#!/usr/bin/env python3

### University of Alberta
#   Authors: Michael Przystupa & Amirmohammad Karimi
#
#
###
import rospy
import numpy as np
import gym
from gym import spaces
import time
from environments.kinova_env import KinovaEnv, Bound
from environments.kinova_api import KinovaInteractor, KinovaListener


class FullDOFKinovaReacher(KinovaEnv):
    #We want the targets in a subset of what's actually reachable...mostly for safety
    TARGET_DEFAULT_BOUNDS = [
                    Bound( 0.0, np.pi), 
                    Bound(-1.0, 1.0),
                    Bound( 0.0, np.pi),
                    Bound(-1.0, 1.0), 
                    Bound( 0.0, np.pi),
                    Bound(-1.0, 1.0),
                    Bound( 0.0, np.pi),
                ]
    VALID_REWARDS = ['l2', 'l1', 'squared', 'precision', 'action-norm']
    VALID_TARGET_GENERATION = ['kinematic', 'fixed']
    CONTROL_JOINTS = list(range(KinovaEnv.NUM_JOINTS))

    def __init__(self,  dt=0.05,
                        max_velocity=None,
                        joint_bounds=None,
                        target_bounds=None,
                        H=100,
                        target_mode = 'kinematic',
                        fixed_target = [0.61695433, 0.10706768, 1.03322147],
                        reset_error_threshold=0.001, #worked well empirically, too small and you'll get caught in a loop, could be as high as 0.02
                        reset_joint_angles=np.array([0., 0.25, 0.0, -0.3, 0.0, 0.1, 0]),
                        reward_type='l2,precision',
                        precision_tau = 0.01
                        ):
        """
        dt = action_cycle time
        max_action in rad/sec
        """
        super(FullDOFKinovaReacher, self).__init__(ths=np.zeros(7),
                                                dt=dt, use_gripper=True, max_velocity=max_velocity,
                                                H = H)
        
        #Initialized the jount boundaries which are the limit we allow the kinova to move 
        self.joint_bounds = Bound([l.low for l in (self.JOINT_BOUNDS if joint_bounds is None else joint_bounds)],
                [l.high for l in (self.JOINT_BOUNDS if joint_bounds is None else joint_bounds)]) 

        # Set-up target  for initialization
        self.target_mode = target_mode
        self.target_bounds = Bound([l.low for l in (self.JOINT_BOUNDS if target_bounds is None else target_bounds)],
                [l.high for l in (self.TARGET_DEFAULT_BOUNDS if target_bounds is None else target_bounds)])
        self.fixed_target = fixed_target
        self.target = fixed_target
        
        #Set-up reward information
        self.reward = 0.0
        self.reward_type = reward_type.lower().split(',')
        self.reward_info = {}
        for r_t in self.reward_type:
            assert r_t in self.VALID_REWARDS

        # OPEN AI gym business
        
        # setup reset information  
        self.reset_joint_angles =  reset_joint_angles
        self.reset_error_threshold = reset_error_threshold
        self.resetting = False #flag to let callback know when resetting kinova
        self.precision_tau = precision_tau

        #initialize Kinova API
        self.kinova_commands = KinovaInteractor(dt, joint_state_subscriber=self._getFeedbackCallback)
        assert self.kinova_commands.get_is_init_success(), "Could not connect with Kinova"

        self.kinova_listener = KinovaListener(self.CONTROL_JOINTS, self.NUM_JOINTS, dt)

        rospy.on_shutdown(self.close)
        # list of joints to have action in _move_joints
        self.action_space = spaces.Box(low=-self._max_velocity, high=self._max_velocity, shape=(len(self.CONTROL_JOINTS),))
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=(len(self._obs()),), dtype=np.float32)


    def get_cartesian(self): 
        return self.kinova_listener.get_position()

    def get_target(self):
        return self.target

    def seed(self, seed=None):
        np.random.seed(seed)

    def boundary_thread(self):
        print('implement me')

        #literally just listens for boundary 
        while not rospy.is_shutdown():
            if not self.resetting:
                thetas = self.kinova_listener.get_controlled_angles()
                thetas_vel = self.kinova_listener.get_controlled_angular_velocity()

                check_bounds = self.check_boundary(self.joint_bounds.low, self.joint_bounds.high,
                                                    self.thetas, self.thetas_velocity, self.dt)
                #always stop moving joints that will go out of bounds
                jointCmds = self.thetas_velocity  * check_bounds
                self.kinova_commands.publishJointVelCmd(jointCmds)
                
                in_bound = np.any(check_bounds)
                if not in_bound: # if no joint woud be in bound, stop all movement
                    self.kinova_commands.stopMovement()


    def _getFeedbackCallback(self, joint_state, base_cyclic_feedback):
        """
            We listen to several publishers from kinova to get joint_state in radians and end effector position from cyclic feedback
        """
        if rospy.is_shutdown():
            self.stop()
        else:
            self.thetas = np.array(joint_state.position[:self.NUM_JOINTS])
            self.thetas_velocity = np.array(joint_state.velocity[:self.NUM_JOINTS])

            #using sensor as opposed to kinematics is different by something like +/- 0.0003  in euclidean distance
            self.cartesian_end_effector = np.array([base_cyclic_feedback.base.commanded_tool_pose_x,
                                            base_cyclic_feedback.base.commanded_tool_pose_y,
                                            base_cyclic_feedback.base.commanded_tool_pose_z
                                            ])
            self.reward = self._calc_reward(x_t = self.get_target(), x_e = self.get_cartesian() )

            if not self.resetting and hasattr(self, "kinova_commands"):
                check_bounds = self.check_boundary(self.joint_bounds.low, self.joint_bounds.high,
                                                    self.thetas, self.thetas_velocity, self.dt)
                #always stop moving joints that will go out of bounds
                jointCmds = self.thetas_velocity  * check_bounds
                self.kinova_commands.publishJointVelCmd(jointCmds)
                
                in_bound = np.any(check_bounds)
                if not in_bound: # if no joint woud be in bound, stop all movement
                    self.kinova_commands.stopMovement()
            
    def _obs(self):
        thetas = self.kinova_listener.get_controlled_angles()
        thetas_velocity = self.kinova_listener.get_controlled_angular_velocity()

        cartesian = self.get_cartesian()
        target = self.get_target()
        c = np.cos(thetas)
        s = np.sin(thetas)
        state = np.concatenate([cartesian, c, s, thetas_velocity, target])

        return state

    def _calc_reward(self, x_t, x_e, thetas_velocity):
        self.reward_info = {}
        
        difference = x_t - x_e

        l2_dist = np.linalg.norm(difference, ord=2)

        self.reward_info['l2'] = -l2_dist

        self.reward_info['l1'] = -np.linalg.norm(difference, ord=1)

        self.reward_info['precision'] = np.exp( -l2_dist / self.precision_tau)

        self.reward_info['action-norm'] = -np.linalg.norm(thetas_velocity, 2) ** 2 #squared action norm

        reward = 0.0
        for rew in self.reward_type:
            reward += self.reward_info[rew]

        return reward * self.dt
        
    def set_target(self):
        if self.target_mode == 'kinematic':
            #since we sample joint in space, kinematics are transforming samples
            #i.e. it is not a uniform distribution
            thetas = self.kinematics.get_thetas()
            #sample angles for each joint
            q_sample = np.zeros(self.NUM_JOINTS)
            for i in range(self.NUM_JOINTS):
                low = self.target_bounds.low[i]
                high = self.target_bounds.high[i]
                bound = Bound(low if low > -np.inf else 0.0, high if high < np.inf else 2*np.pi)
                q_s = np.random.uniform(bound.low, bound.high)
                q_sample[i] = np.clip(q_s, bound.low, bound.high)

            self.kinematics.set_thetas(q_sample)
            self.target = self.kinematics.calcEndEffector().copy()
            self.kinematics.set_thetas(thetas)
        elif self.target_mode == 'fixed':
            self.target = self.fixed_target.copy()

        return self.target.copy()
        
    def step(self, action):
        """
        action must be in [-1 1] then is scaled to max_action ( rad /s )
        """
        if not rospy.is_shutdown():
            action = np.clip(action, a_min=-self._max_velocity, a_max=self._max_velocity)

            # take action
            in_bound = self.check_boundary(self.joint_bounds.low, self.joint_bounds.high, self.kinova_listener.get_thetas(), action, self.dt, 10)
            jointCmds = action  * in_bound

            self.kinova_commands.publishJointVelCmd(jointCmds)
            if not np.any(in_bound):
                self.kinova_commands.stopMovement()

            self.kinova_commands.rate_sleep()

            observation = self._obs()
            thetas_velocity = self.kinova_listener.get_controlled_angular_velocity()
            reward = self._calc_reward(x_t = self.get_target(), x_e = self.get_cartesian(), thetas_velocity= thetas_velocity)

            self.steps += 1
            done = (self.steps >= self.H)
            info = self.reward_info

            return observation, reward, done, info

    def reset(self, target=None, use_target_as_is=False):
        self.use_target_as_is = use_target_as_is #TODO work around for 2DOF setting
        #stopping
        if not rospy.is_shutdown():
            self.kinova_commands._clearFaults()
            self.fixed_target = self.fixed_target if target is None else target
            self.resetting = True
            rospy.loginfo("Resetting Kinova Reacher Environment")

            self.stop()
            rospy.sleep(1.0)
            # move to reset pose
            th_s = self.reset_joint_angles
            success = True
            success &= self.kinova_commands.send_joint_angles(th_s)

            if not success:
                rospy.logerr("Unsuccessfully sent joint angles during reset, closing environment")
                self.close()

            thetas = self.kinova_listener.get_thetas()
            while np.any(np.abs(thetas - self.reset_joint_angles) > self.reset_error_threshold) and not rospy.is_shutdown() and success:
                thetas = self.kinova_listener.get_thetas()
                rospy.sleep(0.01)

            # to make sure low error
            rospy.sleep(0.5)
            # set new target
            self.set_target()

            # reset steps
            self.steps = 0

            self.resetting = False

        return self._obs()

    def render(self, mode='human'):
        #do nothing
        i = 0

    def stop(self):
        self.kinova_commands.stopMovement()
        self.kinova_commands.publishJointVelCmd(np.zeros(self.NUM_JOINTS))

    def close(self):
        rospy.loginfo("closing Kinova 3D Reaching environment")
        self.stop()
        self.kinova_commands.shutdown()


class FOURDOFKinovaReacher(FullDOFKinovaReacher):
    TARGET_DEFAULT_BOUNDS = [
                    Bound( 0.0, 0.0), 
                    Bound( 0.0, 0.0),
                    Bound( 0.0, np.pi),
                    Bound(-1.0, 1.0), 
                    Bound( 0.0, np.pi),
                    Bound(-1.0, 1.0),
                    Bound( 0.0, np.pi),
                ]

    JOINT_BOUNDS = [
                    Bound(0.0, 0.0), 
                    Bound(0.0, 0.0),
                    Bound(-np.inf, np.inf),
                    Bound(-2.5796, 2.5796), 
                    Bound(-np.inf, np.inf),
                    Bound(-2.0996, 2.0996),
                    Bound(0.0, 0.0),
                ]

    def __init__(self,  dt=0.05,
                        max_velocity=None,
                        joint_bounds=None,
                        target_bounds=None,
                        H=100,
                        target_mode = 'kinematic',
                        fixed_target = [0.61695433, 0.10706768, 1.03322147],
                        reset_error_threshold=0.001, #worked well empirically, too small and you'll get caught in a loop, could be as high as 0.02
                        reset_joint_angles=np.array([0., 0.0, 0.0, -0.3, 0.0, 0.1, 0]),
                        reward_type='l2,precision',
                        precision_tau = 0.01
                        ):
        super(FOURDOFKinovaReacher, self).__init__(
                dt= dt,
                max_velocity= max_velocity,
                joint_bounds= joint_bounds,
                target_bounds= target_bounds,
                H= H,
                target_mode = target_mode,
                fixed_target = fixed_target,
                reset_error_threshold= reset_error_threshold, 
                reset_joint_angles= reset_joint_angles,
                reward_type= reward_type,
                precision_tau = precision_tau
                )

        self.control_joints = [2, 3, 4, 5]
        self.stationary_joints = [0, 1, 6]
        print(self.joint_bounds)
        
        for i in self.stationary_joints:
            self.joint_bounds.low[i] = 0.0
            self.joint_bounds.high[i] = 0.0

    def step(self, action):
        #check action dimensions, if full angle, extract only ones we need
        #else, reshape to be 7 DOF & send to super
        if len(action) == 4:
            temp = np.zeros(self.NUM_JOINTS)
            for i, j in enumerate(self.control_joints):
                temp[j] = action[i]
            action = temp
        else:
            for i, j in enumerate(self.stationary_joints):
                action[j] = 0.0

        return super(FOURDOFKinovaReacher, self).step(action)


