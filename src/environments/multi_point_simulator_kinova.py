from environments.kinova_env import KinovaEnv, Bound
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import copy



class MultiPointReacher(KinovaEnv):
    VALID_REWARDS = ['l2', 'l1', 'squared', 'precision', 'action-norm', 'discrete-time-penalty', 'keep-moving']
    VALID_TARGET_GENERATION = [ 'kinematic', 'fixed']

    def __init__(self,
                    dt=0.05,
                    max_velocity=None,
                    control_joints = list(range(7)),
                    joint_bounds=None,
                    H=100,
                    reward_type='l2',
                    precision_tau=0.01,
                    epsilon_threshold=0.1,
                    target_generation = 'kinematic',
                    radius_bound = [0.6, 1.07], #lower bound is psuedo arbitrary
                    phi_bound =[0.0, 1.0], #to be be more cone like shrink upper bound
                    theta_bound = [0.0, np.pi * 2],
                    fixed_target = np.array([0.6, 0.6, 0.6]),
                    points_config="pose_and_origin"
                    ):

        super(MultiPointReacher, self).__init__(ths=np.zeros(7),
                                                dt=dt, use_gripper=True, max_velocity=max_velocity,
                                                H = H)


        self._link_psns = self.kinematics.calc_link_positions()
        self._control_joints = control_joints

        self.ang_v = np.zeros(len(control_joints))

        self.joint_bounds = Bound([l.low for l in self.JOINT_BOUNDS],
                [l.high for l in self.JOINT_BOUNDS]) if joint_bounds is None else joint_bounds

        #hard coded for now
        self.boundary_mode = 'angular'
        self.obs_mode = 'complete'
        self.points_config = points_config


        self._end_effector_pose = np.zeros((3, 4))
        self.target_generation = target_generation.lower()
        self.fixed_target = fixed_target
        self.target = np.zeros(3) if  self.target_generation != 'fixed' else fixed_target

        #OpenAI gym configs
        self.action_space = spaces.Box(low=-1.0* self._max_velocity
                , high=self._max_velocity, shape=(len(control_joints),))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self._obs()),), dtype=np.float32)

        self.reward_type = reward_type.lower().split(',')
        self.precision_tau = precision_tau
        self.epsilon_threshold= epsilon_threshold

        self.steps = 0


        self.reset()
        #rendering configuration things
        self.fig = None
        self.ax = None
        self.robot = None
        self.joints = None
        self.pose = None
        self.targImg = None

    def update_psn(self):
        self._link_psns = self.kinematics.calc_link_positions()
        self._end_effector_pose = self.calc_tip_points()
        #self.kinematics.calc_pose_and_origin()

    def calc_tip_points(self):
        pts = None
        if self.points_config == "pose_and_origin":
            pts = self.kinematics.calc_pose_and_origin()
        elif self.points_config == "pose":
            pts = self.kinematics.calc_pose()
        elif self.points_config == "min_pose_and_origin":
            pts = self.kinematics.calc_min_pos_origin()

        return pts

    def _obs(self):
        x_end_effector = self._end_effector_pose

        th = self.kinematics.get_thetas()
        thvs = self.ang_v.copy()

        c = np.cos(th)
        s = np.sin(th)

        if self.obs_mode == 'complete':
            state = np.concatenate([x_end_effector.flatten('F'), c, s, thvs, self.target.flatten('F')])
        elif self.obs_mode == 'reduced':
            state = np.concatenate([th, thvs, self.target.flatten('F') - x_end_effector.flatten('F')]) #change

        return state

    def _generate_target(self):
        if self.target_generation == 'kinematic':
            #since we sample joint in space, kinematics are transforming samples
            #i.e. it is not a uniform distribution
            thetas = self.kinematics.get_thetas()
            #sample angles for each joint

            q_sample = np.zeros(self.NUM_JOINTS)
            for i in range(self.NUM_JOINTS):
                if i in self._control_joints:
                    low = self.joint_bounds.low[i]
                    high = self.joint_bounds.high[i]
                    bound = Bound(low if low > -np.inf else 0.0, high if high < np.inf else 2*np.pi)
                    q_s = np.random.uniform(bound.low, bound.high)
                    q_sample[i] = np.clip(q_s, bound.low, bound.high)

            self.kinematics.set_thetas(q_sample)
            self.target = self.calc_tip_points()
            self.kinematics.set_thetas(thetas)
        elif self.target_generation == "fixed":
            self.target = self.fixed_target
        else:
            raise Exception("Invalid target generation {}".format(self.target_generation))


    def _calc_reward(self, state, action, state_p):
        if self.obs_mode == 'complete':
            x_t = state_p[-3:]
            x_e = state_p[0:3]

            x_prev = state[0:3]

            difference = x_t - x_e
        elif self.obs_mode == 'reduced':
            difference = state_p[-3:]
            x_prev = state[-3:]

        l2_dist = np.linalg.norm(difference, ord=2)

        reward = 0.0
        if 'l2' in self.reward_type:
            reward += -l2_dist
        if 'l1' in self.reward_type:
            reward += -np.linalg.norm(difference, ord=1)
        if 'precision' in self.reward_type:
            reward += np.exp( -l2_dist / self.precision_tau)

        #The idea is to penalize agent for being far away from target, the below supposedly helps to encourage moving to target faster
        if 'discrete-time-penalty' in self.reward_type and l2_dist >= self.epsilon_threshold:
            reward += -1.0 #it is bad to be some ball away from target

        if 'keep-moving' in self.reward_type and l2_dist >= self.epsilon_threshold:
            reward += np.linalg.norm(x_e - x_prev, 2) #good to move away from current position

        if 'action-norm' in self.reward_type:
            reward += -np.linalg.norm(action, 2) ** 2 #squared action norm

        return reward

    def step(self, a=None):
        if a is not None:
            self.ang_v = np.clip(a, -1.0 * self._max_velocity, self._max_velocity)

        if self.boundary_mode == 'angular':
            low = self.joint_bounds.low
            high = self.joint_bounds.high
            current = self.kinematics.get_thetas()

            in_bound = self.check_boundary(low, high, current, self.ang_v, self.dt)
            self.ang_v = self.ang_v * in_bound
            #check all angles within boudns
        elif self.boundary_mode == 'position':
            print('check position')


        thetas = self.kinematics.get_thetas()
        self.kinematics.set_thetas(thetas + self.ang_v * self.dt)
        self.update_psn()

        self.steps += 1
        done = self.H <= self.steps

        state = self._obs()
        action = self.ang_v #not 100% sure this is right choice for this or not :/
        reward = self._calc_reward(self.prev_state, action, state)
        self.prev_state = state

        return state, reward * self.dt, done, {}

    def reset(self, thetas=np.array([0., 0.5, 0, -0.5, 0, 0.5, 0]),
            target = None):

        self.fixed_target = target if target is not None else self.fixed_target

        self.kinematics.set_thetas(thetas)
        self.update_psn()
        self._generate_target()

        self.prev_state = self._obs()
        self.steps = 0
        return self.prev_state

    def _initFig(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        bound = sum(self.link_lengths)
        x = bound
        y = bound
        z = bound

        self.ax.set(xlim=(-x, x), ylim=(-y, y), zlim=(-0, z))
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])

    def joint_dim(self, i):
        return np.array([l[i] for l in self._link_psns])

    def render(self, mode='human'):

        if self.fig is None and self.ax is None:
            self._initFig()

        targ_x = self.target[0,:]
        targ_y = self.target[1,:]
        targ_z = self.target[2,:]
        
        if self.targImg is None:
            self.targImg = self.ax.scatter(targ_x, targ_y, targ_z)
        else:
            self.targImg._offsets3d = (targ_x, targ_y, targ_z)

        x = self._end_effector_pose[0,:]
        y = self._end_effector_pose[1,:]
        z = self._end_effector_pose[2,:]

        if self.pose is None:
            self.pose = self.ax.scatter(x,y, z)
        else:
            self.pose._offsets3d = (x, y, z)



        x = self.joint_dim(0)
        y = self.joint_dim(1)
        z = self.joint_dim(2)

        if self.joints is None:
            self.joints = self.ax.scatter(x, y, z)
        else:
            self.joints._offsets3d = (x, y, z)

        if self.robot is None:
            self.robot,  = self.ax.plot(x, y, z)
        else:
            self.robot.set_xdata(x)
            self.robot.set_ydata(y)
            self.robot.set_3d_properties(z)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        


        

        return None


