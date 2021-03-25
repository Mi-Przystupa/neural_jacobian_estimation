#!/usr/bin/env python

### University of Alberta
#   Authors: Michael Przystupa & Amirmohammad Karimi
#
#
###
import atexit #ros might handle this...
import sys
import rospy
from kortex_driver.srv import *
from kortex_driver.msg import *
import numpy as np
from ExplicitKinematics import ExplicitKinematics
from DHKinematics import DHKinematics
import gym
from gym import spaces
import argparse
import actionlib
import time
from sensor_msgs.msg import JointState

success = True

class KinovaReacher(gym.core.Env):
    def __init__(self,  dof=7,
                        episode_length=100,
                        dt = 0.05,
                        max_action=np.array([0.5,0.5]),
                        angle_boundary_low = np.array([- np.pi/4, - np.pi/4, - np.pi/4, - np.pi/4, - np.pi/4,- 3*np.pi/4, - np.pi/4]),
                        angle_boundary_high = np.array([+ np.pi/4, + np.pi/4, + np.pi/4, + np.pi/4, + np.pi/4,+ 3*np.pi/4, + np.pi/4]),
                        position_boundary_low = np.array([0.0,-0.5]),
                        position_boundary_high = np.array([0.7,0.5]),
                        target_angle_boundary_low = np.array([- np.pi/4, - np.pi/4, - np.pi/4, - np.pi/4, - np.pi/4,- 3*np.pi/4, - np.pi/4]),
                        target_angle_boundary_high = np.array([+ np.pi/4, + np.pi/4, + np.pi/4, + np.pi/4, + np.pi/4,+ 3*np.pi/4, + np.pi/4]),
                        target_xy_boundary_low = np.array([0.2,-0.25]),
                        target_xy_boundary_high = np.array([0.6,0.25]),
                        target_mode = 'position',
                        boundary_mode = 'position',
                        obs_mode = 'complete'
                        ):
        """
        dt = action_cycle time
        max_action in rad/sec
        """
        self.dt = dt
        self.max_action = max_action

        try:
            #rospy.init_node('example_full_arm_movement_python')
            rospy.init_node('kinova_reacher_full_arm')

            self.dt = dt

            self.HOME_ACTION_IDENTIFIER = 2

            # Get node params

            self.robot_name = rospy.get_param('~robot_name', "my_gen3")


            self.degrees_of_freedom = rospy.get_param("/" + self.robot_name + "/degrees_of_freedom", dof)

            if self.degrees_of_freedom > dof:
                self.degrees_of_freedom = dof

            self.is_gripper_present = rospy.get_param("/" + self.robot_name + "/is_gripper_present", False)

            rospy.loginfo("Using robot_name " + self.robot_name + " , robot has " + str(self.degrees_of_freedom) + " degrees of freedom and is_gripper_present is " + str(self.is_gripper_present))

            # Init the services
            clear_faults_full_name = '/' + self.robot_name + '/base/clear_faults'
            rospy.wait_for_service(clear_faults_full_name)
            self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)

            #Used to handle reseting robot position
            read_action_full_name = '/' + self.robot_name + '/base/read_action'
            rospy.wait_for_service(read_action_full_name)
            self.read_action = rospy.ServiceProxy(read_action_full_name, ReadAction)
            execute_action_full_name = '/' + self.robot_name + '/base/execute_action'
            rospy.wait_for_service(execute_action_full_name)
            self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)

            #Enables sending grip commands
            send_gripper_command_full_name = '/' + self.robot_name + '/base/send_gripper_command'
            rospy.wait_for_service(send_gripper_command_full_name)
            self.send_gripper_command = rospy.ServiceProxy(send_gripper_command_full_name, SendGripperCommand)

            #handles sending joint commands to robots
            send_joint_speed_command_full_name = '/' + self.robot_name +'/base/send_joint_speeds_command'
            rospy.wait_for_service(send_joint_speed_command_full_name)
            self.send_joint_speeds_command = rospy.ServiceProxy(send_joint_speed_command_full_name, SendJointSpeedsCommand)
            #handles acquiring measured joint angles
            get_joint_angles_command_full_name = '/' + self.robot_name + '/base/get_measured_joint_angles'
            rospy.wait_for_service(get_joint_angles_command_full_name)
            self.get_joint_angles_command = rospy.ServiceProxy(get_joint_angles_command_full_name, GetMeasuredJointAngles)

            #Acquire Pose
            get_pose_command_full_name = '/' + self.robot_name + '/base/get_measured_cartesian_pose'
            rospy.wait_for_service(get_pose_command_full_name)
            self.get_pose_command = rospy.ServiceProxy(get_pose_command_full_name, GetMeasuredCartesianPose)

            #
            play_joint_trajectory_full_name = '/' + self.robot_name + '/base/play_joint_trajectory'
            rospy.wait_for_service(play_joint_trajectory_full_name)
            self.play_joint_trajectory = rospy.ServiceProxy(play_joint_trajectory_full_name, PlayJointTrajectory)


            # subscriber to robot feedback message
            topic_name = '/' + self.robot_name +'/base_feedback/joint_state'
            self.sub = rospy.Subscriber(topic_name, JointState, self._getFeedbackCallback)

            # publisher to act on robot ( joint_velocities)
            topic_name = '/' + self.robot_name + '/in/joint_velocity'
            self.pub = rospy.Publisher(topic_name, Base_JointSpeeds , queue_size=1)

        except Exception as e:
            print("error thrown")
            print(e)
            self.is_init_success = False
        else:
            self.is_init_success = True


        if self.is_init_success:


            #register handles to stop robot gracefully e.g. killing program
            rospy.on_shutdown(self.shutdown)

            try:
                rospy.delete_param("/kortex_examples_test_results/full_arm_movement_python")
            except:
                pass

        # kinematics
        self.kinematics = DHKinematics(self._zeroAngleVec() )
        self.L1 = 0.3143
        self.L2 = 0.1674 + 0.120
        self.z_offset = 0.7052
        self.x = 0.6021 # is robot_z - z_offset
        self.y = 0      # is robot_x
        # self.end_effector_position = np.array([0.6021, 0])
        self.th1 = 0    # first active_joint
        self.th2 = 0    # second active_joint
        self.dth1 = 0   #first joint velocity
        self.dth2 = 0   #second joint velocity
        self.all_joints = self._zeroAngleVec() # array of all joint angles
        self.active_joints = [3, 5]
        self.rate = rospy.Rate(1/self.dt)
        target_robot_coordinate = self.kinematics.calcEndEffector() # just initialisation
        self.target = np.array([target_robot_coordinate[2] - self.z_offset, target_robot_coordinate[0]])
        self.in_boundary = True
        #arbitrary choice right now, reminder: Kinova is in DEGREES NO its rad
        self.boundary_mode = boundary_mode
        self.angle_boundary_low = angle_boundary_low
        self.angle_boundary_high = angle_boundary_high
        self.position_boundary_low = position_boundary_low
        self.position_boundary_high = position_boundary_high

        self.target_mode = target_mode
        self.target_angle_boundary_low = target_angle_boundary_low
        self.target_angle_boundary_high = target_angle_boundary_high
        self.target_xy_boundary_low = target_xy_boundary_low
        self.target_xy_boundary_high = target_xy_boundary_high

        # RL
        # list of joints to have action in _move_joints
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.obs_mode = obs_mode
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.active_joints = [3,5]
        self.time_steps = 0
        self.episode_length = episode_length
        # setup reset_joint_angles
        self.reset_joint_angles = self._zeroAngleVec()
        (th1,th2)=self.inv_kin(0.4,0)
        self.reset_joint_angles[self.active_joints] = [th1[0],th2[0]]
        self.reset_error_threshold = 0.02

        # grab sport ball
        self._send_gripper_command(0.3)


    # def argumentParser(self,argument):
    # 	""" Argument parser """
    # 	parser = argparse.ArgumentParser(description='Drive robot joint to command position')
    # 	parser.add_argument('kinova_robotType', metavar='kinova_robotType', type=str, default='j2n6a300',
    #                     help='kinova_RobotType is in format of: [{j|m|r|c}{1|2}{s|n}{4|6|7}{s|a}{2|3}{0}{0}]. eg: j2n6a300 refers to jaco v2 6DOF assistive 3fingers. Please be noted that not all options are valided for different robot types.')
    # 	args_ = parser.parse_args(argument)
    # 	prefix = args_.kinova_robotType + "_"
    # 	nbJoints = int(args_.kinova_robotType[3])
	#     return prefix, nbJoints
    def seed(self, seed=None):
        np.random.seed(seed)

    def getIsInitSuccess(self):
        return self.is_init_success

    def getJointAngles(self):
        joint_angs = self.get_joint_angles_command().output.joint_angles
        angles = [j.value for j in joint_angs]
        #angs.output.joint_angles[6].value
        return angles

    def getPose(self):
        pose = self.get_pose_command().output
        return pose

    def getBaseFeedback_xyz(self):
        feedback = rospy.wait_for_message("/" + self.robot_name + "/base_feedback", BaseCyclic_Feedback)
        #gives tool pose x, y, z, in meters
        return [feedback.base.commanded_tool_pose_x,
                feedback.base.commanded_tool_pose_y,
                feedback.base.commanded_tool_pose_z]

    def _send_gripper_command(self, value):

        # Initialize the request
        # This works for the Robotiq Gripper 2F_85
        # Close the gripper
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION

        rospy.loginfo("Sending the gripper command...")

        # Call the service
        try:
            self.send_gripper_command(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SendGripperCommand")
            return False
        else:
            return True

    def _sendRobotHome(self):
        # The Home Action is used to home the robot. It cannot be deleted and is always ID #2:
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        try:
            res = self.read_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ReadAction")
            return False
        # Execute the HOME action if we could read it
        else: #if this is called then the try statement worked...
            # What we just read is the input of the ExecuteAction service
            req = ExecuteActionRequest()
            req.input = res.output
            rospy.loginfo("Sending the robot home...")
            try:
                self.execute_action(req)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ExecuteAction")
                return False
            else:
                return True

    def _rotateWrist(self,th_d):
        print('rotating wrist test')
        self._moveSingleJoint(th_d, 6, "wrist")

    def _zeroVelocityVec(self): #changed to np.array
        return np.array([0.0 for i in range(self.degrees_of_freedom)])

    def _zeroAngleVec(self):
        return np.array([0.0 for i in range(self.degrees_of_freedom)])

    def _moveSingleJoint(self, th_d, joint, name=None):
        """
            th_d: angular velocity you want joint to move at
            joint: integer from 0 - 6 that you want to move
            name: identifier for joint, if provided None defaults to: "joint {joint}", {joint} is input provided
        """
        if name is None:
            name = "joint {}".format(joint)

        th_ds = self._zeroVelocityVec()
        th_ds[joint] = th_d

        rospy.loginfo("Sending the robot {} speed".format(name))
        return self._moveJoints(th_ds)

    def _moveJoints(self, th_ds):
        #Create struct to send via ros
        temp_base = Base_JointSpeeds()
        for i, v in enumerate(th_ds):
            temp_vel = JointSpeed()
            temp_vel.joint_identifier = i
            temp_vel.value = v
            temp_base.joint_speeds.append(temp_vel)

        # Send the angles
        try:
            self.send_joint_speeds_command(temp_base)
        except rospy.ServiceException:
            rospy.logerr("Failed to send jointspeed")
            return False
        else:
            return True

    def _stopMovement(self):
        #print("stopping movement")
        #set all joints to 0
        jointCmds = self._zeroVelocityVec()
        # print(rospy.get_time())
        self._publishJointVelCmd(jointCmds)

    def _clearFaults(self):
        #from kinova example code, need to clear faults else robot doesn't move
        try:
            self.clear_faults()
        except rospy.ServiceException:
            rospy.logerr("Failed to call ClearFaults")
            return False
        else:
            rospy.loginfo("Cleared the faults successfully")
            rospy.sleep(2.5)
            return True

    def send_cartesian_pose(self, x, y , z):
        # Get the actual cartesian pose to increment it
        # You can create a subscriber to listen to the base_feedback
        # Here we only need the latest message in the topic though
        feedback = rospy.wait_for_message("/" + self.robot_name + "/base_feedback", BaseCyclic_Feedback)

        req = PlayCartesianTrajectoryRequest()

        req.input.target_pose.x = x#feedback.base.commanded_tool_pose_x #+ 0.30
        req.input.target_pose.y = y#feedback.base.commanded_tool_pose_y #+ 0.15
        req.input.target_pose.z = z#feedback.base.commanded_tool_pose_z + 0.15
        req.input.target_pose.theta_x = feedback.base.commanded_tool_pose_theta_x #- 90
        req.input.target_pose.theta_y = feedback.base.commanded_tool_pose_theta_y #+ 35
        req.input.target_pose.theta_z = feedback.base.commanded_tool_pose_theta_z #+ 35

        pose_speed = CartesianSpeed()
        pose_speed.translation = 0.1
        pose_speed.orientation = 15

        # The constraint is a one_of in Protobuf. The one_of concept does not exist in ROS
        # To specify a one_of, create it and put it in the appropriate list of the oneof_type member of the ROS object :
        req.input.constraint.oneof_type.speed.append(pose_speed)

        # Call the service
        rospy.loginfo("Sending the robot to the cartesian pose...")
        try:
            self.play_cartesian_trajectory(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call PlayCartesianTrajectory")
            return False
        else:
            return True

    def _send_joint_angles(self, angles):
        '''
        angles in rad -pi to pi
        '''
        print('angles', angles)
        angles = angles * 180 / np.pi
        angles[np.where(angles < 0)] += 360
        # Create the list of angles
        req = PlayJointTrajectoryRequest()
        # req.input.constraint.type = 2
        # req.input.constraint.value = 60
        # req.input.conctrained_joint
        for i, v in enumerate(angles):
            temp_angle = JointAngle()
            # temp_angle = ConstrainedJointAngle()
            temp_angle.joint_identifier = i
            temp_angle.value = v
            # temp_angle.constraint.type = 2 # joint_constrait_speed
            # temp_angle.constraint.value = 80 # max rad/secffort
            # print(v)
            req.input.joint_angles.joint_angles.append(temp_angle)

            # req.input.constraint.value = 30

        # Send the angles
        rospy.loginfo("Sending the robot to angles {}...".format(angles))
        try:
            self.play_joint_trajectory(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call PlayJointTrajectory")
            return False
        else:
            return True

    def _publishJointVelCmd(self, jointCmds):

        jointCmd = Base_JointSpeeds()
        jointCmd.duration = 0
        for i, v in enumerate(jointCmds):
            temp_JointSpeed = JointSpeed()
            temp_JointSpeed.joint_identifier = i
            temp_JointSpeed.value = v
            temp_JointSpeed.duration = 0
            jointCmd.joint_speeds.append(temp_JointSpeed)

        self.pub.publish(jointCmd)

    def _getFeedbackCallback(self, data):
        '''print(rospy.get_time() - self.t_old)
        self.t_old = rospy.get_time()
        self.x = data.base.tool_pose_z - self.z_offset
        self.y = data.base.tool_pose_x
        self.th1 = data.actuators[self.active_joints[0]].position
        self.th1 *= (np.pi/180)
        self.dth1 = data.actuators[self.active_joints[0]].velocity
        if self.th1 > np.pi:
            self.th1 -= 2*np.pi
        self.th2 = data.actuators[self.active_joints[1]].position
        self.dth2 = data.actuators[self.active_joints[1]].velocity

        self.th2 *= (np.pi/180)
        if self.th2 > np.pi:
            self.th2 -= 2*np.pi'''
        # print(rospy.get_time() - self.t_old)
        self.all_joints = np.array(data.position[:self.degrees_of_freedom])
        self.all_joints_velocity = np.array(data.velocity[:self.degrees_of_freedom])
        self.t_old = rospy.get_time()
        self.th1 = data.position[self.active_joints[0]]
        self.th2 = data.position[self.active_joints[1]]
        self.dth1 = data.velocity[self.active_joints[0]]
        self.dth2 = data.velocity[self.active_joints[1]]
        self.kinematics.ths = data.position[:self.degrees_of_freedom]
        robot_coordinate = self.kinematics.calcEndEffector()
        self.x = robot_coordinate[2] - self.z_offset
        self.y = robot_coordinate[0]


        if self.boundary_mode == 'angular':
            self.in_boundary = self.check_boundary(self.angle_boundary_low, self.angle_boundary_high, self.all_joints, self.all_joints_velocity, self.dt)
        elif self.boundary_mode == 'position':
            # calculate next xy point
            k = 7
            th_s = self.all_joints + self.all_joints_velocity*self.dt*k

            self.kinematics.ths = th_s
            next_point_robot = self.kinematics.calcEndEffector()
            next_point = np.array([next_point_robot[2] - self.z_offset, next_point_robot[0]])
            self.in_boundary = self.check_boundary(self.position_boundary_low, self.position_boundary_high, next_point,0,0)

        if self.in_boundary == False:
            #print('out of in_boundary')
            self._stopMovement()

    @staticmethod
    def check_boundary(low, high, current, rate, dt):
        safety_coefficient = 10
        next = current + safety_coefficient * rate * dt
        if np.all( next > low) and np.all( next < high):
            return True

        return False


    def _obs(self):

        obs = np.array([self.x, self.y, np.cos(self.th1), np.sin(self.th1),  np.cos(self.th2), np.sin(self.th2), self.dth1, self.dth2, self.target[0], self.target[1]])
        return obs

    def _calcReward(self):
        dist = np.linalg.norm(self.target[0:2]-np.array([self.x, self.y]))
        return self.dt * (-dist + np.exp(-(dist**2)/0.01))

    def set_target(self):

        if self.target_mode == 'position':
            while True:
                A = np.random.uniform(  low=self.target_xy_boundary_low,
                                        high=self.target_xy_boundary_high)
                L = np.linalg.norm(A)
                if L <= (self.L1 + self.L2) and L >= np.abs(self.L1 - self.L2):
                    (th1s, th2s) = self.inv_kin(A[0], A[1])
                    th_s1 = self._zeroAngleVec()
                    th_s1[self.active_joints] = [th1s[0], th2s[0]]
                    th_s2 = self._zeroAngleVec()
                    th_s2[self.active_joints] = [th1s[1], th2s[1]]
                    if self.check_boundary(self.angle_boundary_low, self.angle_boundary_high, th_s1, 0, self.dt) or self.check_boundary(self.angle_boundary_low, self.angle_boundary_high, th_s2, 0, self.dt):
                        self.target = A
                        break

        elif self.target_mode == 'angle':
            th1 = np.random.uniform(low=self.target_angle_boundary_low[self.active_joints[0]] ,
                                    high=self.target_angle_boundary_high[self.active_joints[0]])

            th2 = np.random.uniform(low=self.target_angle_boundary_low[self.active_joints[1]] ,
                                    high=self.target_angle_boundary_high[self.active_joints[1]])
            th_s = self._zeroAngleVec()
            th_s[self.active_joints] = [th1, th2]
            self.kinematics.ths = th_s
            target_robot_coordinate = self.kinematics.calcEndEffector()
            self.target = np.array([target_robot_coordinate[2] - self.z_offset, target_robot_coordinate[0]])

    def step(self, action):
        """
        action must be in [-1 1] then is scaled to max_action ( rad /s )
        """
        action = self.max_action * np.clip(action, a_min=-1, a_max=1)

        # take action
        jointCmds = self._zeroVelocityVec()
        jointCmds[self.active_joints] = action
        if self.boundary_mode == 'angular':
            self.in_boundary = self.check_boundary(self.angle_boundary_low, self.angle_boundary_high, self.all_joints, jointCmds, self.dt)

        elif self.boundary_mode == 'position':
            # calculate next xy point
            k = 5
            th_s = self.all_joints
            th_s[self.active_joints] = [self.th1+action[0]*self.dt*k, self.th2+action[1]*self.dt*k]
            self.kinematics.ths = th_s
            next_point_robot = self.kinematics.calcEndEffector()
            next_point = np.array([next_point_robot[2] - self.z_offset, next_point_robot[0]])
            self.in_boundary = self.check_boundary(self.position_boundary_low, self.position_boundary_high, next_point,0,0)

        if self.in_boundary == True:
            self._publishJointVelCmd(jointCmds)
        else:
            self._stopMovement()

        self.rate.sleep()       # this maintains the frequency as defined in init (20 Hz for now)
        # time.sleep(self.dt)

        # rospy.spin()
        observation = self._obs()
        reward = self._calcReward()
        self.steps += 1
        if self.steps == self.episode_length:
            done = True
        else:
            done = False

        return observation, reward, done, None

    def reset(self):
        #print("reset something")
        #stopping
        if not rospy.is_shutdown():
            self._stopMovement()
            rospy.sleep(1.0)
            # move to reset pose
            th_s = self.reset_joint_angles
            success = True
            success &= self._send_joint_angles(th_s)
            # t = time.time()
            print(np.max(np.abs(self.all_joints - self.reset_joint_angles)))
            time.sleep(np.max(np.abs(self.all_joints - self.reset_joint_angles)) / 0.6)
            # print(time.time() - t)
            # while np.any(np.abs(self.all_joints - self.reset_joint_angles) > self.reset_error_threshold): # this means wait until getting reset joint position
            #     # print(self.all_joints - self.reset_joint_angles)
            #     success &= self._send_joint_angles(th_s)
            #     time.sleep(1.0)
            # to make sure low error
            rospy.sleep(0.5)
            # set new target
            self.set_target()

            # reset steps
            self.steps = 0

        return self._obs()

    def inv_kin(self,tx,ty):
        l1 = self.L1
        l2 = self.L2
        l3 = (tx**2+ty**2)**0.5
        th0 = np.arctan2(ty,tx)
        a = np.arccos((l1**2 + l3**2 - l2**2)/(2*l1*l3))
        th1 = np.array([a, -a])+th0
        a = np.arccos((l1**2 + l2**2 - l3**2)/(2*l1*l2))
        th2 = np.array([a-np.pi, np.pi-a])
        return (th1,th2)

    def shutdown(self):
        print("shutdown stuff gracefully")
        print("resetting robot configuration")
        self.reset()
        print("Unsubscribing Updating State")
        #unregister because we don't need to keep updating state variables
        self.sub.unregister()
        print("Manually shutting down ROS")
        #this call manually stops ROS so any subscribes we forgot should stop
        rospy.signal_shutdown("KinovaEnvironment is shutting down. Stopping Ros")
        sys.exit('Exiting due to shutdown')



if __name__ == "__main__":
    print("hello world")

    reacher = KinovaReacher()
    success = reacher.getIsInitSuccess()

    time.sleep(1)
    if success:
        #*******************************************************************************
        # Make sure to clear the robot's faults else it won't move if it's already in fault
        success &= reacher._clearFaults()
        #*******************************************************************************

        obs = reacher.reset()
        for _ in range(4):
            t0 = rospy.get_time()
            obs = reacher.reset()
            done = False
            for t in range(reacher.episode_length):
                action = np.random.uniform(-1,1,2)
                # if t < reacher.episode_length / 2.0:
                #    action = np.array([1, 0])
                # else:
                #    action = np.array([-1, 0])

                obs, r, done, info = reacher.step(action=action)
                if done or rospy.is_shutdown():
                    break
                print(obs, rospy.get_time()-t0)
                t0 = rospy.get_time()
