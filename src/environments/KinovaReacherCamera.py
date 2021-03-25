#!/usr/bin/env python

### University of Alberta
#   Authors: Michael Przystupa & Amirmohammad Karimi
#
#
###
from KinovaReacher import KinovaReacher
from std_msgs.msg import String
import rospy
import json
import numpy as np
from sensor_msgs.msg import JointState
import time
import sys
import cv2

class KinovaReacherCamera(KinovaReacher):
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
                        obs_mode = 'complete',
                        camera_publisher_topic = 'tracker_status',
                        camera_listener_topic = 'target_image'

                        ):

        super(KinovaReacherCamera,self).__init__(dof,
                                                episode_length,
                                                dt,
                                                max_action,
                                                angle_boundary_low,
                                                angle_boundary_high,
                                                position_boundary_low,
                                                position_boundary_high,
                                                target_angle_boundary_low,
                                                target_angle_boundary_high,
                                                target_xy_boundary_low,
                                                target_xy_boundary_high,
                                                target_mode ,
                                                boundary_mode ,
                                                obs_mode)
        # if self.is_init_success:
        self.sub_cam = rospy.Subscriber(camera_publisher_topic, String, self._updateTrackerStateCallback)
        self.pub_cam = rospy.Publisher(camera_listener_topic, String)
        self.cam_width = 640
        self.cam_height = 320
        self.tracker_lost = False
        self.tracker_lost_timout = 2
        self.capture_flag = True
        self.x_cam = 0
        self.y_cam = 0
        #clear faults
        self._clearFaults()
        time.sleep(0.1)

        # self.reset()

        self.find_transform()

    def _updateTrackerStateCallback(self, data):
        # print(type(data))
        # print(rospy.get_time() - self.t_old)
        self.t_old = rospy.get_time()
        dict = json.loads(data.data)
        self.x_cam = dict['center_psn'][0]
        self.y_cam = dict['center_psn'][1]
        self.tracker_lost = not dict['track_success']
        # print(time.time() - dict['last_frame_time'])

    def _obs(self):
        obs = np.array([self.x_cam/self.cam_width, self.y_cam/self.cam_height, np.cos(self.th1), np.sin(self.th1),  np.cos(self.th2), np.sin(self.th2), self.dth1, self.dth2, self.target_image[0]/self.cam_width, self.target_image[1]/self.cam_height])
        return obs

    def _calcReward(self):
        dist = np.linalg.norm(self.target_image[0:2]/np.array([self.cam_width, self.cam_height])-np.array([self.x_cam, self.y_cam])/np.array([self.cam_width, self.cam_height]))
        return self.dt * (-dist + np.exp(-(dist**2)/0.01))

    def set_target(self):
        # first generate target in robot space
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

        # then transform to image space
        self.target_image = self.transform(self.target.reshape(1,2), self.robot2image_mat ).reshape(2)

        #send it
        message = {
                "target_image" : self.target_image.tolist(),
                "capture_flag" : self.capture_flag
        }

        message = json.dumps(message)


        rospy.loginfo(message)
        self.pub_cam.publish(message)

    def step(self, action):
        """
        action must be in [-1 1] then is scaled to max_action ( rad /s )
        """
        action = self.max_action * np.clip(action, a_min=-1, a_max=1)

        # take action
        jointCmds = self._zeroVelocityVec()
        jointCmds[self.active_joints] = action
        # print(rospy.get_time())

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

        if self.in_boundary == True and (not self.tracker_lost):
            self._publishJointVelCmd(jointCmds)
        else:
            self._stopMovement()
            if self.tracker_lost :
                print('tracker lost')
                success = self.wait_for_tracker(self.tracker_lost_timout)
                if not success:
                    sys.exit('tracker lost track timeout passed')



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

    def wait_for_tracker(self, timeout):
        t0 = time.time()
        success = True
        while self.tracker_lost :
            if (time.time()-t0) > timeout:
                success = False
                break
            time.sleep(0.05)
        return success

    def find_transform(self):
        points_robot = []
        points_image = []
        N = 4
        # collect four measurements
        self._stopMovement()
        rospy.sleep(1.0)
        for i in np.linspace(0,N,N):
            joint_angles = self._zeroAngleVec()
            # print((i - N/2)/ N* 2 * np.pi /6 )
            joint_angles[self.active_joints[0]] = (i - N/2)/ N* 2 * np.pi /8
            joint_angles[self.active_joints[1]] = (i - N/2)/ N* 2 * np.pi /2
            self._send_joint_angles(joint_angles)
            rospy.sleep(3)
            points_robot.append([self.x, self.y])
            points_image.append([self.x_cam, self.y_cam])

        points_robot = np.array(points_robot,dtype=np.float32)
        # points_robot = np.array([[0,0],,dtype=np.float32)
        points_image = np.array(points_image,dtype=np.float32)
        self.imge2robot_mat = cv2.findHomography(points_image, points_robot)[0]
        self.robot2image_mat = np.linalg.inv(self.imge2robot_mat)


        # print(cv2.perspectiveTransform(src=points_image.T, m=self.imge2robot_transform))
        print(self.transform(points_robot, self.robot2image_mat))

    def transform(self,a,H):
        b = H.dot ( np.concatenate((a.T,np.ones((1,a.shape[0]))),axis=0)).T
        b /= b[:,-1:]
        return b[:,:-1]

    def shutdown(self):
        print("shutdown stuff gracefully")
        print("resetting robot configuration")
        self.capture_flag = False
        self.reset()
        print("Unsubscribing Updating State")
        #unregister because we don't need to keep updating state variables
        self.sub.unregister()
        print("Manually shutting down ROS")
        #this call manually stops ROS so any subscribes we forgot should stop
        rospy.signal_shutdown("KinovaEnvironment is shutting down. Stopping Ros")
        sys.exit('Exiting due to shutdown')

if __name__ == '__main__':
    print("hello world")

    reacher = KinovaReacherCamera()
    success = reacher.getIsInitSuccess()

    time.sleep(1)
    if success:
        #*******************************************************************************
        # Make sure to clear the robot's faults else it won't move if it's already in fault
        success &= reacher._clearFaults()
        #*******************************************************************************

        obs = reacher.reset()
        for _ in range(8):
            t0 = rospy.get_time()
            obs = reacher.reset()
            done = False
            for t in range(reacher.episode_length):
                action = np.random.uniform(-1,1,2)
                #if t < reacher.episode_length / 2.0:
                #    action = np.array([0.5, 0.5])
                #else:
                #    action = np.array([-1.0, -1.0])

                obs, r, done, info = reacher.step(action=action)
                if done or rospy.is_shutdown():
                    break
                print(obs, rospy.get_time()-t0)
                t0 = rospy.get_time()
