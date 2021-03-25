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
from KinovaReacher import KinovaReacher
from CameraReward import CameraReward
import time


class KinovaInternalCameraReacher(KinovaReacher):

    def __init__(self, dof=7, episode_length=100, dt = 0.05, max_action=np.array([0.5,0.5]),
            initBB=None, tracker_type='kcf', width_scale=500, src=0):

        self.cam_rew = CameraReward(initBB=initBB, tracker_type=tracker_type, width_scale=width_scale, src=src)
        self.curr_rew = 0.0

        self.target_psn = np.array([0.0, 0.0])
        self.center_psn = np.array([0.0, 0.0])
        self.updateTrackerState()

        super(KinovaInternalCameraReacher,self).__init__(dof, episode_length, dt, max_action)
        
        if self.is_init_success:
            #Set-up Camera Reward
            # subscriber to robot feedback message
            #self.sub.unregister()
            # publisher to act on robot ( joint_velocities)
            topic_name = '/' + self.robot_name + '/in/joint_velocity'
            self.pub = rospy.Publisher(topic_name, Base_JointSpeeds , queue_size=1)

            topic_name = '/' + self.robot_name +'/base_feedback/joint_state'
            self.sub_cam = rospy.Subscriber(topic_name, JointState, self._updateTrackerStateCallback)
            rospy.on_shutdown(self.shutdown)

    def updateTrackerState(self):
        self.cam_rew.updateTracker()

        #Update position and center information
        self.target_psn = self.cam_rew.getTarget()
        center = self.cam_rew.getTrackerCenter() 
        self.center_psn = np.array([center[1], center[0]])

        #calculate the reward
        self.curr_rew = self.cam_rew.calcReward()

    def _updateTrackerStateCallback(self, data):
        #intention is to align updating values with Frame from Camera
        #Update tracker
        self.updateTrackerState()
    
    def _obs(self):
        target = self.cam_rew.getTarget()
        obs = np.array([self.center_psn[0], self.center_psn[0], np.cos(self.th1), np.sin(self.th1),  np.cos(self.th2), np.sin(self.th2), self.dth1, self.dth2, target[0], target[1]])
        return obs

    def reset(self):
        super(KinovaInternalCameraReacher, self).reset()
        if not rospy.is_shutdown():
            rospy.sleep(1)
            self.cam_rew.resetTracker()

        return self._obs() 
    
    def step(self, action):
        action = self.max_action * np.clip(action, a_min=-1, a_max=1)

        # take action
        jointCmds = self._zeroVelocityVec()
        jointCmds[self.active_joints] = action
        # print(rospy.get_time())
        self.in_boundry = self.check_boundary(self.angle_boundary_low, self.angle_boundary_high, self.all_joints, jointCmds, self.dt)
        if self.in_boundry:
            self._publishJointVelCmd(jointCmds)
        else:
            self._stopMovement()

        time.sleep(self.dt)

        # rospy.spin()
        observation = self._obs()
        rewards = self.curr_rew 
        self.steps += 1
        done  = self.steps >= self.episode_length or not self.cam_rew.getSuccess()
                
        return observation, rewards, done, None

    def render(self):
        #this will screw up action/state alignment
        self.cam_rew.render()

    def shutdown(self):
        print("Shutdown KinovaExplicitCameraReacher")
        super(KinovaInternalCameraReacher, self).shutdown()
        self.sub_cam.unregister()


if __name__ == "__main__":
    print("Validating CameraReacherEnv")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str,
            help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="kcf",
            help="OpenCV object tracker type")
    ap.add_argument("-record", "-r", action="store_true",default=False,
            help="flag to record video")
    ap.add_argument("-genTarg", action="store_true",default=False,
            help="flag to keep generating random targets")
    ap.add_argument("-source", "-s", type=int, default=-1,
            help="specify camera source")
    args = vars(ap.parse_args())

    print(args.keys())

    reacher = KinovaInternalCameraReacher(src=args["source"], tracker_type=args["tracker"])
    success = reacher.getIsInitSuccess()
    time.sleep(1)
    if success:
        #*******************************************************************************
        # Make sure to clear the robot's faults else it won't move if it's already in fault
        success &= reacher._clearFaults()
        #*******************************************************************************

        obs = reacher.reset()
        for e in range(10):
            obs = reacher.reset()
            for t in range(reacher.episode_length):
                if not rospy.is_shutdown():
                    t0 = rospy.get_time()
                    action = np.random.uniform(-1,1,2)
                    obs, r, done, info = reacher.step(action=action)
                    reacher.render()
                    if done:
                        break
                    print(obs,r, rospy.get_time()-t0)
