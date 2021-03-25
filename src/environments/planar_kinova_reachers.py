from environments.kinova_3D_reaching import FullDOFKinovaReacher
from environments.kinova_env import Bound
from environments.kinova_api import KinovaListener
from environments.kinova_homography import KinovaHomography
from environments.camera_publisher import CameraUpdateTrackerInfo

import json
import numpy as np
import gym
from gym import spaces

class TwoJointPlanarKinova(FullDOFKinovaReacher): 
    CONTROL_JOINTS = [3, 5] #joints we control is joint 4 & 6 but is indexed from 0

    TWO_DOF_DEFAULT_BOUNDARY = [
                    Bound( -0.0, 0.0), 
                    Bound( -0.0, 0.0), 
                    Bound( -0.0, 0.0), 
                    Bound( -2.0, 2.0),  #control joint 1
                    Bound( -0.0, 0.0), 
                    Bound( -2.0, 2.0), #control joint 2
                    Bound( -0.0, 0.0), 
                ]
    JOINT_BOUNDS = TWO_DOF_DEFAULT_BOUNDARY
    TARGET_DEFAULT_BOUNDS = TWO_DOF_DEFAULT_BOUNDARY

    L1 = 0.3143
    L2 = 0.1674 + 0.120
    Z_OFFSET = 0.7052

    """
        Move robot in it's (Y, Z) plane
        We can pretend it is (Y=X, Z = Y) for interpretation purposes
    """

    def __init__(self,  dt=0.05,
                        max_velocity=None,
                        joint_bounds=TWO_DOF_DEFAULT_BOUNDARY,
                        target_bounds=TWO_DOF_DEFAULT_BOUNDARY,
                        H=100,
                        target_mode = 'kinematic',
                        fixed_target = [ 0.4738, 0.188],
                        reset_error_threshold=0.001, #worked well empirically, too small and you'll get caught in a loop, could be as high as 0.02
                        reset_joint_angles= np.array([-0.5, 0.5]),
                        reward_type='l2,precision',
                        precision_tau = 0.01):

        super(TwoJointPlanarKinova, self).__init__(
                        dt=dt,
                        max_velocity=max_velocity,
                        joint_bounds=joint_bounds,
                        target_bounds=target_bounds,
                        H=H,
                        target_mode = target_mode,
                        fixed_target = [fixed_target[1] , 0., fixed_target[0] + self.Z_OFFSET],
                        reset_error_threshold=0.001, #worked well empirically, too small and you'll get caught in a loop, could be as high as 0.02
                        reset_joint_angles=np.array([0., 0.0, 0.0, reset_joint_angles[0], 0.0, reset_joint_angles[1], 0]),
                        reward_type='l2,precision',
                        precision_tau = precision_tau)
        self.action_space = spaces.Box(low=-self._max_velocity, high=self._max_velocity, shape=(len(self.CONTROL_JOINTS),))
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=(len(self._obs()),), dtype=np.float32)

        #TODO: I have no idea why, but occasionally I observe the inverse jacobian controller completely fails occasionally, usually boundary targets though

    def _transform_coords(self, vec):
        #Converts (x,y,z) ==> (x,y) plane treating new_x = z, new_y = x
        new_x = vec[2] - self.Z_OFFSET
        new_y = vec[0]
        return np.array([new_x, new_y])

    def get_cartesian(self): 
        position = self.kinova_listener.get_position()
        return self._transform_coords(position)

    def get_target(self):
        coords = self._transform_coords(self.target)
        return np.array([coords[0], coords[1]])  

    def step(self, action):
        #check action dimensions, if full angle, extract only ones we need
        #else, reshape to be 7 DOF & send to super
        if len(action) == 2:
            temp = np.zeros(self.NUM_JOINTS)
            temp[self.CONTROL_JOINTS[0]] = action[0]
            temp[self.CONTROL_JOINTS[1]] = action[1]
            action = temp

        return super(TwoJointPlanarKinova, self).step(action)


class TwoJointVisualPlanarKinova(TwoJointPlanarKinova):

    def __init__(self, dt=0.05,
                        max_velocity=None,
                        joint_bounds=None,
                        target_bounds=None,
                        H=100,
                        target_mode = 'kinematic',
                        fixed_target = np.array([150, 150]),
                        reset_error_threshold=0.001, #worked well empirically, too small and you'll get caught in a loop, could be as high as 0.02
                        reset_joint_angles= np.array([-0.5, 0.5]),
                        reward_type='l2,precision',
                        precision_tau = 0.01,
                        camera_listener='tracker_status', #environment listens for info on tracker
                        camera_publisher='target_image', #in environment to publish information about target img   
                        video_save_path = 'vide.avi'
                        ):

        super(TwoJointVisualPlanarKinova, self).__init__(
                                                        dt = dt,
                                                        max_velocity=max_velocity,
                                                        joint_bounds= joint_bounds,
                                                        target_bounds= target_bounds,
                                                        H= H,
                                                        target_mode = target_mode,
                                                        fixed_target = fixed_target,
                                                        reset_error_threshold= reset_error_threshold,
                                                        reset_joint_angles=  reset_joint_angles,
                                                        reward_type= reward_type,
                                                        precision_tau = precision_tau
                                                        )

        self.kinova_listener = KinovaListener(self.CONTROL_JOINTS, self.NUM_JOINTS, dt, camera_listener)

        self.homography = KinovaHomography(self.kinova_commands, self.kinova_listener, 
                                            self.CONTROL_JOINTS, self.NUM_JOINTS, 
                                            robot_preprocess=self._transform_coords)

        self.pub_cam = CameraUpdateTrackerInfo(camera_publisher) 
        self.target = fixed_target
        print("camera dimensions are fixed in initialization")
        self.cam_width = 640
        self.cam_height = 480



        self.action_space = spaces.Box(low=-self._max_velocity, high=self._max_velocity, shape=(len(self.CONTROL_JOINTS),))
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=(len(self._obs()),), dtype=np.float32)

        self.video_save_path = video_save_path


    def set_target(self):
        #Get target from robot coordinate frame as always
        target = super(TwoJointVisualPlanarKinova, self).set_target()

        if self.use_target_as_is:
            #work around for experiment when loading targets
            self.target = target
        else:
            # then transform to image space
            target = self._transform_coords(target).reshape(1, 2)
            self.target = self.homography.robot_to_image(target).reshape(2)

        self.publish_to_camera(True, self.video_save_path)
        
    def publish_to_camera(self, capture_target, save_path='./video.avi'):
        #send it
        message = {
                "target_image" : self.target.tolist(),
                "capture_flag" : capture_target, 
                "save_path" : save_path
        }

        message = json.dumps(message)
        repeat = 10
        for _ in range(repeat):
            self.pub_cam.publish(message)

    def normalize_cam_coords(self, coords):
        coords[0] = coords[0] /self.cam_width
        coords[1] = coords[1] / self.cam_height
        return coords


    def get_cartesian(self): 
        position = self.kinova_listener.get_position().copy()
        if len(position) <= 2: #initialization issue where coords are in actual cartesian
            position = self.normalize_cam_coords(position)
        return position

    def get_target(self):
        target = self.target.copy()
        if len(target) <= 2: #initilization issue, this is work around
            target = self.normalize_cam_coords(target)
        return target 

    def close(self):
        self.publish_to_camera(False)
        super(TwoJointVisualPlanarKinova, self).close()
 
