import sys
try:
    #TODO this is a hack, at minimum should be done s.t. it'll work for aaaany ros distribution
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
except Exception as e:
    print(e)
    print("no ros kinetic found in path")

import numpy as np
import rospy
import cv2

class KinovaHomography:
    """
    class that handles transformations between kinova's coordinate system and image coordinate system

    """

    def __init__(self, kinova_sender, kinova_listener, 
                        control_joints, num_joints,
                        robot_preprocess=None,
                        num_pts=4, error_thresh=0.002,
                        save_homography='./homography.npy'):

        self.num_pts = num_pts
        self.kinova_sender = kinova_sender
        self.kinova_listener = kinova_listener
        self.control_joints = control_joints
        self.num_joints = num_joints
        self.error_thresh = error_thresh 
        self.robot_preprocess = self.default_robot_preprocess if robot_preprocess is None else robot_preprocess

        self.save_homography = save_homography
        self.find_transform()

    def default_robot_preprocess(self, coord):
        #default behavior is to just take 1st 2 coordinates
        return np.array([coord[0], coord[1]])

    def find_transform(self):
        load_success = False
        points_robot = []
        points_image = []

        try:
            load_file = np.load(self.save_homography, allow_pickle=True).item()
            print(load_file)
            load_success = True
            points_robot = load_file['robot']
            points_image = load_file['image']
        except Exception as e:
            print("Could not find homography, recalibrating")
            print("error:", e)

        if not load_success:
            N = self.num_pts
            # collect four measurements
            self.kinova_sender.stopMovement()
            rospy.sleep(1.0)
            #TODO: make this more general.... 
            for i in np.linspace(0,N,N):
                joint_angles = np.zeros(self.num_joints)

                joint_angles[self.control_joints[0]] = (i - N/2)/ N* 2 * np.pi /8
                joint_angles[self.control_joints[1]] = (i - N/2)/ N* 2 * np.pi /2

                self.kinova_sender.send_joint_angles(joint_angles)
                thetas = self.kinova_listener.get_thetas()
                while np.any(np.abs(thetas - joint_angles) > self.error_thresh) and not rospy.is_shutdown():
                    thetas = self.kinova_listener.get_thetas()
                    rospy.sleep(0.1)
                rospy.sleep(2.0)

                robot = self.robot_preprocess(self.kinova_listener.get_cartesian_robot())
                camera = self.kinova_listener.get_position()
                
                points_robot.append(robot)
                points_image.append(camera)

            points_robot = np.array(points_robot,dtype=np.float32)
            points_image = np.array(points_image,dtype=np.float32)
            
            save_file = {'robot': points_robot, 'image': points_image}
            np.save(self.save_homography, save_file)

        self.imge2robot_mat = cv2.findHomography(points_image, points_robot)[0]
        self.robot2image_mat = np.linalg.inv(self.imge2robot_mat)
        
        print(self.transform(points_robot, self.robot2image_mat))

    def transform(self,a,H):
        b = H.dot( np.concatenate((a.T,np.ones((1,a.shape[0]))),axis=0)).T
        b /= b[:,-1:]
        return b[:,:-1]

    def robot_to_image(self, coord):
        return self.transform(coord, self.robot2image_mat)

    def image_to_robot(self, coord):
        return self.transform(coord, self.imge2robot_mat)

