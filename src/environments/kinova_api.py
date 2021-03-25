import sys
import rospy
import numpy as np
import json
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, String
from environments.time_filter import Subscriber, ApproximateTimeSynchronizer
#from kortex_driver.srv import *
from kortex_driver.srv import PlayJointTrajectory, Base_ClearFaults, SendGripperCommand,SendGripperCommandRequest, PlayJointTrajectoryRequest
from kortex_driver.msg import BaseCyclic_Feedback, Base_JointSpeeds, JointSpeed, Finger, GripperMode, JointAngle


class KinovaInteractor:
    """
    This class deals with sending commands to the kinova, either by publishers or subscribers
    """

    def __init__(self, dt, joint_state_subscriber=None):
        try:
            rospy.init_node('kinova_sender')
        except:
            print("node already exists")

        try:
            # Get node params
            self.robot_name = rospy.get_param('~robot_name', "my_gen3")
            self.degrees_of_freedom = rospy.get_param("/" + self.robot_name + "/degrees_of_freedom", 7)
            self.is_gripper_present = rospy.get_param("/" + self.robot_name + "/is_gripper_present", False)

            out = "Using robot_name {} robot has {} degrees of freedom and is_gripper present is {}".format(self.robot_name, str(self.degrees_of_freedom), str(self.is_gripper_present))
            rospy.loginfo(out)

            # Init clear fault 
            clear_faults_full_name = '/' + self.robot_name + '/base/clear_faults'
            rospy.wait_for_service(clear_faults_full_name)
            self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)

            #Service for resetting position of Kinova
            play_joint_trajectory_full_name = '/' + self.robot_name + '/base/play_joint_trajectory'
            rospy.wait_for_service(play_joint_trajectory_full_name)
            self.play_joint_trajectory = rospy.ServiceProxy(play_joint_trajectory_full_name, PlayJointTrajectory)

            #For sending gripping
            send_gripper_command_full_name = '/' + self.robot_name + '/base/send_gripper_command'
            rospy.wait_for_service(send_gripper_command_full_name)
            self.send_gripper_command = rospy.ServiceProxy(send_gripper_command_full_name, SendGripperCommand)

            # publisher to act on robot ( joint_velocities)
            topic_name = '/' + self.robot_name + '/in/joint_velocity'
            self.pub = rospy.Publisher(topic_name, Base_JointSpeeds , queue_size=1)

            #stop publisher
            topic_name = '/' + self.robot_name +'/in/stop'
            self.stop_command = rospy.Publisher(topic_name, Empty, queue_size=1)


        except Exception as e:
            rospy.logerr("Error thrown during initialization: {}".format(e))
            self.is_init_success = False
        else:
            self.is_init_success = True

        if self.is_init_success:
            self._clearFaults()
            #register handles to stop robot gracefully e.g. killing program
            self.rate = rospy.Rate(1.0 / dt) 
            rospy.on_shutdown(self.shutdown)
 
    def rate_sleep(self):
        self.rate.sleep()

    def get_is_init_success(self):
        return self.is_init_success

    def send_joint_angles(self, angles):
        '''
        angles in rad -pi to pi
        '''
        angles = angles * 180 / np.pi
        #Trajectory generator REQUIRES positive angles
        angles[np.where(angles < 0)] += 360
        # Create the list of angles
        req = PlayJointTrajectoryRequest()
        for i, v in enumerate(angles):
            temp_angle = JointAngle()
            temp_angle.joint_identifier = i
            temp_angle.value = v
            req.input.joint_angles.joint_angles.append(temp_angle)

        # Send the angles
        rospy.loginfo("Sending the robot to angles {}...".format(angles))
        try:
            self.play_joint_trajectory(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call PlayJointTrajectory")
            return False
        else:
            return True

    def publishJointVelCmd(self, jointCmds):
        jointCmd = Base_JointSpeeds()
        jointCmd.duration = 0
        for i, v in enumerate(jointCmds):
            temp_JointSpeed = JointSpeed()
            temp_JointSpeed.joint_identifier = i
            temp_JointSpeed.value = v
            temp_JointSpeed.duration = 0
            jointCmd.joint_speeds.append(temp_JointSpeed)

        self.pub.publish(jointCmd)

    def stopMovement(self):
        rospy.loginfo("Stopping Kinova Movement")
        self.stop_command.publish(Empty())

    def gripper_command(self, value):

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

    def shutdown(self):
            rospy.loginfo("Shutdown Gracefully")
            #unregister because we don't need to keep updating state variables
            self.stopMovement()

            #self.sub.unregister()
            #this call manually stops ROS so any subscribes we forgot should stop
            rospy.signal_shutdown("KinovaEnvironment is shutting down. Stopping Ros")
            sys.exit('Exiting due to shutdown')


class KinovaListener:

    """
        Listens for information from the kinova, as well as for camera information
    """

    def __init__(self, control_joints, num_joints, dt, camera_listener=None):
        try:
            rospy.init_node('kinova_listener')
        except:
            print("node already exists")

        # Get node params
        self.robot_name = rospy.get_param('~robot_name', "my_gen3")
        self.degrees_of_freedom = rospy.get_param("/" + self.robot_name + "/degrees_of_freedom", 7)
        self.is_gripper_present = rospy.get_param("/" + self.robot_name + "/is_gripper_present", False)

        use_camera = camera_listener is not None

        # subscriber to robot feedback message
        topic_name = '/' + self.robot_name +'/base_feedback/joint_state'
        joint_state = Subscriber(topic_name, JointState)

        topic_name = '/' + self.robot_name +'/base_feedback'
        position_feedback = Subscriber(topic_name, BaseCyclic_Feedback)

        subscribers = [joint_state, position_feedback]
        if use_camera:
            print("todo: these should be passed in as inputs")
            camera_feedback = Subscriber(camera_listener, String)#, self._updateTrackerStateCallback)
            subscribers.append(camera_feedback)


        
        self.sub =  ApproximateTimeSynchronizer(subscribers, 1, 
                                                    slop=dt, allow_headerless=True)
        if use_camera:
            print("Getting position from camera")
            self.sub.registerCallback(self.camera_state_listener)
        else:
            print("Getting position from robot")
            self.sub.registerCallback(self.robot_state_listener)


        #Initialize variables to store state information
        self.num_joints = num_joints
        self.control_joints = control_joints

        self.thetas = np.array([0. for _ in range(num_joints)]) # array of all joint angles
        self.thetas_velocity = np.array([0. for _ in range(num_joints)])
        self.position = np.zeros(2 if use_camera else 3)
        self.tracker_lost = False

    
    def get_position_dim(self):
        return len(self.position)

    def get_thetas(self):
        return self.thetas

    def get_angulra_velocity(self):
        return self.thetas_velocity

    def get_controlled_angles(self):
        return self.thetas[self.control_joints]
        
    def get_controlled_angular_velocity(self):
        return self.thetas_velocity[self.control_joints]

    def get_position(self):
        return self.position

    def get_cartesian_robot(self):
        return self.cartesian_end_effector

    def robot_state_listener(self, joint_state, base_feedback):
        if not rospy.is_shutdown():
            self.thetas = np.array(joint_state.position[:self.num_joints])
            self.thetas_velocity = np.array(joint_state.velocity[:self.num_joints])

            #using sensor as opposed to kinematics is different by something like +/- 0.0003  in euclidean distance
            self.position = np.array([base_feedback.base.commanded_tool_pose_x,
                                            base_feedback.base.commanded_tool_pose_y,
                                            base_feedback.base.commanded_tool_pose_z
                                            ])
            self.cartesian_end_effector = self.position

    def camera_state_listener(self, joint_state, base_feedback, camera_feedback):
        if not rospy.is_shutdown():
            self.thetas = np.array(joint_state.position[:self.num_joints])
            self.thetas_velocity = np.array(joint_state.velocity[:self.num_joints])

            dict = json.loads(camera_feedback.data)
            self.position = np.array([dict['center_psn'][0], dict['center_psn'][1]])
            self.cartesian_end_effector = np.array([base_feedback.base.commanded_tool_pose_x,
                                            base_feedback.base.commanded_tool_pose_y,
                                            base_feedback.base.commanded_tool_pose_z
                                            ])

            self.tracker_lost = not dict['track_success']

    def isTrackerLost(self):
        #only useful if using camera
        return self.tracker_lost
