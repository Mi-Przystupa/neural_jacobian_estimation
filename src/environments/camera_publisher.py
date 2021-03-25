#!/usr/bin/env python
try:
    from simple_tracker import Tracker
except:
    from environments.simple_tracker import Tracker
    
from std_msgs.msg import String
import numpy as np
import rospy
import json
import time

class CameraPublisher:
    def __init__(self, source=0, show_position=True):
        rospy.init_node('camerapublisher', anonymous=True)
        self.pub = rospy.Publisher('tracker_status', String, queue_size=1)
        self.sub = rospy.Subscriber('target_image', String, self.listen,queue_size=1)
        self.rate = rospy.Rate(40) 
        self.tracker =  Tracker(source=source,  show_position= show_position)
        self.target_image = None
        self.capture_flag = False
        self.save_path = './video.avi'


    def publish(self):
        while not rospy.is_shutdown():
            #Update position and center information
            # if self.tracker.success == False:
            #     self.tracker.initTracker()

            self.tracker.updateTracker()

            self.tracker.render(self.target_image, self.capture_flag, save_path = self.save_path)

            center = self.tracker.getTrackerCenter()
            
            video_size = self.tracker.frame.shape
            center_psn = [0, 0] # np.array([center[0], center[1]])
            center_psn[0] = center[0]
            center_psn[1] = center[1]
            print("video size", video_size, "center", center, self.target_image)
            #480, 640 is output of camera dimensions

            dims = self.tracker.getDims()
            message = {
                    'center_psn': center_psn, #.tolist(),
                    'width': dims[1],
                    'height': dims[0],
                    'track_success' : self.tracker.success,
                    'last_frame_time' : self.tracker.last_frame_time,
                    'publish_time' : time.time()
            }

            message = json.dumps(message)

            rospy.loginfo(message)
            self.pub.publish(message)
            self.rate.sleep()

    def listen(self, data):
        dict = json.loads(data.data)
        self.target_image = dict['target_image']
        self.capture_flag = dict['capture_flag']
        self.save_path = dict['save_path']

class CameraUpdateTrackerInfo:
    def __init__(self, camera_publisher):
        self.pub = rospy.Publisher(camera_publisher, String, queue_size=1)

    def publish(self, message):
        self.pub.publish(message)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tracker", type=str, default="kcf",
            help="OpenCV object tracker type")
    ap.add_argument("-source", "-s", type=int, default=0,
            help="specify camera source")
    ap.add_argument('-show_position', type=bool, default=1,
            help="to show position")
    args = vars(ap.parse_args())
    print(args.keys())

    tracker_type = args["tracker"]
    src = args["source"]
    try:
        publisher = CameraPublisher(args["source"], args["show_position"])

        publisher.publish()
    except rospy.ROSInterruptException:
        pass

