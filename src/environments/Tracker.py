#!/usr/bin/env python

#this class is based on tutorial code from the following link:
#https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/

import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
except Exception as e:
    print("no ros kinetic found in path")

import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS

import imutils #pip install --upgrade imutils
import cv2
import time
import atexit
import rospy
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from camera_stuff import detector


class Tracker:
    #
    OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create, #want accuracy and tolerate lower fps
            "kcf": cv2.TrackerKCF_create, #happy medimum of fps vs accuracy
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create, #author did not recommend
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create #fast tracking(more fps), less accurate
    }

    def __init__(self, initBB=None, tracker_type='kcf', width_scale=640, src=0, handleTarget=False, attempt_autoinit=50):
        #take first camera it can

        self.vs = VideoStream(src=src).start()

        time.sleep(1.0)
        self.width_scale = width_scale
        #initialized after you get the frame
        self.dims = None
        self.updateDims()

        self.attempt_autoinit = attempt_autoinit
        self.tracker_type = tracker_type

        self.initTracker(initBB)

        self.box = self.initBB
        self.success = True

        if handleTarget:
            #define bounds where robot is located
            print("Specify where robot exists")
            self.robotBound = self.specifyBoundingBox()


            #define target Generation box

            print("Specify where targets should be generated in image")
            self.targetRange = self.specifyBoundingBox()

            #specify Target
            self.targ_psn = np.array([0, 0])
            self.generateTarget()
        else:
            self.handleTarget = handleTarget

        #fps information
        self.fps = FPS()
        self.fps.start()

        #clean up
        atexit.register(self.shutdown)

    def init_box(self):
        for _ in range(self.attempt_autoinit):
            frame = self.readFrame()
            success, self.initBB = detector.detect(frame)
            if success:
                break
            time.sleep(0.05)
        success = False
        return success

    def initTracker(self, initBB=None):
        #Set-up Tracker
        if initBB is None:
            success = self.init_box()
            #initialize tracker bounding box if unspecified
            if not self.init_box():
                self.initBB = self.specifyBoundingBox()
        else:
            self.initBB = initBB
        frame = self.readFrame()
        self.curr_frame = frame
        self.tracker = self.OPENCV_OBJECT_TRACKERS[self.tracker_type]()
        self.tracker.init(frame, self.initBB)

    def specifyBoundingBox(self):
        frame = self.readFrame()
        boundingBox = cv2.selectROI("Frame", frame,fromCenter=False,
                    showCrosshair=True)
        return boundingBox

    def inRobotBox(self, v_w, v_h):
        (x, y, w, h) = [int(v) for v in self.robotBound]
        inWidth = x <= v_w and v_w <= x + w
        inHeight = y <= v_h and v_h <= y + h
        return inWidth and inHeight

    def getTargetRange(self):
        (w_low,h_low, w, h) = [int(v) for v in self.targetRange]
        w_high = w_low + w
        h_high = h_low + h
        return w_low, w_high, h_low, h_high

    def generateTarget(self):
        if self.handleTarget:
            w_low, w_high, h_low, h_high = self.getTargetRange()
            targ_h = np.random.uniform(h_low, h_high)
            targ_w = np.random.uniform(w_low, w_high)
            while self.inRobotBox(targ_w, targ_h):
                #generate target outside bounding box
                targ_h = np.random.uniform(h_low, h_high)
                targ_w = np.random.uniform(w_low, w_high)

            self.targ_psn = np.array([targ_h, targ_w])
        else:
            self.targ_psn = np.array([0.0, 0.0])

    def resetTracker(self):
        self.box = self.initBB
        self.initTracker()

    def getTarget(self):
        return self.targ_psn

    def readFrame(self):
        frame = self.vs.read()
        frame = imutils.resize(frame, width=self.width_scale)
        return frame

    def updateDims(self):
        frame = self.readFrame()
        (H, W) = frame.shape[:2]
        self.dims = (H, W)

    def getDims(self):
        return self.dims

    def setWidthScale(self, width_scale):
        self.width_scale = width_scale
        self.updateDims()

    def getSuccess(self):
        return self.success

    def updateTracker(self):
        frame = self.readFrame()
        (self.success, self.box) = self.tracker.update(frame)
        self.curr_frame = frame

        #collect FPS information
        self.fps.update()
        self.fps.stop()

        return self.success

    def calcReward(self):
        center = self.getTrackerCenter()
        #it's backwards
        center = np.array([center[1], center[0]])
        diff = self.targ_psn - center
        return -np.linalg.norm(diff, 2)

    def getTrackerCenter(self):
        (x, y, w, h) = [int(v) for v in self.box]
        return np.array([x + w / 2, y + h / 2])

    def getInfo(self):
        info = [
            ("Tracker", self.tracker_type),
            ("Success", "Yes" if self.success else "No"),
            ("FPS", "{:.2f}".format(self.fps.fps())),
            ("Reward", "{:.2f}".format(self.calcReward() if self.handleTarget else 0.0))
        ]
        return info

    def render(self):
        #Visualize tracking box
        (x, y, w, h) = [int(v) for v in self.box]
        cv2.rectangle(self.curr_frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2)
        center = self.getTrackerCenter()
        cv2.circle(self.curr_frame, (center[0], center[1]), radius=5, color=(0, 255,0),thickness=-1)

        if self.handleTarget:
            #visualize center of box
            targ_h = int(self.targ_psn[0])
            targ_w = int(self.targ_psn[1])
            cv2.circle(self.curr_frame, (targ_w, targ_h), radius=5, color=(0, 255,0),thickness=-1)

            #Visualize area no target should appear
            (x, y, w, h) = [int(v) for v in self.robotBound]
            cv2.rectangle(self.curr_frame, (x, y), (x + w, y + h),
                    (0, 0, 255), 2)

            #Visualize area targets generated 
            (x, y, w, h) = [int(v) for v in self.targetRange]
            cv2.rectangle(self.curr_frame, (x, y), (x + w, y + h),
                    (255, 0, 0), 2)

        if self.handleTarget:
            #visualize center of box
            targ_h = int(self.targ_psn[0])
            targ_w = int(self.targ_psn[1])
            cv2.circle(self.curr_frame, (targ_w, targ_h), radius=5, color=(0, 255,0),thickness=-1)

            #Visualize area no target should appear
            (x, y, w, h) = [int(v) for v in self.robotBound]
            cv2.rectangle(self.curr_frame, (x, y), (x + w, y + h),
                    (0, 0, 255), 2)

            #Visualize area targets generated
            (x, y, w, h) = [int(v) for v in self.targetRange]
            cv2.rectangle(self.curr_frame, (x, y), (x + w, y + h),
                    (255, 0, 0), 2)

        info = self.getInfo()
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(self.curr_frame, text, (10, self.dims[0] - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        cv2.imshow("Frame", self.curr_frame)
        #if you don't call waitKey the screen immediately disappears
        key = cv2.waitKey(1) & 0xFF

        return self.curr_frame


    def shutdown(self):
        print("shut down the show, kill everything")
        #clean up video stream and all that
        #kill video stream
        self.vs.stop()
        #shutdown any openCV windows
        cv2.destroyAllWindows()

if __name__ == "__main__":
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
    ap.add_argument("-source", "-s", type=int, default=1,
            help="specify camera source")
    args = vars(ap.parse_args())
    print(args.keys())


    # extract the OpenCV version info
    (major, minor) = cv2.__version__.split(".")[:2]
    print(major, minor)
    cam_rew = Tracker(src=args["source"], tracker_type=args["tracker"])
    dims = cam_rew.getDims()
    writer = None
    if args["record"]:
        forcc = cv2.VideoWriter_fourcc(*"MJPG")

        writer = cv2.VideoWriter("example.avi", forcc, 20,
                            (int(dims[1]), int(dims[0])), True)
    while True:
        success = cam_rew.updateTracker()
        frame = cam_rew.render()
        if args["genTarg"]:
            cam_rew.generateTarget()
        if writer is not None:
            writer.write(frame)
        if not success:
            break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

