#!/usr/bin/env python3
import sys
try:
    #TODO this is a hack, at minimum should be done s.t. it'll work for aaaany ros distribution
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
except Exception as e:
    print(e)
    print("no ros kinetic found in path")

import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS

import imutils 
import cv2
import time
import atexit
import rospy
import os

this_folder_abs_path = os.path.abspath(os.path.dirname(__file__))
class Tracker:
    def __init__(self, source=0, show_position=True):
        self.cap = cv2.VideoCapture(source)
        time.sleep(1)
        self.x = 0
        self.y = 0
        self.success = False
        self.capture_flag = False
        self.capture_counter = 0
        self.show_position = show_position
    def updateTracker(self):

        ret, image = self.cap.read()
        if ret:
            self.last_frame_time = time.time()
            self.frame = image.copy()
            image = cv2.GaussianBlur(image,(3,3),0)
            output = image.copy()
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_red = np.array([30,50,100])
            upper_red = np.array([36,255,255])
            mask = cv2.inRange(hsv, lower_red, upper_red)

            kernel = np.ones((5,5),np.uint8)
            mask = cv2.erode(mask,kernel,iterations = 3)
            mask = cv2.dilate(mask,kernel,iterations = 6)
            res = cv2.bitwise_and(image,image, mask= mask)

            hsv2 = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
            lower_red = np.array([28,18,75])
            upper_red = np.array([40,255,255])
            mask2 = cv2.inRange(hsv2, lower_red, upper_red)
            res2 = cv2.bitwise_and(res,res, mask= mask2)

            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # edge = cv2.Canny(gray, 20, 220)
            # edge = cv2.bitwise_and(edge,edge, mask= mask)
            circles = None
            # circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, dp=1, minDist=10,param1=5, param2=8, minRadius=10, maxRadius=13)
            # cv2.imshow("res", res2)
            I = np.where(mask2 > 0)

            print(len(I[1]))

            # cv2.circle(output, (x, y), 10, (0, 255, 0), 4)
            # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # show the output image
            # cv2.imshow("output", np.hstack([image, output]))
            if len(I[0] > 100) :
                self.x = np.mean(I[1])
                self.y = np.mean(I[0])
                self.success = True
            else:
                # self.x = 0
                # self.y = 0
                self.success = False
        else:
            # self.x = 0
            # self.y = 0
            self.success = False

    def getTrackerCenter(self):
        return (self.x, self.y)

    def render(self,target=None, capture_flag=False, save_path='./video.avi'):
        if self.capture_flag == False and capture_flag == True:
            print('starting capture', save_path)
            self.capture_flag = True
            self.writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640,480))

        if self.capture_flag == True and capture_flag == False:
            self.capture_flag = False
            self.writer.release()

        output = self.frame.copy()
        x = np.int32(self.x)
        y = np.int32(self.y)
        #if self.show_position:
            #cv2.circle(output, (x, y), 10, (255, 100, 0), 4)
            #cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        if target is not None:
            x = np.int32(target[0])
            y = np.int32(target[1])
            cv2.circle(output, (x, y), 10, (0, 255, 0), -1)
        cv2.imshow('output', output)
        key = cv2.waitKey(1) & 0xFF
        # if key == ord('S'):
        #     self.writer_flag
        if self.capture_flag:
            print("capturing", save_path)
            self.writer.write(output)

    def getDims(self):
        return self.frame.shape
