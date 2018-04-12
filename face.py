#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 as cv
from time import sleep
from PIL import ImageGrab
import os
import threading

cv.namedWindow('webcam', cv.WINDOW_NORMAL)
cv.resizeWindow('webcam', 1024, 1024)
cv.moveWindow('webcam', 0, 0)

cam = cv.VideoCapture(0)
face_img = cv.imread('face.png')
trows, tcols = face_img.shape[:2]
while True:
    try:
        dev_ok, cam_img = cam.read()
        if dev_ok is not True:
            print('dev status:', dev_ok)
            break

        mark_color = None
        mark_text = ''
        # find face
        method = cv.TM_SQDIFF_NORMED
        result = cv.matchTemplate(face_img, cam_img, method)
        # we want the minimum squared difference
        mn, mx, mnLoc, _ = cv.minMaxLoc(result)
        # Locate possible position
        MPx, MPy = mnLoc
        # Check quality
        need_score = 0.5  # (mx + mn) * 0.3
        if mn > need_score:  # bigger number is worse
            mark_color = (0, 0, 255)
            mark_text = 'Not Found'
        else:
            mark_color = (0, 255, 0)
            mark_text = 'Found'
        # show result
        if mark_text == 'Found':
            cv.rectangle(cam_img, (MPx, MPy), (MPx + tcols, MPy + trows), mark_color, 3)
        cv.putText(cam_img, mark_text, (5, 30), cv.FONT_HERSHEY_DUPLEX, 1, mark_color, 2)
        cv.putText(cam_img, 'Score: ' + str(mn), (5, 60), cv.FONT_HERSHEY_DUPLEX, 1, mark_color, 2)
        cv.putText(cam_img, 'Max: ' + str(mx), (5, 90), cv.FONT_HERSHEY_DUPLEX, 1, mark_color, 2)
        cv.putText(cam_img, 'boundary: ' + str(need_score), (5, 120), cv.FONT_HERSHEY_DUPLEX, 1, mark_color, 2)
        cv.imshow('webcam', cam_img)
        cv.waitKey(1)
    except KeyboardInterrupt:
        break

cam.release()
