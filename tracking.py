"""
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: © 2022 Renaud Bastien, Stephane Sanchez <stephane.sanchez@ut-capitole.fr>
Université Toulouse Capitole - Institut de Recherche en Informatique de Toulouse
Real Expression Artificial Life (REVA) Research Team
https://www.irit.fr/departement/calcul-intensif-simulation-optimisation/reva/
"""

# -*- coding: utf-8 -*-

import datetime
from datetime import timedelta
import ctypes as ct
import math
from tkinter import Y
from cv2 import fastNlMeansDenoising
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from Camera import camera3d as cam
import cv2
import time
import os
import pyrealsense2 as rs
import threading
from datetime import date
import sys
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
import socket
import random
from collections import deque
from collections.abc import Iterable
from filterpy.kalman import KalmanFilter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import Geometry3D as g3d

import tracking_parameters as tp
from FishTracker import fishTracker as ft

from PIL import Image

import os

# importlib.reload(matrixNet)
# importlib.reload(cam)
# importlib.reload(tools)
# importlib.reload(ft)

WRITING_HOST = "127.0.0.1"  # The server's hostname or IP address
WRITING_PORT = 65432        # The port used by the server
writing_socket = None

#pathFile = os.path.normpath(os.path.dirname(os.path.realpath(__file__)) + "\\Data")
pathFile = "./Data"
#print("File path : " + pathFile+"\n")
todayFolderPath = ""
experimentFolder = ""
tracked_filename = ""
world_filename = ""

cameraThread = None
frames_stream = []
stream_config = []   # 0 : depth = True/False, 1 : rgb = True/False, 
                     # 2 : ir = True/False, 3 : show = "depth"/"rgb"/"ir"
stream_state = 'pause' # 'pause', 'play', 'stop'

trackvisuThread = None
tracking_stream = []
tracking_stream_state = 'pause' # 'pause', 'play', 'stop'

depthRes = 2 #848x480, 30 fps, refers to cam.resDepth
irRes = 2 #848x480, 30 fps, refers to cam.resDepth
rgbRes = 4 #848x480, 30 fps, refers to cam.resRGB

#Tracking data string
blob_data = []
tracked_data = []
world_data = []
writing_state = 'pause' # 'pause', 'play', 'stop'

depthRes = 2 #848x480, 30 fps, refers to cam.resDepth
irRes = 2 #848x480, 30 fps, refers to cam.resDepth
rgbRes = 4 #848x480, 30 fps, refers to cam.resRGB

displayFrameIndex = 0
trackingFrameIndex = 0

visu = False

playback = None
config = None

def round(x, base=1):
    return base * np.round(x/base)

def round(x,base = 1):
    return base * np.round(x/base)

def defineKalman(P = 1.0,Q = .01,R =0.01,dt = 1/tp.cst_framerate,dim = 4):
    nx = 2*dim
    f = KalmanFilter (nx, dim)
    f.x = np.zeros((nx,))

    f.F = np.eye(nx) + dt*np.eye(nx,k = dim)


    f.H = np.eye(dim,nx)


    f.Q = np.zeros((nx,nx))
    T33 = dt ** 3 / 3.0
    T22 = dt ** 2 / 2.0
    T = dt
    for k in range(0,dim):
        f.Q[k,k] = T33
        f.Q[k+dim,k+dim] = T
        f.Q[k+dim,k] = T22
        f.Q[k,k+dim] = T22
    f.Q = Q* f.Q
    f.P *= P
    f.R = np.eye(dim)*R

    return f 

def computeDistanceWithDisparity(xL, xR):
    distance = 0
    
    disparity = xL - xR + tp.cst_disparity_shift

    if disparity>0:
        baseline = 0.050183285 #m #50.183285 mm  #50.0 #50.183285 #mm
        focal = 423.574 #0.5 * 848 / math.tan(math.pi/4) #423.574 #pixels
        distance = focal * (baseline / disparity) #* depth_scale    #meters ?

    return disparity, distance

def computeDistanceWithDisparity2(disparity):
    distance = 0
    
    if disparity>0:
        baseline = 0.050183285 #m #50.183285 mm  #50.0 #50.183285 #mm
        focal = 423.574 #0.5 * 848 / math.tan(math.pi/4) #423.574 #pixels
        distance = focal * (baseline / disparity) #* depth_scale    #meters ?

    return distance

def computeWaterDistanceToCam():
    wpoint = g3d.Point(tp.cst_watersurface_point[0], tp.cst_watersurface_point[1], tp.cst_watersurface_point[2])
    wnorm = g3d.Vector(tp.cst_watersurface_norm[0], tp.cst_watersurface_norm[1], tp.cst_watersurface_norm[2])
    wplane = g3d.Plane(wpoint, wnorm)

    campoint = g3d.Point(0., 0., 0.)
    waterdistance = g3d.distance(wplane, campoint)
    return waterdistance

def computeWaterDistanceToCenter():
    wpoint = g3d.Point(tp.cst_watersurface_point[0], tp.cst_watersurface_point[1], tp.cst_watersurface_point[2])
    wnorm = g3d.Vector(tp.cst_watersurface_norm[0], tp.cst_watersurface_norm[1], tp.cst_watersurface_norm[2])
    wplane = g3d.Plane(wpoint, wnorm)

    cpoint = g3d.Point(tp.cst_bowl[0], tp.cst_bowl[1], tp.cst_bowl[2])
    waterdistance = g3d.distance(wplane, cpoint)
    return waterdistance


def createFolder(path,params = []):
    if not os.path.exists(path):
        os.makedirs(path)
    for param in params:
        path = path + '/' + str(param)
        if not os.path.exists(path):
            os.makedirs(path)
    return path
        
def checkPath(path,params = []):
    for param in params:
        path = path + '/' + str(param)
    return path

def connectWritingServer():
    print("\n... Connecting to writing server ..\n")
    global writing_socket
    writing_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    writing_socket.connect((WRITING_HOST, WRITING_PORT))
    
def sendWritingData(dataToSend):    
    global writing_socket
    message = dataToSend.encode('utf-8')
    try:
        writing_socket.sendto(message, (WRITING_HOST, WRITING_PORT))
    except:
        pass
   
def showCameraStream():
    # Create opencv window to render image in
    cv2.namedWindow("Camera Stream", cv2.WINDOW_AUTOSIZE)
    
    # depth = True
    # rgb = True
    # ir = True
    # show = 'rgb'
    
    global stream_config
    global frames_stream
    global stream_state
    global tracking_stream_state

    global trackvisuThread

    while True:
        if stream_state == 'play' or stream_state == 'pause' :
            if len(stream_config) > 3 and stream_state == 'play':
                depth = stream_config[0]
                rgb = stream_config[1]
                ir = stream_config[2]
                show = stream_config[3]
            
                if len(frames_stream) > 0:
                    frames = frames_stream.pop(0)
                    #Get frames
                    i = 0
                    if depth:
                        depth_image = frames[i]
                        i += 1
                    if rgb:
                        rgb_image = frames[i]
                        i += 1
                    if ir:
                        ir_image = frames[i]
                        i += 1    
                    
                    if show == 'depth':
                        cv2.imshow("Camera Stream", depth_image)    
                    elif show == 'rgb':
                        image = rgb_image
                        # Render image in opencv window
                        cv2.imshow("Camera Stream", image)
                    elif show == 'ir':
                        image = ir_image
                        # Render image in opencv window
                        cv2.imshow("Camera Stream", image)
        
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                message = f"close"
                sendWritingData(message)
                writing_socket.close()
                saveTrackingData()
                tracking_stream_state = 'stop'
                if trackvisuThread is not None:
                    trackvisuThread.join()
                    cv2.destroyAllWindows()
                    sys.exit(0)
                else:
                    cv2.destroyAllWindows()
                    sys.exit(0)
        if stream_state == 'stop':
            cv2.destroyAllWindows()
            break

def showCameraStream_dyn():
    # Create opencv window to render image in
    cv2.namedWindow("Camera Stream", cv2.WINDOW_AUTOSIZE)
    
    global stream_config
    global frames_stream
    global stream_state
    global writing_socket

    global displayFrameIndex

    global tracking_stream_state

    global trackvisuThread

    while True:
        if stream_state == 'play' or stream_state == 'pause' :
            if stream_state == 'play':
                if len(frames_stream) > 0:
                    frames = frames_stream.pop(0)
                    # Safety
                    displayFrameIndex = min(displayFrameIndex, len(frames)-1)
                    #Get frames
                    image = frames[displayFrameIndex]
                    frames_stream = []
                    cv2.imshow("Camera Stream", image)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                message = f"close"
                sendWritingData(message)
                writing_socket.close()
                saveTrackingData()
                tracking_stream_state = 'stop'
                if trackvisuThread is not None:
                    trackvisuThread.join()
                    cv2.destroyAllWindows()
                    sys.stdout.flush()
                    sys.exit(0)
                else:
                    cv2.destroyAllWindows()
                    sys.stdout.flush()
                    sys.exit(0)
            # Press 'n'' to switch frame
            elif key & 0xFF == ord('n') and len(frames) > 0 and stream_state == 'play':
                displayFrameIndex = (displayFrameIndex+1)%len(frames)        
        elif stream_state == 'stop':
            cv2.destroyWindow("Camera Stream")
            break

def showTrackingStream():
    # Create opencv window to render image in
    cv2.namedWindow("Tracking Stream", cv2.WINDOW_AUTOSIZE)
    #cv2.moveWindow("Tracking Stream", 40,30)
    
    global tracking_stream
    global tracking_stream_state
    global writing_socket


    global trackingFrameIndex
    
    global stream_state
    global cameraThread

    while True:
        if tracking_stream_state == 'play' or tracking_stream_state == 'pause' :
            if (tracking_stream_state == 'play'):
                if len(tracking_stream) > 0:
                    frames = tracking_stream.pop(0)
                    # Safety
                    trackingFrameIndex = min(trackingFrameIndex, len(frames)-1)
                    #Get frames
                    image = frames[trackingFrameIndex]
                    tracking_stream = []          
                    cv2.imshow("Tracking Stream", image)
                      
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                message = f"close"
                sendWritingData(message)
                writing_socket.close()
                saveTrackingData()
                stream_state = 'stop'
                if cameraThread is not None:
                    cameraThread.join()
                    cv2.destroyAllWindows()
                    sys.stdout.flush()
                    sys.exit(0)
                else:
                    cv2.destroyAllWindows()
                    sys.stdout.flush()
                    sys.exit(0)
            # Press 'n'' to switch frame
            elif key & 0xFF == ord('n') and tracking_stream_state == 'play':
                if len(frames) > 0 :
                    trackingFrameIndex = (trackingFrameIndex+1)%len(frames)
        elif tracking_stream_state == 'stop':
            cv2.destroyWindow("Tracking Stream")
            break
        
def saveTrackingData():
    print("\n. Start saving tracking data .")
    global tracked_data
    global world_data
    global experimentFolder

    global tracked_filename
    global world_filename

    tname = ""
    wname = ""

    if tracked_filename != "":
        tname = tracked_filename[0:-4] +"_safe.csv"
    else:
        tname = experimentFolder + "/tracked_safe.csv"
    
    if world_filename == "":
        wname = experimentFolder + "/world_safe.csv"
    else:
        wname = world_filename[0:-4] +"_safe.csv"

    tracked_file = open(tname, 'a')
    world_file = open(wname, 'a')
    
    #tracked_file = open(experimentFolder + "/tracked_safe.csv", 'a')
    #world_file = open(experimentFolder + "/world_safe.csv", 'a')

    while len(tracked_data) > 0:
        #tracked_file = open(experimentFolder + "/tracked.data", 'a')
        s = tracked_data.pop(0)
        tracked_file.write(s)
        #tracked_file.flush()
    tracked_file.close()
    
    while len(world_data) > 0:
        #world_file = open(experimentFolder + "/world.data", 'a')
        s = world_data.pop(0)
        world_file.write(s)
        #world_file.flush()
    world_file.close()
    print("\n. Saving tracking data done.")

def createBlobDetector():
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = False
    
    #Laplacian Threshold
    params.minThreshold = 0
    params.maxThreshold = 300
    
    #Area
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 60

    #Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1
    params.maxCircularity = 0.7
    
    #Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87
    params.maxConvexity = 1.0
    
    #Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 0.25
    
    detector = cv2.SimpleBlobDetector_create(params)
    return detector

def backGroundSubstractor(frames, type='KNN'):
    if type == 'MOG2':
        fgbg = cv2.createBackgroundSubtractorMOG2()
        fgbg.setDetectShadows(False)
        #fgbg.setHistory(5)
        #fgbg.setNMixtures(5)
        #fgbg.setVarThreshold(16)
        #fgbg.setBackgroundRatio(0.1) 
    else:
        fgbg = cv2.createBackgroundSubtractorKNN()
        fgbg.setDetectShadows(False)
        #fgbg.setDist2Threshold(100.0)
        #fgbg.setHistory(5)
    
    for frame in frames:
        fgmask = fgbg.apply(frame, learningRate=-1)
    return fgbg

def backgroundSubstract(fgbg, frame, learningRate=-1):
    tmp = np.copy(frame)
    fgmask=fgbg.apply(tmp,learningRate=learningRate)
    
    tmp[fgmask==0]=0
    return tmp
    
def backgroundSubstract2(fgbg,frame,blur=False,level=10,nMin = 2000,learningRate=0):
    tmp = np.copy(frame)
    fgmask=fgbg.apply(tmp,learningRate=learningRate)
    kernel = np.ones((3,3),np.uint8)

    erosion = cv2.erode(fgmask,kernel,iterations = 1)
    dilate = cv2.dilate(erosion,kernel,iterations = 1)
    tmp[fgmask==0]=0
    #tmp[dilate<.7]=0
    if np.isnan(frame).all() or len(frame[~np.isnan(frame)])<2000:
        print('all nan')
    
    return tmp,fgmask,dilate

def frameSmoother(frame,level = 10):
    temp = np.copy(frame)
    temp[np.isnan(frame)]=0
    temp = cv2.blur(temp,(level,level))
    temp[np.isnan(frame)]=np.nan
    return temp

def crossCorrArg(A,B):
    C = []
    for step in range(0,80):
        if step == 0:
            C.append(np.sum(np.abs(A-B)))
        else:
            C.append(np.sum(np.abs(A[:,:-step]-B[:,step:])))
    return np.argmin(C)

def crossCorr(A,B):
    C = []
    for step in range(0,100):
        if step == 0:
            C.append(np.sum(np.abs(A-B)))
        else:
            C.append(np.sum(np.abs(A[:,:-step]-B[:,step:])))
    return C

def createMask(frame,rate):
    #ret,thres = cv2.threshold(frame,60,255,cv2.THRESH_BINARY)
    ret,thres = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    contours,hierarchy = cv2.findContours(thres ,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
    
    maskCircle = [0,0,0]
    for contour in contours:
        (x,y),radius = cv2.minEnclosingCircle(contour)
        center = (int(x),int(y))
        radius = int(radius * rate)
        #cv2.circle(frame,center,radius,(255,0,0),2)
        if maskCircle[0]<radius:
            maskCircle[0] = int(radius)
            maskCircle[1:] = center
    mask = frame * 0
    cv2.circle(mask,maskCircle[1:],maskCircle[0], 1,-1)
    return mask

def createMaskAlternate(frame,rate=1.0):
    global todayFolderPath

    img = np.asanyarray(frame)
    gray = img.copy()
    gray_blurred = cv2.blur(gray, (3,3))

    circles = cv2.HoughCircles(gray_blurred, 
                               cv2.HOUGH_GRADIENT, 1, 100, param1 = 200,
                               param2 = 20, minRadius = 150, maxRadius = 200)
    
    maskCircle = [0,0,0]
    # ensure at least some circles were found
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0]:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            center = (int(x),int(y))
            radius = r
            if maskCircle[0]<radius:
                maskCircle[0] = int(radius*rate)
                maskCircle[1:] = center
        mask = frame * 0
        cv2.circle(mask,maskCircle[1:],maskCircle[0], 1,-1)
    return mask

def computeBowlCircle(frame_in):
    global todayFolderPath

    img = np.asanyarray(frame_in)
    #img = np.asanyarray(Image.fromarray(img.astype(np.uint8)))
    gray = img.copy()
    gray_blurred = cv2.blur(gray, (3,3))

    circles = cv2.HoughCircles(gray_blurred, 
                               cv2.HOUGH_GRADIENT, 1, 100, param1 = 200,
                               param2 = 20, minRadius = 150, maxRadius = 200)
    
    # ensure at least some circles were found
    print(str(circles))
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0]:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(img, (x, y), r, (255, 255, 255), 4)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 255), -1)
    return img, circles[0]

def monoMask():
    global rgbRes
    global irRes
    global depthRes
    rgbRes = tp.cst_resRGB
    depthRes = tp.cst_resDepthIR
    irRes = tp.cst_resDepthIR

    print(cam.resDepth[depthRes])

    x = cam.resDepth[depthRes][0]
    y = cam.resDepth[depthRes][1]
    mask = np.ones((y, x))
    return mask

def getMask(rate=1.0):
    print("* ---- Starting mask creation ---- *")
    
    #Streaming
    global frames_stream
    global stream_state

    # Start camera
    print(". Start camera .")
    global rgbRes
    global irRes
    global depthRes
    rgbRes = tp.cst_resRGB
    depthRes = tp.cst_resDepthIR
    irRes = tp.cst_resDepthIR
    
    camera = cam.Camera(rgbMode=rgbRes, depthMode=depthRes, infraredMode=irRes, 
                        rgb = False, depth = True, infrared = True, filtre = False)
    #camera = cam.Camera(infraredMode=2,depthMode=2,infrared = True,rgb = False,depth=True,filtre = False)
    camera.setAdvancedDepth()
    camera.laserSwitch()
    camera.alignToIR(True)
    camera.setExposureIR(2500) #10000
    camera.setGainIR(80) #16
    camera.laserOff()

    print(". Start camera done .")

    print(".. Mask creation ..")
    
    stream_state = 'play'

    # Get image 
    #X = camera.getFramesAligned()
    
    # Get frameset
    colorizer = rs.colorizer()
    for i in range(0, 5):
        frames = camera.pipeline.wait_for_frames()
        aligned_frames = camera.align.process(frames)
        depthMap = aligned_frames.get_depth_frame()
        depthColor = colorizer.colorize(depthMap)
        ir = aligned_frames.get_infrared_frame()
        X=[]
        X.append(np.asanyarray(depthColor.get_data()))
        X.append(np.asanyarray(ir.get_data()))
        # Copy data to streamer
        frames_stream.append(X.copy())

    # Create mask
    mask = createMask(X[1],rate)
    print(".. Mask creation done ..")

    # Control
    print("... Saving mask ...")
    global experimentFolder
        
    cv2.imwrite(todayFolderPath + "/X1.png", X[1])
    cv2.imwrite(todayFolderPath + "/mask.png", mask*255)
    cv2.imwrite(todayFolderPath + "/mask_X1.png", mask*X[1])
    print("... Saving mask done ...")
    
    # Stop camera
    print(". Stop camera .")
    stream_state = 'pause'
    camera.stop()
    
    print("* ---- Mask creation done ---- *")
    return mask

def computeMatchingPositions(nfish, posL, posR):
    epsilon = 50 #pixels
    matches = []
    success = True

    if len(posR) != nfish or len(posL) != nfish:
        success = False
    else :
        idx = []
        for i in range(0, nfish):
            idx.append(i)

        for i in range(0, nfish):
            pair = []
            pair.append(i)
            candidates = []
            for j in range(0, len(idx)):
                delta = abs(posL[i][1]-posR[idx[j]][1])
                #print(delta)
                if ( delta <= epsilon ):
                    candidates.append(idx[j])
            if len(candidates) == 0:
                success = False
                break
            else:
                d_min = 10000
                idx_min = -1
                for k in range(0, len(candidates)):
                    d = posL[candidates[k]][0] - posR[i][0]
                    #print(d)
                    if d > 0 and d < d_min:
                        idx_min = candidates[k]
                        d_min = d
                if idx_min > -1:
                    pair.append(idx_min)
                    matches.append(pair)
                    idx.remove(idx_min)
                else:
                    success = False
                    break
    #print(matches)
    return success, matches

def computeMatchingPairs(A, B):
    # Compute pairwise distance matrix
    cost_matrix = cdist(A, B, metric='euclidean')

    # Apply the constraint: Only allow matches where x_A > x_B, otherwise set a large cost
    large_value = 1e6  # Large cost to prevent invalid assignments
    for i in range(len(A)):
        for j in range(len(B)):
            if A[i, 0] <= B[j, 0]:  # If x_A is not greater than x_B, make the cost very high
                cost_matrix[i, j] = large_value

    # Solve the optimal assignment problem (Hungarian Algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create matching pairs
    matching_pairs = [(A[i].tolist(), B[j].tolist()) for i, j in zip(row_ind, col_ind)]

    return matching_pairs

def adjustRefraction(pos):
    posr = [pos[0], pos[1], pos[2]]
        
    #Adjust refraction : h_t = (n_air/n_water) * h_r
    n_air = 1.000293
    n_water = 1.333
    
    delta_z = pos[2] - tp.cst_watersurface_dist

    if (delta_z > 0):
        delta_zr = delta_z * n_water / n_air
        posr[2] = pos[2] + delta_zr

    #diff = posr[2] - pos[2]
    #print("\n" + str(pos) + " " + str(posr) + " " + str(diff))
    return posr

def changeToBowlOrigin(pos):
    posr = [pos[0], pos[1], pos[2]]

    posr[0] = pos[0] - tp.cst_watersurface_center[0]
    posr[1] = pos[1] - tp.cst_watersurface_center[1] 
    posr[2] = pos[2] - tp.cst_watersurface_center[2]

    #print("\n" + str(pos) + " | " + str(posr))
    return posr

def createDepthMap(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 5  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-125,
        numDisparities=256, #5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=1,
        speckleWindowSize=256,
        speckleRange=32,
        preFilterCap=64,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.5
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    
    #filteredImg = cv2.normalize(src=displ, dst=displ, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    #filteredImg = np.uint8(filteredImg)

    return filteredImg

def do3DTracking(fish, framerate, pipeline):
    mn = tp.cst_mn

    # To stream camera output
    global frames_stream
    global stream_config
    global stream_state
    stream_config = [True, False, False, 'depth']
    stream_state = 'play'
    
    # To stream tracking output
    global tracking_stream
    global tracking_stream_state
    tracking_stream_state = 'play'
                    
    global blob_data
    global tracked_data
    global world_data
    global writing_state
    
    global tracked_filename
    global world_filename

    blob_data = []
    tracked_data = []
    world_data = []

    global experimentFolder

    left_ir_id = 1
    right_ir_id = 2

    frameseen = 0

    # Create Tracker Objects
    timeSteps = 0
    freq = 1.0/framerate
    t_1 = 0
    t0 = 0
    t1 = 0
    ts = 0

    history = []
    consecutive_xyz = []
    consecutive_phi = []
    kalman = []

    for i in range(fish):
        hf = deque([])
        history.append(hf)
        consecutive_phi.append(0)
        consecutive_xyz.append(0)
        kalman.append(defineKalman(P = 1,Q =2*10**-4,R =10**-5,dt=1.0/framerate,dim = 3))

    #----------------- Get essentials data -------------------------------------#
    # Save one image of the bowl
    frames = pipeline.wait_for_frames(10000)
    #aligned_frames = frames #rs.align(rs.stream.infrared).process(frames)
    irFrameL = frames.get_infrared_frame(left_ir_id)

    ircaption = np.asanyarray(irFrameL.get_data())
    cv2.imwrite(experimentFolder + "/ircaption.png", ircaption)
    print("\n>>>>>>>>> Experimental setup data <<<<<<<<<<<<<<\n")
    intrinsics = irFrameL.get_profile().as_video_stream_profile().get_intrinsics()
    print("\nIR Camera intrinsics : " + str(intrinsics) + "\n")

    depthFrame = frames.get_depth_frame()
    intrinsics3D = depthFrame.get_profile().as_video_stream_profile().get_intrinsics()
    print("\nDepth intrinsics : " + str(intrinsics3D) + "\n")

    #----------------- Compute camera distance to water ------------------------#
    tp.cst_watersurface_dist = computeWaterDistanceToCam()
    print("\nWater distance to camera = " + str(tp.cst_watersurface_dist) + "\n")

    #wdcenter = computeWaterDistanceToCenter()
    #print("->>> water distance to center = " + str(wdcenter) + " \n")

    #----------------- Compute bowl position in pictures    --------------------#
    frames = pipeline.wait_for_frames(10000)
    # Left - 3D aligned with left
    framecircle = frames.get_infrared_frame(left_ir_id).as_frame().get_data()
    imgcL, centerL = computeBowlCircle(framecircle)
    tp.cst_centerleft = [centerL[0][0], centerL[0][1]]
    tp.cst_radiusleft = centerL[0][2]
    cv2.imwrite(todayFolderPath + "/circle_L.png",imgcL)
    print("\nCenter & radius from left : " + str(tp.cst_centerleft) + " | " + str(tp.cst_radiusleft) + "\n")


    #---------------- Compute bowl center position in camera world -------------------#
    xc = tp.cst_centerleft[0]
    yc = tp.cst_centerleft[1]
    d = tp.cst_watersurface_dist
    tp.cst_watersurface_center = rs.rs2_deproject_pixel_to_point(intrinsics, [xc, yc], d)

    print("\nWater Surface center at : " + str(tp.cst_watersurface_center) + "\n")

    #------------------- Create mask ------------------------------------------#
    print(">>>>>>>> Creating masks <<<<<<<", end=" ")
    frames = pipeline.wait_for_frames(10000)
    frameseen += 1
    frame = frames.get_infrared_frame(left_ir_id) 
    frameL = np.asanyarray(frame.as_frame().get_data())
    
    mask = []
    mask.append(createMaskAlternate(frameL,0.915))
    cv2.imwrite(todayFolderPath + "/irL.png",frameL)
    cv2.imwrite(todayFolderPath + "/mask0.png",mask[0])
    cv2.imwrite(todayFolderPath + "/ir_mask_L.png",frameL*mask[0])
    print(" >>>>>>>> Done ! <<<<<<<\n")

    print("\n********** START TRACKING ! ************\n")

    # Streaming loop
    for i in range(mn*60*framerate) :
        # print(timeSteps)
        # Get frameset 
        frames = pipeline.wait_for_frames()
        frameseen += 1

        if timeSteps == 0:
            t0 = time.time_ns()
            t1 = t0
            t_1 = t0
        else :
            t = time.time_ns()
            t_1 = t1
            t1 = t - t0
            dt = (t1 - t_1) * 0.000000001
            ts = t1 * 0.000000001
            #print("t0: "+ str(t0) + " t1: " + str(ts) + " dt: " + str(dt) + " ")

        #aligned_frames = rs.align(rs.stream.depth).process(frames)
        #depthMap = aligned_frames.get_depth_frame()
        #irFrameL = aligned_frames.get_infrared_frame(left_ir_id)

        depthMap = frames.get_depth_frame()
        irFrameL = frames.get_infrared_frame(left_ir_id)
        
        dMap = mask[0]*np.copy(np.asanyarray(depthMap.as_frame().get_data()))
        irL = mask[0]*np.copy(np.asanyarray(irFrameL.as_frame().get_data()))
        
        if timeSteps == 0:  # initialization
            # Init fish tank
            tank = ft.FishTank(fish, framerate, dMap, dMap, irL, mask[0], 3, 1, False)
            
        else:
            # 3D tracking from Depth Map
            
            # Do tracking !
            tank.updateFishes(dMap, dMap, irL)
            flag = True
            pos = []
            
            for k in range(0, fish):
                pos.append(np.copy(tank.fishList[k].stateCorr()[0]))
                        
                # Compute 3D tracking data

                # Store tracking data
                #s_tracked = str(i) + " " + str(t1)
                #s_world = str(i) + " " + str(t1)

                s_tracked = str(i) + " " + str(frameseen*freq)
                
                #s_world = str(i) + " " + str(frameseen*freq)
                s_world = str(i) + " " + str(ts)

                x = pos[k][0][0]
                y = pos[k][1][0]
                z = pos[k][2][0]

                s_tracked = s_tracked + " " + str(x)+ " " + str(y)+ " " + str(z) + " " # + str(pos[3])+ " " + str(pos[4])+ " " + str(pos[5])
                    
                try:
                    #sintrinsics = depthMap.profile.as_video_stream_profile().intrinsics
                    #distance = z/1000.
                    distance = depthMap.get_distance(int(x), int(y))
                    pos_world = rs.rs2_deproject_pixel_to_point(intrinsics3D, [x, y], distance)
                    
                    #pos_world_r = adjustRefraction(pos_world)
                    pos_world_r = changeToBowlOrigin(pos_world)  

                    #Kalman filter
                    kalman[k].update(pos_world_r)
                    kalman[k].predict()
                    pos_world_r = kalman[k].x
                        
                    #Compute chi and phi --- add noise mean filter 
                    phi = 0.0
                    chi = 0.0
                    #print(history)
                    #print(str(h))
                    #if h > 0:
                        
                    if consecutive_phi[k] >= tp.cst_hist_phi:
                        # h = history[k][-1][0] - history[k][-consecutive_phi][0]
                        #vx = (history[k][-1][2][0] - history[k][-consecutive_phi][2][0]) / h
                        #vy = (history[k][-1][2][1] - history[k][-consecutive_phi][2][1]) / h
                        #phi = np.arctan2(vy, vx)
                        #print(str(consecutive_xyz[k]) + "| " + str(history[k]))

                        h = ts - history[k][-consecutive_phi[k]][0]
                        vx = (pos_world_r[0] - history[k][-consecutive_phi[k]][2][0]) / h
                        vy = (pos_world_r[1] - history[k][-consecutive_phi[k]][2][1]) / h
                        vz = (pos_world_r[2] - history[k][-consecutive_phi[k]][2][2]) / h
                        phi = np.arctan2(vy, vx)
                        if abs(vz)<=1.0 :
                            chi = np.arcsin(vz)
                        else :
                            chi = 0.0
                    
                    '''
                    if consecutive_xyz[k] >= tp.cst_hist_xyz:
                        #print(str(consecutive_xyz[k]) + "| " + str(history[k]))
                        new_pos = [pos_world_r[0], pos_world_r[1], pos_world_r[2]]
                        for h in range(0, consecutive_xyz[k]-1):
                            new_pos[0] += history[k][-1-h][2][0]
                            new_pos[1] += history[k][-1-h][2][1]
                            new_pos[2] += history[k][-1-h][2][2]
                        new_pos[0] = new_pos[0]/consecutive_xyz[k]
                        new_pos[1] = new_pos[1]/consecutive_xyz[k]
                        new_pos[2] = new_pos[2]/consecutive_xyz[k]
                        #print(str(pos_world_r) + " | " + str(new_pos))
                        pos_world_r = new_pos
                    '''
                    consecutive_xyz[k] = min(consecutive_xyz[k]+1, tp.cst_hist_xyz)
                    consecutive_phi[k] = min(consecutive_phi[k]+1, tp.cst_hist_phi)
                    history[k].append((ts, dt, pos_world_r))
                    if len(history[k]) > tp.cst_histwindow:
                        history[k].popleft()

                    s_world = s_world + " " + str(pos_world_r[0]) + " " + str(pos_world_r[1]) + " " + str(pos_world_r[2]) + " " + str(phi) + " " + str(chi)
                
                except Exception as e:
                    flag = False
                    print(str(e))
                    s_world = s_world + " " + "999 999 999 999 999"
                
                s_tracked = s_tracked + "\n"
                s_world = s_world + "\n"

                if flag:
                    #print(".", end=" ")   
                    #tracked_data.append(s_tracked)
                    world_data.append(s_world)

                    #message = f"{fish} {tracked_filename} {s_tracked} {world_filename} {s_world}"
                    message = f"{fish} {world_filename} {s_world}"
                    sendWritingData(message)
            
        if visu:
            images=[]
            colorizer = rs.colorizer()
            depthColor = colorizer.colorize(depthMap)
            images.append(np.asanyarray(depthColor.get_data()))
            images.append(np.asanyarray(irFrameL.get_data()))
            images.append(np.asanyarray(dMap*100))
            frames_stream.append(images.copy())

            deathMapName = experimentFolder + "\deathMap_" + str(timeSteps) + ".png";
            cv2.imwrite(deathMapName, np.asanyarray(depthColor.get_data()))
            irLName = experimentFolder + "\irL_" + str(timeSteps) + ".png";
            cv2.imwrite(irLName, np.asanyarray(irFrameL.get_data()))
            dmName = experimentFolder + "\dm_" + str(timeSteps) + ".png";
            cv2.imwrite(dmName, np.asanyarray(dMap*100))
            
            # Feed tracking stream
            trackingImages = []
            tm = tank.showTank_new()
            trackingImages.append(tm)
            tracking_stream.append(trackingImages.copy())

            tmName = experimentFolder + "\\tankm_" + str(timeSteps) + ".png";
            cv2.imwrite(tmName, tm)
            
        timeSteps += 1

    print("\n*************  Tracking done ! *************")
    print("*         Saving data, pease wait ...      *")
    print("********************************************\n")

    saveTrackingData()

    print("\n*************  Data saved ! *************")
    print("\n------> Please get data in " + experimentFolder)

def doStereoTracking(fish, framerate, pipeline):
    mn = tp.cst_mn

    global playback
    global config

    # To stream camera output
    global frames_stream
    global stream_config
    global stream_state
    stream_config = [True, False, False, 'depth']
    stream_state = 'play'
    
    # To stream tracking output
    global tracking_stream
    global tracking_stream_state
    tracking_stream_state = 'play'
                    
    global blob_data
    global tracked_data
    global world_data
    global writing_state
    
    global tracked_filename
    global world_filename

    blob_data = []
    tracked_data = []
    world_data = []

    global experimentFolder

    #sbm = cv2.StereoBM_create(numDisparities=64, blockSize=7)
    #sbm.SADWindowSize = SWS
    #sbm.setPreFilterType(1)
    #sbm.setPreFilterSize(17)
    #sbm.setPreFilterCap(32)
    #sbm.setMinDisparity(MDS)
    #sbm.setNumDisparities(NOD)
    #sbm.setTextureThreshold(TTH)
    #sbm.setUniquenessRatio(UR)
    #sbm.setSpeckleRange(32)
    #sbm.setSpeckleWindowSize(1)

    #disparityTransform = rs.disparity_transform()
    #detector = createBlobDetector()


    left_ir_id = 1
    right_ir_id = 2

    frameseen = 0

    # Create Tracker Objects
    timeSteps = 0
    freq = 1.0/framerate
    t_1 = 0
    t0 = 0
    t1 = 0
    ts = 0

    history = []
    consecutive_xyz = []
    consecutive_phi = []
    if (tp.cst_noisefiltering==2):
        kalman = []

    for i in range(fish):
        hf = deque([])
        history.append(hf)
        consecutive_phi.append(0)
        consecutive_xyz.append(0)
        if (tp.cst_noisefiltering==2):
            kalman.append(defineKalman(P = 1,Q =2*10**-4,R =10**-5,dt=1.0/framerate,dim = 3))

    #----------------- Get essentials data -------------------------------------#
    # Save one image of the bowl
    frames = pipeline.wait_for_frames(10000)
    #aligned_frames = frames #rs.align(rs.stream.infrared).process(frames)
    irFrameL = frames.get_infrared_frame(left_ir_id)

    ircaption = np.asanyarray(irFrameL.get_data())
    cv2.imwrite(experimentFolder + "/ircaption.png", ircaption)
    print("\n>>>>>>>>> Experimental setup data <<<<<<<<<<<<<<\n")
    intrinsics = irFrameL.get_profile().as_video_stream_profile().get_intrinsics()
    print("\nCamera intrinsics : " + str(intrinsics) + "\n")

    #depthFrame = frames.get_depth_frame()
    #intrinsics = depthFrame.get_profile().as_video_stream_profile().get_intrinsics()
    #print("\nDepth intrinsics : " + str(intrinsics) + "\n")

    #----------------- Compute camera distance to water ------------------------#
    tp.cst_watersurface_dist = computeWaterDistanceToCam()
    print("\nWater distance to camera = " + str(tp.cst_watersurface_dist) + "\n")

    #wdcenter = computeWaterDistanceToCenter()
    #print("->>> water distance to center = " + str(wdcenter) + " \n")

    #----------------- Compute bowl position in pictures    --------------------#
    frames = pipeline.wait_for_frames(10000)
    # Left
    framecircle = frames.get_infrared_frame(left_ir_id).as_frame().get_data()
    imgcL, centerL = computeBowlCircle(framecircle)
    tp.cst_centerleft = [centerL[0][0], centerL[0][1]]
    tp.cst_radiusleft = centerL[0][2]
    cv2.imwrite(todayFolderPath + "/circle_L.png",imgcL)
    print("\nCenter & radius from left : " + str(tp.cst_centerleft) + " | " + str(tp.cst_radiusleft) + "\n")

    # Right
    framecircle = frames.get_infrared_frame(right_ir_id).as_frame().get_data()
    imgcR, centerR = computeBowlCircle(framecircle)
    tp.cst_centerright = [centerR[0][0], centerR[0][1]]
    tp.cst_radiusright = centerR[0][2]
    cv2.imwrite(todayFolderPath + "/circle_R.png",imgcR)
    print("\nCenter & radius from right : " + str(tp.cst_centerright) + " | " + str(tp.cst_radiusright) + "\n")

    #---------------- Compute bowl center position in camera world -------------------#
    xc = tp.cst_centerleft[0]
    yc = tp.cst_centerleft[1]
    d = tp.cst_watersurface_dist
    tp.cst_watersurface_center = rs.rs2_deproject_pixel_to_point(intrinsics, [xc, yc], d)

    print("\nWater Surface center at : " + str(tp.cst_watersurface_center) + "\n")

    #------------------- Create mask ------------------------------------------#
    print(">>>>>>>> Creating masks <<<<<<<", end=" ")
    frames = pipeline.wait_for_frames(10000)
    frameseen += 1
    frame = frames.get_infrared_frame(left_ir_id) 
    frameL = np.asanyarray(frame.as_frame().get_data())
    frame = frames.get_infrared_frame(right_ir_id)
    frameR = np.asanyarray(frame.as_frame().get_data())
    mask = []
    mask.append(createMaskAlternate(frameL,0.965))
    mask.append(createMaskAlternate(frameR,0.965))
    cv2.imwrite(todayFolderPath + "/irL.png",frameL)
    cv2.imwrite(todayFolderPath + "/mask0.png",mask[0])
    cv2.imwrite(todayFolderPath + "/mask1.png", mask[1])
    cv2.imwrite(todayFolderPath + "/ir_mask_L.png",frameL*mask[0])
    cv2.imwrite(todayFolderPath + "/ir_mask_R.png", frameR*mask[1])
    print(" >>>>>>>> Done ! <<<<<<<\n")

    print("\n>>>>>> Creating background substractors <<<<<<<", end=" ")

    # Initialize background substractors with 200 images
    irFrames = []
    irFrames.append([])
    irFrames.append([])

    for k in range(0,200):
        frames = pipeline.wait_for_frames(10000)
        frameseen += 1
        irFrames[0].append(mask[0]*np.copy(np.asanyarray(frames.get_infrared_frame(left_ir_id).as_frame().get_data())))
        irFrames[1].append(mask[1]*np.copy(np.asanyarray(frames.get_infrared_frame(right_ir_id).as_frame().get_data())))
    fgbg = []
    #fgbg.append(backGroundSubstractor(irFrames[0], type="MOG2"))
    #fgbg.append(backGroundSubstractor(irFrames[1], type="MOG2"))
    fgbg.append(backGroundSubstractor(irFrames[0]))
    fgbg.append(backGroundSubstractor(irFrames[1]))


    print(" >>>>>>>> Done ! <<<<<<<\n")

    print("\n********** START TRACKING ! ************\n")

    pipeline.stop()
    pipeline.start(config)
    playback.set_real_time(True)

    # set the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Streaming loop
    for i in range(mn*60*framerate) :
        # print(timeSteps)
        # Get frameset 
        test = False
        #while (test == False):
        
        pbtime = 0
        pbtime2 = 0

        #try:
        
        frames = pipeline.wait_for_frames(10000)
        """
        fsize=0
        while (fsize==0):
            #frames = pipeline.poll_for_frames()
            frames = pipeline.wait_for_frames()
            fsize = frames.size()
            if (fsize > 0):
                pbtime = playback.get_position()
                print("Time 1 : " + str(pbtime))
            else:
                frames = pipeline.wait_for_frames()
                fsize = frames.size()
        
        fsize=0
        """
        frameseen += 1
        #except:
        """
        playback.pause()
        pbtime2 = playback.get_position()
        print("Time 2 : " + str(pbtime2))
        delta = timedelta(days=0, hours=0, seconds=(pbtime+5000)/1000000000, microseconds=0, milliseconds=0, minutes=0, weeks=0)
        print("Delta : " + str(delta))
        #pipeline.stop()
        #pipeline.start(config)
        #playback.set_real_time(True)
        playback.seek(delta)
        #playback.resume()
        #frames = pipeline.wait_for_frames(500)
        test = False
        """

        if timeSteps == 0:
            t0 = time.time_ns()
            t1 = t0
            t_1 = t0
        else :
            t = time.time_ns()
            t_1 = t1
            t1 = t - t0
            dt = (t1 - t_1) * 0.000000001
            ts = t1 * 0.000000001
            #print("t0: "+ str(t0) + " t1: " + str(ts) + " dt: " + str(dt) + " ")

        #aligned_frames = frames #rs.align(rs.stream.depth).process(frames)
        
        depthMap = frames.get_depth_frame()
        irFrameL = frames.get_infrared_frame(left_ir_id)
        irFrameR = frames.get_infrared_frame(right_ir_id)

        dMap = np.copy(np.asanyarray(depthMap.as_frame().get_data()))

        # Left
        irL = mask[0]*np.copy(np.asanyarray(irFrameL.as_frame().get_data()))
        
        irLBlur = cv2.GaussianBlur(irL, (5,5), 0)
        #irLBlur = cv2.medianBlur(irLBlur0, 5)
        irLBT = cv2.adaptiveThreshold(irLBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
        #ret, irLBT = cv2.threshold(irLBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        irBL = cv2.bitwise_not(irLBT)*mask[0]
        
        
        #irBL = cv2.morphologyEx(irBLM, cv2.MORPH_OPEN, kernel)

        """
        irLTh = cv2.threshold(irL, 0, 255, cv2.THRESH_OTSU)[1]
        irBLI = cv2.bitwise_not(irLTh)*mask[0]
        irBL = cv2.dilate(irBLI, kernel, iterations=2)
        """
        #irBL = backgroundsSubstract(fgbg[0],irLTh,learningRate = 0.)
               
        # Apply transform
        """
        irBL = cv2.GaussianBlur(irBL, (7,7), 1)
        #irBL_closed = cv2.morphologyEx(irBL, cv2.MORPH_CLOSE, kernel)
        irBL_eroded = cv2.morphologyEx(irBL, cv2.MORPH_OPEN, kernel)
        #irBL = irBL_eroded
        #irBL = cv2.divide(irBL, irBL_eroded, scale=255)
        irBL = cv2.threshold(irBL_eroded, 0, 255, cv2.THRESH_OTSU)[1]
        """
        
        # Right
        irR = mask[1]*np.copy(np.asanyarray(irFrameR.as_frame().get_data()))

        irRBlur = cv2.GaussianBlur(irR, (5,5), 0)
        #irRBlur = cv2.medianBlur(irRBlur0, 5)
        irRBT = cv2.adaptiveThreshold(irRBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
        #ret, irRBT = cv2.threshold(irRBlur,100,255,cv2.THRESH_BINARY)
        irBR = cv2.bitwise_not(irRBT)*mask[1]
        #irBR = cv2.morphologyEx(irBRM, cv2.MORPH_OPEN, kernel)
        
        """
        irRTh = cv2.threshold(irR, 0, 255, cv2.THRESH_OTSU)[1]
        irBRI = cv2.bitwise_not(irRTh)*mask[1]
        irBR = cv2.dilate(irBRI, kernel, iterations=2)
        """

        #irBR = backgroundSubstract(fgbg[1],irR,learningRate = 0.)
                         
        # Apply transform
        """
        irBR = cv2.GaussianBlur(irBR, (7,7), 1)
        #irBR_closed = cv2.morphologyEx(irBR, cv2.MORPH_CLOSE, kernel)
        irBR_eroded = cv2.morphologyEx(irBR, cv2.MORPH_OPEN, kernel)
        #irBR = irBR_eroded
        irBR = cv2.threshold(irBR_eroded, 0, 255, cv2.THRESH_OTSU)[1]
        """

        #dMapTest = createDepthMap(irL, irR);

        """
        irRG = cv2.blur(irR, (3,3))
        irLG = cv2.blur(irL, (3,3))
        keypointsR = detector.detect(irR)
        keypointsL = detector.detect(irL)
        irBRkp = cv2.drawKeypoints(irRG, keypointsR, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        irBLkp = cv2.drawKeypoints(irLG, keypointsL, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        """

        if timeSteps == 0:  # initialization
            # Init fish tank
            tankL = ft.FishTank(fish, framerate, irBL, dMap, irL, mask[0], 2, 0, False)
            tankR = ft.FishTank(fish, framerate, irBR, dMap, irR, mask[1], 2, 0, False)
            
        else:
            # Stereo tracking from IR pictures
            
            # Do tracking !
            tankL.updateFishes(irBL, dMap, irL)
            tankR.updateFishes(irBR, dMap, irR)

            posL = []
            posR = []

            for k in range(0, fish):
                posL.append(np.copy(tankL.fishList[k].stateCorr()[0]))
                posR.append(np.copy(tankR.fishList[k].stateCorr()[0]))
            
            # Match left image fish with right image ones

            matchingOk, matches = computeMatchingPositions(fish, posL, posR)

            #print(matches)

            if (matchingOk and len(matches)==fish):

                # Compute 3D trackink data
                flag = True

                # Store tracking data
                #s_tracked = str(i) + " " + str(t1)
                #s_world = str(i) + " " + str(t1)

                s_tracked = str(i) + " " + str(frameseen*freq)
                
                #s_world = str(i) + " " + str(frameseen*freq)
                s_world = str(i) + " " + str(ts)

                for k in range(0, fish):
                    # Compute disparity             
                    xL = posL[k][0][0]
                    yL = posL[k][1][0]

                    xR = posR[matches[k][1]][0][0]
                    yR = posR[matches[k][1]][1][0]

                    # Adjust xR if needed
                    if abs(int(yL) - int(yR))>1:
                        pcR = tankR.fishList[matches[k][1]].fishPoints
                        if isinstance(pcR, Iterable):
                            rPoints = []
                            for p in pcR:
                                if p[1] >= int(yL)-1 and p[1] <= int(yL) + 1 :
                                    rPoints.append(p)
                            if len(rPoints) > 0:
                                new_xR = 0
                                for p in rPoints:
                                    new_xR += p[0]
                                new_xR = new_xR / len(rPoints)
                                #print("yL: " + str(int(yL)) + " | yR: " + str(int(yR)) + " | xL: " 
                                #      + str(xL) + " | xR: " + str(xR) + " | nxR: " + str(new_xR))
                                xR = new_xR

                    #Compute disparity & distance
                    disparity, distance = computeDistanceWithDisparity(xL, xR)
                    
                    #print(str(dMapTest))
                    #disparityTest = dMapTest[int(yL)][int(xL)]
                    #distTest = computeDistanceWithDisparity2(disparityTest)
                    #print("Disparity : " + str(disparity) + " - Test : " + str(disparityTest) + " - D : " + str(distance) + " - Dtest : " + str(distTest))

                    s_tracked = s_tracked + " " + str(xL)+ " " + str(yL)+ " " + str(disparity) #+ " " + str(pos[3])+ " " + str(pos[4])+ " " + str(pos[5])
                    
                    try:
                        if disparity > 0:
                            #intrinsics = irFrameL.get_profile().as_video_stream_profile().get_intrinsics()
                            #intrinsics = depthMap.profile.as_video_stream_profile().intrinsics
                            
                            pos_world = rs.rs2_deproject_pixel_to_point(intrinsics, [xL, yL], distance)

                            #pos_world_r = adjustRefraction(pos_world)
                            pos_world_r = changeToBowlOrigin(pos_world)

                            #Kalman filter
                            if (tp.cst_noisefiltering==2):
                                kalman[k].update(pos_world_r)
                                kalman[k].predict()
                                pos_world_r[0] = kalman[k].x[0]
                                pos_world_r[1] = kalman[k].x[1]
                                pos_world_r[2] = kalman[k].x[2]

                            if pos_world_r[2] < 0.:
                                #print("1 : ", pos_world_r, "/n")
                                pos_world_r[2] = history[k][-1][2][2]
                                #print("2 : ", pos_world_r, "/n")
                            """
                            distanceRS = depthMap.get_distance(int(xL), int(yL))
                            pos_world_rs = rs.rs2_deproject_pixel_to_point(intrinsics, [xL, yL], distanceRS)
                            
                            print(sL + " " + sR + " " + str(disparity) 
                            + " d1: " + str(distance) + " d2: " + str(distanceRS) 
                            + " " + str(pos_world) + " " + str(pos_world_rs))
                            """
                            
                            #Compute chi and phi --- add noise mean filter 
                            phi = 0.0
                            chi = 0.0
                            #print(history)
                            #print(str(h))
                            #if h > 0:
                            
                            if consecutive_phi[k] >= tp.cst_hist_phi:
                                h = ts - history[k][-consecutive_phi[k]][0]
                                vx = (pos_world_r[0] - history[k][-consecutive_phi[k]][2][0]) / h
                                vy = (pos_world_r[1] - history[k][-consecutive_phi[k]][2][1]) / h
                                vz = (pos_world_r[2] - history[k][-consecutive_phi[k]][2][2]) / h
                                v = np.sqrt(vx**2+vy**2 + 1e-8)

                                phi = np.arctan2(vy, vx)
                                chi = np.arctan(vz/v)
                            
                            if (tp.cst_noisefiltering == 1):
                                if consecutive_xyz[k] >= tp.cst_hist_xyz:
                                    #print(str(consecutive_xyz[k]) + "| " + str(history[k]))
                                    new_pos = [pos_world_r[0], pos_world_r[1], pos_world_r[2]]
                                    for h in range(0, consecutive_xyz[k]-1):
                                        new_pos[0] += history[k][-1-h][2][0]
                                        new_pos[1] += history[k][-1-h][2][1]
                                        new_pos[2] += history[k][-1-h][2][2]
                                    new_pos[0] = new_pos[0]/consecutive_xyz[k]
                                    new_pos[1] = new_pos[1]/consecutive_xyz[k]
                                    new_pos[2] = new_pos[2]/consecutive_xyz[k]
                                    #print(str(pos_world_r) + " | " + str(new_pos))
                                    pos_world_r = new_pos
                            
                            consecutive_xyz[k] = min(consecutive_xyz[k]+1, tp.cst_hist_xyz)
                            consecutive_phi[k] = min(consecutive_phi[k]+1, tp.cst_hist_phi)
                            history[k].append((ts, dt, pos_world_r))
                            if len(history[k]) > tp.cst_histwindow:
                                history[k].popleft()

                            s_world = s_world + " " + str(pos_world_r[0]) + " " + str(pos_world_r[1]) + " " + str(pos_world_r[2]) + " " + str(phi) + " " + str(chi)
                        else:
                            flag = False
                            consecutive_xyz[k] = 0
                            consecutive_phi[k] = 0
                    except Exception as e:
                        print(str(e))
                        s_world = s_world + " " + "999 999 999 999 999"
                
                #s_tracked = s_tracked + "\n"
                s_world = s_world + "\n"

                if flag:
                    #print(".", end=" ")   
                    #tracked_data.append(s_tracked)
                    world_data.append(s_world)

                    #message = f"{fish} {tracked_filename} {s_tracked} {world_filename} {s_world}"
                    message = f"{fish} {world_filename} {s_world}"
                    sendWritingData(message)
            
            if visu:
                images=[]
                #images.append(dMapTest)
                colorizer = rs.colorizer()
                depthColor = colorizer.colorize(depthMap)
                images.append(np.asanyarray(depthColor.get_data()))
                images.append(np.asanyarray(irFrameL.get_data()))
                images.append(np.asanyarray(irFrameR.get_data()))
                images.append(irBL)
                images.append(irBR)
                frames_stream.append(images.copy())

                deathMapName = experimentFolder + "\deathMap_" + str(timeSteps) + ".png";
                cv2.imwrite(deathMapName, np.asanyarray(depthColor.get_data()))
                irLName = experimentFolder + "\irL_" + str(timeSteps) + ".png";
                cv2.imwrite(irLName, np.asanyarray(irFrameL.get_data()))
                irRName = experimentFolder + "\irR_" + str(timeSteps) + ".png";
                cv2.imwrite(irRName, np.asanyarray(irFrameR.get_data()))
                irBLName = experimentFolder + "\irBL_" + str(timeSteps) + ".png";
                cv2.imwrite(irBLName, np.asanyarray(irBL))
                irBRName = experimentFolder + "\irBR_" + str(timeSteps) + ".png";
                cv2.imwrite(irBRName, np.asanyarray(irBR))
                
                # Feed tracking stream
                trackingImages = []
                tankLimg = tankL.showTank_new() 
                trackingImages.append(tankLimg)
                tankRimg = tankR.showTank_new()
                trackingImages.append(tankRimg)
                tracking_stream.append(trackingImages.copy())

                tankLName = experimentFolder + "\TankL_" + str(timeSteps) + ".png";
                cv2.imwrite(tankLName, np.asanyarray(tankLimg))
                tankRName = experimentFolder + "\TankR_" + str(timeSteps) + ".png";
                cv2.imwrite(tankRName, np.asanyarray(tankRimg))
                

        timeSteps += 1

    print("\n*************  Tracking done ! *************")
    print("*         Saving data, pease wait ...      *")
    print("********************************************\n")

    saveTrackingData()

    print("\n*************  Data saved ! *************")
    print("\n------> Please get data in " + experimentFolder)


def trackFromFile(nfish=1, mn=90, framerate=30, verbose=True, visu=False, path="", filename=""):
    global experimentFolder
    global tracked_filename
    global world_filename

    global playback
    global config    

    # Number of fished to track
    if nfish  > 0:
        fish = nfish

    # Data file
    bagFile = path + "/" + filename
  
    # Check if the given file have bag extension
    if os.path.splitext(filename)[1] != ".bag":
        print("The given file is not of correct file format.\n")
        print("Only .bag files are accepted\n")
        exit()

    # Files for saving data filename[0:-4]
    tracked_filename = experimentFolder + "/" + filename[0:-4] +"_tracked.csv"
    world_filename = experimentFolder + "/" + filename[0:-4] +"_world.csv"

    try:
        #---------------------- Create pipeline -------------------------------------------#
        print("\n>>>>>>> Configuring RealSense Pipeline <<<<<<<\n")
        print("Creating pipeline ...", end=" ")
        pipeline = rs.pipeline()
        print("\nDone !")
        # Create a config object
        print("\nConfiguring RealSense....", end=" ")
        config = rs.config()
        print("\nDone !")
        # Tell config that we will use a recorded device from file to be used
        # +by the pipeline through playback.
        print("\nConfiguring source file: ",
              bagFile,
              "...",
              end=" ")
        config.enable_device_from_file(file_name=bagFile, repeat_playback=False)
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        playback = profile.get_device().as_playback()
        playback.set_real_time(True)
        print("\nDone !\n")

        if tp.cst_stereo :
            doStereoTracking(fish, framerate, pipeline)
        else :
            do3DTracking(fish, framerate, pipeline)
        
    finally:
        message = f"close"
        sendWritingData(message)
        saveTrackingData()
        pass



def trackFish(fish=1, mn=90, framerate=30, verbose=True, visu=False):
    global todayFolderPath
    
    print("* ---- Starting fish tracking ---- *")

    # Start camera
    print(". Start camera .")
    global rgbRes
    global irRes
    global depthRes
    rgbRes =  tp.cst_resRGB #0
    depthRes = tp.cst_resDepthIR #2
    irRes = tp.cst_resDepthIR #2

    camera = cam.Camera(rgbMode=rgbRes, depthMode=depthRes, infraredMode=irRes, 
                        rgb = False, depth = True, infrared = True, filtre = False)
    
    #camera = cam.Camera(infraredMode=2,depthMode=2,infrared = True,rgb = False,depth=True,filtre = False)
    camera.setAdvancedDepth()
    #camera.laserOff()
    camera.laserSwitch()
    #camera.alignToIR(True)
    camera.setExposureIR(tp.cst_exposure) #2500
    camera.setGainIR(tp.cst_gain) #80
    print(". Start camera done .")
    
    try:
        # Create pipeline
        print("Get RealSense Pipeline....", end=" ")
        pipeline = camera.pipeline
        print("Done !")
        # Create a config object
        print("Get RealSense config....", end=" ")
        config = camera.config
        print("Done !")
        
        profile = camera.profile #= pipeline.start(config)
        
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        if tp.cst_stereo :
            doStereoTracking(fish, framerate, pipeline)
        else :
            do3DTracking(fish, framerate, pipeline)
    finally:
        message = f"close"
        sendWritingData(message)
        saveTrackingData()
        pass

#-------------------------------------------------------------------------------
# MAIN
#-------------------------------------------------------------------------------

def interactiveMenu():
    #np.set_printoptions(threshold=sys.maxsize)

    global todayFolderPath
    global experimentFolder
    global tracked_filename
    global world_filename
    global visu
    global cameraThread
    global trackvisuThread
    global tracking_stream_state
    global stream_state
    global writing_socket
    global writing_state

    starting_string = "\n###### --- Tracking Fish ! --- ######\n"
    ending_string = "\n###### --- Tracking Fish Done !!! --- ######\n"
    
    print(starting_string)
      
    #Create folder with today date
    today = date.today() 
    todayFolderPath = createFolder(pathFile,[str(today)])
    todayFolderPath = os.path.normpath(todayFolderPath)
    print("\n***  The folder for today experiments is " + todayFolderPath + "  ***\n")

     # Testing with file
    TestTrackingFromFile = tp.cst_trackfromfile
    
    # First make sure that projector and IR lamps are OFF, etc.
    while True:
        print("\n!!! Please make sure that :\n\tBowl is full of water\n\tFish are in the bowl\n\tProjector is on\n\tIR lamps are on !!!\n")
        key = input("Press \'y\' to continue or \'n\' to stop ...\n")
        if (key == "Y" or key == "y"):
            break;  
        elif (key == "N" or key == "n"):
            sys.exit(0)
    
    # 0- Ask how many fish
    nfish = 0
    while True:
        print("\n!!! Please answer next question !!!\n")
        n = input("How many fish to track ? Enter a number or n to stop ...\n")
        if (n.isnumeric()):
            nfish = int(n)    
            print("\nYou will track " + str(nfish) + " fish.")
            print("--> Do you wish to continue ?")
            key = input("Press \'y\' to continue, \'n\' to stop or any other key to enter a new value ...\n")
            if (key == "Y" or key == "y"):
                break;  
            elif (key == "N" or key == "n"):
                sys.exit(0)
        elif (n == "N" or n == "n"):
            sys.exit(0)                          
            
    # Create folder to store data
    t = time.localtime(time.time())
    f = str(nfish)+"Fish"
    p = str(t.tm_hour).zfill(2) + "-" + str(t.tm_min).zfill(2)
    experimentFolder = os.path.normpath(createFolder(todayFolderPath, [f, p]))
    print("Tracking data is stored in " + experimentFolder + "\n")

    global visu
    visu = False
    while True:
        print("\n!!! Do you want to vizualise tracking ? !!!\n")
        key = input("Press \'y\' (yes) or \'n\' (no), \'q\'to stop ...\n")
        if (key == "Y" or key == "y"):
            visu  = True
            break
        elif (key == "N" or key == "n"):
            visu  = False
            break
        elif (key == "Q" or key == "q"):
            sys.exit(0)
    
    if visu:      
        # Starting camera stream
        cameraThread = threading.Thread(target=showCameraStream_dyn)
        cameraThread.daemon = True       
        cameraThread.start()
        
        # Starting tracking stream
        trackvisuThread = threading.Thread(target=showTrackingStream)
        trackvisuThread.daemon = True 
        trackvisuThread.start()
        
    while True:
        print("\n!!!               Ready to track ?                 !!!")
        #print("!!!   Please check that writing server is running  !!!\n")
        key = input("Press \'y\' to proceed or \'n\' to stop ...\n")
        if (key == "Y" or key == "y"):
            visu  = True
            break
        elif (key == "N" or key == "n"):
            stream_state = 'stop'
            tracking_stream_state = 'stop'
            writing_state = 'stop'
            sys.exit(0)
    
    if writing_socket is None:
        connectWritingServer()

    if TestTrackingFromFile:
        mn = tp.cst_mn
        fps = tp.cst_framerate
        p = tp.cst_path
        f = tp.cst_bagfilename
        trackFromFile(nfish, mn, fps, False, visu, p, f)
    else :
        tracked_filename = experimentFolder + "/tracked.csv"
        world_filename = experimentFolder + "/world.csv"
        trackFish(nfish, tp.cst_mn, tp.cst_framerate, False, visu)
        
    while True:
        print(ending_string)
        input("Press any key to stop ...\n")
        stream_state = 'stop'
        tracking_stream_state = 'stop'
        writing_state = 'stop'
        sys.exit(0)

def menu():
    #np.set_printoptions(threshold=sys.maxsize)
    global todayFolderPath
    global experimentFolder
    global tracked_filename
    global world_filename
    global visu
    global cameraThread
    global trackvisuThread
    global tracking_stream_state
    global stream_state
    global writing_socket
    global writing_state

    starting_string = "\n###### --- Tracking Fish ! --- ######\n"
    ending_string = "\n###### --- Tracking Fish Done !!! --- ######\n"
    
    print(starting_string)
      
    #Create folder with today date
    today = date.today() 
    todayFolderPath = createFolder(pathFile,[str(today)])
    print("\n***  The folder for today experiments is " + todayFolderPath + "  ***\n")

     # Testing with file
    TestTrackingFromFile = tp.cst_trackfromfile
    
    # 0- Ask how many fish
    nfish = tp.cst_nFish
            
    # Create folder to store data
    t = time.localtime(time.time())
    f = str(nfish)+"Fish"
    p = str(t.tm_hour).zfill(2) + "-" + str(t.tm_min).zfill(2)
    experimentFolder = createFolder(todayFolderPath, [f, p])
    print("Tracking data is stored in " + experimentFolder + "\n")

    visu = True
   
    if visu:      
        # Starting camera stream
        cameraThread = threading.Thread(target=showCameraStream_dyn)
        cameraThread.daemon = True       
        cameraThread.start()
        
        # Starting tracking stream
        trackvisuThread = threading.Thread(target=showTrackingStream)
        trackvisuThread.daemon = True 
        trackvisuThread.start()
        
    while True:
        print("\n!!!               Ready to track ?                 !!!")
        #print("!!!   Please check that writing server is running  !!!\n")
        key = input("Press \'y\' to proceed or \'n\' to stop ...\n")
        if (key == "Y" or key == "y"):
            visu  = True
            break
        elif (key == "N" or key == "n"):
            stream_state = 'stop'
            tracking_stream_state = 'stop'
            writing_state = 'stop'
            sys.exit(0)
    
    if writing_socket is None:
        connectWritingServer()

    if TestTrackingFromFile:
        mn = tp.cst_mn
        fps = tp.cst_framerate
        p = tp.cst_path
        f = tp.cst_bagfilename
        
        trackFromFile(nfish, mn, fps, False, visu, p, f)
            #IRTrackFromFile2(nfish, mn, fps, False, visu, p, f)
    else:
        tracked_filename = experimentFolder + "/tracked.csv"
        world_filename = experimentFolder + "/world.csv"
        trackFish(nfish, tp.cst_mn, tp.cst_framerate, False, visu)
    
    while True:
        print(ending_string)
        input("Press any key to stop ...\n")
        stream_state = 'stop'
        tracking_stream_state = 'stop'
        writing_state = 'stop'
        sys.exit(0)

if __name__ == '__main__':
    if tp.cst_interactiveMenu:
        interactiveMenu()
    else:
        menu()    