#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: © 2022 Boris Lenseigne, Renaud Bastien, Stephane Sanchez <stephane.sanchez@ut-capitole.fr>
Université Toulouse Capitole - Institut de Recherche en Informatique de Toulouse
Real Expression Artificial Life (REVA) Research Team
https://www.irit.fr/departement/calcul-intensif-simulation-optimisation/reva/
"""

#speed up mgrid, define once

# in pointclmoudfromamp, indexes define in one line

from sys import exit
# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
#from moviepy.video.io.bindings import mplfig_to_npimage
import csv
from .exTensionFilterTracker import ExtensionFilter
import math

# Global variables and magic numbers
# Just something small enough to be considered zero
cst_Epsilon = 0.0000001
# Thresholding of the mahanalobis distance according to 3 DoF chi squared law
# [0.072, 0.115, 0.216, 0.352, 0.584, 6.251, 7.815, 9.348, 11.345, 12.838]
cst_ChiSquaredThreshold = 12.838 #5.0 #9.348 #12.838 #12.838
# Typical fish length in pixels (rather under than over estimate)
cst_FishSize = 5
# Never to exceed fish size
cst_MaxFishSize = 50
# Arbitrary high distance.
cst_AbsoluteMaxDist = 100000
# Boundaries for valid depth range. Since it seems that de bowl
# itself is not visible (zero values pixels), we use this assumption
# for now and do not restrain the depth in distance. Adjust if needed
#cst_MinDepth = 1
#cst_MaxDepth = 65535  # 16 bit data
cst_MinDepth = 1
cst_MaxDepth = 65535

# Minimal number of points to consider when we want to initialize a new Fish
cst_minValidFishPoints = 3
# Frames per second
cst_FPS = 30
# number of fishes
cst_nFishes = 10
# tracking mode - 0 : frame - 1 : depth map - 2 : composite
cst_trackingMode = 1
# dimensions
cst_dims = 3

#visualisation
cst_show_ir = True
cst_show_coord = False 

class Fish:
    """
    A class that represents a single fish in the tank.

    A class that represents a single fish in the tank. It associates a tracker,
    some metadata and a few utilities.
    """
    
    def __init__(self, name, fps=cst_FPS, nDims = cst_dims, startPos=None, index=-1, verbose=True):
        """
        Create a fish.

        Parameters
        ----------
        name : String
            name of the fish
        startPos : tuple
            start position of the fish, if known
        index: int
            index of the fish (eg. in a fish list). Defaults to -1.
        Returns
        -------
        None.

        """
        self.verbose = verbose
        self.index = index
        self.name = name[0]
        ########################
        # Status flags come here
        ########################
        # How many turns we allow without getting a measurement before
        # reseting the filter.
        self.maxPredictionOnlyTurns = 5
        # Some numerical instability occured the tracker needs to be reset
        self.needReset = False
        self.timeUpdated = False
        self.measUpdated = False

        self.dims = nDims

        if startPos is not None:
            if self.dims>2:
                X0 = np.array([[startPos[0]],
                            [startPos[1]],
                            [startPos[2]],
                            [50],
                            [50],
                            [1]])
            else:
                X0 = np.array([[startPos[0]],
                            [startPos[1]],
                            [50],
                            [50]])
        else:
            X0 = None
        self.predictionNoise = 1
        self.measurementNoise = 1
        self.tracker = ExtensionFilter(self.predictionNoise,
                                       self.measurementNoise,
                                       X0,
                                       nDims = self.dims,
                                       fps=fps)
        # List of 3D points identified as being part of the fish at each time
        # step.It is "None" if there was no measurement
        self.fishPoints = None

    def setVerbose(self, verbose=True):
        self.verbose = verbose

    def dManaPred(self, points):
        """
        Mahanalobis from the the predicted fish position.

        Mahanalobis distance between the predicted fish position and a 3D
        point array.

        Parameters
        ----------
        point : n x 3 array of points ([[x, y, z]])
            The 3D point we're interested row by row.

        Returns
        -------
        nx1 array: The Mahanalobis distance between the predicted
        (prior)position and the requested points.

        """
        fishCovMat = self.tracker.X_kkm
        fishState = self.tracker.x_kkm
        #if np.any(np.iscomplex(fishState)):
        #    print("fishTraker : line 139 ... fishState has complex values")
        fishPos = np.transpose(fishState[0:self.dims])
        return self.dMana(fishPos, fishCovMat, points)

    def dManaCorr(self, points):
        """
        Mahanalobis from the the correected fish position.

        Mahanalobis distance between the corrected fish position and a 3D
        point array.

        Parameters
        ----------
        point : n x 3 array of points ([[x, y, z]])
            The 3D point we're interested row by row.

        Returns
        -------
        nx1 array: The Mahanalobis distance between the corrected (posterior)
        position and the requested points.

        """
        fishCovMat = self.tracker.X_kk
        fishState = self.tracker.x_kk
        #if np.any(np.iscomplex(fishState)):
        #    print("fishTraker : line 162 ... fishState has complex values")
        fishPos = np.transpose(fishState[0:self.dims])
        return self.dMana(fishPos, fishCovMat, points)

    def dManaPredError(self, points):
        """
        Mahanalobis from the the predicted fish position and estimation error.

        Mahanalobis distance between the predicted fish position and a 3D
        point array. Considering the prediction error covariance matrix

        Parameters
        ----------
        point : n x 3 array of points ([[x, y, z]])
            The 3D point we're interested row by row.

        Returns
        -------
        nx1 array: The Mahanalobis distance between the corrected (posterior)
        position and the requested points.

        """
        fishCovMat = self.tracker.P_kkm[0:self.dims, 0:self.dims]
        fishState = self.tracker.x_kkm
        #if np.any(np.iscomplex(fishState)):
        #    print("fishTraker : line 186 ... fishState has complex values")
        fishPos = np.transpose(fishState[0:self.dims])
        return self.dMana(fishPos, fishCovMat, points)

    def dMana(self, centrum, covMat, points):
        """
        Mahanalobis distance from a given distribution.

        Parameters
        ----------
        centrum: 1x3 array
            Barycentrum of the fish
        covMat: 3x3 array
            The covariance matrix to consider.
        point: n x 3 array of points ([[x, y, z]])
            The 3D point we're interested row by row.
        Returns
        -------
        nx1 array: The Mahanalobis distance between the desired point cloud
        the requested distribution.

        """
        fishCovMatInv = np.linalg.inv(covMat)
        #if np.any(np.iscomplex(fishCovMatInv)):
        #    print("fishTraker : line 207 ... complex values")
        # mu = np.repeat(centrum, points.shape[0], axis=0)
        # offset = (points - mu)
        dSquare = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            pt = points[i, :]
            offset = (pt - centrum)
            dSquare[i] = np.matmul(offset,
                                   np.matmul(fishCovMatInv,
                                             np.transpose(offset)))[0, 0]
        dSquareRoot = np.sqrt(dSquare)
        #if np.any(np.iscomplex(dSquareRoot)):
        #    print("fishTraker : line 226 ... dSquareRoot has complex values")
        
        return dSquareRoot

    def dCartPred(self, points):
        """
        Cartesian from the the predicted fish position.

        Cartesian distance between the predicted fish position and a 3D
        point array.

        Parameters
        ----------
        point : n x 3 array of points ([[x, y, z]])
            The 3D point we're interested row by row.

        Returns
        -------
        a Float: The cartesian distance between the predicted (prior)position
        and the requested points.

        """
        fishState = self.tracker.x_kkm
        #if np.any(np.iscomplex(fishState)):
        #    print("fishTraker : line 256 ... fishState has complex values")
        fishPos = np.transpose(fishState[0:self.dims])
        return self.dCart(fishPos, points)

    def dCartCorr(self, points):
        """
        Cartesian from the the correected fish position.

        Cartesian distance between the corrected fish position and a 3D
        point array.

        Parameters
        ----------
        point : n x 3 array of points ([[x, y, z]])
            The 3D point we're interested row by row.

        Returns
        -------
        a Float: The Cartesian distance between the corrected (posterior)
        position and the requested points.

        """
        fishState = self.tracker.x_kk
        #if np.any(np.iscomplex(fishState)):
        #    print("fishTraker : line 280 ... fishState has complex values")
        fishPos = np.transpose(fishState[0:self.dims])
        return self.dCart(fishPos, points)

    def dCart(self, centrum, points):
        """
        Cartesian distance beetwen a point (centrum) and a point cloud.

        Parameters
        ----------
        point: n x 3 array of points ([[x, y, z]])
            The 3D point we're interested row by row.
        centrum: 1x3 array
            The point we want to measure the distance from
        covMat: 3x3 array
            Fish extension covariance matrix
        Returns
        -------
        a Float: The Mahanalobis distance between the desired point cloud
        the requested points.

        """
        mu = np.repeat(centrum, points.shape[0], axis=0)
        offset = (points - mu)
        dSquare = np.sum(offset * offset, axis=1)

        dSquareRoot = np.sqrt(dSquare)

        #if np.any(np.iscomplex(dSquareRoot)):
        #    print("fishTraker : line 308 ... dSquareRoot has complex values")

        return dSquareRoot

    def statePred(self):
        """
        Return the predicted state of the fish.

        Returns
        -------
        x_kkm : 6x1 array
            State vector [[x, y, z, dx, dy, dz]]
        X_kkm : 3 x 3 array
            fish spread covariance matrix.

        """
        x_kkm = self.tracker.x_kkm
        X_kkm = self.tracker.X_kkm
        return (x_kkm, X_kkm)

    def stateCorr(self):
        """
        Return the corrected state of the fish.

        Returns
        -------
        x_kk : 6x1 array
            State vector [[x, y, z, dx, dy, dz]]
        X_kk : 3 x 3 array
            fish spread covariance matrix.

        """
        x_kk = self.tracker.x_kk
        X_kk = self.tracker.X_kk
        return (x_kk, X_kk)

    def setPos(self, pos):
        """
        Update the corrected fish position

        Parameters
        ----------
        pos : 3x1 array
            x, y, z position.

        Returns
        -------
        None.

        """
        for i in range(0, self.dims):
            self.tracker.x_kk[i, 0] = pos[i]

        #self.tracker.x_kk[0, 0] = pos[0]
        #self.tracker.x_kk[1, 0] = pos[1]
        #self.tracker.x_kk[2, 0] = pos[2]

    def setExtension(self, pointCloud):
        """
        Update the extension of the fish.
        Update the extension of the fish with the covariance matrix of the
        point cloud. The method updates the corrected extension X_kk.

        Parameters
        ----------
        pointCloud : n x 3 array
            Fish point cloud[[x, y, z]]

        Returns
        -------
        None.

        """
        #print(len(pointCloud))
        if len(pointCloud) > 2 :
            X_kk = np.cov(np.transpose(pointCloud))

            #if np.any(np.iscomplex(X_kk)):
            #    print("fishTraker : line 381 ... X_kk has complex values")

            # Ensure the matrix is non singular (when all points have the same Z)
            # . Since its symetric, we ensure there are no zeros on the diagonal.
            for i in range(self.tracker.nDims):
                if (X_kk[i, i] < cst_Epsilon):
                    X_kk[i, i] = cst_Epsilon

            self.tracker.X_kk = X_kk
            self.tracker.alpha_kk = pointCloud.shape[0]

    def reInit(self, pointCloud):
        """
        Reinitializes the tracker with a given point cloud.

        Reinitializes the tracker in the center of the cloud and updates
        the spread accordingly. The speed is left unchanged

        Parameters
        ----------
        pos : 3x1 array
            x, y, z position.
        Extension : 3x3 array
            Extension covariance matrix (symetric semi-definite positive)

        Returns
        -------
        4 x 1 array The difference between old and new state vector.

        """
        oldState = self.tracker.x_kk
        self.fishPoints = None
        if (pointCloud.size > 0):
            self.fishPoints = pointCloud
            pos = np.mean(pointCloud, axis=0)
            self.setPos(pos)
            self.setExtension(pointCloud)
            self.timeUpdated = False
            self.measUpdated = False
            self.needReset = False
            self.tracker.predictionOnlyTurns = 0
        return (oldState - self.tracker.x_kk)

    def timeUpdate(self):
        """
        Perform a time update step ofthe fish tracker.

        Returns
        -------
        None.

        """
        self.tracker.timeUpdate()
        self.timeUpdated = True
        self.measUpdated = False

    def measurementUpdate(self, fishPoints):
        """
        Perform a correction step given a point cloud.

        Parameters
        ----------
        fishPoints : n x 3 array
            [x,y,z] points belonging to the fish

        Returns
        -------
        None.

        """
        
        if (fishPoints.size > 0):
        #if (fishPoints.size > cst_minValidFishPoints):
            self.fishPoints = fishPoints
            yBar = np.transpose(np.array([np.mean(fishPoints, axis=0)]))
            # If there is a single measurement, the covariance matrix
            # computation goes wrong this fixes it.
            if fishPoints.shape[0] == 1:
                YBar = np.identity(self.tracker.nDims)
            else:
                YBar = np.cov(np.transpose(fishPoints))
            nk = fishPoints.shape[0]
            self.tracker.measurementUpdate(yBar, YBar, nk)
            if self.verbose:
                print("fish.index = ", self.index, " Update")
        else:
            self.tracker.propagatePrediction()
            self.fishPoints = None
            if self.verbose:
                print("fish.index = ", self.index, " Propagate")
        if self.tracker.hadComplexValues is True : #or fishPoints.size<cst_minValidFishPoints:
            #    self.reInit(fishPoints)
            self.needReset = True
            print("fish.index = ", self.index, " Has complex values, reset !")
        if self.tracker.predictionOnlyTurns >= self.maxPredictionOnlyTurns:
            self.needReset = True
            if self.verbose :
                print("fish.index = ",
                    self.index,
                    " predictionOnlyTurns = ",
                    self.tracker.predictionOnlyTurns,
                    " Reset !")
        self.timeUpdated = False
        self.measUpdated = True


class FishTank:
    """
    A Class that represents the fish tank.

    It receives deph maps and has fishes.
    Tracking of the fishes is performed in 3D in the coordinate system of the
    depth map, with its origin in the top left corner and the Y axis pointing
    downwards.
    """
    
    pctCount = 0

    def __init__(self, nFishes, fps, frame, depthMap, irFrame, mask, nDims = cst_dims, trackingMode = cst_trackingMode, verbose=True):
        """
        Initializes a fish Tank with fishes

        Parameters
        ----------
        nFishes : int
            number of fishes in the tank.
        depthMap : Intel RealSense deth map
            The first depth map of the data set.
        mask : h x w array
            Binary mask for valid points in the RS frames

        Returns
        -------
        None.

        """
        self.verbose = verbose

        self.fps = fps

        self.needReset = False  # True if one of the fishes is lost for too long
        
        if (mask.ndim > 2):
            self.mask = mask[:, :, 0]  # keep only one color band for masking
        else :
            self.mask = mask

        self.mask[self.mask>0] = 1

        self.nFishes = nFishes

        self.dims = nDims
        self.trackingMode = trackingMode

        # load fish names list and select funny names for the fishes
        csvReader = csv.reader(open('./names.csv'), delimiter=",")
        self.fishNames = list(csvReader)
        seed = np.random.randint(0, len(self.fishNames))
        self.fishList = []

        self.frame = frame
        self.depthMap = depthMap
        self.irFrame = irFrame
        
        if self.trackingMode == 0:
            self.trackingImage = np.copy(self.frame)
            self.YX = np.mgrid[0:self.trackingImage.shape[0], 0:self.trackingImage.shape[1]]
        elif self.trackingMode == 1:
            self.trackingImage = np.copy(self.depthMap)
            self.YX = np.mgrid[0:self.trackingImage.shape[0], 0:self.trackingImage.shape[1]]
        elif self.trackingMode == 2:
            self.trackingImage = np.copy(self.frame)
            self.YX = np.mgrid[0:self.trackingImage.shape[0], 0:self.trackingImage.shape[1]]

        self.initFishes(seed)
        # Display the tank.
        # The figure where we show everything
        #self.fig = plt.figure()
        #self.ax = self.fig.add_subplot(1, 1, 1)
        #self.colorizer = rs.colorizer()

    def updateFishes(self, frame, depthMap, irFrame):
        """
        Update the status of nthe fish when provided with a new depth map.

        Parameters
        ----------
        dMapFrame : RealSense frame objet
            A new frame.

        Returns
        -------
        None.

        """
        self.frame = frame
        self.depthMap = depthMap
        self.irFrame = irFrame

        if self.trackingMode == 0:
            self.trackingImage = np.copy(self.frame)
        elif self.trackingMode == 1:
            self.trackingImage = np.copy(self.depthMap)
        elif self.trackingMode == 2:
            self.trackingImage = np.copy(self.frame)

        if self.needReset is True:
            self.resetFishes()
        else:
            pointCloud = self.pointCloudFromMap()
            
            distToFish = np.zeros((pointCloud.shape[0], self.nFishes))

            # Assign points to the fishes according to Mahanalobis distance.
            # We add a Threshold following chi squared law.
            # This avoids assigning irrelevant points.
            for i in range(self.nFishes):
                fish = self.fishList[i]
                # Apply the time update step
                fish.timeUpdate()
                searchCovMat = (fish.tracker.X_kk +
                                fish.tracker.P_kkm[0:self.dims, 0:self.dims] +
                                fish.tracker.R)
                state, _ = fish.statePred()
                pos = np.transpose(state[0:self.dims])
                distToFish[:, i] = fish.dMana(pos,
                                              searchCovMat,
                                              pointCloud)
                # Increase the Mahanalobis distance threshold if there was
                # no measurements for some time.
                # adjustedChiSquareThreshold = (cst_ChiSquaredThreshold *
                #                               (fish.tracker.predictionOnlyTurns + 1))
                # Filter out invalid points to far away to belong to any fish
                distToFish[distToFish > cst_ChiSquaredThreshold] = cst_AbsoluteMaxDist
            idx = np.any((distToFish < cst_AbsoluteMaxDist), axis=1)
            distToFish = distToFish[idx, :]
            pointCloud = pointCloud[idx, :]
            # Gather the points for each fish
            pointToFish = np.argmin(distToFish, axis=1)

            for i in range(self.nFishes):
                # measurement update step
                fish = self.fishList[i]
                fishPoints = pointCloud[pointToFish == i, :]
                fish.measurementUpdate(fishPoints)
                self.needReset = (self.needReset or fish.needReset)


    def initFishes(self, seed):
        """
        Find fishes in a depth Map.

        Find fishes in a depth Map and initializes the fishes in the tank.
        It assumes that all fishes are visible in the first frame
        Fishes are found using a Monte-Carlo process.

        Parameters
        ----------
        seed: int
            Random seed to pick up names for the fishes
        Returns
        -------
        None.

        """
        # Get all points potentially belonging to fishes.
        pointCloud = self.pointCloudFromMap()
        # Init fishes
        fishPointCloud = pointCloud
        for i in range(self.nFishes):
            if fishPointCloud.size > 0:
                fishCenter = fishPointCloud[0, :]
            else:
                if self.dims > 2:
                    fishCenter = [0, 0, 0]
                else:
                    fishCenter = [0, 0]

            fish = Fish(self.fishNames[i],    #self.fishNames[seed + i % len(self.fishNames)],
                        self.fps,
                        self.dims,
                        fishCenter,
                        i,
                        self.verbose)
            fishPointCloud, fish = self.initSingleFish(fishPointCloud, fish)
            self.fishList.append(fish)

    def initSingleFish(self, pointCloud, fish):
        """
        Initialization of the first fish in the point Cloud.

        Parameters
        ----------
        pointCloud : n x 3 array
            [[x,y,z]] point cloud
        fish : a fish
            The fish to initialize
        Returns
        -------
        The point cloud minus the point selected for the current fish,
        The fish updated from the pointCloud

        """

        distToFish = fish.dCartCorr(pointCloud)
        #print(distToFish)
        fishPoints = distToFish < cst_FishSize
        fish.reInit(pointCloud[fishPoints, :])
        # 1.4 Iterate
        converged = False

        while not converged:
            # Assign points to the fish
            searchCovMat = (fish.tracker.X_kk +
                            fish.tracker.P_kkm[0:self.dims, 0:self.dims] +
                            fish.tracker.R)
            state, _ = fish.statePred()
            pos = np.transpose(state[0:self.dims])
            # Restrain the search area to a reasonable size for a fish
            distCardToFish = fish.dCartPred(pointCloud)
            distManaToFish = fish.dMana(pos,
                                        searchCovMat,
                                        pointCloud)

            fishPoints = np.logical_and(distManaToFish < cst_ChiSquaredThreshold,
                                        distCardToFish < cst_MaxFishSize)
            stateShiftVector = fish.reInit(pointCloud[fishPoints, :])
            posOffset = stateShiftVector[0:self.dims, 0]
            converged = (np.matmul(np.transpose(posOffset),
                                   posOffset) < cst_Epsilon)
            if self.verbose:
                print("initFish : ", np.matmul(np.transpose(posOffset), posOffset))
        # Once converged we remove the selected points form the point cloud
        pointCloud = pointCloud[np.invert(fishPoints), :]
        return (pointCloud, fish)

    def resetFishes(self):
        """
        Perform an extensive point search and resets the fishes in the tank.

         Find fishes in a depth Map and resets the fishes in the tank.
         It assumes that all fishes are visible in the current frame
         Fishes are found using a Monte-Carlo process.

         Parameters
         ----------
         seed: int
             Random seed to pick up names for the fishes
         Returns
         -------
         None.

         """
        self.needReset = False
        # 1- Get all points potentially belonging to fishes.
        pointCloud = self.pointCloudFromMap()

        fishPointCloud = pointCloud
        for i in range(self.nFishes):
            fish = self.fishList[i]
            # Check wether there are enough siginficant points to build a fish
            if fishPointCloud.shape[0] < cst_minValidFishPoints:
                self.needReset = True
                break
            else:
                fish.setPos(fishPointCloud[0, :])
                fishPointCloud, fish = self.initSingleFish(fishPointCloud,
                                                           fish)

    def pointCloudFromMap(self):
        """
        Return XYZ coordinates of relevant points in the current depth Map

        Parameters
        ----------

        Returns
        -------
        None.

        """
        # 1- Select all points potentially belonging to fishes.
        # 1.1 Apply masking
        if self.trackingMode == 2:
            tImage = self.trackingImage
            tImage[self.mask == 0] = 0

            depthImage = self.depthMap
            depthImage[self.mask == 0] = 0
            
            # 1.2 Thresholding, get x,y,z point cloud
            pointMap = np.zeros(tImage.shape)
            #pointMap[tImage >= cst_MinDepth] = tImage[tImage >= cst_MinDepth]
            #pointMap[pointMap > cst_MaxDepth] = 0
            idx = (tImage >= cst_MinDepth) & (tImage < cst_MaxDepth)
            pointMap[idx] = tImage[idx]

            dMap = np.zeros(tImage.shape)
            #dMap[depthImage >= cst_MinDepth] = depthImage[depthImage >= cst_MinDepth]
            #dMap[dMap > cst_MaxDepth] = 0
            idx = (depthImage >= cst_MinDepth) & (depthImage < cst_MaxDepth)
            dMap[idx] = depthImage[idx]

            pointMap[pointMap > 0] = dMap[pointMap > 0]

            #YX = np.mgrid[0:tImage.shape[0], 0:tImage.shape[1]]
            Y = self.YX[0]
            X = self.YX[1]
            Y = Y[pointMap > 0]
            X = X[pointMap > 0]
            if self.dims>2:
                Z = pointMap[pointMap > 0]
                pointCloud = np.stack((X, Y, Z), axis=1)
            else:
                pointCloud = np.stack((X, Y), axis=1)
        else:
            tImage = self.trackingImage
        
            tImage[self.mask == 0] = 0
            
            #self.trackingImage = tImage

            # 1.2 Thresholding, get x,y,z point cloud
            pointMap = np.zeros(tImage.shape)
            #pointMap[tImage >= cst_MinDepth] = tImage[tImage >= cst_MinDepth]
            #pointMap[pointMap > cst_MaxDepth] = 0
            idx = (tImage >= cst_MinDepth) & (tImage < cst_MaxDepth)
            pointMap[idx] = tImage[idx]
            
            Y = self.YX[0]
            X = self.YX[1]
            idx = (pointMap > 0)
            Y = Y[idx]
            X = X[idx]
            if self.dims>2:
                Z = pointMap[idx]
                pointCloud = np.stack((X, Y, Z), axis=1)
            else:
                pointCloud = np.stack((X, Y), axis=1)
        return pointCloud

    def draw2DCovMatEllipse(self,
                            covMat,
                            pos=(0, 0),
                            nstd=10,
                            edgeColor='red'):
        """
        Create a plot of the covariance confidence ellipse.

        Create a plot of the covariance confidence ellipse from a covariance
        Matrix. Stolen and adapted from:
        https://stats.stackexchange.com/questions/361017/proper-way-of-estimating-the-covariance-error-ellipse-in-2d/361334

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.
        covMat : 2x2 array
            Covariance Matrix
        pos : tuple
            The center of the matrix
        n_std : float
            The number of standard deviations to determine the ellipse's
            radiuses.
        edgeColor: see matplotlib documentation
            The color to draw with
        Returns
        -------
        matplotlib.patches.Ellipse
        Other parameters
        ----------------
        """

        # Eigenvalues and eigenvectors of the covariance matrix.
        if covMat.shape[0] > 2:
            covMat = covMat[0:2, 0:2]
        vals, vecs = np.linalg.eigh(covMat)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * nstd * np.sqrt(vals)
        # Ellipse
        ellipse1 = Ellipse(xy=pos, width=width, height=height, angle=theta,
                           edgecolor=edgeColor, fc='None', lw=2, zorder=4)
        self.ax.add_patch(ellipse1)
       # plt.show()

    def add2DCovMatEllipse( self,
                            image,
                            covMat,
                            pos=(0, 0),
                            nstd=10,
                            edgeColor=(0,0,255)):
        """
        Create a plot of the covariance confidence ellipse.

        Create a plot of the covariance confidence ellipse from a covariance
        Matrix. Stolen and adapted from:
        https://stats.stackexchange.com/questions/361017/proper-way-of-estimating-the-covariance-error-ellipse-in-2d/361334

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.
        covMat : 2x2 array
            Covariance Matrix
        pos : tuple
            The center of the matrix
        n_std : float
            The number of standard deviations to determine the ellipse's
            radiuses.
        edgeColor: see matplotlib documentation
            The color to draw with
        Returns
        -------
        matplotlib.patches.Ellipse
        Other parameters
        ----------------
        """

        # Eigenvalues and eigenvectors of the covariance matrix.
        if covMat.shape[0] > 2:
            covMat = covMat[0:2, 0:2]
        vals, vecs = np.linalg.eigh(covMat)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * nstd * np.sqrt(vals)
        
        if not math.isnan(width) and not math.isnan(height):
            img = cv2.ellipse(image, pos, 
                              (int(width), int(height)),
                              theta, 0, 360, edgeColor, 1)
        else:
            img = image
        return img

    def showTank_new(self):
        colors = np.array([(255, 0, 0), (0,255,0), (0,0,255) , 
                           (255,255,0), (0,255,255), (255,0,255), 
                           (138,0,255), (0, 138, 255), (138,255,0),
                           (255,0,138), (0,255,138), (255,255,255)])
        
        global cst_show_ir
        global cst_show_coord

        if cst_show_ir:
            colorImage = cv2.cvtColor(self.irFrame*self.mask, cv2.COLOR_GRAY2RGB)
        else :
            colorImage = cv2.cvtColor(self.trackingImage*self.mask, cv2.COLOR_GRAY2RGB)
        
        #print(depthColorImage.shape, end=" ")

        indexes = []
        for i in range(self.nFishes):
            indexes.append(i)
        
        """    
        done = False
        while not done:
            done=True
            for i in range(self.nFishes-1):
                fj = self.fishList[indexes[i]]
                fj1 = self.fishList[indexes[i+1]]
                print("-------------- " + str(i) + " ------" + str(fj1))
                if fj.fishPoints.size < fj1.fishPoints.size:
                    tmp = indexes[i]
                    indexes[i]=indexes[i+1]
                    indexes[i+1]=tmp
                    done=False
        """    

        # Draw fish
        for i in range(self.nFishes):
            fish = self.fishList[indexes[i]] #self.fishList[i]
            fishState, fishExt = fish.stateCorr()
            fishState = np.real(fishState)
            
            #print(np.iscomplex(fishState), fishState[0], fishState[1])
            #if np.any(np.iscomplex(fishState)):
            #    print("fishTraker : line 880 ... fishState has complex values")

            #coloring blob
            if fish.fishPoints is not None:
                cpt = 0
                for p in fish.fishPoints :
                    colorImage[int(p[1]), int(p[0])] = colors[i]
                    
            # Draw fish ellipse
            #depthColorImage = self.add2DCovMatEllipse(depthColorImage, 
            #                                          fishExt, 
            #                                          pos=(int(fishState[0]), int(fishState[1])))
            

            # Point at fish position
            colorImage = cv2.circle(    colorImage, 
                                        (int(fishState[0]), int(fishState[1])), 
                                        2, (200,200,200), -1)
            s1 = " " + str(fish.index) #+ " : " + fish.name 
            colorImage = cv2.putText(   colorImage,
                                        s1, 
                                        (int(fishState[0]), int(fishState[1]) + 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                        (255,255,255),
                                        1)
            
            if cst_show_coord:
                s2 = "("
                for i in range(0, self.dims-1):
                    s2 = s2 + str(round(fishState[i,0], 2)) + ", "
                s2 = s2 + str(round(fishState[self.dims-1,0], 2)) + ")"
                colorImage = cv2.putText(   colorImage,
                                            s2, 
                                            (int(fishState[0]), int(fishState[1]) + 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                            (255,0,0),
                                            1)
        return colorImage

    def showTank(self):
        """
        Display the tank.
        Shows the depth map with the identified fishes overlaid

        Returns
        -------
        None.

        """
                
        colors = np.array(['.c', '.m', '.y', '.r', '.b', '.g', '.w'])
        # Colorize depth frame to jet colormap
        depthColorFrame = self.colorizer.colorize(self.dMapFrame)

        # Convert depth_frame to numpy array to render image in opencv
        depthColorImage = np.asanyarray(depthColorFrame.get_data())
        
        # Render the depth map
        self.ax.clear()
        self.ax.imshow(depthColorImage)
        # Draw he fishes
        for i in range(self.nFishes):
            fish = self.fishList[i]
            fishState, fishExt = fish.stateCorr()
            self.ax.plot(fishState[0], fishState[1], '+r')
            self.ax.text(fishState[0], fishState[1] + 5,
                         (fish.name + " \nz = " +
                          str(round(fishState[2, 0], 2))),
                         color='r')
            # print("Fish : ", fish.index, " Ext: ", fishExt)
            self.draw2DCovMatEllipse(fishExt, pos=(fishState[0], fishState[1]))
            
            if fish.fishPoints is not None:
                self.ax.plot(fish.fishPoints[:, 0],
                             fish.fishPoints[:, 1], colors[i])
            
        # plt.draw()
        # plt.pause(0.05)
        # plt.show(block=False)
        # if (FishTank.pctCount % 100 == 0):
        #     plt.savefig("./Data/Pictures/saved" + str(FishTank.pctCount) + ".jpg")
        # FishTank.pctCount = FishTank.pctCount + 1
        # plt.show()
        
        #img = mplfig_to_npimage(self.fig)

        img = None
        """
        if (FishTank.pctCount % 1 == 0):
            self.fig.canvas.draw()
            img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        FishTank.pctCount = FishTank.pctCount + 1
        """
        return img

if __name__ == '__main__':
    # Default, test, data set
    # Number of fished to track
    defaultNFishes = cst_nFishes
    # Data file
    defaultBagFile = './Data/1Fish/20220217_112309.bag'

    # Alternative data can be given on the command line
    parser = argparse.ArgumentParser(
        description="Read recorded bag file and tracks fishes in the depth map.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        help="Path to the bag file")
    parser. add_argument("-n",
                         "--nfish",
                         type=int,
                         help="Number of fishes in the tank")
    # Parse the command line arguments to an object
    args = parser.parse_args()
    # Safety if no parameter have been given
    if not args.input:
        bagFile = defaultBagFile
        nFishes = defaultNFishes
    else:
        # Check if the given file have bag extension
        if os.path.splitext(args.input)[1] != ".bag":
            print("The given file is not of correct file format.")
            print("Only .bag files are accepted")
            exit()
        else:
            bagFile = args.input
            nFishes = args.nfish

    try:
        # Create pipeline
        print("Creating RealSense Pipeline....", end=" ")
        pipeline = rs.pipeline()
        print("Done !")
        # Create a config object
        print("Configuring RealSense....", end=" ")
        config = rs.config()
        print("Done !")
        # Tell config that we will use a recorded device from file to be used
        # +by the pipeline through playback.
        print("Configuring source file: ",
              bagFile,
              "...",
              end=" ")
        rs.config.enable_device_from_file(config, bagFile)

        print("Done !")
        # Configure the pipeline to stream the depth stream
        # Change this parameters according to the recorded bag file resolution
        print("Enabling Streams....", end=" ")
        config.enable_stream(rs.stream.depth, rs.format.z16, cst_FPS)
        print("Done !")
        # Start streaming from file
        print("Starting stream....", end=" ")
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        print("Done !")
        # Create colorizer object
        #colorizer = rs.colorizer()
        # load mask file
        maskFile = bagFile[0:-4] + '_mask.png'
        print(maskFile)
        mask = cv2.imread(maskFile)
        # Create Tracker Objects
        timeSteps = 0
        # Streaming loop
        while True:
            # Get frameset of depth
            frames = pipeline.wait_for_frames()
            depthMap = frames.get_depth_frame()
            print("time = ", timeSteps)
            if timeSteps == 0:  # initialization
                tank = FishTank(nFishes, depthMap, mask)
                tank.showTank()
            else:
                tank.updateFishes(depthMap)
                # if timeSteps > 1904:  # 1250
                #     print("Wait ! ")
                tank.showTank()
            timeSteps += 1

    finally:

        pass
