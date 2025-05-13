"""
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: © 2022 Renaud Bastien, Stephane Sanchez <stephane.sanchez@ut-capitole.fr>
Université Toulouse Capitole - Institut de Recherche en Informatique de Toulouse
Real Expression Artificial Life (REVA) Research Team
https://www.irit.fr/departement/calcul-intensif-simulation-optimisation/reva/
"""

import pyrealsense2 as rs
import numpy as np
import time

resRGB = [[1920,1080,30],[1280,720,60],[1280,720,30],[848,480,90],[848,480,30],[640,480,60], [640,480,30]]
resDepth = [[1280,720,30],[848,480,90],[848,480,30],[640,480,90], [640,480,30]] 


class Camera:
    def __init__(self,rgb = True,depth = True,infrared = False,rgbMode = 0,depthMode = 0,infraredMode = 0,
                        laser=True,exposureIR = 5000,exposureRGB = 78,filtre = False):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.depth = depth
        self.rgb = rgb
        self.infrared = infrared
        self.laser = laser
        self.reference = False
        self.referenceFrame = None
        self.rgbMode = rgbMode
        self.depthMode = depthMode
        self.infraredMode = infraredMode
                      
        if depth:
            self.config.enable_stream(rs.stream.depth, resDepth[depthMode][0], resDepth[depthMode][1], rs.format.z16, resDepth[depthMode][2])
        if rgb:
            self.config.enable_stream(rs.stream.color, resRGB[rgbMode][0], resRGB[rgbMode][1], rs.format.bgr8, resRGB[rgbMode][2])
        if infrared:
            self.config.enable_stream(rs.stream.infrared, 1,resDepth[infraredMode][0], resDepth[infraredMode][1], rs.format.y8, resDepth[infraredMode][2])   
            self.config.enable_stream(rs.stream.infrared, 2,resDepth[infraredMode][0], resDepth[infraredMode][1], rs.format.y8, resDepth[infraredMode][2])        

        self.profile = self.pipeline.start(self.config)
        self.filter = filtre

        if rgb:
            self.rgbSensor = self.profile.get_device().first_color_sensor()
            self.rgbSensor.set_option(rs.option.enable_auto_exposure,0)
            self.rgbSensor.set_option(rs.option.exposure,exposureRGB)
        if depth:
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_sensor.set_option(rs.option.emitter_enabled,laser)
            self.depth_sensor.set_option(rs.option.enable_auto_exposure,0)
            self.depth_sensor.set_option(rs.option.exposure,exposureIR)
            self.depth_sensor.set_option(rs.option.depth_units,0.0001)   
        if self.filter:
            self.setFilter()
            
        self.alignSteam = "depth" #Align to right ir sensor if "ir", to color img if "color"
        self.align = rs.align(rs.stream.depth) #Align to defaut ir left sensor
    
    def testFrames(self):
        frames = self.pipeline.wait_for_frames()
        return frames

    def getFrames(self):
        frames = self.pipeline.wait_for_frames()
        frame = []
        if self.depth:
            depthImage = frames.get_depth_frame()
            if self.filter:
                depthImage = self.spatialFilter.process(depthImage)
                depthImage = self.temporalFilter.process(depthImage)
            colorizer = rs.colorizer()
            colored = colorizer.colorize(self.depthImage)
            frame.append(np.asanyarray(colored.get_data()))
            #frame.append(self.convertFrameToNumpy(depthImage))
        if self.rgb:
            frame.append(self.convertFrameToNumpy(frames.get_color_frame()))
        if self.infrared:
            frame.append(self.convertFrameToNumpy(frames.get_infrared_frame(1)))
            frame.append(self.convertFrameToNumpy(frames.get_infrared_frame(2)))
        return frame

    def convertFrameToNumpy(self,frame):
        return np.asanyarray(frame.as_frame().get_data())


    def getFramesAligned(self):
        frames = self.pipeline.wait_for_frames()
        frame = []
        aligned_frames = self.align.process(frames)
        if self.depth:
            self.aligned_depth_frame = aligned_frames.get_depth_frame()
            self.depth_intrinsics = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            if self.filter:
                self.aligned_depth_frame = self.spatialFilter.process(self.aligned_depth_frame)
                self.aligned_depth_frame = self.temporalFilter.process(self.aligned_depth_frame)
            
            #frame.append(self.convertFrameToNumpy(self.aligned_depth_frame))

            colorizer = rs.colorizer()
            colored = colorizer.colorize(self.aligned_depth_frame)
            frame.append(np.asanyarray(colored.get_data()))
        if self.rgb:
            color_frame = aligned_frames.get_color_frame()
            self.color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            frame.append(self.convertFrameToNumpy(color_frame))
        if self.infrared:
            infrared_frameL = aligned_frames.get_infrared_frame(1)
            infrared_frameR = aligned_frames.get_infrared_frame(2)
            self.infrared_intrinsics = infrared_frameL.profile.as_video_stream_profile().intrinsics
            frame.append(self.convertFrameToNumpy(infrared_frameL))
            frame.append(self.convertFrameToNumpy(infrared_frameR))
        
        return frame


    def stop(self):
        self.pipeline.stop()

    def laserSwitch(self):
        self.laser = not self.laser
        self.depth_sensor.set_option(rs.option.emitter_enabled,self.laser)
        
    def laserOff(self):
        self.laser = False
        self.depth_sensor.set_option(rs.option.emitter_enabled,self.laser)
    
    def laserOn(self):
        self.laser = True
        self.depth_sensor.set_option(rs.option.emitter_enabled,self.laser)
    

    def alignToImg(self,stream="depth"):
        if stream == "color":
            align_to = rs.stream.color
            self.alignStream = "color"
        elif stream == "ir":
            align_to = rs.stream.infrared
            self.alignStream = "ir"
        else:
            align_to = rs.stream.depth
            self.alignStream = "depth"
        self.align = rs.align(align_to)

    def getReferenceFrame(self):
        self.referenceFrame = None
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        self.referenceFrame = aligned_frames.get_depth_frame()
        if self.filter:
            self.referenceFrame = self.spatialFilter.process(self.referenceFrame)
            self.referenceFrame = self.temporalFilter.process(self.referenceFrame)
        return self.convertFrameToNumpy(self.referenceFrame)
        

    def getIntrinsics(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        if self.depth:
            aligned_depth_frame = aligned_frames.get_depth_frame()
            self.depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        if self.rgb:
            color_frame = aligned_frames.get_color_frame()
            self.color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        if self.infrared:
            infrared_frame = aligned_frames.get_infrared_frame(1)
            self.infrared_intrinsics = infrared_frame.profile.as_video_stream_profile().intrinsics
            

    def getReferenceDistance(self,step = 1):
        self.getIntrinsics()
        self.reference = True
        frame = []
        if self.alignStream == "depth" or self.alignStream == "ir":
            nx = resDepth[self.rgbMode][0]
            ny = resDepth[self.rgbMode][1]
            intrinsics = self.infrared_intrinsics
        elif self.alignStream == "color":
            nx = resRGB[self.rgbMode][0]
            ny = resRGB[self.rgbMode][1]
            intrinsics = self.color_intrinsics
        imShape = (ny,nx)
        self.referenceDistance = np.zeros(imShape)
        for l in range(0,step):
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depthFrame = aligned_frames.get_depth_frame()
            distance = np.zeros(imShape)
            for k in range(0,nx):
                for j in range(0,ny): 
                    distance[j,k] = depthFrame.get_distance(k, j)
            self.referenceDistance += distance/step
        return self.referenceDistance


    def getWorldPosition(self,x,y):
        distance = self.aligned_depth_frame.get_distance(x, y)
        if self.alignStream == "color":
            intrinsics = self.color_intrinsics
        else:
            intrinsics = self.infrared_intrinsics
        distancePoint = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], distance)
        return distancePoint

    def getPixelFromPosition(self,x,y,z):
        if self.alignStream == "color":
            intrinsics = self.color_intrinsics
        else:
            intrinsics = self.infrared_intrinsics
        pixel = rs.rs2_project_point_to_pixel(intrinsics, [x, y, z])
        return pixel

    def simulateDepth(self,X):
        self.getIntrinsics()
        if self.alignStream == "color":
            size = (resRGB[self.rgbMode][1], resRGB[self.rgbMode][0],3)
        else:
            size = (resDepth[self.infraredMode][1], resDepth[self.infraredMode][0],3) 
        depth = np.zeros(size) 
        nx = np.zeros(size[:2])
        for k in range(0,np.shape(X)[0]):
            pixel = self.getPixelFromPosition(X[k,0],X[k,1],X[k,2])

            try:
                if pixel[1]>0 and pixel[0]>0 and pixel[1]<720 and pixel[0]<1280: 
                    depth[int(pixel[1]),int(pixel[0]),:] += X[k,:]
                    nx[int(pixel[1]),int(pixel[0])] += 1

            #    print(depth[int(np.floor(pixel[1])),int(np.floor(pixel[0])),:])
            except:
                pass
        nx = np.moveaxis(np.tile(nx,(3,1,1)),0,2)
        return depth/nx

    def setFilter(self,temporal = True,spatial = True):
        self.filter = True
        self.spatialFilter = rs.spatial_filter()
        self.temporalFilter = rs.temporal_filter()
        #print(self.spatialFilter.get_supported_options())
        self.spatialFilter.set_option(rs.option.filter_magnitude,2)
        self.spatialFilter.set_option(rs.option .filter_smooth_alpha,.5)    
        self.spatialFilter.set_option(rs.option.filter_smooth_delta,10)
        #print(self.spatialFilter.get_option_range( rs.option.holes_fill ))
        self.spatialFilter.set_option(rs.option.holes_fill,5)
        self.temporalFilter.set_option(rs.option.filter_smooth_alpha,.5)    
        self.temporalFilter.set_option(rs.option.filter_smooth_delta,10)
         
    def setHighAccuracy(self):
        self.depth_sensor.set_option(rs.option.visual_preset, 3)


    def setAdvancedDepth(self,deepSeaMedianThreshold = 796,deepSeaNeighborThreshold = 108,
                            deepSeaSecondPeakThreshold = 647,lrAgreeThreshold = 10,
                            minusDecrement = 2,plusIncrement = 25,
                            scoreThreshA = 4,scoreThreshB = 2893,
                            textureCountThreshold = 0,textureDifferenceThreshold = 1722,
                            rsmRemoveThresh = 86):
        advancedMode = rs.rs400_advanced_mode(self.profile.get_device())
        advancedMode.toggle_advanced_mode(True)
        time.sleep(5)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if self.depth:
            self.config.enable_stream(rs.stream.depth, resDepth[self.depthMode][0], resDepth[self.depthMode][1], rs.format.z16, resDepth[self.depthMode][2])
        if self.rgb:
            self.config.enable_stream(rs.stream.color, resRGB[self.rgbMode][0], resRGB[self.rgbMode][1], rs.format.bgr8, resRGB[self.rgbMode][2])
        if self.infrared:
            self.config.enable_stream(rs.stream.infrared, 1,resDepth[self.infraredMode][0], resDepth[self.infraredMode][1], rs.format.y8, resDepth[self.infraredMode][2])
            self.config.enable_stream(rs.stream.infrared, 2,resDepth[self.infraredMode][0], resDepth[self.infraredMode][1], rs.format.y8, resDepth[self.infraredMode][2])        
        self.profile = self.pipeline.start(self.config)
        advancedMode = rs.rs400_advanced_mode(self.profile.get_device())

        control = advancedMode.get_depth_control()

        #control.deepSeaMedianThreshold = deepSeaMedianThreshold
        #control.deepSeaNeighborThreshold = deepSeaNeighborThreshold
        #control.deepSeaSecondPeakThreshold = deepSeaSecondPeakThreshold
        #control.lrAgreeThreshold = lrAgreeThreshold
        #control.minusDecrement = minusDecrement
        #control.plusIncrement = plusIncrement
        #control.scoreThreshA = scoreThreshA
        #control.scoreThreshB = scoreThreshB
        #control.textureCountThreshold = textureCountThreshold
        #control.textureDifferenceThreshold = textureDifferenceThreshold

        #advancedMode.set_depth_control(control)
        self.setHighAccuracy()
        rsm = advancedMode.get_rsm()
        rsm.rsmBypass = 0
        rsm.diffThresh = 1.65625
        rsm.sloRauDiffThresh = 0.78125
        rsm.removeThresh = rsmRemoveThresh
        advancedMode.set_rsm(rsm)

        self.advancedMode = advancedMode
        self.control = control
        self.depth_sensor.set_option(rs.option.depth_units,0.0001)   


    def setExposureIR(self,exposureIR):
        self.depth_sensor.set_option(rs.option.exposure,exposureIR)

    def setGainIR(self,gainIR):
        self.depth_sensor.set_option(rs.option.gain,gainIR)

    def setExposureRGB(self,exposureRGB):
        self.rgbSensor.set_option(rs.option.exposure,exposureRGB)



    


