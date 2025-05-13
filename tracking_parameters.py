"""
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: © 2022 Renaud Bastien, Stephane Sanchez <stephane.sanchez@ut-capitole.fr>
Université Toulouse Capitole - Institut de Recherche en Informatique de Toulouse
Real Expression Artificial Life (REVA) Research Team
https://www.irit.fr/departement/calcul-intensif-simulation-optimisation/reva/
"""

# File parameters if tracking from file
cst_interactiveMenu = False
cst_trackfromfile = True
cst_path =  "D:/" #"D:/1-Fish-New/" #"E:/videos manips/10/" #"E:/videos manips/" #"D:/2Fish-FondBleu-Nov2023/" #"./Data/Records" #"E:/FishVR-RealSense-DATA/10May2022" #"E:/Records"   
cst_bagfilename = "20240212_165156.bag" #"20240111_154624.bag"  #"20231123_110523.bag" #"20221117_103330.bag" #"20221109_112631.bag" #"20221004_163849.bag"  #"20220909_145941.bag"  #"20220510_144822.bag"    #20220909_145941.bag" #"20220705_112726.bag"

# Stereo or 3D tracking
cst_stereo = False
cst_disparity_shift = 0
cst_exposure = 10000
cst_gain = 77

# auto create mask
cst_createmask = False

# Tracking parameters
cst_nFish = 10
cst_mn = 240
cst_framerate = 30
cst_hist_xyz = 3
cst_hist_phi = 5
cst_histwindow = 10
cst_noisefiltering = 2 # 0: No filter / 1: Mean / 2: Kalman

# Resolution and framerate
# RGB = [[1920,1080,30],[1280,720,60],[1280,720,30],[848,480,90],[848,480,30],[640,480,60], [640,480,30]]
# Depth = [[1280,720,30],[848,480,90],[848,480,30],[640,480,90], [640,480,30]] 
cst_resRGB = 2
cst_resDepthIR = 2

# Calibration and real data 
# From bowl.data file
cst_bowl = [-0.00906913,  0.01295541,  0.38836542]
cst_bowl_radius = 0.26155618496900956

# From watersurface.data file
cst_watersurface_point = [-0.19302024,  0.17076398,  0.47497158]
cst_watersurface_norm = [-0.56383963,  0.60408307,  0.56317716]

# Needed parameters to be computed - DO NO EDIT !
cst_watersurface_dist = 0.
cst_watersurface_center = [0., 0., 0.]

cst_camleft = [0.,0.,0.]
cst_camright = [0.05,0.,0.]
cst_centerleft = [0.,0.]

cst_centerright = [0.,0.]
cst_radiusleft = 0
cst_radiusright = 0

