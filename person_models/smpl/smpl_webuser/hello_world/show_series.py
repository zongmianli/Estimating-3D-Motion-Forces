'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


Please Note:
============
This is a demo version of the script for driving the SMPL model with python.
We would be happy to receive comments, help and suggestions on improving this code 
and in making it available on more platforms. 


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]
- OpenCV [http://opencv.org/downloads.html] 
  --> (alternatively: matplotlib [http://matplotlib.org/downloads.html])


About the Script:
=================
This script demonstrates loading the smpl model and rendering it using OpenDR 
to render and OpenCV to display (or alternatively matplotlib can also be used
for display, as shown in commented code below). 

This code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Create an OpenDR scene (with a basic renderer, camera & light)
  - Render the scene using OpenCV / matplotlib


Running the Hello World code:
=============================
Inside Terminal, navigate to the smpl/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python render_smpl.py


'''

import time
import cv2
import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl_webuser.serialization import load_model

## Load SMPL model (here we load the female model)
m = load_model('../../models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

datapath = '/sequoia/data2/zoli/datapublic/SURREAL/smpl_data/smpl_data.npz'
datamass = np.load(datapath)
dirpose = datamass['pose_h36m_S9_Directions 1']

outputpath = '/sequoia/data2/zoli/datazml/h36m_est_shape/temp/'


m.betas[:] = np.zeros(m.betas.size)

## Create OpenDR renderer
#rn = ColoredRenderer()

## Assign attributes to renderer
w, h = (640, 480)

for i in range(9400, dirpose.shape[0]):
    print 'frame = {}'.format(i+1)
    m.pose[:] = dirpose[i,:]
    m.pose[0] = np.pi
    m.pose[1] = 0
    m.pose[2] = 0
    ## Create OpenDR renderer
    rn = ColoredRenderer()

    rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
    rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn.set(v=m, f=m.f, bgcolor=np.zeros(3))

    ## Construct point light source
    rn.vc = LambertianPointLight(
        f=m.f,
        v=rn.v,
        num_verts=len(m),
        light_pos=np.array([-1000,-1000,-2000]),
        vc=np.ones_like(m)*.9,
        light_color=np.array([1., 1., 1.]))

    cv2.imshow('render_SMPL', rn.r)
    print ('..Print any key while on the display window')
    #time.sleep(0.5)
    cv2.waitKey(0)

cv2.destroyAllWindows()


## Could also use matplotlib to display
#import matplotlib.pyplot as plt
#plt.ion()
#plt.imshow(rn.r)
#plt.show()
## import pdb; pdb.set_trace()
