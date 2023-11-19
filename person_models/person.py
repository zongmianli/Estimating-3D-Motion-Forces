import math
import numpy as np
import numpy.linalg as LA
import pickle as pk

import pinocchio as se3
from pinocchio.utils import *
from pinocchio.explog import exp,log

from lib.utils import *
from .limb import Limb
try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    from lib.display import Visual
except ImportError:
    print("person.py: viewer not imported")


class Person(Limb):
    '''
    Creates a kinematic tree to represent human body, with:
    * Free flyer encoded as quaternion (6DOF, configuration nq=7, configuration velocity nv=6).
    * 5 limbs:
        - left leg (4 joints, 12DOF, encoded as 16D quaternion vector)
        - right leg (4 joints, 12DOF, encoded as 16D quaternion vector)
        - spine (5 joints, 15DOF, encoded as 20D quaternion vector)
        - left arm (5 joints, 15DOF, encoded as 20D quaternion vector)
        - right arm (5 joints, 15DOF, encoded as 20D quaternion vector)
    '''
    def __init__(self, name,
                       smpl_joints_neutral_pose,
                       openpose_keypoints_neutral_pose,
                       inertia_path=None,
                       viewer=None,
                       opacity=1.):
        self.name = name
        self.vis = False
        if viewer is not None:
            self.vis = True
            self.viewer = viewer
            self.visuals = []
        self.joint_size = 0.02
        self.model = se3.Model()
        # Use the y-axis as the direction of gravity
        model = self.model
        gravity = model.gravity
        gravity.linear = np.matrix([0.,-gravity.linear[2],0.]).T

        self.nj = 24 # number of joints
        #              joint names       SMPL  Pinocchio
        jointNames = [  'pelvis',       # 0      0
                        'l_hip',        # 1      1
                        'r_hip',           # 2      5
                        'spine_0',         # 3      9
                        'l_knee',          # 4      2
                        'r_knee',         # 5      6
                        'spine_1',      # 6     10
                        'l_ankle',         # 7      3
                        'r_ankle',        # 8      7
                        'spine_2',         # 9     11
                        'l_toes',          # 10     4
                        'r_toes',         # 11     8
                        'spine_3',      # 12    12
                        'l_scapula',    # 13    14
                        'r_scapula',      # 14    19
                        'spine_4',        # 15    13
                        'l_shoulder',      # 16    15
                        'r_shoulder',     # 17    20
                        'l_elbow',         # 18    16
                        'r_elbow',        # 19    21
                        'l_wrist',         # 20    17
                        'r_wrist',        # 21    22
                        'l_fingers',    # 22    18
                        'r_fingers' ]   # 23    23

        self.jointNames = [name+'_'+jointNames[i] for i in range(self.nj)]
        self.bodyNames = [name+'_'+jointNames[i]+'_link' for i in range(self.nj)]
        # a standard configuration vector for Pinocchio
        self.nqPino = 99 # 7 + 4*16 + 4*7 = 99
        self.nq = 75

        if inertia_path is not None:
            self.inertia_path = inertia_path
            # load human inertia from Galo's biomechanics model
            self.inertias = self.loadInertias(inertia_path)
        else:
            self.inertias = {}
            for j in range(24):
                randY = se3.Inertia.Random()
                self.inertias[self.bodyNames[j]] = (randY.mass, randY.inertia)
        # 3D joint positions of SMPL skeleton in neutral pose
        self.smpl_joints_neutral_pose = smpl_joints_neutral_pose
        # 3D joint positions of Pinocchio skeleton in neutral pose
        joint_ids_smpl = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]
        self.joint_positions_neutral_pose = smpl_joints_neutral_pose[joint_ids_smpl]
        # 3D joint positions of Openpose skeleton in neutral pose
        self.openpose_keypoints_neutral_pose = openpose_keypoints_neutral_pose
        # create the whole body kinematic tree
        self.CreateHumanBody(opacity=opacity)
        self.CreateOpenposeKeypoints(opacity=opacity)
        # create data for Pinocchio
        self.data = self.model.createData()
        # generate decoration tree for person model
        self.decoration = self.generateDecoration()


    def loadInertias(self, pathToInertia):
        with open(pathToInertia, 'rb') as inputf:
            inertias_temp = pk.load(inputf, encoding='latin-1')
            inertias = {}
            for k in inertias_temp.keys():
                inertias[self.name+'_'+k] = inertias_temp[k]
            print("inertias loaded from "+pathToInertia)
            return inertias


    def deleteVisuals(self):
        if hasattr(self, 'viewer'):
            # Remove visuals on the human skeleton
            for i in range(self.nj):
                self.viewer[self.jointNames[i]].delete()
                self.viewer[self.bodyNames[i]].delete()
                if i in [7,8]:
                    for k in range(4):
                        self.viewer[self.jointNames[i]+'_ctt'+str(k)]

            # Remove the virtual markers of Openpose joints
            for name in self.keypoint_names:
                self.viewer['openpose_'+name].delete()
        else:
            print('Viewer does not exist. Nothing deleted.')


    def computeRelativeTranslation(self, j3dPos):
        '''given j3d positions, calculate relative translation between linked joints'''
        leftLegIdx = [0,1,4,7,10]
        rightLegIdx = [0,2,5,8,11]
        spineIdx = [0,3,6,9,12,15]
        leftArmIdx = [9,13,16,18,20,22]
        rightArmIdx = [9,14,17,19,21,23]

        baseTrans     = j3dPos[0]
        leftLegTrans  = j3dPos[leftLegIdx[1:]] - j3dPos[leftLegIdx[:-1]]
        rightLegTrans = j3dPos[rightLegIdx[1:]] - j3dPos[rightLegIdx[:-1]]
        spineTrans    = j3dPos[spineIdx[1:]] - j3dPos[spineIdx[:-1]]
        leftArmTrans  = j3dPos[leftArmIdx[1:]] - j3dPos[leftArmIdx[:-1]]
        rightArmTrans = j3dPos[rightArmIdx[1:]] - j3dPos[rightArmIdx[:-1]]

        return baseTrans,leftLegTrans,rightLegTrans,spineTrans,leftArmTrans,rightArmTrans


    def CreateHumanBody(self, opacity=1.):
        '''create a kinematic tree for the whole person body using createLimb function.'''

        colorPicker = { 'yellow':[255,215,0], # left arm
                        'green' :[127,255,0], # right arm
                        'violet':[138,43,226],
                        'blue'  :[30,144,255],
                        'tomato':[255,0,0], # spine
                        'dark blue': [0, 51, 102], # ground
                        'grey': [204, 204, 204], # limbs
                        'pink' : [255,0,255], # left leg
                        'cyan' : [0,255,255], # right leg
                        'white' :[255,255,255], }

        # compute relative translation between linked joints
        (baseTrans,leftLegTrans,rightLegTrans,spineTrans,leftArmTrans,rightArmTrans) = self.computeRelativeTranslation(self.smpl_joints_neutral_pose)
        # add ground-pelvis joint
        jointPlacement = se3.SE3(eye(3), np.matrix(baseTrans).T)
        self.model.addJoint(0, se3.JointModelFreeFlyer(), jointPlacement, self.jointNames[0])
        # append body to pelvis joint
        bodyName = self.bodyNames[0]
        m = self.inertias[bodyName][0]
        I = self.inertias[bodyName][1]
        c = zero(3)
        bodyInertia = se3.Inertia(m, c, I)
        self.model.appendBodyToJoint(1, bodyInertia, se3.SE3.Identity())
        # add visualization
        if self.vis:
            self.viewer[self.jointNames[0]].set_object(
                g.Sphere(self.joint_size),
                g.MeshLambertMaterial(
                    color=rgb_to_hex(colorPicker['white']),
                    reflectivity=0.5,
                    opacity=opacity))
            self.visuals.append(
                Visual(self.jointNames[0], se3.SE3.Identity(), 1, 'j') )

        # create left leg
        leftLegIndex = [1, 4, 7, 10]
        leftLegJointNames = [self.jointNames[i] for i in leftLegIndex]
        leftLegBodyNames = [self.bodyNames[i] for i in leftLegIndex]
        leftLegColors =   [ colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white'] ] # the last indicates links' color!
        self.createLimb(1, leftLegTrans,
                            leftLegJointNames,
                            leftLegBodyNames,
                            leftLegColors,
                            self.inertias,
                            opacity)

        # create right leg
        rightLegIndex = [2, 5, 8, 11]
        rightLegJointNames = [self.jointNames[i] for i in rightLegIndex]
        rightLegBodyNames = [self.bodyNames[i] for i in rightLegIndex]
        rightLegColors =  [ colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white']] # the last indicates links' color!
        self.createLimb(1, rightLegTrans,
                            rightLegJointNames,
                            rightLegBodyNames,
                            rightLegColors,
                            self.inertias,
                            opacity)

        # create spine
        spineIndex = [3, 6, 9, 12, 15]
        spineJointNames = [self.jointNames[i] for i in spineIndex]
        spineBodyNames = [self.bodyNames[i] for i in spineIndex]
        spineColors = [     colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white']] # the last indicates links' color!
        self.createLimb(1, spineTrans,
                            spineJointNames,
                            spineBodyNames,
                            spineColors,
                            self.inertias,
                            opacity)

        # create left arm
        leftArmIndex = [13, 16, 18, 20, 22]
        leftArmJointNames = [self.jointNames[i] for i in leftArmIndex]
        leftArmBodyNames = [self.bodyNames[i] for i in leftArmIndex]
        leftArmColors = [   colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white'] ] # the last indicates links' color!
        self.createLimb(12, leftArmTrans,
                            leftArmJointNames,
                            leftArmBodyNames,
                            leftArmColors,
                            self.inertias,
                            opacity)

        # create right arm
        rightArmIndex = [14, 17, 19, 21, 23]
        rightArmJointNames = [self.jointNames[i] for i in rightArmIndex]
        rightArmBodyNames = [self.bodyNames[i] for i in rightArmIndex]
        rightArmColors = [  colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white'],
                            colorPicker['white'] ] # the last indicates links' color!
        self.createLimb(12, rightArmTrans,
                            rightArmJointNames,
                            rightArmBodyNames,
                            rightArmColors,
                            self.inertias,
                            opacity)


    def CreateOpenposeKeypoints(self, opacity=1.):
        '''
        Define Openpose on our human model as operational frames.
        '''
        nkeypoints = self.openpose_keypoints_neutral_pose.shape[0] # 18
        self.keypoint_names = ['nose', 'neck',
                    'r_shoulder', 'r_elbow', 'r_wrist',
                    'l_shoulder', 'l_elbow', 'l_wrist',
                    'r_hip', 'r_knee', 'r_ankle',
                    'l_hip', 'l_knee', 'l_ankle',
                    'r_eye', 'l_eye', 'r_ear', 'l_ear']
        self.keypoint_colors = [[255,153,51], # nose: orange
                    [255,153,51], # neck: orange
                    [127,255,0],[127,255,0],[127,255,0], # right arm: green
                    [255,215,0],[255,215,0],[255,215,0], # left arm: yellow
                    [0,255,255],[0,255,255],[0,255,255], # right leg: cyan
                    [255,0,255],[255,0,255],[255,0,255], # left leg: pink
                    [127,255,0], # right eye: green
                    [255,215,0], # left eye: yellow
                    [0,255,255], # right ear: cyan
                    [255,0,255]] # left ear: pink

        keypoint_size = self.joint_size
        keypoint_names = ['openpose_'+self.keypoint_names[i] for i in range(nkeypoints)]
        parent_joint_ids = [13, 11, 19, 20, 21, 14, 15, 16, 0, 5, 6, 0, 1, 2, 13, 13, 13, 13]
        for k in range(nkeypoints):
            keypoint_name = keypoint_names[k]
            parent_joint_id = parent_joint_ids[k] # Pinocchio model counts joints from 1
            # compute the placement of the keypoint wrt its parent joint frame
            keypoint_positions_wrt_parent_joint = \
                self.openpose_keypoints_neutral_pose[k] - self.joint_positions_neutral_pose[parent_joint_id]
            keypoint_placement = se3.SE3(np.matrix(np.eye(3)),
                                         np.matrix(keypoint_positions_wrt_parent_joint).T)
            # attach the keypoint to its parent joint
            parent_frame_id = self.model.nframes-1 # what is a "parent frame"?
            self.model.addFrame(se3.Frame(
                                 keypoint_name,
                                 parent_joint_id + 1,
                                 parent_frame_id,
                                 keypoint_placement,
                                 se3.FrameType.OP_FRAME))
            if self.vis:
                self.viewer[keypoint_name].set_object(
                    g.Sphere(keypoint_size),
                    g.MeshLambertMaterial(
                        color=rgb_to_hex(self.keypoint_colors[k]),
                        reflectivity=0.5,
                        opacity=opacity))
                self.visuals.append(Visual(keypoint_name, se3.SE3.Identity(),
                                           parent_frame_id+1, 'f'))
