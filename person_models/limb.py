import math
import numpy as np
import numpy.linalg as LA

import pinocchio as se3
from pinocchio.utils import *
from pinocchio.explog import exp,log

from lib.utils import *
try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    from lib.display import Visual
except ImportError:
    print "limb.py: viewer not imported"

class Limb:
    '''
    The class Limb defines a kinematic chain of >=2 joints.
    All joints are spherical joints, whose rotations are represented as quaternions.

    The members of the class are:
    * viewer: a display encapsulating a gepetto viewer client to create 3D objects and place them.
    * model: the kinematic tree of the model.
    * data: the temporary variables to be used by the kinematic algorithms.
    * visuals: the list of all the 'visual' 3D objects to render the model, each element of the list being
    an object Visual.
    '''

    def __init__(self, viewer=None):
        self.vis = False
        if viewer is not None:
            self.vis = True
            self.viewer = viewer
            self.visuals = []
        self.joint_size = 0.02
        self.model = se3.Model.BuildEmptyModel()
        self.createLimb()
        self.data = self.model.createData()


    def createLimb( self,
                    rootId=0,
                    jointTrans=None,
                    jointNames=None,
                    bodyNames=None,
                    bodyColors=None,
                    inertias=None,
                    opacity=1. ):

        if jointTrans is None:
            raise ValueError('jointTrans is missing!')
        if jointNames is None:
            raise ValueError('jointNames is missing!')
        if bodyNames is None:
            raise ValueError('bodyNames is missing!')
        if bodyColors is None:
            raise ValueError('bodyColors is missing!')
        if inertias is None:
            raise ValueError('inertias is missing!')

        limbColor = bodyColors[-1]

        jointSize = self.joint_size
        jointId = rootId

        for i in range(len(jointNames)):
            # add joints
            jointName = jointNames[i]
            jointColor= bodyColors[i]
            jointPlacement = se3.SE3(eye(3), np.matrix(jointTrans[i]).T)
            jointId = self.model.addJoint(jointId,
                                          se3.JointModelSpherical(),
                                          jointPlacement,
                                          jointName )
            # append body to joint
            bodyName = bodyNames[i]
            m = inertias[bodyName][0]
            I = inertias[bodyName][1]
            if i<len(jointNames)-1:
                # COM is at the midpoint between two joints
                c = (np.matrix(jointTrans[i+1]).T)/2.
            else:
                # rough estimation of end effector's COM
                c = (np.matrix(jointTrans[i]).T)/2.4
            bodyInertia = se3.Inertia(m, c, I)
            self.model.appendBodyToJoint(jointId,
                                         bodyInertia,
                                         se3.SE3.Identity())
            # add visualization
            if self.vis:
                self.viewer[jointName].set_object(
                        g.Sphere(0.3*jointSize),
                        g.MeshLambertMaterial(
                                color=rgb_to_hex(jointColor),
                                reflectivity=0.5,
                                opacity=opacity))
                self.visuals.append(
                        Visual(jointName, se3.SE3.Identity(), jointId, 'j'))

            # add visuals for special joints
            vec0 = np.array([0.,0.,1.])
            if jointName in [self.name+'_l_ankle', self.name+'_r_ankle']:
                # create visual
                s = 1.36 # scaling foot size
                footLength = LA.norm(jointTrans[i+1][::2])*s # compute distance in xz plane
                footWidth = footLength/2.
                footThickness = jointSize
                if self.vis:
                    self.viewer[bodyName].set_object(
                            g.Box([footWidth/2.,
                                   footThickness/2.,
                                   footLength]),
                            g.MeshLambertMaterial(
                                    color=rgb_to_hex(limbColor),
                                    reflectivity=0.5,
                                    opacity=opacity))

                # find visual placement and add visual
                vec1Temp = jointTrans[i+1].copy()
                vec1Temp[1] = 0.
                vec1 = vec1Temp/LA.norm(vec1Temp)
                rotationMat = rotation_matrix(vec0, vec1)
                translation = vec1Temp*(2-s)/2
                translation[1] = jointTrans[i+1][1]
                solePlacement = se3.SE3(rotationMat, np.matrix(translation).T)
                if self.vis:
                    self.visuals.append(
                            Visual(bodyName, solePlacement, jointId, 'j'))

                # save contact point positions expressed in ankle frame
                if jointName == self.name+'_l_ankle':
                    self.leftSoleCttPos = np.zeros((4,3))
                else:
                    self.rightSoleCttPos = np.zeros((4,3))

                # add contact points
                cttLocalPos = np.array(
                    [[-footWidth/2., -footThickness/2., -footLength/2.], # p1
                     [-footWidth/2., -footThickness/2.,  footLength/2.], # p2
                     [ footWidth/2., -footThickness/2.,  footLength/2.], # p3
                     [ footWidth/2., -footThickness/2., -footLength/2.]])# p4

                for k in range(4):
                    cttPtName = jointName+'_ctt'+str(k)
                    cttPtSize = jointSize*0.8
                    cttPos = solePlacement.act(np.matrix(cttLocalPos[k]).T)
                    # save contact positions
                    if jointName == self.name+'_l_ankle':
                        self.leftSoleCttPos[k] = cttPos.getA().reshape(-1)
                    else:
                        self.rightSoleCttPos[k] = cttPos.getA().reshape(-1)
                    cttPlacement = se3.SE3(np.matrix(np.eye(3)), cttPos)
                    # or alternatively
                    parentFid = self.model.nframes-1
                    self.model.addFrame(
                        se3.Frame(cttPtName, jointId, parentFid,
                                  cttPlacement, se3.FrameType.OP_FRAME))

            elif jointName in [self.name+'_l_toes', self.name+'_r_toes']:
                if self.vis:
                    # create visual
                    toeLength = LA.norm(jointTrans[i][::2])/4. # nearly 1/4 foot length
                    toeWidth = LA.norm(jointTrans[i][::2])*0.8
                    toeThickness = jointSize*0.8
                    self.viewer[bodyName].set_object(
                            g.Box([toeWidth/2.,
                                   toeThickness/2.,
                                   toeLength]),
                            g.MeshLambertMaterial(
                                    color=rgb_to_hex(limbColor),
                                    reflectivity=0.5,
                                    opacity=opacity))

                    # find visual placement and add visual
                    vec1Temp = jointTrans[i].copy()
                    vec1Temp[1] = 0.
                    vec1 = vec1Temp/LA.norm(vec1Temp)
                    rotationMat = rotation_matrix(vec0, vec1)
                    translation = vec1Temp/8.
                    self.visuals.append(Visual(bodyName,
                            se3.SE3(rotationMat, np.matrix(translation).T),
                            jointId, 'j'))
            elif jointName in [self.name+'_spine_4']:
                if self.vis:
                    limbLength = LA.norm(jointTrans[i])
                    self.viewer[bodyName].set_object(
                            g.Box([jointSize,
                                   jointSize,
                                   limbLength]),
                            g.MeshLambertMaterial(
                                    color=rgb_to_hex(limbColor),
                                    reflectivity=0.5,
                                    opacity=opacity))

                    # find visual placement and add visual
                    vec1 = jointTrans[i]/LA.norm(jointTrans[i])
                    rotationMat = rotation_matrix(vec0, vec1)
                    self.visuals.append(Visual(bodyName,
                            se3.SE3(rotationMat, np.matrix(jointTrans[i]).T),
                            jointId, 'j'))

            elif jointName in [self.name+'_l_fingers', self.name+'_r_fingers']:
                if self.vis:
                    # create visual:
                    limbLength = LA.norm(jointTrans[i])
                    self.viewer[bodyName].set_object(
                            g.Box([jointSize,
                                   jointSize,
                                   limbLength/1.4]),
                            g.MeshLambertMaterial(
                                    color=rgb_to_hex(limbColor),
                                    reflectivity=0.5,
                                    opacity=opacity))

                    # find visual placement and add visual
                    vec1 = jointTrans[i]/LA.norm(jointTrans[i])
                    rotationMat = rotation_matrix(vec0, vec1)
                    self.visuals.append(Visual(bodyName,
                            se3.SE3(rotationMat, (np.matrix(jointTrans[i]).T)/2.4),
                            jointId, 'j'))

            # add visuals for ordinary joints
            else:
                if self.vis:
                    # create visual
                    limbLength = LA.norm(jointTrans[i+1])
                    self.viewer[bodyName].set_object(
                            g.Box([jointSize,
                                   jointSize,
                                   limbLength]),
                            g.MeshLambertMaterial(
                                    color=rgb_to_hex(limbColor),
                                    reflectivity=0.5,
                                    opacity=opacity))

                    # find visual placement and add visual
                    vec1 = jointTrans[i+1]/LA.norm(jointTrans[i+1])
                    rotationMat = rotation_matrix(vec0, vec1)
                    self.visuals.append(Visual(bodyName,
                            se3.SE3(rotationMat, (np.matrix(jointTrans[i+1]).T)/2.),
                            jointId, 'j'))


    def generateDecoration(self, model=None):
        # load the kinematic tree, i.e. the se3::Model object in Pinocchio
        model = self.model if model is None else model
        # initialize the decoration matrix with -1 value
        decoration = np.zeros((model.njoints-1, 5)).astype(int) - 1
        decoration = np.matrix(decoration)
        cumulateQIndex = 0
        cumulateQPinoIndex = 0
        for j in range(model.njoints-1):
            # joint ID, beginning with 0
            decoration[j,0] = j
            # ID of parent joint
            decoration[j,1] = model.parents[j+1]-1
            # joint type encoded as integers
            jointTypeStr = model.joints[j+1].shortname()
            if jointTypeStr=='JointModelFreeFlyer':
                decoration[j,2] = 1
            elif jointTypeStr=='JointModelSpherical':
                decoration[j,2] = 2
            elif jointTypeStr=='JointModelRX':
                decoration[j,2] = 3
            elif jointTypeStr=='JointModelRY':
                decoration[j,2] = 4
            elif jointTypeStr=='JointModelRZ':
                decoration[j,2] = 5
            elif jointTypeStr=='JointModelPX':
                decoration[j,2] = 6
            elif jointTypeStr=='JointModelPY':
                decoration[j,2] = 7
            elif jointTypeStr=='JointModelPZ':
                decoration[j,2] = 8
            # index in configuration vector that we start reading the joint
            decoration[j,3] = cumulateQIndex
            if decoration[j,2]==1:
                cumulateQIndex += 6
            elif decoration[j,2]==2:
                cumulateQIndex += 3
            elif decoration[j,2] in [3,4,5,6,7,8]:
                cumulateQIndex += 1
            else:
                print "warning (generateDecoration): unknown joint type!"
            # index in Pinocchio configuration vector we start reading the joint
            # note that Pinocchio uses quaternion for rotation
            decoration[j,4] = cumulateQPinoIndex
            if decoration[j,2]==1:
                cumulateQPinoIndex += 7
            elif decoration[j,2]==2:
                cumulateQPinoIndex += 4
            elif decoration[j,2] in [3,4,5,6,7,8]:
                cumulateQPinoIndex += 1
            else:
                print "warning (generateDecoration): unknown joint type!"
        return decoration


    def display(self, qPino, updateData=True):
        if hasattr(self, 'viewer'):
            if updateData:
                se3.framesForwardKinematics(self.model, self.data, qPino)
            for visual in self.visuals:
                if visual.parent_type=='j': # joint
                    visual.Place( self.viewer, self.data.oMi[visual.parent_id] )
                elif visual.parent_type=='f': # operational frame
                    visual.Place( self.viewer, self.data.oMf[visual.parent_id] )
        else:
            raise ValueError('The body model does not have a viewer!')
