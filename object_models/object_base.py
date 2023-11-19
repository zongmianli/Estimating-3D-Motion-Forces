import math
import numpy as np
import numpy.linalg as LA
from abc import ABCMeta, abstractmethod

import pinocchio as se3
from pinocchio.utils import *
from pinocchio.explog import exp,log

from lib.utils import rgb_to_hex
try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    from lib.display import Visual
except ImportError:
    print("object_base.py: Visual not imported")

class ObjectBase:
    '''
    The abstract base class ObjectBase defines a free-floating root joint and basis functionalities of an object.
    Some important attributes:
    viewer: Meshcat visualizer for creating 3D objects and place them.
    visuals: the list of all the 'visual' 3D objects to render the robot, each element of the list being an object Visual.
    model: the kinematic tree of the object model.
    '''
    __metaclass__ = ABCMeta

    def __init__(self,
                 num_contacts=2,
                 num_keypoints=1,
                 viewer=None):
        self.num_contacts = num_contacts # often =2 for left/right contact
        self.num_keypoints = num_keypoints # stick end is the only keypoint
        self.vis = False
        if viewer is not None:
            self.vis = True
            self.viewer = viewer
            self.visuals = []
            self.joint_size = 0.02
            self.colors = {
                "joints": [255,255,255],
                "links": [153,153,153],
                "contact_points": [255,255,255],
                "left_hand_contact": [255,214,0],
                "right_hand_contact": [127,255,0],
                "base_joint": [255,25,25], # handle end
                "keypoints": [25,255,25] # tool head
            }

        self.model = se3.Model()
        # use the y-axis as the direction of gravity
        model = self.model
        gravity = model.gravity
        gravity.linear = np.matrix([0.,-gravity.linear[2],0.]).T
        # arguments that will be updated later
        self.name = None
        self.njoints = 0
        self.data = None


    def CreateBaseJoint(self, inertia, placement):
        '''
        Defines a free-floating base joint for object 6D pose.
        '''
        if self.model.njoints == 1:
            base_id = self.model.addJoint(
                0, se3.JointModelFreeFlyer(), placement, self.name)
            self.model.appendBodyToJoint(
                base_id, inertia, se3.SE3.Identity())
            if self.vis:
                if self.name == "ground":
                    visual_name = "ground_base"
                    self.viewer[visual_name].set_object(
                        g.Box([5., 0.0001, 5.]),
                        g.MeshLambertMaterial(
                            color=rgb_to_hex([102,51,0]),
                            reflectivity=0.5,
                            opacity=0.8))
                    self.visuals.append(
                        Visual(visual_name, se3.SE3.Identity(), base_id, 'j'))
                else:
                    visual_name = "object_base"
                    visual_size = self.joint_size
                    visual_color = self.colors["base_joint"]
                    self.viewer[visual_name].set_object(
                        g.Sphere(visual_size),
                        g.MeshLambertMaterial(
                            color=rgb_to_hex(visual_color),
                            reflectivity=0.5,
                            opacity=1.))
                    self.visuals.append(
                        Visual(visual_name, se3.SE3.Identity(), base_id, 'j'))
        else:
            raise ValueError("The model is not empty!")
        return base_id


    def CreateKeypoints(self, names, parent_ids, inertia):
        '''
        Create keypoints given the lists of names, parent_ids and inertiae.
        The keypoints are implemented as Pinocchio joints.
        '''
        # Sanity check
        num_keypoints = self.num_keypoints
        if (num_keypoints!=len(names) or num_keypoints!=len(parent_ids) or num_keypoints!=len(inertia)):
            raise ValueError("Sanity check failed: num_keypoints!=len(names) or num_keypoints!=len(parent_ids) or num_keypoints!=len(inertia)")

        for i in range(num_keypoints):
            joint_id = self.model.addJoint(
                parent_ids[i],
                se3.JointModelPZ(),
                se3.SE3.Identity(),
                names[i])
            self.model.appendBodyToJoint(
                joint_id,
                inertia[i],
                se3.SE3.Identity())
            if self.vis:
                visual_name = names[i]
                visual_color = self.colors["keypoints"]
                visual_size = self.joint_size
                self.viewer[visual_name].set_object(
                    g.Sphere(visual_size),
                    g.MeshLambertMaterial(
                        color=rgb_to_hex(visual_color),
                        reflectivity=0.5,
                        opacity=1.))
                self.visuals.append(
                    Visual(visual_name, se3.SE3.Identity(), joint_id, 'j'))


    def Display(self, q_pino, q_ctt, q_endpt=None, update_data=True):
        if hasattr(self, "viewer"):
            if update_data:
                if self.name == "ground":
                    # update operational frames for ground contact points
                    # we put x and z translation to column 0 and 1, respectively
                    q_ctt = q_ctt.reshape((-1,2))
                    for fid in range(1, self.num_contacts+1):
                        contact_pos = self.model.frames[fid].placement.translation
                        contact_pos[[0,2]] = q_ctt[fid-1]
                        self.model.frames[fid].placement.translation = contact_pos
                    se3.framesForwardKinematics(self.model, self.data, q_pino)
                else:
                    # object contact points are implemented as virtual joints
                    q = np.vstack((q_pino, q_ctt, q_endpt))
                    se3.forwardKinematics(self.model, self.data, q)
            for visual in self.visuals:
                if visual.parent_type=='j': # joint
                    visual.Place(self.viewer, self.data.oMi[visual.parent_id])
                elif visual.parent_type=='f': # operational frame
                    visual.Place(self.viewer, self.data.oMf[visual.parent_id])
        else:
            raise ValueError("The object does not have a viewer!")


    def DeleteVisuals(self, list_visuals=None):
        '''
        Remove a set of object visuals from the 3D scene.
        All object visuals will be deleted if list_visuals is None.
        '''
        delete_all = True if list_visuals is None else False
        if hasattr(self, "viewer"):
            # Retrieve the list of visuals and remove the visuals from the GUI
            visuals_new = []
            for visual in self.visuals:
                name = visual.name
                if delete_all or name in list_visuals:
                    self.viewer[visual.name].delete()
                else:
                    visuals_new.append(visual)
            # Save the new list of visuals
            self.visuals = visuals_new
        else:
            print("viewer does not exist. Nothing deleted.")


    def GenerateDecoration(self):
        '''
        Generates the decoration matrix of a kinematic tree that is created using Pinocchio.
        Note that we ignore the virtual joints (e.g. contact points and object keypoints).
        '''
        # initialize the decoration matrix with -1 value
        num_joints = self.njoints - 1 # exclude the universe joint
        if self.name != "ground":
            num_joints -= self.num_contacts + self.num_keypoints
        decoration = np.zeros((num_joints, 5)).astype(int) - 1
        decoration = np.matrix(decoration)
        cumulateQIndex = 0
        cumulateQPinoIndex = 0
        for j in range(num_joints):
            # joint ID, beginning with 0
            decoration[j,0] = j
            # ID of parent joint
            decoration[j,1] = self.model.parents[j+1]-1
            # joint type encoded as integers
            jointTypeStr = self.model.joints[j+1].shortname()
            if jointTypeStr=="JointModelFreeFlyer":
                decoration[j,2] = 1
            elif jointTypeStr=="JointModelSpherical":
                decoration[j,2] = 2
            elif jointTypeStr=="JointModelRX":
                decoration[j,2] = 3
            elif jointTypeStr=="JointModelRY":
                decoration[j,2] = 4
            elif jointTypeStr=="JointModelRZ":
                decoration[j,2] = 5
            elif jointTypeStr=="JointModelPX":
                decoration[j,2] = 6
            elif jointTypeStr=="JointModelPY":
                decoration[j,2] = 7
            elif jointTypeStr=="JointModelPZ":
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
                print("warning (generateDecoration): unknown joint type!")
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
                print("warning (generateDecoration): unknown joint type!")
        return decoration


    @abstractmethod
    def CreateObject():
        pass


    @abstractmethod
    def CreateContactPoints():
        pass
