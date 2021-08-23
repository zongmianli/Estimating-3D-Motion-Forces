import numpy as np
import numpy.linalg as LA
from abc import ABCMeta, abstractmethod

import pinocchio as se3
from pinocchio.utils import *
from pinocchio.explog import exp, log

from lib.utils import *
try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    from lib.display import Visual
except ImportError:
    print "ground_plane.py: Visual not imported"
from object_base import ObjectBase


class GroundPlane(ObjectBase):
    '''
    This class represents the ground plane using a 6D free-floating joint.
    '''

    def CreateObject(self):
        # update basic info
        self.name = "ground"
        # create base joint and contact points
        base_id = self.CreateBaseJoint(se3.Inertia.Zero(), se3.SE3.Identity())
        self.CreateContactPoints(base_id)
        self.njoints = self.model.njoints
        # create Pinocchio data object
        self.data = self.model.createData()
        # generate decoration tree
        self.decoration = self.GenerateDecoration()


    def CreateContactPoints(self, parent_joint_id):
        for i in range(self.num_contacts):
            # create operational frame with ID i+1 (frame 0 is the universe)
            # i is the ID of parent frame (or previous frame) of frame i+1
            name = "{0}_contact_point_{1}".format(self.name, i)
            self.model.addFrame(
                se3.Frame(name, parent_joint_id, i, se3.SE3.Identity(),
                          se3.FrameType.OP_FRAME))

            if self.vis:
                self.viewer[name].set_object(
                    g.Sphere(self.joint_size),
                    g.MeshLambertMaterial(
                        color=rgb_to_hex(self.colors["contact_points"]),
                        reflectivity=0.5,
                        opacity=0.8))

                self.visuals.append(
                    Visual(name, se3.SE3.Identity(), i+1, 'f'))
