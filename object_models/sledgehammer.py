import numpy as np
import numpy.linalg as LA
from abc import ABCMeta, abstractmethod

import pinocchio as se3
from pinocchio.utils import *
from pinocchio.explog import exp, log

from object_base import ObjectBase
try:
    from lib.display import Visual
except ImportError:
    print("sledgehammer.py: Visual not imported")

class Tool(ObjectBase):
    '''
    Creates a kinematic model for sledgehammer.
    '''

    def CreateObject(self, params):
        # update basic info
        self.name = "sledgehammer"
        # [width, height, length]
        head_shape = None if "head_shape" not in params else params["head_shape"]
        stick_length = .5 if "stick_length" not in params else params["stick_length"]
        stick_radius = 0.015 if "stick_radius" not in params else params["stick_radius"]
        mass = 2. if "mass" not in params else params["mass"]
        self.params = {"head_shape": head_shape,
                       "stick_length": stick_length,
                       "stick_radius": stick_radius,
                       "mass": mass}
        self.joint_names = ["hammer_base"]
        self.body_names = ["hammer_stick"]
        if head_shape is not None:
            self.body_names.append("hammer_head")
        # create base joint and assign inertia
        I = eye(3)
        c = zero(3)
        inertia = se3.Inertia(mass, c, I)
        base_id = self.CreateBaseJoint(inertia, se3.SE3.Identity())
        if self.vis:
            name = self.body_names[0]
            self.viewer.gui.addCylinder(
                    'world/'+name,
                    stick_radius,
                    stick_length,
                    self.colors["links"])
            placement = se3.SE3(eye(3), np.matrix([0., 0., stick_length/2.]).T)
            self.visuals.append(Visual('world/'+name, placement, base_id, 'j'))
            if head_shape is not None:
                name = self.body_names[1]
                self.viewer.gui.addBox('world/'+name, head_shape[0],
                                    head_shape[1], head_shape[2],
                                    self.colors["links"])
                placement = se3.SE3.Identity()
                self.visuals.append(Visual('world/'+name, placement, base_id, 'j'))
        # create contact points and endpoints
        self.CreateContactPoints(base_id)
        self.CreateKeypoints(base_id)
        self.njoints = self.model.njoints # is this used later?
        # create Pinocchio data object
        self.data = self.model.createData()
        # generate decoration tree
        self.decoration = self.GenerateDecoration()

    def CreateContactPoints(self, parent_joint_id):
        '''
        Defines contact points as (virtual) leaf joints on the object kinematics tree.
        '''
        self.contact_point_names = []
        for i in range(self.num_contacts):
            name = "{0}_contact_point_{1}".format(self.name, i)
            joint_id = self.model.addJoint(
                parent_joint_id,
                se3.JointModelPZ(),
                se3.SE3.Identity(),
                name)
            # NOTE: is it necessary to append body to joint
            self.model.appendBodyToJoint(
                joint_id,
                se3.Inertia(1e-6, zero(3), eye(3)),#se3.Inertia.Zero(), # virtual joint has no inertia
                se3.SE3.Identity())
            self.contact_point_names.append(name)
            if self.vis:
                if i==0:
                    color = self.colors["left_hand_contact"]
                elif i==1:
                    color = self.colors["right_hand_contact"]
                else:
                    color = self.colors["contact_points"]
                self.viewer.gui.addSphere(
                    "world/"+name, 1.5*self.joint_size, color)
                self.visuals.append(
                    Visual("world/"+name, se3.SE3.Identity(), joint_id, 'j'))
