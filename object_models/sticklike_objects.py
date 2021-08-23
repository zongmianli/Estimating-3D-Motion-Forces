import numpy as np
import numpy.linalg as LA
from abc import ABCMeta, abstractmethod

import pinocchio as se3
from pinocchio.utils import *
from pinocchio.explog import exp, log

from object_base import ObjectBase
try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    from lib.display import Visual
except ImportError:
    print "sticklike_objects.py: Visual not imported"
from lib.utils import load_parameters_from_txt,rgb_to_hex,rotation_matrix


class StickLikeObject(ObjectBase):
    '''
    Modelling stick-like objects such as hammer, spade, barbell and scythe.
    '''
    def CreateObject(self, params_path):
        params = load_parameters_from_txt(params_path)
        name = params["name"][0]
        fix_contact_points = params["fix_contact_points"][0]
        handle_length_bounds = params["handle_length_bounds"][:2]
        base_mass = params["base_mass"][0]
        head_mass = params["head_mass"][0]
        oc_mot_s4 = params["oc_mot_s4"][0]

        self.name = name
        self.fix_contact_points = fix_contact_points
        self.handle_length_bounds = handle_length_bounds
        self.oc_mot_s4 = oc_mot_s4

        # Create the base joint (corresponding to the handle end)
        # (float)mass, (object)lever, (object)inertia
        base_inertia = se3.Inertia(base_mass, zero(3), eye(3))
        self.base_id = self.CreateBaseJoint(base_inertia, se3.SE3.Identity())

        if self.vis:
            handle_length = (handle_length_bounds[0]+handle_length_bounds[1])/2.
            self.UpdateHandleVisuals(handle_length)
        # create contact points and endpoints
        self.CreateContactPoints()
        head_inertia = se3.Inertia(head_mass, zero(3), eye(3))
        self.CreateKeypoints(["object_head"], [self.base_id], [head_inertia])
        self.njoints = self.model.njoints # is this used later?
        # create Pinocchio data object
        self.data = self.model.createData()
        # generate decoration tree
        self.decoration = self.GenerateDecoration()

        # Delete visuals for virtual objects
        if self.vis and self.name=="virtual_object":
            self.DeleteVisuals()


    def CreateContactPoints(self):
        '''
        Defines contact points as (virtual) leaf joints on the object kinematics tree.
        '''
        for i in range(self.num_contacts):
            name = "object_contact_point_{0}".format(i)
            joint_id = self.model.addJoint(
                self.base_id,
                se3.JointModelPZ(),
                se3.SE3.Identity(),
                name)
            # NOTE: is it necessary to append body to joint
            self.model.appendBodyToJoint(
                joint_id,
                se3.Inertia(1e-6, zero(3), eye(3)), # virtual joint should not have
                se3.SE3.Identity())

            if self.vis:
                if i==0:
                    color = self.colors["left_hand_contact"]
                elif i==1:
                    color = self.colors["right_hand_contact"]
                else:
                    color = self.colors["contact_points"]
                self.viewer[name].set_object(
                    g.Sphere(1.5*self.joint_size),
                    g.MeshLambertMaterial(
                        color=rgb_to_hex(color),
                        reflectivity=0.5,
                        opacity=1.))
                self.visuals.append(
                    Visual(name, se3.SE3.Identity(), joint_id, 'j'))


    def UpdateHandleVisuals(self, handle_length):
        '''
        Create a 3D model of the object's handle given its length.
        If an old model already exists in the GUI, it will be removed from 
        the scene before the new model is created.
        '''
        name = "object_handle"
        # Delete handle visual if already exists
        self.DeleteVisuals(list_visuals=[name])

        # Add new handle visual
        handle_radius = 0.015
        self.viewer[name].set_object(
            g.Cylinder(handle_length, radius=handle_radius),
            g.MeshLambertMaterial(
                color=rgb_to_hex(self.colors["links"]),
                reflectivity=0.5,
                opacity=1.))

        vec0 = np.array([0.,0.,1.])
        vec1 = np.array([0.,1.,0.])
        rotationMat = rotation_matrix(vec0, vec1)
        placement = se3.SE3(rotationMat,
                            np.matrix([0., 0., handle_length/2.]).T)
        self.visuals.append(
            Visual(name, placement, self.base_id, 'j'))
