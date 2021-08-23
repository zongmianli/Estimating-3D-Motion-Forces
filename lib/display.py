import numpy as np
import numpy.linalg as LA
from pinocchio.utils import *
from pinocchio.explog import exp,log
import pinocchio as se3
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from lib.utils import rgb_to_hex

class Visual:
    '''
    Class representing one 3D mesh of the body parts, to be attached to a
    joint. The class contains:
    * the name of the 3D objects inside Gepetto viewer.
    * the ID of the joint in the kinematic tree to which the body is attached.
    * the placement of the body with respect to the joint frame.
    This class is only used in the list Limb.visuals (see below).
    '''
    def __init__(self, name, placement, parent_id=None, parent_type=None):
        self.name = name
        self.placement = placement # placement of the body wrt joint, i.e. jointMbody
        if parent_id is not None:
            self.parent_id = parent_id # ID of the parent joint or frame
        if parent_type is not None:
            self.parent_type = parent_type # 'j' (joint) or 'f' (frame)


    def Place(self, viewer, oMi):
        oMbody = oMi*self.placement
        viewer[self.name].set_transform(oMbody.homogeneous.getA())


class PointCloud:
    '''
    This class makes easier the display of a set of points (either in 2D or in 3D).
    '''
    def __init__(self, cloud_name, viewer, names,
                 colors=None,
                 opacity=0.8,
                 size=0.03):
        self.cloud_name = cloud_name
        self.viewer = viewer
        self.num_points = len(names)
        self.names = [cloud_name+'_'+names[i] for i in range(self.num_points)]
        if colors is None or len(colors) != self.num_points:
            colors = [[255,255,255] for i in range(self.num_points)]
        self.colors = colors
        self.opacity = opacity
        self.size = size
        self.CreateVisuals()


    def CreateVisuals(self):
        size = self.size
        for i in range(self.num_points):
            name = self.names[i]
            color = self.colors[i]
            self.viewer[name].set_object(
                    g.Sphere(size),
                    g.MeshLambertMaterial(
                            color=rgb_to_hex(color),
                            reflectivity=0.5,
                            opacity=self.opacity))


    def DeleteVisuals(self):
        for name in self.names:
            self.viewer[name].delete()


    def Display(self, q):
        for i in range(self.num_points):
            name = self.names[i]
            placement = se3.SE3(eye(3), q[(3*i):(3*i+3),0])
            self.viewer[name].set_transform(placement.homogeneous.getA())


    def Display2d(self, q):
        translation = np.matrix(np.zeros(3)).T
        for i in range(self.num_points):
            name = self.names[i]
            translation[:2, 0] = q[(3*i):(3*i+2),0]
            placement = se3.SE3(eye(3), translation)
            self.viewer[name].set_transform(placement.homogeneous.getA())
