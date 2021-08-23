import time
import numpy as np
import numpy.linalg as LA
import cPickle as pickle
from os import makedirs, remove
from os.path import join, exists, abspath, dirname, basename, isfile
from scipy.optimize import fmin_bfgs, fmin_slsqp, minimize
from glob import glob

import pinocchio as se3
from pinocchio.utils import *
from pinocchio.explog import exp,log

from person_models.person import Person
try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    from lib.display import PointCloud
    from lib.video_recorder import VideoRecorder # TODO
except ImportError:
    print "optimizer.py: PointCloud,VideoRecorder not imported"
from lib.utils import *


class Optimizer(object):

    def __init__(self, video_name,
              seq_len,
              overlap,
              person_model,
              person_loader,
              object_model=None,
              object_loader=None,
              ground_model=None,
              ground_loader=None,
              contact=None,
              camera=None,
              pose_prior=None,
              viewer=None,
              save_dir=None):
        # initialize the attributes
        self.video_name = video_name
        self.seq_len = seq_len
        self.overlap = overlap
        self.person_model = person_model
        self.person_loader = person_loader
        self.nf = person_loader.nt_
        self.dt = person_loader.dt_
        if object_model is not None and object_loader is not None:
            self.object_model = object_model
            self.object_loader = object_loader
        if ground_model is not None and ground_loader is not None:
            self.ground_model = ground_model
            self.ground_loader = ground_loader
        if contact is not None:
            self.contact = contact
        if camera is not None:
            self.camera = camera
        if pose_prior is not None:
            self.pose_prior = pose_prior
        if viewer is not None:
            self.viewer = viewer
            self.generate_force_visual_names()

        # get useful information
        self.list_seqs, self.nseqs = self.GenerateSlidingWindows(
            self.nf, seq_len=seq_len, overlap=overlap)
        self.save_dir = abspath("results") if save_dir is None else save_dir
        if not exists(self.save_dir):
            makedirs(self.save_dir)


    def GenerateSlidingWindows(self, nf, seq_len=10, overlap=2):
        '''
        generates a list of "sliding windows" of fixed length with a given overlap value.
        '''
        list_seqs = []
        f_begin = 0
        while f_begin <= nf:
            f_end = min(f_begin+seq_len, nf)
            list_seqs.append(range(f_begin, f_end))
            if f_end == nf:
                break
            f_begin += seq_len - overlap

        nseqs = len(list_seqs)
        return list_seqs, nseqs


    def generate_force_visual_names(self, vis_cttForce=True,
                                    vis_muscleTorque=False):
        # copy friction generators and contact forces from person_loader
        self.gen6dLeftFoot = self.person_loader.friction_cone_generators_left_foot_
        self.gen6dRightFoot = self.person_loader.friction_cone_generators_right_foot_
        self.gen6dGround = self.person_loader.friction_cone_generators_
        self.Compute6dGroundContactForces()
        if vis_cttForce:
            self.names_forceObj = []
            self.names_forceObj_GT = []
            for n in range(self.contact.num_object_contact_joints):
                self.names_forceObj.append("forceObj_"+str(n+1))
                self.names_forceObj_GT.append("forceObj_GT_"+str(n+1))
            self.names_forceGd = []
            self.names_forceGd_GT = []
            for n in range(self.contact.num_ground_contact_joints):
                self.names_forceGd.append("forceGd_"+str(n+1))
                self.names_forceGd_GT.append("forceGd_GT_"+str(n+1))

        if vis_muscleTorque:
            self.names_muscleTorque = []
            for j in range(1,24):
                self.names_muscleTorque.append("muscleTorque_"+str(j))


    def DeleteForceVisuals(self):
        '''
        Delete forces visuals.
        '''
        if hasattr(self, 'viewer'):
            if hasattr(self, 'names_forceObj'):
                for name in self.names_forceObj:
                    self.viewer[name+'f'].delete()
                    self.viewer[name+'t'].delete()
            if hasattr(self, 'names_forceGd'):
                for name in self.names_forceGd:
                    self.viewer[name+'f'].delete()
                    self.viewer[name+'t'].delete()
            if hasattr(self, 'names_forceObj_GT'):
                for name in self.names_forceObj_GT:
                    self.viewer[name+'f'].delete()
                    self.viewer[name+'t'].delete()
            if hasattr(self, 'names_forceGd_GT'):
                for name in self.names_forceGd_GT:
                    self.viewer[name+'f'].delete()
                    self.viewer[name+'t'].delete()
            if hasattr(self, 'names_muscleTorque'):
                for name in self.names_muscleTorque:
                    self.viewer[name].delete()
        else:
            print 'Viewer does not exist. Nothing deleted.'


    def DeleteGroundTruthForceVisuals(self):
        '''
        Delete Ground Truth forces visuals.
        '''
        if hasattr(self, 'viewer'):
            if hasattr(self, 'names_forceObj_GT'):
                for name in self.names_forceObj_GT:
                    self.viewer[name+'f'].delete()
                    self.viewer[name+'t'].delete()
            if hasattr(self, 'names_forceGd_GT'):
                for name in self.names_forceGd_GT:
                    self.viewer[name+'f'].delete()
                    self.viewer[name+'t'].delete()
        else:
            print 'Viewer does not exist. Nothing deleted.'


    def Compute6dGroundContactForces(self):
        contact_mapping = self.contact.contact_mapping
        contact_state = self.contact.contact_states
        list_ground_contact_joints = self.contact.list_ground_contact_joints
        num_ground_contact_joints = self.contact.num_ground_contact_joints
        ground_friction = self.person_loader.ground_friction_.copy()
        self.seqLocalForceGround = np.matrix(np.zeros((
            6*num_ground_contact_joints, self.nf)))
        for i in range(self.nf):
            ground_friction_vec = ground_friction[:, i]
            for n in range(num_ground_contact_joints):
                j = list_ground_contact_joints[n]
                # compute contact force if joint j is in contact at time step i
                if contact_state[j,i] > 0:
                    fid = contact_mapping[j]
                    if j not in [3, 7]:
                        # c is a frame whose origin is at j, and whose axis are in parallel to world axis
                        coefs = ground_friction_vec[(4*(fid-1)):(4*(fid-1)+4), 0]
                        cPhic = self.gen6dGround * coefs
                    else:
                        coefs = ground_friction_vec[(4*(fid-1)):(4*(fid-1)+16), 0]
                        if j == 3:
                            cPhic = self.gen6dLeftFoot * coefs
                        else:
                            cPhic = self.gen6dRightFoot * coefs
                    # cPhic: spatial forces expressed in local frames of person joint j
                    self.seqLocalForceGround[(6*n):(6*n+6), i] = cPhic.copy()


    def ExpressLocalForcesInParkourFrame(self, person_config):
        '''
        Transforms the coordinate frame of the estimated local contact
        forces to the joint frames defined by the Parkour dataset.
        '''

        list_ground_contact_joints = self.contact.list_ground_contact_joints
        list_object_contact_joints = self.contact.list_object_contact_joints
        num_ground_contact_joints = self.contact.num_ground_contact_joints
        num_object_contact_joints = self.contact.num_object_contact_joints

        contact_forces_parkour = np.zeros((self.nf, 4, 6))

        joint_contact_force_mapping = [
            0, # 0
            0, # 1
            1, # 2 l_knee
            1, # 3 l_ankle
            1, # 4 l_toes
            0, # 5
            2, # 6 r_knee
            2, # 7 r_ankle
            2, # 8 r_toes
            0, # 9
            0, # 10
            0, # 11
            0, # 12
            0, # 13
            0, # 14
            0, # 15
            0, # 16
            0, # 17
            3, # 18 l_hand
            0, # 19
            0, # 20
            0, # 21
            0, # 22
            4] # 23 r_hand

        self.Compute6dGroundContactForces()
        self.seqForceObject = self.object_loader.contact_force_.copy()

        # Transformation from "contact" frame to Parkour joint frame
        parkourR_c = np.matrix(
            [[-1,0,0],[0,0,-1],[0,-1,0]]).astype(float)
        parkourM_c = se3.SE3(parkourR_c, zero(3))

        for i in range(self.nf):
            cPhicGd = self.seqLocalForceGround[:, i]
            cPhicObj = self.seqForceObject[:, i]

            se3.framesForwardKinematics(self.person_model.model,
                                        self.person_model.data,
                                        person_config[:,i])

            for n in range(num_ground_contact_joints):
                j = list_ground_contact_joints[n]
                k = joint_contact_force_mapping[j]-1
                if k==-1: # joint j corresponds to no force in Parkour dataset
                    continue
                # Convert ground contact forces:
                # 1. from local joint frame to the "contact" frame
                f_local = se3.Force(cPhicGd[(6*n):(6*n+6),0])
                if j not in [3,7]:
                    f_c = f_local
                else:
                    cR_local = self.person_model.data.oMi[j+1].rotation
                    cM_local = se3.SE3(cR_local, zero(3))
                    f_c = cM_local.act(f_local)

                # 2. from "contact" frame to Parkour joint frame
                f_parkour = parkourM_c.act(f_c)
                contact_forces_parkour[i,k,:] += \
                    f_parkour.vector.getA().reshape(-1)

            for n in range(num_object_contact_joints):
                j = list_object_contact_joints[n]
                k = joint_contact_force_mapping[j]-1
                if k==-1: # joint j corresponds to no force in Parkour dataset
                    continue
                # Convert object contact forces:
                # 1. from local joint frame to the "contact" frame
                f_local = se3.Force(cPhicObj[(6*n):(6*n+6),0])
                cR_local = self.person_model.data.oMi[j+1].rotation
                cM_local = se3.SE3(cR_local, zero(3))
                f_c = cM_local.act(f_local)

                # 2. from "contact" frame to Parkour joint frame
                f_parkour = parkourM_c.act(f_c)
                contact_forces_parkour[i,k,:] += \
                    f_parkour.vector.getA().reshape(-1)

        return contact_forces_parkour


    def ExpressParkourForcesInLocalJointFrame(self, person_config,
                                              contact_forces_parkour):
        '''
        Transforms the coordinate frame of the input 6D forces from the
        joint frames used by Parkour dataset, to the local joint frames
        defined by Pinocchio. This function is (roughly) the inverse of
        ExpressLocalForcesInParkourFrame. The input variable
        contact_forces_parkour is an array of size
        (num_frames_in_video x num_contact_forces x 6)
        '''

        list_ground_contact_joints = self.contact.list_ground_contact_joints
        list_object_contact_joints = self.contact.list_object_contact_joints

        seqLocalForceGround_GT = np.zeros((
            6*self.contact.num_ground_contact_joints, self.nf))
        seqForceObject_GT = np.zeros((
            6*self.contact.num_object_contact_joints, self.nf))
        seqLocalForceGround_GT = np.matrix(seqLocalForceGround_GT)
        seqForceObject_GT = np.matrix(seqForceObject_GT)

        joints_of_interest = [3, 7, 18, 23] # l_sole, r_sole, l_hand, r_hand

        for i in range(self.nf):
            se3.framesForwardKinematics(self.person_model.model,
                                        self.person_model.data,
                                        person_config[:,i])

            # loop over the joints of interest
            for k in range(len(joints_of_interest)):
                j = joints_of_interest[k]

                # Convert object/ground contact forces
                # 1. from Parkour joint frame to "contact" frame
                f_parkour = se3.Force(
                    np.matrix(contact_forces_parkour[i,k,:]).T)

                parkourR_c = np.matrix(
                    [[-1,0,0],[0,0,-1],[0,-1,0]]).astype(float)
                cM_parkour = se3.SE3(parkourR_c.transpose(), zero(3))
                f_c = cM_parkour.act(f_parkour)

                # 2. from "contact" frame to local joint frame
                cR_local = self.person_model.data.oMi[j+1].rotation
                localM_c = se3.SE3(cR_local.transpose(), zero(3))
                f_local = localM_c.act(f_c)
                if j in [3,7] and j in list_ground_contact_joints:
                    n = list_ground_contact_joints.index(j)
                    seqLocalForceGround_GT[(6*n):(6*n+6), i] = \
                        f_local.vector
                elif j in [18, 23] and j in list_object_contact_joints:
                    n = list_object_contact_joints.index(j)
                    seqForceObject_GT[(6*n):(6*n+6), i] = f_local.vector

        return seqLocalForceGround_GT, seqForceObject_GT


    def DisplayForces(self, q_person_pino,
                            cPhicObj=None,
                            cPhicGd=None,
                            cPhicObj_GT=None,
                            cPhicGd_GT=None,
                            tauMuscle=None):

        person = self.person_model
        se3.framesForwardKinematics(person.model, person.data, q_person_pino)
        list_object_contact_joints = self.contact.list_object_contact_joints
        list_ground_contact_joints = self.contact.list_ground_contact_joints

        color_linfor = [255, 0,0] # red
        color_torque = [0,0, 235] # blue
        color_linfor_gt = [255,255,0]# yellow
        color_torque_gt = [0, 255,0]# green
        if cPhicObj is not None:
            for n in range(len(list_object_contact_joints)):
                j = list_object_contact_joints[n]
                oMc = person.data.oMi[j+1]
                self.PlaceForceArrow(self.names_forceObj[n], oMc,
                                     cPhicObj[(6*n):(6*n+6),0],
                                     color_linfor, color_torque)
        if cPhicGd is not None:
            for n in range(len(list_ground_contact_joints)):
                j = list_ground_contact_joints[n]
                if j in [3, 7]:
                    oMc = person.data.oMi[j+1]
                else:
                    oMc = se3.SE3(eye(3), person.data.oMi[j+1].translation)
                self.PlaceForceArrow(self.names_forceGd[n], oMc,
                                     cPhicGd[(6*n):(6*n+6),0],
                                      color_linfor, color_torque)
        if cPhicObj_GT is not None:
            for n in range(len(list_object_contact_joints)):
                j = list_object_contact_joints[n]
                self.PlaceForceArrow(self.names_forceObj_GT[n],
                                     person.data.oMi[j+1],
                                     cPhicObj_GT[(6*n):(6*n+6),0],
                                     color_linfor_gt, color_torque_gt)
        if cPhicGd_GT is not None:
            for n in range(len(list_ground_contact_joints)):
                j = list_ground_contact_joints[n]
                self.PlaceForceArrow(self.names_forceGd_GT[n],
                                     person.data.oMi[j+1],
                                     cPhicGd_GT[(6*n):(6*n+6),0],
                                     color_linfor_gt, color_torque_gt)

        if tauMuscle is not None:
            for j in range(1,24):
                self.PlaceMuscleTorqueArrow(self.names_muscleTorque[j-1],
                                            person.data.oMi[j+1],
                                            tauMuscle[(6+3*(j-1)):(6+3*(j-1)+3),0])


    def PlaceMuscleTorqueArrow(self, name, oMc, tau):
        color_muscle_torque = [255,255,0] # yellow
        val_tau = max(LA.norm(tau), 1e-4)
        dir_tau = tau/val_tau
        dir0 = np.matrix([1.,0.,0.]).T
        oMtau = oMc * se3.SE3(rotation_matrix(dir0, dir_tau), zero(3))
        # Place visuals
        vertices = [[0.,0.,0.], [val_tau/10.,0.,0.]] # normalize torque
        vertices = np.array(vertices).T.astype(np.float32)
        self.viewer[name].set_object(
            g.LineSegments(
                g.PointsGeometry(vertices),
                g.MeshLambertMaterial(color=rgb_to_hex(color_muscle_torque))))
        self.viewer[name].set_transform(oMtau.homogeneous.getA())


    def PlaceForceArrow(self, name, oMc, cPhic, color_linfor, color_torque):
        name_linfor = name+'f'
        name_torque = name+'t'
        linfor = cPhic[:3,0]
        torque = cPhic[3:,0]
        val_linfor = max(LA.norm(linfor), 1e-4)
        val_torque = max(LA.norm(torque), 1e-4)
        dir_linfor = linfor/val_linfor
        dir_torque = torque/val_torque
        dir0 = np.matrix([1.,0.,0.]).T
        oMlinfor = oMc * se3.SE3(rotation_matrix(dir0, dir_linfor),
                                 zero(3))
        oMtorque = oMc * se3.SE3(rotation_matrix(dir0, dir_torque), zero(3))

        # Place linear force
        vertices = [[0.,0.,0.], [val_linfor/728.22,0.,0.]] # normalize linfor
        vertices = np.array(vertices).T.astype(np.float32)
        self.viewer[name_linfor].set_object(
            g.LineSegments(
                g.PointsGeometry(vertices),
                g.MeshBasicMaterial(color=rgb_to_hex(color_linfor))))
        self.viewer[name_linfor].set_transform(oMlinfor.homogeneous.getA())

        # Place torque
        vertices = [[0.,0.,0.], [val_torque/728.22,0.,0.]] # normalize torque
        vertices = np.array(vertices).T.astype(np.float32)
        self.viewer[name_torque].set_object(
            g.LineSegments(
                g.PointsGeometry(vertices),
                g.MeshBasicMaterial(color=rgb_to_hex(color_torque))))
        self.viewer[name_torque].set_transform(oMtorque.homogeneous.getA())


    def SaveDecisionVariables(self,
                              save_path,
                              frame_start=None,
                              frame_end=None,
                              save_object_endpoints=False,
                              save_forces=False,
                              solver_options=None,
                              stage_options=None,
                              stage_weights=None):
        '''
        Save the decision variables that have been optimized.
        '''
        params = {}
        # Basic info
        params["fps"] = 1./self.dt
        params["focal_length"] = self.camera.focal_length_

        # Person: save config
        params["config_person"] = self.person_loader.config_.copy()

        # Object: save config, config_contact and config_keypoints
        params["config_object"] = self.object_loader.config_.copy()
        params["config_object_contact"] = self.object_loader.config_contact_.copy()
        params["config_object_keypoints"] = self.object_loader.config_keypoints_.copy()

        # Ground: save config and config_contact
        params["config_ground"] = self.ground_loader.config_.copy()
        params["config_ground_contact"] = self.ground_loader.config_contact_.copy()

        # Other useful info
        params["joint_3d_positions_person"] = self.person_loader.joint_3d_positions_.copy()
        params["keypoint_3d_positions_object"] = self.object_loader.keypoint_3d_positions_.copy()
        params["contact_mapping"] = self.contact.contact_mapping.copy()
        params["contact_states"] = self.contact.contact_states.copy()

        # Optionally, save 2D measurements
        if save_object_endpoints:
            params["keypoint_2d_reprojected"] = self.object_loader.keypoint_2d_reprojected_.copy()
            params["endpoint_2d_positions"] = self.object_loader.endpoint_2d_positions_.copy()
        if save_forces:
            params["ground_friction"] = self.person_loader.ground_friction_.copy()
            params["object_contact_forces"] = self.object_loader.contact_force_.copy()

        # save optimization parameteres
        if solver_options is not None:
            params["solver_options"] = solver_options
        if stage_options is not None:
            params["stage_options"] = stage_options
        if stage_weights is not None:
            params["stage_weights"] = stage_weights
        if frame_start is not None:
            params["frame_start"] = frame_start
        if frame_end is not None:
            params["frame_end"] = frame_end
        # write data to file
        if not exists(dirname(save_path)):
            makedirs(dirname(save_path))
        with open(save_path, 'w') as f:
            pickle.dump(params, f)
            print('Decision variables saved to {}'.format(save_path))

    def RestoreDecisionVariables(self, pkl_path, virtual_object):
        '''
        Load the decision varialbes from file and recover the dataloader's state.
        '''
        with open(pkl_path, 'r') as f:
            # load data
            params = pickle.load(f)
            fps = params["fps"]
            focal_length = params["focal_length"]
            config_person = params["config_person"]
            config_ground = params["config_ground"]
            config_ground_contact = params["config_ground_contact"]

            # camera: focal_length
            self.camera.focal_length_ = focal_length
            self.camera.UpdateProjectionMatrix()

            # person: config, ground contact forces
            # # DEBUGGING:
            # q_person_temp = config_person[:,26].copy()
            # for i in range(22,30):
            #     config_person[:, i] = q_person_temp.copy()

            self.person_loader.LoadConfig(config_person, fps)
            self.person_loader.UpdateKeypoint2dReprojected(self.camera)
            self.person_loader.UpdateJoint2dReprojected(self.camera)

            # ground: config, config_contact
            self.ground_loader.LoadConfig(config_ground, fps)
            self.ground_loader.LoadConfigContact(config_ground_contact)

            # object: config, config_contact, config_keypoints
            if not virtual_object:
                config_object = params["config_object"]
                config_object_contact = params["config_object_contact"]
                config_object_keypoints = params["config_object_keypoints"]
                self.object_loader.LoadConfig(config_object, fps)
                self.object_loader.LoadConfigContact(config_object_contact)
                self.object_loader.LoadConfigKeypoints(config_object_keypoints)
                self.object_loader.UpdateKeypoint2dReprojected(self.camera)

            #
            if "ground_friction" in params.keys():
                self.person_loader.ground_friction_ = params["ground_friction"]
            if "object_contact_forces" in params.keys():
                self.object_loader.contact_force_ = params["object_contact_forces"]
            print('Decision variables restored from {}'.format(pkl_path))

    #######################################
    def GuessDepth(self, focal_length):
        '''
        initializes the depth of human body via similar triangles using the diameter of human head
        '''
        # head size in 3D: approximate head diameter using 3x neck length
        head_size_3d = 3*LA.norm(self.person_model.smpl_joints_neutral_pose[15] - self.person_model.smpl_joints_neutral_pose[12]) # scalar
        # head size in 2D
        joint_2d_positions = self.person_loader.joint_2d_positions_ # 42 x nf matrix
        neck_id = 12
        neck_positions = joint_2d_positions[(3*neck_id):(3*neck_id+2),:] # 2 x nf matrix
        head_top_id = 13
        head_top_positions = joint_2d_positions[(3*head_top_id):(3*head_top_id+2),:] # 2 x nf matrix
        head_size_2d = LA.norm(head_top_positions-neck_positions, axis=0) # 1D array of size nf
        # recover the 3D depth via similar triangles
        depth_guess = focal_length * np.tile(head_size_3d, self.nf) / head_size_2d # 1D array of size nf
        # copy the depth into person_loader.config_
        # note that the python binding does not support block-wise assignment
        config_temp = self.person_loader.config_.copy() # convert the Eigen::MatrixXd object to a numpy matrix
        config_temp[2,:] = depth_guess # numpy matrix supports block-wise assignment
        self.person_loader.config_ = config_temp # copy back

    def PreprocessDataToVisualize(self, remove_depth=False, virtual_object=False):
        '''
        preprocess data to visualize. This function should be called before the kinematic model is rendered.
        '''
        # move away the ground contact points if there is no contact
        contact_state = self.contact.contact_states
        contact_mapping = self.contact.contact_mapping
        joints_having_ground_contact = self.contact.joints_having_ground_contact
        ground_config_contact = self.ground_loader.config_contact_.copy()
        object_config_contact = self.object_loader.config_contact_.copy()
        for i in range(self.nf):
            # search the contact points that are not used at frame i
            keypoint_ids = []
            for j in joints_having_ground_contact:
                if contact_state[j, i] != 2:
                    if j in [3, 7]: # left/right ankle correspond to 4 contact points each
                        for k in range(4):
                            keypoint_ids.append(contact_mapping[j]+k)
                    else: # the other joints correspond to 1 contact point each
                        keypoint_ids.append(contact_mapping[j])
            # move away the unused keypoints
            for k in keypoint_ids:
                ground_config_contact[2*(k-1), i] = 100. # minus 1 because k begin from 1
        self.ground_config_contact = ground_config_contact
        self.object_config_contact = object_config_contact
        self.object_config_keypoints = self.object_loader.config_keypoints_.copy()

        # we put the subject around the world origin for a better visualization
        # this is done by subtracting the person and object/ground's z-coordinate by their mean depth
        self.person_config_translated = self.person_loader.config_pino_.copy()
        self.person_config_init_translated = self.person_loader.config_pino_init_.copy()
        self.ground_config_translated = self.ground_loader.config_pino_.copy()
        self.object_config_translated = self.object_loader.config_pino_.copy()
        if remove_depth:
            depth_offset = np.mean(self.person_loader.config_[2,:])
            self.person_config_translated[2,:] -= depth_offset
            self.person_config_init_translated[2,:] -= depth_offset
            self.ground_config_translated[2,:] -= depth_offset
            if not virtual_object:
                self.object_config_translated[2,:] -= depth_offset

    def ClearScene(self):
        self.DeleteForceVisuals()
        self.DeleteGroundTruthForceVisuals()
        self.person_model.deleteVisuals()
        self.viewer['/Axes'].set_property("visible", False)
        #self.viewer.ShowXYZAxis(False)
        self.object_model.DeleteVisuals()
        self.ground_model.DeleteVisuals()



    def PlaySequence(self,
                     frames=None,
                     show_person=True,
                     show_object=False,
                     show_object_2d=False,
                     show_ground=False,
                     show_person_baseline=False,
                     show_object_baseline=False,
                     show_openpose_joints=True,
                     show_reprojected_joints=False,
                     show_reprojected_keypoints=True,
                     show_force=False,
                     show_force_gt=False,
                     remove_depth=False,
                     virtual_object=False,
                     camera_angle=0.,
                     screenshot_folder=None,
                     pause=False):
        '''
        Play person, object & ground pose using Gepetto-viewer.
        '''
        # quick check
        if not hasattr(self, 'person_loader'):
            raise ValueError("Optimizer.PlaySequence: person data is missing "+ \
                             "(data member person_loader of type DataloaderPerson).")
        if (show_object or show_object_baseline) and not hasattr(self, 'object_loader'):
            raise ValueError("Optimizer.PlaySequence: object data is missing "+ \
                             "(data member object_loader of type DataloaderObject).")
        if show_ground and not hasattr(self, 'ground_loader'):
            raise ValueError("Optimizer.PlaySequence: ground data is missing "+ \
                             "(data member ground_loader of type DataloaderObject).")
        # options
        rescale_2d = 1/300. # the original pixel positions are to large to render in the viewer

        # gather data to play
        frames = range(self.nf) if frames is None else frames
        person_loader = self.person_loader
        object_loader = self.object_loader if hasattr(self, 'object_loader') else None
        ground_loader = self.ground_loader if hasattr(self, 'ground_loader') else None

        # compute offset on the depth
        self.PreprocessDataToVisualize(remove_depth, virtual_object)

        # create another person model with visuals if necessary
        if show_person and show_person_baseline:
            person_baseline = Person("person_baseline",
                            self.person_model.smpl_joints_neutral_pose,
                            self.person_model.openpose_keypoints_neutral_pose,
                            inertia_path=self.person_model.inertia_path,
                            viewer=self.viewer,
                            opacity=0.2)
        # create another tool model with visuals if necessary
        if show_object:# and show_object_baseline:
            handle_length = self.object_loader.config_keypoints_[0,0]
            self.object_model.UpdateHandleVisuals(handle_length)

        if show_openpose_joints:
            names_openpose_joints = self.person_model.keypoint_names
            colors_openpose_joints = self.person_model.keypoint_colors
            visuals_openpose_joints = PointCloud('openpose_joints',
                                        self.viewer,
                                        names=names_openpose_joints,
                                        colors=colors_openpose_joints,
                                        opacity=0.2,
                                        size=0.02)

        if show_reprojected_keypoints:
            person_loader.UpdateKeypoint2dReprojected(self.camera)
            names_openpose_joints = self.person_model.keypoint_names
            colors_openpose_joints = self.person_model.keypoint_colors
            visuals_reprojected_keypoints = \
                PointCloud('reprojected_keypoints',
                           self.viewer,
                           names=names_openpose_joints,
                           colors=colors_openpose_joints,
                           opacity=1.,
                           size=0.02)

        if show_reprojected_joints:
            # | limb name | limb color |
            # | --------- | ---------- |
            # | pelvis    | red        |
            # | left leg  | pink       |
            # | right leg | cyan       |
            # | spine     | tomato     |
            # | left arm  | yellow     |
            # | right arm | green      |
            person_loader.UpdateJoint2dReprojected(self.camera)
            names_reprojected_joints = \
                ['pelvis',
                 'l_hip','l_knee','l_ankle','l_toes',
                 'r_hip','r_knee','r_ankle','r_toes',
                 'spine_0','spine_1','spine_2','spine_3', 'spine_4',
                 'l_scapula','l_shoulder','l_elbow','l_wrist','l_fingers',
                 'r_scapula','r_shoulder','r_elbow','r_wrist','r_fingers']
            colors_reprojected_joints = \
                [[255,0,0],
                 [255,0,255],[255,0,255],[255,0,255],[255,0,255],
                 [0,255,255],[0,255,255],[0,255,255],[0,255,255],
                 [255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],
                 [255,215,0],[255,215,0],[255,215,0],[255,215,0],[255,215,0],
                 [127,255,0],[127,255,0],[127,255,0],[127,255,0],[127,255,0]]
            visuals_reprojected_joints = \
                PointCloud('reprojected_joints',
                           self.viewer,
                           names=names_reprojected_joints,
                           colors=colors_reprojected_joints,
                           opacity=1.,
                           size=0.02)

        if show_object_2d:
            print("Object colors: handle end is Red; head is Green!")
            names_object_endpoints = ['handle_tip', 'tool_head']
            colors_object_endpoints = [[255,0,0], [0,255,0]]
            visuals_reprojected_object_endpoints = \
                PointCloud('reprojected_object_endpoints',
                           self.viewer,
                           names=names_object_endpoints,
                           colors=colors_object_endpoints,
                           opacity=1.,
                           size=0.03)
            visuals_object_endpoints = \
                PointCloud('object_endpoints',
                           self.viewer,
                           names=names_object_endpoints,
                           colors=colors_object_endpoints,
                           opacity=.5,
                           size=0.05)

        if show_force:
            self.Compute6dGroundContactForces()
            self.seqForceObject = self.object_loader.contact_force_.copy()

        if not hasattr(self, "seqForceObject_GT") or not hasattr(self, "seqLocalForceGround_GT"):
            show_force_gt = False

        if screenshot_folder is not None:
            cache_folder = join(dirname(screenshot_folder),
                                "screenshot_cache")
            recorder = VideoRecorder(self.viewer, cache_folder,
                                     screenshot_folder)

        # Change camera view
        set_meshcat_camera_view(self.viewer, camera_angle)
        self.viewer['/Axes'].set_property("visible", False)

        # Visualize frame by frame
        for i in frames:

            if show_person:
                self.person_model.display(self.person_config_translated[:,i])
                if show_person_baseline:
                    person_baseline.display(self.person_config_init_translated[:,i])
            if show_openpose_joints:
                visuals_openpose_joints.Display2d(person_loader.joint_2d_positions_[:,i]*rescale_2d)
            if show_reprojected_keypoints:
                visuals_reprojected_keypoints.Display2d(person_loader.keypoint_2d_reprojected_[:,i]*rescale_2d)
            if show_reprojected_joints:
                visuals_reprojected_joints.Display2d(person_loader.joint_2d_reprojected_[:,i]*rescale_2d)
            if show_object_2d and not virtual_object:
                # default order in endpoint_2d_positions_:
                # first 3 entries correspond to handle_tip
                # last 3 entries correspond to tool_head
                visuals_object_endpoints.Display2d(object_loader.endpoint_2d_positions_[:,i]*rescale_2d)
                visuals_reprojected_object_endpoints.Display2d(object_loader.keypoint_2d_reprojected_[:,i]*rescale_2d)
            #elif show_person_baseline:
                #self.person_model.display(person_loader.config_baseline_pino_[:,i])
            # visualize tool CAD model
            if show_object:
                # Update object visuals
                q_object_translated = self.object_config_translated[:,i]
                # display tool only if there is hand contact
                if 1 not in self.contact.contact_states[:, i].tolist():
                    q_object_translated[0, 0] = 1000.
                self.object_model.Display(q_object_translated,
                                        self.object_config_contact[:, i],
                                        q_endpt=self.object_config_keypoints[:,i],
                                        update_data=True)

            # visualize ground plane
            if show_ground:
                self.ground_model.Display(self.ground_config_translated[:, i],
                                    self.ground_config_contact[:, i],
                                    q_endpt=None,
                                    update_data=True)
            else:
                self.ground_model.DeleteVisuals()

            if show_force:
                #self.DeleteForceVisuals()
                if show_force_gt:
                    #self.DeleteGroundTruthForceVisuals()
                    self.DisplayForces(self.person_config_translated[:, i],
                                       cPhicObj=self.seqForceObject[:, i],
                                       cPhicGd=self.seqLocalForceGround[:, i],
                                       cPhicObj_GT=self.seqForceObject_GT[:, i],
                                       cPhicGd_GT=self.seqLocalForceGround_GT[:, i],
                                       tauMuscle=None)
                else:
                    self.DisplayForces(self.person_config_translated[:, i],
                                       cPhicObj=self.seqForceObject[:, i],
                                       cPhicGd=self.seqLocalForceGround[:, i])

            if screenshot_folder is not None:
                print("#{}. Saving screenshot to {}".format(i, screenshot_folder))
                save_name = "%06d" % i
                recorder.CaptureScreen(save_name,
                                       extension='jpg',
                                       delay=0.1)
            elif pause:
                raw_input('#{}. Press any key to continue...'.format(i))
            else:
                time.sleep(self.dt)


        # remove shadow if exists
        if show_person and show_person_baseline:
            person_baseline.deleteVisuals()
        # remove 2d joints if exists
        if show_openpose_joints:
            visuals_openpose_joints.DeleteVisuals()
        if show_reprojected_keypoints:
            visuals_reprojected_keypoints.DeleteVisuals()
        if show_reprojected_joints:
            visuals_reprojected_joints.DeleteVisuals()
        if show_object_2d:
            visuals_object_endpoints.DeleteVisuals()
            visuals_reprojected_object_endpoints.DeleteVisuals()
        self.viewer['/Axes'].set_property("visible", True)
        #self.viewer.ShowXYZAxis(True)

