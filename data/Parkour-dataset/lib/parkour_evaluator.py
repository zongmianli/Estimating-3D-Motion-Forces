import logging
import argparse
import time
import numpy as np
import numpy.linalg as LA
import pickle as pk
from os import makedirs, remove
from os.path import join, exists, abspath, dirname, basename, isfile

from helpers.maths import rotation_matrix, procrustes


# List of human joints considered in Parkour dataset
# --------------------
# joint name        id
# --------------------
# left hip           0
# left knee          1
# left ankle         2
# left toes          3
# right hip          4
# right knee         5
# right ankle        6
# right toes         7
# left shoulder      8
# left elbow         9
# left wrist        10
# left fingers      11
# right shoulder    12
# right elbow       13
# right wrist       14
# right fingers     15
# --------------------

def compute_mean_force_errors(force_errors, mask_uncaptured_forces):

    num_frames = force_errors.shape[0]
    num_force_types = force_errors.shape[1]
    if num_force_types!=mask_uncaptured_forces.shape[1] or \
       num_frames!=mask_uncaptured_forces.shape[0]:
        raise ValueError("Should never happen!")
    mean_force_errors = np.zeros(num_force_types) # 4
    for k in range(num_force_types):
        num_captured_forces = \
            num_frames - np.sum(mask_uncaptured_forces[:,k], axis=0)
        mean_force_errors[k] = \
            np.sum(force_errors[:,k], axis=0)/num_captured_forces

    return mean_force_errors

class ParkourEvaluator(object):

    def __init__(self, gt_dir, video_name):
        self.video_name = video_name
        self.action = video_name.split('_')[0]
        self.path_to_contact_forces_local = join(
            gt_dir, "contact_forces_local.pkl")
        self.path_to_joint_3d_positions = join(
            gt_dir, "joint_3d_positions.pkl")
        self.LoadGroundTruthData()
        self.mpjpe = {}
        self.joint_errors = {}
        self.mean_joint_errors = {}
        self.joint_3d_positions_pred_aligned = {}


    def LoadGroundTruthData(self):
        '''
        Load ground truth 3D human motion and contact forces
        '''
        self.joint_3d_positions = None
        with open(self.path_to_joint_3d_positions, 'rb') as f:
            data = pk.load(f, encoding='latin-1')[self.video_name]
            self.joint_3d_positions = data["joint_3d_positions"]
            self.joint_names = data["joint_names"]
            self.joint_marker_mapping = data["joint_marker_mapping"]
            self.fps = data["fps"]
            self.num_joints = len(self.joint_names)
            self.num_frames_in_video = self.joint_3d_positions.shape[0]
            print('  Ground truth joint positions loaded from {}'.format(
                self.path_to_joint_3d_positions))

        with open(self.path_to_contact_forces_local, 'rb') as f:
            data = pk.load(f, encoding='latin-1')[self.video_name]
            self.contact_forces_local = data["contact_forces_local"]
            self.contact_forces_names = data["contact_forces_names"]
            self.num_contact_forces = len(self.contact_forces_names)
            self.mask_uncaptured_forces = data["mask_uncaptured_forces"]
            if self.fps != data["fps"]:
                raise ValueError("Should never happen!")
            print('  Ground truth contact forces loaded from {}'.format(
                self.path_to_contact_forces_local))


    def Evaluate3DPoses(self, joint_3d_positions_pred):
        '''
        Given an array joint_3d_positions_pred of size
        (num_frames_in_video x num_joints x 3), this function aligns
        the predicted joint positions with the ground truth by solving
        an orthogonal Procrustes problem.
        '''
        num_frames = self.num_frames_in_video-3
        joint_errors = np.zeros((num_frames, self.num_joints))
        joint_3d_positions_pred_aligned = np.zeros(
            joint_3d_positions_pred.shape)

        for i in range(num_frames):
            skeleton_pred = joint_3d_positions_pred[i]*1000. # meter -> mm
            skeleton_ground_truth = \
                self.joint_3d_positions[i]*1000. # meter -> mm
            R, t, s, skeleton_pred_aligned = procrustes(
                skeleton_pred, skeleton_ground_truth)
            joint_3d_positions_pred_aligned[i] = skeleton_pred_aligned
            joint_errors[i] = LA.norm(
                joint_3d_positions_pred_aligned[i] - skeleton_ground_truth,
                axis=1)

        self.joint_3d_positions_pred_aligned["procrustes"] = \
            joint_3d_positions_pred_aligned
        self.joint_errors["procrustes"] = joint_errors
        self.mean_joint_errors["procrustes"] = np.mean(joint_errors, axis=0)
        self.mpjpe["procrustes"] = np.mean(
            self.mean_joint_errors["procrustes"])


    def EvaluateContactForces(self, contact_forces_pred):
        '''
        Given an array contact_forces_pred of size
        (num_frames_in_video x num_contact_forces x 6),
        this function computes the error of the linear and the moment
        component between the input forces and the ground truth.
        '''
        num_frames = self.num_frames_in_video-3
        linear_force_errors = np.zeros((num_frames, self.num_contact_forces))
        torque_errors = np.zeros((num_frames, self.num_contact_forces))
        for i in range(num_frames):
            linear_forces_pred = contact_forces_pred[i][:,:3]
            torques_pred = contact_forces_pred[i][:,3:]
            linear_forces_ground_truth = \
                self.contact_forces_local[i][:,:3] # 4x3 array
            torques_ground_truth = \
                self.contact_forces_local[i][:,3:] # 4x3 array
            mask = 1 - np.tile(
                self.mask_uncaptured_forces[i].astype(float),
                (3, 1)).T # 4x3 indicator array of captured forces

            linear_force_errors[i] = LA.norm(
                mask*(linear_forces_pred - linear_forces_ground_truth),
                axis=1)
            torque_errors[i] = LA.norm(
                mask*(torques_pred - torques_ground_truth), axis=1)

        self.linear_force_errors = linear_force_errors
        self.torque_errors = torque_errors
        self.mean_linear_force_errors = compute_mean_force_errors(
            linear_force_errors, self.mask_uncaptured_forces[:num_frames,:])
        self.mean_torque_errors = compute_mean_force_errors(
            torque_errors, self.mask_uncaptured_forces[:num_frames,:])


    def SaveResults(self, save_path, save_forces=True):

        results = {"action": self.action,
                   "joint_errors": self.joint_errors,
                   "mean_joint_errors": self.mean_joint_errors,
                   "mpjpe": self.mpjpe}
        if save_forces:
            results["linear_force_errors"] = self.linear_force_errors
            results["torque_errors"] = self.torque_errors
            results["mask_uncaptured_forces"] = self.mask_uncaptured_forces
            results["mean_linear_force_errors"] = \
                self.mean_linear_force_errors
            results["mean_torque_errors"] = self.mean_torque_errors
        if exists(save_path):
            with open(save_path, 'rb') as f:
                data = pk.load(f, encoding='latin-1')
                data[self.video_name] = results
        else:
            data = {self.video_name: results}
            if not exists(dirname(save_path)):
                makedirs(dirname(save_path))
        with open(save_path, 'wb') as f:
            pk.dump(data, f)
            print('  Evaluation results saved to {}'.format(save_path))



