import numpy as np
import numpy.linalg as LA
from glob import glob
from os import makedirs, remove
from os.path import join, exists, abspath, dirname, basename, isfile

import pinocchio as pin
from pinocchio.utils import *


def load_object_2d_endpoints(
        path_object_2d_endpoints,
        frame_start,
        frame_end,
        path_scores=None):
    '''
    Load object 2D endpoints from the user-provided path.
    This function return a (6, nf) matrix, i.e. (x, y, confidence) * 2 endpoints = 6
    '''
    # npts: number of keypoints in the txt file. By default we consider 2 keypoints at each video frame: the first point as the head of the object and the second as the end of the bar.

    npts = 2
    with open(path_object_2d_endpoints, 'r') as f:
        data = np.loadtxt(f)
        if data.shape[0] > 0:
            fids = data[::npts,0].astype(int)
            endPts = np.zeros((fids.shape[0], 0)).astype(int)
            # In the raw object output, the 1st endpoint is red, the 2nd is green
            for n in range(npts):
                endPts = np.concatenate((endPts, data[n::npts,1:]), axis=1)
        else:
            fids = []
            endPts = np.array([])


    # Load scores
    if path_scores is not None:
        with open(path_scores, 'r') as f:
            data = np.loadtxt(f)
            # fids_check = data_scores[:,0].astype(int)
            scores = data[:,1]
            nf = scores.shape[0]

    # parse the endPts and scores to a (6, nf) matrix
    endpoint_2d_positions = np.zeros((nf, 6))
    for i in range(len(fids)):
        fid = fids[i]
        endpoints = endPts[i]
        # get the correct order of the endpoints
        handle_end = 2
        head = 0
        # our Pinocchio object model assume index 0 as the end of handle and 1 as the object's head
        endpoint_2d_positions[fid,:2] = endpoints[handle_end:(handle_end+2)]
        endpoint_2d_positions[fid,3:5] = endpoints[head:(head+2)]

        if path_scores is not None:
            score = scores[fid]
        else:
            score = 1.

        endpoint_2d_positions[fid,2] = score
        endpoint_2d_positions[fid,5] = score

    endpoint_2d_positions = np.matrix(endpoint_2d_positions).T
    endpoint_2d_positions = endpoint_2d_positions[:, (frame_start-1):frame_end]  # verbose

    return endpoint_2d_positions

def compute_object_config_initial_guesses(
        object_model,
        object_data,
        object_loader,
        person_loader,
        object_init_rotation_candidates=None):
    '''
    Generate a set of initial guesses for the object configurations
    '''
    person_joint_3d_positions = person_loader.joint_3d_positions_
    config_object_initial_guesses = []

    nf = person_joint_3d_positions.shape[1]
    object_init_positions = (person_joint_3d_positions[18*3:(18*3+3),:] + person_joint_3d_positions[23*3:(23*3+3),:])/2.# initialize object translation with hand positions

    if object_init_rotation_candidates is None:
        object_init_rotation_candidates = [[0., 1e-1, 0.],
                                           [0., np.pi/2., 0.],
                                           [0., np.pi, 0.],
                                           [0., -np.pi/2., 0.]]

    for rot in object_init_rotation_candidates:
        object_init_rotations_tiled = np.tile(np.matrix(rot).T, (1, nf))
        config_object_guess = np.concatenate(
            (object_init_positions, object_init_rotations_tiled), axis=0)

        # Refine config_object_guess for barbell sequences, such that the center
        # of the stick handle (where the contact points are initialized), lies
        # in the center of the person's hand positions for each frame
        if object_loader.name_ == "barbell":
            object_loader.LoadConfig(config_object_guess, object_loader.fps_)
            nq_pino = object_loader.nq_pino_
            nq_contact = object_loader.nq_contact_
            nq_keypoints = object_loader.nq_keypoints_
            nq_stacked = nq_pino + nq_keypoints + nq_contact
            config_stacked = np.matrix(np.zeros((nq_stacked, nf)))
            config_stacked[:nq_pino,:] = object_loader.config_pino_.copy()
            config_stacked[nq_pino:(nq_pino+nq_contact),:] = object_loader.config_contact_.copy()

            num_contacts = object_loader.num_contacts_
            for i in range(nf):
                # Compute the centroid position of the object contact points
                pin.forwardKinematics(object_model, object_data, config_stacked[:,i])
                centroid_object_contacts = zero(3)
                for k in range(num_contacts):
                    centroid_object_contacts += object_data.oMi[object_loader.njoints_+1+k].translation
                centroid_object_contacts /= num_contacts
                # Offset the object translation by the difference between the centroid
                # and the center of the person's hands
                offset = config_object_guess[:3,i] - centroid_object_contacts
                config_object_guess[:3,i] += offset

        config_object_initial_guesses.append(config_object_guess)

    return config_object_initial_guesses


def LoadObject2dEndpoints(video_name, npts=2):
    '''
    load object 2D endpoints and confidence values
    return a (6, nf) matrix, i.e. (x, y, confidence) * 2 endpoints = 6
    --
    npts: number of keypoints in the txt file. By default we consider 2 keypoints at each video frame: the first point as the head of the object and the second as the end of the bar.
    '''
    if 'hammer' in video_name:
        endpts_path = join('object_detect', 'endpoints_corrected', video_name+'_endpoints_corrected.txt')
    else:
        endpts_path = join('object_detect', 'endpoints', video_name+'_endpoints.txt')
    scores_path = join('object_detect', 'scores', video_name+'_scores.txt')
    # load endpoints
    with open(endpts_path, 'r') as f:
        data_endpts = np.loadtxt(f)
        if data_endpts.shape[0] > 0:
            fids = data_endpts[::npts,0].astype(int)
            endPts = np.zeros((fids.shape[0], 0)).astype(int)
            for n in range(npts):
                endPts = np.concatenate((endPts, data_endpts[n::npts,1:]), axis=1)
        else:
            fids = []
            endPts = np.array([])
    # load confidence values
    with open(scores_path, 'r') as f:
        data_scores = np.loadtxt(f)
        # fids_check = data_scores[:,0].astype(int)
        scores = data_scores[:,1]
        nf = scores.shape[0]

    # correct the order of object endpoints.
    # In object raw output, the 1st endpoint is red, the 2nd is green
    # barbell: 0 - right point, 1 - left point
    # scythe:  0 - handle end, 1 - head
    # spade: 0 - head, 1 - handle end
    # hammer: it dependes on the video
    endpoint_orders = {
        "barbell_0002": (0, 2),
        "barbell_0003": (0, 2),
        "barbell_0007": (0, 2),
        "barbell_0008": (0, 2),
        "barbell_0010": (0, 2),
        "hammer_0001": (0, 2),
        "hammer_0003": (0, 2),
        "hammer_0006": (2, 0),
        "hammer_0007": (2, 0),
        "hammer_0010": (0, 2),
        "scythe_0001": (0, 2),
        "scythe_0002": (0, 2),
        "scythe_0003": (0, 2),
        "scythe_0005": (0, 2),
        "scythe_0006": (0, 2),
        "spade_0001": (2,0),
        "spade_0002": (2,0),
        "spade_0003": (2,0),
        "spade_0007": (2,0),
        "spade_0008": (2,0)
    }

    # parse the endPts and scores to a (6, nf) matrix
    endpoint_2d_positions = np.zeros((nf, 6))
    for i in range(len(fids)):
        fid = fids[i]
        endpoints = endPts[i]
        # get the correct order of the endpoints
        handle_end = endpoint_orders[video_name][0]
        head = endpoint_orders[video_name][1]
        # our Pinocchio object model assume index 0 as the end of handle and 1 as the object's head
        endpoint_2d_positions[fid,:2] = endpoints[handle_end:(handle_end+2)]
        endpoint_2d_positions[fid,2] = scores[fid]
        endpoint_2d_positions[fid,3:5] = endpoints[head:(head+2)]
        endpoint_2d_positions[fid,5] = scores[fid]

    return np.matrix(endpoint_2d_positions).T


def estimate_object_handle_length(
        tool_name,
        joint_2d_positions,
        openpose_keypoints_neutral_pose,
        endpoint_2d_positions,
        ratio=.8):
    '''
    Estimate the object handle's 3D length accoring to its relative 2D length w.r.t. the person's torso size estimated by Openpose.
    The main steps include:
    1. Compute the person's average torso length (in 2D).
    2. Compute the average 2D length of the object.
    3. Compute the relative length of the object handle w.r.t. the person's torso in 2D.
    4. The 3D object length is obtained from the relative length and the person's 3D torso length (output by HMR).
    If there's no object 2D endpoint is detected, we use fixed scales for different types of objects
    '''

    # Estimate torso 2D length
    # Compute 2D lengths of (detected) torso links frame by frame,
    # and stack the results to the list torso_links_2d_len. The estimated
    # torso 2D length is finally obtained by averaging torso_links_2d_len.
    torso_link_ids = [[5,8], # l_shoulder, r_hip
                      [2,11]] # r_shoulder, l_hip

    joint_2d_positions = joint_2d_positions.T.getA().reshape((-1,18,3))
    nf = joint_2d_positions.shape[0]

    torso_links_2d_lengths = []
    for i in range(nf):
        # Iterate over torso links
        for l,joint_ids in enumerate(torso_link_ids):
            # For link #l, make sure the joints are detected
            link_exists = True
            for j in joint_ids:
                confidence = joint_2d_positions[i,j,-1]
                if confidence <= .05:
                    link_exists = False
                    break

            if link_exists:
                joint_1 = joint_2d_positions[i,joint_ids[0],:2]
                joint_2 = joint_2d_positions[i,joint_ids[1],:2]
                torso_links_2d_lengths.append(LA.norm(joint_1-joint_2))

    # Sort per-frame torso lengths in descending order
    torso_links_2d_lengths = sorted(torso_links_2d_lengths, reverse=True)
    # Compute average 2D torso length using a propotion (determined by ratio)
    # of the per-frame results.
    nf_detected = len(torso_links_2d_lengths)
    nf_effective = int(ratio*nf_detected)
    nf_start = (nf_detected-nf_effective)/2
    nf_end = min(nf_start+nf_effective, nf_detected)
    torso_2d_len = np.mean(torso_links_2d_lengths[nf_start:nf_end])

    # Compute torso 3D length
    torso_links_3d_len = [None] * 2
    for l,joint_ids in enumerate(torso_link_ids):
        joint_1 = openpose_keypoints_neutral_pose[joint_ids[0],:3]
        joint_2 = openpose_keypoints_neutral_pose[joint_ids[1],:3]
        torso_links_3d_len[l] = LA.norm(joint_1-joint_2)

    torso_3d_len = np.mean(np.array(torso_links_3d_len))

    # Estimate object handle 2D length using
    endpoint_2d_positions = endpoint_2d_positions.T.getA().reshape((-1,2,3))
    if not endpoint_2d_positions.shape[0]==nf:
        print("Check failed: endpoint_2d_positions.shape[0]==nf ({0:d} vs {1:d})".format(endpoint_2d_positions.shape[0], nf))

    handle_2d_lengths = []
    for i in range(nf):
        # Compute handle lengths for frames with detected endpoints
        object_detected = True
        for k in range(2):
            confidence = endpoint_2d_positions[i,k,-1]
            if confidence <= .05:
                object_detected = False
                break

        if object_detected:
            handle_end_pos = endpoint_2d_positions[i,0,:2]
            tool_head_pos = endpoint_2d_positions[i,1,:2]
            handle_2d_lengths.append(LA.norm(handle_end_pos-tool_head_pos))

    nf_detected = len(handle_2d_lengths)
    if nf_detected > 0:
        # Sort per-frame handle lengths in descending order
        handle_2d_lengths = sorted(handle_2d_lengths, reverse=True)
        # Compute average 2D handle length using a propotion (determined by ratio)
        # of the per-frame results.
        nf_effective = int(ratio*nf_detected)
        nf_start = (nf_detected-nf_effective)/2
        nf_end = min(nf_start+nf_effective, nf_detected)
        handle_2d_len = np.mean(handle_2d_lengths[nf_start:nf_end])
        scale = handle_2d_len/torso_2d_len
    else:
        # Maunally set scale if no object is detected at any time
        if tool_name == "hammer":
            scale = 1.
        elif tool_name == "scythe":
            scale = 1.
        elif tool_name == "spade":
            scale = 1.
        elif tool_name == "barbell":
            scale = 3.7 # 4.6, 4.65, 2.63, 2.93
        else:
            raise ValueError("Unknown tool name: {0:s}!".format(tool_name))
        print("(object_pose.py) No object is detected, use default scale for estimating object 3D length ...")
        handle_2d_len = 0.

    # Compute 3D handle length using similar triangles
    handle_3d_len = torso_3d_len*scale

    # Print info
    print("(object_pose.py) Estimated {0:s} length:".format(tool_name))
    print("  torso_2d_len: {0:.2f} (pixels)".format(torso_2d_len))
    print("  handle_2d_len: {0:.2f} (pixels)".format(handle_2d_len))
    print("  torso_3d_len: {0:.2f} (m)".format(torso_3d_len))
    print("  Output handle_3d_len: {0:.2f} (m) (with scale {1:.2f})".format(handle_3d_len, scale))

    return handle_3d_len
