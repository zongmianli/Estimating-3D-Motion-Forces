import numpy as np
import numpy.linalg as LA
import chumpy as ch
import pickle as pk
from person_models.smpl.smpl_webuser.serialization import load_model


def ComputeJoint3dPositionsInNeutralPose(model_path, betas,
                                         adjust_coco_joints=True):
    '''
    Computes the 3d positions of SMPL and Cocoplus joints
    --------------------------------
    | Joint name     | Cocoplus ID |
    |----------------+-------------|
    | Right ankle    |           0 |
    | Right knee     |           1 |
    | Right hip      |           2 |
    | Left hip       |           3 |
    | Left knee      |           4 |
    | Left ankle     |           5 |
    | Right wrist    |           6 |
    | Right elbow    |           7 |
    | Right shoulder |           8 |
    | Left shoulder  |           9 |
    | Left elbow     |          10 |
    | Left wrist     |          11 |
    | Neck           |          12 |
    | Head top       |          13 |
    | nose           |          14 |
    | left_eye       |          15 |
    | right_eye      |          16 |
    | left_ear       |          17 |
    | right_ear      |          18 |
    --------------------------------
    '''

    # Get canonical SMPL joint 3D positions
    smpl_model = load_model(model_path)
    n_betas = betas.shape[0]
    joints_3d_dirs = np.dstack([smpl_model.J_regressor.dot(
        smpl_model.shapedirs[:, :, i]) for i in range(n_betas)])
    joint_3d_positions_smpl = ch.array(joints_3d_dirs).dot(betas) + \
        smpl_model.J_regressor.dot(smpl_model.v_template.r)
    joint_3d_positions_smpl = joint_3d_positions_smpl.r

    # Get Cocoplus joint 3D positions
    with open(model_path, 'rb') as f:
        cocoplus_regressor = pk.load(f, encoding='latin-1')["cocoplus_regressor"]
        joints_3d_dirs = np.dstack([cocoplus_regressor.dot(
            smpl_model.shapedirs[:, :, i]) for i in range(n_betas)])
        joint_3d_positions_cocoplus = ch.array(joints_3d_dirs).dot(betas) + \
            cocoplus_regressor.dot(smpl_model.v_template.r)
        joint_3d_positions_cocoplus = joint_3d_positions_cocoplus.r.copy()

        # Adjust hip and neck positions to better fit to Openpose joints
        if adjust_coco_joints:
            # l_hip: the same x position as l_knee
            joint_3d_positions_cocoplus[3,0] = \
                joint_3d_positions_cocoplus[4,0]

            # r_hip: the same x position as r_knee
            joint_3d_positions_cocoplus[2,0] = \
                joint_3d_positions_cocoplus[1,0]

            # neck: the average y of l and r shoulders
            joint_3d_positions_cocoplus[12,1] = \
                np.mean(joint_3d_positions_cocoplus[8:10,1])

        joint_ids_cocoplus = [14, 12, 8, 7, 6,  9, 10, 11,  2,
                               1,  0, 3, 4, 5, 16, 15, 18, 17]
        joint_3d_positions_openpose = \
            joint_3d_positions_cocoplus[joint_ids_cocoplus]

    return joint_3d_positions_smpl, joint_3d_positions_openpose


def LoadOpenposeOutputs(openpose_path, video_name, frame_start, frame_end,
                        scale_hip_confidence=0.8, scale_head_confidence=0.5):

    # Loading 2d poses
    with open(openpose_path, 'rb') as f:
        joints_2d = pk.load(f, encoding='latin-1')[video_name][:,:,:3].astype(float) # nf x 18 x 3
        (nframes, num_joints, ndim) = joints_2d.shape

    # The 3rd dimension of joints_2d represent (x, y, confidence score).
    # We slightly reduce the confidence scores of hip and head joints by a
    # factor since their definitions are quite different between Openpose
    # and our human model
    joint_ids_hip = [8, 11] # right & left hip joints in Openpose
    joints_2d[:, joint_ids_hip, 2] = \
        scale_hip_confidence * joints_2d[:, joint_ids_hip, 2]
    joint_ids_head = [0, 14, 15, 16, 17] # head "joints" in Openpose
    joints_2d[:, joint_ids_head, 2] = \
        scale_head_confidence * joints_2d[:, joint_ids_head, 2]

    joints_2d = joints_2d.reshape((nframes, ndim*num_joints))
    return np.matrix(joints_2d[(frame_start-1):frame_end]).T


def ReplaceUnconfidentOpenposeJointsWithHMRJoints(
        joint_2d_positions_openpose,
        joint_2d_positions_hmr,
        list_openpose_joints=None,
        confidence_threshold=0.66,
        preserve_confidence_score=True):
    '''
    Replace unconfident openpose 2D joints with HMR joints.
    Note that the confidence score from Openpose is kept in the output.
    '''

    # Mapping from 18 Cocoplus joints to Openpose joints
    # ------------------------------------------
    # | Joint name | Openpose ID | Cocoplus ID |
    # |------------+-------------+-------------|
    # | nose       |           0 |          14 |
    # | neck       |           1 |          12 |
    # | r_shoulder |           2 |           8 |
    # | r_elbow    |           3 |           7 |
    # | r_wrist    |           4 |           6 |
    # | l_shoulder |           5 |           9 |
    # | l_elbow    |           6 |          10 |
    # | l_wrist    |           7 |          11 |
    # | r_hip      |           8 |           2 |
    # | r_knee     |           9 |           1 |
    # | r_ankle    |          10 |           0 |
    # | l_hip      |          11 |           3 |
    # | l_knee     |          12 |           4 |
    # | l_ankle    |          13 |           5 |
    # | r_eye      |          14 |          16 |
    # | l_eye      |          15 |          15 |
    # | r_ear      |          16 |          18 |
    # | l_ear      |          17 |          17 |
    # ------------------------------------------
    joint_ids_hmr = [14, 12, 8, 7, 6,  9, 10, 11,  2,
                      1,  0, 3, 4, 5, 16, 15, 18, 17]
    num_joints = len(joint_ids_hmr) # 18

    # Convert from (3*njoints, nframes) matrix to (nframes, njoints, 3) array
    joint_2d_positions_temp = joint_2d_positions_openpose.T.getA().reshape(
        (-1, num_joints, 3))
    nframes = joint_2d_positions_temp.shape[0]
    if nframes != joint_2d_positions_hmr.shape[0]:
        print(nframes)
        print(joint_2d_positions_hmr.shape[0])
        raise ValueError("joint_2d_positions_temp.shape[0] != "
                         "joint_2d_positions_hmr.shape[0]!")
    list_openpose_joints = \
        range(18) if list_openpose_joints is None else list_openpose_joints
    for j in list_openpose_joints:
        j_hmr = joint_ids_hmr[j]
        for i in range(nframes):
            confidence_score = joint_2d_positions_temp[i, j, 2]
            if confidence_score < confidence_threshold:
                joint_2d_positions_temp[i, j, :2] = \
                    joint_2d_positions_hmr[i, j_hmr, :2].copy()
                if not preserve_confidence_score:
                    joint_2d_positions_temp[i, j, 2] = 1.
    joint_2d_positions_temp = joint_2d_positions_temp.reshape(
        (nframes, 3*num_joints))
    return np.matrix(joint_2d_positions_temp).T

#
def LoadHmrOutputs(hmr_path, video_name, frame_start, frame_end):
    print(hmr_path)
    with open(hmr_path, 'rb') as f:
        data = pk.load(f, encoding='latin-1')[video_name]

    shapes = data['shapes'][(frame_start-1):frame_end]
    trans = data['trans'][(frame_start-1):frame_end]
    poses = data['poses'][(frame_start-1):frame_end]
    cams = data['cams'][(frame_start-1):frame_end]
    joint_2d_positions = data['joint_2d_positions'][(frame_start-1):frame_end]

    # Set betas as the mean of HMR output
    betas = np.mean(shapes, axis=0)

    # Convert config_smpl to config
    config_smpl = np.concatenate((trans, poses), axis=1)
    config_smpl = np.matrix(config_smpl).T
    joint_ids_pino = [0, 1, 5, 9, 2, 6,
                      10, 3, 7, 11, 4, 8,
                      12, 14, 19, 13, 15, 20,
                      16, 21, 17, 22, 18, 23]
    config = np.matrix(np.zeros(config_smpl.shape))

    # Copy the first 3 rows of configSMPL to config
    config[:3, :] = config_smpl[:3, :]

    # Relocate the rest of the rows (axis angles)
    for i in range(24):
        j = joint_ids_pino[i]
        config[(3*j+3):(3*(j+1)+3), :] = config_smpl[(3*i+3):(3*(i+1)+3), :]

    # Get camera focal length
    flength = np.mean(cams[:, 0], axis=0)
    return config, betas, flength, joint_2d_positions
