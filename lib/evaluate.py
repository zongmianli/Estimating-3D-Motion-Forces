import pickle as pk
from os import makedirs, remove
from os.path import join, exists, abspath, dirname, basename, isfile


def evaluate_handtool_sequence(
        video_name,
        action,
        evaluator,
        optimizer,
        verbose=False,
        save_path=None):
    '''
    Compare the estimated 3D motion with the ground truth 3D joint positions
    at keyframes for the input Handtool video.
    '''

    # Extract person 3D joint positions at the keyframes
    # Note that the original keyframes count from 1
    keyframes = evaluator.keyframes[video_name]
    keyframes = [k-1 for k in keyframes]
    num_keyframes = len(keyframes)

    joint_3d_positions = \
        optimizer.person_loader.joint_3d_positions_[:, keyframes].T#.getA()
    joint_3d_positions = joint_3d_positions.reshape((num_keyframes, 24, 3))

    # Mapping from the IDs of the 12 joints considered in the Handtool
    # dataset to the joints in our human model
    # -------------------------------------
    # | Joint name | Parkour ID | Ours ID |
    # | ---------- | ---------- | ------- |
    # | r_ankle    |          0 |       7 |
    # | r_knee     |          1 |       6 |
    # | r_hip      |          2 |       5 |
    # | l_hip      |          3 |       1 |
    # | l_knee     |          4 |       2 |
    # | l_ankle    |          5 |       3 |
    # | r_wrist    |          6 |      22 |
    # | r_elbow    |          7 |      21 |
    # | r_shoulder |          8 |      20 |
    # | l_shoulder |          9 |      15 |
    # | l_elbow    |         10 |      16 |
    # | l_wrist    |         11 |      17 |
    # -------------------------------------
    joint_mapping = [7, 6, 5, 1, 2, 3, 22, 21, 20, 15, 16, 17]

    data_dict = {'joint_3d_positions': joint_3d_positions[:,joint_mapping,:]}
    sequence_data_pred = {(action, video_name): data_dict}

    # Compute 3D joint errors
    joint_errors, _, mpjpe = evaluator.ComputeMotionErrors(
        sequence_data_pred, verbose=verbose)

    # Save evaluation results
    if save_path is not None:
        results = dict()
        results["action"] = action
        results["joint_errors"] = joint_errors
        results["mpjpe"] = mpjpe
        data = {video_name: results}
        if not exists(dirname(save_path)):
            makedirs(dirname(save_path))
        with open(save_path, 'w') as f:
            pk.dump(data, f)
            print("(evaluate.py) Evaluation results saved to: \n"
                  " - {}".format(save_path))


def evaluate_parkour_sequence(evaluator, optimizer,
                              eval_forces=False,
                              #eval_baseline=False,
                              save_path=None):
    '''
    Computes the 3D motion and force estimation errors with the ground truth
    values in the LAAS Parkour dataset.
    '''

    # Mapping from the IDs of the 16 joints considered in the Parkour
    # dataset to the joints in our human model
    # -------------------------------------
    # | Joint name | Parkour ID | Ours ID |
    # | ---------- | ---------- | ------- |
    # | l_hip      |          0 |       1 |
    # | l_knee     |          1 |       2 |
    # | l_ankle    |          2 |       3 |
    # | l_toes     |          3 |       4 |
    # | r_hip      |          4 |       5 |
    # | r_knee     |          5 |       6 |
    # | r_ankle    |          6 |       7 |
    # | r_toes     |          7 |       8 |
    # | l_shoulder |          8 |      15 |
    # | l_elbow    |          9 |      16 |
    # | l_wrist    |         10 |      17 |
    # | l_fingers  |         11 |      18 |
    # | r_shoulder |         12 |      20 |
    # | r_elbow    |         13 |      21 |
    # | r_wrist    |         14 |      22 |
    # | r_fingers  |         15 |      23 |
    # -------------------------------------
    joint_mapping = \
        [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 20, 21, 22, 23]

    seqLocalForceGround_GT, seqForceObject_GT = \
        optimizer.ExpressParkourForcesInLocalJointFrame(
            optimizer.person_loader.config_pino_.copy(),
            evaluator.contact_forces_local)
    optimizer.seqLocalForceGround_GT = seqLocalForceGround_GT
    optimizer.seqForceObject_GT = seqForceObject_GT

    # Evalute estimated 3D human motion
    # print(optimizer.person_loader.joint_3d_positions_.shape) # nq, nframes
    joint_3d_positions_all = \
        optimizer.person_loader.joint_3d_positions_.T.reshape(
            (optimizer.nf, -1, 3)) # nf x njoints x 3 array
    joint_3d_positions_pred = joint_3d_positions_all[:,joint_mapping,:]
    evaluator.Evaluate3DPoses(joint_3d_positions_pred)
    print("(evaluate.py) 3D motion estimation errors:\n"
          " - MPJPE (mm): {0:.2f}".format(
              evaluator.mpjpe['procrustes']))

    # # Compute 3D motion errors for baseline method
    # if eval_baseline:
    #     joint_3d_positions_all_init = \
    #         optimizer.person_loader.joint_3d_positions_init_.T.getA().reshape((optimizer.nf, -1, 3)) # nf x njoints x 3 array
    #     joint_3d_positions_baseline = \
    #         joint_3d_positions_all_init[:,joint_mapping,:]
    #     evaluator.Evaluate3DPoses(joint_3d_positions_baseline)
    #     print(" - MPJPE (baseline): {0:.2f} mm".format(
    #         evaluator.mpjpe['procrustes']))

    # Compute force estimation errors
    if eval_forces:
        contact_forces_pred = optimizer.ExpressLocalForcesInParkourFrame(
            optimizer.person_loader.config_pino_.copy())
        contact_forces_pred[0] = contact_forces_pred[2].copy()
        contact_forces_pred[1] = contact_forces_pred[2].copy()
        evaluator.EvaluateContactForces(contact_forces_pred)
        print("(evaluate.py) Force estimation errors "
              "{0}:".format(evaluator.contact_forces_names))
        print(" - Mean lin. force errors (Newton): {0}".format(
            evaluator.mean_linear_force_errors))
        print(" - Mean torque errors (Newton-metre): {0}".format(
            evaluator.mean_torque_errors))

    if save_path is not None:
        evaluator.SaveResults(save_path, save_forces=eval_forces)
