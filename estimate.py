import argparse
import ffmpeg
import sys
import shutil
import numpy as np
import numpy.linalg as LA
import pickle as pk
from glob import glob
from os import makedirs, remove
from os.path import join, exists, abspath, dirname, basename, isfile

import pinocchio as se3
from pinocchio.utils import *
from pinocchio.explog import exp, log

import lib.human_pose as human_pose
import lib.object_pose as object_pose
import lib.contact_states as lib_contact

try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
except ImportError:
    print("estimate.py: Viewer not imported")

import lib.solver as solver
from lib.solver import Camera, PosePriorGmm, DataloaderPerson, DataloaderObject, Minimize
import lib.utils as utils
from person_models.person import Person
from object_models.ground_plane import GroundPlane
from object_models.sticklike_objects import StickLikeObject
from optimizer import Optimizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Estimating 3D motion and forces of human-object "
        "interactions from Internet Videos")

    parser.add_argument("experiment", nargs='?', help="Experiment ID")
    parser.add_argument("action", nargs='?', help="Action name")
    parser.add_argument("video_path", nargs='?', help="Path to input video")
    parser.add_argument("--path-person-2d-poses", type=str,
                        help="Path to person 2D poses data")
    parser.add_argument("--path-contact-states", type=str, default=None,
                        help="Path to recognized contact states")
    parser.add_argument("--path-object-2d-keypoints", type=str, default=None,
                        help="Path to object 2D keypoints data")
    parser.add_argument("--path-init-person-3d-poses", type=str, default=None,
                        help="Path the initial person 3D poses.")
    parser.add_argument("--person-models-dir", type=str, default=None,
                        help="Path to the person models directory")
    parser.add_argument("--object-models-dir", type=str, default=None,
                        help="Path to the object models directory")
    parser.add_argument("--gt-dir", type=str, default=None,
                        help="Path to the directory containing ground truth"
                        "data (required for evaluation)")
    parser.add_argument("--evaluator-dir", type=str, default=None,
                        help="Path to the evluation library (required for"
                        " evaluation)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Path to the output directory")
    parser.add_argument("--restore-path", type=str, default=None,
                        help="Restore decision variables from path")
    parser.add_argument("--save-results", default=False, action="store_true",
                        help="Set to True to save decision variables")
    parser.add_argument("--save-name", type=str, default="test",
                        help="Decision variables are saved to path: "
                        "${out_dir}/${experiment}/${save_name}/*.pkl")
    parser.add_argument("--visualize", default=False, action="store_true",
                        help="Turn on visualization")
    parser.add_argument("--screenshot-folder", type=str, default=None,
                        help="Path to the folder for saving screenshots")
    parser.add_argument("--dry-run", default=False, action="store_true",
                        help="Run without the optimization stages")
    parser.add_argument("--frame-start", type=int, default=1,
                        help="Frame number of the first timestep")
    parser.add_argument("--frame-end", type=int, default=-1,
                        help="Frame number of the last timestep to optimize"
                        "(the very last frame by default)")
    parser.add_argument("--virtual-object", default=False,
                        action="store_true",
                        help="Turn on vritual object mode")
    parser.add_argument("--from-stage", type=int, default=1,
                        help="Run from optimization stage #")
    parser.add_argument("--until-stage", type=int, default=7,
                        help="Stop at optimization stage #")
    parser.add_argument("--max-num-it", type=int, default=20,
                        help="Maximum number of iterations")
    parser.add_argument("--fun-tolerance", type=float, default=1e-8,
                        help="Function tolerance for Ceres")
    parser.add_argument("--print-summary", type=int, default=1,
                        help="Print ceres summary")
    parser.add_argument("--evaluate-problem", default=False,
                        action="store_true",
                        help="Print residual values after optimization")

    # Optim-related parameters
    parser.add_argument("--seq_len", type=int, default=50,
                        help="length of the 'sliding window' to optimize")
    parser.add_argument("--overlap", type=int, default=2,
                        help="overlap between neighbour sliding windows")
    parser.add_argument("--add_p_com", type=int, default=1,
                        help="set to 1 to add the center of mass term "
                        "in optimization, set to 0 otherwise")
    parser.add_argument("--p_data", type=float, default=1., help="")
    parser.add_argument("--o_jvel", type=float, default=1e-2, help="")
    parser.add_argument("--p_jvel", type=float, default=1e-2, help="")
    parser.add_argument("--p_jacc", type=float, default=1e-3, help="")
    parser.add_argument("--p_pose", type=float, default=4., help="")
    parser.add_argument("--p_torq_b", type=float, default=1e-2, help="")
    parser.add_argument("--p_torq", type=float, default=1e-4, help="")
    parser.add_argument("--o_data", type=float, default=5., help="")
    parser.add_argument("--o_jacc", type=float, default=1e-3, help="")
    parser.add_argument("--o_torq_b", type=float, default=5e-2, help="")
    parser.add_argument("--oc_mot", type=float, default=1e4, help="")
    parser.add_argument("--gc_mot", type=float, default=1e3, help="")
    parser.add_argument("--c_type_f", type=float, default=10., help="")
    parser.add_argument("--c_type_s", type=float, default=1., help="")
    parser.add_argument("--radius_com", type=float, default=1e-1, help="")
    parser.add_argument("--p_cvel", type=float, default=1e-2, help="")
    parser.add_argument("--oc_for", type=float, default=1e-6, help="")
    parser.add_argument("--gc_for", type=float, default=1e-6, help="")
    parser.add_argument("--oc_flm", type=float, default=5e-4, help="")
    parser.add_argument("--gc_flm", type=float, default=1e-5, help="")
    parser.add_argument("--o_3dpt", type=float, default=1e4, help="")
    parser.add_argument("--p_3dpt", type=float, default=1e2, help="")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="confidence threshold for filtering and "
                        "replacing openpose 2D joints with HMR 2D joints.")

    args = parser.parse_args()
    experiment = args.experiment
    action = args.action
    video_path = args.video_path
    path_person_2d_poses = args.path_person_2d_poses
    path_contact_states = args.path_contact_states
    path_object_2d_keypoints = args.path_object_2d_keypoints
    path_init_person_3d_poses = args.path_init_person_3d_poses
    person_models_dir=args.person_models_dir
    object_models_dir=args.object_models_dir
    gt_dir = args.gt_dir
    evaluator_dir = args.evaluator_dir
    out_dir = args.out_dir
    restore_path = args.restore_path
    save_results = args.save_results
    save_name = args.save_name
    visualize = args.visualize
    screenshot_folder = args.screenshot_folder
    dry_run = args.dry_run
    frame_start = args.frame_start
    frame_end = args.frame_end
    virtual_object = args.virtual_object
    from_stage = args.from_stage
    until_stage = args.until_stage
    max_num_it = args.max_num_it
    fun_tolerance = args.fun_tolerance
    print_summary = args.print_summary
    evaluate_problem = args.evaluate_problem

    skip_stage5 = True
    skip_stage6 = True

    # Retrieve useful info
    video_name = basename(video_path).split('.')[0]

    # Optim-related parameters
    seq_len = args.seq_len
    overlap = args.overlap
    add_p_com = args.add_p_com
    p_data = args.p_data
    o_jvel = 0. if virtual_object else args.o_jvel
    p_jvel = args.p_jvel
    p_jacc = args.p_jacc
    p_pose = args.p_pose
    p_torq_b = args.p_torq_b
    p_torq = args.p_torq
    o_data = 0. if virtual_object else args.o_data
    o_jacc = 0. if virtual_object else args.o_jacc
    o_torq_b = 0. if virtual_object else args.o_torq_b
    oc_mot = 0. if virtual_object else args.oc_mot
    gc_mot = args.gc_mot
    c_type_f = args.c_type_f
    c_type_s = args.c_type_s
    radius_com = args.radius_com
    p_cvel = args.p_cvel
    oc_for = args.oc_for
    gc_for = args.gc_for
    oc_flm = args.oc_flm
    gc_flm = args.gc_flm
    o_3dpt = 0. if virtual_object else args.o_3dpt
    p_3dpt = args.p_3dpt
    confidence_threshold = args.confidence_threshold

    # Visualization-related parameters
    show_object = not virtual_object
    show_object_2d = False
    show_ground = True
    show_openpose_joints = False
    show_reprojected_joints = False
    show_reprojected_keypoints = False
    show_force = True
    show_force_gt = True
    remove_depth = True

    # ------------------------------------------------------------------
    print("(estimate.py) Loading video info ...")

    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] \
                         if stream['codec_type'] == 'video'), None)
    frame_size = np.array([int(video_stream['width']),
                           int(video_stream['height'])])
    nframes = int(video_stream['nb_frames'])
    fps = video_stream['r_frame_rate'].split('/')
    fps = float(fps[0])/float(fps[1])

    if not 1<=frame_start<=nframes:
        raise ValueError("Should never happen: 1<=frame_start<=nframes "
                         "(1<={0:d}<={1:d})".format(frame_start, nframes))
    if frame_end == -1:
        frame_end = nframes
    elif not 1<=frame_end<=nframes:
        raise ValueError("Should never happen: 1<=frame_end<=nframes "
                         "(1<={0:d}<={1:d})".format(frame_end, nframes))

    nf = frame_end - frame_start + 1
    friction_angle = np.pi/8.
    fze_gd_rot = 1.

    # ------------------------------------------------------------------
    print("(estimate.py) Initializing human 3D motion trajectory ...")

    config_person, betas, flength, joint_2d_positions_hmr = \
        human_pose.LoadHmrOutputs(path_init_person_3d_poses, video_name,
                                  frame_start, frame_end)


    # ------------------------------------------------------------------
    print("(estimate.py) Loading human pose prior ...")

    with open(join(person_models_dir,"gmm_pose_prior.pkl"), 'rb') as f:
        prior_data = pk.load(f, encoding='latin-1')
        pose_prior = PosePriorGmm(
            prior_data["means"], prior_data["precs"],
            prior_data["weights"], prior_data["prefix"])


    # ------------------------------------------------------------------
    print("(estimate.py) Loading human 2D joint positions ...")

    joint_2d_positions = human_pose.LoadOpenposeOutputs(
        path_person_2d_poses, video_name, frame_start, frame_end)

    # (optional) comment if needed
    # Replace unconfident Openpose ankle joints with 3D HMR joints
    # reprojected in 2D image
    joint_2d_positions = \
        human_pose.ReplaceUnconfidentOpenposeJointsWithHMRJoints(
            joint_2d_positions,
            joint_2d_positions_hmr,
            list_openpose_joints=[10, 13], # r_ankle, l_ankle
            confidence_threshold=confidence_threshold,
            preserve_confidence_score=False)

    # ------------------------------------------------------------------
    print("(estimate.py) Loading object 2D keypoint positions ...")

    if virtual_object:
        # Althought virtual objects have no endpoint detection, for
        # compativility reason it is required to create a zero data matrix
        # in appropriate size.
        endpoint_2d_positions = np.matrix(np.zeros((6, nf)))
    else:
        raise ValueError("Please add the '--virtual-object' option to run this commit")

    # ------------------------------------------------------------------
    print("(estimate.py) Loading contact states ...")

    contact = lib_contact.ContactStates(
        path_contact_states, video_name, action, frame_start, frame_end)

    # ------------------------------------------------------------------
    viewer = None
    if visualize:
        print("(estimate.py) Initializing Viewer ...")
        viewer = meshcat.Visualizer()
        viewer.open()
        viewer['/Grid'].set_property("visible", True)
        viewer['/Axes'].set_property("visible", True)


    # ------------------------------------------------------------------
    print("(estimate.py) Creating person model and dataloader ...")

    # Load joint 3D positions in neutral pose
    path_to_human_model = join(person_models_dir, "smpl", "models",
                               "neutral_smpl_with_cocoplus_reg.pkl")
    path_to_human_inertia = join(person_models_dir, "full_body_inertia.pkl")
    smpl_joints_neutral_pose, openpose_keypoints_neutral_pose = \
        human_pose.ComputeJoint3dPositionsInNeutralPose(
            path_to_human_model, betas, adjust_coco_joints=True)

    # Create person kinematic model
    person = Person("person",
                    smpl_joints_neutral_pose,
                    openpose_keypoints_neutral_pose,
                    inertia_path=path_to_human_inertia,
                    viewer=viewer)

    # Normalize axis angles
    config_person = utils.normalizeRotationAngles(
        config_person, person.decoration, align_opt=1, angle_opt=1)

    # Initialize person dataloader
    person_loader = DataloaderPerson(
        person.model,
        person.data,
        person.decoration.astype(float),
        config_person,
        fps,
        friction_angle,
        contact.num_ground_contacts,
        np.matrix(contact.contact_mapping.astype(float)).T,
        np.matrix(contact.contact_states.astype(float)),
        np.matrix(contact.contact_types.astype(float)),
        joint_2d_positions)

    joint_3d_positions_init = person_loader.joint_3d_positions_.copy()

    # ------------------------------------------------------------------
    print("(estimate.py) Creating ground model and dataloader ...")

    ground = GroundPlane(num_contacts=contact.num_ground_contacts,
                         num_keypoints=0,
                         viewer=viewer)

    ground.CreateObject()

    # Initialize ground configuration
    ground_initial_position = np.mean(config_person[:3, :], axis=1)
    ground_initial_rotation = np.matrix([0.,0.,1e-16]).T
    ground_initial_pose = np.concatenate((ground_initial_position, ground_initial_rotation))
    config_ground = np.tile(ground_initial_pose, (1, nf))
    config_ground_contact = np.matrix(np.zeros((ground.num_contacts*2, nf)))

    # Initialize ground dataloader
    ground_loader = DataloaderObject(
        ground.model,
        ground.data,
        ground.decoration.astype(float),
        config_ground,
        fps,
        "ground",
        config_ground_contact,
        np.matrix(np.zeros((1, nf))), # config_keypoints has no meaning for ground
        np.matrix(np.zeros((6, nf))), # endpoint_2d_positions has no meaning for ground
        False)


    # ------------------------------------------------------------------
    print("(estimate.py) Creating object model and dataloader ...")

    tool = StickLikeObject(num_contacts=contact.num_object_contacts,
                           num_keypoints=1,
                           viewer=viewer)

    if not virtual_object:
        path_to_object_params = join(
            object_models_dir, "parameters", action+".txt")
    else:
        path_to_object_params = join(
            object_models_dir, "parameters", "virtual_object.txt")

    tool.CreateObject(path_to_object_params)

    # Initialize config_object. It will be computed at later stages
    config_object = 1e-6*np.matrix(np.ones((6, nf)))

    # Compute object handle length
    if not virtual_object:
        handle_3d_len = object_pose.estimate_object_handle_length(
            action,
            joint_2d_positions,
            openpose_keypoints_neutral_pose,
            endpoint_2d_positions)
    else:
        handle_3d_len = 1.
    config_object_keypoints = handle_3d_len * np.matrix(np.ones((1, nf)))

    # Initialize contact point positions at the middle of the handle
    contact_initial_position = handle_3d_len/2.
    config_object_contact = contact_initial_position * np.matrix(
        np.ones((tool.num_contacts, nf)))

    # Initialize object dataloader
    object_loader = DataloaderObject(
        tool.model,
        tool.data,
        tool.decoration.astype(float),
        config_object,
        fps,
        tool.name,
        config_object_contact,
        config_object_keypoints,
        endpoint_2d_positions,
        virtual_object)


    # ------------------------------------------------------------------
    print("(estimate.py) Initializing projective camera model ...")

    cam_skew = 0.
    focal_length = np.matrix([flength, flength]).T.astype(float)
    principal_point = np.matrix(frame_size/2.).T
    cam_rotation = np.matrix(np.eye(3))
    cam_translation = np.matrix([0., 0., 0.]).T
    camera = Camera(
        cam_skew,
        focal_length,
        principal_point,
        cam_rotation,
        cam_translation)


    # ------------------------------------------------------------------
    print("(estimate.py) Initializing trajectory estimator ...")

    optimizer = Optimizer(
        video_name,
        seq_len,
        overlap,
        person,
        person_loader,
        object_model=tool,
        object_loader=object_loader,
        ground_model=ground,
        ground_loader=ground_loader,
        contact=contact,
        camera=camera,
        pose_prior=pose_prior,
        viewer=viewer,
        save_dir=join(out_dir, experiment))


    if restore_path is not None:
        optimizer.RestoreDecisionVariables(restore_path, virtual_object)

    # ------------------------------------------------------------------
    print("(estimate.py) Initializing evaluator ...")

    evaluator = None
    sys.path.append(evaluator_dir)
    if action in ('kv', 'mu', 'pu', 'sv'):
        from lib.evaluate import evaluate_parkour_sequence
        from parkour_evaluator import ParkourEvaluator

        evaluator = ParkourEvaluator(gt_dir, video_name)

    elif action in ('barbell', 'hammer', 'scythe', 'spade'):
        from lib.evaluate import evaluate_handtool_sequence
        from handtool_evaluator import HandtoolEvaluator

        evaluator = HandtoolEvaluator(gt_dir)
    else:
        raise ValueError("Unknown action: {}".format(action))

    stage_comments = (
        "Recover the 6d pose for the human basis joint",  # s1
        "Recover human 3D pose from 2D",  # s2
        "Joint refinement of camera focal length and human 3D pose",  # s3
        "Recover object 3D pose from contact states and human joints",  # s4
        "Joint estimation of human, object pose and contact positions",  # s5
        "Recover forces under dynamics constraints",  # s6
        "Joint estimation of 3D poses and contact forces")  # s7


    # stage 1 estimate person 3D poses from 2D
    #  - stage 1a, recover the 6D pose for the person's base joint
    #  - stage 1b,
    # stage 2 estiamte ground plan from person poses
    # stage 3 estimate object 3D poses from 2D
    # stage 4 joint estimation of 3D motion and contact forces


    # stage options (listed vertically)
    #     0 cam_variable,               * always 0 for Parkour sequences
    #     1 update_6d_basis_only,       * 1 for stage 1 et 0 for the rest
    #     2 measure_torso_joints_only,  * always 0
    #     3 freeze_6d_basis,            * always 0
    #     4 ground_plane_constrained,   * always 1
    #     5 freeze_ground_rotation      * always 1 for Parkour sequences
    #     6 CoM
    # (transposed list) fze == freeze
    # cam_var, 6d_only, torso_only, fze_6d, gd_cons, fze_gd_rot, add_p_com, 3D_ctt, smooth_oc, fze_obj_len, fze_q_o, fze_ctt_o
    #       0,       1,          2,      3,       4,          5,         6,      7,         8,           9,      10,        11
    stage_options = \
        [[ 0.,      1.,         0.,    -1.,     -1.,        -1.,       -1.,     -1,         0,          -1,      -1,        -1 ],  # 1
         [ 0.,      0.,         0.,     0.,     -1.,        -1., add_p_com,     -1,         0,          -1,      -1,        -1 ],  # 2
         [ 0.,      1.,         0.,    -1.,      1., fze_gd_rot,       -1.,     -1,         0,          -1,      -1,        -1 ],  # 3
         [-1.,      0.,        -1.,    -1.,     -1.,        -1.,       -1.,      1,         1,           1,       0,         0 ],  # 4
         [ 0.,      0.,         0.,     0.,     -1.,        -1., add_p_com,      1,         1,           1,       0,         0 ],  # 5
         [-1.,     -1.,        -1.,    -1.,     -1.,        -1.,       -1.,     -1,        -1,           1,      -1,        -1 ],  # 6
         [ 0.,      0.,         0.,     0.,     -1.,        -1., add_p_com,      1,         1,           1,       0,         0 ]]  # 7

    # set the weights of the different energy terms
    # index description
    #     0 person data                          * fixed
    #     1 person depth                         * fixed
    #     2 person velocity                      *
    #     3 person acceleration                  *
    #     4 person pose                          *
    #     5 person torque basis (sqrt)           *
    #     6 person torque (sqrt)                 *
    #     7 object data                          *
    #     8 object acceleration                  *
    #     9 object torque basis (sqrt)           *
    #    10 contact space: object                *
    #    11 contact space: ground                *
    #    12 contact time: fixed contact (sqrt)   *
    #    13 contact time: moving contact (sqrt)  *
    #    14 person center of mass:               * fixed

    # (transposed list)
    #     p_data, o_jvel, p_jvel, p_jacc, p_pose, p_torq_b, p_torq, o_data, o_jacc, o_torq_b, oc_mot, gc_mot, c_type_f, c_type_s, radius_com, p_cvel, oc_for, gc_for, oc_flm, gc_flm, o_3dpt, p_3dpt
    #          0,      1,      2,      3,      4,        5,      6,      7,      8,        9,     10,     11,       12,       13,         14,     15,     16,     17,     18,     19,     20,     21
    stage_weights =  \
        [[p_data,     -1, p_jvel, p_jacc,     -1,       -1,     -1,     -1,     -1,       -1,     -1,     -1, c_type_f, c_type_s,         -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1], # stage 1
         [p_data,     -1, p_jvel, p_jacc, p_pose,       -1,     -1,     -1,     -1,       -1,     -1,     -1, c_type_f, c_type_s, radius_com, p_cvel,     -1,     -1,     -1,     -1,     -1, p_3dpt], # stage 2
         [p_data,     -1, p_jvel, p_jacc,     -1,       -1,     -1,     -1,     -1,       -1,     -1, gc_mot, c_type_f, c_type_s,         -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1], # stage 3
         [   -1., o_jvel,     -1,     -1,     -1,       -1,     -1, o_data, o_jacc,     -1,tool.oc_mot_s4,     -1, c_type_f, c_type_s,         -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1], # stage 4
         [p_data,     -1, p_jvel, p_jacc, p_pose,       -1,     -1,     -1,     -1,       -1, oc_mot, gc_mot, c_type_f, c_type_s, radius_com, p_cvel,     -1,     -1,     -1,     -1,     -1, p_3dpt], # stage 5
         [   -1.,     -1,     -1,     -1,     -1, p_torq_b, p_torq,     -1,     -1, o_torq_b,     -1,     -1,       -1,       -1,         -1,     -1, oc_for, gc_for,     -1,     -1,     -1,     -1], # stage 6
         [p_data, o_jvel, p_jvel, p_jacc, p_pose, p_torq_b, p_torq, o_data, o_jacc, o_torq_b, oc_mot, gc_mot, c_type_f, c_type_s, radius_com, p_cvel, oc_for, gc_for,     -1,     -1, o_3dpt, p_3dpt]] # stage 7

    # Choose the optimization stages to run
    stages_to_run = []
    if not dry_run:
        for s in range(from_stage, until_stage+1):
            # Skip stage 4 for virtual object.
            if s==4 and virtual_object:
                continue
            # Optionally skip stage 5
            if s==5 and skip_stage5:
                continue
            # Optionally skip stage 6
            if s==6 and skip_stage6:
                continue

            stages_to_run.append(s)

        # Initialize Google Logging if there are optimization stages to run
        if len(stages_to_run) >= 1:
            solver.InitLogging()

    stage_with_all_frames = [1, 2, 3, 4, 5, 6, 7] # number of the stages in which we optimize over all frames # TODO(26/10/2019)

    # loop over the chosen stages
    for s in stages_to_run:
        # Get the number of trials to run at stage s
        if s==4:
            # Generate a set of initial guesses for the object configurations
            config_object_initial_guesses = object_pose.compute_object_config_initial_guesses(
                tool.model,
                tool.data,
                optimizer.object_loader,
                optimizer.person_loader,
                object_init_rotation_candidates=[[0., 1e-1, 0.],
                                                 [0., np.pi/2., 0.],
                                                 [0., np.pi, 0.],
                                                 [0., -np.pi/2., 0.]])

            num_trials = len(config_object_initial_guesses)
        else:
            num_trials = 1

        # Create a temporary folder for data saving and loading
        temp_folder = join(
                out_dir, experiment, save_name, "{0}_s{1}_trials".format(video_name, s))
        if not exists(temp_folder):
            makedirs(temp_folder)

        # Initialize an array of infinity numbers
        cost_trials = np.full([num_trials], np.inf)

        for n in range(num_trials):
            print("Stage {0} starts: {1} (trial {2}/{3})".format(s, stage_comments[s-1], n+1, num_trials))

            if s==4:
                # Update object configurations
                optimizer.object_loader.LoadConfig(config_object_initial_guesses[n], fps)
                # Reset keypoints and contact points
                optimizer.object_loader.LoadConfigContact(config_object_contact)
                optimizer.object_loader.LoadConfigKeypoints(config_object_keypoints)
                optimizer.object_loader.UpdateKeypoint2dReprojected(optimizer.camera)

            # Trick to avoid the bug reported in #f9e5c52 and solved in #7b92da4
            if not virtual_object:
                q_test = optimizer.object_loader.config_pino_[:,1].copy()
                q_test_m1 = optimizer.object_loader.config_pino_[:,0].copy()
                vq_test = se3.difference(optimizer.object_model.model, q_test_m1, q_test)

            # Start the optimization
            total_cost = 0.
            nseqs = optimizer.nseqs if s not in stage_with_all_frames else 1

            for i in range(nseqs): # nseqs. or for debugging: only run on three sequences
                frames = optimizer.list_seqs[i] if s not in stage_with_all_frames else range(nf)
                timestep_begin = frames[0]
                timestep_end = frames[-1]

                print("Stage {0}: processing frame #{1} - {2} / {3} (sequence {4} / {5})".format(
                    s, timestep_begin, timestep_end, nf-1, i, nseqs-1))
                solver_options = \
                    [print_summary,  # print_summary,
                     s,  # number of stage,
                     timestep_begin,  # timestep_begin,
                     timestep_end,  # timestep_end,
                     max_num_it,  # max_num_iterations,
                     2.,  # Trust region method: 1. Dogleg; 2. L.-M.
                     fun_tolerance,  # function_tolerance,
                     float(True), #  update_state_every_iteration
                     float(evaluate_problem), # evaluate_problem
                     tool.fix_contact_points,
                     tool.handle_length_bounds[0], # lower bound for searching the handle length
                     tool.handle_length_bounds[1]] # upper bound for searching the handle length

                final_cost = solver.Minimize(
                    np.matrix(stage_weights[s-1]).T,
                    np.matrix(stage_options[s-1]).T,
                    np.matrix(solver_options).T,
                    optimizer.person_loader,
                    optimizer.object_loader,
                    optimizer.ground_loader,
                    optimizer.pose_prior,
                    optimizer.camera)

                # Consider the current trial as a failure if zero cost is obtained
                # (Ceres probably failed to compute in this case).
                if final_cost < 1e-10:
                    total_cost = np.inf
                    break
                else:
                    total_cost += final_cost

            # Save data to a temporay folder
            if num_trials > 1:
                trial_save_path = join(temp_folder, "trial_{0}.pkl".format(n+1))
                optimizer.SaveDecisionVariables(
                    trial_save_path,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    save_object_endpoints=True,
                    save_forces=True,
                    solver_options=solver_options,
                    stage_options=stage_options,
                    stage_weights=stage_weights)
            cost_trials[n] = total_cost

        if num_trials > 1:
            print("cost_trials: ")
            print(cost_trials)

            # Keep the parameters leading to the lowest total_cost
            trial_restore_path = join(
                temp_folder, "trial_{0}.pkl".format(np.argmin(cost_trials) + 1))
            optimizer.RestoreDecisionVariables(trial_restore_path, virtual_object)

        # Remove the temporary I/O folder
        shutil.rmtree(temp_folder)

        if save_results:
            # ------------------------------------------------------------------
            print("(estimate.py) Stage {0}: saving results ...".format(s))

            save_path = join(
                out_dir, experiment, save_name, "{0}_s{1}.pkl".format(video_name, s))
            optimizer.SaveDecisionVariables(
                save_path,
                frame_start=frame_start,
                frame_end=frame_end,
                save_object_endpoints=True,
                save_forces=True,
                solver_options=solver_options,
                stage_options=stage_options,
                stage_weights=stage_weights)

        # Run evaluation for parkour sequences
        if evaluator is not None:
            if action in ('kv', 'mu', 'pu', 'sv'):
                eval_forces = False
                save_path = join(out_dir, experiment, "evaluation",
                                 "{0}_s{1}".format(save_name, s),
                                 "{0}.pkl".format(video_name))
                evaluate_parkour_sequence(evaluator, optimizer,
                                          eval_forces=eval_forces,
                                          save_path=save_path)
            elif action in ('barbell', 'hammer', 'scythe', 'spade'):
                evaluate_handtool_sequence(
                    video_name,
                    action,
                    evaluator,
                    optimizer,
                    verbose=True,
                    save_path=join(out_dir, experiment, "evaluation", "{0}_s{1}".format(save_name, s), "{0}.pkl".format(video_name)))

        # visualizing the estimated human-object trajectories
        if visualize:
            camera_angle = 90.
            camera_angle_rad = np.radians(camera_angle)
            optimizer.PlaySequence(frames=None,
                                   show_person=True,
                                   show_object=show_object,
                                   show_object_2d=show_object_2d,
                                   show_ground=show_ground,
                                   show_person_baseline=False,
                                   show_object_baseline=False,
                                   show_openpose_joints=show_openpose_joints,
                                   show_reprojected_joints=show_reprojected_joints,
                                   show_reprojected_keypoints=show_reprojected_keypoints,
                                   show_force=show_force,
                                   show_force_gt=show_force_gt,
                                   remove_depth=remove_depth,
                                   virtual_object=virtual_object,
                                   camera_angle=camera_angle_rad,
                                   pause=False)

    if visualize:
        camera_angle = 90.
        camera_angle_rad = np.radians(camera_angle)
        optimizer.PlaySequence(frames=None,
                               show_person=True,
                               show_object=show_object,
                               show_object_2d=show_object_2d,
                               show_ground=show_ground,
                               show_person_baseline=False,
                               show_object_baseline=False,
                               show_openpose_joints=show_openpose_joints,
                               show_reprojected_joints=show_reprojected_joints,
                               show_reprojected_keypoints=show_reprojected_keypoints,
                               show_force=show_force,
                               show_force_gt=show_force_gt,
                               remove_depth=remove_depth,
                               virtual_object=virtual_object,
                               camera_angle=camera_angle_rad,
                               pause=False)

        # Clean up visuals
        optimizer.ClearScene()
