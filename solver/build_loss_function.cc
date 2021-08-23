#include "solver.h"
#include "cost_functors/contact_ground_spatial.h"
#include "cost_functors/contact_object_spatial.h"
//#include "cost_functors/contact_symmetric.h"
#include "cost_functors/contact_temporal.h"
//#include "cost_functors/horizontal_object.h"
#include "cost_functors/limit_ground_force.h"
#include "cost_functors/limit_object_force.h"
#include "cost_functors/object_3d_errors.h"
#include "cost_functors/object_acceleration.h"
#include "cost_functors/object_data.h"
#include "cost_functors/object_torque.h"
#include "cost_functors/object_velocity.h"
#include "cost_functors/person_3d_errors.h"
#include "cost_functors/person_acceleration.h"
#include "cost_functors/person_center_of_mass.h"
#include "cost_functors/person_data.h"
#include "cost_functors/person_depth.h"
#include "cost_functors/person_pose.h"
#include "cost_functors/person_torque.h"
#include "cost_functors/person_velocity.h"
#include "cost_functors/smoothing_cartesian_velocity.h"
#include "cost_functors/smoothing_ground_force.h"
#include "cost_functors/smoothing_object_force.h"

// New cost terms with local parameterization
//#include "cost_terms/object_data.h"

// Print str to stdout. Print blank spaces of the same length as str
// if print_str is set to true.
void PrintStringToStdout(std::string str, bool print_str)
{
    if(print_str)
    {
        std::cout << str;
    }
    else
    {
        int str_size = str.size();
        for (int n=0; n<str_size; n++)
        {
            std::cout << " ";
        }
    }
}



void BuildLossFunction(
    ceres::Problem &problem,
    std::vector<std::string> &residual_block_names,
    std::vector<std::string> &residual_block_types,
    std::vector<ceres::ResidualBlockId> &residual_block_ids,
    int timestep_begin,
    int timestep_end,
    const Eigen::VectorXd &stage_weights,
    const Eigen::VectorXd &stage_options,
    const bool fix_object_contact_points,
    DataloaderPerson &person_loader,
    DataloaderObject &object_loader,
    DataloaderObject &ground_loader,
    PosePriorGmm &pose_prior,
    Camera &camera)
{
    // ------------------------------------------------------------------
    // Resolve stage weights
    // ------------------------------------------------------------------
    double w_p_data = stage_weights(0);
    double w_o_jvel = stage_weights(1);
    double w_p_jvel = stage_weights(2);
    double w_p_jacc = stage_weights(3);
    double w_p_pose = stage_weights(4);
    double w_p_torq_b = stage_weights(5);
    double w_p_torq = stage_weights(6);
    double w_o_data = stage_weights(7);
    double w_o_jacc = stage_weights(8);
    double w_o_torq_b = stage_weights(9);
    double w_oc_mot = stage_weights(10);
    double w_gc_mot = stage_weights(11);
    double w_c_type_f = stage_weights(12);
    double w_c_type_s = stage_weights(13);
    double radius_com = stage_weights(14);
    double w_p_cvel = stage_weights(15);
    double w_oc_for = stage_weights(16);
    double w_gc_for = stage_weights(17);
    double w_oc_flm = stage_weights(18);
    double w_gc_flm = stage_weights(19);
    double w_o_3dpt = stage_weights(20);
    double w_p_3dpt = stage_weights(21);
    std::cout << "----- stage_weights -----" << std::endl;
    std::cout << "w_p_data " << w_p_data << std::endl;
    std::cout << "w_o_jvel " << w_o_jvel << std::endl;
    std::cout << "w_p_jvel " << w_p_jvel << std::endl;
    std::cout << "w_p_jacc " << w_p_jacc << std::endl;
    std::cout << "w_p_pose " << w_p_pose << std::endl;
    std::cout << "w_p_torq_b " << w_p_torq_b << std::endl;
    std::cout << "w_p_torq " << w_p_torq << std::endl;
    std::cout << "w_o_data " << w_o_data << std::endl;
    std::cout << "w_o_jacc " << w_o_jacc << std::endl;
    std::cout << "w_o_torq_b " << w_o_torq_b << std::endl;
    std::cout << "w_oc_mot " << w_oc_mot << std::endl;
    std::cout << "w_gc_mot " << w_gc_mot << std::endl;
    std::cout << "w_c_type_f " << w_c_type_f << std::endl;
    std::cout << "w_c_type_s " << w_c_type_s << std::endl;
    std::cout << "radius_com " << radius_com << std::endl;
    std::cout << "w_p_cvel " << w_p_cvel << std::endl;
    std::cout << "w_oc_for " << w_oc_for << std::endl;
    std::cout << "w_gc_for " << w_gc_for << std::endl;
    std::cout << "w_oc_flm " << w_oc_flm << std::endl;
    std::cout << "w_gc_flm " << w_gc_flm << std::endl;
    std::cout << "w_o_3dpt " << w_o_3dpt << std::endl;
    std::cout << "w_p_3dpt " << w_p_3dpt << std::endl;
    std::cout << "----- END: stage_weights -----" << std::endl;

    // ------------------------------------------------------------------
    // Resolve stage_options
    // ------------------------------------------------------------------
    int cam_variable = (int)stage_options(0); // 0
    int update_6d_basis_only = (int)stage_options(1); // 0
    int measure_torso_joints_only = (int)stage_options(2); // 0
    int freeze_6d_basis = (int)stage_options(3); // -1
    int ground_plane_constrained = (int)stage_options(4); // -1
    int freeze_ground_rotation = (int)stage_options(5); // -1
    int add_p_com = stage_options(6);
    int enforce_3D_distance = (int)stage_options(7); // we only minimize 2D distance between object and human joints if object is not detected
    bool smooth_oc = (bool)stage_options(8);
    int freeze_handle_length = (int)stage_options(9);
    std::cout << "----- stage_options -----" << std::endl;
    std::cout << "cam_variable " << cam_variable << std::endl;
    std::cout << "update_6d_basis_only " << update_6d_basis_only << std::endl;
    std::cout << "measure_torso_joints_only " << measure_torso_joints_only << std::endl;
    std::cout << "freeze_6d_basis " << freeze_6d_basis << std::endl;
    std::cout << "ground_plane_constrained " << ground_plane_constrained << std::endl;
    std::cout << "freeze_ground_rotation " << freeze_ground_rotation << std::endl;
    std::cout << "add_p_com " << add_p_com << std::endl;
    std::cout << "enforce_3D_distance " << enforce_3D_distance << std::endl;
    std::cout << "smooth_oc " << smooth_oc << std::endl;
    std::cout << "freeze_handle_length " << freeze_handle_length << std::endl;
    std::cout << "----- END: stage_options -----" << std::endl;

    std::cout << "----- adding residual blocks -----" << std::endl;
    bool has_object_contact, has_ground_contact;
    bool cost_term_is_added = false;
    std::string str_to_print;
    ceres::ResidualBlockId block_id;

    // Get object information
    bool virtual_object = object_loader.get_is_virtual_object();

    // Print the header line
    std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "| time |          p_data | o_data | p_pose | oc_mot | gc_mot |      p_torq | o_torq | p_jvel | p_cvel | p_jacc | o_jvel | o_jacc | c_type | oc_for | gc_for | oc_flm | gc_flm | p_com |" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;

    for (int i = timestep_begin; i <= timestep_end; i++)
    {
        std::cout << "| #" << std::setw(3) << std::setfill('0') << i << " | ";
        // ------------------------------------------------------------------
        // Data term: 2D re-projection error
        // ------------------------------------------------------------------
        if (w_p_data > 0)
        {
            ceres::CostFunction *cost_function_reprojection_error =
                CostFunctorPersonData::Create(i,
                                              &person_loader,
                                              &camera,
                                              (bool)cam_variable,
                                              (bool)update_6d_basis_only,
                                              (bool)measure_torso_joints_only);
            if (cam_variable == 1)
            {
                //std::cout << "(cam) ";
                block_id = problem.AddResidualBlock(
                    cost_function_reprojection_error,
                    new ceres::HuberLoss(1.0),
                    person_loader.mutable_config(i),
                    camera.mutable_focal_length());
            }
            else
            {
                block_id = problem.AddResidualBlock(
                    cost_function_reprojection_error,
                    new ceres::HuberLoss(1.0),
                    person_loader.mutable_config(i));
            }
            cost_term_is_added = true;
            str_to_print = "p_data ";
            // Save block information for later evaluation
            std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | p_data ";
            residual_block_names.push_back(block_name.str());
            residual_block_types.push_back("p_data");
            residual_block_ids.push_back(block_id);
        }
        else
        {
            // print 17 blank spaces otherwise
            str_to_print = "                ";
        }
        PrintStringToStdout(str_to_print, cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        // ------------------------------------------------------------------
        // Penalize 3D distance between person joints and their initializations
        // ------------------------------------------------------------------
        if (w_p_3dpt > 0)
        {
            ceres::CostFunction *cost_function_person_3d_errors =
                CostFunctorPerson3dErrors::Create(i, &person_loader);
            ceres::LossFunction *loss_function_person_3d_errors =
                new ceres::ScaledLoss(NULL, w_p_3dpt, ceres::TAKE_OWNERSHIP);
            block_id = problem.AddResidualBlock(
                cost_function_person_3d_errors,
                loss_function_person_3d_errors,
                person_loader.mutable_config(i));
            cost_term_is_added = true;
            // Save block information for later evaluation
            std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | p_3dpt ";
            residual_block_names.push_back(block_name.str());
            residual_block_types.push_back("p_3dpt");
            residual_block_ids.push_back(block_id);
        }
        PrintStringToStdout("p_3dpt ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        // For a non-virtual object, we minimize keypoint reprojection errors if the object is detected at the current timestep
        bool object_is_detected = false;
        if (!virtual_object)
        {
            object_is_detected =
                object_loader.get_endpoint_2d_positions_column(i).squaredNorm() > 1e-10;
        }

        if (object_is_detected && w_o_data > 0 && (!virtual_object))
        {
            ceres::CostFunction *cost_function_obj_reproj =
                CostFunctorObjectData::Create(i, &object_loader, &camera);
            // ceres::LossFunction *loss_function_obj_reproj =
            //     new ceres::ScaledLoss(NULL, w_o_data, ceres::TAKE_OWNERSHIP);
            // problem.AddResidualBlock(cost_function_obj_reproj,
            //                          loss_function_obj_reproj,
            //                          object_loader.mutable_config(i),
            //                          object_loader.mutable_config_keypoints(0));
            // NOTE: add robustifier?
            block_id = problem.AddResidualBlock(
                cost_function_obj_reproj,
                new ceres::HuberLoss(1.0),
                object_loader.mutable_config_pino(i),
                object_loader.mutable_config_keypoints(0));
            cost_term_is_added = true;
            // Save block information for later evaluation
            std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | o_data ";
            residual_block_names.push_back(block_name.str());
            residual_block_types.push_back("o_data");
            residual_block_ids.push_back(block_id);
        }
        PrintStringToStdout("o_data ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        // ------------------------------------------------------------------
        // Penalize 3D distance between joints and their initializations
        // ------------------------------------------------------------------
        if (w_o_3dpt > 0 && (!virtual_object))
        {
            ceres::CostFunction *cost_function_object_3d_errors =
                CostFunctorObject3dErrors::Create(i, &object_loader);
            ceres::LossFunction *loss_function_object_3d_errors =
                new ceres::ScaledLoss(NULL, w_o_3dpt, ceres::TAKE_OWNERSHIP);
            block_id = problem.AddResidualBlock(
                cost_function_object_3d_errors,
                loss_function_object_3d_errors,
                object_loader.mutable_config_pino(i),
                object_loader.mutable_config_keypoints(0));
            cost_term_is_added = true;
            // Save block information for later evaluation
            std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | o_3dpt ";
            residual_block_names.push_back(block_name.str());
            residual_block_types.push_back("o_3dpt");
            residual_block_ids.push_back(block_id);
        }
        PrintStringToStdout("o_3dpt ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        // ------------------------------------------------------------------
        // Prior on 3D human poses
        // ------------------------------------------------------------------
        if (w_p_pose > 0)
        {
            ceres::CostFunction *cost_function_p_pose =
                CostFunctorPersonPose::Create(i, &person_loader, &pose_prior);
            ceres::LossFunction *loss_function_p_pose =
                new ceres::ScaledLoss(NULL, w_p_pose, ceres::TAKE_OWNERSHIP);
            block_id = problem.AddResidualBlock(
                cost_function_p_pose,
                loss_function_p_pose,
                person_loader.mutable_config(i));
            cost_term_is_added = true;
            // Save block information for later evaluation
            std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | p_pose ";
            residual_block_names.push_back(block_name.str());
            residual_block_types.push_back("p_pose");
            residual_block_ids.push_back(block_id);
        }
        PrintStringToStdout("p_pose ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        // ------------------------------------------------------------------
        // Physical plausibility of the motion
        // ------------------------------------------------------------------

        // Contact motion model for person-object contacts
        if (w_oc_mot > 0 && (!virtual_object))
        {
            // Contact Motion term:
            // We penalize the Euclidean distance between person joints
            // and their corresponding contact points defined on the object.
            // 3D distance is minimized if enforce_3D_distance is true,
            // otherwise it minimizes 2D distance.
            ceres::CostFunction *cost_contact_object_spatial =
                CostFunctorContactObjectSpatial::Create(
                    i,
                    &person_loader,
                    &object_loader,
                    (bool)enforce_3D_distance);
            if (cost_contact_object_spatial != NULL)
            {
                ceres::LossFunction *loss_contact_object_spatial =
                    new ceres::ScaledLoss(
                        NULL, w_oc_mot, ceres::TAKE_OWNERSHIP);
                // NOTE: The current contact motion model is "hard", which is not
                //       robust to contact recognition errors.
                // TODO: try add robustifier to loss function?

                // Specify the contact type: sliding contact v.s. fixed contact
                // NOTE: Currently, the contact type is specified "globally",
                //       that is to say, either all contact points are fixed, or all
                //       of them are sliding contacts.
                // TODO: Implement contact types per joint, per contact phase

                if (fix_object_contact_points)
                {
                    block_id = problem.AddResidualBlock(
                        cost_contact_object_spatial,
                        loss_contact_object_spatial,
                        person_loader.mutable_config(i),
                        object_loader.mutable_config_pino(i),
                        object_loader.mutable_config_contact(0));
                }
                else
                {
                    block_id = problem.AddResidualBlock(
                        cost_contact_object_spatial,
                        loss_contact_object_spatial,
                        person_loader.mutable_config(i),
                        object_loader.mutable_config_pino(i),
                        object_loader.mutable_config_contact(i));
                }
                cost_term_is_added = true;
                // Save block information for later evaluation
                std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | oc_mot ";
                residual_block_names.push_back(block_name.str());
                residual_block_types.push_back("oc_mot");
                residual_block_ids.push_back(block_id);
            }
            PrintStringToStdout("oc_mot ", cost_term_is_added);
            std::cout << "| "; cost_term_is_added = false;

            // // Symmetric Contact term:
            // // For barbell videos, we check if both left and right hand are
            // // in contact with object. If this is the case, we penalize 
            // // the distance between the center of left and right hand 
            // // contact points and the center of barbell bar
            // if (add_contact_symmetric > 0)
            // {
            //     ceres::CostFunction *cost_contact_symmetric =
            //         CostFunctorContactSymmetric::Create(
            //             i,
            //             &person_loader,
            //             &object_loader);
            //     if (cost_contact_symmetric != NULL)
            //     {
            //         ceres::LossFunction *loss_contact_symetric =
            //             new ceres::ScaledLoss(
            //                 NULL, w_oc_mot, ceres::TAKE_OWNERSHIP); // NOTE: we set w_oc_sym == w_oc_mot here
            //         // if (sliding_object_contact==0)
            //         // {
            //         //     problem.AddResidualBlock(
            //         //         cost_contact_symmetric,
            //         //         loss_contact_symetric,
            //         //         object_loader.mutable_config_keypoints(0),
            //         //         object_loader.mutable_config_contact(0));
            //         // }
            //         problem.AddResidualBlock(
            //             cost_contact_symmetric,
            //             loss_contact_symetric,
            //             object_loader.mutable_config_keypoints(0),
            //             object_loader.mutable_config_contact(i));
            //         cost_term_is_added = true;
            //     }
            // }
            // PrintStringToStdout("oc_sym ", cost_term_is_added);
            // std::cout << "| "; cost_term_is_added = false;
        }
        else
        {
            // otherwise fill the line with blank spaces
            std::cout << "       |        | ";
        }

        // Contact motion model for person-ground contacts
        if (w_gc_mot > 0)
        {
            ceres::CostFunction *cost_contact_ground_spatial =
                CostFunctorContactGroundSpatial::Create(
                    i,
                    &person_loader,
                    &ground_loader,
                    (bool)update_6d_basis_only);
            if (cost_contact_ground_spatial != NULL)
            {
                ceres::LossFunction *loss_contact_ground_spatial =
                    new ceres::ScaledLoss(
                        NULL, w_gc_mot, ceres::TAKE_OWNERSHIP);
                // NOTE: The current contact motion model is "hard", which is not
                //       robust to contact recognition errors.
                // TODO: try add robustifier to loss function?

                block_id = problem.AddResidualBlock(
                    cost_contact_ground_spatial,
                    loss_contact_ground_spatial,
                    person_loader.mutable_config(i),
                    ground_loader.mutable_config(0), // NOTE: use 0 or timestep_begin?
                    ground_loader.mutable_config_contact(i));
                cost_term_is_added = true;
                // Save block information for later evaluation
                std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | gc_mot ";
                residual_block_names.push_back(block_name.str());
                residual_block_types.push_back("gc_mot");
                residual_block_ids.push_back(block_id);
            }
        }
        PrintStringToStdout("gc_mot ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        // Person torque term (with full-body dynamics and friction model);
        Eigen::Vector2d weights_person_troque;
        weights_person_troque << w_p_torq_b, w_p_torq;
        ceres::CostFunction *cost_function_person_torque =
            CostFunctorPersonTorque::Create(
                i,
                &person_loader,
                &object_loader,
                weights_person_troque,
                &has_object_contact,
                &has_ground_contact);
        if (i >= 2 && w_p_torq_b > 0 && w_p_torq > 0)
        {
            if (has_object_contact && has_ground_contact)
            {
                str_to_print = "(og) p_torq ";
                block_id = problem.AddResidualBlock(
                    cost_function_person_torque,
                    NULL,
                    person_loader.mutable_config(i),
                    person_loader.mutable_config(i - 1),
                    person_loader.mutable_config(i - 2),
                    object_loader.mutable_contact_force(i),
                    person_loader.mutable_ground_friction(i));
            }
            else if (has_object_contact)
            {
                str_to_print = " (o) p_torq ";
                block_id = problem.AddResidualBlock(
                    cost_function_person_torque,
                    NULL,
                    person_loader.mutable_config(i),
                    person_loader.mutable_config(i - 1),
                    person_loader.mutable_config(i - 2),
                    object_loader.mutable_contact_force(i));
            }
            else if (has_ground_contact)
            {
                str_to_print = " (g) p_torq ";
                block_id = problem.AddResidualBlock(
                    cost_function_person_torque,
                    NULL,
                    person_loader.mutable_config(i),
                    person_loader.mutable_config(i - 1),
                    person_loader.mutable_config(i - 2),
                    person_loader.mutable_ground_friction(i));
            }
            else
            {
                str_to_print = "     p_torq ";
                block_id = problem.AddResidualBlock(
                    cost_function_person_torque,
                    NULL,
                    person_loader.mutable_config(i),
                    person_loader.mutable_config(i - 1),
                    person_loader.mutable_config(i - 2));
            }

            // Ground contact forces should lie in their friction cones
            // if (has_ground_contact)
            // {
            //     for (int k = 0; k<person_loader.get_nq_ground_friction(); k++)
            //     {
            //         problem.SetParameterLowerBound(
            //             person_loader.mutable_ground_friction(i),
            //             k,
            //             0.0);
            //     }
            // }
            cost_term_is_added = true;
            // Save block information for later evaluation
            std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | p_torq ";
            residual_block_names.push_back(block_name.str());
            residual_block_types.push_back("p_torq");
            residual_block_ids.push_back(block_id);
        }
        else
        {
            str_to_print = "            ";
        }
        PrintStringToStdout(str_to_print, cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        // Object torque term
        if (i >= 2 && w_o_torq_b > 0 && has_object_contact && (!virtual_object))
        {
            Eigen::Vector2d weights_object_troque;
            weights_object_troque << w_o_torq_b, w_o_torq_b;
            ceres::CostFunction *cost_function_object_torque =
                CostFunctorObjectTorque::Create(
                    i,
                    &person_loader,
                    &object_loader,
                    weights_object_troque);
            if (fix_object_contact_points)
            {
                block_id = problem.AddResidualBlock(
                    cost_function_object_torque,
                    NULL,
                    object_loader.mutable_config_pino(i),
                    object_loader.mutable_config_pino(i - 1),
                    object_loader.mutable_config_pino(i - 2),
                    object_loader.mutable_config_contact(0),
                    object_loader.mutable_config_keypoints(0),
                    object_loader.mutable_contact_force(i));
            }
            else
            {
                block_id = problem.AddResidualBlock(
                    cost_function_object_torque,
                    NULL,
                    object_loader.mutable_config_pino(i),
                    object_loader.mutable_config_pino(i - 1),
                    object_loader.mutable_config_pino(i - 2),
                    object_loader.mutable_config_contact(i),
                    object_loader.mutable_config_keypoints(0),
                    object_loader.mutable_contact_force(i));
            }
            cost_term_is_added = true;
            // Save block information for later evaluation
            std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | o_torq ";
            residual_block_names.push_back(block_name.str());
            residual_block_types.push_back("o_torq");
            residual_block_ids.push_back(block_id);
        }
        PrintStringToStdout("o_torq ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        // ------------------------------------------------------------------
        // Smoothness of the trajectories
        // ------------------------------------------------------------------

        // Reduce the person's joint space velocities
        // NOTE: we use v_q and a_q instead of spatial velocities
        if (i >= 1 && w_p_jvel > 0)
        {
            ceres::CostFunction *cost_function_velocity =
                CostFunctorPersonVelocity::Create(
                    i,
                    &person_loader,
                    (bool)update_6d_basis_only);
            ceres::LossFunction *loss_function_velocity =
                new ceres::ScaledLoss(
                    NULL, w_p_jvel, ceres::TAKE_OWNERSHIP);
            block_id = problem.AddResidualBlock(
                cost_function_velocity,
                loss_function_velocity,
                person_loader.mutable_config(i),
                person_loader.mutable_config(i - 1));
            cost_term_is_added = true;
            // Save block information for later evaluation
            std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | p_jvel ";
            residual_block_names.push_back(block_name.str());
            residual_block_types.push_back("p_jvel");
            residual_block_ids.push_back(block_id);
        }
        PrintStringToStdout("p_jvel ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        // Reduce the person joints' cartesian velocities
        if (i >= 1 && w_p_cvel > 0)
        {
            // Eigen::VectorXi list_joints_to_smooth(12);
            // list_joints_to_smooth << 1,2,3,5,6,7,15,16,17,20,21,22;
            //Eigen::VectorXi list_joints_to_smooth(8);
            //list_joints_to_smooth << 2,3,6,7,16,17,21,22;
            Eigen::VectorXi list_joints_to_smooth(4);
            list_joints_to_smooth << 2,6,16,21; // elbows and knees
            ceres::CostFunction *cost_function_cartesian_vel =
                CostFunctorSmoothingCartesianVelocity::Create(
                    i,
                    &person_loader,
                    list_joints_to_smooth);
            ceres::LossFunction *loss_function_cartesian_vel =
                new ceres::ScaledLoss(
                    NULL, w_p_cvel, ceres::TAKE_OWNERSHIP);
            block_id = problem.AddResidualBlock(
                cost_function_cartesian_vel,
                loss_function_cartesian_vel,
                person_loader.mutable_config(i),
                person_loader.mutable_config(i - 1));
            cost_term_is_added = true;
            // Save block information for later evaluation
            std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | p_cvel ";
            residual_block_names.push_back(block_name.str());
            residual_block_types.push_back("p_cvel");
            residual_block_ids.push_back(block_id);
        }
        PrintStringToStdout("p_cvel ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        // Reduce the person's joint space accelerations
        if (i >= 2 && w_p_jacc > 0)
        {
            ceres::CostFunction *cost_function_acceleration =
                CostFunctorPersonAcceleration::Create(
                    i,
                    &person_loader,
                    (bool)update_6d_basis_only);
            ceres::LossFunction *loss_function_acceleration =
                new ceres::ScaledLoss(
                    NULL, w_p_jacc, ceres::TAKE_OWNERSHIP);
            block_id = problem.AddResidualBlock(
                cost_function_acceleration,
                loss_function_acceleration,
                person_loader.mutable_config(i),
                person_loader.mutable_config(i - 1),
                person_loader.mutable_config(i - 2));
            cost_term_is_added = true;
            // Save block information for later evaluation
            std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | p_jacc ";
            residual_block_names.push_back(block_name.str());
            residual_block_types.push_back("p_jacc");
            residual_block_ids.push_back(block_id);
        }
        PrintStringToStdout("p_jacc ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        // Reduce the object's joint space velocities
        if (i >= 1 && w_o_jvel > 0 && (!virtual_object))
        {
            ceres::CostFunction *cost_function_obj_vel =
                CostFunctorObjectVelocity::Create(
                    i, &object_loader);
            ceres::LossFunction *loss_function_obj_vel =
                new ceres::ScaledLoss(
                    NULL, w_o_jvel, ceres::TAKE_OWNERSHIP);
            block_id = problem.AddResidualBlock(
                cost_function_obj_vel,
                loss_function_obj_vel,
                object_loader.mutable_config_pino(i),
                object_loader.mutable_config_pino(i - 1));
            cost_term_is_added = true;
            // Save block information for later evaluation
            std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | o_jvel ";
            residual_block_names.push_back(block_name.str());
            residual_block_types.push_back("o_jvel");
            residual_block_ids.push_back(block_id);
        }
        PrintStringToStdout("o_jvel ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        // Reduce the object's joint space accelerations
        if (i >= 2 && w_o_jacc > 0 && (!virtual_object))
        {
            // exclude the last time step since otherwise v_q is unconstrainted
            ceres::CostFunction *cost_function_obj_acc =
                CostFunctorObjectAcceleration::Create(
                    i, &object_loader);
            ceres::LossFunction *loss_function_obj_acc =
                new ceres::ScaledLoss(
                    NULL, w_o_jacc, ceres::TAKE_OWNERSHIP);
            block_id = problem.AddResidualBlock(
                cost_function_obj_acc,
                loss_function_obj_acc,
                object_loader.mutable_config_pino(i),
                object_loader.mutable_config_pino(i - 1),
                object_loader.mutable_config_pino(i - 2));
            cost_term_is_added = true;
            // Save block information for later evaluation
            std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | o_jacc ";
            residual_block_names.push_back(block_name.str());
            residual_block_types.push_back("o_jacc");
            residual_block_ids.push_back(block_id);
        }
        PrintStringToStdout("o_jacc ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;


        // The "contact type" term:
        // Reduce the 3D cartesian velocities of a set of person joints.
        // The weights on the different joints depends on the joint types:
        // it put a high weight for the joints in "fixed" contact with
        // or ground, and put a low weight for the human joints in
        // "sliding" contacts with object/ground
        if (i >= 1 && w_c_type_f > 0 && w_c_type_s > 0)
        {
            Eigen::Vector2d weights_c_type;
            weights_c_type << w_c_type_f, w_c_type_s;
            ceres::CostFunction *cost_function_contact_temporal =
                CostFunctorContactTemporal::Create(
                    i,
                    &person_loader,
                    &object_loader,
                    weights_c_type,
                    (bool)update_6d_basis_only,
                    (bool)smooth_oc,
                    &has_object_contact,
                    virtual_object);
            // the pointer == NULL means there is no contact joint at timestep i
            if (cost_function_contact_temporal != NULL)
            {
                // Three conditions for adding object config as variables:
                // 1. We explicitely model the object (virtual_object is false).
                // 2. There is at least one joint which is in contact with object at frame i and frame i-1 (has_object_contact will be set to true in this case).
                // 3. We plan to smooth the object contact joints (smooth_oc is true).
                if (!virtual_object && has_object_contact && smooth_oc)
                {
                    block_id = problem.AddResidualBlock(
                        cost_function_contact_temporal,
                        NULL, // loss_function_contact_temporal,
                        person_loader.mutable_config(i),
                        person_loader.mutable_config(i - 1),
                        object_loader.mutable_config_pino(i),
                        object_loader.mutable_config_pino(i - 1));
                }
                else
                {
                    block_id = problem.AddResidualBlock(
                        cost_function_contact_temporal,
                        NULL, // loss_function_contact_temporal,
                        person_loader.mutable_config(i),
                        person_loader.mutable_config(i - 1));
                }
                cost_term_is_added = true;
                // Save block information for later evaluation
                std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | c_type ";
                residual_block_names.push_back(block_name.str());
                residual_block_types.push_back("c_type");
                residual_block_ids.push_back(block_id);
            }
        }
        PrintStringToStdout("c_type ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        // Smooth the object and the ground contact forces
        if (i >= 3 && w_oc_for > 0)
        {
            ceres::CostFunction *cost_function_smoothing_object_force =
                    CostFunctorSmoothingObjectForce::Create(
                        i,
                        &person_loader,
                        &object_loader);
            if (cost_function_smoothing_object_force!= NULL)
            {
                ceres::LossFunction *loss_function_smoothing_object_force =
                    new ceres::ScaledLoss(
                        NULL, w_oc_for, ceres::TAKE_OWNERSHIP);
                block_id = problem.AddResidualBlock(
                    cost_function_smoothing_object_force,
                    loss_function_smoothing_object_force,
                    object_loader.mutable_contact_force(i),
                    object_loader.mutable_contact_force(i - 1));
                cost_term_is_added = true;
                // Save block information for later evaluation
                std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | oc_for ";
                residual_block_names.push_back(block_name.str());
                residual_block_types.push_back("oc_for");
                residual_block_ids.push_back(block_id);
            }
        }
        PrintStringToStdout("oc_for ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        if (i >= 3 && w_gc_for > 0)
        {
            ceres::CostFunction *cost_function_smoothing_ground_force =
                    CostFunctorSmoothingGroundForce::Create(
                        i, &person_loader);
            if (cost_function_smoothing_ground_force!= NULL)
            {
                ceres::LossFunction *loss_function_smoothing_ground_force =
                    new ceres::ScaledLoss(
                        NULL, w_gc_for, ceres::TAKE_OWNERSHIP);
                block_id = problem.AddResidualBlock(
                    cost_function_smoothing_ground_force,
                    loss_function_smoothing_ground_force,
                    person_loader.mutable_ground_friction(i),
                    person_loader.mutable_ground_friction(i - 1));
                cost_term_is_added = true;
                // Save block information for later evaluation
                std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | gc_for ";
                residual_block_names.push_back(block_name.str());
                residual_block_types.push_back("gc_for");
                residual_block_ids.push_back(block_id);
            }
        }
        PrintStringToStdout("gc_for ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        // Reduce the magnitude of object contact forces
        if ( i>=2 && w_oc_flm > 0)
        {
            ceres::CostFunction *cost_function_limit_object_force =
                CostFunctorLimitObjectForce::Create(
                    i,
                    &person_loader,
                    &object_loader);
            if (cost_function_limit_object_force!= NULL)
            {
                ceres::LossFunction *loss_function_limit_object_force =
                    new ceres::ScaledLoss(
                        NULL, w_oc_flm, ceres::TAKE_OWNERSHIP);
                block_id = problem.AddResidualBlock(
                    cost_function_limit_object_force,
                    loss_function_limit_object_force,
                    object_loader.mutable_contact_force(i));
                cost_term_is_added = true;
                // Save block information for later evaluation
                std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | oc_flm ";
                residual_block_names.push_back(block_name.str());
                residual_block_types.push_back("oc_flm");
                residual_block_ids.push_back(block_id);
            }
        }
        PrintStringToStdout("oc_flm ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        // Reduce the magnitude of ground contact forces
        if ( i>=2 && w_gc_flm > 0)
        {
            ceres::CostFunction *cost_function_limit_ground_force =
                CostFunctorLimitGroundForce::Create(
                    i,
                    &person_loader);
            if (cost_function_limit_ground_force!= NULL)
            {
                ceres::LossFunction *loss_function_limit_ground_force =
                    new ceres::ScaledLoss(
                        NULL, w_gc_flm, ceres::TAKE_OWNERSHIP);
                block_id = problem.AddResidualBlock(
                    cost_function_limit_ground_force,
                    loss_function_limit_ground_force,
                    person_loader.mutable_ground_friction(i));
                cost_term_is_added = true;
                // Save block information for later evaluation
                std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | gc_flm ";
                residual_block_names.push_back(block_name.str());
                residual_block_types.push_back("gc_flm");
                residual_block_ids.push_back(block_id);
            }
        }
        PrintStringToStdout("gc_flm ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        // ------------------------------------------------------------------
        // Other terms (not mentioned in the first submission)
        // ------------------------------------------------------------------

        // Person center of mass term
        if (add_p_com > 0)
        {
            ceres::CostFunction *cost_function_p_com =
                CostFunctorPersonCenterOfMass::Create(
                    i,
                    &person_loader,
                    radius_com);
            if (cost_function_p_com!= NULL)
            {
                // ceres::LossFunction *loss_function_p_com =
                //     new ceres::ScaledLoss(NULL, add_p_com, ceres::TAKE_OWNERSHIP);
                block_id = problem.AddResidualBlock(
                    cost_function_p_com,
                    NULL,
                    person_loader.mutable_config(i));
                cost_term_is_added = true;
                // Save block information for later evaluation
                std::ostringstream block_name; block_name <<  "| #" << std::setw(3) << std::setfill('0') << i << " | p_com ";
                residual_block_names.push_back(block_name.str());
                residual_block_types.push_back("p_com");
                residual_block_ids.push_back(block_id);
            }
        }
        PrintStringToStdout("p_com ", cost_term_is_added);
        std::cout << "| "; cost_term_is_added = false;

        std::cout << std::endl;
    }
    std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
}
