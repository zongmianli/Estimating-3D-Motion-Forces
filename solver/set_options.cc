#include "solver.h"

void SetOptions(ceres::Problem &problem,
                int stage,
                int timestep_begin,
                int timestep_end,
                const Eigen::VectorXd &stage_options,
                const bool fix_object_contact_points,
                const Eigen::Vector2d &handle_length_bounds,
                DataloaderPerson &person_loader,
                DataloaderObject &object_loader,
                DataloaderObject &ground_loader)
{
    int freeze_6d_basis = (int)stage_options(3);
    int ground_plane_constrained = (int)stage_options(4); // 1 or 2 (for scythe)
    int freeze_ground_rotation = (int)stage_options(5); // -1
    int freeze_handle_length = (int)stage_options(9); //

    bool freeze_object_config_pino = (int)stage_options(10);
    bool freeze_object_contact = (int)stage_options(11);

    std::cout << "setting Solver Options ... " << std::endl;
    std::cout << "freeze_6d_basis " << freeze_6d_basis << std::endl;
    std::cout << "ground_plane_constrained " << ground_plane_constrained << std::endl;
    std::cout << "freeze_ground_rotation " << freeze_ground_rotation << std::endl;
    std::cout << "freeze_handle_length " << freeze_handle_length << std::endl;
    std::cout << "fix_object_contact_points " << fix_object_contact_points << std::endl;
    std::cout << "handle_length_bounds == " << std::endl << handle_length_bounds.transpose() << std::endl;

    std::cout << "freeze_object_config_pino == " << freeze_object_config_pino << std::endl;
    std::cout << "freeze_object_contact == " << freeze_object_contact << std::endl;


    switch (stage)
    {
    case 1:
    {
        break;
    }
    case 2:
    {
        break;
    }
    case 3:
    {
        break;
    }
    case 4:
    {
        //set constant parameter blocks
        for (int i = timestep_begin-2; i <= timestep_end; i++)
        {
            if (i >= 0)
            {
                double * parameters = person_loader.mutable_config(i);
                if (problem.HasParameterBlock(parameters))
                {
                    problem.SetParameterBlockConstant(parameters);
                }
            }
        }

        break;
    }
    case 5:
    {
        double * parameters;
        // // TODO: should we really freeze object config?
        // for (int i = timestep_begin - 2; i <= timestep_end; i++)
        // {
        //     if (i >= 0)
        //     {
        //         parameters = object_loader.mutable_config_pino(i);
        //         if (problem.HasParameterBlock(parameters))
        //         {
        //             problem.SetParameterBlockConstant(parameters);
        //             std::cout << "object_config_pino freeze at frame " << i << std::endl;
        //         }
        //     }
        // }

        // set constant the prefix parameters blocks
        for (int i = timestep_begin - 2; i < timestep_begin; i++)
        {
            if (i >= 0)
            {
                parameters = person_loader.mutable_config(i);
                if (problem.HasParameterBlock(parameters))
                {
                    problem.SetParameterBlockConstant(parameters);
                    //std::cout << "* person_loader.mutable_config(i): i = " << i << std::endl;
                }
                parameters = object_loader.mutable_config_contact(i);
                if (problem.HasParameterBlock(parameters))
                {
                    problem.SetParameterBlockConstant(parameters);
                    //std::cout << "* object_loader.mutable_config_contact(i): i = " << i << std::endl;
                }
            }
        }

        // freeze ground 6D pose
        parameters = ground_loader.mutable_config(0);
        if (problem.HasParameterBlock(parameters))
        {
            problem.SetParameterBlockConstant(parameters);
        }
        break;
    }
    case 6:
    {
        // set constant parameter blocks
        double *parameters;
        for (int i = timestep_begin - 2; i <= timestep_end; i++)
        {
            if (i >= 0)
            {
                parameters = person_loader.mutable_config(i);
                if (problem.HasParameterBlock(parameters))
                {
                    problem.SetParameterBlockConstant(parameters);
                }
                parameters = object_loader.mutable_config_pino(i);
                if (problem.HasParameterBlock(parameters))
                {
                    problem.SetParameterBlockConstant(parameters);
                }
                parameters = object_loader.mutable_config_contact(i);
                if (problem.HasParameterBlock(parameters))
                {
                    problem.SetParameterBlockConstant(parameters);
                }
            }
        }

        // set constant the prefix parameters blocks
        for (int i = timestep_begin - 2; i < timestep_begin; i++)
        {
            if (i >= 0)
            {
                parameters = object_loader.mutable_contact_force(i);
                if (problem.HasParameterBlock(parameters))
                {
                    problem.SetParameterBlockConstant(parameters);
                }
                parameters = person_loader.mutable_ground_friction(i);
                if (problem.HasParameterBlock(parameters))
                {
                    problem.SetParameterBlockConstant(parameters);
                }
            }
        }
        break;
    }
    case 7:
        double *parameters;
        // set constant the prefix parameters blocks
        for (int i = timestep_begin - 2; i < timestep_begin; i++)
        {
            if (i >= 0)
            {
                parameters = person_loader.mutable_config(i);
                if (problem.HasParameterBlock(parameters))
                {
                    problem.SetParameterBlockConstant(parameters);
                }
                parameters = object_loader.mutable_config_pino(i);
                if (problem.HasParameterBlock(parameters))
                {
                    problem.SetParameterBlockConstant(parameters);
                }
                parameters = object_loader.mutable_config_contact(i);
                if (problem.HasParameterBlock(parameters))
                {
                    problem.SetParameterBlockConstant(parameters);
                }
            }
        }

        // // Freeze toes and fingers
        // std::vector<int> joints_to_freeze = {4, 8, 18, 23}; // toes and fingers
        // FreezeJoints(problem,
        //              timestep_begin,
        //              timestep_end,
        //              person_loader,
        //              joints_to_freeze);

        // freeze ground 6D pose
        parameters = ground_loader.mutable_config(0);
        if (problem.HasParameterBlock(parameters))
        {
            problem.SetParameterBlockConstant(parameters);
        }
        break;
    }

    if (freeze_6d_basis == 1)
    {
        // set up contact entries
        std::vector<int> constant_entries;
        for (int k=0; k<3; k++)
        {
            constant_entries.push_back(k);
        }
        // set parameterizations
        for (int i = 0; i < person_loader.get_nt(); i++)
        {
            double * parameters = person_loader.mutable_config(i);
            if (problem.HasParameterBlock(parameters))
            {
                ceres::SubsetManifold *parameterization_constant_entries =
                    new ceres::SubsetManifold(75, constant_entries);
                problem.SetManifold(parameters,
                                    parameterization_constant_entries);
            }
        }
    }

    if (ground_plane_constrained == 1)
    {
        // freeze ground x, y rotations
        std::vector<int> constant_entries; 
        constant_entries.push_back(0); // x cooridinate
        constant_entries.push_back(2); // z cooridinate 
        constant_entries.push_back(4); // rotation along axis y
        if (freeze_ground_rotation == 1)
        {
            constant_entries.push_back(3); // rotation along axis x
            constant_entries.push_back(5); // rotation along axis z
        }
        else if (freeze_ground_rotation == 2)
        {
            // free z rotation
            constant_entries.push_back(3); // rotation along axis x
        }
        double * parameters = ground_loader.mutable_config(0);
        if (problem.HasParameterBlock(parameters))
        {
            ceres::SubsetManifold *parameterization_constant_entries =
                new ceres::SubsetManifold(6, constant_entries);
            problem.SetManifold(parameters,
                                parameterization_constant_entries);
        }
    }


    // Set feasible range for object contact points
    int nq_contact_object = object_loader.get_nq_contact();
    for (int i = 0; i < object_loader.get_nt(); i++)
    {
        double * parameters = object_loader.mutable_config_contact(i);
        if (problem.HasParameterBlock(parameters))
        {
            for(int k=0; k<nq_contact_object; k++)
            {
                problem.SetParameterLowerBound(
                    parameters,
                    k,
                    0);
                problem.SetParameterUpperBound(
                    parameters,
                    k,
                    *object_loader.mutable_config_keypoints(0));
            }
        }
        if (i==0 && fix_object_contact_points)
        {
            break;
        }
    }

    // Set feasible range for the end of the object's stick handle
    double * parameters = object_loader.mutable_config_keypoints(0);
    if (problem.HasParameterBlock(parameters))
    {
        if (freeze_handle_length == 1)
        {
            problem.SetParameterBlockConstant(parameters);
        }
        else
        {
            for(int k=0; k<object_loader.get_nq_keypoints(); k++)
            {
                problem.SetParameterLowerBound(
                    object_loader.mutable_config_keypoints(0),
                    k,
                    handle_length_bounds(0));
                problem.SetParameterUpperBound(
                    object_loader.mutable_config_keypoints(0),
                    k,
                    handle_length_bounds(1));
            }
        }
    }

    // Freeze object configuration
    if (freeze_object_config_pino == 1)
    {
        for (int i = 0; i < object_loader.get_nt(); i++)
        {
            double * parameters = object_loader.mutable_config_pino(i);
            if (problem.HasParameterBlock(parameters))
            {
                problem.SetParameterBlockConstant(parameters);
                //std::cout << "object_config_pino freeze at frame " << i << std::endl;
            }
        }
    }

    // Freeze object config contact
    if (freeze_object_contact == 1)
    {
        for (int i = 0; i < object_loader.get_nt(); i++)
        {
            double * parameters = object_loader.mutable_config_contact(i);
            if (problem.HasParameterBlock(parameters))
            {
                problem.SetParameterBlockConstant(parameters);
            }
        }
    }
}

void FreezeJoints(ceres::Problem &problem,
                  int timestep_begin,
                  int timestep_end,
                  DataloaderPerson &person_loader,
                  std::vector<int> &joints_to_freeze)
{
    int num_joints_to_freeze = (int)joints_to_freeze.size();
    Eigen::MatrixXd person_decoration = person_loader.get_decoration();

    int jid; // joint id (from 0 to 23)
    int joint_type; // joint type
    int joint_index; // index of the position from which joint j's configuration starts

    std::vector<int> constant_entries;
    for (int n=0; n<num_joints_to_freeze; n++)
    {
        jid = joints_to_freeze[(size_t)n];
        joint_type = person_decoration(jid, 2);
        joint_index = person_decoration(jid, 3);
        if (joint_type == 1)
        { // free-floating joint: 6 entries
            for (int k=0; k<6; k++)
            {
                constant_entries.push_back(joint_index+k);
            }
        }
        else if (joint_type == 2)
        { // spherical joint: 3 entries
            for (int k=0; k<3; k++)
            {
                constant_entries.push_back(joint_index+k);
            }
        }
        else
        { // all other types of joint: 1 entry
            constant_entries.push_back(joint_index);
        }
    }

    // set parameterizations
    double * parameters;
    for (int i = timestep_begin; i <= timestep_end; i++)
    {
        parameters = person_loader.mutable_config(i);
        if (problem.HasParameterBlock(parameters))
        {
            ceres::SubsetManifold *parameterization_constant_entries =
                new ceres::SubsetManifold(75, constant_entries);
            problem.SetManifold(parameters,
                                parameterization_constant_entries);
        }
    }
}
