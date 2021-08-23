#include "solver.h"

// Initialize Google Logging.
// This function should only be called once.
void InitLogging()
{
    google::InitGoogleLogging("solver");
    FLAGS_logtostderr = 1; // print logging messages to stderr
}

double Minimize(
    const Eigen::VectorXd &stage_weights,
    const Eigen::VectorXd &stage_options,
    const Eigen::VectorXd &ceres_options,
    DataloaderPerson &person_loader,
    DataloaderObject &object_loader,
    DataloaderObject &ground_loader,
    PosePriorGmm &pose_prior,
    Camera &camera)
{
    // unwrap options
    bool print_summary = (bool) ceres_options(0);
    int stage = (int) ceres_options(1);
    int timestep_begin = (int) ceres_options(2);
    int timestep_end = (int) ceres_options(3);
    int max_num_iterations = (int) ceres_options(4);
    int trust_region_method = (int) ceres_options(5);
    double function_tolerance = ceres_options(6);
    bool update_state_every_iteration = (bool) ceres_options(7);
    bool evaluate_problem = (bool) ceres_options(8);
    bool fix_object_contact_points = (bool) ceres_options(9);
    Eigen::Vector2d handle_length_bounds = ceres_options.segment<2>(10);

    // initialize the problem
    LOG(INFO) << "*** stage " << stage << " ***" << std::endl;
    ceres::Problem ceres_problem;

    // set up the objective function to minimize
    std::vector<std::string> residual_block_names;
    std::vector<std::string> residual_block_types;
    std::vector<ceres::ResidualBlockId> residual_block_ids;

    BuildLossFunction(
        ceres_problem,
        residual_block_names,
        residual_block_types,
        residual_block_ids,
        timestep_begin,
        timestep_end,
        stage_weights,
        stage_options,
        fix_object_contact_points,
        person_loader,
        object_loader,
        ground_loader,
        pose_prior,
        camera);

    // set up stage options
    SetOptions(ceres_problem,
               stage,
               timestep_begin,
               timestep_end,
               stage_options,
               fix_object_contact_points,
               handle_length_bounds,
               person_loader,
               object_loader,
               ground_loader);

    // set up Ceres options
    ceres::Solver::Options options;
    options.max_num_iterations = max_num_iterations;
    options.function_tolerance = function_tolerance;
    options.update_state_every_iteration = update_state_every_iteration;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    switch (trust_region_method)
    {
    case 1:
        options.trust_region_strategy_type = ceres::DOGLEG;
        break;
    case 2:
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        break;
    }
    //options.dogleg_type = ceres::TRADITIONAL_DOGLEG;
    ceres::Solver::Summary ceres_summary;

    // solve the problem
    ceres::Solve(options, &ceres_problem, &ceres_summary);

    // print out Ceres summary
    if (print_summary)
    {
        LOG(INFO) << ceres_summary.FullReport() << "\n";
    }

    // Compute the final objective function value
    double final_cost = 0.0;
    ceres_problem.Evaluate(
        ceres::Problem::EvaluateOptions(), &final_cost, nullptr, nullptr, nullptr);

    if (evaluate_problem)
    {
        // Compute the each residual block values and print results
        ceres::Problem::EvaluateOptions evaluate_options;
        // Print the header line
        std::cout << "Solver Evaluation Results (per residual block and total costs)" << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "| time | term | value | residual size | residual values " << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << std::fixed << std::setprecision(2);

        std::vector<std::string> terms_to_evaluate;
        // Personalize the type of residual blocks to evaluate
        terms_to_evaluate.push_back("p_torq");

        // Evaluate per residual block costs
        std::string name;
        for (int n=0; n<residual_block_names.size(); n++)
        {
            if (std::find(terms_to_evaluate.begin(), terms_to_evaluate.end(), residual_block_types[n]) == terms_to_evaluate.end())
            {
                continue;
            }
            std::cout << residual_block_names[n];
            std::vector<ceres::ResidualBlockId> residual_block_id = {residual_block_ids[n]};
            evaluate_options.residual_blocks = residual_block_id;
            double cost = 0.0;
            std::vector<double> residuals;
            ceres_problem.Evaluate(evaluate_options, &cost, &residuals, nullptr, nullptr);
            std::cout << "| "  << std::setfill (' ') << std::setw(6) << cost << " | " << std::setw(2) << residuals.size() << " | ";

            for (int i=0; i<residuals.size(); i++)
            {
                std::cout << std::setw(6) << residuals[i] << " ";
                // Only display the first 10 numbers
                if (i>=10)
                {
                    std::cout << "... ";
                    break;
                }
            }
            std::cout << std::endl;
        }
        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "| Final cost : " << std::setprecision(6) << std::scientific << final_cost << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;
    }

    // Update person loader data
    person_loader.UpdateJoint2dReprojected(&camera);
    person_loader.UpdateKeypoint2dReprojected(&camera);
    person_loader.UpdateConfigPino();

    // Update ground loader data
    // Copy ground config at frame 0 to the rest frames
    Eigen::MatrixXd config_ground = ground_loader.get_config();
    for (int i=1; i<ground_loader.get_nt(); i++)
    {
        config_ground.col(i) = config_ground.col(0);
    }
    ground_loader.LoadConfig(config_ground, ground_loader.get_fps());
    ground_loader.UpdateConfigPino();

    // Update object loader data
    if (!object_loader.get_is_virtual_object())
    {
        // Update object config_ from config_pino_
        object_loader.UpdateConfig();

        // If fix_object_contact_points is true, we should
        // copy contact config at frame 0 to the rest frames
        if (fix_object_contact_points)
        {
            Eigen::MatrixXd object_config_contact = object_loader.get_config_contact();
            for (int i=1; i<object_loader.get_nt(); i++)
            {
                object_config_contact.col(i) = object_config_contact.col(0);
            }
            object_loader.LoadConfigContact(object_config_contact);
        }

        // Copy keypoint config at frame 0 to the rest frames
        Eigen::MatrixXd config_keypoints = object_loader.get_config_keypoints();
        // Print the optimal keypoint configuration
        std::cout << "Optimal keypoint configuration: " << config_keypoints.col(0).transpose() << std::endl;
        for (int i=1; i<object_loader.get_nt(); i++)
        {
            config_keypoints.col(i) = config_keypoints.col(0);
        }
        object_loader.LoadConfigKeypoints(config_keypoints);
        // Compute the 2D reprojection of object keypoints
        object_loader.UpdateKeypoint2dReprojected(&camera);
    }
    return final_cost;
}
