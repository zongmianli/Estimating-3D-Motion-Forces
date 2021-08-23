#ifndef __SOLVER_H__
#define __SOLVER_H__

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <glog/logging.h>

#include "camera.h"
#include "pose_prior_gmm.h"
#include "dataloader/dataloader_person.h"
#include "dataloader/dataloader_object.h"


void PrintStringToStdout(std::string str, bool print_str);

void InitLogging();

double Minimize(
    const Eigen::VectorXd &stage_weights,
    const Eigen::VectorXd &stage_options,
    const Eigen::VectorXd &ceres_options,
    DataloaderPerson &person_loader,
    DataloaderObject &object_loader,
    DataloaderObject &ground_loader,
    PosePriorGmm &pose_prior,
    Camera &camera);

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
    Camera &camera);

void SetOptions(
    ceres::Problem &problem,
    int stage,
    int timestep_begin,
    int timestep_end,
    const Eigen::VectorXd &stage_options,
    const bool fix_object_contact_points,
    const Eigen::Vector2d &handle_length_bounds,
    DataloaderPerson &person_loader,
    DataloaderObject &object_loader,
    DataloaderObject &ground_loader);

void FreezeJoints(
    ceres::Problem &problem,
    int timestep_begin,
    int timestep_end,
    DataloaderPerson &person_loader,
    std::vector<int> &joints_to_freeze);

#endif // #ifndef __SOLVER_H__
