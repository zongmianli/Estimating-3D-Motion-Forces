#ifndef __SMOOTHING_CARTESIAN_VELOCITY_H__
#define __SMOOTHING_CARTESIAN_VELOCITY_H__

#include <ceres/ceres.h>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "../dataloader/dataloader_person.h"

using namespace Eigen;
using namespace std;
using namespace pinocchio;

struct CostFunctorSmoothingCartesianVelocity
{
  typedef ceres::DynamicNumericDiffCostFunction<CostFunctorSmoothingCartesianVelocity>
      CostFunctionSmoothingCartesianVelocity;

  CostFunctorSmoothingCartesianVelocity(int i,
                                        DataloaderPerson *person_loader,
                                        VectorXi list_joints_to_smooth)
  {
      i_ = i;
      person_loader_ = person_loader;
      nq_ = person_loader_->get_nq();
      nq_pino_ = person_loader_->get_nq_pino();
      dt_ = person_loader->get_dt();
      list_joints_to_smooth_ = list_joints_to_smooth;
      num_joints_to_smooth_ = list_joints_to_smooth.rows();
  }

  bool operator()(double const *const *parameters, double *residual) const
  {
    // get q
    const double *const q = *parameters;
    // get q_minus1
    const double *const q_minus1 = *(parameters + 1);

    VectorXd q_mat, q_minus1_mat;
    q_mat = Eigen::Map<const VectorXd>(q,nq_,1);
    q_minus1_mat = Eigen::Map<const VectorXd>(q_minus1,nq_,1);
    
    person_loader_->UpdateConfigPino(i_, q_mat);
    person_loader_->UpdateConfigPino(i_-1, q_minus1_mat);

    VectorXd q_pino, q_pino_minus1;
    q_pino = Eigen::Map<const VectorXd>(person_loader_->mutable_config_pino(i_),nq_pino_,1);
    q_pino_minus1 = Eigen::Map<const VectorXd>(person_loader_->mutable_config_pino(i_-1),nq_pino_,1);
    assert(q_pino.size() == person_loader_->get_nq_pino() && "q_pino of wrong dimension");
    assert(q_pino_minus1.size() == person_loader_->get_nq_pino() && "q_pino_minus1 of wrong dimension");

    pinocchio::forwardKinematics(person_loader_->model_, person_loader_->data_, q_pino);
    int joint_id;
    for (int i = 0; i < num_joints_to_smooth_; i++)
    {
        joint_id = list_joints_to_smooth_(i);
        const Vector3d &joint_position_3d = person_loader_->data_.oMi[(size_t)(joint_id + 1)].translation();
        for (int k = 0; k < 3; k++)
        {
            residual[3 * i + k] = joint_position_3d(k);
        }
    }

    pinocchio::forwardKinematics(person_loader_->model_, person_loader_->data_, q_pino_minus1);
    for (int i = 0; i < num_joints_to_smooth_; i++)
    {
        joint_id = list_joints_to_smooth_(i);
        const Vector3d &joint_position_3d_minus1 = person_loader_->data_.oMi[(size_t)(joint_id + 1)].translation();
        for (int k = 0; k < 3; k++)
        {
            residual[3 * i + k] = (residual[3 * i + k] - joint_position_3d_minus1(k)) / dt_;
        }
    }
    return true;
  }

  static ceres::CostFunction *Create(int i,
                                     DataloaderPerson *person_loader,
                                     VectorXi list_joints_to_smooth)
  {
      CostFunctorSmoothingCartesianVelocity *cost_functor =
          new CostFunctorSmoothingCartesianVelocity(i, person_loader, list_joints_to_smooth);
      CostFunctionSmoothingCartesianVelocity *cost_function =
          new CostFunctionSmoothingCartesianVelocity(cost_functor);

      int nq_person = person_loader->get_nq();
      cost_function->AddParameterBlock(nq_person);
      cost_function->AddParameterBlock(nq_person);
      int num_joints_to_smooth = list_joints_to_smooth.rows();
      cost_function->SetNumResiduals(3 * num_joints_to_smooth);
      return cost_function;
  }

private:
  int i_;
  int nq_;
  int nq_pino_;
  int num_joints_to_smooth_;
  double dt_;
  VectorXi list_joints_to_smooth_;
  DataloaderPerson *person_loader_;
};

#endif // ifndef __SMOOTHING_CARTESIAN_VELOCITY_H__
