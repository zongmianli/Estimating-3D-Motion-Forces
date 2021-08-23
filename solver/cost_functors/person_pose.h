#ifndef __PERSON_POSE_HPP__
#define __PERSON_POSE_HPP__

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "../dataloader/dataloader_person.h"
#include "../pose_prior_gmm.h"

using namespace Eigen;
using namespace std;
using namespace pinocchio;

struct CostFunctorPersonPose
{
  typedef ceres::DynamicNumericDiffCostFunction<CostFunctorPersonPose>
      CostFunctionPersonPoseMog;

  CostFunctorPersonPose(int i,
                        DataloaderPerson *person_loader,
                        PosePriorGmm *pose_prior)
  {
    i_ = i; // num of time step
    person_loader_ = person_loader;
    pose_prior_ = pose_prior;
  }

  bool operator()(double const *const *parameters, double *residual) const
  {
    const double *const q = *parameters;
    Matrix<double, 75, 1> q_mat(q);
    // compute probability density
    VectorXd log_likelihood = pose_prior_->ComputeLogLikelihood(q_mat);
    for (int i = 0; i < 70; i++)
    {
      residual[i] = log_likelihood(i);
    }
    return true;
  }

  static ceres::CostFunction *Create(int i,
                                     DataloaderPerson *person_loader,
                                     PosePriorGmm *pose_prior)
  {
    CostFunctorPersonPose *cost_functor =
        new CostFunctorPersonPose(i, person_loader, pose_prior);
    CostFunctionPersonPoseMog *cost_function =
        new CostFunctionPersonPoseMog(cost_functor);
    // person config parameters
    int nq = person_loader->get_nq();
    cost_function->AddParameterBlock(nq);
    // number of residuals
    cost_function->SetNumResiduals(70);
    return cost_function;
  }

private:
  int i_;
  DataloaderPerson *person_loader_;
  PosePriorGmm *pose_prior_;
};

#endif // ifndef __PERSON_POSE_HPP__
