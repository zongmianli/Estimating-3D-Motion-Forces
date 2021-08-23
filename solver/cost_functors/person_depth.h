#ifndef __PERSON_DEPTH_H__
#define __PERSON_DEPTH_H__

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "../dataloader/dataloader_person.h"

using namespace std;

struct CostFunctorPersonDepth
{
  typedef ceres::DynamicNumericDiffCostFunction<CostFunctorPersonDepth>
      CostFunctionPersonDepth;

  CostFunctorPersonDepth(int i,
                         DataloaderPerson *person_loader)
  {
    i_ = i; // num of time step
    person_loader_ = person_loader;
    VectorXd config = person_loader_->get_config_column(i_);
    depth_guess_ = config(2);
  }

  bool operator()(double const *const *parameters, double *residual) const
  {
    //cout << "ckpt: depth" << endl;
    // get current depth
    double depth = (*parameters)[2];
    residual[0] = depth - depth_guess_;
    return true;
  }

  static ceres::CostFunction *Create(int i,
                                     DataloaderPerson *person_loader)
  {
    CostFunctorPersonDepth *cost_functor =
        new CostFunctorPersonDepth(i, person_loader);
    CostFunctionPersonDepth *cost_function =
        new CostFunctionPersonDepth(cost_functor);
    // 6D vector representing human basis pose
    cost_function->AddParameterBlock(6);
    // number of residuals
    cost_function->SetNumResiduals(1);
    return cost_function;
  }

private:
  int i_;
  double depth_guess_;
  DataloaderPerson *person_loader_;
};

#endif // ifndef __PERSON_DEPTH_H__
