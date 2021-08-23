#ifndef __PERSON_ACCELERATION_H__
#define __PERSON_ACCELERATION_H__

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "../dataloader/dataloader_person.h"

using namespace std;

struct CostFunctorPersonAcceleration
{
  typedef ceres::DynamicNumericDiffCostFunction<CostFunctorPersonAcceleration>
      CostFunctionPersonAcceleration;

  CostFunctorPersonAcceleration(int i,
                             DataloaderPerson *person_loader,
                             bool update_6d_basis_only)
  {
    i_ = i; // num of time step
    person_loader_ = person_loader;
    double dt = person_loader->get_dt();
    dt_square_ = dt * dt;
    if (update_6d_basis_only)
    {
      nq_ = 6;
    }
    else
    {
      nq_ = person_loader_->get_nq();
    }
  }

  bool operator()(double const *const *parameters, double *residual) const
  {
    // get q
    const double *const q = *parameters;
    // get q_minus1
    const double *const q_minus1 = *(parameters + 1);
    // get q_minus2
    const double *const q_minus2 = *(parameters + 2);
    for (int i = 0; i < nq_; i++)
    {
      residual[i] = (q[i] - 2 * q_minus1[i] + q_minus2[i]) / dt_square_;
    }
    return true;
  }

  static ceres::CostFunction *Create(int i,
                                     DataloaderPerson *person_loader,
                                     bool update_6d_basis_only)
  {
    CostFunctorPersonAcceleration *cost_functor =
        new CostFunctorPersonAcceleration(i, person_loader, update_6d_basis_only);
    CostFunctionPersonAcceleration *cost_function =
        new CostFunctionPersonAcceleration(cost_functor);
    // person config parameters
    int nq = cost_functor->get_nq();
    cost_function->AddParameterBlock(nq);
    cost_function->AddParameterBlock(nq);
    cost_function->AddParameterBlock(nq);
    // number of residuals
    cost_function->SetNumResiduals(nq);
    return cost_function;
  }

  int get_nq()
  {
    return nq_;
  }

private:
  int i_;
  int nq_;
  double dt_square_;
  DataloaderPerson *person_loader_;
};

#endif // ifndef __PERSON_ACCELERATION_H__
