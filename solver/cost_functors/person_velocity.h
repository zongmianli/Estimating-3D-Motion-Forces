#ifndef __PERSON_VELOCITY_H__
#define __PERSON_VELOCITY_H__

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "../dataloader/dataloader_person.h"

using namespace std;

struct CostFunctorPersonVelocity
{
  typedef ceres::DynamicNumericDiffCostFunction<CostFunctorPersonVelocity>
      CostFunctionPersonVelocity;

  CostFunctorPersonVelocity(int i,
                             DataloaderPerson *person_loader,
                             bool update_6d_basis_only)
  {
    i_ = i; // num of time step
    person_loader_ = person_loader;
    dt_ = person_loader->get_dt();
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
    // penalize the basis movement more than the rest joints
    for (int i = 0; i < 6; i++)
    {
      residual[i] = 10*(q[i] - q_minus1[i]) / dt_;
    }
    if (nq_ > 6)
    {
      for (int i = 6; i < nq_; i++)
      {
        residual[i] = (q[i] - q_minus1[i]) / dt_;
      }
    }
    // penalizes the velocity along the depth direction
    residual[2] = residual[2]*10.0;
    return true;
  }

  static ceres::CostFunction *Create(int i,
                                     DataloaderPerson *person_loader,
                                     bool update_6d_basis_only)
  {
    CostFunctorPersonVelocity *cost_functor =
        new CostFunctorPersonVelocity(i, person_loader, update_6d_basis_only);
    CostFunctionPersonVelocity *cost_function =
        new CostFunctionPersonVelocity(cost_functor);
    // add parameter blocks
    int nq = cost_functor->get_nq(); // 6 or 75
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
  double dt_;
  DataloaderPerson *person_loader_;
};

#endif // ifndef __PERSON_VELOCITY_H__
