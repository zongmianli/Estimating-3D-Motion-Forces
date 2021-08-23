#ifndef __CONTACT_GROUND_SPATIAL_H__
#define __CONTACT_GROUND_SPATIAL_H__

#include <ceres/ceres.h>

#include "../dataloader/dataloader_person.h"
#include "../dataloader/dataloader_object.h"

using namespace std;
using namespace Eigen;
using namespace pinocchio;

struct CostFunctorContactGroundSpatial
{
  typedef ceres::DynamicNumericDiffCostFunction<CostFunctorContactGroundSpatial> CostFunctionContactGroundSpatial;
  
  CostFunctorContactGroundSpatial(int i,
                                  DataloaderPerson *person_loader,
                                  DataloaderObject *ground_loader,
                                  bool update_6d_basis_only);
  
  bool operator()(double const *const *parameters, double *residual) const
  {
    return Evaluate(parameters,residual,NULL);
  }
  
  bool Evaluate(double const *const * parameters,
                double * residual,
                double ** jacobians) const;
  
  static ceres::CostFunction *Create(int i,
                                     DataloaderPerson *person_loader,
                                     DataloaderObject *ground_loader,
                                     bool update_6d_basis_only);
  
  int get_nq_person() const
  {
    return nq_person_;
  }

  int get_num_contact_points() const
  {
    return num_contact_points_;
  }
  
private:
  int i_;
  int nq_person_;
  VectorXd q_person_init_; // initial human configuration vector (75d)
  int num_contact_joints_;
  int num_contact_points_;
  int total_num_contact_points_;
  vector<int> contact_joints_;
  VectorXi contact_mapping_;
  DataloaderPerson *person_loader_;
  DataloaderObject *ground_loader_;
};

#endif // #ifndef __CONTACT_GROUND_SPATIAL_H__
