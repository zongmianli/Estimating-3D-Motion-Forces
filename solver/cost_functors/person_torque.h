#ifndef __PERSON_TORQUE_H__
#define __PERSON_TORQUE_H__

#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#include <math.h>
#include <ceres/ceres.h>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/frames.hpp"


#include "../dataloader/dataloader_person.h"
#include "../dataloader/dataloader_object.h"

using namespace Eigen;
using namespace std;
using namespace pinocchio;

struct CostFunctorPersonTorque
{
  typedef ceres::DynamicNumericDiffCostFunction<CostFunctorPersonTorque>
  CostFunctionPersonTorque;
  typedef Matrix<double, 6, 1> Vector6d;
  typedef pinocchio::Data::Matrix6x Matrix6x;
  
  CostFunctorPersonTorque(int i,
                          DataloaderPerson *person_loader,
                          DataloaderObject *object_loader,
                          const VectorXd & weights);
  
  bool Evaluate(double const *const * parameters,
                double * residual,
                double ** jacobians) const;
  
  bool operator()(double const *const *parameters, double *residual) const
  {
      return Evaluate(parameters,residual,NULL);
  }
  
  static ceres::CostFunction *Create(int i,
                                     DataloaderPerson *person_loader,
                                     DataloaderObject *object_loader,
                                     const VectorXd &weights,
                                     bool *has_object_contact,
                                     bool *has_ground_contact);
  
  bool has_object_contact() const
  {
    return num_object_contact_joints_>0;
  }
  
  bool has_ground_contact() const
  {
    return num_ground_contact_joints_>0;
  }
  
private:
  int i_;
  int nq_person_;
  VectorXd weights_;
  DataloaderPerson *person_loader_;
  DataloaderObject *object_loader_;
  int num_object_contact_joints_;
  int num_ground_contact_joints_;
  vector<int> object_contact_joints_;
  vector<int> ground_contact_joints_;
  VectorXi contact_mapping_;
  double dt_;
  double dt_square_;
  // double friction_angle_;
  Matrix6x friction_cone_generators_;
  Matrix6x friction_cone_generators_left_foot_;
  Matrix6x friction_cone_generators_right_foot_;
  bool debug_;
};

#endif // ifndef __PERSON_TORQUE_H__
