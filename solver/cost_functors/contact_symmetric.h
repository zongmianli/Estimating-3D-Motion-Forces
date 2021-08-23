#ifndef __CONTACT_SYMMETRIC_H__
#define __CONTACT_SYMMETRIC_H__

#include <ceres/ceres.h>

#include "../dataloader/dataloader_person.h"
#include "../dataloader/dataloader_object.h"

using namespace Eigen;
using namespace std;
using namespace pinocchio;

struct CostFunctorContactSymmetric
{
  typedef ceres::DynamicNumericDiffCostFunction<CostFunctorContactSymmetric>
      CostFunctionContactSymmetric;

  CostFunctorContactSymmetric(int k_left_fingers, int k_right_fingers)
  {
      k_left_fingers_ = k_left_fingers;
      k_right_fingers_ = k_right_fingers;
  }

  bool operator()(double const *const *parameters, double *residual) const
  {
    // get q_keypoint
    const double *const q_keypoint = *parameters;
    // get q_contact
    const double *const q_contact = *(parameters + 1);
    // average left and right finger contact points and minus center position of the stick
    residual[0] = q_contact[k_left_fingers_] + q_contact[k_right_fingers_] - q_keypoint[0];
    return true;
  }

  static ceres::CostFunction *Create(int i,
                                     DataloaderPerson *person_loader,
                                     DataloaderObject *object_loader)
  {
      // check if both left and right hand are in contact with object. 
      // If this is the case, we penalize the distance between the center
      // of left and right hand contact points and the center of barbell bar
      VectorXi contact_states = person_loader->get_contact_states_column(i);
      // if left fingers (18) and right fingers (23) are both in contact
      int j_left_fingers = 18;
      int j_right_fingers = 23;
      if (contact_states(j_left_fingers) == 1 && contact_states(j_right_fingers) == 1)
      {
          // get the id of contact points corresponding to two hands
          VectorXi contact_mapping = person_loader->get_contact_mapping();
          int k_left_fingers = contact_mapping(j_left_fingers) - 1;
          int k_right_fingers = contact_mapping(j_right_fingers) - 1;
          CostFunctorContactSymmetric *cost_functor =
              new CostFunctorContactSymmetric(k_left_fingers, k_right_fingers);
          CostFunctionContactSymmetric *cost_function =
              new CostFunctionContactSymmetric(cost_functor);
          // add parameter blocks
          cost_function->AddParameterBlock(1); // a single keypoint (endpoint)
          int nq_contact = object_loader->get_nq_contact();
          cost_function->AddParameterBlock(nq_contact);
          // number of residuals
          cost_function->SetNumResiduals(1); // one residual
          return cost_function;
      }
      return NULL;
  }

private:
  int k_left_fingers_;
  int k_right_fingers_;
};

#endif // ifndef __CONTACT_SYMMETRIC_H__
