#ifndef __HORIZONTAL_OBJECT_H__
#define __HORIZONTAL_OBJECT_H__

#include <ceres/ceres.h>

#include "../dataloader/dataloader_object.h"

using namespace Eigen;
using namespace std;
using namespace pinocchio;

struct CostFunctorHorizontalObject
{
  typedef ceres::DynamicNumericDiffCostFunction<CostFunctorHorizontalObject>
      CostFunctionHorizontalObject;

  CostFunctorHorizontalObject(int i,
                              DataloaderObject *object_loader)
  {
      i_ = i;
      object_loader_ = object_loader;
      nq_pino_object_ = object_loader->get_nq_pino(); // 7: base translation + quaternion rotation
      nq_contact_ = object_loader->get_nq_contact(); // 2 or 3: left&right hand contact, plus neck
      nq_keypoints_ = object_loader->get_nq_keypoints(); // 1: stick end
      num_keypoints_ = object_loader->get_num_keypoints(); // 1: stick end
      njoints_object_ = object_loader->get_njoints(); // 1
      num_contact_points_ = object_loader->get_num_contacts(); // 2 or 3: left&right hand contact, plus neck
      assert(nq_pino_object_ == 7 && "nq_pino_object_ != 7");
      assert(nq_keypoints_ == 1 && "nq_keypoints_ != 1");
      assert(num_keypoints_ == 1 && "num_keypoints_ != 1");
      assert(njoints_object_ == 1 && "njoints_object_ != 1");
  }

  bool operator()(double const *const *parameters, double *residual) const
  {
    // get q_object
    const double *const q_object = *parameters;
    // get q_keypoints
    const double *const q_keypoints = *(parameters + 1);

    // convert q_object to q_object_pino: replace axis-angles with quaternions
    Matrix<double, 6, 1> q_object_mat_in(q_object);
    VectorXd q_object_mat = q_object_mat_in;
    object_loader_->UpdateConfigPino(i_, q_object_mat);
    const double *q_object_pino = object_loader_->mutable_config_pino(i_);
    // update object contact point positions via forward kinematics
    VectorXd q_stacked = VectorXd::Zero(nq_pino_object_ + nq_contact_ + nq_keypoints_);
    for (int i = 0; i < nq_pino_object_; i++)
    {
      q_stacked(i) = *(q_object_pino + i);
    }
    // here we ignore contact positions for they are not used
    for (int i = 0; i < nq_keypoints_; i++)
    {
      q_stacked(i + nq_pino_object_ + nq_contact_) = *(q_keypoints + i);
    }
    pinocchio::forwardKinematics(object_loader_->model_, object_loader_->data_, q_stacked);

    // compute residual blocks
    Vector3d endpoint1_3d_position = object_loader_->data_.oMi[1].translation();
    Vector3d endpoint2_3d_position = object_loader_->data_.oMi[njoints_object_ + num_contact_points_ + 1].translation();
    // penalize the vertical distance between two endpoints
    residual[0] = endpoint1_3d_position(1) - endpoint2_3d_position(1);
    return true;
  }

  static ceres::CostFunction *Create(int i,
                                     DataloaderObject *object_loader)
  {
      CostFunctorHorizontalObject *cost_functor =
          new CostFunctorHorizontalObject(i, object_loader);
      CostFunctionHorizontalObject *cost_function =
          new CostFunctionHorizontalObject(cost_functor);
      // object config
      cost_function->AddParameterBlock(object_loader->get_nq());
      // keypoint config
      cost_function->AddParameterBlock(object_loader->get_nq_keypoints());
      // number of residuals
      cost_function->SetNumResiduals(1);
      return cost_function;
  }

private:
  int i_;
  int nq_pino_object_;
  int nq_contact_;
  int nq_keypoints_;
  int njoints_object_;
  int num_contact_points_;
  int num_keypoints_;
  DataloaderObject *object_loader_;
};

#endif // ifndef __HORIZONTAL_OBJECT_H__
