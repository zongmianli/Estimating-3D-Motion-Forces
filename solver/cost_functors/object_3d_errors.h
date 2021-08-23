#ifndef __OBJECT_3D_ERRORS_H__
#define __OBJECT_3D_ERRORS_H__

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "../dataloader/dataloader_object.h"

using namespace Eigen;
using namespace std;
using namespace pinocchio;

struct CostFunctorObject3dErrors
{
    typedef ceres::DynamicNumericDiffCostFunction<CostFunctorObject3dErrors> CostFunctionObject3dErrors;
    typedef Eigen::Matrix<double, 4, 1> Vector4d;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<double, 7, 1> Vector7d;
    typedef Eigen::Map<const Vector6d> MapConstVector6d;
    typedef Eigen::Map<const Vector7d> MapConstVector7d;
    typedef Eigen::Map<const Vector4d> MapConstVector4d;

    CostFunctorObject3dErrors(
        int i,
        DataloaderObject *object_loader);

    bool operator()(double const *const *parameters, double *residual) const
    {
        return Evaluate(parameters, residual, NULL);
    }

    bool Evaluate(double const *const * parameters,
                  double * residual,
                  double ** jacobians) const;

    static ceres::CostFunction *Create(
        int i,
        DataloaderObject *object_loader);

private:
    int i_;
    int njoints_object_;
    int nq_pino_object_;
    int nq_contact_;
    int nq_keypoints_;
    int num_contact_points_;
    int num_keypoints_;
    VectorXd keypoint_3d_positions_;
    DataloaderObject *object_loader_;

};

#endif // ifndef __OBJECT_3D_ERRORS_H__
