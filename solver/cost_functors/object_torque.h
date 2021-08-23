#ifndef __OBJECT_TORQUE_H__
#define __OBJECT_TORQUE_H__

#include <math.h>
#include <ceres/ceres.h>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "../dataloader/dataloader_person.h"
#include "../dataloader/dataloader_object.h"

using namespace Eigen;
using namespace std;
using namespace pinocchio;

struct CostFunctorObjectTorque
{
    typedef ceres::DynamicNumericDiffCostFunction<CostFunctorObjectTorque>
        CostFunctionObjectTorque;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<double, 7, 1> Vector7d;
    //typedef Eigen::Map<Vector6d> MapVector6d;
    typedef Eigen::Map<const Vector6d> MapConstVector6d;
    typedef Eigen::Map<const Vector7d> MapConstVector7d;
    typedef Eigen::Matrix<double, 9, 1> Vector9d;
    typedef Eigen::Map<const Eigen::VectorXd> ConstMapVectorXd;
    typedef Eigen::Map<Eigen::VectorXd> MapVectorXd;

    CostFunctorObjectTorque(
        int i,
        DataloaderPerson *person_loader,
        DataloaderObject *object_loader,
        const VectorXd &weights);

    bool Evaluate(
        double const *const * parameters,
        double * residual,
        double ** jacobians) const;

    bool operator()(double const *const *parameters, double *residual) const
    {
        return Evaluate(parameters, residual, NULL);
    }

    static ceres::CostFunction *Create(
        int i,
        DataloaderPerson *person_loader,
        DataloaderObject *object_loader,
        const Eigen::VectorXd &weights);

    int get_nv_stacked();

private:
    int i_;
    int njoints_object_;
    int nq_pino_object_;
    int nq_contact_;
    int nq_keypoints_;
    int nq_stacked_;
    int nv_object_;
    int nv_stacked_;
    Eigen::VectorXd weights_;
    DataloaderPerson *person_loader_;
    DataloaderObject *object_loader_;
    int num_object_contact_joints_;
    std::vector<int> object_contact_joints_;
    Eigen::VectorXi contact_mapping_;
    double dt_;
};

#endif // ifndef __OBJECT_TORQUE_H__
