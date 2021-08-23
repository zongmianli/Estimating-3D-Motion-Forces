#ifndef __OBJECT_ACCELERATION_H__
#define __OBJECT_ACCELERATION_H__

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "pinocchio/algorithm/joint-configuration.hpp"

#include "../dataloader/dataloader_object.h"

using namespace std;

struct CostFunctorObjectAcceleration
{
    typedef ceres::DynamicNumericDiffCostFunction<CostFunctorObjectAcceleration>
    CostFunctionObjectAcceleration;
    typedef Matrix<double, 7, 1> Vector7d;
    typedef Eigen::Map<const Vector7d> MapConstVector7d;

    CostFunctorObjectAcceleration(
        int i,
        DataloaderObject *object_loader)
    {
        i_ = i;
        object_loader_ = object_loader;
        dt_ = object_loader->get_dt();
        nq_ = object_loader_->get_nq(); // 6
        nq_pino_ = object_loader->get_nq_pino();    // 7
        int nq_contact = object_loader->get_nq_contact();     // 1 or 2
        int nq_keypoints = object_loader->get_nq_keypoints(); // 2 endpoints
        nq_stacked_ = nq_pino_ + nq_contact + nq_keypoints;
    }

    bool operator()(double const *const *parameters, double *residual) const
    {
        Eigen::VectorXd config_stacked[3];
        for (int k=0; k<=2; k++)
        {
            const double *const q_pino_object = parameters[k]; // k = 0,1,2 correspond to i_, i_-1 and i_-2
            Eigen::VectorXd q_pino_object_mat = MapConstVector7d(q_pino_object);
            object_loader_->set_config_pino_column(i_-k, q_pino_object_mat);
            Eigen::VectorXd q_stacked = Eigen::VectorXd::Zero(nq_stacked_);
            q_stacked.head<7>() = MapConstVector7d(object_loader_->mutable_config_pino(i_-k));

            config_stacked[k] = q_stacked;
        }

        Eigen::VectorXd vq_object = pinocchio::difference(
            object_loader_->model_, config_stacked[1], config_stacked[0]).head(nq_)/dt_;
        Eigen::VectorXd vq_object_minus1 = pinocchio::difference(
            object_loader_->model_, config_stacked[2], config_stacked[1]).head(nq_)/dt_;

        Eigen::Map<Eigen::VectorXd>(residual, nq_, 1).noalias() = (vq_object - vq_object_minus1)/dt_;

        return true;
    }

    static ceres::CostFunction *Create(int i,
                                       DataloaderObject *object_loader)
    {
        CostFunctorObjectAcceleration *cost_functor =
            new CostFunctorObjectAcceleration(i, object_loader);
        CostFunctionObjectAcceleration *cost_function =
            new CostFunctionObjectAcceleration(cost_functor);
        // Object config parameters
        int nq_pino_object = object_loader->get_nq_pino();
        cost_function->AddParameterBlock(nq_pino_object);
        cost_function->AddParameterBlock(nq_pino_object);
        cost_function->AddParameterBlock(nq_pino_object);
        // number of residuals
        cost_function->SetNumResiduals(object_loader->get_nq());
        return cost_function;
    }

private:
    int i_;
    int nq_;
    int nq_pino_;
    int nq_stacked_;
    double dt_;
    DataloaderObject *object_loader_;
};

#endif // ifndef __OBJECT_ACCELERATION_H__
