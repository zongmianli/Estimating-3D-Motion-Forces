#ifndef __OBJECT_VELOCITY_H__
#define __OBJECT_VELOCITY_H__

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "pinocchio/algorithm/joint-configuration.hpp"

#include "../dataloader/dataloader_object.h"

struct CostFunctorObjectVelocity
{
    typedef ceres::DynamicNumericDiffCostFunction<CostFunctorObjectVelocity>
    CostFunctionObjectVelocity;
    //typedef Matrix<double, 6, 1> Vector6d;
    typedef Matrix<double, 7, 1> Vector7d;
    //typedef Eigen::Map<const Vector6d> MapConstVector6d;
    typedef Eigen::Map<const Vector7d> MapConstVector7d;

    CostFunctorObjectVelocity(
        int i,
        DataloaderObject *object_loader)
    {
        i_ = i; // num of time step
        object_loader_ = object_loader;
        dt_ = object_loader->get_dt();
        nq_ = object_loader_->get_nq(); // 6
        nq_pino_ = object_loader->get_nq_pino();    // 7
        int nq_contact = object_loader->get_nq_contact();     // often 2
        int nq_keypoints = object_loader->get_nq_keypoints(); // 2 endpoints
        nq_stacked_ = nq_pino_ + nq_contact + nq_keypoints;
    }

    bool operator()(double const *const *parameters, double *residual) const
    {
        // Get q_stacked
        const double *const q_pino = *parameters;
        Eigen::VectorXd q_pino_mat = MapConstVector7d(q_pino);
        object_loader_->set_config_pino_column(i_, q_pino_mat);
        Eigen::VectorXd q_stacked = Eigen::VectorXd::Zero(nq_stacked_);
        q_stacked.head<7>() = MapConstVector7d(object_loader_->mutable_config_pino(i_));

        // Get q_stacked at previous frame
        const double *const q_pino_minus1 = *(parameters + 1);
        Eigen::VectorXd q_pino_minus1_mat = MapConstVector7d(q_pino_minus1);
        object_loader_->set_config_pino_column(i_-1, q_pino_minus1_mat);
        Eigen::VectorXd q_stacked_minus1 = Eigen::VectorXd::Zero(nq_stacked_);
        q_stacked_minus1.head<7>() = MapConstVector7d(object_loader_->mutable_config_pino(i_-1));

        Eigen::Map<Eigen::VectorXd>(residual, nq_, 1).noalias() =
            pinocchio::difference(object_loader_->model_, q_stacked_minus1, q_stacked).head(nq_)/dt_;

        return true;
    }

    static ceres::CostFunction *Create(
        int i,
        DataloaderObject *object_loader)
    {
        CostFunctorObjectVelocity *cost_functor= 
            new CostFunctorObjectVelocity(i, object_loader);
        CostFunctionObjectVelocity *cost_function =
            new CostFunctionObjectVelocity(cost_functor);
        // Object config parameters
        int nq_pino_object = object_loader->get_nq_pino();
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

#endif // ifndef __OBJECT_VELOCITY_H__
