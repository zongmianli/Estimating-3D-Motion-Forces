#ifndef __OBJECT_DATA_H__
#define __OBJECT_DATA_H__

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "../dataloader/dataloader_object.h"
#include "../camera.h"

using namespace Eigen;
using namespace std;
using namespace pinocchio;

struct CostFunctorObjectData
{
    typedef ceres::DynamicNumericDiffCostFunction<CostFunctorObjectData> CostFunctionObjectData;
    typedef Eigen::Matrix<double, 4, 1> Vector4d;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<double, 7, 1> Vector7d;
    typedef Eigen::Map<const Vector6d> MapConstVector6d;
    typedef Eigen::Map<const Vector7d> MapConstVector7d;
    typedef Eigen::Map<const Vector4d> MapConstVector4d;

    CostFunctorObjectData(int i,
                          DataloaderObject *object_loader,
                          Camera *camera)
    {
        // cout << "CostFunctorObjectData: in" << endl;
        i_ = i; // num of time step
        njoints_object_ = object_loader->get_njoints(); // 1
        // cout << "njoints_object_: " << njoints_object_ << endl;
        if (njoints_object_!= 1)
        {
            assert(false && "Must never happened: njoints_object_!= 1");
        }
        // length of a contact configuration vector
        nq_pino_object_ = object_loader->get_nq_pino();    // 7: base translation + quaternion rotation
        // cout << "nq_pino_object_: " << nq_pino_object_ << endl;
        nq_contact_ = object_loader->get_nq_contact();     // 2: left&right hand contact
        // cout << "nq_contact_: " << nq_contact_ << endl;
        nq_keypoints_ = object_loader->get_nq_keypoints(); // 1: head
        // cout << "nq_keypoints_: " << nq_keypoints_ << endl;
        num_contact_points_ = object_loader->get_num_contacts(); // 2: left&right hand contact
        // cout << "num_contact_points_: " << num_contact_points_ << endl;
        if (nq_keypoints_!=1)
        {
            assert(false && "Must never happened: nq_keypoints_!=1");
        }
        if (num_contact_points_!= 2)
        {
            assert(false && "Must never happened: num_contact_points_!= 2");
        }
        num_keypoints_ = object_loader->get_num_keypoints(); // 1
        // cout << "num_keypoints_: " << num_keypoints_ << endl;
        endpoint_2d_positions_ = object_loader->get_endpoint_2d_positions_column(i_); // handle end + head
        camera_ = camera;
        object_loader_ = object_loader;
        // cout << "CostFunctorObjectData: out" << endl;
    }

    bool operator()(double const *const *parameters, double *residual) const
    {
        const double *const q_pino_object = *parameters;
        const double *const q_keypoints = *(parameters + 1);

        // Normalize the quaternions and save to the dataloader
        VectorXd q_pino_object_mat = MapConstVector7d(q_pino_object);
        object_loader_->set_config_pino_column(i_, q_pino_object_mat); 

        // Get q_stacked
        VectorXd q_stacked = VectorXd::Zero(nq_pino_object_ + nq_contact_ + nq_keypoints_);
        q_stacked.head<7>() = MapConstVector7d(object_loader_->mutable_config_pino(i_));
        // here we ignore contact positions for they are not used
        for (int i = 0; i < nq_keypoints_; i++)
        {
            q_stacked(i + nq_pino_object_ + nq_contact_) = *(q_keypoints + i);
        }

        pinocchio::forwardKinematics(object_loader_->model_, object_loader_->data_, q_stacked);

        // compute residual blocks
        Vector3d endpoint_position_3d;
        Vector2d projected_endpoint;
        double endpoint_confidence;
        for (int i = 0; i < num_keypoints_ + 1; i++)
        {
            if (i == 0)
            { // The head is the "default" keypoint
                endpoint_position_3d = object_loader_->data_.oMi[1].translation();
            }
            else
            { // other keypoints, e.g. the handle end
                endpoint_position_3d = object_loader_->data_.oMi[njoints_object_ + num_contact_points_ + i].translation();
            }

            projected_endpoint = camera_->Project(endpoint_position_3d);
            endpoint_confidence = endpoint_2d_positions_(3 * i + 2);

            for (int k = 0; k < 2; k++)
            {
                residual[2 * i + k] = endpoint_confidence *
                    (projected_endpoint(k) - endpoint_2d_positions_(3 * i + k));
            }
        }

        return true;
    }

    static ceres::CostFunction *Create(int i,
                                       DataloaderObject *object_loader,
                                       Camera *camera)
    {
        CostFunctorObjectData *cost_functor =
            new CostFunctorObjectData(i, object_loader, camera);
        CostFunctionObjectData *cost_function =
            new CostFunctionObjectData(cost_functor);
        // object config
        int nq_pino_object = object_loader->get_nq_pino();
        cost_function->AddParameterBlock(nq_pino_object);
        // keypoint config
        int nq_keypoints = object_loader->get_nq_keypoints();
        cost_function->AddParameterBlock(nq_keypoints);
        // number of residuals
        int num_keypoints = object_loader->get_num_keypoints();
        cost_function->SetNumResiduals(2 * (num_keypoints + 1));
        return cost_function;
    }

private:
    int i_;
    int njoints_object_;
    int nq_pino_object_;
    int nq_contact_;
    int nq_keypoints_;
    int num_contact_points_;
    int num_keypoints_;
    VectorXd endpoint_2d_positions_;
    Camera *camera_;
    DataloaderObject *object_loader_;
};

#endif // ifndef __OBJECT_DATA_H__
