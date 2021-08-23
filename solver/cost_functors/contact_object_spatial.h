#ifndef __CONTACT_OBJECT_SPATIAL_H__
#define __CONTACT_OBJECT_SPATIAL_H__

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <glog/logging.h>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "../dataloader/dataloader_person.h"
#include "../dataloader/dataloader_object.h"

using namespace std;
using namespace Eigen;
using namespace pinocchio;

struct CostFunctorContactObjectSpatial
{
    typedef ceres::DynamicNumericDiffCostFunction<CostFunctorContactObjectSpatial> CostFunctionContactObjectSpatial;
    typedef Matrix<double, 6, 1> Vector6d;
    typedef Matrix<double, 7, 1> Vector7d;
    typedef Eigen::Map<const Vector6d> MapConstVector6d;
    typedef Eigen::Map<const Vector7d> MapConstVector7d;

    CostFunctorContactObjectSpatial(int i,
                                    DataloaderPerson *person_loader,
                                    DataloaderObject *object_loader,
                                    bool enforce_3D_distance)
    {
        // cout << "(enter obj_spa) " << endl;
        i_ = i;         // num of time step
        njoints_object_ = object_loader->get_njoints(); // 1
        // the object is considered as detected only if the confidence scores of the two endpoints are high enough
        VectorXd endpoint_2d_positions = object_loader->get_endpoint_2d_positions_column(i_);
        double endpoint_confidence = (endpoint_2d_positions(2)+endpoint_2d_positions(5))/2;
        if(endpoint_confidence > 0.1)
        {
            object_detected_ = true;
        }
        else
        {
            object_detected_ = false;
        }
        enforce_3D_distance_ = enforce_3D_distance;
        // cout << "njoints_object_:" << njoints_object_<< endl;
        VectorXi contact_states = person_loader->get_contact_states_column(i_);
        int nj = (int)contact_states.rows();
        // cout << "contact_joints_:" << endl;
        for (int j = 0; j < nj; j++)
        {
            if (contact_states(j) == 1)
            {
                contact_joints_.push_back(j);
            }
        }
        num_contact_joints_ = (int)contact_joints_.size();
        // since each human joint can have at most 1 contact point, we set
        num_contact_points_ = num_contact_joints_;
        // length of a contact configuration vector
        nq_pino_object_ = object_loader->get_nq_pino();    // 7
        nq_contact_ = object_loader->get_nq_contact();     // often 2
        nq_keypoints_ = object_loader->get_nq_keypoints(); // 2 endpoints
        nq_stacked_ = nq_pino_object_ + nq_contact_ + nq_keypoints_;
        // get contact mapping
        contact_mapping_ = person_loader->get_contact_mapping();
        // save dataloaders
        person_loader_ = person_loader;
        object_loader_ = object_loader;
        // cout << "(leave obj_spa) " << endl;
    }

    bool operator()(double const *const *parameters, double *residual) const
    {
        const double *const q_person = *parameters;
        const double *const q_pino_object = *(parameters + 1);
        const double *const q_contact = *(parameters + 2);

        // convert q_person to q_person_pino
        Matrix<double, 75, 1> q_person_mat_in(q_person);
        VectorXd q_person_mat = q_person_mat_in;
        person_loader_->UpdateConfigPino(i_, q_person_mat);
        Matrix<double, 99, 1> q_person_pino(person_loader_->mutable_config_pino(i_));
        // update person joint positions via forward kinematics
        pinocchio::forwardKinematics(person_loader_->model_, person_loader_->data_, q_person_pino);

        // // convert q_object to q_object_pino: replace axis-angles with quaternions
        // Vector6d q_object_mat = MapConstVector6d(q_object);
        // //std::cout << "q_object_mat == " << q_object_mat.transpose() << std::endl;
        // Eigen::VectorXd q_object_mat_xd(q_object_mat);
        // object_loader_->UpdateConfigPino(i_, q_object_mat_xd);
        // double * q_object_pino = object_loader_->mutable_config_pino(i_);

        // Normalize the quaternions and save to the dataloader
        VectorXd q_pino_object_mat = MapConstVector7d(q_pino_object);
        object_loader_->set_config_pino_column(i_, q_pino_object_mat);

        // update object contact point positions via forward kinematics
        VectorXd q_stacked = VectorXd::Zero(nq_stacked_);
        q_stacked.head<7>() = MapConstVector7d(object_loader_->mutable_config_pino(i_));
        // for(int i=0; i<nq_pino_object_; i++)
        // {
        //     q_stacked(i) = *(q_object_pino+i);
        // }
        for(int i=0; i<nq_contact_; i++)
        {
            q_stacked(i+nq_pino_object_) = *(q_contact+i);
        }
        // here we ignore keypoint positions for they are not used
        // const double * q_keypoints = object_loader_->mutable_config_keypoints();
        // for(int i=0; i<nq_keypoints_; i++)
        // {
        //     q_stacked(i+nq_pino_object_+nq_contact_) = *(q_keypoints+i);
        // }
        pinocchio::forwardKinematics(object_loader_->model_, object_loader_->data_, q_stacked);

        // compute residual blocks
        Vector3d joint_position;
        Vector3d contact_position;
        int j;
        for (int c = 0; c < num_contact_joints_; c++)
        {
            j = contact_joints_[c];
            // get person joint j's position (j+1 in Pinocchio) if it is in contact
            joint_position = person_loader_->data_.oMi[j + 1].translation();
            // find the corresponding contact point from the object model
            contact_position = object_loader_->data_.oMi[njoints_object_+contact_mapping_(j)].translation();
            if (object_detected_ || enforce_3D_distance_)
            {
                for (int k = 0; k < 3; k++)
                { // xyz
                    residual[3 * c + k] = joint_position(k) - contact_position(k);
                }
            }
            else
            {
                for (int k = 0; k < 2; k++)
                { // xy
                    residual[2 * c + k] = joint_position(k) - contact_position(k);
                }
            }
        }
        return true;
    }

    static ceres::CostFunction *Create(int i,
                                       DataloaderPerson *person_loader,
                                       DataloaderObject *object_loader,
                                       bool enforce_3D_distance)
    {
        CostFunctorContactObjectSpatial *cost_functor =
            new CostFunctorContactObjectSpatial(i,
                                                person_loader,
                                                object_loader,
                                                enforce_3D_distance);
        CostFunctionContactObjectSpatial *cost_function = 
            new CostFunctionContactObjectSpatial(cost_functor);
        // minimize 3D distance if object is detected, otherwise minimize 2D distance only
        int num_residuals;
        bool object_detected = cost_functor->get_object_detected();
        if (object_detected || enforce_3D_distance)
        {
            num_residuals = 3 * cost_functor->get_num_contact_points();
        }
        else
        {
            num_residuals = 2 * cost_functor->get_num_contact_points();
        }
        if (num_residuals > 0)
        {
            // add parameter blocks
            int nq_person = person_loader->get_nq();
            int nq_pino_object = object_loader->get_nq_pino();
            int nq_contact = object_loader->get_nq_contact();
            cost_function->AddParameterBlock(nq_person);
            cost_function->AddParameterBlock(nq_pino_object);
            cost_function->AddParameterBlock(nq_contact);
            // number of residuals
            cost_function->SetNumResiduals(num_residuals);
            return cost_function;
        }
        return NULL;
    }

    bool get_object_detected()
    {
        return object_detected_;
    }

    int get_num_contact_points()
    {
        return num_contact_points_;
    }

private:
    int i_;
    bool object_detected_;
    bool enforce_3D_distance_; // true if we wish to minimize 3D distance even if the object is not detected.
    int njoints_object_;
    int nq_pino_object_;
    int nq_contact_;
    int nq_keypoints_;
    int nq_stacked_;
    int num_contact_joints_;
    int num_contact_points_;
    vector<int> contact_joints_;
    VectorXi contact_mapping_;
    DataloaderPerson *person_loader_;
    DataloaderObject *object_loader_;
};

#endif // #ifndef __CONTACT_OBJECT_SPATIAL_H__
