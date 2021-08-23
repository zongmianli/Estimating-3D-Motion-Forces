#ifndef __PERSON_CENTER_OF_MASS_H__
#define __PERSON_CENTER_OF_MASS_H__

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"

#include "../dataloader/dataloader_person.h"

using namespace Eigen;
using namespace std;
using namespace pinocchio;

// This cost function penalizes first computes the positions of the person's center of mass
// and the center of the supporting polygon formed by the ground contact points.
// It then penalizes the horizontal distance between the two centers.
struct CostFunctorPersonCenterOfMass
{
    typedef ceres::DynamicNumericDiffCostFunction<CostFunctorPersonCenterOfMass> CostFunctionPersonCenterOfMass;
    CostFunctorPersonCenterOfMass(int i,
                                  DataloaderPerson *person_loader,
                                  double supporting_radius)
    {
        i_ = i; // num of time step
        person_loader_ = person_loader;
        supporting_radius_ = supporting_radius;
        // contact states at current and previous frame
        VectorXi contact_states = person_loader->get_contact_states_column(i_);
        // search the person joints that are in fixed contact with the ground at timestep i_
        int njoints = contact_states.rows();
        //cout << "(";
        for (int j=0; j<njoints; j++)
        {
            if (contact_states(j) == 2)
            {
                ground_contact_joints_.push_back(j);
                //cout << j << " "; // plot the joint ids to debug
            }
        }
        //cout << ") ";
        num_ground_contact_joints_ = ground_contact_joints_.size();
    }

    bool operator()(double const *const *parameters, double *residual) const
    {
        // recover person configuration vector at timestep i_ and i_ - 1
        const double *const q = *parameters;
        Matrix<double, 75, 1> q_mat_in(q);
        VectorXd q_mat = q_mat_in;
        person_loader_->UpdateConfigPino(i_, q_mat);
        Matrix<double, 99, 1> q_pino(person_loader_->mutable_config_pino(i_));

        // Compute the person's center of mass position
        // // Option 1: we approximate person center of mass by the pelvis joint
        // Vector3d center_of_mass = person_loader_->data_.oMi[1].translation();
        // Option 2: compute the center of mass using Pinocchio
        Eigen::Vector3d center_of_mass = pinocchio::centerOfMass(
            person_loader_->model_,
            person_loader_->data_,
            q_pino); 

        // Compute joint positions at timestep i_ via forward kinematics
        // NOTE: centerOfMass already calls forwardKinematics, no need to compute it again
        // pinocchio::forwardKinematics(person_loader_->model_,
        //                        person_loader_->data_,
        //                        q_pino);
        // compute the center of the supporting area (polygon)

        int j;
        Vector3d center_of_supporting_area_3d = Vector3d::Zero();
        for (int i = 0; i < num_ground_contact_joints_; i++)
        {
            j = ground_contact_joints_[i];
            center_of_supporting_area_3d += person_loader_->data_.oMi[j + 1].translation();
        }
        center_of_supporting_area_3d /= num_ground_contact_joints_;

        Vector3d distance_3d = center_of_supporting_area_3d - center_of_mass;

        Vector2d horizontal_distance;
        horizontal_distance << abs(distance_3d(0)), abs(distance_3d(2)); // xz distance
        for (int k = 0; k < 2; k++)
        {
            // the residual is zero if the horizontal distance (along xz) is less than supporting_radius_
            // it is non-zero otherwise
            if (horizontal_distance(k) < supporting_radius_)
            {
                residual[k] = 0.;
            }
            else
            {
                residual[k] = 300*(horizontal_distance(k)-supporting_radius_);
            }
        }
        //std::cout << "(com) horizontal_distance == " << horizontal_distance.transpose() << std::endl;
        
        return true;
    }

    static ceres::CostFunction *Create(int i,
                                       DataloaderPerson *person_loader,
                                       double supporting_radius)
    {
        CostFunctorPersonCenterOfMass *cost_functor =
            new CostFunctorPersonCenterOfMass(i,
                                              person_loader,
                                              supporting_radius);
        CostFunctionPersonCenterOfMass *cost_function =
            new CostFunctionPersonCenterOfMass(cost_functor);
        int num_residuals = cost_functor->get_num_ground_contact_joints() > 0 ? 2 : 0;
        if (num_residuals>0)
        {
            // add parameter blocks
            int nq = person_loader->get_nq(); // nq_person
            cost_function->AddParameterBlock(nq);
            // x,z distances
            cost_function->SetNumResiduals(num_residuals);
            return cost_function;
        }
        return NULL;
    }

    int get_num_ground_contact_joints()
    {
        return num_ground_contact_joints_;
    }

  private:
    int i_;
    DataloaderPerson *person_loader_;
    vector<int> ground_contact_joints_;
    int num_ground_contact_joints_;
    double supporting_radius_;
};

#endif // #ifndef __PERSON_CENTER_OF_MASS_H__
