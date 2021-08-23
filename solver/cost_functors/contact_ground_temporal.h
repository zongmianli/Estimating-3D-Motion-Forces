#ifndef __CONTACT_GROUND_TEMPORAL_H__
#define __CONTACT_GROUND_TEMPORAL_H__

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "../dataloader/dataloader_person.h"
#include "../dataloader/dataloader_object.h"

using namespace std;
using namespace Eigen;
using namespace pinocchio;

struct CostFunctorContactGroundTemporal
{
    typedef ceres::DynamicNumericDiffCostFunction<CostFunctorContactGroundTemporal> CostFunctionContactGroundTemporal;

    CostFunctorContactGroundTemporal(int i,
                                     DataloaderPerson *person_loader,
                                     DataloaderObject *ground_loader,
                                     const VectorXd &weights)
    {
        i_ = i; // num of time step
        dt_ = person_loader->get_dt();
        // contact states at current and previous frame
        VectorXi contact_states = person_loader->get_contact_states_column(i_);
        VectorXi contact_states_prev = person_loader->get_contact_states_column(i_ - 1);
        // joint contact types at current and previous frame
        VectorXi contact_types = person_loader->get_contact_types_column(i_);
        int nj = contact_states.rows();
        num_contact_points_ = 0;
        for (int j = 0; j < nj; j++)
        {
            if (contact_states(j) == 2 && contact_states_prev(j) == 2)
            {
                contact_joints_.push_back(j);
                contact_types_.push_back(contact_types(j));
                if (j == 3 || j == 7)
                {
                    // left and right ankle (foot) have 4 ground contact points each
                    num_contact_points_ += 4;
                }
                else
                {
                    // other joints have at most one ground contact point
                    num_contact_points_++;
                }
            }
        }
        num_contact_joints_ = contact_joints_.size();
        // get contact mapping
        contact_mapping_ = person_loader->get_contact_mapping();
        // save weights
        weights_ = weights;
        // save dataloaders
        ground_loader_ = ground_loader;
    }

    bool operator()(double const *const *parameters, double *residual) const
    {
        const double *const q_contact = *parameters;
        const double *const q_contact_0 = *(parameters + 1);

        int j, fid;
        double w;
        int count_contact = 0;
        for (int c = 0; c < num_contact_joints_; c++)
        {
            j = contact_joints_[c];
            fid = contact_mapping_[j];
            switch (contact_types_[c])
            {
            case 1:
                w = weights_(0); // fixed contact 
                break;
            case 2:
                w = weights_(1); // moving contact
                break;
            default:
                cout << "unknown contact type! " << endl;
                break;
            }
            if (j != 3 && j != 7)
            {
                for (int k = 0; k < 2; k++)
                {
                    residual[2 * count_contact + k] = w * (q_contact[2 * (fid - 1) + k] - q_contact_0[2 * (fid - 1) + k]) / dt_;
                }
                count_contact++;
            }
            else
            {
                for (int n = 0; n < 4; n++)
                {
                    for (int k = 0; k < 2; k++)
                    {
                        residual[2 * count_contact + k] = w * (q_contact[2 * (fid - 1 + n) + k] - q_contact_0[2 * (fid - 1 + n) + k]) / dt_;
                    }
                    count_contact++;
                }
            }
        }
        return true;
    }

    static ceres::CostFunction *Create(int i,
                                       DataloaderPerson *person_loader,
                                       DataloaderObject *ground_loader,
                                       const VectorXd &weights)
    {
        CostFunctorContactGroundTemporal *cost_functor =
            new CostFunctorContactGroundTemporal(i, person_loader, ground_loader, weights);
        CostFunctionContactGroundTemporal *cost_function =
            new CostFunctionContactGroundTemporal(cost_functor);
        int num_residuals = 2 * cost_functor->get_num_contact_points();
        if (num_residuals > 0)
        {
            // add parameter blocks
            int nq_contact = ground_loader->get_nq_contact();
            cost_function->AddParameterBlock(nq_contact);
            cost_function->AddParameterBlock(nq_contact);
            // number of residuals
            cost_function->SetNumResiduals(num_residuals);
            return cost_function;
        }
        return NULL;
    }

    int get_num_contact_points()
    {
        return num_contact_points_;
    }

  private:
    int i_;
    int num_contact_joints_;
    int num_contact_points_;
    double dt_;
    vector<int> contact_joints_;
    vector<int> contact_types_;
    VectorXi contact_mapping_;
    VectorXd weights_;
    DataloaderObject *ground_loader_;
};

#endif // #ifndef __CONTACT_GROUND_TEMPORAL_H__
