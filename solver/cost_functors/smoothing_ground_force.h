#ifndef __SMOOTHING_GROUND_FORCE_H__
#define __SMOOTHING_GROUND_FORCE_H__

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "../dataloader/dataloader_person.h"
#include "../dataloader/dataloader_object.h"

using namespace Eigen;
using namespace std;
using namespace pinocchio;

struct CostFunctorSmoothingGroundForce
{
    typedef ceres::DynamicNumericDiffCostFunction<CostFunctorSmoothingGroundForce> CostFunctionSmoothingGroundForce;

    CostFunctorSmoothingGroundForce(int i,
                                     DataloaderPerson *person_loader)
    {
        i_ = i; // num of time step
        dt_ = person_loader->get_dt();
        // contact states at current and previous frame
        VectorXi contact_states = person_loader->get_contact_states_column(i_);
        VectorXi contact_states_prev = person_loader->get_contact_states_column(i_ - 1);
        int nj = contact_states.rows();
        num_contact_points_ = 0;
        for (int j = 0; j < nj; j++)
        {
            if (contact_states(j) == 2 && contact_states_prev(j) == 2)
            {
                contact_joints_.push_back(j);
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
    }

    bool operator()(double const *const *parameters, double *residual) const
    {
        const double *const f_contact = *parameters;
        const double *const f_contact_0 = *(parameters + 1);

        int j, fid;
        int count_contact = 0;
        for (int c = 0; c < num_contact_joints_; c++)
        {
            j = contact_joints_[c];
            fid = contact_mapping_[j];
            if (j != 3 && j != 7)
            {
                for (int k = 0; k < 4; k++)
                {
                    residual[4 * count_contact + k] = (f_contact[4 * (fid - 1) + k] - f_contact_0[4 * (fid - 1) + k]) / dt_;
                }
                count_contact++;
            }
            else
            {
                for (int n = 0; n < 4; n++)
                {
                    for (int k = 0; k < 4; k++)
                    {
                        residual[4 * count_contact + k] = (f_contact[4 * (fid - 1 + n) + k] - f_contact_0[4 * (fid - 1 + n) + k]) / dt_;
                    }
                    count_contact++;
                }
            }
        }
        return true;
    }

    static ceres::CostFunction *Create(int i,
                                       DataloaderPerson *person_loader)
    {
        CostFunctorSmoothingGroundForce *cost_functor =
            new CostFunctorSmoothingGroundForce(i, person_loader);
        CostFunctionSmoothingGroundForce *cost_function =
            new CostFunctionSmoothingGroundForce(cost_functor);
        int num_residuals = 4 * cost_functor->get_num_contact_points();
        if (num_residuals > 0)
        {
            // add parameter blocks
            int nq_ground_friction = person_loader->get_nq_ground_friction();
            cost_function->AddParameterBlock(nq_ground_friction);
            cost_function->AddParameterBlock(nq_ground_friction);
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
    VectorXi contact_mapping_;
};

#endif // #ifndef __SMOOTHING_GROUND_FORCE_H__
