#ifndef __SMOOTHING_OBJECT_FORCE_H__
#define __SMOOTHING_OBJECT_FORCE_H__

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "../dataloader/dataloader_person.h"
#include "../dataloader/dataloader_object.h"

using namespace Eigen;
using namespace std;
using namespace pinocchio;

struct CostFunctorSmoothingObjectForce
{
    typedef ceres::DynamicNumericDiffCostFunction<CostFunctorSmoothingObjectForce> CostFunctionSmoothingObjectForce;

    CostFunctorSmoothingObjectForce(int i,
                                     DataloaderPerson *person_loader)
    {
        i_ = i; // num of time step
        dt_ = person_loader->get_dt();
        // contact states at current and previous frame
        VectorXi contact_states = person_loader->get_contact_states_column(i_);
        VectorXi contact_states_prev = person_loader->get_contact_states_column(i_ - 1);
        int nj = contact_states.rows();
        for (int j = 0; j < nj; j++)
        {
            if (contact_states(j) == 1 && contact_states_prev(j) == 1)
            {
                contact_joints_.push_back(j);
            }
        }
        num_contact_joints_ = contact_joints_.size();
        // since each human joint can have at most 1 contact point, we set
        num_contact_points_ = num_contact_joints_;
        // get contact mapping
        contact_mapping_ = person_loader->get_contact_mapping();
    }

    bool operator()(double const *const *parameters, double *residual) const
    {
        const double *const f_contact = *parameters;
        const double *const f_contact_0 = *(parameters + 1);

        int j, fid;
        for (int c = 0; c < num_contact_joints_; c++)
        {
            j = contact_joints_[c];
            fid = contact_mapping_[j];
            for (int k = 0; k < 6; k++)
            {
                residual[6 * c + k] = (f_contact[6 * (fid - 1) + k] - f_contact_0[6 * (fid - 1) + k]) / dt_;
            }
        }
        return true;
    }

    static ceres::CostFunction *Create(int i,
                                       DataloaderPerson *person_loader,
                                       DataloaderObject *object_loader)
    {
        CostFunctorSmoothingObjectForce *cost_functor =
            new CostFunctorSmoothingObjectForce(i, person_loader);
        CostFunctionSmoothingObjectForce *cost_function =
            new CostFunctionSmoothingObjectForce(cost_functor);
        int num_residuals = 6*cost_functor->get_num_contact_points();
        if (num_residuals > 0)
        {
            // add parameter blocks
            int nq_contact_force = object_loader->get_nq_contact_force();
            cost_function->AddParameterBlock(nq_contact_force);
            cost_function->AddParameterBlock(nq_contact_force);
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

#endif // #ifndef __SMOOTHING_OBJECT_FORCE_H__
