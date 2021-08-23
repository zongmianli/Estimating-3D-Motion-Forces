#ifndef __CONTACT_OBJECT_TEMPORAL_H__
#define __CONTACT_OBJECT_TEMPORAL_H__

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

struct CostFunctorContactObjectTemporal
{
    typedef ceres::DynamicNumericDiffCostFunction<CostFunctorContactObjectTemporal> CostFunctionContactObjectTemporal;

    CostFunctorContactObjectTemporal(int i,
                                     DataloaderPerson *person_loader,
                                     DataloaderObject *object_loader,
                                     const VectorXd &weights)
    {
        cout << "(enter obj_temp)" << endl;
        i_ = i; // num of time step
        dt_ = person_loader->get_dt();
        // contact states at current and previous frame
        VectorXi contact_states = person_loader->get_contact_states_column(i_);
        VectorXi contact_states_prev = person_loader->get_contact_states_column(i_ - 1);
        // joint contact types at current and previous frame
        VectorXi contact_types = person_loader->get_contact_types_column(i_);
        int nj = contact_states.rows();
        for (int j = 0; j < nj; j++)
        {
            if (contact_states(j) == 1 && contact_states_prev(j) == 1)
            {
                contact_joints_.push_back(j);
                contact_types_.push_back(contact_types(j));
            }
        }
        num_contact_joints_ = contact_joints_.size();
        // since each human joint can have at most 1 contact point, we set
        num_contact_points_ = num_contact_joints_;
        // get contact mapping
        contact_mapping_ = person_loader->get_contact_mapping();
        // save weights
        weights_ = weights;
        // save dataloaders
        object_loader_ = object_loader;
        cout << "(leave obj_temp)" << endl;
    }

    bool operator()(double const *const *parameters, double *residual) const
    {
        const double *const q_contact = *parameters;
        const double *const q_contact_0 = *(parameters + 1);

        int j, fid;
        double w;
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
            residual[c] = w * (q_contact[fid - 1] - q_contact_0[fid - 1]) / dt_;
        }
        return true;
    }

    static ceres::CostFunction *Create(int i,
                                       DataloaderPerson *person_loader,
                                       DataloaderObject *object_loader,
                                       const VectorXd &weights)
    {
        CostFunctorContactObjectTemporal *cost_functor =
            new CostFunctorContactObjectTemporal(i, person_loader, object_loader, weights);
        CostFunctionContactObjectTemporal *cost_function =
            new CostFunctionContactObjectTemporal(cost_functor);
        int num_residuals = cost_functor->get_num_contact_points();
        if (num_residuals > 0)
        {
            // add parameter blocks
            int nq_contact = object_loader->get_nq_contact();
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
    DataloaderObject *object_loader_;
};

#endif // #ifndef __CONTACT_OBJECT_TEMPORAL_H__
