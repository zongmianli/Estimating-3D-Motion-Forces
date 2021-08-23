#include "limit_object_force.h"


CostFunctorLimitObjectForce::CostFunctorLimitObjectForce(
    int i,
    DataloaderPerson *person_loader)
{
    Eigen::VectorXi contact_states = person_loader->get_contact_states_column(i);
    int nj = contact_states.rows();
    for (int j = 0; j < nj; j++)
    {
        if (contact_states(j) == 1)
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


bool CostFunctorLimitObjectForce::Evaluate(
    double const *const * parameters,
    double * residual,
    double ** jacobians) const
{
    const double *const f_contact = parameters[0];

    int j, fid;
    for (int c = 0; c < num_contact_joints_; c++)
    {
        j = contact_joints_[c];
        fid = contact_mapping_(j);
        for (int k = 0; k < 6; k++)
        {
            residual[6 * c + k] = f_contact[6 * (fid - 1) + k];
        }
    }

    if(jacobians)
    {

    }
    return true;
}

ceres::CostFunction *CostFunctorLimitObjectForce::Create(
    int i,
    DataloaderPerson *person_loader,
    DataloaderObject *object_loader)
{
    CostFunctorLimitObjectForce *cost_functor =
        new CostFunctorLimitObjectForce(i, person_loader);
    CostFunctionLimitObjectForce *cost_function =
        new CostFunctionLimitObjectForce(cost_functor);
    int num_residuals = 6*cost_functor->get_num_contact_points();
    if (num_residuals > 0)
    {
        int nq_contact_force = object_loader->get_nq_contact_force();
        cost_function->AddParameterBlock(nq_contact_force);
        cost_function->SetNumResiduals(num_residuals);
        return cost_function;
    }
    return NULL;
}


int CostFunctorLimitObjectForce::get_num_contact_points()
{
    return num_contact_points_;
}
