#include "limit_ground_force.h"


CostFunctorLimitGroundForce::CostFunctorLimitGroundForce(
    int i,
    DataloaderPerson *person_loader)
{
    Eigen::VectorXi contact_states = person_loader->get_contact_states_column(i);
    int nj = contact_states.rows();
    num_contact_points_ = 0;
    for (int j = 0; j < nj; j++)
    {
        if (contact_states(j) == 2)
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


bool CostFunctorLimitGroundForce::Evaluate(
    double const *const * parameters,
    double * residual,
    double ** jacobians) const
{
    const double *const f_contact = parameters[0];

    int j, fid;
    int count_contact = 0;
    for (int c = 0; c < num_contact_joints_; c++)
    {
        j = contact_joints_[c];
        fid = contact_mapping_(j);
        if (j != 3 && j != 7)
        {
            for (int k = 0; k < 4; k++)
            {
                residual[4 * count_contact + k] = f_contact[4 * (fid - 1) + k];
            }
            count_contact++;
        }
        else
        {
            for (int n = 0; n < 4; n++)
            {
                for (int k = 0; k < 4; k++)
                {
                    residual[4 * count_contact + k] = f_contact[4 * (fid - 1 + n) + k];
                }
                count_contact++;
            }
        }
    }

    if(jacobians)
    {

    }
    return true;
}

ceres::CostFunction *CostFunctorLimitGroundForce::Create(
    int i,
    DataloaderPerson *person_loader)
{
    CostFunctorLimitGroundForce *cost_functor =
        new CostFunctorLimitGroundForce(i, person_loader);
    CostFunctionLimitGroundForce *cost_function =
        new CostFunctionLimitGroundForce(cost_functor);
    int num_residuals = 4 * cost_functor->get_num_contact_points();
    // debugging
    //std::cout << "num_residuals == " << num_residuals << std::endl;
    if (num_residuals > 0)
    {
        int nq_ground_friction = person_loader->get_nq_ground_friction();
        cost_function->AddParameterBlock(nq_ground_friction);
        cost_function->SetNumResiduals(num_residuals);
        return cost_function;
    }
    return NULL;
}

int CostFunctorLimitGroundForce::get_num_contact_points()
{
    return num_contact_points_;
}
