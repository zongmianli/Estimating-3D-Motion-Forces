#include "person_3d_errors.h"

CostFunctorPerson3dErrors::CostFunctorPerson3dErrors(
    int i,
    DataloaderPerson *person_loader)
{
    i_ = i; // num of time step
    person_loader_ = person_loader;
    nq_ = person_loader_->get_nq(); // 75d vector
    nq_pino_ = person_loader_->get_nq_pino(); // 99d vector

    // Measure the 12 limb joints only
    int joints_interest[] = {1, 2, 3, 5, 6, 7, 15, 16, 17, 20, 21, 22};
    for (int k = 0; k < 12; k++)
    {
        joint_ids_.push_back(joints_interest[k]);
    }
    njoints_ = (int)joint_ids_.size();

    // Save the 3D locations of the 18 joints (relative to the base joint)
    Eigen::VectorXd joint_3d_positions = person_loader_->get_joint_3d_positions_column(i_);
    joint_3d_positions_rel_ = joint_3d_positions - (joint_3d_positions.head<3>()).replicate(person_loader_->get_njoints(),1);
}

bool CostFunctorPerson3dErrors::Evaluate(
    double const *const * parameters,
    double * residual,
    double ** jacobians) const
{
    const double *const q = parameters[0];
    VectorXd q_mat = Eigen::Map<const VectorXd>(q,nq_,1);

    // convert 3D axis-angles to a 4D quaternions
    person_loader_->UpdateConfigPino(i_, q_mat);
    VectorXd q_pino = Eigen::Map<VectorXd>(
        person_loader_->mutable_config_pino(i_), nq_pino_, 1);
    assert(q_pino.size() == nq_pino_ && "q_pino of wrong dimension");

    // forwardKinematics
    pinocchio::forwardKinematics(person_loader_->model_, person_loader_->data_, q_pino);
    // Get the base joint position
    const Vector3d & base_position = person_loader_->data_.oMi[1].translation();

    for (int i = 0; i < njoints_; i++)
    {
        int joint_id = joint_ids_[(size_t)i]; // the id number of an openpose joint
        // Compute the joint position relative to the base joint
        const Vector3d & joint_position_rel = person_loader_->data_.oMi[(size_t)joint_id + 1].translation() - base_position;

        for (int k = 0; k < 3; k++)
        {
            residual[3 * i + k] = (joint_position_rel(k) - joint_3d_positions_rel_(3 * joint_id + k));
        }
    }

    if(jacobians)
    {
        // do nothing
    }

    return true;
}

ceres::CostFunction * CostFunctorPerson3dErrors::Create(
    int i,
    DataloaderPerson *person_loader)
{
    CostFunctorPerson3dErrors *cost_functor =
        new CostFunctorPerson3dErrors(i, person_loader);
    CostFunctionPerson3dErrors *cost_function =
        new CostFunctionPerson3dErrors(cost_functor);
    // person config parameters
    int nq = cost_functor->get_nq();
    cost_function->AddParameterBlock(nq);
    // number of residuals
    int njoints = cost_functor->get_njoints();
    cost_function->SetNumResiduals(3 * njoints);
    return cost_function;
}

int CostFunctorPerson3dErrors::get_njoints() const
{
    return njoints_;
}

int CostFunctorPerson3dErrors::get_nq() const
{
    return nq_;
}
