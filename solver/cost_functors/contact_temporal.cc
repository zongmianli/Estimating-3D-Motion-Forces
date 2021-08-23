#include "contact_temporal.h"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"

CostFunctorContactTemporal::CostFunctorContactTemporal(
    int i,
    DataloaderPerson *person_loader,
    DataloaderObject *object_loader,
    const Eigen::VectorXd &weights,
    bool update_6d_basis_only,
    bool smooth_oc,
    bool virtual_object)
{
    i_ = i; // num of time step
    person_loader_ = person_loader;
    object_loader_ = object_loader;
    q_init_ = person_loader_->get_config_column(i_);
    q_minus1_init_ = person_loader_->get_config_column(i_-1);
    dt_ = person_loader->get_dt();
    if (update_6d_basis_only)
    {
        nq_person_effective_ = 6;
    }
    else
    {
        nq_person_effective_ = person_loader_->get_nq();
    }
    // contact states at current and previous frame
    Eigen::VectorXi contact_states = person_loader->get_contact_states_column(i_);
    Eigen::VectorXi contact_states_prev = person_loader->get_contact_states_column(i_ - 1);
    // contact types at current frame
    Eigen::VectorXi contact_types = person_loader->get_contact_types_column(i_);
    // search the person joints that are in fixed contact with the ground at timestep i_

    int njoints_person = (int)contact_states.rows();
    for (int j = 0; j < njoints_person; j++)
    {
        if (contact_states(j) == 2 && contact_states_prev(j) == 2)
        {
            ground_contact_joints_.push_back(j);
            ground_contact_types_.push_back(contact_types(j));
        }
        // Hide object contact joints if not smooth_oc
        if (smooth_oc)
        {
            if (contact_states(j) == 1 && contact_states_prev(j) == 1)
            {
                object_contact_joints_.push_back(j);
                object_contact_types_.push_back(contact_types(j));
            }
        }
    }
    num_ground_contact_joints_ = (int)ground_contact_joints_.size();
    num_object_contact_joints_ = (int)object_contact_joints_.size();
    // // debug:
    // std::cout << "num_ground_contact_joints_" << num_ground_contact_joints_ << std::endl;
    // std::cout << "num_object_contact_joints_" << num_object_contact_joints_ << std::endl;

    virtual_object_ = virtual_object;
    weights_ = weights;
}

bool CostFunctorContactTemporal::Evaluate(
    double const *const *parameters,
    double *residual,
    double ** jacobians) const
{
    // recover person configuration vector at timestep i_ and i_ - 1
    const double *const q = parameters[0];
    const double *const q_minus1 = parameters[1];
    // Eigen::VectorXd q_person = ConstMapVectorXd(
    //     q_person, nq_person_effective_, 1);
    // Eigen::VectorXd q_person_minus1 = ConstMapVectorXd(
    //     q_person_minus1, nq_person_effective_, 1);
    // Eigen::VectorXd q_mat = ConstMapVectorXd(
    //     q, nq_person_effective_, 1);
    // Eigen::VectorXd q_mat_minus1 = ConstMapVectorXd(
    //     q_minus1, nq_person_effective_, 1);
    Eigen::VectorXd q_mat, q_mat_minus1;
    if (nq_person_effective_ == 6)
    {
        Eigen::Matrix<double, 6, 1> q_basis(q);
        Eigen::Matrix<double, 6, 1> q_basis_minus1(q_minus1);
        q_mat = q_init_;
        q_mat_minus1 = q_minus1_init_;
        q_mat.block<6, 1>(0, 0) = q_basis;
        q_mat_minus1.block<6, 1>(0, 0) = q_basis_minus1;
    }
    else if (nq_person_effective_ == 75)
    {
        Eigen::Matrix<double, 75, 1> q_mat_in(q);
        Eigen::Matrix<double, 75, 1> q_mat_minus1_in(q_minus1);
        q_mat = q_mat_in;
        q_mat_minus1 = q_mat_minus1_in;
    }
    else
    {
        LOG(FATAL) << "unknown value for nq_person_effective_" << std::endl;
    }

    Eigen::Vector3d p_object = Eigen::Vector3d::Zero();
    Eigen::Vector3d p_object_minus1 = Eigen::Vector3d::Zero();
    if (!virtual_object_ && has_object_contact())
    {
        const double *const q_pino_object = parameters[2];
        const double *const q_pino_object_minus1 = parameters[3];

        p_object = Eigen::Map<const Eigen::Vector3d>(
            q_pino_object); // only consider object's position
        p_object_minus1 = Eigen::Map<const Eigen::Vector3d>(
            q_pino_object_minus1); // only consider object's position
    }

    // computing joint positions at timestep i_ via forward kinematics
    person_loader_->UpdateConfigPino(i_, q_mat);
    Eigen::Matrix<double, 99, 1> q_pino(person_loader_->mutable_config_pino(i_));
    pinocchio::forwardKinematics(person_loader_->model_,
                                 person_loader_->data_,
                                 q_pino);
    int j;
    Eigen::Vector3d p_joint;
    for (int i = 0; i < num_ground_contact_joints_; i++)
    {
        j = ground_contact_joints_[i];
        p_joint = person_loader_->data_.oMi[j + 1].translation();
        for (int k = 0; k < 3; k++)
        {
            // xyz
            residual[3*i + k] = p_joint(k);
        }
    }
    
    for (int i = 0; i < num_object_contact_joints_; i++)
    {
        j = object_contact_joints_[i];
        p_joint = person_loader_->data_.oMi[j + 1].translation();
        for (int k = 0; k < 3; k++)
        {
            // xyz
            residual[3*(i+num_ground_contact_joints_) + k] = p_joint(k) - p_object(k);
        }
    }
    
    // computing joint positions at timestep i_ - 1 and update the residual block
    person_loader_->UpdateConfigPino(i_ - 1, q_mat_minus1);
    Eigen::Matrix<double, 99, 1> q_pino_minus1(person_loader_->mutable_config_pino(i_ - 1));
    pinocchio::forwardKinematics(person_loader_->model_,
                                 person_loader_->data_,
                                 q_pino_minus1);
    
    Eigen::Vector3d p_joint_minus1;
    double w; // weight
    int residual_idx;
    for (int i = 0; i < num_ground_contact_joints_; i++)
    {
        j = ground_contact_joints_[i];
        switch (ground_contact_types_[i])
        {
        case 1:
            w = weights_(0); // fixed contact
            break;
        case 2:
            w = weights_(1); // moving contact
            break;
        default:
            std::cout << "unknown contact type! '" << std::endl;
            break;
        }
        p_joint_minus1 = person_loader_->data_.oMi[j + 1].translation();
        for (int k = 0; k < 3; k++)
        {
            residual_idx = 3 * i + k;
            // xyz
            residual[residual_idx] = w * (residual[residual_idx] - p_joint_minus1(k)) / dt_;
        }
    }

    for (int i = 0; i < num_object_contact_joints_; i++)
    {
        j = object_contact_joints_[i];
        switch (object_contact_types_[i])
        {
        case 1:
            w = weights_(0); // fixed contact
            break;
        case 2:
            w = weights_(1); // moving contact
            break;
        default:
            std::cout << "unknown contact type! '" << std::endl;
            break;
        }
        p_joint_minus1 = person_loader_->data_.oMi[j + 1].translation();
        for (int k = 0; k < 3; k++)
        {
            residual_idx = 3*(i+num_ground_contact_joints_) + k;
            // xyz
            residual[residual_idx] = w * (residual[residual_idx] - (p_joint_minus1(k)-p_object_minus1(k))) / dt_;
        }
    }

    if(jacobians)
    {
        // Do something here
    }

    return true;
}


ceres::CostFunction * CostFunctorContactTemporal::Create(
    int i,
    DataloaderPerson *person_loader,
    DataloaderObject *object_loader,
    const Eigen::VectorXd &weights,
    bool update_6d_basis_only,
    bool smooth_oc,
    bool *has_object_contact,
    bool virtual_object)
{
    CostFunctorContactTemporal *cost_functor =
        new CostFunctorContactTemporal(i,
                                       person_loader,
                                       object_loader,
                                       weights,
                                       update_6d_basis_only,
                                       smooth_oc,
                                       virtual_object);
    CostFunctionContactTemporal *cost_function =
        new CostFunctionContactTemporal(cost_functor);
    *has_object_contact = cost_functor->has_object_contact();

    int num_residuals = 3*cost_functor->get_num_contact_joints();
    if (num_residuals > 0)
    {
        // add parameter blocks
        int nq_person_effective =
            cost_functor->get_nq_person_effective(); // 6 or 75
        cost_function->AddParameterBlock(nq_person_effective);
        cost_function->AddParameterBlock(nq_person_effective);

        int nq_pino_object = object_loader->get_nq_pino();
        if (!virtual_object && (*has_object_contact) && smooth_oc)
        {
            cost_function->AddParameterBlock(nq_pino_object); // q_pino_object at i
            cost_function->AddParameterBlock(nq_pino_object); // q_pino_object at i - 1
        }
        // number of residuals
        cost_function->SetNumResiduals(num_residuals);
        return cost_function;
    }
    return NULL;
}

int CostFunctorContactTemporal::get_nq_person_effective()
{
    return nq_person_effective_;
}

int CostFunctorContactTemporal::get_num_contact_joints()
{
    return num_object_contact_joints_+num_ground_contact_joints_;
}

bool CostFunctorContactTemporal::has_object_contact() const
{
    return num_object_contact_joints_>0;
}

bool CostFunctorContactTemporal::has_ground_contact() const
{
    return num_ground_contact_joints_>0;
}



