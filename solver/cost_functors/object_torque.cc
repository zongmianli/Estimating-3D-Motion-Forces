#include "pinocchio/spatial/force.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include "object_torque.h"

CostFunctorObjectTorque::CostFunctorObjectTorque(
    int i,
    DataloaderPerson *person_loader,
    DataloaderObject *object_loader,
    const VectorXd &weights)
{
    i_ = i;
    weights_ = weights;
    person_loader_ = person_loader;
    object_loader_ = object_loader;
    njoints_object_ = object_loader->get_njoints(); // 1
    nq_pino_object_ = object_loader->get_nq_pino(); // 7
    nq_contact_ = object_loader->get_nq_contact();
    nq_keypoints_ = object_loader->get_nq_keypoints();
    nq_stacked_ = nq_pino_object_ + nq_contact_ + nq_keypoints_;
    nv_object_ = object_loader->model_.nv - nq_contact_ - nq_keypoints_; // = 6
    nv_stacked_ = nv_object_ + nq_contact_ + nq_keypoints_;
    // get person-object contact information
    VectorXi contact_states = person_loader->get_contact_states_column(i_);
    int njoints_person = (int)contact_states.rows();
    for (int j = 0; j < njoints_person; j++)
    {
        if (contact_states(j) == 1)
        {
            object_contact_joints_.push_back(j);
            //std::cout << " object_contact_joints_(j) == " << j << std::endl;
        }
    }
    num_object_contact_joints_ = (int)object_contact_joints_.size();
    contact_mapping_ = person_loader->get_contact_mapping();
    dt_ = person_loader->get_dt();

    // std::cout << "***** DEBUGGING *****" << std::endl;
    // std::cout << " i_ == " << i_ << std::endl;
    // std::cout << " weights_ == (0.05 0.05) " << weights_.transpose() << std::endl;
    // std::cout << " njoints_object_== (1) " << njoints_object_ << std::endl;
    // std::cout << " nq_pino_object_ == (7) " << nq_pino_object_ << std::endl;
    // std::cout << " nq_contact_ == (2) " << nq_contact_ << std::endl;
    // std::cout << " nq_keypoints_ == (1) " << nq_keypoints_ << std::endl;
    // std::cout << " nq_stacked_ == (10) " << nq_stacked_ << std::endl;
    // std::cout << " nv_stacked_ == (9) " << nv_stacked_ << std::endl;
    // std::cout << " nv_object_ == (6) " << nv_object_ << std::endl;
    // std::cout << " num_object_contact_joints_ == (2) " << num_object_contact_joints_ << std::endl;
    // std::cout << " dt_ == (0.033) " << dt_ << std::endl;
    // std::cout << "***** end: DEBUGGING *****" << std::endl;
    //LOG(FATAL) << "Check" << std::endl;
}

bool CostFunctorObjectTorque::Evaluate(
    double const *const * parameters,
    double * residual,
    double ** jacobians) const
{
    const double *const q_contact = parameters[3];
    const double *const q_keypoints = parameters[4];
    const double *const q_contact_force = parameters[5];

    // Compute q_stacked at frames i_, i_-1 and i_-2
    // Note that we do not update q_contact
    //Eigen::VectorXd ** config_stacked = nullptr;
    Eigen::VectorXd config_stacked[3];
    for (int k=0; k<=2; k++)
    {
        const double *const q_pino_object = parameters[k]; // k = 0,1,2 correspond to i_, i_-1 and i_-2
        Eigen::VectorXd q_pino_object_mat = MapConstVector7d(q_pino_object);
        object_loader_->set_config_pino_column(i_-k, q_pino_object_mat);
        Eigen::VectorXd q_stacked = Eigen::VectorXd::Zero(nq_stacked_);
        q_stacked.head(nq_pino_object_) = MapConstVector7d(object_loader_->mutable_config_pino(i_-k));
        q_stacked.tail(nq_keypoints_) = ConstMapVectorXd(q_keypoints, nq_keypoints_, 1);
        config_stacked[k] = q_stacked;
    }
    // std::cout << "q_pino_object (i)" << std::endl << MapConstVector7d(object_loader_->mutable_config_pino(i_)).transpose() << std::endl;
    // std::cout << "config_stacked[0].transpose = " << std::endl << config_stacked[0].transpose() << std::endl;
    // std::cout << "q_pino_object (i-1)" << std::endl << MapConstVector7d(object_loader_->mutable_config_pino(i_-1)).transpose() << std::endl;
    // std::cout << "config_stacked[1].transpose = " << std::endl << config_stacked[1].transpose() << std::endl;
    // std::cout << "q_pino_object (i-2)" << std::endl << MapConstVector7d(object_loader_->mutable_config_pino(i_-2)).transpose() << std::endl;
    // std::cout << "config_stacked[2].transpose = " << std::endl << config_stacked[2].transpose() << std::endl;

    // Eigen::VectorXd vq_object = pinocchio::difference(
    //     object_loader_->model_, *(*config_stacked + 1), **config_stacked)/dt_;
    // Eigen::VectorXd vq_object_minus1 = pinocchio::difference(
    //     object_loader_->model_, *(*config_stacked + 2), *(*config_stacked + 1))/dt_;
    // Eigen::VectorXd vq_dot_object = (vq_object - vq_object_minus1)/dt_;

    Eigen::VectorXd vq_object = pinocchio::difference(
        object_loader_->model_, config_stacked[1], config_stacked[0])/dt_;
    Eigen::VectorXd vq_object_minus1 = pinocchio::difference(
        object_loader_->model_, config_stacked[2], config_stacked[1])/dt_;
    Eigen::VectorXd vq_dot_object = (vq_object - vq_object_minus1)/dt_;


    // Copy q_stacked at frame i_ and assign q_contact
    Eigen::VectorXd q_stacked_with_contact = config_stacked[0];//**config_stacked;
    q_stacked_with_contact.segment(nq_pino_object_, nq_contact_) = ConstMapVectorXd(q_contact, nq_contact_, 1);

    // compute object inertia matrix M(q)
    pinocchio::crba(object_loader_->model_, object_loader_->data_, config_stacked[0]); // **config_stacked);
    Eigen::MatrixXd M_object(object_loader_->model_.nv, object_loader_->model_.nv);
    M_object.triangularView<Eigen::Upper>() = object_loader_->data_.M.triangularView<Eigen::Upper>();
    M_object.transpose().triangularView<Eigen::StrictlyUpper>() = M_object.triangularView<Eigen::StrictlyUpper>();

    VectorXd b_object = pinocchio::rnea(
        object_loader_->model_,
        object_loader_->data_,
        config_stacked[0],//**config_stacked,
        vq_object,
        VectorXd::Zero(nv_stacked_));

    // compute the full body Jacobian
    pinocchio::computeJointJacobians(object_loader_->model_, object_loader_->data_, q_stacked_with_contact);

    // compute object contact torque
    int j; // id of person joint
    int fid; // id of contact points
    VectorXd object_contact_torque = VectorXd::Zero(nv_stacked_);
    Vector6d object_contact_force = VectorXd::Zero(6);
    // initialize jacobian_world: the robot Jocobian of a joint j expressed in the world frame
    pinocchio::Data::Matrix6x jacobian_world(6, nv_stacked_);

    for (int i = 0; i < num_object_contact_joints_; i++)
    {
        j = object_contact_joints_[(size_t)i]; // joint j in contact with object
        fid = contact_mapping_[j];
        // get the robot Jacobian expressed in world frame
        jacobian_world.setZero();
        pinocchio::getJointJacobian(
            object_loader_->model_,
            object_loader_->data_,
            (pinocchio::JointIndex)(njoints_object_ + fid),
            WORLD,
            jacobian_world);
        object_contact_force.noalias() = person_loader_->data_.oMi[(size_t)j + 1].toDualActionMatrix() * (-MapConstVector6d(q_contact_force + 6 * (fid - 1)));
        object_contact_torque.noalias() += jacobian_world.transpose() * object_contact_force;
    }
    // cout << "object_contact_torque == " << endl;
    // cout << object_contact_torque.transpose() << endl;
    // compute object torque (should equal to zero as object has no actuator)
    Eigen::VectorXd torque_object = M_object * vq_dot_object + b_object - object_contact_torque;
    // cout << "torque_object" << endl;
    // cout << torque_object.transpose() << endl;
    // save residuals
    // the first 6 entries should be zero as the object is not actuated
    // we put a high weight here as a soft equality constraint

    MapVectorXd(residual, nv_stacked_, 1).noalias() = weights_(0) * torque_object.head(nv_stacked_);

    if(jacobians)
    {
        // Do something here
    }

    return true;
}



ceres::CostFunction * CostFunctorObjectTorque::Create(
    int i,
    DataloaderPerson *person_loader,
    DataloaderObject *object_loader,
    const Eigen::VectorXd &weights)
{
    CostFunctorObjectTorque *cost_functor =
        new CostFunctorObjectTorque(i,
                                    person_loader,
                                    object_loader,
                                    weights);
    CostFunctionObjectTorque *cost_function =
        new CostFunctionObjectTorque(cost_functor);

    int nq_pino_object = object_loader->get_nq_pino();
    cost_function->AddParameterBlock(nq_pino_object);
    cost_function->AddParameterBlock(nq_pino_object);
    cost_function->AddParameterBlock(nq_pino_object);

    int nq_contact = object_loader->get_nq_contact();
    cost_function->AddParameterBlock(nq_contact);

    int nq_keypoints = object_loader->get_nq_keypoints();
    cost_function->AddParameterBlock(nq_keypoints);

    int nq_contact_force = object_loader->get_nq_contact_force();
    cost_function->AddParameterBlock(nq_contact_force);

    // Set number of residuals
    cost_function->SetNumResiduals(cost_functor->get_nv_stacked());
    return cost_function;
}

int CostFunctorObjectTorque::get_nv_stacked()
{
    return nv_stacked_;
}
