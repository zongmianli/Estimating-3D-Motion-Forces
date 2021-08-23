
#include "contact_ground_spatial.h"
#include "pinocchio/algorithm/frames.hpp"

CostFunctorContactGroundSpatial::CostFunctorContactGroundSpatial(int i,
                                                                 DataloaderPerson *person_loader,
                                                                 DataloaderObject *ground_loader,
                                                                 bool update_6d_basis_only)
{
  // cout << "ckpt 1." << endl;
  i_ = i;         // num of time step

  if (update_6d_basis_only)
  {
    nq_person_ = 6; // the 6d pose of human basis joint
  }
  else
  {
    nq_person_ = person_loader->get_nq(); // 75d vector
  }
  q_person_init_ = person_loader->get_config_column(i_);

  VectorXi contact_states = person_loader->get_contact_states_column(i_);
  int nj = (int)contact_states.rows();
  num_contact_points_ = 0;
  for (int j = 0; j < nj; j++)
  {
    if (contact_states(j) == 2)
    {
      contact_joints_.push_back(j);
      if (j == 3 || j == 7)
      { // left and right ankle (foot) have 4 ground contact points each
        num_contact_points_ += 4;
      }
      else
      { // other joints have at most one ground contact point
        num_contact_points_++;
      }
    }
  }
  num_contact_joints_ = (int)contact_joints_.size();
  // total number of ground contact points (frames)
  // note that not all the OP contact frames are used in the current time step
  total_num_contact_points_ = ground_loader->model_.nframes - 1; // number of ground contact points
                                                                 // get contact mapping
  contact_mapping_ = person_loader->get_contact_mapping();
  // cout << "num_contact_joints_: " << num_contact_joints_ << endl;
  // cout << "total_num_contact_points_: " << total_num_contact_points_ << endl;
  // cout << "contact_mapping_: " << contact_mapping_ << endl;
  
  // save dataloaders
  person_loader_ = person_loader;
  ground_loader_ = ground_loader;
  // cout << "ckpt 2." << endl;
}

bool CostFunctorContactGroundSpatial::Evaluate(double const *const * parameters,
                                               double * residual,
                                               double ** jacobians) const
{
  const double *const q_person = parameters[0];
  const double *const q_ground = parameters[1];
  const double *const q_contact = parameters[2];
  
  typedef Eigen::Map<const Eigen::VectorXd> ConstMapVectorXd;
  
  // convert q_person to q_person_pino
  //Eigen::VectorXd q_person_mat = ConstMapVectorXd(q_person,person_loader_->get_nq(),1);

  Eigen::VectorXd q_person_mat;
  if (nq_person_ == 6) //only the free flyer
  {
    typedef Eigen::Matrix<double,6,1> Vector6d;
    q_person_mat = q_person_init_;
    q_person_mat.head<6>() = Eigen::Map<const Vector6d>(q_person);
  }
  else if (nq_person_ == person_loader_->get_nq())
  {
    q_person_mat = Eigen::Map<const VectorXd>(q_person,nq_person_,1);
  }
  else
  {
    LOG(FATAL) << "unknown value for nq_person_" << endl;
  }

  person_loader_->UpdateConfigPino(i_, q_person_mat);
  Eigen::VectorXd q_person_pino = ConstMapVectorXd(person_loader_->mutable_config_pino(i_),person_loader_->get_nq_pino(),1);
  
  // convert q_ground to q_ground_pino
  Matrix<double, 6, 1> q_ground_mat_in(q_ground);
  VectorXd q_ground_mat = q_ground_mat_in;
  ground_loader_->UpdateConfigPino(i_, q_ground_mat);
  Matrix<double, 7, 1> q_ground_pino(ground_loader_->mutable_config_pino(i_));
  
  // update person joint positions via forward kinematics
  pinocchio::framesForwardKinematics(person_loader_->model_, person_loader_->data_, q_person_pino);
  
  // update ground contact point positions
  for (int n = 0; n < total_num_contact_points_; n++)
  {
    SE3::Vector3 & contact_pos = ground_loader_->model_.frames[(size_t)(n + 1)].placement.translation();
    contact_pos(0) = q_contact[2 * n];     // x
    contact_pos(2) = q_contact[2 * n + 1]; // z
  }
  pinocchio::framesForwardKinematics(ground_loader_->model_, ground_loader_->data_, q_ground_pino);
  
  // compute residual blocks
  Vector3d joint_position;
  Vector3d contact_position;
  int j, fid;
  int count_contact = 0;
  for (int c = 0; c < num_contact_joints_; c++)
  {
    j = contact_joints_[(size_t)c];
    fid = contact_mapping_(j);
    if (j != 3 && j != 7)
    {
      // if joint j is on the ground
      // get joint j's position (j+1 in Pinocchio)
      const Vector3d & joint_position = person_loader_->data_.oMi[(size_t)(j + 1)].translation();
      // get joint j's contact position on the ground
      const Vector3d & contact_position = ground_loader_->data_.oMf[(size_t)fid].translation();
      
      Eigen::Map<Vector3d>(residual + (3 * count_contact)) = joint_position - contact_position;
      count_contact++;
    }
    else if (j == 3)
    { // left ankle joint
      for (int n = 0; n < 4; n++)
      {
        const Vector3d & joint_position = person_loader_->data_.oMf[(size_t)(1 + n)].translation();
        const Vector3d & contact_position = ground_loader_->data_.oMf[(size_t)(fid + n)].translation();
        
        Eigen::Map<Vector3d>(residual + (3 * count_contact)) = joint_position - contact_position;
        count_contact++;
      }
    }
    else
    { // right ankle joint
      for (int n = 0; n < 4; n++)
      {
        const Vector3d & joint_position = person_loader_->data_.oMf[(size_t)(5 + n)].translation();
        const Vector3d & contact_position = ground_loader_->data_.oMf[(size_t)(fid + n)].translation();
        
        Eigen::Map<Vector3d>(residual + (3 * count_contact)) = joint_position - contact_position;
        count_contact++;
      }
    }
  }
  
  if(jacobians)
  {
    
  }
  return true;
}

ceres::CostFunction * CostFunctorContactGroundSpatial::Create(int i,
                                                              DataloaderPerson *person_loader,
                                                              DataloaderObject *ground_loader,
                                                              bool update_6d_basis_only)
{
  CostFunctorContactGroundSpatial *cost_functor =
  new CostFunctorContactGroundSpatial(i,
                                      person_loader,
                                      ground_loader,
                                      update_6d_basis_only);
  CostFunctionContactGroundSpatial *cost_function =
  new CostFunctionContactGroundSpatial(cost_functor);
  int num_residuals = 3 * cost_functor->get_num_contact_points();
  if (num_residuals > 0)
  {
    // add parameter blocks
    int nq_person = cost_functor->get_nq_person();
    int nq_ground = ground_loader->get_nq();
    int nq_contact = ground_loader->get_nq_contact();
    cost_function->AddParameterBlock(nq_person);
    cost_function->AddParameterBlock(nq_ground);
    cost_function->AddParameterBlock(nq_contact);
    cost_function->SetNumResiduals(num_residuals);
    return cost_function;
  }
  return NULL;
}
