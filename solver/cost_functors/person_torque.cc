
#include "person_torque.h"
#include "pinocchio/algorithm/rnea.hpp"

CostFunctorPersonTorque::CostFunctorPersonTorque(int i,
                        DataloaderPerson *person_loader,
                        DataloaderObject *object_loader,
                        const VectorXd & weights)
{
  i_ = i; // number of timestep (eq. video frame index)
  weights_ = weights; // weights before the cost terms
  person_loader_ = person_loader;
  object_loader_ = object_loader;
  nq_person_ = person_loader->get_nq(); // length of the configuration vector q_{person or object} (with axis-angles)
                                        // get person-object & person-ground contact information
  contact_mapping_ = person_loader->get_contact_mapping();
  MatrixXi::ColXpr contact_states = person_loader->get_contact_states_column(i_);
  int njoints_person = (int)contact_states.rows();
  for (int j = 0; j < njoints_person; j++)
  {
    if (contact_states(j) == 1) // 1: joint j is in contact with the object
    {
      object_contact_joints_.push_back(j);
      //std::cout << "**person_torque.cc** object_contact_joint : " << j << std::endl; // debugging
    }
    else if (contact_states(j) == 2) // 2: joint j is in contact with the ground
    {
      ground_contact_joints_.push_back(j);
      //std::cout << "**person_torque.cc** ground_contact_joint : " << j << std::endl; // debugging
    }
  }
  num_object_contact_joints_ = (int)object_contact_joints_.size();
  num_ground_contact_joints_ = (int)ground_contact_joints_.size();

  friction_cone_generators_ = person_loader_->get_friction_cone_generators();
  friction_cone_generators_left_foot_ = person_loader_->get_friction_cone_generators_left_foot();
  friction_cone_generators_right_foot_ = person_loader_->get_friction_cone_generators_right_foot();
  // save dt_ and dt_square_
  dt_ = person_loader->get_dt();
  dt_square_ = dt_ * dt_;

  debug_ = false;
  if (debug_)
  {
      std::cout << "**person_torque.cc** i: " << i_ << std::endl;
      std::cout << "**person_torque.cc** nq_person_ : " << nq_person_ << std::endl;
      std::cout << "**person_torque.cc** weights_: " << weights_.transpose() << std::endl;
      std::cout << "**person_torque.cc** num_object_contact_joints : " << num_object_contact_joints_ << std::endl;
      std::cout << "**person_torque.cc** num_ground_contact_joints : " << num_ground_contact_joints_ << std::endl;
      std::cout << "**person_torque.cc** contact_mapping_ : " << contact_mapping_.transpose() << std::endl; // DEBUG
      std::cout << "**person_torque.cc** dt_ == " << dt_ << std::endl;
      std::cout << "**person_torque.cc** dt_square_ == " << dt_square_ << std::endl;
      std::cout << "**person_torque.cc** friction_cone_generators_ == " << std::endl;
      std::cout << friction_cone_generators_ << std::endl;
      std::cout << "**person_torque.cc** friction_cone_generators_left_foot_ == " << std::endl;
      std::cout << friction_cone_generators_left_foot_ << std::endl;
      std::cout << "**person_torque.cc** friction_cone_generators_right_foot_ == " << std::endl;
      std::cout << friction_cone_generators_right_foot_ << std::endl;
  }
}

bool CostFunctorPersonTorque::Evaluate(double const *const * parameters,
                                       double * residual,
                                       double ** jacobians) const
{
  const double *const q_person = parameters[0];
  const double *const q_person_minus1 = parameters[1];
  const double *const q_person_minus2 = parameters[2];
  const double * q_contact_force;
  const double * q_ground_friction;

  if (num_object_contact_joints_>0 && num_ground_contact_joints_>0)
  {
    q_contact_force  = parameters[3];
    q_ground_friction= parameters[4];
  }
  else if (num_object_contact_joints_>0)
  {
    q_contact_force  = parameters[3];
  }
  else if (num_ground_contact_joints_>0)
  {
    q_ground_friction = parameters[3];
  }
  else
    assert(false && "Must never happened");

  typedef Eigen::Map<const Eigen::VectorXd> ConstMapVectorXd;
  typedef Eigen::Map<Eigen::VectorXd> MapVectorXd;
  // compute q_person_pino
  Eigen::VectorXd q_person_mat = ConstMapVectorXd(q_person,person_loader_->get_nq(),1);
  person_loader_->UpdateConfigPino(i_, q_person_mat);
  Eigen::VectorXd q_person_pino = ConstMapVectorXd(person_loader_->mutable_config_pino(i_),person_loader_->get_nq_pino(),1);

  // compute person velocity and acceleration from backward differentiation
  Eigen::VectorXd q_person_minus1_mat = ConstMapVectorXd(q_person_minus1,person_loader_->get_nq(),1);
  Eigen::VectorXd q_person_minus2_mat = ConstMapVectorXd(q_person_minus2,person_loader_->get_nq(),1);
  // VectorXd vq_person_mat = VectorXd::Zero(nq_person_);
  // VectorXd vq_dot_person_mat = VectorXd::Zero(nq_person_);
  Eigen::VectorXd vq_person_mat = (q_person_mat - q_person_minus1_mat) / dt_;
  Eigen::VectorXd vq_dot_person_mat = (q_person_mat - 2*q_person_minus1_mat + q_person_minus2_mat) / dt_square_;

  const pinocchio::Model & model = person_loader_->model_;
  pinocchio::Data & data = person_loader_->data_;

  pinocchio::container::aligned_vector<Data::Force> fext((size_t)model.njoints,Data::Force::Zero());
  typedef Eigen::Map<Vector6d> MapVector6d;
  typedef Eigen::Map<const Vector6d> ConstMapVector6d;

  // compute object contact torque
  if (num_object_contact_joints_ > 0)
  {
    for (int i = 0; i < num_object_contact_joints_; i++)
    {
      int j = object_contact_joints_[(size_t)i]; // joint j in contact with object
      int fid = contact_mapping_[j];     // ID of the contact frame allocated to joint j
      ConstMapVector6d fmap(q_contact_force + 6 * (fid - 1),6,1);
      pinocchio::ForceRef< ConstMapVector6d > fref(fmap);
      fext[(size_t)(j + 1)] += fref;
    }
  }

  // compute ground contact torque
  if (num_ground_contact_joints_ > 0)
  {

    double coef;
    for (int i = 0; i < num_ground_contact_joints_; i++)
    {
      pinocchio::Force ground_contact_force = pinocchio::Force::Zero();

      int j = ground_contact_joints_[(size_t)i];  // joint j in contact with ground
      int fid = contact_mapping_[j]; // ID of the contact frame allocated to joint j
      const pinocchio::JointIndex joint_id = (pinocchio::JointIndex)j+1;
      // cout << "j == " << j << endl;
      if (j != 3 && j != 7)
      {
        // express ground contact force in frame c_prime
        // the frame c_prime is defined by contact point c and world axis
        for (int k = 0; k < 4; k++)
        {
          coef = *(q_ground_friction + 4 * (fid - 1) + k);
          pinocchio::ForceRef<const Matrix6x::ConstColXpr> fref(friction_cone_generators_.col(k));
          ground_contact_force += coef * fref;
        }
        // get cMj: the displacement of frame j wrt the contact frame c
        // note that frame c is defined s.t. c share the origin of joint j, and is parallel to the world frame o
        // thus cMj is only rotation

        const Matrix3d & oRj = person_loader_->data_.oMi[joint_id].rotation();
        const pinocchio::SE3 cMj = pinocchio::SE3(oRj, Vector3d::Zero(3)); // just change frame orientation
        fext[joint_id] += cMj.actInv(ground_contact_force);

      }
      else
      {
        if (j == 3)
        { // left ankle joint
          for (int k = 0; k < 16; k++)
          {
            coef = *(q_ground_friction + 4 * (fid - 1) + k);
            pinocchio::ForceRef<const Matrix6x::ConstColXpr> fref(friction_cone_generators_left_foot_.col(k));
            ground_contact_force += coef * fref;
          }
        }
        else if (j==7)
        {
          // right ankle joint
          for (int k = 0; k < 16; k++)
          {
            coef = *(q_ground_friction + 4 * (fid - 1) + k);
            pinocchio::ForceRef<const Matrix6x::ConstColXpr> fref(friction_cone_generators_right_foot_.col(k));
            ground_contact_force += coef * fref;
          }
        }
        else
        {
          assert(false && "must never happened");
        }

        fext[joint_id] += ground_contact_force;
      }
    }
  }

  const VectorXd & torque_person = pinocchio::rnea(model,data,q_person_pino,vq_person_mat,vq_dot_person_mat,fext);

  MapVectorXd(residual,nq_person_,1).head<6>().noalias() = weights_(0) * torque_person.head<6>();
  MapVectorXd(residual,nq_person_,1).tail(nq_person_-6).noalias() = weights_(1) * torque_person.tail(nq_person_-6);

  if(jacobians)
  {
    // Do something here
  }

  return true;
}

ceres::CostFunction * CostFunctorPersonTorque::Create(int i,
                                                      DataloaderPerson *person_loader,
                                                      DataloaderObject *object_loader,
                                                      const VectorXd &weights,
                                                      bool *has_object_contact,
                                                      bool *has_ground_contact)
{
  CostFunctorPersonTorque *cost_functor =
  new CostFunctorPersonTorque(i,
                              person_loader,
                              object_loader,
                              weights);
  CostFunctionPersonTorque *cost_function =
  new CostFunctionPersonTorque(cost_functor);
  // person config parameters
  int nq_person = person_loader->get_nq();
  cost_function->AddParameterBlock(nq_person);
  cost_function->AddParameterBlock(nq_person);
  cost_function->AddParameterBlock(nq_person);
  // object contact force
  *has_object_contact = cost_functor->has_object_contact();
  if (*has_object_contact)
  {
    int nq_contact_force = object_loader->get_nq_contact_force();
    cost_function->AddParameterBlock(nq_contact_force);
  }
  // ground friction force
  *has_ground_contact = cost_functor->has_ground_contact();
  if (*has_ground_contact)
  {
      //std::cout << "*** person_torque.cc: has_ground_contact is true" << std::endl;
    int nq_ground_friction = person_loader->get_nq_ground_friction();
    cost_function->AddParameterBlock(nq_ground_friction);
    //std::cout << "*** person_torque.cc: nq_ground_friction: " << nq_ground_friction << std::endl;
  }
  // number of residuals
  cost_function->SetNumResiduals(nq_person);
  return cost_function;
}
