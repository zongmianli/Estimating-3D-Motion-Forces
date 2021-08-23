#ifndef __PERSON_DATA_H__
#define __PERSON_DATA_H__

#include <ceres/ceres.h>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "../dataloader/dataloader_person.h"
#include "../camera.h"

using namespace Eigen;
using namespace std;
using namespace pinocchio;

// Mapping from our joints to Deepercut joints
// 0 Right ankle  7
// 1 Right knee   6
// 2 Right hip    5
// 3 Left hip     1
// 4 Left knee    2
// 5 Left ankle   3
// 6 Right wrist  22
// 7 Right elbow  21
// 8 Right shoulder 20
// 9 Left shoulder  15
// 10 Left elbow    16
// 11 Left wrist    17
// 12 Neck           -
// 13 Head top       -

// Mapping from our joints to Openpose Joints
// 0 nose          OF9
// 1 neck           -
// 2 right shoulder 20
// 3 right elbow    21
// 4 right wrist    22
// 5 left shoulder  15
// 6 left elbow     16
// 7 left wrist     17
// 8 right hip       5
// 9 right knee      6
// 10 right ankle    7
// 11 left hip       1
// 12 left knee      2
// 13 left ankle     3
// 14 right eye   OF10
// 15 left eye    OF11
// 16 right ear   OF12
// 17 left ear    OF13

// Mapping from Cocoplus Joints to Openpose joints
// 0 nose           14
// 1 neck           12
// 2 right shoulder  8
// 3 right elbow     7
// 4 right wrist     6
// 5 left shoulder   9
// 6 left elbow     10
// 7 left wrist     11
// 8 right hip       2
// 9 right knee      1
// 10 right ankle    0
// 11 left hip       3
// 12 left knee      4
// 13 left ankle     5
// 14 right eye     16
// 15 left eye      15
// 16 right ear     18
// 17 left ear      17

// Openpose joints,   parent Pinocchio joints     relative position (to compute)
// 0 nose             13 spine 4 (head) 
// 1 neck             11 spine 2
// 2 right shoulder   19 r_scapula
// 3 right elbow      20 r_shoulder
// 4 right wrist      21 r_elbow
// 5 left shoulder    14 l_scapula
// 6 left elbow       15 l_shoulder
// 7 left wrist       16 l_elbow
// 8 right hip         0 pelvis
// 9 right knee        5 r_hip
// 10 right ankle      6 r_knee
// 11 left hip         0 pelvis
// 12 left knee        1 l_hip
// 13 left ankle       2 l_knee
// 14 right eye       13 spine 4 (head) 
// 15 left eye        13 spine 4 (head) 
// 16 right ear       13 spine 4 (head) 
// 17 left ear        13 spine 4 (head) 
// joint_ids_smpl = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]
// parent_joint_ids = [13, 11, 19, 20, 21, 14, 15, 16, 0, 5, 6, 0, 1, 2, 13, 13, 13, 13]
// loop over 18 keypoints
// keypoint_positions_wrt_parent_joint = np.zeros((18, 3))
// for each keypoint, compute: 1) its relative translation wrt parent joint

struct CostFunctorPersonData
{
  typedef ceres::DynamicNumericDiffCostFunction<CostFunctorPersonData> CostFunctionPersonData;

  CostFunctorPersonData(int i,
                        DataloaderPerson *person_loader,
                        Camera *camera,
                        bool cam_variable,
                        bool update_6d_basis_only,
                        // bool update_69d_pose_only,
                        bool measure_torso_joints_only)
  {
    i_ = i; // num of time step
    person_loader_ = person_loader;
    camera_ = camera;
    cam_variable_ = cam_variable;
    if (update_6d_basis_only)
    {
      nq_ = 6; // the 6d pose of human basis joint
      // if (update_69d_pose_only)
      // {
      //   cout << "warning: update_6d_basis_only and update_69d_pose_only are both true!" << endl;
      // }
    }
    // else if (update_69d_pose_only)
    // {
    //   nq_ = 69; // axis-angles of 23 human joints (excluding the basis joint)
    // }
    else
    {
      nq_ = person_loader_->get_nq(); // 75d vector
    }
    //cout << "nq_==" << nq_ << endl;
    vector<int> joint_ids_temp;
    int njoints_temp;
    if (measure_torso_joints_only)
    {
      int joints_interest[] = {0, 1, 2, 5, 8, 11, 14, 15, 16, 17};
      njoints_temp = 10;
      for (int k = 0; k < njoints_temp; k++)
      {
        joint_ids_temp.push_back(joints_interest[k]);
      }
    }
    else
    {
      int joints_interest[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
      njoints_temp = 18;
      for (int k = 0; k < njoints_temp; k++)
      {
        joint_ids_temp.push_back(joints_interest[k]);
      }
    }
    // vector<int> joint_ids_temp;
    // int njoints_temp;
    // if (torso_only)
    // {
    //   cout << "(torso) ";
    //   nq_ = 6; // human basis configuration vector is 6-dimensional
    //   int joints_interest[] = {0, 1, 2, 5, 8, 11, 14, 15, 16, 17};
    //   njoints_temp = 10;
    //   for(int k=0; k<njoints_temp; k++)
    //   {
    //     joint_ids_temp.push_back(joints_interest[k]);
    //   }
    // }
    // else
    // {
    //   nq_ = person_loader_->get_nq(); // 75-dimensional vector
    //   int joints_interest[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    //   njoints_temp = 18;
    //   for(int k=0; k<njoints_temp; k++)
    //   {
    //     joint_ids_temp.push_back(joints_interest[k]);
    //   }
    // }
    // 
    q_init_ = person_loader_->get_config_column(i_);
    joint_2d_positions_ = person_loader_->get_joint_2d_positions_column(i_);
    // fill the list of detected openpose joints, joint_ids_
    int joint_id;
    double confidence_score;
    for (int k=0; k<njoints_temp; k++)
    {
      joint_id = joint_ids_temp[(size_t)k];
      confidence_score = joint_2d_positions_(3 * joint_id + 2);
      // add joint_id only if the joint is detected by Openpose
      if (confidence_score > 0.05)
      {
        joint_ids_.push_back(joint_id);
      }
    }
    njoints_ = (int)joint_ids_.size();
    std::cout << "(" << std::setw(2) << std::setfill('0') << njoints_ << " pts) ";
  }

  bool Evaluate(double const *const * parameters,
                double * residual,
                double ** jacobians) const
  {
    //cout << "ckpt: reprojection error" << endl;
    const double *const q = parameters[0];
    
    VectorXd q_mat;
    if (nq_ == 6) //only the free flyer
    {
      typedef Eigen::Matrix<double,6,1> Vector6d;
      q_mat = q_init_;
      q_mat.head<6>() = Eigen::Map<const Vector6d>(q);
    }
    // else if (nq_ == 69)
    // {
    //   Matrix<double, 69, 1> q_pose(q);
    //   q_mat = q_init_;
    //   q_mat.block<69, 1>(6, 0) = q_pose;
    // }
    else if (nq_ == person_loader_->get_nq())
    {
//      Matrix<double, 75, 1> q_mat_in(q);
      q_mat = Eigen::Map<const VectorXd>(q,nq_,1);
    }
    else
    {
      LOG(FATAL) << "unknown value for nq_" << endl;
    }
    
    if (cam_variable_)
    { // update camera projection matrix with new focal length
      const double *const x_cam = parameters[1];
      const double & focal_length = x_cam[0];
      camera_->complete_focal_length(focal_length);
      camera_->UpdateProjectionMatrix();
    }
    
    // convert 3D axis-angles to a 4D quaternions
    person_loader_->UpdateConfigPino(i_, q_mat);
    VectorXd q_pino = Eigen::Map<VectorXd>(person_loader_->mutable_config_pino(i_),person_loader_->get_nq_pino(),1);
    assert(q_pino.size() == person_loader_->get_nq_pino() && "q_pino of wrong dimension");
    
    // forwardKinematics
    pinocchio::framesForwardKinematics(person_loader_->model_, person_loader_->data_, q_pino);
    

    double confidence_score;
    Vector2d projected_joint;
    for (int i = 0; i < njoints_; i++)
    {
      int joint_id = joint_ids_[(size_t)i]; // the id number of an openpose joint
      confidence_score = joint_2d_positions_(3 * joint_id + 2);
      //cout << "confidence_score==" << confidence_score << endl;
      const Vector3d & joint_position_3d = person_loader_->data_.oMf[(size_t)joint_id + 9].translation(); // the 8 first operational frames are the foot contact points
      projected_joint = camera_->Project(joint_position_3d);
      //cout << "projected_joint==" << projected_joint << endl;
      for (int k = 0; k < 2; k++)
      {
        residual[2 * i + k] = confidence_score *
            (projected_joint(k) - joint_2d_positions_(3 * joint_id + k));
      }
    }
    
    if(jacobians)
    {
      // do nothing
    }
    
    return true;
  }

  bool operator()(double const *const *parameters, double *residual) const
  {
    return Evaluate(parameters,residual,NULL);
  }

  static ceres::CostFunction *Create(int i,
                                     DataloaderPerson *person_loader,
                                     Camera *camera,
                                     bool cam_variable,
                                     bool update_6d_basis_only,
                                    //  bool update_69d_pose_only,
                                     bool measure_torso_joints_only)
  {
    CostFunctorPersonData *cost_functor =
        new CostFunctorPersonData(i,
                                  person_loader,
                                  camera,
                                  cam_variable,
                                  update_6d_basis_only,
                                  // update_69d_pose_only,
                                  measure_torso_joints_only);
    CostFunctionPersonData *cost_function =
        new CostFunctionPersonData(cost_functor);
    // person config parameters
    int nq = cost_functor->get_nq();
    cost_function->AddParameterBlock(nq);
    if (cam_variable)
    {
      cost_function->AddParameterBlock(1);
    }
    // number of residuals
    int njoints = cost_functor->get_njoints();
    //cout << "njoints==" << njoints << endl;
    cost_function->SetNumResiduals(2 * njoints);
    return cost_function;
  }

  int get_njoints() const
  {
    return njoints_;
  }

  int get_nq() const
  {
    return nq_;
  }

private:
  int i_;
  int nq_;
  int njoints_;
  vector<int> joint_ids_;
  VectorXd q_init_; // initial human configuration vector (75d)
  VectorXd joint_2d_positions_;
  Camera *camera_;
  bool cam_variable_;
  DataloaderPerson *person_loader_;
};

#endif // ifndef __PERSON_DATA_H__
