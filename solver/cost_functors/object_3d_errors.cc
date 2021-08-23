#include "object_3d_errors.h"

CostFunctorObject3dErrors::CostFunctorObject3dErrors(
    int i,
    DataloaderObject *object_loader)
{
    // cout << "CostFunctorObject3dErrors: in" << endl;
    i_ = i; // num of time step
    njoints_object_ = object_loader->get_njoints(); // 1
    // cout << "njoints_object_: " << njoints_object_ << endl;
    if (njoints_object_!= 1)
    {
        assert(false && "Must never happened: njoints_object_!= 1");
    }
    // length of a contact configuration vector
    nq_pino_object_ = object_loader->get_nq_pino();    // 7: base translation + quaternion rotation
    // cout << "nq_pino_object_: " << nq_pino_object_ << endl;
    nq_contact_ = object_loader->get_nq_contact();     // 2: left&right hand contact
    // cout << "nq_contact_: " << nq_contact_ << endl;
    nq_keypoints_ = object_loader->get_nq_keypoints(); // 1: head
    // cout << "nq_keypoints_: " << nq_keypoints_ << endl;
    num_contact_points_ = object_loader->get_num_contacts(); // 2: left&right hand contact
    // cout << "num_contact_points_: " << num_contact_points_ << endl;
    if (nq_keypoints_!=1)
    {
        assert(false && "Must never happened: nq_keypoints_!=1");
    }
    if (num_contact_points_!= 2)
    {
        assert(false && "Must never happened: num_contact_points_!= 2");
    }
    num_keypoints_ = object_loader->get_num_keypoints(); // 1
    // cout << "num_keypoints_: " << num_keypoints_ << endl;
    // Retrieve the reference 3D keypoint positions
    keypoint_3d_positions_ = object_loader->get_keypoint_3d_positions_column(i_); // handle end + head
    object_loader_ = object_loader;
}

bool CostFunctorObject3dErrors::Evaluate(
    double const *const * parameters,
    double * residual,
    double ** jacobians) const
{
    const double *const q_pino_object = *parameters;
    const double *const q_keypoints = *(parameters + 1);
    // Normalize the quaternions and save to the dataloader
    VectorXd q_pino_object_mat = MapConstVector7d(q_pino_object);
    object_loader_->set_config_pino_column(i_, q_pino_object_mat); 
    // Get q_stacked
    VectorXd q_stacked = VectorXd::Zero(nq_pino_object_ + nq_contact_ + nq_keypoints_);
    q_stacked.head<7>() = MapConstVector7d(object_loader_->mutable_config_pino(i_));
    // here we ignore contact positions for they are not used
    for (int i = 0; i < nq_keypoints_; i++)
    {
        q_stacked(i + nq_pino_object_ + nq_contact_) = *(q_keypoints + i);
    }
    pinocchio::forwardKinematics(object_loader_->model_, object_loader_->data_, q_stacked);
    // compute residual blocks
    Vector3d keypoint_position;
    for (int i = 0; i < num_keypoints_ + 1; i++)
    {
        if (i == 0)
        { // The head is the "default" keypoint
            keypoint_position = object_loader_->data_.oMi[1].translation();
        }
        else
        { // other keypoints, e.g. the handle end
            keypoint_position = object_loader_->data_.oMi[njoints_object_ + num_contact_points_ + i].translation();
        }
     for (int k = 0; k < 3; k++)
        {
            residual[3 * i + k] = (keypoint_position(k) - keypoint_3d_positions_(3 * i + k));
        }
    }
 return true;
}

ceres::CostFunction * CostFunctorObject3dErrors::Create(
    int i,
    DataloaderObject *object_loader)
{
    CostFunctorObject3dErrors *cost_functor =
        new CostFunctorObject3dErrors(i, object_loader);
    CostFunctionObject3dErrors *cost_function =
        new CostFunctionObject3dErrors(cost_functor);
    // object config
    int nq_pino_object = object_loader->get_nq_pino();
    cost_function->AddParameterBlock(nq_pino_object);
    // keypoint config
    int nq_keypoints = object_loader->get_nq_keypoints();
    cost_function->AddParameterBlock(nq_keypoints);
    // number of residuals
    int num_keypoints = object_loader->get_num_keypoints();
    cost_function->SetNumResiduals(3 * (num_keypoints + 1));
    return cost_function;
}
