#include "dataloader.h"

Dataloader::Dataloader(pinocchio::Model &model,
                       pinocchio::Data &data,
                       const Eigen::MatrixXd &decoration)
    : model_(model),
      data_(data),
      decoration_(decoration.cast<int>()) {}

// Updates the time difference, dt_, number of time steps, nt_,
// configuration vectors, config_ and config_pino_,
// configuration velocities, velopcity_
void Dataloader::LoadConfig(const Eigen::MatrixXd &config, double fps)
{
    if (config.cols() != nt_)
    {
        LOG(FATAL) << "config.cols() != nt_!" << std::endl;
    }
    if (config.rows() != nq_)
    {
        LOG(FATAL) << "config.rows() != nq_!" << std::endl;
    }
    config_ = config;
    fps_ = fps;
    dt_ = 1.0/fps;
    velocity_ = Eigen::MatrixXd::Zero(nq_, nt_);
    UpdateConfigVelocity();
    config_pino_ = Eigen::MatrixXd::Zero(nq_pino_, nt_);
    UpdateConfigPino();
}

// Converts a 3 x N matrix whose columns are axis-angle rotation vectors,
// to a 4 x N matrix of unit quaternions under the (x,y,z,w) convention.
Eigen::MatrixXd Dataloader::AxisAngleToQuat(const Eigen::MatrixXd & axis_angles)
{
    int num_cols = (int)axis_angles.cols();
    Eigen::MatrixXd quats = Eigen::MatrixXd::Zero(4, num_cols);
    // get rotation axis and angles
    Eigen::MatrixXd angles = axis_angles.colwise().norm();
    Eigen::MatrixXd axis = axis_angles.cwiseQuotient(angles.replicate(3, 1));
    // multiply the three first cols by sin(angle/2)
    quats.block(0, 0, 3, num_cols) =
        axis.cwiseProduct((angles / 2).array().sin().matrix().replicate(3, 1));
    // last col is cos(angle/2)
    quats.row(3) = (angles / 2).array().cos();
    return quats;
}

// Converts a 4 x N matrix whose columns are quaternions under
// the (x,y,z,w) convention, to a 3 x N matrix of axis-angle rotations.
// The function normalizes the quaternions to norm 1 before convertion.
Eigen::MatrixXd Dataloader::QuatToAxisAngle(const Eigen::MatrixXd & quats)
{
    int num_cols = (int)quats.cols();
    // normalzie the columns of quats
    Eigen::MatrixXd norms = quats.colwise().norm();
    Eigen::MatrixXd quats_temp = quats.cwiseQuotient(norms.replicate(4, 1));
    // get thetas
    Eigen::MatrixXd thetas = quats_temp.row(3).array().acos().matrix() * 2;
    // get rotation axis u
    Eigen::MatrixXd u = quats_temp.block(0, 0, 3, num_cols).cwiseQuotient((thetas / 2).array().sin().matrix().replicate(3, 1));
    // get axis-angle vectors
    return u.cwiseProduct(thetas.replicate(3, 1));
}

// Converts axis-angle blocks in the input matrix config_ (of size nq x nf)
// to quaternions, and then returns a nqPino x nf matrix config_pino_.
void Dataloader::UpdateConfigPino()
{
    int joint_type;
    int joint_index;
    int joint_index_pino;
    for (int j = 0; j < njoints_; j++)
    {
        joint_type = decoration_(j, 2);
        joint_index = decoration_(j, 3);      // the index in q that is used to read to the joint
        joint_index_pino = decoration_(j, 4); // the index in q_pino ...
        if (joint_type == 1)
        { // free-floating joint: copy linear, convert angular to quaternion
            config_pino_.block(joint_index_pino, 0, 3, nt_) =
                config_.block(joint_index, 0, 3, nt_);
            config_pino_.block(joint_index_pino + 3, 0, 4, nt_) =
                AxisAngleToQuat(config_.block(joint_index + 3, 0, 3, nt_));
        }
        else if (joint_type == 2)
        { // spherical joint: convert axis-angle to quaternion
            config_pino_.block(joint_index_pino, 0, 4, nt_) =
                AxisAngleToQuat(config_.block(joint_index, 0, 3, nt_));
        }
        else
        { // all other types of joint: just copy the configuration
            config_pino_.block(joint_index_pino, 0, 1, nt_) =
                config_.block(joint_index, 0, 1, nt_);
        }
    }
}

void Dataloader::UpdateConfigPino(const Eigen::MatrixXd &config)
{
    config_ = config;
    UpdateConfigPino();
}

// Converts the axis-angles in the Eigen vector q_mat to quaternions
// and then saves the resultant configuration vector to time step t
void Dataloader::UpdateConfigPino(int t, Eigen::VectorXd &q_mat)
{
    int joint_type;
    int joint_index;
    int joint_index_pino;
    for (int j = 0; j < njoints_; j++)
    {
        joint_type = decoration_(j, 2);
        joint_index = decoration_(j, 3);      // the index in q that is used to read to the joint
        joint_index_pino = decoration_(j, 4); // the index in q_pino ...
        if (joint_type == 1)
        { // free-floating joint: copy linear, convert angular to quaternion
            config_pino_.block<3, 1>(joint_index_pino, t) = q_mat.block<3, 1>(joint_index, 0);
            config_pino_.block<4, 1>(joint_index_pino + 3, t) =
                AxisAngleToQuat(q_mat.block<3, 1>(joint_index + 3, 0));
        }
        else if (joint_type == 2)
        { // spherical joint: convert axis-angle to quaternion
            config_pino_.block<4, 1>(joint_index_pino, t) =
                AxisAngleToQuat(q_mat.block<3, 1>(joint_index, 0));
        }
        else
        { // all other types of joint: just copy the configuration
            config_pino_(joint_index_pino, t) = q_mat(joint_index, 0);
        }
    }
}

// Do the inverse of UpdateConfig()
// Converts quaternion blocks blocks in the input matrix config_pino_ (of size nqPino x nf)
// to axis-angle, and then returns a nq x nf matrix config_.
void Dataloader::UpdateConfig()
{
    int joint_type;
    int joint_index;
    int joint_index_pino;
    for (int j = 0; j < njoints_; j++)
    {
        joint_type = decoration_(j, 2);
        joint_index = decoration_(j, 3);      // the index in q that is used to read to the joint
        joint_index_pino = decoration_(j, 4); // the index in q_pino ...
        if (joint_type == 1)
        { // free-floating joint: copy linear, convert angular to quaternion
            config_.block(joint_index, 0, 3, nt_) =
                config_pino_.block(joint_index_pino, 0, 3, nt_);
            config_.block(joint_index + 3, 0, 3, nt_) =
                QuatToAxisAngle(config_pino_.block(joint_index_pino + 3, 0, 4, nt_));
        }
        else if (joint_type == 2)
        { // spherical joint: convert axis-angle to quaternion
            config_.block(joint_index, 0, 3, nt_) =
                QuatToAxisAngle(config_pino_.block(joint_index_pino, 0, 4, nt_));
        }
        else
        { // all other types of joint: just copy the configuration
            config_.block(joint_index, 0, 1, nt_) =
                config_pino_.block(joint_index_pino, 0, 1, nt_);
        }
    }
}

void Dataloader::UpdateConfig(const Eigen::MatrixXd &config_pino)
{
    config_pino_ = config_pino;
    UpdateConfig();
}

void Dataloader::UpdateConfig(int t, Eigen::VectorXd &q_pino_mat)
{
    int joint_type;
    int joint_index;
    int joint_index_pino;
    for (int j = 0; j < njoints_; j++)
    {
        joint_type = decoration_(j, 2);
        joint_index = decoration_(j, 3);      // the index in q that is used to read to the joint
        joint_index_pino = decoration_(j, 4); // the index in q_pino ...
        if (joint_type == 1)
        { // free-floating joint: copy linear, convert angular to quaternion
            config_pino_.block<3, 1>(joint_index, t) = q_pino_mat.block<3, 1>(joint_index_pino, 0);
            config_pino_.block<3, 1>(joint_index + 3, t) =
                QuatToAxisAngle(q_pino_mat.block<4, 1>(joint_index_pino + 3, 0));
        }
        else if (joint_type == 2)
        { // spherical joint: convert axis-angle to quaternion
            config_pino_.block<3, 1>(joint_index, t) =
                QuatToAxisAngle(q_pino_mat.block<4, 1>(joint_index_pino, 0));
        }
        else
        { // all other types of joint: just copy the configuration
            config_pino_(joint_index, t) = q_pino_mat(joint_index_pino, 0);
        }
    }
}

// Computes configuration velocities using backward difference,
// i.e., v_q[i] = (q[i] - q[i-1])/dt. The velocity at time step 0
// is copied from time step 1.
void Dataloader::UpdateConfigVelocity()
{
    velocity_.rightCols(nt_ - 1) = (config_.rightCols(nt_ - 1) - config_.leftCols(nt_ - 1)) / dt_;
    velocity_.col(0) = velocity_.col(1);
}

void Dataloader::UpdateConfigVelocity(const Eigen::MatrixXd &config)
{
    config_ = config;
    UpdateConfigVelocity();
}

// Computes configuration velocity at one time step using backward difference.
// The input time step t must lie between 0 and nt_-1.
void Dataloader::UpdateConfigVelocity(int t)
{
    if (t >= 1 && t <= nt_ - 1)
    {
        velocity_.col(t) = (config_.col(t) - config_.col(t - 1)) / dt_;
    }
    else if (t == 0)
    {
        velocity_.col(0) = (config_.col(1) - config_.col(0)) / dt_; // == velocity_.col(1);
    }
    else
    {
        LOG(FATAL) << "incorrect frame number!" << std::endl;
    }
}

// Class accessors and mutators
int Dataloader::get_nq()
{
    return nq_;
}
int Dataloader::get_njoints()
{
    return njoints_;
}
int Dataloader::get_nq_pino()
{
    return nq_pino_;
}
int Dataloader::get_nq_ground_friction()
{
    return nq_ground_friction_;
}
int Dataloader::get_nt()
{
    return nt_;
}
double Dataloader::get_fps()
{
    return fps_;
}
double Dataloader::get_dt()
{
    return dt_;
}
Eigen::MatrixXd Dataloader::get_decoration()
{
    return decoration_.cast<double>(); // EigenPy does not support Eigen::MatrixXi
}
Eigen::MatrixXd & Dataloader::get_config()
{
    return config_;
}
Eigen::MatrixXd::ColXpr Dataloader::get_config_column(int i)
{
    return config_.col(i);
}
void Dataloader::set_config(const Eigen::MatrixXd &config)
{
    config_ = config;
}
Eigen::MatrixXd & Dataloader::get_velocity()
{
    return velocity_;
}
Eigen::MatrixXd & Dataloader::get_config_pino()
{
    return config_pino_;
}

// Update config_pino_
// NOTE: this function does not normalize quaternions
void Dataloader::set_config_pino(const Eigen::MatrixXd &config_pino)
{
    config_pino_ = config_pino;
}

// Normalize the quaternions in the input configuration vector.
// And then update the column i of config_pino_.
void Dataloader::set_config_pino_column(int i, Eigen::VectorXd &q_pino_mat)
{
    // Sanity check
    if (q_pino_mat.rows()!= nq_pino_)
    {
        LOG(FATAL) << "Check failed: q_pino_mat.rows()!= nq_pino_" << std::endl;
    }

    int joint_type;
    int joint_index;
    int joint_index_pino;
    for (int j = 0; j < njoints_; j++)
    {
        joint_type = decoration_(j, 2);
        joint_index_pino = decoration_(j, 4); // the index in q_pino ...
        if (joint_type == 1)
        { // free-floating joint: copy linear, convert angular to quaternion
            config_pino_.block<3, 1>(joint_index_pino, i) = q_pino_mat.block<3, 1>(joint_index_pino, 0);
            config_pino_.block<4, 1>(joint_index_pino + 3, i) =
                (q_pino_mat.block<4, 1>(joint_index_pino + 3, 0)).normalized();
        }
        else if (joint_type == 2)
        { // spherical joint: convert axis-angle to quaternion
            config_pino_.block<4, 1>(joint_index_pino, i) =
                (q_pino_mat.block<4, 1>(joint_index_pino, 0)).normalized();
        }
        else
        { // all other types of joint: just copy the configuration
            config_pino_(joint_index_pino, i) = q_pino_mat(joint_index_pino, 0);
        }
    }
}

Eigen::MatrixXd & Dataloader::get_config_init()
{
    return config_init_;
}
Eigen::MatrixXd & Dataloader::get_config_pino_init()
{
    return config_pino_init_;
}
Eigen::MatrixXd & Dataloader::get_ground_friction()
{
    return ground_friction_;
}
void Dataloader::set_ground_friction(const Eigen::MatrixXd &ground_friction)
{
    ground_friction_ = ground_friction;
}
double * Dataloader::mutable_config(int i)
{
    return config_.data() + i*nq_;
}
// double * Dataloader::mutable_config(int i, int shift)
// {
//     return config_.data() + i*nq_ + shift;
// }
double * Dataloader::mutable_velocity(int i)
{
    return velocity_.data() + i*nq_;
}
// double * Dataloader::mutable_velocity(int i, int shift)
// {
//     return velocity_.data() + i*nq_ + shift;
// }
double * Dataloader::mutable_config_pino(int i)
{
    return config_pino_.data() + i*nq_pino_;
}
// double * Dataloader::mutable_config_pino(int i, int shift)
// {
//     return config_pino_.data() + i*nq_pino_ + shift;
// }
double * Dataloader::mutable_ground_friction(int i)
{
    return ground_friction_.data() + i*nq_ground_friction_;
}
