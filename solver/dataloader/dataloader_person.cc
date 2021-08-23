#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "dataloader_person.h"

DataloaderPerson::DataloaderPerson(pinocchio::Model &model,
                                   pinocchio::Data &data,
                                   const Eigen::MatrixXd &decoration,
                                   const Eigen::MatrixXd &config,
                                   double fps,
                                   double friction_angle,
                                   int num_ground_contact_points,
                                   const Eigen::VectorXd &contact_mapping,
                                   const Eigen::MatrixXd &contact_states,
                                   const Eigen::MatrixXd &contact_types,
                                   const Eigen::MatrixXd &joint_2d_positions)
: Dataloader(model, data, decoration)
{
    // initialize basic info
    njoints_ = model.njoints - 1; // ignore the 'universe' joint
    nq_ = model.nv;
    nq_pino_ = model.nq;
    nt_ = (int)config.cols();
    LoadConfig(config, fps);
    config_init_ = config_;
    config_pino_init_ = config_pino_;
    // load all data
    LoadContactInfo(contact_mapping, contact_states, contact_types);
    LoadJoint2dPositions(joint_2d_positions);
    // initialize joint positions in 2D and 3D
    joint_3d_positions_ = Eigen::MatrixXd::Zero(3*njoints_, nt_);
    joint_2d_reprojected_ = Eigen::MatrixXd::Zero(3*njoints_, nt_);
    UpdateJoint3dPositions();
    // initialize keypoint positions in 2D and 3D
    nkeypoints_ = 18; // 18 Openpose "keypoints"
    keypoint_3d_positions_ = Eigen::MatrixXd::Zero(3*nkeypoints_, nt_);
    keypoint_2d_reprojected_ = Eigen::MatrixXd::Zero(3*nkeypoints_, nt_);
    UpdateKeypoint3dPositions();
    // initialize ground friction force
    // nq_ground_friction_: note that each ankle joint has 4 contact points, at contact point is assigned a 4D force
    nq_ground_friction_ = num_ground_contact_points*4;
    ground_friction_ = Eigen::MatrixXd::Constant(nq_ground_friction_, nt_, 1.0);
    // compute friction cone generators for soles and other joints
    SetFrictionConeGenerators(friction_angle);
}

void DataloaderPerson::LoadContactInfo(const Eigen::VectorXd &contact_mapping,
                                       const Eigen::MatrixXd &contact_states,
                                       const Eigen::MatrixXd &contact_types)
{
    if (contact_states.cols() != nt_)
    {
        LOG(FATAL) << "contact_states.cols() != nt_!" << std::endl;
    }
    if (contact_types.cols() != nt_)
    {
        LOG(FATAL) << "contact_types.cols() != nt_!" << std::endl;
    }
    contact_states_ = contact_states.cast<int>();
    contact_types_ = contact_types.cast<int>();
    contact_mapping_ = contact_mapping.cast<int>();
}

void DataloaderPerson::LoadJoint2dPositions(const Eigen::MatrixXd &joint_2d_positions)
{
    if (joint_2d_positions.cols() != nt_)
    {
        LOG(FATAL) << "joint_2d_positions.cols() != nt_!" << std::endl;
    }
    joint_2d_positions_ = joint_2d_positions;
}

void DataloaderPerson::UpdateJoint3dPositions()
{
    UpdateConfigPino();
    Eigen::VectorXd q_pino(get_nq_pino(),1);
    for (int i = 0; i < nt_; i++)
    {
        q_pino = Eigen::Map<Eigen::VectorXd>(mutable_config_pino(i),get_nq_pino());
        pinocchio::forwardKinematics(model_, data_, q_pino);
        for (int j = 0; j < njoints_; j++)
        {
            joint_3d_positions_.block(3 * j, i, 3, 1) = data_.oMi[(size_t)(j + 1)].translation();
        }
    }
}

void DataloaderPerson::UpdateJoint2dReprojected(Camera *camera)
{
    UpdateJoint3dPositions();
    Eigen::Vector3d joint_position_3d;
    for (int i = 0; i < nt_; i++)
    {
        for (int j = 0; j < njoints_; j++)
        {
            joint_position_3d = joint_3d_positions_.block(3 * j, i, 3, 1);
            joint_2d_reprojected_.block(3 * j, i, 2, 1) = camera->Project(joint_position_3d);
        }
    }
}

void DataloaderPerson::UpdateKeypoint3dPositions()
{
    UpdateConfigPino();
    Eigen::VectorXd q_pino(get_nq_pino(),1);
    for (int i = 0; i < nt_; i++)
    {
        q_pino = Eigen::Map<Eigen::VectorXd>(mutable_config_pino(i),get_nq_pino());
        pinocchio::framesForwardKinematics(model_, data_, q_pino);
        for (int k = 0; k < nkeypoints_; k++)
        {
            keypoint_3d_positions_.block(3 * k, i, 3, 1) = data_.oMf[(size_t)(k + 9)].translation(); // the 8 first operational frames are the foot contact points
        }
    }
}

void DataloaderPerson::UpdateKeypoint2dReprojected(Camera *camera)
{
    UpdateKeypoint3dPositions();
    Eigen::Vector3d keypoint_position_3d;
    for (int i = 0; i < nt_; i++)
    {
        for (int k = 0; k < nkeypoints_; k++)
        {
            keypoint_position_3d = keypoint_3d_positions_.block(3 * k, i, 3, 1);
            keypoint_2d_reprojected_.block(3 * k, i, 2, 1) = camera->Project(keypoint_position_3d);
        }
    }
}

void DataloaderPerson::SetFrictionConeGenerators(double friction_angle)
{
    // create friction cone generators for ground contacts
    friction_angle_ = friction_angle; // M_PI/6.0;
    double sin_angle = sin(friction_angle_);
    double cos_angle = cos(friction_angle_);
    double minus_sin_angle = -sin_angle;
    // 3D generators
    Eigen::Matrix<double, 3, 4> generators_3d;
    generators_3d << minus_sin_angle, 0.0, sin_angle, 0.0,
        cos_angle, cos_angle, cos_angle, cos_angle,
        0.0, sin_angle, 0.0, minus_sin_angle;
    // Adding zeros to form 6D generators
    // Here we should reverse the friction cone bacause the 3D generators 
    // are expressed in the world frame whose y-axis points towards the gravity
    friction_cone_generators_ = Eigen::MatrixXd::Zero(6, 4);
    friction_cone_generators_.topRows<3>() = -generators_3d; 
    // 6D generators corresponding to left/right sole
    friction_cone_generators_left_foot_ = ComputeFrictionConeGeneratorsForFoot(generators_3d, 1);
    friction_cone_generators_right_foot_ = ComputeFrictionConeGeneratorsForFoot(generators_3d, 2);
}

Eigen::MatrixXd DataloaderPerson::ComputeFrictionConeGeneratorsForFoot(const Eigen::Matrix<double, 3, 4> &generators_3d, int option)
{
    int frame_id; // frame id of the 1st foot contact point
    if (option == 1)
    { // left foot
        frame_id = 1;
    }
    else if (option == 2)
    { // right foot
        frame_id = 5;
    }
    else
    {
        LOG(FATAL) << "ComputeFrictionConeGeneratorsForFoot: incorrect option value!" << std::endl;
    }
    pinocchio::Data::Matrix6x friction_cone_generators_foot = pinocchio::Data::Matrix6x::Zero(6, 16);
    // Eigen::Matrix3d p_contact_foot_cross;
    for (int i = 0; i < 4; i++)
    {
        const Eigen::Vector3d & p_contact_foot = model_.frames[(size_t)(frame_id + i)].placement.translation();
//            p_contact_foot_cross = ComputePCross(p_contact_foot);
        for (int k = 0; k < 4; k++)
        {
            friction_cone_generators_foot.col(4 * i + k).head<3>() = generators_3d.col(k);
            friction_cone_generators_foot.col(4 * i + k).tail<3>() = p_contact_foot.cross(generators_3d.col(k));
        }
    }
    return friction_cone_generators_foot;
}

// Class accessors and mutators
Eigen::VectorXi & DataloaderPerson::get_contact_mapping()
{
    return contact_mapping_;
}

const Eigen::VectorXi & DataloaderPerson::get_contact_mapping() const
{
    return contact_mapping_;
}

Eigen::MatrixXi & DataloaderPerson::get_contact_states()
{
    return contact_states_;
}

const Eigen::MatrixXi & DataloaderPerson::get_contact_states() const
{
    return contact_states_;
}

Eigen::MatrixXi::ColXpr DataloaderPerson::get_contact_states_column(int i)
{
    return contact_states_.col(i);
}

Eigen::MatrixXi::ConstColXpr DataloaderPerson::get_contact_states_column(int i) const
{
    return contact_states_.col(i);
}


Eigen::MatrixXi & DataloaderPerson::get_contact_types()
{
    return contact_types_;
}

const Eigen::MatrixXi & DataloaderPerson::get_contact_types() const
{
    return contact_types_;
}

Eigen::MatrixXi::ColXpr DataloaderPerson::get_contact_types_column(int i)
{
    return contact_types_.col(i);
}

Eigen::MatrixXi::ConstColXpr DataloaderPerson::get_contact_types_column(int i) const
{
    return contact_types_.col(i);
}

Eigen::MatrixXd & DataloaderPerson::get_joint_2d_positions()
{
    return joint_2d_positions_;
}

const Eigen::MatrixXd & DataloaderPerson::get_joint_2d_positions() const
{
    return joint_2d_positions_;
}

Eigen::MatrixXd::ColXpr DataloaderPerson::get_joint_2d_positions_column(int i)
{
    return joint_2d_positions_.col(i);
}

Eigen::MatrixXd::ConstColXpr DataloaderPerson::get_joint_2d_positions_column(int i) const
{
    return joint_2d_positions_.col(i);
}

Eigen::MatrixXd & DataloaderPerson::get_joint_3d_positions()
{
    return joint_3d_positions_;
}

const Eigen::MatrixXd & DataloaderPerson::get_joint_3d_positions() const
{
    return joint_3d_positions_;
}

Eigen::VectorXd DataloaderPerson::get_joint_3d_positions_column(int i)
{
    return joint_3d_positions_.col(i);
}

Eigen::MatrixXd & DataloaderPerson::get_joint_2d_reprojected()
{
    return joint_2d_reprojected_;
}

const Eigen::MatrixXd & DataloaderPerson::get_joint_2d_reprojected() const
{
    return joint_2d_reprojected_;
}

Eigen::MatrixXd & DataloaderPerson::get_keypoint_3d_positions()
{
    return keypoint_3d_positions_;
}

const Eigen::MatrixXd & DataloaderPerson::get_keypoint_3d_positions() const
{
    return keypoint_3d_positions_;
}

Eigen::MatrixXd & DataloaderPerson::get_keypoint_2d_reprojected()
{
    return keypoint_2d_reprojected_;
}

const Eigen::MatrixXd & DataloaderPerson::get_keypoint_2d_reprojected() const
{
    return keypoint_2d_reprojected_;
}

Eigen::MatrixXd & DataloaderPerson::get_friction_cone_generators()
{
    return friction_cone_generators_;
}

const Eigen::MatrixXd & DataloaderPerson::get_friction_cone_generators() const
{
    return friction_cone_generators_;
}

Eigen::MatrixXd & DataloaderPerson::get_friction_cone_generators_left_foot()
{
    return friction_cone_generators_left_foot_;
}

const Eigen::MatrixXd & DataloaderPerson::get_friction_cone_generators_left_foot() const
{
    return friction_cone_generators_left_foot_;
}

Eigen::MatrixXd & DataloaderPerson::get_friction_cone_generators_right_foot()
{
    return friction_cone_generators_right_foot_;
}

const Eigen::MatrixXd & DataloaderPerson::get_friction_cone_generators_right_foot() const
{
    return friction_cone_generators_right_foot_;
}
