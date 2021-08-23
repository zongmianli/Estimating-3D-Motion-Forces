#include "pinocchio/algorithm/kinematics.hpp"

#include "dataloader_object.h"

DataloaderObject::DataloaderObject(pinocchio::Model &model,
                                   pinocchio::Data &data,
                                   const Eigen::MatrixXd &decoration,
                                   const Eigen::MatrixXd &config,
                                   double fps,
                                   std::string name,
                                   const Eigen::MatrixXd &config_contact,
                                   const Eigen::MatrixXd &config_keypoints,
                                   const Eigen::MatrixXd &endpoint_2d_positions,
                                   bool is_virtual_object)
: Dataloader(model, data, decoration)
{
    // initialize basic info
    set_name(name);
    nt_ = (int)config.cols();
    nq_contact_ = (int)config_contact.rows();
    is_virtual_object_ = is_virtual_object;
    if (is_virtual_object)
    {
        // initialize person-object contact force
        num_contacts_ = nq_contact_;
        nq_contact_force_ = num_contacts_ * 6;
        contact_force_ = Eigen::MatrixXd::Zero(nq_contact_force_, nt_);
    }
    else
    {
        // Note that ground contact point can vary on a plane
        // while other types of contact vary on a line segment
        if (name_ == "ground")
        {
            num_contacts_ = nq_contact_ / 2;
            nq_keypoints_ = 0;
            num_keypoints_ = 0;
            njoints_ = model.njoints - 1;
            nq_ = model.nv;
            nq_pino_ = model.nq;
            // no forces
            nq_contact_force_ = 0;
            nq_ground_friction_ = 0;
        }
        else
        {
            num_contacts_ = nq_contact_;
            nq_keypoints_ = (int)config_keypoints.rows();
            num_keypoints_ = nq_keypoints_;
            LoadConfigKeypoints(config_keypoints);
            LoadEndpoint2dPositions(endpoint_2d_positions);
            njoints_ = model.njoints - 1 - num_contacts_ - num_keypoints_; // ignore the 'universe' joint
            nq_ = model.nv - nq_contact_ - nq_keypoints_;
            nq_pino_ = model.nq - nq_contact_ - nq_keypoints_;
            // initialize person-object contact force
            nq_contact_force_ = num_contacts_ * 6;
            contact_force_ = Eigen::MatrixXd::Zero(nq_contact_force_, nt_);
            // initialize ground friction force
            nq_ground_friction_ = 4; // we assume a single object-ground contact point
            ground_friction_ = Eigen::MatrixXd::Zero(nq_ground_friction_, nt_);
        }
        // // for debugging
        std::cout << name_ << std::endl;
        std::cout << "njoints_ = " << njoints_ << std::endl;
        std::cout << "nq_ = " << nq_ << std::endl;
        std::cout << "nq_pino_ = " << nq_pino_ << std::endl;
        std::cout << "num_contacts_ = " << num_contacts_ << std::endl;
        std::cout << "nq_keypoints_ = " << nq_keypoints_ << std::endl;
        std::cout << "num_keypoints_ = " << num_keypoints_ << std::endl;
        LoadConfig(config, fps);
        config_init_ = config_;
        config_pino_init_ = config_pino_;
        LoadConfigContact(config_contact);
        if ((!is_virtual_object) && (name_!="ground"))
        {
            // save endpoints 3D and 2D positions
            keypoint_3d_positions_ = Eigen::MatrixXd::Zero(3*2, nt_);
            keypoint_2d_reprojected_ = Eigen::MatrixXd::Zero(3*2, nt_);
            UpdateKeypoint3dPositions();
            std::cout << "keypoints Initialized!" << std::endl;
        }
    }
}

// void DataloaderPerson::UpdateConfigPino(int t, const double *q)
// {
//     Eigen::Matrix<double, 6, 1> q_mat(q);
//     UpdateConfigPino(t, q_mat);
// }

// Updates contact positions stored in the dataloader, while updating 
// the dimension of contact configuration vector, nq_contact_,
// and the number of contact points, num_contacts_.
void DataloaderObject::LoadConfigContact(const Eigen::MatrixXd &config_contact)
{
    if (config_contact.cols() != nt_)
    {
        LOG(FATAL) << "config_contact_.cols() != nt_!" << std::endl;
    }
    if (config_contact.rows() != nq_contact_)
    {
        LOG(FATAL) << "config_contact.rows() != nq_contact_!" << std::endl;
    }
    config_contact_ = config_contact;
}

void DataloaderObject::LoadConfigKeypoints(const Eigen::MatrixXd &config_keypoints)
{
    if (config_keypoints.cols() != nt_)
    {
        LOG(FATAL) << "config_keypoints.cols() != nt_!" << std::endl;
    }
    if (config_keypoints.rows() != nq_keypoints_)
    {
        LOG(FATAL) << "config_keypoints.rows()!= nq_keypoints_!" << std::endl;
    }
    config_keypoints_ = config_keypoints;
}

void DataloaderObject::LoadEndpoint2dPositions(const Eigen::MatrixXd &endpoint_2d_positions)
{
    if (endpoint_2d_positions.cols() != nt_)
    {
        LOG(FATAL) << "endpoint_2d_positions.cols() != nt_!" << std::endl;
    }
    if (endpoint_2d_positions.rows() != 6)
    {
        LOG(FATAL) << "endpoint_2d_positions.rows() != 6!" << std::endl;
    }
    endpoint_2d_positions_ = endpoint_2d_positions;
}

void DataloaderObject::UpdateKeypoint3dPositions()
{
    UpdateConfigPino();
    Eigen::VectorXd q_stacked = Eigen::VectorXd::Zero(nq_pino_+nq_contact_+nq_keypoints_);
    // VectorXd q_stacked(get_nq_pino()+nq_contact_+nq_keypoints_,1);
    for (int i = 0; i < nt_; i++)
    {
        const double *q_object_pino = mutable_config_pino(i);
        const double *q_keypoints = mutable_config_keypoints(i);
        for (int k = 0; k < nq_pino_; k++)
        {
            q_stacked(k) = *(q_object_pino + k);
        }
        // here we ignore contact positions for they are not used
        for (int k = 0; k < nq_keypoints_; k++)
        {
            q_stacked(k + nq_pino_ + nq_contact_) = *(q_keypoints + k);
        }
        // q_stacked = Map<VectorXd>(mutable_config_pino(i),get_nq_pino());
        pinocchio::forwardKinematics(model_, data_, q_stacked);
        // update head position
        keypoint_3d_positions_.block(0, i, 3, 1) = data_.oMi[1].translation(); // the 8 first operational frames are the foot contact points
        // update stick end position
        keypoint_3d_positions_.block(3, i, 3, 1) = data_.oMi[njoints_ + num_contacts_ +1].translation(); // the 8 first operational frames are the foot contact points
    }
}

void DataloaderObject::UpdateKeypoint2dReprojected(Camera *camera)
{
    UpdateKeypoint3dPositions();
    Eigen::Vector3d keypoint_position_3d;
    for (int i = 0; i < nt_; i++)
    {
        for (int k = 0; k < 2; k++)
        {
            keypoint_position_3d = keypoint_3d_positions_.block(3 * k, i, 3, 1);
            keypoint_2d_reprojected_.block(3 * k, i, 2, 1) = camera->Project(keypoint_position_3d);
        }
    }
}

// Class accessors and mutators
std::string DataloaderObject::get_name()
{
    return name_;
}
void DataloaderObject::set_name(std::string name)
{
    name_ = name;
}
int DataloaderObject::get_nq_contact()
{
    return nq_contact_;
}
int DataloaderObject::get_num_contacts()
{
    return num_contacts_;
}
int DataloaderObject::get_nq_contact_force()
{
    return nq_contact_force_;
}
int DataloaderObject::get_nq_keypoints()
{
    return nq_keypoints_;
}
int DataloaderObject::get_num_keypoints()
{
    return num_keypoints_;
}
bool DataloaderObject::get_is_virtual_object()
{
    return is_virtual_object_;
}

Eigen::MatrixXd DataloaderObject::get_config_contact()
{
    return config_contact_;
}
Eigen::MatrixXd DataloaderObject::get_contact_force()
{
    return contact_force_;
}
void DataloaderObject::set_contact_force(const Eigen::MatrixXd &contact_force)
{
    contact_force_ = contact_force;
}
Eigen::MatrixXd DataloaderObject::get_config_keypoints()
{
    return config_keypoints_;
}
Eigen::MatrixXd DataloaderObject::get_endpoint_2d_positions()
{
    return endpoint_2d_positions_;
}
Eigen::VectorXd DataloaderObject::get_endpoint_2d_positions_column(int i)
{
    return endpoint_2d_positions_.col(i);
}

Eigen::MatrixXd & DataloaderObject::get_keypoint_3d_positions()
{
    return keypoint_3d_positions_;
}

const Eigen::MatrixXd & DataloaderObject::get_keypoint_3d_positions() const
{
    return keypoint_3d_positions_;
}

Eigen::VectorXd DataloaderObject::get_keypoint_3d_positions_column(int i)
{
    return keypoint_3d_positions_.col(i);
}

Eigen::MatrixXd & DataloaderObject::get_keypoint_2d_reprojected()
{
    return keypoint_2d_reprojected_;
}

const Eigen::MatrixXd & DataloaderObject::get_keypoint_2d_reprojected() const
{
    return keypoint_2d_reprojected_;
}

double * DataloaderObject::mutable_config_contact(int i)
{
    return config_contact_.data() + i*nq_contact_;
}
double * DataloaderObject::mutable_contact_force(int i)
{
    return contact_force_.data() + i*nq_contact_force_;
}
double * DataloaderObject::mutable_config_keypoints(int i)
{
    return config_keypoints_.data() + i*nq_keypoints_;
}
