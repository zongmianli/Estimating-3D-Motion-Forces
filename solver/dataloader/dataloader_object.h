#ifndef __DATALOADER_OBJECT_H__
#define __DATALOADER_OBJECT_H__

#include <string>

#include "dataloader.h"
#include "../camera.h"

class DataloaderObject : public Dataloader
{
public:
    DataloaderObject(pinocchio::Model &model,
                     pinocchio::Data &data,
                     const Eigen::MatrixXd &decoration,
                     const Eigen::MatrixXd &config,
                     double fps,
                     std::string name,
                     const Eigen::MatrixXd &config_contact,
                     const Eigen::MatrixXd &config_keypoints,
                     const Eigen::MatrixXd &endpoint_2d_positions,
                     bool is_virtual_object);
    // void UpdateJoint3dPositions();
    // void UpdateConfigPino(int t, const double *q);
    // void UpdateConfig(int t, const double *q_pino);

    void LoadConfigContact(const Eigen::MatrixXd &config_contact);
    void LoadConfigKeypoints(const Eigen::MatrixXd &config_keypoints);
    void LoadEndpoint2dPositions(const Eigen::MatrixXd &endpoint_2d_positions);
    void UpdateKeypoint3dPositions();
    void UpdateKeypoint2dReprojected(Camera *camera);

    // Class accessors and mutators
    std::string get_name();
    void set_name(std::string name);
    int get_nq_contact();
    int get_num_contacts();
    int get_nq_contact_force();
    int get_nq_keypoints();
    int get_num_keypoints();
    bool get_is_virtual_object();
    Eigen::MatrixXd get_config_contact();
    Eigen::MatrixXd get_contact_force();
    void set_contact_force(const Eigen::MatrixXd &contact_force);
    Eigen::MatrixXd get_config_keypoints();
    Eigen::MatrixXd get_endpoint_2d_positions();
    Eigen::VectorXd get_endpoint_2d_positions_column(int i);

    Eigen::MatrixXd & get_keypoint_3d_positions();
    const Eigen::MatrixXd & get_keypoint_3d_positions() const;
    Eigen::VectorXd get_keypoint_3d_positions_column(int i);

    Eigen::MatrixXd & get_keypoint_2d_reprojected();
    const Eigen::MatrixXd & get_keypoint_2d_reprojected() const;

    double * mutable_config_contact(int i);
    double * mutable_contact_force(int i);
    double * mutable_config_keypoints(int i);

private:
    std::string name_;
    int nq_contact_;
    int nq_contact_force_;
    int nq_keypoints_;
    int num_contacts_;
    int num_keypoints_;
    bool is_virtual_object_;
    Eigen::MatrixXd config_contact_;
    Eigen::MatrixXd contact_force_;
    Eigen::MatrixXd config_keypoints_;
    Eigen::MatrixXd endpoint_2d_positions_;
    Eigen::MatrixXd keypoint_3d_positions_;
    Eigen::MatrixXd keypoint_2d_reprojected_;
};

#endif // ifndef __DATALOADER_OBJECT_H__
