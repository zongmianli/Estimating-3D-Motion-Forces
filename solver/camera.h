#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <Eigen/Core>

// the Camera class simplifies the operations with the perspective projection
class Camera
{
public:
    Camera(double skew,
           const Eigen::Vector2d &focal_length,
           const Eigen::Vector2d &principal_point,
           const Eigen::Matrix3d &cam_rotation,
           const Eigen::Vector3d &cam_translation);
    void UpdateProjectionMatrix();
    Eigen::Vector2d Project(const Eigen::Vector3d & point_3d);
    // Class accessors and mutators
    double get_skew();
    void set_skew(double skew);
    Eigen::Vector2d get_focal_length();
    void set_focal_length(const Eigen::Vector2d & focal_length);
    void complete_focal_length(double flength);
    Eigen::Vector2d get_principal_point();
    void set_principal_point(const Eigen::Vector2d & principal_point);
    Eigen::Matrix3d get_cam_rotation();
    void set_cam_rotation(const Eigen::Matrix3d & cam_rotation);
    Eigen::Vector3d get_cam_translation();
    void set_cam_translation(const Eigen::Vector3d & cam_translation);
    Eigen::MatrixXd get_calibration_mat();
    Eigen::MatrixXd get_projection_mat();
    double * mutable_focal_length();
    
private:
    double skew_;
    Eigen::Vector2d focal_length_;
    Eigen::Vector2d principal_point_;
    Eigen::Matrix3d cam_rotation_;
    Eigen::Vector3d cam_translation_;
    Eigen::MatrixXd calibration_mat_; // 3x3 camera calibration matrix
    Eigen::MatrixXd projection_mat_;  // 3x4 camera projection matrix
};

#endif // #ifndef __CAMERA_H__
