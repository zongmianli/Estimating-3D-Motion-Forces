#include "camera.h"

Camera::Camera(double skew,
               const Eigen::Vector2d &focal_length,
               const Eigen::Vector2d &principal_point,
               const Eigen::Matrix3d &cam_rotation,
               const Eigen::Vector3d &cam_translation)
    : skew_(skew),
      focal_length_(focal_length),
      principal_point_(principal_point),
      cam_rotation_(cam_rotation),
      cam_translation_(cam_translation)
{
    UpdateProjectionMatrix();
}

void Camera::UpdateProjectionMatrix()
{
    // compute camera calibration matrix K
    Eigen::Matrix3d calibration_mat = Eigen::Matrix3d::Zero();
    calibration_mat(0, 0) = focal_length_(0);
    calibration_mat(1, 1) = focal_length_(1);
    calibration_mat(0, 1) = skew_;
    calibration_mat(0, 2) = principal_point_(0);
    calibration_mat(1, 2) = principal_point_(1);
    calibration_mat(2, 2) = 1.0;
    calibration_mat_ = calibration_mat;
    // compute camera projection matrix P = KR[I | −C]
    typedef Eigen::Matrix<double,3,4> Matrix34;
    Matrix34 right_mat = Matrix34::Identity();
    right_mat.block<3, 1>(0, 3) = - cam_translation_;
    projection_mat_.noalias() = (calibration_mat_ * cam_rotation_) * right_mat;
}

Eigen::Vector2d Camera::Project(const Eigen::Vector3d & point_3d)
{
    Eigen::Vector4d point_3d_homo;
    point_3d_homo << point_3d, 1.;
    Eigen::Vector3d point_2d_homo = projection_mat_*point_3d_homo;
    return point_2d_homo.block<2,1>(0,0)/point_2d_homo(2);
}

// Class accessors and mutators
double Camera::get_skew()
{
    return skew_;
}

void Camera::set_skew(double skew)
{
    skew_ = skew;
}

Eigen::Vector2d Camera::get_focal_length()
{
    return focal_length_;
}

void Camera::set_focal_length(const Eigen::Vector2d  & focal_length)
{
    focal_length_ = focal_length;
}

void Camera::complete_focal_length(double flength)
{
    focal_length_(0) = flength;
    focal_length_(1) = flength;
}

Eigen::Vector2d Camera::get_principal_point()
{
    return principal_point_;
}

void Camera::set_principal_point(const Eigen::Vector2d & principal_point)
{
    principal_point_ = principal_point;
}

Eigen::Matrix3d Camera::get_cam_rotation()
{
    return cam_rotation_;
}

void Camera::set_cam_rotation(const Eigen::Matrix3d & cam_rotation)
{
    cam_rotation_ = cam_rotation;
}

Eigen::Vector3d Camera::get_cam_translation()
{
    return cam_translation_;
}

void Camera::set_cam_translation(const Eigen::Vector3d & cam_translation)
{
    cam_translation_ = cam_translation;
}

Eigen::MatrixXd Camera::get_calibration_mat()
{
    return calibration_mat_;
}

Eigen::MatrixXd Camera::get_projection_mat()
{
    return projection_mat_;
}

double * Camera::mutable_focal_length()
{
    return focal_length_.data();
}
