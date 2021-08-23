#include "bindings.h"
#include "../camera.h"

void ExposeCamera()
{
    bp::class_<Camera>("Camera",
                       bp::init<double,
                       const Eigen::Vector2d &,
                       const Eigen::Vector2d &,
                       const Eigen::Matrix3d &,
                       const Eigen::Vector3d &>())
        .def("UpdateProjectionMatrix", &Camera::UpdateProjectionMatrix)
        .def("Project", &Camera::Project)
        .add_property("focal_length_", &Camera::get_focal_length, &Camera::set_focal_length)
        .add_property("principal_point_", &Camera::get_principal_point, &Camera::set_principal_point)
        .add_property("skew_", &Camera::get_skew, &Camera::set_skew)
        .add_property("cam_rotation_", &Camera::get_cam_rotation, &Camera::set_cam_rotation)
        .add_property("cam_translation_", &Camera::get_cam_translation, &Camera::set_cam_translation)
        .add_property("calibration_mat_", &Camera::get_calibration_mat)
        .add_property("projection_mat_", &Camera::get_projection_mat);
}
