#include "bindings.h"
#include "../dataloader/dataloader_object.h"

void ExposeDataloaderObject()
{
    bp::class_<DataloaderObject, bp::bases<Dataloader> >("DataloaderObject",
                                                         bp::init<pinocchio::Model &,
                                                         pinocchio::Data &,
                                                         const Eigen::MatrixXd &,
                                                         const Eigen::MatrixXd &,
                                                         double,
                                                         std::string,
                                                         const Eigen::MatrixXd &,
                                                         const Eigen::MatrixXd &,
                                                         const Eigen::MatrixXd &,
                                                         bool>())
        .def("LoadConfigContact", &DataloaderObject::LoadConfigContact)
        .def("LoadConfigKeypoints", &DataloaderObject::LoadConfigKeypoints)
        .def("LoadEndpoint2dPositions", &DataloaderObject::LoadEndpoint2dPositions)
        .def("UpdateKeypoint3dPositions", &DataloaderObject::UpdateKeypoint3dPositions)
        .def("UpdateKeypoint2dReprojected", &DataloaderObject::UpdateKeypoint2dReprojected)
        .add_property("name_", &DataloaderObject::get_name, &DataloaderObject::set_name)
        .add_property("nq_contact_", &DataloaderObject::get_nq_contact)
        .add_property("num_contacts_", &DataloaderObject::get_num_contacts)
        .add_property("nq_contact_force_", &DataloaderObject::get_nq_contact_force)
        .add_property("nq_keypoints_", &DataloaderObject::get_nq_keypoints)
        .add_property("num_keypoints_", &DataloaderObject::get_num_keypoints)
        .add_property("config_contact_", &DataloaderObject::get_config_contact)
        .add_property("contact_force_", &DataloaderObject::get_contact_force, &DataloaderObject::set_contact_force)
        .add_property("config_keypoints_", &DataloaderObject::get_config_keypoints)
        .add_property("endpoint_2d_positions_", &DataloaderObject::get_endpoint_2d_positions)
        .add_property("keypoint_3d_positions_", bp::make_function((Eigen::MatrixXd & (DataloaderObject::*)())&DataloaderObject::get_keypoint_3d_positions,
                                                                  bp::return_value_policy<bp::return_by_value>()))
        .add_property("keypoint_2d_reprojected_", bp::make_function((Eigen::MatrixXd & (DataloaderObject::*)())&DataloaderObject::get_keypoint_2d_reprojected,
                                                                    bp::return_value_policy<bp::return_by_value>()));
}
