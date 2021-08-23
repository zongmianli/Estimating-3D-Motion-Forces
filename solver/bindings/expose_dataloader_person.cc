#include "bindings.h"
#include "../dataloader/dataloader_person.h"

void ExposeDataloaderPerson()
{
    bp::class_<DataloaderPerson, bp::bases<Dataloader> >("DataloaderPerson",
                                                         bp::init<pinocchio::Model &,
                                                         pinocchio::Data &,
                                                         const Eigen::MatrixXd &,
                                                         const Eigen::MatrixXd &,
                                                         double,
                                                         double,
                                                         int,
                                                         const Eigen::VectorXd &,
                                                         const Eigen::MatrixXd &,
                                                         const Eigen::MatrixXd &,
                                                         const Eigen::MatrixXd &>())
        .def("LoadContactInfo", &DataloaderPerson::LoadContactInfo)
        .def("LoadJoint2dPositions", &DataloaderPerson::LoadJoint2dPositions)
        .def("UpdateJoint3dPositions", &DataloaderPerson::UpdateJoint3dPositions)
        .def("UpdateJoint2dReprojected", &DataloaderPerson::UpdateJoint2dReprojected)
        .def("UpdateKeypoint3dPositions", &DataloaderPerson::UpdateKeypoint3dPositions)
        .def("UpdateKeypoint2dReprojected", &DataloaderPerson::UpdateKeypoint2dReprojected)
        // .add_property("contact_mapping_", &DataloaderPerson::get_contact_mapping)
        // .add_property("contact_states_", &DataloaderPerson::get_contact_states)
        // .add_property("contact_types_", &DataloaderPerson::get_contact_types)
        .add_property("joint_2d_positions_", bp::make_function((Eigen::MatrixXd & (DataloaderPerson::*)())&DataloaderPerson::get_joint_2d_positions,
                                                               bp::return_value_policy<bp::return_by_value>()))
        .add_property("joint_3d_positions_", bp::make_function((Eigen::MatrixXd & (DataloaderPerson::*)())&DataloaderPerson::get_joint_3d_positions,
                                                               bp::return_value_policy<bp::return_by_value>()))
        .add_property("joint_2d_reprojected_", bp::make_function((Eigen::MatrixXd & (DataloaderPerson::*)())&DataloaderPerson::get_joint_2d_reprojected,
                                                                 bp::return_value_policy<bp::return_by_value>()))
        .add_property("keypoint_3d_positions_", bp::make_function((Eigen::MatrixXd & (DataloaderPerson::*)())&DataloaderPerson::get_keypoint_3d_positions,
                                                                  bp::return_value_policy<bp::return_by_value>()))
        .add_property("keypoint_2d_reprojected_", bp::make_function((Eigen::MatrixXd & (DataloaderPerson::*)())&DataloaderPerson::get_keypoint_2d_reprojected,
                                                                    bp::return_value_policy<bp::return_by_value>()))
        .add_property("friction_cone_generators_", bp::make_function((Eigen::MatrixXd & (DataloaderPerson::*)())&DataloaderPerson::get_friction_cone_generators,
                                                                     bp::return_value_policy<bp::return_by_value>()))
        .add_property("friction_cone_generators_left_foot_", bp::make_function((Eigen::MatrixXd & (DataloaderPerson::*)())&DataloaderPerson::get_friction_cone_generators_left_foot,
                                                                               bp::return_value_policy<bp::return_by_value>()))
        .add_property("friction_cone_generators_right_foot_", bp::make_function((Eigen::MatrixXd & (DataloaderPerson::*)())&DataloaderPerson::get_friction_cone_generators_right_foot,
                                                                                bp::return_value_policy<bp::return_by_value>()));
}
