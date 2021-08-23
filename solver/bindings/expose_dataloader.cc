#include "bindings.h"
#include "../dataloader/dataloader.h"

void ExposeDataloader()
{
    bp::class_<Dataloader>("Dataloader",
                           bp::init<pinocchio::Model &,
                           pinocchio::Data &,
                           const Eigen::MatrixXd &>())
        .def("LoadConfig", &Dataloader::LoadConfig)
        .def("AxisAngleToQuat", &Dataloader::AxisAngleToQuat)
        .def("QuatToAxisAngle", &Dataloader::QuatToAxisAngle)
        .def("UpdateConfigPino",
             (void (Dataloader::*)()) & Dataloader::UpdateConfigPino)
        .def("UpdateConfigPino",
             (void (Dataloader::*)(const Eigen::MatrixXd &)) & Dataloader::UpdateConfigPino)
        .def("UpdateConfigPino",
             (void (Dataloader::*)(int, Eigen::VectorXd &)) & Dataloader::UpdateConfigPino)
        .def("UpdateConfigVelocity",
             (void (Dataloader::*)()) & Dataloader::UpdateConfigVelocity)
        .def("UpdateConfigVelocity",
             (void (Dataloader::*)(const Eigen::MatrixXd &)) & Dataloader::UpdateConfigVelocity)
        .def("UpdateConfigVelocity",
             (void (Dataloader::*)(int)) & Dataloader::UpdateConfigVelocity)
        //.def_readonly("model_", &Dataloader::model_)
        //.def_readwrite("data_", &Dataloader::data_)
        .add_property("njoints_", &Dataloader::get_njoints)
        .add_property("nq_", &Dataloader::get_nq)
        .add_property("nq_pino_", &Dataloader::get_nq_pino)
        .add_property("nq_ground_friction_", &Dataloader::get_nq_ground_friction)
        .add_property("nt_", &Dataloader::get_nt)
        .add_property("fps_", &Dataloader::get_fps)
        .add_property("dt_", &Dataloader::get_dt)
        .add_property("decoration_", &Dataloader::get_decoration)
        .add_property("config_",
                      bp::make_function(&Dataloader::get_config,
                                        bp::return_value_policy<bp::return_by_value>()),
                      &Dataloader::set_config)
        .add_property("velocity_",
                      bp::make_function(&Dataloader::get_velocity,
                                        bp::return_value_policy<bp::return_by_value>()))
        .add_property("config_pino_",
                      bp::make_function(&Dataloader::get_config_pino,
                                        bp::return_value_policy<bp::return_by_value>()),
                      &Dataloader::set_config_pino)
        .add_property("config_init_",
                      bp::make_function(&Dataloader::get_config_init,
                                        bp::return_value_policy<bp::return_by_value>()))
        .add_property("config_pino_init_",
                      bp::make_function(&Dataloader::get_config_pino_init,
                                        bp::return_value_policy<bp::return_by_value>()))
        .add_property("ground_friction_",
                      bp::make_function(&Dataloader::get_ground_friction,
                                        bp::return_value_policy<bp::return_by_value>()),
                      &Dataloader::set_ground_friction);
}
