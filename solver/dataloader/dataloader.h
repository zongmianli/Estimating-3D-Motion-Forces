#ifndef __DATALOADER_H__
#define __DATALOADER_H__

#include <Eigen/Core>
#include <glog/logging.h>
//#include <math.h>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"

//
//
//#include "../camera.h"

// A class that is used for access/setting person data for the solver
class Dataloader
{
public:
    Dataloader(pinocchio::Model &model,
               pinocchio::Data &data,
               const Eigen::MatrixXd &decoration);

    void LoadConfig(const Eigen::MatrixXd &config, double fps);
    // virtual void UpdateJoint3dPositions() = 0;

    static Eigen::MatrixXd AxisAngleToQuat(const Eigen::MatrixXd &axis_angles);
    static Eigen::MatrixXd QuatToAxisAngle(const Eigen::MatrixXd &quats);

    void UpdateConfigPino();
    void UpdateConfigPino(const Eigen::MatrixXd &config);
    void UpdateConfigPino(int t, Eigen::VectorXd &q_mat);
    void UpdateConfig();
    void UpdateConfig(const Eigen::MatrixXd &config_pino);
    void UpdateConfig(int t, Eigen::VectorXd &q_pino_mat);
    void UpdateConfigVelocity();
    void UpdateConfigVelocity(const Eigen::MatrixXd &config);
    void UpdateConfigVelocity(int t);

    pinocchio::Model & model_;
    pinocchio::Data & data_;

    // Class accessors and mutators
    int get_njoints();
    int get_nq();
    int get_nq_pino();
    int get_nq_ground_friction();
    int get_nt();
    double get_fps();
    double get_dt();

    Eigen::MatrixXd get_decoration();
    Eigen::MatrixXd & get_config();
    Eigen::MatrixXd::ColXpr get_config_column(int i);
    void set_config(const Eigen::MatrixXd &config);
    Eigen::MatrixXd & get_velocity();
    Eigen::MatrixXd & get_config_pino();
    void set_config_pino(const Eigen::MatrixXd &config_pino);
    void set_config_pino_column(int i, Eigen::VectorXd &q_pino_mat);
    Eigen::MatrixXd & get_config_init();
    Eigen::MatrixXd & get_config_pino_init();
    Eigen::MatrixXd & get_ground_friction();
    void set_ground_friction(const Eigen::MatrixXd &ground_friction);

    double * mutable_config(int i);
    // double * mutable_config(int i, int shift);
    double * mutable_velocity(int i);
    // double * mutable_velocity(int i, int shift);
    double * mutable_config_pino(int i);
    // double * mutable_config_pino(int i, int shift);
    double * mutable_ground_friction(int i);

protected:
    int njoints_; // number of joints
    int nq_;      // dimension of the configuration vector with axis angle (75) (here nv_ == nq_)
    int nq_pino_; // dimension of the pinocchio configuration vector with quaternions (99) (here nv_pino_ != nq_pino_)
    int nq_ground_friction_; // dimension of ground friction force vector
    int nt_;      // number of time steps
    double fps_;
    double dt_;
    const Eigen::MatrixXi decoration_;
    Eigen::MatrixXd config_;
    Eigen::MatrixXd velocity_;
    Eigen::MatrixXd config_pino_;
    Eigen::MatrixXd config_init_;
    Eigen::MatrixXd config_pino_init_;
    Eigen::MatrixXd ground_friction_;
};

#endif // ifndef __DATALOADER_H__
