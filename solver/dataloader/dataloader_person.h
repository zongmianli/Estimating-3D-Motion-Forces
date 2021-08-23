#ifndef __DATALOADER_PERSON_H__
#define __DATALOADER_PERSON_H__

#include "dataloader.h"
#include "../camera.h"

class DataloaderPerson : public Dataloader
{
public:
    DataloaderPerson(pinocchio::Model &model,
                     pinocchio::Data &data,
                     const Eigen::MatrixXd &decoration,
                     const Eigen::MatrixXd &config,
                     double fps,
                     double friction_angle,
                     int num_ground_contact_points,
                     const Eigen::VectorXd &contact_mapping,
                     const Eigen::MatrixXd &contact_states,
                     const Eigen::MatrixXd &contact_types,
                     const Eigen::MatrixXd &joint_2d_positions);
    // void UpdateJoint3dPositions();
    // void UpdateConfigPino(int t, const double *q);
    // void UpdateConfig(int t, const double *q_pino);

    void LoadContactInfo(const Eigen::VectorXd &contact_mapping,
                         const Eigen::MatrixXd &contact_states,
                         const Eigen::MatrixXd &contact_types);
    void LoadJoint2dPositions(const Eigen::MatrixXd &joint_2d_positions);
    void UpdateJoint3dPositions();
    void UpdateJoint2dReprojected(Camera *camera);
    void UpdateKeypoint3dPositions();
    void UpdateKeypoint2dReprojected(Camera *camera);
    void SetFrictionConeGenerators(double friction_angle);
    Eigen::MatrixXd ComputeFrictionConeGeneratorsForFoot(const Eigen::Matrix<double, 3, 4> &generators_3d, int option);

    // Class accessors and mutators
    const Eigen::VectorXi & get_contact_mapping() const;
    Eigen::VectorXi & get_contact_mapping();

    const Eigen::MatrixXi & get_contact_states() const;
    Eigen::MatrixXi & get_contact_states();

    Eigen::MatrixXi::ColXpr get_contact_states_column(int i);
    Eigen::MatrixXi::ConstColXpr get_contact_states_column(int i) const;

    Eigen::MatrixXi & get_contact_types();
    const Eigen::MatrixXi & get_contact_types() const;

    Eigen::MatrixXi::ColXpr get_contact_types_column(int i);
    Eigen::MatrixXi::ConstColXpr get_contact_types_column(int i) const;

    Eigen::MatrixXd & get_joint_2d_positions();
    const Eigen::MatrixXd & get_joint_2d_positions() const;

    Eigen::MatrixXd::ColXpr get_joint_2d_positions_column(int i);
    Eigen::MatrixXd::ConstColXpr get_joint_2d_positions_column(int i) const;

    Eigen::MatrixXd & get_joint_3d_positions();
    const Eigen::MatrixXd & get_joint_3d_positions() const;
    Eigen::VectorXd get_joint_3d_positions_column(int i);

    Eigen::MatrixXd & get_joint_2d_reprojected();
    const Eigen::MatrixXd & get_joint_2d_reprojected() const;

    Eigen::MatrixXd & get_keypoint_3d_positions();
    const Eigen::MatrixXd & get_keypoint_3d_positions() const;

    Eigen::MatrixXd & get_keypoint_2d_reprojected();
    const Eigen::MatrixXd & get_keypoint_2d_reprojected() const;

    Eigen::MatrixXd & get_friction_cone_generators();
    const Eigen::MatrixXd & get_friction_cone_generators() const;

    Eigen::MatrixXd & get_friction_cone_generators_left_foot();
    const Eigen::MatrixXd & get_friction_cone_generators_left_foot() const;

    Eigen::MatrixXd & get_friction_cone_generators_right_foot();
    const Eigen::MatrixXd & get_friction_cone_generators_right_foot() const;

private:
    Eigen::VectorXi contact_mapping_;
    Eigen::MatrixXi contact_states_;
    Eigen::MatrixXi contact_types_;
    Eigen::MatrixXd joint_2d_positions_;
    Eigen::MatrixXd joint_3d_positions_;
    Eigen::MatrixXd joint_2d_reprojected_;
    // we define Openpose joints as keypoints of our human model
    int nkeypoints_;
    Eigen::MatrixXd keypoint_3d_positions_;
    Eigen::MatrixXd keypoint_2d_reprojected_;
    // ground friction
    double friction_angle_;
    Eigen::MatrixXd friction_cone_generators_;
    Eigen::MatrixXd friction_cone_generators_left_foot_;
    Eigen::MatrixXd friction_cone_generators_right_foot_;
};

#endif // ifndef __DATALOADER_PERSON_H__
