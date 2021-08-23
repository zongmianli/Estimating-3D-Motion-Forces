#ifndef __CONTACT_TEMPORAL_H__
#define __CONTACT_TEMPORAL_H__

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "../dataloader/dataloader_person.h"
#include "../dataloader/dataloader_object.h"

// This cost function penalizes the linear velocities of joint j at timestep i
// for ground contact joints: penalize velocity in world frame
// for object contact joints: penalize velocity in object frame!

struct CostFunctorContactTemporal
{
    typedef ceres::DynamicNumericDiffCostFunction<CostFunctorContactTemporal> CostFunctionContactTemporal;
    typedef Eigen::Map<const Eigen::VectorXd> ConstMapVectorXd;

    CostFunctorContactTemporal(int i,
                               DataloaderPerson *person_loader,
                               DataloaderObject *object_loader,
                               const Eigen::VectorXd &weights,
                               bool update_6d_basis_only,
                               bool smooth_oc,
                               bool virtual_object);

    bool Evaluate(double const *const * parameters,
                  double * residual,
                  double ** jacobians) const;

    bool operator()(double const *const *parameters, double *residual) const
    {
        return Evaluate(parameters, residual, NULL);
    }

    static ceres::CostFunction *Create(int i,
                                       DataloaderPerson *person_loader,
                                       DataloaderObject *object_loader,
                                       const Eigen::VectorXd &weights,
                                       bool update_6d_basis_only,
                                       bool smooth_oc,
                                       bool *has_object_contact,
                                       bool virtual_object);

    int get_nq_person_effective();

    int get_num_contact_joints();

    bool has_object_contact() const;

    bool has_ground_contact() const;

    private:
    int i_;
    DataloaderPerson *person_loader_;
    DataloaderObject *object_loader_;
    Eigen::VectorXd q_init_; // initial human configuration vector (75d)
    Eigen::VectorXd q_minus1_init_; // initial human configuration vector (75d)
    double dt_;
    int nq_person_effective_;
    int num_object_contact_joints_;
    int num_ground_contact_joints_;
    std::vector<int> object_contact_joints_;
    std::vector<int> ground_contact_joints_;
    std::vector<int> object_contact_types_;
    std::vector<int> ground_contact_types_;
    Eigen::VectorXd weights_;
    bool virtual_object_;
};

#endif // #ifndef __CONTACT_TEMPORAL_H__
