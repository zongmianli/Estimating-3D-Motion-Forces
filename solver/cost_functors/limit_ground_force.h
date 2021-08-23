#ifndef __LIMIT_GROUND_FORCE_H__
#define __LIMIT_GROUND_FORCE_H__

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "../dataloader/dataloader_person.h"
//#include "../dataloader/dataloader_object.h"

struct CostFunctorLimitGroundForce
{
    typedef ceres::DynamicNumericDiffCostFunction<CostFunctorLimitGroundForce> CostFunctionLimitGroundForce;

    CostFunctorLimitGroundForce(
        int i,
        DataloaderPerson *person_loader);

    bool operator()(
        double const *const *parameters,
        double *residual) const
    {
        return Evaluate(parameters, residual, NULL);
    }

    bool Evaluate(
        double const *const * parameters,
        double * residual,
        double ** jacobians) const;

    static ceres::CostFunction *Create(
        int i,
        DataloaderPerson *person_loader);

    int get_num_contact_points();

  private:
    int num_contact_joints_;
    int num_contact_points_;
    std::vector<int> contact_joints_;
    Eigen::VectorXi contact_mapping_;
};

#endif // #ifndef __LIMIT_GROUND_FORCE_H__
