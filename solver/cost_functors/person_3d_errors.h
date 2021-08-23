#ifndef __PERSON_3D_ERRORS_H__
#define __PERSON_3D_ERRORS_H__

#include <ceres/ceres.h>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

#include "../dataloader/dataloader_person.h"

using namespace Eigen;
using namespace std;
using namespace pinocchio;

struct CostFunctorPerson3dErrors
{
    typedef ceres::DynamicNumericDiffCostFunction<CostFunctorPerson3dErrors> CostFunctionPerson3dErrors;

    CostFunctorPerson3dErrors(
        int i,
        DataloaderPerson *person_loader);

    bool Evaluate(
        double const *const * parameters,
        double * residual,
        double ** jacobians) const;

    bool operator()(double const *const *parameters, double *residual) const
    {
        return Evaluate(parameters, residual, NULL);
    }

    static ceres::CostFunction *Create(
        int i,
        DataloaderPerson *person_loader);

    int get_njoints() const;
    int get_nq() const;

private:
    int i_;
    int nq_;
    int nq_pino_;
    int njoints_;
    vector<int> joint_ids_;
    VectorXd joint_3d_positions_rel_;
    DataloaderPerson *person_loader_;
};

#endif // ifndef __PERSON_3D_ERRORS_H__
