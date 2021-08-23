#include "bindings.h"
#include "../pose_prior_gmm.h"

void ExposePosePriorGmm()
{
    bp::class_<PosePriorGmm>("PosePriorGmm",
                             bp::init<const Eigen::MatrixXd &,
                             const Eigen::MatrixXd &,
                             const Eigen::VectorXd &,
                             int>())
        .def("ConfigPinoToSmpl", &PosePriorGmm::ConfigPinoToSmpl)
        .def("ComputeMinComponent", &PosePriorGmm::ComputeMinComponent)
        .def("ComputeLogLikelihood", &PosePriorGmm::ComputeLogLikelihood)
        .add_property("means_", &PosePriorGmm::get_means, &PosePriorGmm::set_means)
        .add_property("precs_", &PosePriorGmm::get_precs, &PosePriorGmm::set_precs)
        .add_property("weights_", &PosePriorGmm::get_weights, &PosePriorGmm::set_weights)
        .add_property("prefix_", &PosePriorGmm::get_prefix)
        .add_property("nq_", &PosePriorGmm::get_nq)
        .add_property("num_gaussians_", &PosePriorGmm::get_num_gaussians)
        .add_property("min_component_idx_", &PosePriorGmm::get_min_component_idx)
        .add_property("sqrt_minus_log_weights_", &PosePriorGmm::get_sqrt_minus_log_weights)
        .add_property("prec_transpose_over_sqrt2_", &PosePriorGmm::get_prec_transpose_over_sqrt2)
        .add_property("log_likelihoods_", &PosePriorGmm::get_log_likelihoods)
        .add_property("log_likelihoods_scalar_", &PosePriorGmm::get_log_likelihoods_scalar);
}
