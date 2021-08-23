#include "pose_prior_gmm.h"

PosePriorGmm::PosePriorGmm(const Eigen::MatrixXd &means,
                           const Eigen::MatrixXd &precs,
                           const Eigen::VectorXd &weights,
                           int prefix)
    : means_(means),
      precs_(precs),
      weights_(weights),
      prefix_(prefix)
{
    nq_ = (int) means.rows();
    num_gaussians_ = (int) means.cols();
    min_component_idx_ = -1;
    // pre-compute sqrt_minus_log_weights_
    Eigen::VectorXd minus_log_weights = -weights.array().log();
    sqrt_minus_log_weights_ = minus_log_weights.array().sqrt();
    // initialize log-likehood vectors and save sqrt_minus_log_weights_ in their last entries
    log_likelihoods_ = Eigen::MatrixXd::Zero(nq_+1, num_gaussians_);
    log_likelihoods_.bottomRows<1>() = sqrt_minus_log_weights_.transpose();
    log_likelihoods_scalar_ = Eigen::VectorXd::Zero(num_gaussians_);
    // pre-compute prec_transpose_over_sqrt2_
    prec_transpose_over_sqrt2_ = Eigen::MatrixXd::Zero(num_gaussians_ * nq_, nq_);
    for (int i = 0; i < num_gaussians_; i++)
    {
        prec_transpose_over_sqrt2_.block(i * nq_, 0, nq_, nq_) = precs.block(i * nq_, 0, nq_, nq_).transpose() / sqrt(2.0);
    }
    // initialize joint_ids_smpl_to_pino_
    int joint_ids_smpl_to_pino[24] = {0, 1, 5, 9, 2, 6,
                                      10, 3, 7, 11, 4, 8,
                                      12, 14, 19, 13, 15, 20,
                                      16, 21, 17, 22, 18, 23};
    for(int i=0; i<24; i++)
    {
        joint_ids_smpl_to_pino_.push_back(joint_ids_smpl_to_pino[i]);
    }
}

Eigen::VectorXd PosePriorGmm::ConfigPinoToSmpl(const Eigen::VectorXd &q_mat)
{
    int joint_id_pino;
    Eigen::VectorXd q_smpl = Eigen::VectorXd::Zero(75);
    for (int j = 0; j < 24; j++)
    {
        joint_id_pino = joint_ids_smpl_to_pino_[(size_t)j];
        q_smpl.block<3, 1>(3 * j + 3, 0) = q_mat.block<3, 1>(3 * joint_id_pino + 3, 0);
    }
    return q_smpl;
}

// ComputeMinComponent is called before any call to ComputeLogLikelihood
// it computes the minus log-likelihood vectors for all the Gaussians
// and then saves the index of the Gaussian with the maximum log-likelihood
void PosePriorGmm::ComputeMinComponent(const Eigen::VectorXd &q_smpl)
{
    // update log_likelihoods_
    for (int i = 0; i < num_gaussians_; i++)
    {
        log_likelihoods_.block(0,i,nq_,1) = prec_transpose_over_sqrt2_.block(i * nq_, 0, nq_, nq_) *
            (q_smpl.bottomRows(nq_) - means_.col(i));
    }
    // update log_likelihoods_scalar_
    log_likelihoods_scalar_ = log_likelihoods_.array().pow(2).matrix().colwise().sum().transpose();
    // compute the minimum minus-log-likelihood and save index
    int min_index = 0;
    double min_value = log_likelihoods_scalar_(0);
    for (int i = 1; i < num_gaussians_; i++)
    {
        if (log_likelihoods_scalar_(i) < min_value)
        {
            min_value = log_likelihoods_scalar_(i);
            min_index = i;
        }
    }
    min_component_idx_ = min_index;
}

// returns a vector whose square norm is the approximated minus log-likehood
Eigen::VectorXd PosePriorGmm::ComputeLogLikelihood(const Eigen::VectorXd &q_mat)
{
    Eigen::VectorXd q_smpl = ConfigPinoToSmpl(q_mat);
    ComputeMinComponent(q_smpl);
    return log_likelihoods_.col(min_component_idx_);
}

// Class accessors and mutators
Eigen::MatrixXd PosePriorGmm::get_means()
{
    return means_;
}
void PosePriorGmm::set_means(const Eigen::MatrixXd &means)
{
    means_ = means;
}
Eigen::MatrixXd PosePriorGmm::get_precs()
{
    return precs_;
}
void PosePriorGmm::set_precs(const Eigen::MatrixXd &precs)
{
    precs_ = precs;
}
Eigen::VectorXd PosePriorGmm::get_weights()
{
    return weights_;
}
void PosePriorGmm::set_weights(const Eigen::VectorXd &weights)
{
    weights_ = weights;
}
int PosePriorGmm::get_prefix()
{
    return prefix_;
}
int PosePriorGmm::get_nq()
{
    return nq_;
}
int PosePriorGmm::get_num_gaussians()
{
    return num_gaussians_;
}
int PosePriorGmm::get_min_component_idx()
{
    return min_component_idx_;
}
Eigen::VectorXd PosePriorGmm::get_sqrt_minus_log_weights()
{
    return sqrt_minus_log_weights_;
}
Eigen::MatrixXd PosePriorGmm::get_prec_transpose_over_sqrt2()
{
    return prec_transpose_over_sqrt2_;
}
Eigen::MatrixXd PosePriorGmm::get_log_likelihoods()
{
    return log_likelihoods_;
}
Eigen::VectorXd PosePriorGmm::get_log_likelihoods_scalar()
{
    return log_likelihoods_scalar_;
}
