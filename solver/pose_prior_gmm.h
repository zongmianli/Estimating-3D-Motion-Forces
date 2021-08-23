#ifndef __POSE_PRIOR_GMM_H__
#define __POSE_PRIOR_GMM_H__

#include <Eigen/Core>
#include <vector>
// the PosePriorGmm class simplifies the computation of Gaussian mixture model
// arguments:
// means - (69, 8) matrix
// precs - (8*69, 69) matrix
// weights - (8, 1) matrix
// prefix - int
class PosePriorGmm
{
public:
    PosePriorGmm(const Eigen::MatrixXd &means,
                 const Eigen::MatrixXd &precs,
                 const Eigen::VectorXd &weights,
                 int prefix);
    Eigen::VectorXd ConfigPinoToSmpl(const Eigen::VectorXd &q_mat);
    void ComputeMinComponent(const Eigen::VectorXd &q_smpl);
    Eigen::VectorXd ComputeLogLikelihood(const   Eigen::VectorXd &q_mat);
    // Class accessors and mutators
    Eigen::MatrixXd get_means();
    void set_means(const Eigen::MatrixXd &means);
    Eigen::MatrixXd get_precs();
    void set_precs(const Eigen::MatrixXd &precs);
    Eigen::VectorXd get_weights();
    void set_weights(const Eigen::VectorXd &weights);
    int get_prefix();
    int get_nq();
    int get_num_gaussians();
    int get_min_component_idx();
    Eigen::VectorXd get_sqrt_minus_log_weights();
    Eigen::MatrixXd get_prec_transpose_over_sqrt2();
    Eigen::MatrixXd get_log_likelihoods();
    Eigen::VectorXd get_log_likelihoods_scalar();

private:
    Eigen::MatrixXd means_;
    Eigen::MatrixXd precs_;
    Eigen::VectorXd weights_;
    int prefix_;
    int nq_;
    int num_gaussians_;
    int min_component_idx_;
    Eigen::VectorXd sqrt_minus_log_weights_;
    Eigen::MatrixXd prec_transpose_over_sqrt2_;
    Eigen::MatrixXd log_likelihoods_;
    Eigen::VectorXd log_likelihoods_scalar_;
    std::vector<int> joint_ids_smpl_to_pino_;
};

#endif // #ifndef __POSE_PRIOR_GMM_H__
