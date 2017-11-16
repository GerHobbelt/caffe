#ifndef CAFFE_ADAGRAD_SOLVERS_HPP_
#define CAFFE_ADAGRAD_SOLVERS_HPP_

#include <string>
#include <vector>

#include "caffe/solver.hpp"
#include "sgd_solver.hpp"

namespace caffe {

/**
 * @brief Optimizes the parameters of a Net using
 *        ada grad.
 */
template <typename Dtype>
class AdaGradSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaGradSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { constructor_sanity_check(); }
  explicit AdaGradSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }
  virtual inline const char* type() const { return "AdaGrad"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with AdaGrad.";
  }

  DISABLE_COPY_AND_ASSIGN(AdaGradSolver);
};

}  // namespace caffe

#endif  // CAFFE_ADAGRAD_SOLVERS_HPP_
