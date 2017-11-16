#ifndef CAFFE_ADADELTA_SOLVERS_HPP_
#define CAFFE_ADADELTA_SOLVERS_HPP_

#include <string>
#include <vector>

#include "caffe/solver.hpp"
#include "sgd_solver.hpp"

namespace caffe {

/**
 * @brief Optimizes the parameters of a Net using
 *        Ada Delta solver.
 */
template <typename Dtype>
class AdaDeltaSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaDeltaSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { AdaDeltaPreSolve(); }
  explicit AdaDeltaSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AdaDeltaPreSolve(); }
  virtual inline const char* type() const { return "AdaDelta"; }

 protected:
  void AdaDeltaPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdaDeltaSolver);
};

}  // namespace caffe

#endif  // CAFFE_ADADELTA_SOLVERS_HPP_
