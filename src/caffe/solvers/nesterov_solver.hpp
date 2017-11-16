#ifndef CAFFE_NESTEROV_SOLVERS_HPP_
#define CAFFE_NESTEROV_SOLVERS_HPP_

#include <string>
#include <vector>

#include "caffe/solver.hpp"
#include "sgd_solver.hpp"

namespace caffe {

/**
 * @brief Optimizes the parameters of a Net using
 *        nestevrov solver.
 */
template <typename Dtype>
class NesterovSolver : public SGDSolver<Dtype> {
 public:
  explicit NesterovSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) {}
  explicit NesterovSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) {}
  virtual inline const char* type() const { return "Nesterov"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(NesterovSolver);
};

}  // namespace caffe

#endif  // CAFFE_NESTEROV_SOLVERS_HPP_
