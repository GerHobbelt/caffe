#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"

/**
 Forward declare std::thread instead of including <thread>
 */
namespace std { class thread; }

namespace caffe {

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread() : thread_(), interruption_requested_(false) {}
  virtual ~InternalThread();

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  void StartInternalThread();

  /** Will not return until the internal thread has exited. */
  void StopInternalThread();

  bool is_started() const;

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  virtual void InternalThreadEntry() {}

  /* Should be tested when running loops to exit when requested. */
  bool must_stop();

 private:
  void entry(int device, Caffe::Brew mode, int rand_seed,
      int solver_count, int solver_rank, bool multiprocess);

  shared_ptr<std::thread> thread_;
  bool interruption_requested_;
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
