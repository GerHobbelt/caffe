#include <mutex>
#include <string>
#include <thread>

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template<typename T>
BlockingQueue<T>::BlockingQueue() {
  abort_ = false;
}

template<typename T>
void BlockingQueue<T>::push(const T& t) {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.push(t);
  }
  condition_.notify_one();
}

template<typename T>
bool BlockingQueue<T>::try_pop(T* t) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  queue_.pop();
  return true;
}

template<typename T>
T BlockingQueue<T>::pop(const string& log_on_wait) {
  std::unique_lock<std::mutex> lock(mutex_);

  while (queue_.empty() && !abort_) {
    if (!log_on_wait.empty()) {
      LOG_EVERY_N(INFO, 1000)<< log_on_wait;
    }
    condition_.wait(lock);
  }

  if (abort_) {
    if (abort_) throw operation_aborted();
  }

  T t = queue_.front();
  queue_.pop();
  return t;
}

template<typename T>
bool BlockingQueue<T>::try_peek(T* t) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  return true;
}

template<typename T>
T BlockingQueue<T>::peek() {
  std::unique_lock<std::mutex> lock(mutex_);

  while (queue_.empty() && !abort_) {
    condition_.wait(lock);
  }

  if (abort_) {
    if (abort_) throw operation_aborted();
  }

  return queue_.front();
}

template<typename T>
size_t BlockingQueue<T>::size() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return queue_.size();
}


template<typename T>
void BlockingQueue<T>::abort() {
  std::unique_lock<std::mutex> lock(mutex_);
  abort_ = true;
  condition_.notify_all();
}
template class BlockingQueue<Batch<float>*>;
template class BlockingQueue<Batch<double>*>;

}  // namespace caffe
