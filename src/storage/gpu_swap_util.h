#ifndef MXNET_GPU_SWAP_UTIL_H_
#define MXNET_GPU_SWAP_UTIL_H_
#include <string>

namespace mxnet {
static inline std::string MBString(size_t size) {
  return std::to_string(size / 1e6) + "MB";
}

static inline std::string GBString(size_t size) {
  return std::to_string(size / 1e9) + "GB";
}

} // namespace mxnet

#endif // MXNET_GPU_SWAP_UTIL_H_
