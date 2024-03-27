#include "common/result_helper.h"

#include <cmath>

bool CheckResultSame(float* out, float* ref, int size, float eps) {
  for (int i = 0; i < size; i++) {
    // if (fabs(out[i] - ref[i]) > eps) {
    if (out[i] != ref[i]) {
      return false;
    }
  }

  return true;
}
