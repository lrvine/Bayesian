#include "bayesian.h"
#include <iostream>

namespace baysian {

// calculate the accuracy
void bayesian::accuracy(int *outcome, int *result) {
  double correct = 0;  // store the number of correct predictions

  for (int i = 0; i < testInstances; i++)  // count the number of correct predictions
  {
    if (outcome[i] == result[i]) correct++;
#ifdef DEBUG
    std::cout << "predict to be " << outcome[i] << " is actually " << result[i]
              << std::endl;
#endif
  }
  std::cout << "Total " << testInstances << " data have " << correct
            << " correct predictions" << std::endl;
  double percentage = correct / testInstances;  // calculate the accuracy
  std::cout << "Accuracy is " << percentage * 100 << "%" << std::endl;
}

}  // namespace baysian
