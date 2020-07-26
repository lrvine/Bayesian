#ifndef BAYESIAN_H_
#define BAYESIAN_H_

#include "machinelearning.h"

namespace machinelearning {
namespace baysian {

class Bayesian : public MachineLearning {
 protected:
  void ParseConfiguration(char *);
  std::vector<double> output_class_cnt_;
  // this array store the total number of
  // each decision's class in training data
  std::vector<int> is_discrete_;
  // this array store the information about each attribute
  // is continuous or not
  std::vector<int> num_class_for_each_attribute_;
  // this array store the number of classes of each attribute
  int num_attributes_;                 // store the number of attributes
  int num_output_class_;               // the number of output classes
};

}  // namespace baysian
}  // namespace machinelearning
#endif
