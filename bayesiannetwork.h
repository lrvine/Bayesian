#ifndef BAYESIANNETWORK_H_
#define BAYESIANNETWORK_H_

#include "bayesian.h"

namespace machinelearning {
namespace baysian {

class BayesianNetwork : public Bayesian {
 public:
  BayesianNetwork(char *);
  // initialize all the information we need from training data
  std::vector<int> Predict(char *, bool);
  // calculate the probability of each choice and choose the greatest one as our
  // prediction
  void Train(char *);

 private:
  std::vector<std::vector<std::vector<long double> > >
      conditional_probability_table_;
  std::vector<std::vector<int> > nodes_parents_;
};

template <class Type>
struct KeyAndTwoValue {
  double key;
  Type value1;
  Type value2;
};

}  // namespace baysian
}  // namespace machinelearning
#endif