#ifndef BAYESIANNETWORK_H_
#define BAYESIANNETWORK_H_

#include "bayesian.h"

namespace baysian {

class BayesianNetwork : public Bayesian {
 private:
  long double ***cpt;
  int **parent;

 public:
  BayesianNetwork(char *);
  ~BayesianNetwork();
  // initialize all the information we need from training data
  void predict(char *);
  // calculate the probability of each choice and choose the greatest one as our
  // prediction
  void train(char *);
};

template <class Type>
struct data {
  double key;
  Type value1;
  Type value2;
};

}  // namespace baysian

#endif