#ifndef NAIVEBAYESIAN_H_
#define NAIVEBAYESIAN_H_

#include <vector>

#include "bayesian.h"

namespace baysian {

class NaiveBayesian : public Bayesian {
 public:
  void Train(char *);
  void Predict(char *);
  // calculate the probability of each choice
  // and choose the greatest one as our prediction
  NaiveBayesian(char *);
  // initialize all the information we need from training data
  ~NaiveBayesian();
  // release memory
 private:
  long double **probabilityTable;
};

}  // namespace baysian

#endif