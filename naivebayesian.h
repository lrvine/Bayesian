#ifndef NAIVEBAYESIAN_H_
#define NAIVEBAYESIAN_H_

#include <vector>

#include "bayesian.h"

namespace machinelearning {
namespace baysian {

class NaiveBayesian : public Bayesian {
 public:
  NaiveBayesian(char *);
  // initialize all the information we need from training data
  ~NaiveBayesian();
  // release memory
  void Train(char *);
  std::vector<int> Predict(char *, bool);
  // calculate the probability of each choice
  // and choose the greatest one as our prediction

 private:
  long double **probabilityTable;
};

}  // namespace baysian
}  // namespace machinelearning
#endif