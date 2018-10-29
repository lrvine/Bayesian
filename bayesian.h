#ifndef bayesian_h
#define bayesian_h

namespace baysian {

class bayesian {
 protected:
  int trainInstances;  // store the number of training instances
  int testInstances;   // store the number of testing instances
  int attributes;      // store the number of attributes
  // calculate the probability of each choice and choose the greatest one as our
  // prediction
  void accuracy(int *, int *);  // claculate the accuracy
};

}  // namespace baysian
#endif
