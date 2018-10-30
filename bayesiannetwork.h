#include "bayesian.h"
namespace baysian {

class bayesianNetwork : public bayesian {
 public:
  bayesianNetwork(char *, char *, char *);
  // initialize all the information we need from training data
  void predict(long double ***, int *, double *, int **, char *);
  // calculate the probability of each choice and choose the greatest one as our
  // prediction
  void predict(char *);
  void train(char *);
};

template <class Type>
struct data {
  double key;
  Type value1;
  Type value2;
};

}  // namespace baysian
