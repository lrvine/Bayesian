#include "bayesian.h"
namespace baysian {

class naiveBayesian : public bayesian {
 protected:
  void classifier(long double **, int *, double *, int *, char *);
  // calculate the probability of each choice and choose the greatest one as our
  // prediction
 public:
  naiveBayesian(char *, char *, char *);
  // initialize all the information we need from training data
};

}  // namespace baysian
