#include "bayesian.h"
namespace baysian {

class bayesianNetwork : public bayesian {
 private:
  long double ***cpt;
  double *classCount;  // this array store the total number of each
                       // decision's class in training data
  int *discrete;
  int *classNum;  // this array store the number of classes of each attribute
  int **parent;

 public:
  bayesianNetwork(char *);
  ~bayesianNetwork();
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
