#ifndef bayesian_h
#define bayesian_h

namespace baysian {

class bayesian {
 protected:
  double *classCount;  // this array store the total number of each
                       // decision's class in training data
  int *discrete;  // this array store the information about each attribute is
                  // continuous or not
  int *classNum;  // this array store the number of classes of each attribute
  int trainInstances;  // store the number of training instances
  int testInstances;   // store the number of testing instances
  int attributes;      // store the number of attributes
  int outputClassNum;  // the number of output classes

  void accuracy(int *, int *);  // claculate the accuracy
 public:
  virtual void predict(char *) = 0;
  // calculate the probability of each choice and choose the greatest one as our
  // prediction
  virtual void train(char *) = 0;
};

}  // namespace baysian
#endif
