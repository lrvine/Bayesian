#ifndef ML_H_
#define ML_H_

namespace machinelearning {

class MachineLearning {
 public:
  virtual void Train(char *) = 0;
  virtual void Predict(char *) = 0;
  void Accuracy(int *, int *) const;

 protected:
  virtual void ParseConfiguration(char *) = 0;
  int num_train_instances_;  // store the number of training instances
  int num_test_instances_;   // store the number of testing instances
};

}  // namespace machinelearning
#endif
