#ifndef BAYESIAN_H_
#define BAYESIAN_H_

namespace baysian {

class Bayesian {
 public:
  virtual void Train(char *) = 0;
  virtual void Predict(char *) = 0;
  // calculate the probability of each choice
  // and choose the greatest one as our prediction
 protected:
  double *num_class_for_each_attributes_;
  // this array store the total number of
  // each decision's class in training data
  int *is_discrete_;  // this array store the information about each attribute
                      // is continuous or not
  int *num_class_for_each_attribute_;  // this array store the number of classes
                                       // of each attribute
  int num_train_instances_;            // store the number of training instances
  int num_test_instances_;             // store the number of testing instances
  int num_attributes_;                 // store the number of attributes
  int num_output_class_;               // the number of output classes

  virtual void Accuracy(int *, int *) const;  // claculate the Accuracy
  void ParseConfiguration(char *);
};

}  // namespace baysian
#endif
