#include "bayesian.h"

#include <fstream>
#include <iostream>

namespace baysian {

// calculate the accuracy
void bayesian::accuracy(int *outcome, int *truth) {
  double correct = 0;  // store the number of correct predictions

  for (int i = 0; i < testInstances; i++)
  // count the number of correct predictions
  {
    if (outcome[i] == truth[i]) correct++;
#ifdef DEBUG
    std::cout << "predict to be " << outcome[i] << " is actually " << truth[i]
              << std::endl;
#endif
  }
  std::cout << "Total " << testInstances << " data have " << correct
            << " correct predictions" << std::endl;
  double percentage = correct / testInstances;  // calculate the accuracy
  std::cout << "Accuracy is " << percentage * 100 << "%" << std::endl;
}

void bayesian::parse_configuration(char *cfg_file) {
  std::ifstream configure;
  configure.open(cfg_file);
  if (!configure) {
    std::cout << "Can't open configuration file!" << std::endl;
    return;
  }

  configure >> trainInstances >> testInstances >> attributes;
  // read the number of training instances and attributes

  discrete = new int[attributes];
  // this array store the information about each attribute is continuous or not
  for (int i = 0; i < attributes; i++) configure >> discrete[i];
  //  read the information about continuous or not

  classNum = new int[attributes + 1];
  // this array store the number of classes of each attribute

  for (int i = 0; i <= attributes; i++) {  // read the number of classes
    configure >> classNum[i];
    if (discrete[i])  // set classNum as 2 for continuous data
      classNum[i] = 2;
  }

  outputClassNum = classNum[attributes];
  classCount = new double[outputClassNum];
  // this array store the total number of each decision's class in training data
  for (int i = 0; i < outputClassNum; i++) classCount[i] = 0;

  configure.close();
}

}  // namespace baysian
