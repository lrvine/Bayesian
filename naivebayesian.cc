#include "naivebayesian.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace machinelearning {
namespace baysian {

// initialize all the information we need from training data
NaiveBayesian::NaiveBayesian(char *cfg_file) {
  std::cout << "Init NaiveBayesian" << std::endl;
  ParseConfiguration(cfg_file);
}

void NaiveBayesian::Train(char *train_file) {
  std::ifstream trainingDataFile;
  std::string Buf;
  trainingDataFile.open(train_file);
  if (!trainingDataFile) {
    std::cout << "Can't open training data file!" << std::endl;
    return;
  }

  // this "probabilityTable" store the count of every possible combination
  // and divide each of them by the total occurences
  probabilityTable.resize(num_attributes_ * num_output_class_);
  for (int i = 0; i < num_attributes_; ++i) {
    if (is_discrete_[i] == 1)  // if this attribute is is_discrete_
    {
      for (int j = (i * num_output_class_); j < ((i + 1) * num_output_class_);
           ++j)
        probabilityTable[j].resize(num_class_for_each_attribute_[i], 0);
    } else if (is_discrete_[i] == 0)  // if this attribute is continuous
    {
      for (int j = (i * num_output_class_); j < ((i + 1) * num_output_class_);
           ++j)
        probabilityTable[j].resize(2, 0);
      // the first one store mean , the second store the standard deviation
    }
  }

  // use a array to store each instance for further processing
  std::vector<double> oneLine((num_attributes_ + 1), 0);

  // store the information of each instance into probabilityTable
  for (int i = 1; i <= num_train_instances_; ++i) {
    getline(trainingDataFile, Buf);
    std::stringstream lineStream(Buf);

    for (int j = 0; j <= num_attributes_; ++j) {
      // read one instance for processing
      getline(lineStream, Buf, ',');
      oneLine[j] = stod(Buf);
    }

    output_class_cnt_[static_cast<int>(oneLine[num_attributes_]) -
                      1]++;  // count the result

    for (int j = 0; j < num_attributes_; ++j) {
      if (is_discrete_[j] == 1)  // if this attribute is is_discrete_
      {
        probabilityTable[j * num_output_class_ +
                         static_cast<int>(oneLine[num_attributes_]) -
                         1][static_cast<int>(oneLine[j]) - 1]++;
      } else if (is_discrete_[j] == 0)  // if this attribute is continuous
      {
        probabilityTable[j * num_output_class_ +
                         static_cast<int>(oneLine[num_attributes_]) - 1][0] +=
            oneLine[j];
        probabilityTable[j * num_output_class_ +
                         static_cast<int>(oneLine[num_attributes_]) - 1][1] +=
            pow(oneLine[j], 2);
      }
    }
  }

  trainingDataFile.close();
  // processing the information in the probabilityTable to get the proabability
  for (int t = 0; t < num_attributes_; ++t) {
    if (is_discrete_[t] == 1)  // if this attribute is is_discrete_
    {
      for (int d = 0; d < num_output_class_; ++d) {
        int correction = 0;
        for (int o = 0; o < num_class_for_each_attribute_[t]; ++o)
        // this loop judge weather there is zero occurence of some conjuction
        // if it dose, then do Laplacian correction
        {
          if (probabilityTable[(t * num_output_class_ + d)][o] == 0) {
            correction = num_class_for_each_attribute_[t];
            for (int p = 0; p < num_class_for_each_attribute_[t]; ++p) {
              probabilityTable[(t * num_output_class_ + d)][p]++;
            }
            break;
          }
        }
        for (int w = 0; w < num_class_for_each_attribute_[t]; ++w)
        // claculate every conjuction's contribution of probability
        {
          probabilityTable[(t * num_output_class_ + d)][w] /=
              (output_class_cnt_[d] + correction);
        }
      }
    } else if (is_discrete_[t] == 0)
    // if this attribute is continuous,we assume it's Gaussian distribution
    // claculate the mean and standard deviation of each continuous attribute
    {
      for (int h = 0; h < num_output_class_; ++h) {
        long double a0 =
            pow(probabilityTable[(t * num_output_class_ + h)][0], 2) /
            output_class_cnt_[h];
        long double a1 = probabilityTable[(t * num_output_class_ + h)][1] - a0;
        long double a2 = a1 / output_class_cnt_[h];
        long double a3 = sqrt(a2);
        probabilityTable[(t * num_output_class_ + h)][1] = a3;
        probabilityTable[(t * num_output_class_ + h)][0] /=
            output_class_cnt_[h];
      }
    }
  }
  // calculate the probability of each resulting class
  for (int probIndex = 0; probIndex < num_output_class_; ++probIndex)
    output_class_cnt_[probIndex] =
        output_class_cnt_[probIndex] / num_train_instances_;
}

// calculate the probability of each choice and choose the greatest one as our
// prediction
std::vector<int> NaiveBayesian::Predict(char *test_file, bool has_truth = 1) {
  std::vector<int> outcome(num_test_instances_, 0);
  // this vector store our prediciton
  std::vector<int> truth(num_test_instances_, 0);
  // this vector store the real result for comparison
  std::vector<int> oneLine((num_attributes_), 0);
  // store each instance for processing
  std::vector<long double> decision((num_output_class_), 0);
  // store the probability of each choice

  std::ifstream testInputFile(test_file);
  if (!testInputFile) {
    std::cout << "Can't open test data file!" << std::endl;
    return outcome;
  }
  std::cout << "Start prediction " << std::endl;
  std::string Buf;

  for (int i = 0; i < num_test_instances_; ++i) {
    for (int m = 0; m < num_output_class_; ++m) decision[m] = 1;
    // set the array's entries as 1 for each testing instance

    getline(testInputFile, Buf);
    std::stringstream lineStream(Buf);

    for (int u = 0; u < num_attributes_; u++) {
      getline(lineStream, Buf, ',');
      oneLine[u] = stod(Buf);
    }
    // read one instance for prediction
    if (has_truth) {
      getline(lineStream, Buf, ',');
      truth[i] = stod(Buf);
      // store the truth
    }
    for (int x = 0; x < num_output_class_; ++x) {
      for (int j = 0; j < num_attributes_; ++j) {
        if (is_discrete_[j] == 1)  // if this attribute is is_discrete_
        {
          decision[x] *= probabilityTable[(j * num_output_class_) + x]
                                         [static_cast<int>(oneLine[j]) - 1];
        } else if (is_discrete_[j] == 0)
        // if this attribute is continuous , then use the Gaussian distribution
        // formular to calculate it's contribution of probability
        {
          long double a0 = -pow(
              (oneLine[j] - probabilityTable[(j * num_output_class_) + x][0]),
              2);
          long double a1 =
              2 * pow(probabilityTable[(j * num_output_class_) + x][1], 2);
          long double a2 = a0 / a1;
          long double a3 = exp(a2);
          long double a4 =
              (0.39894228 / probabilityTable[(j * num_output_class_) + x][1]) *
              a3;
          decision[x] *= a4;
        }
      }
      decision[x] *= output_class_cnt_[x];
    }
    // decide which choice has the highest probability
    int big = 0;
    long double hug = decision[0];
    for (int j = 1; j < num_output_class_; ++j) {
      if (decision[j] > hug) {
        big = j;
        hug = decision[j];
      }
    }
    outcome[i] = (big + 1);
  }
  if (has_truth) Accuracy(outcome, truth);
  // call function "caauracy" to calculate the accuracy
  return outcome;
}

}  // namespace baysian
}  // namespace machinelearning