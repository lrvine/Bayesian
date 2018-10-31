#include "naivebayesian.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace baysian {

// initialize all the information we need from training data
NaiveBayesian::NaiveBayesian(char *cfg_file) {
  std::cout << "Run NaiveBayesian" << std::endl;
  ParseConfiguration(cfg_file);
}

NaiveBayesian::~NaiveBayesian() {
  // release the memory
#ifdef DEBUG
  std::cout << " release memory " << std::endl;
#endif
  for (int x = 0; x < (num_attributes_ * num_output_class_); x++)
    delete[] probabilityTable[x];
  delete[] probabilityTable;
  delete[] is_discrete_;
  delete[] num_class_for_each_attribute_;
  delete[] num_class_for_each_attributes_;
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
  probabilityTable = new long double *[(num_attributes_ * num_output_class_)];
  for (int j = 0; j < num_attributes_; j++) {
    if (is_discrete_[j] == 1)  // if this attribute is is_discrete_
    {
      for (int i = (j * num_class_for_each_attribute_[num_attributes_]);
           i < (j * num_class_for_each_attribute_[num_attributes_] +
                num_class_for_each_attribute_[num_attributes_]);
           i++)
        probabilityTable[i] = new long double[num_class_for_each_attribute_[j]];
    } else if (is_discrete_[j] == 0)  // if this attribute is continuous
    {
      for (int i = (j * num_class_for_each_attribute_[num_attributes_]);
           i < (j * num_class_for_each_attribute_[num_attributes_] +
                num_class_for_each_attribute_[num_attributes_]);
           i++)
        probabilityTable[i] = new long double[2];
      // the first one store mean , the second store the standard deviation
    }
  }

  // initialize the probabilityTable to be 0
  for (int r = 0; r < num_attributes_; r++) {
    if (is_discrete_[r] == 1) {
      for (int g = (r * num_class_for_each_attribute_[num_attributes_]);
           g < (r * num_class_for_each_attribute_[num_attributes_] +
                num_class_for_each_attribute_[num_attributes_]);
           g++) {
        for (int e = 0; e < num_class_for_each_attribute_[r]; e++)
          probabilityTable[g][e] = 0;
      }
    } else if (is_discrete_[r] == 0) {
      for (int g = (r * num_class_for_each_attribute_[num_attributes_]);
           g < (r * num_class_for_each_attribute_[num_attributes_] +
                num_class_for_each_attribute_[num_attributes_]);
           g++) {
        for (int e = 0; e < 2; e++) probabilityTable[g][e] = 0;
      }
    }
  }

  // use a array to store each instance for further processing
  double *oneLine = new double[num_attributes_ + 1];

  // store the information of each instance into probabilityTable
  for (int i = 1; i <= num_train_instances_; i++) {
    getline(trainingDataFile, Buf);
    std::stringstream lineStream(Buf);

    for (int y = 0; y <= num_attributes_;
         y++) {  // read one instance for processing
      getline(lineStream, Buf, ',');
      oneLine[y] = stod(Buf);
    }

    num_class_for_each_attributes_[static_cast<int>(oneLine[num_attributes_]) -
                                   1]++;  // count the result

    for (int j = 0; j < num_attributes_; j++) {
      if (is_discrete_[j] == 1)  // if this attribute is is_discrete_
      {
        probabilityTable[j * num_class_for_each_attribute_[num_attributes_] +
                         static_cast<int>(oneLine[num_attributes_]) -
                         1][static_cast<int>(oneLine[j]) - 1]++;
      } else if (is_discrete_[j] == 0)  // if this attribute is continuous
      {
        probabilityTable[j * num_class_for_each_attribute_[num_attributes_] +
                         static_cast<int>(oneLine[num_attributes_]) - 1][0] +=
            oneLine[j];
        probabilityTable[j * num_class_for_each_attribute_[num_attributes_] +
                         static_cast<int>(oneLine[num_attributes_]) - 1][1] +=
            pow(oneLine[j], 2);
      }
    }
  }

  delete[] oneLine;
  trainingDataFile.close();
  // processing the information in the protalbe to get the proabability
  for (int t = 0; t < num_attributes_; t++) {
    if (is_discrete_[t] == 1)  // if this attribute is is_discrete_
    {
      for (int d = 0; d < num_class_for_each_attribute_[num_attributes_]; d++) {
        int correction = 0;
        for (int o = 0; o < num_class_for_each_attribute_[t]; o++)
        // this loop judge weather there is zero occurence of some conjuction
        // if it dose, then do Laplacian correction
        {
          if (probabilityTable[(
                  t * num_class_for_each_attribute_[num_attributes_] + d)][o] ==
              0) {
            correction = num_class_for_each_attribute_[t];
            for (int p = 0; p < num_class_for_each_attribute_[t]; p++) {
              probabilityTable[(
                  t * num_class_for_each_attribute_[num_attributes_] + d)][p]++;
            }
            break;
          }
        }
        for (int w = 0; w < num_class_for_each_attribute_[t]; w++)
        // claculate every conjuction's contribution of probability
        {
          probabilityTable[(t * num_class_for_each_attribute_[num_attributes_] +
                            d)][w] /=
              (num_class_for_each_attributes_[d] + correction);
        }
      }
    } else if (is_discrete_[t] == 0)
    // if this attribute is continuous,we assume it's Gaussian distribution
    // claculate the mean and standard deviation of each continuous attribute
    {
      for (int h = 0; h < num_class_for_each_attribute_[num_attributes_]; h++) {
        long double a0 =
            pow(probabilityTable[(
                    t * num_class_for_each_attribute_[num_attributes_] + h)][0],
                2) /
            num_class_for_each_attributes_[h];
        long double a1 =
            probabilityTable[(
                t * num_class_for_each_attribute_[num_attributes_] + h)][1] -
            a0;
        long double a2 = a1 / num_class_for_each_attributes_[h];
        long double a3 = sqrt(a2);
        probabilityTable[(t * num_class_for_each_attribute_[num_attributes_] +
                          h)][1] = a3;
        probabilityTable[(t * num_class_for_each_attribute_[num_attributes_] +
                          h)][0] /= num_class_for_each_attributes_[h];
      }
    }
  }
  // calculate the probability of each resulting class
  for (int probIndex = 0; probIndex < num_output_class_; probIndex++)
    num_class_for_each_attributes_[probIndex] =
        num_class_for_each_attributes_[probIndex] / num_train_instances_;
}

// calculate the probability of each choice and choose the greatest one as our
// prediction
void NaiveBayesian::Predict(char *test_file) {
  std::ifstream testInputFile(test_file);
  if (!testInputFile) {
    std::cout << "Can't open test data file!" << std::endl;
    return;
  }
  std::cout << "Start prediction " << std::endl;
  std::string Buf;

  int *truth = new int[num_test_instances_];  // this array store the real
                                              // result for comparison
  for (int w = 0; w < num_test_instances_; w++) {
    truth[w] = 0;
  }

  int *outcome =
      new int[num_test_instances_];  // this array store our prediciton
  for (int f = 0; f < num_test_instances_; f++) {
    outcome[f] = 0;
  }

  double *oneLine =
      new double[num_attributes_ + 1];  // store each instance for processing
  long double *decision =
      new long double[num_class_for_each_attribute_[num_attributes_]];
  // store the probability of each choice
  for (int a = 0; a < num_test_instances_; a++) {
    for (int m = 0; m < num_class_for_each_attribute_[num_attributes_]; m++)
      decision[m] = 1;
    // set the array's entries as 1 for each testing instance

    getline(testInputFile, Buf);
    std::stringstream lineStream(Buf);

    for (int u = 0; u < num_attributes_; u++) {
      getline(lineStream, Buf, ',');
      oneLine[u] = stod(Buf);
    }
    // read one instance for prediction

    getline(lineStream, Buf, ',');
    truth[a] = stod(Buf);
    // store the truth
    for (int x = 0; x < num_class_for_each_attribute_[num_attributes_]; x++) {
      for (int j = 0; j < num_attributes_; j++) {
        if (is_discrete_[j] == 1)  // if this attribute is is_discrete_
        {
          decision[x] *= probabilityTable
              [(j * num_class_for_each_attribute_[num_attributes_]) + x]
              [static_cast<int>(oneLine[j]) - 1];
        } else if (is_discrete_[j] == 0)
        // if this attribute is continuous , then use the Gaussian distribution
        // formular to calculate it's contribution of probability
        {
          long double a0 = -pow(
              (oneLine[j] -
               probabilityTable
                   [(j * num_class_for_each_attribute_[num_attributes_]) + x]
                   [0]),
              2);
          long double a1 =
              2 *
              pow(probabilityTable
                      [(j * num_class_for_each_attribute_[num_attributes_]) + x]
                      [1],
                  2);
          long double a2 = a0 / a1;
          long double a3 = exp(a2);
          long double a4 =
              (0.39894228 /
               probabilityTable
                   [(j * num_class_for_each_attribute_[num_attributes_]) + x]
                   [1]) *
              a3;
          decision[x] *= a4;
        }
      }
      decision[x] *= num_class_for_each_attributes_[x];
    }
    // decide which choice has the highest probability
    int big = 0;
    long double hug = decision[0];
    for (int v = 1; v < num_class_for_each_attribute_[num_attributes_]; v++) {
      if (decision[v] > hug) {
        big = v;
        hug = decision[v];
      }
    }
    outcome[a] = (big + 1);
  }
  Accuracy(outcome, truth);
  // call function "caauracy" to calculate the accuracy

  // release memory
  delete[] truth;
  delete[] decision;
  delete[] oneLine;
  delete[] outcome;
}

}  // namespace baysian
