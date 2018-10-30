#include "naivebayesian.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace baysian {

// initialize all the information we need from training data
naiveBayesian::naiveBayesian(char *cfg_file) {
  std::cout << "Run naiveBayesian" << std::endl;
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
  for (int z = 0; z < attributes;
       z++)  //  read the information about continuous or not
    configure >> discrete[z];
  classNum = new int[attributes + 1];
  // this array store the number of classes of each attribute
  for (int b = 0; b <= attributes; b++) {  // read the number of classes
    configure >> classNum[b];
    if (discrete[b])  // set classNum as 2 for continuous data
      classNum[b] = 2;
  }

  outputClassNum = classNum[attributes];
  classCount = new double[outputClassNum];
  // this array store the total number of each decision's class in training data
  for (int c = 0; c < outputClassNum; c++) classCount[c] = 0;

  configure.close();
}

void naiveBayesian::train(char *train_file) {
  std::ifstream trainingDataFile;
  std::string Buf;
  trainingDataFile.open(train_file);
  if (!trainingDataFile) {
    std::cout << "Can't open training data file!" << std::endl;
    return;
  }

  // this "probabilityTable" store the count of every possible combination
  // and divide each of them by the total occurences
  probabilityTable = new long double *[(attributes * outputClassNum)];
  for (int j = 0; j < attributes; j++) {
    if (discrete[j] == 1)  // if this attribute is discrete
    {
      for (int i = (j * classNum[attributes]);
           i < (j * classNum[attributes] + classNum[attributes]); i++)
        probabilityTable[i] = new long double[classNum[j]];
    } else if (discrete[j] == 0)  // if this attribute is continuous
    {
      for (int i = (j * classNum[attributes]);
           i < (j * classNum[attributes] + classNum[attributes]); i++)
        probabilityTable[i] = new long double[2];
      // the first one store mean , the second store the standard deviation
    }
  }

  // initialize the probabilityTable to be 0
  for (int r = 0; r < attributes; r++) {
    if (discrete[r] == 1) {
      for (int g = (r * classNum[attributes]);
           g < (r * classNum[attributes] + classNum[attributes]); g++) {
        for (int e = 0; e < classNum[r]; e++) probabilityTable[g][e] = 0;
      }
    } else if (discrete[r] == 0) {
      for (int g = (r * classNum[attributes]);
           g < (r * classNum[attributes] + classNum[attributes]); g++) {
        for (int e = 0; e < 2; e++) probabilityTable[g][e] = 0;
      }
    }
  }

  // use a array to store each instance for further processing
  double *oneLine = new double[attributes + 1];

  // store the information of each instance into probabilityTable
  for (int i = 1; i <= trainInstances; i++) {
    getline(trainingDataFile, Buf);
    std::stringstream lineStream(Buf);

    for (int y = 0; y <= attributes; y++) {  // read one instance for processing
      getline(lineStream, Buf, ',');
      oneLine[y] = stod(Buf);
    }

    classCount[static_cast<int>(oneLine[attributes]) -
               1]++;  // count the result

    for (int j = 0; j < attributes; j++) {
      if (discrete[j] == 1)  // if this attribute is discrete
      {
        probabilityTable[j * classNum[attributes] +
                         static_cast<int>(oneLine[attributes]) -
                         1][static_cast<int>(oneLine[j]) - 1]++;
      } else if (discrete[j] == 0)  // if this attribute is continuous
      {
        probabilityTable[j * classNum[attributes] +
                         static_cast<int>(oneLine[attributes]) - 1][0] +=
            oneLine[j];
        probabilityTable[j * classNum[attributes] +
                         static_cast<int>(oneLine[attributes]) - 1][1] +=
            pow(oneLine[j], 2);
      }
    }
  }

  delete[] oneLine;
  trainingDataFile.close();
  // processing the information in the protalbe to get the proabability
  for (int t = 0; t < attributes; t++) {
    if (discrete[t] == 1)  // if this attribute is discrete
    {
      for (int d = 0; d < classNum[attributes]; d++) {
        int correction = 0;
        for (int o = 0; o < classNum[t]; o++)
        // this loop judge weather there is zero occurence of some conjuction
        // if it dose, then do Laplacian correction
        {
          if (probabilityTable[(t * classNum[attributes] + d)][o] == 0) {
            correction = classNum[t];
            for (int p = 0; p < classNum[t]; p++) {
              probabilityTable[(t * classNum[attributes] + d)][p]++;
            }
            break;
          }
        }
        for (int w = 0; w < classNum[t]; w++)
        // claculate every conjuction's contribution of probability
        {
          probabilityTable[(t * classNum[attributes] + d)][w] /=
              (classCount[d] + correction);
        }
      }
    } else if (discrete[t] == 0)
    // if this attribute is continuous,we assume it's Gaussian distribution
    // claculate the mean and standard deviation of each continuous attribute
    {
      for (int h = 0; h < classNum[attributes]; h++) {
        long double a0 =
            pow(probabilityTable[(t * classNum[attributes] + h)][0], 2) /
            classCount[h];
        long double a1 =
            probabilityTable[(t * classNum[attributes] + h)][1] - a0;
        long double a2 = a1 / classCount[h];
        long double a3 = sqrt(a2);
        probabilityTable[(t * classNum[attributes] + h)][1] = a3;
        probabilityTable[(t * classNum[attributes] + h)][0] /= classCount[h];
      }
    }
  }
  // calculate the probability of each resulting class
  for (int probIndex = 0; probIndex < outputClassNum; probIndex++)
    classCount[probIndex] = classCount[probIndex] / trainInstances;
}

naiveBayesian::~naiveBayesian() {
  // release the memory
#ifdef DEBUG
  std::cout << " release memory " << std::endl;
#endif
  for (int x = 0; x < (attributes * outputClassNum); x++)
    delete[] probabilityTable[x];
  delete[] probabilityTable;
  delete[] discrete;
  delete[] classNum;
  delete[] classCount;
}

// calculate the probability of each choice and choose the greatest one as our
// prediction
void naiveBayesian::predict(char *test_file) {
  std::ifstream testInputFile(test_file);
  if (!testInputFile) {
    std::cout << "Can't open test data file!" << std::endl;
    return;
  }
  std::cout << "Start prediction " << std::endl;
  std::string Buf;

  int *truth = new int[testInstances];  // this array store the real result for
                                        // comparison
  for (int w = 0; w < testInstances; w++) {
    truth[w] = 0;
  }

  int *outcome = new int[testInstances];  // this array store our prediciton
  for (int f = 0; f < testInstances; f++) {
    outcome[f] = 0;
  }

  double *oneLine =
      new double[attributes + 1];  // store each instance for processing
  long double *decision = new long double[classNum[attributes]];
  // store the probability of each choice
  for (int a = 0; a < testInstances; a++) {
    for (int m = 0; m < classNum[attributes]; m++) decision[m] = 1;
    // set the array's entries as 1 for each testing instance

    getline(testInputFile, Buf);
    std::stringstream lineStream(Buf);

    for (int u = 0; u < attributes; u++) {
      getline(lineStream, Buf, ',');
      oneLine[u] = stod(Buf);
    }
    // read one instance for prediction

    getline(lineStream, Buf, ',');
    truth[a] = stod(Buf);
    // store the truth
    for (int x = 0; x < classNum[attributes]; x++) {
      for (int j = 0; j < attributes; j++) {
        if (discrete[j] == 1)  // if this attribute is discrete
        {
          decision[x] *= probabilityTable[(j * classNum[attributes]) + x]
                                         [static_cast<int>(oneLine[j]) - 1];
        } else if (discrete[j] == 0)
        // if this attribute is continuous , then use the Gaussian distribution
        // formular to calculate it's contribution of probability
        {
          long double a0 =
              -pow((oneLine[j] -
                    probabilityTable[(j * classNum[attributes]) + x][0]),
                   2);
          long double a1 =
              2 * pow(probabilityTable[(j * classNum[attributes]) + x][1], 2);
          long double a2 = a0 / a1;
          long double a3 = exp(a2);
          long double a4 =
              (0.39894228 /
               probabilityTable[(j * classNum[attributes]) + x][1]) *
              a3;
          decision[x] *= a4;
        }
      }
      decision[x] *= classCount[x];
    }
    // decide which choice has the highest probability
    int big = 0;
    long double hug = decision[0];
    for (int v = 1; v < classNum[attributes]; v++) {
      if (decision[v] > hug) {
        big = v;
        hug = decision[v];
      }
    }
    outcome[a] = (big + 1);
  }
  accuracy(outcome, truth);
  // call function "caauracy" to calculate the accuracy

  // release memory
  delete[] truth;
  delete[] decision;
  delete[] oneLine;
  delete[] outcome;
}

}  // namespace baysian
