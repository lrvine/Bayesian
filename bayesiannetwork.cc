#include "bayesiannetwork.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

namespace machinelearning {
namespace baysian {

struct KeyCompare {
  bool operator()(const KeyAndTwoValue<int> l, const KeyAndTwoValue<int> r) {
    return l.key > r.key;
  }
};

// initialize all the information we need from training data
BayesianNetwork::BayesianNetwork(char *cfg_file) {
  std::cout << "Init Baysiannetwork" << std::endl;
  ParseConfiguration(cfg_file);
}

void BayesianNetwork::Train(char *train_file) {
  int combinations = 1;
  for (int com = (num_attributes_ - 1); com > 1; --com) combinations += com;

#ifdef DEBUG
  std::cout << "combinatinos " << combinations << std::endl << std::endl;
#endif

  std::ifstream trainingDataFile;
  std::string Buf;
  trainingDataFile.open(train_file);
  if (!trainingDataFile) {
    std::cout << "Can't open training data file!" << std::endl;
    return;
  }

  std::vector<std::vector<int> > rank(combinations);
  for (int idx = 0; idx < combinations; ++idx) rank[idx].resize(2);
  int index = 0;
  for (int i = 0; i < (num_attributes_ - 1); ++i) {
    for (int j = 1; j <= (num_attributes_ - i - 1); ++j) {
      rank[index][0] = i;
      rank[index][1] = (i + j);
      index++;
    }
  }

  std::vector<std::vector<double> > array_type_one(num_attributes_ *
                                                   num_output_class_);
  for (int i = 0; i < num_attributes_; ++i) {
    for (int j = (i * num_output_class_); j < ((i + 1) * num_output_class_);
         ++j)
      array_type_one[j].resize(num_class_for_each_attribute_[i], 0);
  }

  std::vector<std::vector<double> > array_type_two(combinations *
                                                   num_output_class_);
  for (int i = 0; i < combinations; ++i) {
    for (int j = (i * num_output_class_); j < ((i + 1) * num_output_class_);
         ++j) {
      array_type_two[j].resize(num_class_for_each_attribute_[rank[i][0]] *
                                   num_class_for_each_attribute_[rank[i][1]],
                               0);
    }
  }

  std::vector<std::vector<double> > array_type_three(combinations *
                                                     num_output_class_);
  for (int i = 0; i < combinations; ++i) {
    for (int j = (i * num_output_class_); j < ((i + 1) * num_output_class_);
         ++j) {
      array_type_three[j].resize(
          num_class_for_each_attribute_[rank[i][0]] *
              num_class_for_each_attribute_[rank[i][1]],
          0);
    }
  }

  std::vector<int> oneLine((num_attributes_ + 1), 0);

  for (int i = 1; i <= num_train_instances_; ++i) {
    getline(trainingDataFile, Buf);
    std::stringstream lineStream(Buf);

    for (int j = 0; j <= num_attributes_; ++j) {
      getline(lineStream, Buf, ',');
      oneLine[j] = stod(Buf);
    }

    for (int h = 0; h < num_attributes_; ++h)
      array_type_one[h * num_output_class_ + oneLine[num_attributes_] - 1]
                    [oneLine[h] - 1]++;

    for (int h = 0; h < combinations; ++h) {
      array_type_two[h * num_output_class_ + oneLine[num_attributes_] - 1]
                    [(oneLine[rank[h][0]] - 1) *
                         num_class_for_each_attribute_[rank[h][1]] +
                     oneLine[rank[h][1]] - 1]++;

      array_type_three[h * num_output_class_ + oneLine[num_attributes_] - 1]
                      [(oneLine[rank[h][0]] - 1) *
                           num_class_for_each_attribute_[rank[h][1]] +
                       oneLine[rank[h][1]] - 1]++;
    }

    output_class_cnt_[oneLine[num_attributes_] - 1]++;
  }

  trainingDataFile.close();

  for (int t = 0; t < num_attributes_; ++t) {
    for (int d = 0; d < num_output_class_; ++d) {
      int correction = 0;

      for (int e = 0; e < num_class_for_each_attribute_[t]; ++e) {
        if (array_type_one[(t * num_output_class_ + d)][e] == 0) {
          correction = num_class_for_each_attribute_[t];
          for (int p = 0; p < num_class_for_each_attribute_[t]; ++p) {
            array_type_one[(t * num_output_class_ + d)][p]++;
          }
          break;
        }
      }

      for (int w = 0; w < num_class_for_each_attribute_[t]; ++w)
        array_type_one[(t * num_output_class_ + d)][w] /=
            (output_class_cnt_[d] + correction);
    }
  }

  for (int i = 0; i < combinations; ++i) {
    int correction1 = 0;

    for (int d = 0; d < num_output_class_; ++d) {
      for (int e = 0; e < num_class_for_each_attribute_[rank[i][0]] *
                              num_class_for_each_attribute_[rank[i][1]];
           ++e) {
        if (array_type_two[i * num_output_class_ + d][e] == 0) {
          for (int p = 0; p < num_output_class_; ++p) {
            for (int q = 0; q < num_class_for_each_attribute_[rank[i][0]] *
                                    num_class_for_each_attribute_[rank[i][1]];
                 ++q) {
              array_type_two[i * num_output_class_ + p][q]++;
            }
          }

          correction1 = num_output_class_ *
                        num_class_for_each_attribute_[rank[i][0]] *
                        num_class_for_each_attribute_[rank[i][1]];

          break;
        }
      }
    }

    for (int j = 0; j < num_output_class_; ++j) {
      for (int k = 0; k < num_class_for_each_attribute_[rank[i][0]] *
                                num_class_for_each_attribute_[rank[i][1]];
           ++k) {
        array_type_two[i * num_output_class_ + j][k] /=
            (num_train_instances_ + correction1);
      }
    }
  }

  for (int t = 0; t < combinations; ++t) {
    for (int d = 0; d < num_output_class_; ++d) {
      int correction2 = 0;

      for (int o = 0; o < num_class_for_each_attribute_[rank[t][0]] *
                              num_class_for_each_attribute_[rank[t][1]];
           ++o) {
        if (array_type_three[t * num_output_class_ + d][o] == 0) {
          for (int p = 0; p < num_class_for_each_attribute_[rank[t][0]] *
                                  num_class_for_each_attribute_[rank[t][1]];
               ++p) {
            array_type_three[t * num_output_class_ + d][p]++;
          }

          correction2 = num_class_for_each_attribute_[rank[t][0]] *
                        num_class_for_each_attribute_[rank[t][1]];

          break;
        }
      }

      for (int j = 0; j < num_class_for_each_attribute_[rank[t][0]] *
                                num_class_for_each_attribute_[rank[t][1]];
           ++j)
        array_type_three[t * num_output_class_ + d][j] /=
            (output_class_cnt_[d] + correction2);
    }
  }

  std::vector<double> relation(combinations, 0);

  for (int s0 = 0; s0 < combinations; ++s0) {
    double temp = 0;

    for (int s1 = 0; s1 < num_output_class_; ++s1) {
      for (int s2 = 0; s2 < num_class_for_each_attribute_[rank[s0][0]]; ++s2) {
        for (int s3 = 0; s3 < num_class_for_each_attribute_[rank[s0][1]];
             ++s3) {
          temp +=
              (array_type_two[s0 * num_output_class_ + s1]
                             [s2 * num_class_for_each_attribute_[rank[s0][1]] +
                              s3] *
               log10(
                   array_type_three
                       [s0 * num_output_class_ + s1]
                       [s2 * num_class_for_each_attribute_[rank[s0][1]] + s3] /
                   (array_type_one[rank[s0][0] * num_output_class_ + s1][s2] *
                    array_type_one[rank[s0][1] * num_output_class_ + s1][s3])));
        }
      }
    }
    relation[s0] = temp;
  }

  std::priority_queue<KeyAndTwoValue<int>, std::vector<KeyAndTwoValue<int> >,
                      KeyCompare>
      maxweight;
  KeyAndTwoValue<int> elen;

  for (int cast = 0; cast < combinations; ++cast) {
    elen.value1 = rank[cast][0];
    elen.value2 = rank[cast][1];
    elen.key = relation[cast];
    maxweight.push(elen);
  }

  std::vector<int> groups(num_attributes_, 0);

  std::vector<std::vector<int> > graph(num_attributes_);
  for (int i = 0; i < num_attributes_; i++) graph[i].resize(num_attributes_, 0);

  KeyAndTwoValue<int> one_case;
  int base = 1;

  for (int i = 0; i < combinations; ++i) {
    if (!maxweight.empty()) {
      one_case = maxweight.top();
      maxweight.pop();
    }

    if (groups[one_case.value1] != 0 && groups[one_case.value2] != 0 &&
        groups[one_case.value1] == groups[one_case.value2]) {
    } else if (groups[one_case.value1] == 0 &&
               groups[one_case.value1] == groups[one_case.value2]) {
      groups[one_case.value1] = base;
      groups[one_case.value2] = base;
      base++;

      graph[one_case.value1][one_case.value2] = 1;
      graph[one_case.value2][one_case.value1] = 1;
    } else if (groups[one_case.value1] == 0 && groups[one_case.value2] != 0) {
      groups[one_case.value1] = groups[one_case.value2];

      graph[one_case.value1][one_case.value2] = 1;
      graph[one_case.value2][one_case.value1] = 1;
    } else if (groups[one_case.value1] != 0 && groups[one_case.value2] == 0) {
      groups[one_case.value2] = groups[one_case.value1];

      graph[one_case.value1][one_case.value2] = 1;
      graph[one_case.value2][one_case.value1] = 1;
    } else if (groups[one_case.value1] != 0 && groups[one_case.value2] != 0 &&
               groups[one_case.value1] != groups[one_case.value2]) {
      int boss, slave;

      if (groups[one_case.value1] < groups[one_case.value2]) {
        boss = groups[one_case.value1];
        slave = groups[one_case.value2];
      } else {
        boss = groups[one_case.value2];
        slave = groups[one_case.value1];
      }

      for (int scan = 0; scan < num_attributes_; scan++) {
        if (groups[scan] == slave) {
          groups[scan] = boss;
        }
      }

      graph[one_case.value1][one_case.value2] = 1;
      graph[one_case.value2][one_case.value1] = 1;
    }
  }

#ifdef DEBUG
  for (int i = 0; i < num_attributes_; ++i) {
    for (int j = 0; j < num_attributes_; ++j)
      std::cout << graph[i][j] << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif

  std::vector<int> transfer(num_attributes_);
  for (int i = 0; i < num_attributes_; ++i) transfer[i] = num_attributes_;

  transfer[0] = 0;

  for (int i = 0; i < num_attributes_; ++i) {
    int min = (num_attributes_ + 1);
    int point = 0;

    for (int j = 0; j < num_attributes_; ++j) {
      if (min > transfer[j]) {
        min = transfer[j];
        point = j;
      }
    }

    for (int j = 0; j < num_attributes_; ++j) {
      if (graph[point][j] == 1) {
        graph[j][point] = 0;
        transfer[j] = (min + 1);
      }
    }

    transfer[point] = (num_attributes_ + 1);
  }

#ifdef DEBUG
  std::cout << std::endl;
  for (int i = 0; i < num_attributes_; ++i) {
    for (int j = 0; j < num_attributes_; ++j)
      std::cout << graph[i][j] << " ";

    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif

  //------------------------------------------------

  nodes_parents_.resize(num_attributes_);
  // this 2-dimension array store each node's parents
  for (int i = 0; i < num_attributes_; ++i)
    nodes_parents_[i].resize(num_attributes_ + 1, 0);

  // read the information about everyone's parents
  for (int i = 0; i < num_attributes_; ++i) {
    int pama = 1;
    int pamaindex = 1;

    for (int j = 0; j < num_attributes_; ++j) {
      if (graph[j][i] == 1) {
        pama++;
        nodes_parents_[i][pamaindex] = j;
        pamaindex++;
      }
    }

    nodes_parents_[i][0] = pama;
    nodes_parents_[i][pamaindex] = num_attributes_;
  }
  //-------------------------------------------------

  // conditional_probability_table_ is a three dimention array
  // the first dimention is the num_attributes_
  // the last two dimention is the "conditional probability table"
  // for each attribute
  conditional_probability_table_.resize(num_attributes_);
  for (int i = 0; i < num_attributes_; ++i) {
    conditional_probability_table_[i].resize(
        num_class_for_each_attribute_[i]);

    // calculate the appropriate length of the third dimention
    int reg = 1;
    for (int j = 1; j <= nodes_parents_[i][0]; ++j)
      reg *= num_class_for_each_attribute_[nodes_parents_[i][j]];

    for (int j = 0; j < num_class_for_each_attribute_[i]; ++j) {
      conditional_probability_table_[i][j].resize(reg + 1);

      conditional_probability_table_[i][j][0] = reg;

      for (int k = 1; k <= reg; ++k)  // initialize to zero
        conditional_probability_table_[i][j][k] = 0;
    }
  }

  trainingDataFile.open(train_file);
  if (!trainingDataFile) {
    std::cout << "Can't open training data file!" << std::endl;
    return;
  }

  std::vector<double> oneLine_double(num_attributes_ + 1);

  // store the counts of each possible conjunction into
  // conditional_probability_table_
  for (int i = 1; i <= num_train_instances_; ++i) {
    getline(trainingDataFile, Buf);
    std::stringstream lineStream(Buf);

    for (int j = 0; j <= num_attributes_; ++j) {
      getline(lineStream, Buf, ',');
      oneLine_double[j] = stod(Buf);
    }

    for (int j = 0; j < num_attributes_; ++j) {
      int reg_add = 1;
      int reg_mul = 1;

      for (int k = 1; k <= nodes_parents_[j][0]; ++k) {
        reg_add +=
            (reg_mul *
             (static_cast<int>(oneLine_double[nodes_parents_[j][k]]) - 1));
        reg_mul *= num_class_for_each_attribute_[nodes_parents_[j][k]];
      }

      conditional_probability_table_[j][(static_cast<int>(oneLine_double[j]) -
                                         1)][reg_add]++;
    }
  }

  trainingDataFile.close();

  // processing the information in the protalbe to get the proabability of each
  // conjunction
  for (int i = 0; i < num_attributes_; ++i) {
    for (int j = 1; j <= conditional_probability_table_[i][0][0]; ++j) {
      for (int k = 0; k < num_class_for_each_attribute_[i]; ++k) {
        // this loop judge weather there is zero occurence of some conjuction
        // if it dose, then do Laplacian correction
        if (conditional_probability_table_[i][k][j] == 0) {
          for (int p = 0; p < num_class_for_each_attribute_[i]; ++p) {
            conditional_probability_table_[i][p][j]++;
          }
          break;
        }
      }

      int sum = 0;

      for (int w = 0; w < num_class_for_each_attribute_[i]; ++w)
        sum += conditional_probability_table_[i][w][j];

      // claculate every conjuction's contribution of probability
      for (int w = 0; w < num_class_for_each_attribute_[i]; ++w)
        conditional_probability_table_[i][w][j] /= sum;
    }
  }

  // calculate the probability of each resulting class
  for (int i = 0; i < num_output_class_; ++i) {
    output_class_cnt_[i] = output_class_cnt_[i] / num_train_instances_;
#ifdef DEBUG
    std::cout << output_class_cnt_[i] << " ";
#endif
  }
}

// calculate the probability of each choice and choose the greatest one as our
// prediction
std::vector<int> BayesianNetwork::Predict(char *test_file, bool has_truth) {
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
  std::string Buf;

  for (int i = 0; i < num_test_instances_; ++i) {
    getline(testInputFile, Buf);
    std::stringstream lineStream(Buf);
    // set the array's entries as 1 for each testing instance
    for (int j = 0; j < num_output_class_; ++j) decision[j] = 1;

    // read one instance for prediction
    for (int j = 0; j < num_attributes_; ++j) {
      getline(lineStream, Buf, ',');
      oneLine[j] = stod(Buf);
    }
    if (has_truth) {
      getline(lineStream, Buf, ',');
      truth[i] = stod(Buf);
      // store the truth
    }

    // calculate each choice's probability
    for (int x1 = 0; x1 < num_output_class_; ++x1) {
      for (int x2 = 0; x2 < num_attributes_; ++x2) {
        int reg_add = 1;  // objective's position of the third dimention array
        int reg_mul = 1;  // for calculating reg_add

        // the address of our objective is depend on this attribute's parent
        for (int x3 = 1; x3 < nodes_parents_[x2][0]; ++x3) {
          reg_add += (reg_mul *
                      (static_cast<int>(oneLine[nodes_parents_[x2][x3]]) - 1));
          reg_mul *= num_class_for_each_attribute_[nodes_parents_[x2][x3]];
        }
        reg_add += (reg_mul * x1);

        decision[x1] *=
            conditional_probability_table_[x2][static_cast<int>(oneLine[x2]) -
                                               1][reg_add];
      }
      decision[x1] *= output_class_cnt_[x1];
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
  Accuracy(outcome, truth);
  // call function "accuracy" to calculate the accuracy
  return outcome;
}

}  // namespace baysian
}  // namespace machinelearning
