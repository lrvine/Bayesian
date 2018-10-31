#include "bayesiannetwork.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

namespace baysian {

struct data_compare {
  bool operator()(const KeyAndTwoValue<int> l, const KeyAndTwoValue<int> r) {
    return l.key > r.key;
  }
};

// initialize all the information we need from training data
BayesianNetwork::BayesianNetwork(char *cfg_file) {
  std::cout << "Run Baysiannetwork" << std::endl;
  ParseConfiguration(cfg_file);
}

BayesianNetwork::~BayesianNetwork() {
  // release the memory
#ifdef DEBUG
  std::cout << " release memory " << std::endl;
#endif

  for (int x1 = 0; x1 < num_attributes_; x1++) {
    for (int x2 = 0; x2 < num_class_for_each_attribute_[x1]; x2++)
      delete[] conditional_probability_table_[x1][x2];
    delete[] conditional_probability_table_[x1];
  }

  for (int pa = 0; pa < num_attributes_; pa++) delete[] nodes_parents_[pa];

  delete[] nodes_parents_;
  delete[] conditional_probability_table_;
  delete[] num_class_for_each_attribute_;
  delete[] is_discrete_;
  delete[] num_class_for_each_attributes_;
}

void BayesianNetwork::Train(char *train_file) {
  int combinations = 1;
  for (int com = (num_attributes_ - 1); com > 1; com--) combinations += com;

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

  int **rank = new int *[combinations];
  for (int idx = 0; idx < combinations; idx++) rank[idx] = new int[2];
  int index = 0;
  for (int z = 0; z < (num_attributes_ - 1); z++) {
    for (int p = 1; p <= (num_attributes_ - z - 1); p++) {
      rank[index][0] = z;
      rank[index][1] = (z + p);
      index++;
    }
  }

  double **ccc = new double *[num_attributes_ * num_output_class_];
  for (int f = 0; f < num_attributes_; f++) {
    for (int f1 = (f * num_class_for_each_attribute_[num_attributes_]);
         f1 < ((f + 1) * num_class_for_each_attribute_[num_attributes_]); f1++)
      ccc[f1] = new double[num_class_for_each_attribute_[f]];
  }
  for (int f2 = 0; f2 < num_attributes_; f2++) {
    for (int f3 = (f2 * num_class_for_each_attribute_[num_attributes_]);
         f3 < ((f2 + 1) * num_class_for_each_attribute_[num_attributes_]);
         f3++) {
      for (int f4 = 0; f4 < num_class_for_each_attribute_[f2]; f4++)
        ccc[f3][f4] = 0;
    }
  }

  double **aaa = new double *[combinations * num_output_class_];
  for (int f = 0; f < combinations; f++) {
    for (int f1 = (f * num_output_class_); f1 < ((f + 1) * num_output_class_);
         f1++) {
      aaa[f1] = new double[num_class_for_each_attribute_[rank[f][0]] *
                           num_class_for_each_attribute_[rank[f][1]]];

      for (int f2 = 0; f2 < num_class_for_each_attribute_[rank[f][0]] *
                                num_class_for_each_attribute_[rank[f][1]];
           f2++)
        aaa[f1][f2] = 0;
    }
  }

  double **bbb = new double *[combinations * num_output_class_];
  for (int f = 0; f < combinations; f++) {
    for (int f1 = (f * num_output_class_); f1 < ((f + 1) * num_output_class_);
         f1++) {
      bbb[f1] = new double[num_class_for_each_attribute_[rank[f][0]] *
                           num_class_for_each_attribute_[rank[f][1]]];

      for (int f2 = 0; f2 < num_class_for_each_attribute_[rank[f][0]] *
                                num_class_for_each_attribute_[rank[f][1]];
           f2++)
        bbb[f1][f2] = 0;
    }
  }

  int *oneLine = new int[num_attributes_ + 1];

  for (int i = 1; i <= num_train_instances_; i++) {
    getline(trainingDataFile, Buf);
    std::stringstream lineStream(Buf);

    for (int b = 0; b <= num_attributes_; b++) {
      getline(lineStream, Buf, ',');
      oneLine[b] = stod(Buf);
    }

    for (int h = 0; h < num_attributes_; h++)
      ccc[h * num_class_for_each_attribute_[num_attributes_] +
          oneLine[num_attributes_] - 1][oneLine[h] - 1]++;

    for (int h = 0; h < combinations; h++) {
      aaa[h * num_output_class_ + oneLine[num_attributes_] - 1]
         [(oneLine[rank[h][0]] - 1) *
              num_class_for_each_attribute_[rank[h][1]] +
          oneLine[rank[h][1]] - 1]++;

      bbb[h * num_output_class_ + oneLine[num_attributes_] - 1]
         [(oneLine[rank[h][0]] - 1) *
              num_class_for_each_attribute_[rank[h][1]] +
          oneLine[rank[h][1]] - 1]++;
    }

    num_class_for_each_attributes_[oneLine[num_attributes_] - 1]++;
  }

  delete[] oneLine;
  trainingDataFile.close();

  for (int t = 0; t < num_attributes_; t++) {
    for (int d = 0; d < num_class_for_each_attribute_[num_attributes_]; d++) {
      int correction = 0;

      for (int o = 0; o < num_class_for_each_attribute_[t]; o++) {
        if (ccc[(t * num_class_for_each_attribute_[num_attributes_] + d)][o] ==
            0) {
          correction = num_class_for_each_attribute_[t];
          for (int p = 0; p < num_class_for_each_attribute_[t]; p++) {
            ccc[(t * num_class_for_each_attribute_[num_attributes_] + d)][p]++;
          }
          break;
        }
      }

      for (int w = 0; w < num_class_for_each_attribute_[t]; w++)
        ccc[(t * num_class_for_each_attribute_[num_attributes_] + d)][w] /=
            (num_class_for_each_attributes_[d] + correction);
    }
  }

  for (int i = 0; i < combinations; i++) {
    int correction1 = 0;

    for (int d = 0; d < num_output_class_; d++) {
      for (int o = 0; o < num_class_for_each_attribute_[rank[i][0]] *
                              num_class_for_each_attribute_[rank[i][1]];
           o++) {
        if (aaa[i * num_output_class_ + d][o] == 0) {
          for (int p = 0; p < num_output_class_; p++) {
            for (int q = 0; q < num_class_for_each_attribute_[rank[i][0]] *
                                    num_class_for_each_attribute_[rank[i][1]];
                 q++) {
              aaa[i * num_output_class_ + p][q]++;
            }
          }

          correction1 = num_output_class_ *
                        num_class_for_each_attribute_[rank[i][0]] *
                        num_class_for_each_attribute_[rank[i][1]];

          break;
        }
      }
    }

    for (int w1 = 0; w1 < num_output_class_; w1++) {
      for (int w2 = 0; w2 < num_class_for_each_attribute_[rank[i][0]] *
                                num_class_for_each_attribute_[rank[i][1]];
           w2++) {
        aaa[i * num_output_class_ + w1][w2] /=
            (num_train_instances_ + correction1);
      }
    }
  }

  for (int t = 0; t < combinations; t++) {
    for (int d = 0; d < num_output_class_; d++) {
      int correction2 = 0;

      for (int o = 0; o < num_class_for_each_attribute_[rank[t][0]] *
                              num_class_for_each_attribute_[rank[t][1]];
           o++) {
        if (bbb[t * num_output_class_ + d][o] == 0) {
          for (int p = 0; p < num_class_for_each_attribute_[rank[t][0]] *
                                  num_class_for_each_attribute_[rank[t][1]];
               p++) {
            bbb[t * num_output_class_ + d][p]++;
          }

          correction2 = num_class_for_each_attribute_[rank[t][0]] *
                        num_class_for_each_attribute_[rank[t][1]];

          break;
        }
      }

      for (int w2 = 0; w2 < num_class_for_each_attribute_[rank[t][0]] *
                                num_class_for_each_attribute_[rank[t][1]];
           w2++)
        bbb[t * num_output_class_ + d][w2] /=
            (num_class_for_each_attributes_[d] + correction2);
    }
  }

  double *relation = new double[combinations];

  for (int s0 = 0; s0 < combinations; s0++) {
    double tempo = 0;

    for (int s1 = 0; s1 < num_output_class_; s1++) {
      for (int s2 = 0; s2 < num_class_for_each_attribute_[rank[s0][0]]; s2++) {
        for (int s3 = 0; s3 < num_class_for_each_attribute_[rank[s0][1]];
             s3++) {
          tempo +=
              (aaa[s0 * num_output_class_ + s1]
                  [s2 * num_class_for_each_attribute_[rank[s0][1]] + s3] *
               log10(bbb[s0 * num_output_class_ + s1]
                        [s2 * num_class_for_each_attribute_[rank[s0][1]] + s3] /
                     (ccc[rank[s0][0] * num_output_class_ + s1][s2] *
                      ccc[rank[s0][1] * num_output_class_ + s1][s3])));
        }
      }
    }
    relation[s0] = tempo;
  }

  std::priority_queue<KeyAndTwoValue<int>, std::vector<KeyAndTwoValue<int> >,
                      data_compare>
      maxweight;
  KeyAndTwoValue<int> elen;

  for (int cast = 0; cast < combinations; cast++) {
    elen.value1 = rank[cast][0];
    elen.value2 = rank[cast][1];
    elen.key = relation[cast];
    maxweight.push(elen);
  }

  int *groups = new int[num_attributes_];
  for (int v = 0; v < num_attributes_; v++) groups[v] = 0;

  int **graph = new int *[num_attributes_];
  for (int zz1 = 0; zz1 < num_attributes_; zz1++)
    graph[zz1] = new int[num_attributes_];

  for (int k1 = 0; k1 < num_attributes_; k1++) {
    for (int kk1 = 0; kk1 < num_attributes_; kk1++) graph[k1][kk1] = 0;
  }

  KeyAndTwoValue<int> mmm;
  int base = 1;

  for (int combi = 0; combi < combinations; combi++) {
    if (!maxweight.empty()) {
      mmm = maxweight.top();
      maxweight.pop();
    }

    if (groups[mmm.value1] != 0 && groups[mmm.value2] != 0 &&
        groups[mmm.value1] == groups[mmm.value2]) {
    } else if (groups[mmm.value1] == 0 &&
               groups[mmm.value1] == groups[mmm.value2]) {
      groups[mmm.value1] = base;
      groups[mmm.value2] = base;
      base++;

      graph[mmm.value1][mmm.value2] = 1;
      graph[mmm.value2][mmm.value1] = 1;
    } else if (groups[mmm.value1] == 0 && groups[mmm.value2] != 0) {
      groups[mmm.value1] = groups[mmm.value2];

      graph[mmm.value1][mmm.value2] = 1;
      graph[mmm.value2][mmm.value1] = 1;
    } else if (groups[mmm.value1] != 0 && groups[mmm.value2] == 0) {
      groups[mmm.value2] = groups[mmm.value1];

      graph[mmm.value1][mmm.value2] = 1;
      graph[mmm.value2][mmm.value1] = 1;
    } else if (groups[mmm.value1] != 0 && groups[mmm.value2] != 0 &&
               groups[mmm.value1] != groups[mmm.value2]) {
      int boss, slave;

      if (groups[mmm.value1] < groups[mmm.value2]) {
        boss = groups[mmm.value1];
        slave = groups[mmm.value2];
      } else {
        boss = groups[mmm.value2];
        slave = groups[mmm.value1];
      }

      for (int scan = 0; scan < num_attributes_; scan++) {
        if (groups[scan] == slave) {
          groups[scan] = boss;
        }
      }

      graph[mmm.value1][mmm.value2] = 1;
      graph[mmm.value2][mmm.value1] = 1;
    }
  }

#ifdef DEBUG
  for (int atest = 0; atest < num_attributes_; atest++) {
    for (int atest1 = 0; atest1 < num_attributes_; atest1++)
      std::cout << graph[atest][atest1] << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif

  int *transfer = new int[num_attributes_];
  for (int v = 0; v < num_attributes_; v++) transfer[v] = num_attributes_;

  transfer[0] = 0;

  for (int redo = 0; redo < num_attributes_; redo++) {
    int min = (num_attributes_ + 1);
    int point = 0;

    for (int redo1 = 0; redo1 < num_attributes_; redo1++) {
      if (min > transfer[redo1]) {
        min = transfer[redo1];
        point = redo1;
      }
    }

    for (int redo2 = 0; redo2 < num_attributes_; redo2++) {
      if (graph[point][redo2] == 1) {
        graph[redo2][point] = 0;
        transfer[redo2] = (min + 1);
      }
    }

    transfer[point] = (num_attributes_ + 1);
  }

#ifdef DEBUG
  std::cout << std::endl;
  for (int test = 0; test < num_attributes_; test++) {
    for (int test1 = 0; test1 < num_attributes_; test1++)
      std::cout << graph[test][test1] << " ";

    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif

  //------------------------------------------------

  nodes_parents_ = new int *[num_attributes_];
  // this 2-dimension array store each node's parents
  for (int z = 0; z < num_attributes_; z++)
    nodes_parents_[z] = new int[num_attributes_ + 1];

  for (int k = 0; k < num_attributes_; k++) {
    for (int kk = 0; kk <= num_attributes_; kk++) nodes_parents_[k][kk] = 0;
  }

  // read the information about everyone's parents
  for (int e = 0; e < num_attributes_; e++) {
    int pama = 1;
    int pamaindex = 1;

    for (int ee = 0; ee < num_attributes_; ee++) {
      if (graph[ee][e] == 1) {
        pama++;
        nodes_parents_[e][pamaindex] = ee;
        pamaindex++;
      }
    }

    nodes_parents_[e][0] = pama;
    nodes_parents_[e][pamaindex] = num_attributes_;
  }
  //-------------------------------------------------

  // conditional_probability_table_ is a three dimention array
  // the first dimention is the num_attributes_
  // the last two dimention is the "conditional probability table"
  // for each attribute
  conditional_probability_table_ = new long double **[num_attributes_];
  for (int j1 = 0; j1 < num_attributes_; j1++) {
    conditional_probability_table_[j1] =
        new long double *[num_class_for_each_attribute_[j1]];

    // calculate the appropriate length of the third dimention
    int reg = 1;
    for (int j2 = 1; j2 <= nodes_parents_[j1][0]; j2++)
      reg *= num_class_for_each_attribute_[nodes_parents_[j1][j2]];

    for (int j3 = 0; j3 < num_class_for_each_attribute_[j1]; j3++) {
      conditional_probability_table_[j1][j3] = new long double[reg + 1];

      conditional_probability_table_[j1][j3][0] = reg;

      for (int j4 = 1; j4 <= reg; j4++)  // initialize to zero
        conditional_probability_table_[j1][j3][j4] = 0;
    }
  }

  trainingDataFile.open(train_file);
  if (!trainingDataFile) {
    std::cout << "Can't open training data file!" << std::endl;
    return;
  }

  double *oneLine_double = new double[num_attributes_ + 1];

  // store the counts of each possible conjunction into
  // conditional_probability_table_
  for (int i = 1; i <= num_train_instances_; i++) {
    getline(trainingDataFile, Buf);
    std::stringstream lineStream(Buf);

    for (int y = 0; y <= num_attributes_; y++) {
      getline(lineStream, Buf, ',');
      oneLine_double[y] = stod(Buf);
    }

    for (int yy = 0; yy < num_attributes_; yy++) {
      int reg_add = 1;
      int reg_mul = 1;

      for (int yyy = 1; yyy <= nodes_parents_[yy][0]; yyy++) {
        reg_add +=
            (reg_mul *
             (static_cast<int>(oneLine_double[nodes_parents_[yy][yyy]]) - 1));
        reg_mul *= num_class_for_each_attribute_[nodes_parents_[yy][yyy]];
      }

      conditional_probability_table_[yy][(static_cast<int>(oneLine_double[yy]) -
                                          1)][reg_add]++;
    }
  }

  delete[] oneLine_double;
  trainingDataFile.close();

  // processing the information in the protalbe to get the proabability of each
  // conjunction
  for (int t1 = 0; t1 < num_attributes_; t1++) {
    for (int d = 1; d <= conditional_probability_table_[t1][0][0]; d++) {
      for (int o = 0; o < num_class_for_each_attribute_[t1]; o++) {
        // this loop judge weather there is zero occurence of some conjuction
        // if it dose, then do Laplacian correction
        if (conditional_probability_table_[t1][o][d] == 0) {
          for (int p = 0; p < num_class_for_each_attribute_[t1]; p++) {
            conditional_probability_table_[t1][p][d]++;
          }
          break;
        }
      }

      int sum = 0;

      for (int w = 0; w < num_class_for_each_attribute_[t1]; w++)
        sum += conditional_probability_table_[t1][w][d];

      // claculate every conjuction's contribution of probability
      for (int ww = 0; ww < num_class_for_each_attribute_[t1]; ww++)
        conditional_probability_table_[t1][ww][d] /= sum;
    }
  }

  // calculate the probability of each resulting class
  for (int p = 0; p < num_output_class_; p++) {
    num_class_for_each_attributes_[p] =
        num_class_for_each_attributes_[p] / num_train_instances_;
#ifdef DEBUG
    std::cout << num_class_for_each_attributes_[p] << " ";
#endif
  }
}

// calculate the probability of each choice and choose the greatest one as our
// prediction
void BayesianNetwork::Predict(char *test_file) {
  std::ifstream testInputFile(test_file);
  if (!testInputFile) {
    std::cout << "Can't open test data file!" << std::endl;
    return;
  }
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

  long double *decision = new long double[num_output_class_];
  // store the probability of each choice

  for (int a = 0; a < num_test_instances_; a++) {
    getline(testInputFile, Buf);
    std::stringstream lineStream(Buf);
    // set the array's entries as 1 for each testing instance
    for (int m = 0; m < num_output_class_; m++) decision[m] = 1;

    // read one instance for prediction
    for (int u = 0; u <= num_attributes_; u++) {
      getline(lineStream, Buf, ',');
      oneLine[u] = stod(Buf);
    }

    truth[a] = oneLine[num_attributes_];
    // store the truth

    // calculate each choice's probability
    for (int x1 = 0; x1 < num_output_class_; x1++) {
      for (int x2 = 0; x2 < num_attributes_; x2++) {
        int reg_add = 1;  // objective's position of the third dimention array
        int reg_mul = 1;  // for calculating reg_add

        // the address of our objective is depend on this attribute's parent
        for (int x3 = 1; x3 < nodes_parents_[x2][0]; x3++) {
          reg_add += (reg_mul *
                      (static_cast<int>(oneLine[nodes_parents_[x2][x3]]) - 1));
          reg_mul *= num_class_for_each_attribute_[nodes_parents_[x2][x3]];
        }
        reg_add += (reg_mul * x1);

        decision[x1] *=
            conditional_probability_table_[x2][static_cast<int>(oneLine[x2]) -
                                               1][reg_add];
      }
      decision[x1] *= num_class_for_each_attributes_[x1];
    }

    // decide which choice has the highest probability
    int big = 0;
    long double hug = decision[0];
    for (int v = 1; v < num_output_class_; v++) {
      if (decision[v] > hug) {
        big = v;
        hug = decision[v];
      }
    }
    outcome[a] = (big + 1);
  }
  Accuracy(outcome, truth);
  // call function "accuracy" to calculate the accuracy

  // release memory
  delete[] truth;
  delete[] decision;
  delete[] oneLine;
  delete[] outcome;
}

}  // namespace baysian
