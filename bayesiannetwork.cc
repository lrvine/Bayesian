#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#include "bayesiannetwork.h"

namespace baysian {

struct data_compare {
  bool operator()(const data<int> l, const data<int> r) {
    return l.key > r.key;
  }
};

// initialize all the information we need from training data
bayesianNetwork::bayesianNetwork(char *train_file, char *test_file,
                                 char *cfg_file) {
  std::cout << "Run Baysiannetwork" << std::endl;
  std::ifstream configure;
  configure.open(cfg_file);
  if (!configure) {
    std::cout << "! Can't open configuration file!" << std::endl;
    return;
  }

  configure >> trainInstances >> testInstances >>
      attributes;  // read the number of training instances and attributes

  // TODO : implement handling continuous data. This is just a placeholder
  int *discrete = new int[attributes];
  // this array store the information about each attribute is continuous or not
  for (int idx = 0; idx < attributes;
       idx++)  //  read the information about continuous or not
    configure >> discrete[idx];

  int *numclass = new int[attributes + 1];
  // this array store the number of classes of each attribute
  for (int idx = 0; idx <= attributes; idx++)  // read the number of classes
    configure >> numclass[idx];

  double *count = new double[numclass[attributes]];
  // this array store the total number of each decision's class in training data
  for (int c = 0; c < numclass[attributes]; c++) count[c] = 0;

  configure.close();
  int combinations = 1;
  for (int com = (attributes - 1); com > 1; com--) combinations += com;

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
  for (int z = 0; z < (attributes - 1); z++) {
    for (int p = 1; p <= (attributes - z - 1); p++) {
      rank[index][0] = z;
      rank[index][1] = (z + p);
      index++;
    }
  }

  double **ccc = new double *[attributes * numclass[attributes]];
  for (int f = 0; f < attributes; f++) {
    for (int f1 = (f * numclass[attributes]);
         f1 < ((f + 1) * numclass[attributes]); f1++)
      ccc[f1] = new double[numclass[f]];
  }
  for (int f2 = 0; f2 < attributes; f2++) {
    for (int f3 = (f2 * numclass[attributes]);
         f3 < ((f2 + 1) * numclass[attributes]); f3++) {
      for (int f4 = 0; f4 < numclass[f2]; f4++) ccc[f3][f4] = 0;
    }
  }

  double **aaa = new double *[combinations * numclass[attributes]];
  for (int f = 0; f < combinations; f++) {
    for (int f1 = (f * numclass[attributes]);
         f1 < ((f + 1) * numclass[attributes]); f1++) {
      aaa[f1] = new double[numclass[rank[f][0]] * numclass[rank[f][1]]];

      for (int f2 = 0; f2 < numclass[rank[f][0]] * numclass[rank[f][1]]; f2++)
        aaa[f1][f2] = 0;
    }
  }

  double **bbb = new double *[combinations * numclass[attributes]];
  for (int f = 0; f < combinations; f++) {
    for (int f1 = (f * numclass[attributes]);
         f1 < ((f + 1) * numclass[attributes]); f1++) {
      bbb[f1] = new double[numclass[rank[f][0]] * numclass[rank[f][1]]];

      for (int f2 = 0; f2 < numclass[rank[f][0]] * numclass[rank[f][1]]; f2++)
        bbb[f1][f2] = 0;
    }
  }

  int *oneLine = new int[attributes + 1];

  for (int i = 1; i <= trainInstances; i++) {
    getline(trainingDataFile, Buf);
    std::stringstream lineStream(Buf);

    for (int b = 0; b <= attributes; b++) {
      getline(lineStream, Buf, ',');
      oneLine[b] = stod(Buf);
    }

    for (int h = 0; h < attributes; h++)
      ccc[h * numclass[attributes] + oneLine[attributes] - 1][oneLine[h] - 1]++;

    for (int h = 0; h < combinations; h++) {
      aaa[h * numclass[attributes] + oneLine[attributes] - 1]
         [(oneLine[rank[h][0]] - 1) * numclass[rank[h][1]] +
          oneLine[rank[h][1]] - 1]++;

      bbb[h * numclass[attributes] + oneLine[attributes] - 1]
         [(oneLine[rank[h][0]] - 1) * numclass[rank[h][1]] +
          oneLine[rank[h][1]] - 1]++;
    }

    count[oneLine[attributes] - 1]++;
  }

  delete[] oneLine;
  trainingDataFile.close();

  for (int t = 0; t < attributes; t++) {
    for (int d = 0; d < numclass[attributes]; d++) {
      int correction = 0;

      for (int o = 0; o < numclass[t]; o++) {
        if (ccc[(t * numclass[attributes] + d)][o] == 0) {
          correction = numclass[t];
          for (int p = 0; p < numclass[t]; p++) {
            ccc[(t * numclass[attributes] + d)][p]++;
          }
          break;
        }
      }

      for (int w = 0; w < numclass[t]; w++)
        ccc[(t * numclass[attributes] + d)][w] /= (count[d] + correction);
    }
  }

  for (int i = 0; i < combinations; i++) {
    int correction1 = 0;

    for (int d = 0; d < numclass[attributes]; d++) {
      for (int o = 0; o < numclass[rank[i][0]] * numclass[rank[i][1]]; o++) {
        if (aaa[i * numclass[attributes] + d][o] == 0) {
          for (int p = 0; p < numclass[attributes]; p++) {
            for (int q = 0; q < numclass[rank[i][0]] * numclass[rank[i][1]];
                 q++) {
              aaa[i * numclass[attributes] + p][q]++;
            }
          }

          correction1 = numclass[attributes] * numclass[rank[i][0]] *
                        numclass[rank[i][1]];

          break;
        }
      }
    }

    for (int w1 = 0; w1 < numclass[attributes]; w1++) {
      for (int w2 = 0; w2 < numclass[rank[i][0]] * numclass[rank[i][1]]; w2++) {
        aaa[i * numclass[attributes] + w1][w2] /=
            (trainInstances + correction1);
      }
    }
  }

  for (int t = 0; t < combinations; t++) {
    for (int d = 0; d < numclass[attributes]; d++) {
      int correction2 = 0;

      for (int o = 0; o < numclass[rank[t][0]] * numclass[rank[t][1]]; o++) {
        if (bbb[t * numclass[attributes] + d][o] == 0) {
          for (int p = 0; p < numclass[rank[t][0]] * numclass[rank[t][1]];
               p++) {
            bbb[t * numclass[attributes] + d][p]++;
          }

          correction2 = numclass[rank[t][0]] * numclass[rank[t][1]];

          break;
        }
      }

      for (int w2 = 0; w2 < numclass[rank[t][0]] * numclass[rank[t][1]]; w2++)
        bbb[t * numclass[attributes] + d][w2] /= (count[d] + correction2);
    }
  }

  double *relation = new double[combinations];

  for (int s0 = 0; s0 < combinations; s0++) {
    double tempo = 0;

    for (int s1 = 0; s1 < numclass[attributes]; s1++) {
      for (int s2 = 0; s2 < numclass[rank[s0][0]]; s2++) {
        for (int s3 = 0; s3 < numclass[rank[s0][1]]; s3++) {
          tempo += (aaa[s0 * numclass[attributes] + s1]
                       [s2 * numclass[rank[s0][1]] + s3] *
                    log10(bbb[s0 * numclass[attributes] + s1]
                             [s2 * numclass[rank[s0][1]] + s3] /
                          (ccc[rank[s0][0] * numclass[attributes] + s1][s2] *
                           ccc[rank[s0][1] * numclass[attributes] + s1][s3])));
        }
      }
    }
    relation[s0] = tempo;
  }

  std::priority_queue<data<int>, std::vector<data<int> >, data_compare>
      maxweight;
  data<int> elen;

  for (int cast = 0; cast < combinations; cast++) {
    elen.value1 = rank[cast][0];
    elen.value2 = rank[cast][1];
    elen.key = relation[cast];
    maxweight.push(elen);
  }

  int *groups = new int[attributes];
  for (int v = 0; v < attributes; v++) groups[v] = 0;

  int **graph = new int *[attributes];
  for (int zz1 = 0; zz1 < attributes; zz1++) graph[zz1] = new int[attributes];

  for (int k1 = 0; k1 < attributes; k1++) {
    for (int kk1 = 0; kk1 < attributes; kk1++) graph[k1][kk1] = 0;
  }

  data<int> mmm;
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

      for (int scan = 0; scan < attributes; scan++) {
        if (groups[scan] == slave) {
          groups[scan] = boss;
        }
      }

      graph[mmm.value1][mmm.value2] = 1;
      graph[mmm.value2][mmm.value1] = 1;
    }
  }

#ifdef DEBUG
  for (int atest = 0; atest < attributes; atest++) {
    for (int atest1 = 0; atest1 < attributes; atest1++)
      std::cout << graph[atest][atest1] << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif

  int *transfer = new int[attributes];
  for (int v = 0; v < attributes; v++) transfer[v] = attributes;

  transfer[0] = 0;

  for (int redo = 0; redo < attributes; redo++) {
    int min = (attributes + 1);
    int point = 0;

    for (int redo1 = 0; redo1 < attributes; redo1++) {
      if (min > transfer[redo1]) {
        min = transfer[redo1];
        point = redo1;
      }
    }

    for (int redo2 = 0; redo2 < attributes; redo2++) {
      if (graph[point][redo2] == 1) {
        graph[redo2][point] = 0;
        transfer[redo2] = (min + 1);
      }
    }

    transfer[point] = (attributes + 1);
  }

#ifdef DEBUG
  std::cout << std::endl;
  for (int test = 0; test < attributes; test++) {
    for (int test1 = 0; test1 < attributes; test1++)
      std::cout << graph[test][test1] << " ";

    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif

  //------------------------------------------------

  int **parent = new int *[attributes];
  // this 2-dimension array store each node's parents
  for (int z = 0; z < attributes; z++) parent[z] = new int[attributes + 1];

  for (int k = 0; k < attributes; k++) {
    for (int kk = 0; kk <= attributes; kk++) parent[k][kk] = 0;
  }

  // read the information about everyone's parents
  for (int e = 0; e < attributes; e++) {
    int pama = 1;
    int pamaindex = 1;

    for (int ee = 0; ee < attributes; ee++) {
      if (graph[ee][e] == 1) {
        pama++;
        parent[e][pamaindex] = ee;
        pamaindex++;
      }
    }

    parent[e][0] = pama;
    parent[e][pamaindex] = attributes;
  }
  //-------------------------------------------------

  // cpt is a three dimention array
  // the first dimention is the attributes
  // the last two dimention is the "conditional probability table"
  // for each attribute
  long double ***cpt = new long double **[attributes];
  for (int j = 0; j < attributes; j++) {
    cpt[j] = new long double *[numclass[j]];

    // calculate the appropriate length of the third dimention
    int reg = 1;
    for (int jjj = 1; jjj <= parent[j][0]; jjj++)
      reg *= numclass[parent[j][jjj]];

    for (int jj = 0; jj < numclass[j]; jj++) {
      cpt[j][jj] = new long double[reg + 1];

      cpt[j][jj][0] = reg;

      for (int jjjj = 1; jjjj <= reg; jjjj++)  // initialize to zero
        cpt[j][jj][jjjj] = 0;
    }
  }

  trainingDataFile.open(train_file);
  if (!trainingDataFile) {
    std::cout << "Can't open training data file!" << std::endl;
    return;
  }

  double *oneLine_double = new double[attributes + 1];

  // store the counts of each possible conjunction into cpt
  for (int i = 1; i <= trainInstances; i++) {
    getline(trainingDataFile, Buf);
    std::stringstream lineStream(Buf);

    for (int y = 0; y <= attributes; y++) {
      getline(lineStream, Buf, ',');
      oneLine_double[y] = stod(Buf);
    }

    for (int yy = 0; yy < attributes; yy++) {
      int reg_add = 1;
      int reg_mul = 1;

      for (int yyy = 1; yyy <= parent[yy][0]; yyy++) {
        reg_add +=
            (reg_mul * (static_cast<int>(oneLine_double[parent[yy][yyy]]) - 1));
        reg_mul *= numclass[parent[yy][yyy]];
      }

      cpt[yy][(static_cast<int>(oneLine_double[yy]) - 1)][reg_add]++;
    }
  }

  delete[] oneLine_double;
  trainingDataFile.close();

  // processing the information in the protalbe to get the proabability of each
  // conjunction
  for (int t1 = 0; t1 < attributes; t1++) {
    for (int d = 1; d <= cpt[t1][0][0]; d++) {
      for (int o = 0; o < numclass[t1]; o++) {
        // this loop judge weather there is zero occurence of some conjuction
        // if it dose, then do Laplacian correction
        if (cpt[t1][o][d] == 0) {
          for (int p = 0; p < numclass[t1]; p++) {
            cpt[t1][p][d]++;
          }
          break;
        }
      }

      int sum = 0;

      for (int w = 0; w < numclass[t1]; w++) sum += cpt[t1][w][d];

      // claculate every conjuction's contribution of probability
      for (int ww = 0; ww < numclass[t1]; ww++) cpt[t1][ww][d] /= sum;
    }
  }

  // calculate the probability of each resulting class
  for (int p = 0; p < numclass[attributes]; p++) {
    count[p] = count[p] / trainInstances;
#ifdef DEBUG
    std::cout << count[p] << " ";
#endif
  }

  predict(cpt, numclass, count, parent, test_file);
  // call function for classification

  // release the memory
  for (int x1 = 0; x1 < attributes; x1++) {
    for (int x2 = 0; x2 < numclass[x1]; x2++) delete[] cpt[x1][x2];
    delete[] cpt[x1];
  }

  for (int pa = 0; pa < attributes; pa++) delete[] parent[pa];

  delete[] parent;
  delete[] cpt;
  delete[] numclass;
  delete[] discrete;
  delete[] count;
}

// Will migrate old predict function to this format
void bayesianNetwork::predict(char *test_file) {}

// calculate the probability of each choice and choose the greatest one as our
// prediction
void bayesianNetwork::predict(long double ***cpt, int *numclass, double *count,
                              int **parent, char *test_file) {
  std::ifstream testInputFile(test_file);
  if (!testInputFile) {
    std::cout << "Can't open test data file!" << std::endl;
    return;
  }
  std::string Buf;

  int *result = new int[testInstances];  // this array store the real result for
                                         // comparison
  for (int w = 0; w < testInstances; w++) {
    result[w] = 0;
  }

  int *outcome = new int[testInstances];  // this array store our prediciton
  for (int f = 0; f < testInstances; f++) {
    outcome[f] = 0;
  }

  double *oneLine =
      new double[attributes + 1];  // store each instance for processing

  long double *decision = new long double[numclass[attributes]];
  // store the probability of each choice

  for (int a = 0; a < testInstances; a++) {
    getline(testInputFile, Buf);
    std::stringstream lineStream(Buf);
    // set the array's entries as 1 for each testing instance
    for (int m = 0; m < numclass[attributes]; m++) decision[m] = 1;

    // read one instance for prediction
    for (int u = 0; u <= attributes; u++) {
      getline(lineStream, Buf, ',');
      oneLine[u] = stod(Buf);
    }

    result[a] = oneLine[attributes];
    // store the result

    // calculate each choice's probability
    for (int x1 = 0; x1 < numclass[attributes]; x1++) {
      for (int x2 = 0; x2 < attributes; x2++) {
        int reg_add = 1;  // objective's position of the third dimention array
        int reg_mul = 1;  // for calculating reg_add

        // the address of our objective is depend on this attribute's parent
        for (int x3 = 1; x3 < parent[x2][0]; x3++) {
          reg_add +=
              (reg_mul * (static_cast<int>(oneLine[parent[x2][x3]]) - 1));
          reg_mul *= numclass[parent[x2][x3]];
        }
        reg_add += (reg_mul * x1);

        decision[x1] *= cpt[x2][static_cast<int>(oneLine[x2]) - 1][reg_add];
      }
      decision[x1] *= count[x1];
    }

    // decide which choice has the highest probability
    int big = 0;
    long double hug = decision[0];
    for (int v = 1; v < numclass[attributes]; v++) {
      if (decision[v] > hug) {
        big = v;
        hug = decision[v];
      }
    }
    outcome[a] = (big + 1);
  }
  accuracy(outcome, result);
  // call function "accuracy" to calculate the accuracy

  // release memory
  delete[] result;
  delete[] decision;
  delete[] oneLine;
  delete[] outcome;
}

}  // namespace baysian
