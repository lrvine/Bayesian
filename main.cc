#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "bayesiannetwork.h"
#include "naivebayesian.h"

using namespace std;

int main(int argc, char **argv) {
  int method = 0;
  char *train;
  char *input;
  char *cfg;
  clock_t begin;
  clock_t end;
  double time_spent;

  if (argc >= 5) {
    method = atoi(argv[4]);
    train = argv[1];
    input = argv[2];
    cfg = argv[3];
  } else if (argc == 4) {
    train = argv[1];
    input = argv[2];
    cfg = argv[3];
    std::cout << " use default naiveBayesian method" << std::endl;
  } else {
    std::cout << " You need to provide training data, test data, and "
                 "configuration for prediction. Please read README"
              << std::endl;
  }
  begin = clock();
  if (method == 0) {
    baysian::naiveBayesian naive(cfg);
    naive.train(train);
    naive.predict(input);  // call function to do prediction
  } else if (method == 1) {
    baysian::bayesianNetwork bnetwork(train, input, cfg);
  }
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  std::cout << "Time spent " << time_spent << " seconds " << std::endl;
  return 0;
}
