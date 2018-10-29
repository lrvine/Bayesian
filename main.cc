#include <stdlib.h>
#include <iostream>
#include "bayesiannetwork.h"
#include "naivebayesian.h"

using namespace std;

int main(int argc, char **argv) {
  int method = 0;
  char *train;
  char *input;
  char *cfg;

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

  if (method == 0) {
    baysian::naiveBayesian naive(train, input, cfg);
  } else if (method == 1) {
    baysian::bayesianNetwork bnetwork(train, input, cfg);
  }

  return 0;
}
