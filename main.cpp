#include <iostream>
#include <stdlib.h>
#include "naivebayesian.h"
#include "bayesiannetwork.h"

using namespace std;

int main( int argc, char** argv ){

int method=0;
char* train;
char* input;
char* cfg;

if( argc >= 5 ){
	method = atoi(argv[4]);
	train = argv[1];
	input = argv[2];
	cfg = argv[3];
}else if( argc == 4 ){
	train = argv[1];
	input = argv[2];
	cfg = argv[3];
	cout<<" use default NaiveBayesian method"<<endl;
}else {
	cout<<" You need to provide training data, test data, and configuration for prediction. Please read README"<<endl;
}

if( method == 0 ){
  naivebayesian method0(train, input, cfg);
}else if( method == 1 ){
  bayesiannetwork method1(train, input, cfg);
}

return 0;
}


