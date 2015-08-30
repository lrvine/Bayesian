#include <iostream>
#include <stdlib.h>
using std::cout;
using std::endl;

#include "naivebayesian.h"
#include "bayesiannetwork.h"


int main( int argc, char** argv ){

int method=0;
char* train;
char* input;

if( argc >= 4 ){
	method = atoi(argv[3]);
	train = argv[1];
	input = argv[2];
}else if( argc == 3 ){
	train = argv[1];
	input = argv[2];
	cout<<" use default NaiveBayesian method"<<endl;
}else {
	cout<<" You need to provide training data and input data for prediction. Please read README"<<endl;
}

if( method == 1 ){
  bayesiannetwork b(train, input);
}else{
  naivebayesian a(train, input);
}
return 0;
}


