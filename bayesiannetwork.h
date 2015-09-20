#include "bayesian.h"
class bayesiannetwork: public bayesian
{
protected:
	void classifier(long double ***  , int * , double * , int **, char*);
	//calculate the probability of each choice and choose the greatest one as our prediction
public:
	bayesiannetwork(char*, char*, char*); 
	//initialize all the information we need from training data
};





