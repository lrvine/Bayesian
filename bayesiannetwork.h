#include "bayesian.h"
namespace baysian{

class bayesiannetwork: public bayesian
{
protected:
	void classifier(long double ***  , int * , double * , int **, char*);
	//calculate the probability of each choice and choose the greatest one as our prediction
public:
	bayesiannetwork(char*, char*, char*); 
	//initialize all the information we need from training data
};

template<class Type>
struct data
{
	double key;
	Type value1;
	Type value2;
};


}// end of namespace bayesian
