#include "bayesian.h"
#include <iostream>


namespace baysian{

//calculate the accuracy
void bayesian::accuracy(int *outcome , int * result)
{
	double correct=0;// store the number of correct predictions

	for( int i=0 ; i<testinstances; i++)//count the number of correct predictions 
	{
		if (outcome[i]==result[i])
			correct++;
//		std::cout<<"predict to be "<<outcome[i]<<" is actually "<<result[i]<<std::endl;
	}
	std::cout<<"Total "<<testinstances<<" data have "<<correct<<" correct predictions"<< std::endl;
	double percentage=correct/testinstances; // calculate the accuracy
	std::cout<<"Accuracy is "<<percentage*100<<"%"<<std::endl;
}

}// end of namespace bayesian
