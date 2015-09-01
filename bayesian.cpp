#include "bayesian.h"
#include <iostream>

using std::cout;
using std::endl;

//calculate the accuracy
void bayesian::accuracy(int *outcome , int * result)
{


	double correct=0;// store the number of correct predictions

	for( int i=0 ; i<testinstances; i++)//count the number of correct predictions 
	{
		if (outcome[i]==result[i])
			correct++;

//		cout<<"predict to be "<<outcome[i]<<" is actually "<<result[i]<<endl;
	}
	
	cout<<"total "<<testinstances<<" data hve "<<correct<<" correct prediction"<< endl;

	double percentage=correct/testinstances; // calculate the accuracy

	cout<<"accuracy is "<<percentage*100<<"%"<<endl;
}
