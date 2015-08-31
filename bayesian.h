class bayesian
{
protected:
	int traininstances;  //store the number of training instances
	int testinstances;   //store the number of testing instances
	int attributes;      //store the number of attributes
	virtual	void classifier(long double ***  , int * , double * , int **, char*);
	//calculate the probability of each choice and choose the greatest one as our prediction
	void accuracy(int * , int * ); // claculate the accuracy


};
