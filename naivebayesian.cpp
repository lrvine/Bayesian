#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

#include "naivebayesian.h"

using std::ifstream;
using std::cout;
using std::endl;
using std::cerr;
using std::setw;

//initialize all the information we need from training data
naivebayesian::naivebayesian()
{
	ifstream training("data.txt");

    if(!training){cerr<<"Can't open training data file!"<<endl;system("PAUSE"); exit(1);}
    
	training>>traininstances>>attributes; // read the number of training instances and attributes

	int *discrete = new int[attributes]; 
	//this array store the information about each attribute is continuous or not
	for(int z=0; z<attributes ; z++)     //  read the information about continuous or not
		training>>discrete[z];


	int *numclass= new int[attributes+1];  
	//this array store the number of classes of each attribute
	for(int b=0; b<=attributes; b++)     // read the number of classes
		training>>numclass[b];

	double *count = new double[numclass[attributes]];
	//this array store the total number of each decision's class in training data
	for(int c=0; c<numclass[attributes]; c++)
		count[c]=0;


    //this "protable" store the count of every possible combination 
    //and divide each of them by the total occurences	
	long double** protable = new long double*[(attributes*numclass[attributes])]; 
	for(int j=0; j<attributes; j++)
	{
		if (discrete[j]==1)// if this attribute is discrete
		{
			for(int tt=(j*numclass[attributes]) ; tt<(j*numclass[attributes]+numclass[attributes]); tt++)
				protable[tt]=new long double[numclass[j]];
		}
		else if (discrete[j]==0)//if this attribute is continuous
		{
			for(int ttt=(j*numclass[attributes]);ttt<(j*numclass[attributes]+numclass[attributes]);ttt++)
				protable[ttt]=new long double[2]; 
			//the first one store mean , the second store the standard deviation
		}
	}


    //initialize the protable to be 0
	for(int r=0 ; r<attributes; r++)   
	{
		if(discrete[r]==1)
		{
			for( int g=(r*numclass[attributes]);g< (r*numclass[attributes]+numclass[attributes]) ;g++)
			{
				for(int e=0; e <numclass[r]; e++)
					protable[g][e]=0;
			}
		}
		else if (discrete[r]==0) 
		{
			for( int gg=(r*numclass[attributes]);gg<(r*numclass[attributes]+numclass[attributes]);gg++)
			{
				for(int e=0; e < 2; e++)
					protable[gg][e]=0;
			}
		}
	}

	//use a array to store each instance for further processing
	double *temp = new double[attributes+1];

    //store the information of each instance into protable
	for( int i=1 ; i<=traininstances; i++)
	{

		for (int y=0 ; y<=attributes ; y++)//read one instance for processing
			training>>temp[y];

		count[static_cast<int>(temp[attributes]) -1 ]++;//count the result


		for( int jj=0 ; jj<attributes;jj++)
		{
			if(discrete[jj]==1)// if this attribute is discrete
			{
				protable[jj*numclass[attributes]+static_cast<int>(temp[attributes])-1]
				[static_cast<int>(temp[jj])-1]++;
			}
			else if (discrete[jj]==0)//if this attribute is continuous
			{
				protable[ jj*numclass[attributes]+static_cast<int>(temp[attributes])-1 ]
				[0]+=temp[jj];
				protable[ jj*numclass[attributes]+static_cast<int>(temp[attributes])-1 ]
				[1]+=pow( temp[jj] , 2 ) ;
			}
		}
	}

	delete [] temp;


    //processing the information in the protalbe to get the proabability
	for( int t=0 ; t< attributes ; t++)
	{
		if (discrete[t]==1)// if this attribute is discrete
		{
			for ( int d=0 ; d<numclass[attributes] ; d++)
			{	
				int correction=0;

				for (int o=0 ; o < numclass[t] ; o++)
				//this loop judge weather there is zero occurence of some conjuction
				//if it dose, then do Laplacian correction 
				{
					
					if (protable[(t*numclass[attributes]+d)][o]==0 )
					{
						correction=numclass[t];
						for(int p=0 ; p <numclass[t] ; p++)
						{
							protable[(t*numclass[attributes]+d)][p]++;
						}
						break;
					}
				}

				for ( int w=0 ; w<numclass[t] ; w++)
				//claculate every conjuction's contribution of probability 
				{
					protable[(t*numclass[attributes]+d)][w]/=(count[d]+correction);
				}
			}
		}
		else if (discrete[t]==0)
		//if this attribute is continuous,we assume it's Gaussian distribution
		//claculate the mean and standard deviation of each continuous attribute
		{
			for (int h=0 ; h < numclass[attributes] ; h++)
			{
				long double a0=pow( protable[(t*numclass[attributes]+h)][0] , 2 ) / count[h];
				long double a1=protable[(t*numclass[attributes]+h)][1]-a0;
				long double a2=a1/count[h];
				long double a3=sqrt(a2);
				protable[(t*numclass[attributes]+h)][1]=a3;
				
				protable[(t*numclass[attributes]+h)][0]/=count[h];
			}
		}
	}


	//calculate the probability of each resulting class
	for ( int ppp=0 ; ppp<numclass[attributes] ; ppp++)
		count[ppp]=count[ppp]/traininstances;

	classifier(protable , numclass ,  count , discrete);
	//call function for classification

	//release the memory
	for( int x=0; x<(attributes*numclass[attributes]) ; x++)
		delete [] protable[x];

	delete [] protable;
	delete [] discrete;
	delete [] numclass;
	delete [] count;


}


//calculate the probability of each choice and choose the greatest one as our prediction
void naivebayesian::classifier(long double** protable,int*numclass ,double* count ,int *discrete)
{
	ifstream testing("test.txt");

    if(!testing){cerr<<"Can't open training data file!"<<endl; system("PAUSE");exit(1);}

	testing>>testinstances;              //read the number of testing data

	int *result= new int[testinstances]; //this array store the real result for comparison
	for(int w=0; w<testinstances; w++)
	{
		result[w]=0;
	}

	int *outcome=new int[testinstances]; //this array store our prediciton
	for(int f=0; f<testinstances; f++)
	{
		outcome[f]=0;
	}

	double *temp=new double [attributes+1]; //store each instance for processing

	long double *decision=new long double[numclass[attributes]]; 
	// store the probability of each choice

	for( int a=0 ; a<testinstances ; a++)
	{
		for ( int m=0 ; m<numclass[attributes]; m++)
		//set the array's entries as 1 for each testing instance
		decision[m]=1;

		for (int u=0 ; u<=attributes; u++)
		// read one instance for prediction
			testing>>temp[u];

		result[a]=temp[attributes];
		// store the result

		for( int x=0 ; x<numclass[attributes] ; x++)
		{

			for( int j=0 ; j<attributes ; j++)
			{
				if(discrete[j]==1)// if this attribute is discrete
				{
					decision[x] *= protable[(j*numclass[attributes])+x][static_cast<int>(temp[j])-1];
				}
				else if (discrete[j]==0)
				// if this attribute is continuous , then use the Gaussian distribution formular
		        // to calculate it's contribution of probability 
				{
					long double a0=-pow( ( temp[j] - protable[(j*numclass[attributes])+x][0] ) , 2 ); 
					long double a1=2 * pow( protable[(j*numclass[attributes])+x][1] , 2 );
					long double a2=a0/a1;
					long double a3=exp(a2);
					long double a4 =( 0.39894228/protable[(j*numclass[attributes])+x][1] )*a3;
					decision[x] *= a4; 
				}
		
			}
			decision[x]*=count[x];
		}

		
		//decide which choice has the highest probability
		int big=0;                                       
	    long double hug=decision[0];
		for ( int v=1 ; v<numclass[attributes] ; v++)
		{
			if ( decision[v]>hug)
			{
				big=v;
				hug=decision[v];
			}
		}
		outcome[a]=(big+1);

	}

	accuracy ( outcome , result );
	//call function "caauracy" to calculate the accuracy 

	//release memory
	delete [] result;
	delete [] decision;
	delete [] temp;
	delete [] outcome;
}


//calculate the accuracy
void naivebayesian::accuracy(int *outcome , int * result)
{


	double correct=0;// store the number of correct predictions

	for( int i=0 ; i<testinstances; i++)//count the number of correct predictions 
	{
		if (outcome[i]==result[i])
			correct++;

		cout<<"預測"<<outcome[i]<<"   實際為"<<result[i]<<endl;
	}
	
	cout<<"總共 "<<testinstances<<" 資料中 有"<<correct<<" 筆資料正確辨識"<< endl;

	double percentage=correct/testinstances; // calculate the accuracy

	cout<<"準確率為"<<percentage*100<<"%"<<endl;
}