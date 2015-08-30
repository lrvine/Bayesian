#include <iostream>
using std::cout;
using std::endl;

#include <iomanip>
using std::setw;

#include <fstream>
using std::cerr;

#include <cstdlib>

#include "maxheap.h"

template<class Type>
maxheap<Type>::maxheap(int sz )
{
	maxsize = sz; n=0 ;
	heap=new data<Type>[maxsize+1];
}

/*
template<class Type>
void maxheap<Type>::printcontent(int number)
{
	cout<<"key value" <<setw(12)<<"content"<<endl;
    for(int i=1; i<=number; i++)
	{
	cout<<setw(6)<<heap[i].key<<setw(15)<<heap[i].value<<endl;
	}
}
*/

template<class Type>
void maxheap<Type>::insert(const data<Type> x)
{
	if (n==maxsize){heapfull();return;}
    	n++;
	int i;
	for(i=n;;){
		if (i==1)break;
		if (x.key <= heap[i/2].key) break;
		heap[i]=heap[i/2];
		i/=2;
	}
	heap[i]=x;
}

template<class Type>
void maxheap<Type>::deletemax(data<Type>& x)
{
	if(!n){heapempty();return;}
	x=heap[1]; data <Type>k=heap[n]; n--;
	int i;
	int j;
	for(i=1,j=2;j<=n;)
	{
		if(j<n)if(heap[j].key<heap[j+1].key)j++;
		if (k.key>=heap[j].key)break;
		heap[i]=heap[j];
		i=j;j*=2;
	}
	heap[i]=k;
//	return &x;
}

template<class Type>
bool maxheap<Type>::isfull()
{
  if (n==maxsize) return true;
  else return false;
}

template<class Type>
bool maxheap<Type>::isempty()
{
  if (n==0) return true;
  else return false;
}

template<class Type>
void maxheap<Type>::heapfull()
{
  cerr<<"heap is already full !!"<<endl; exit(1);
}

template<class Type>
void maxheap<Type>::heapempty()
{
  cerr<<"heap is empty !!"<<endl;exit(1);
}

