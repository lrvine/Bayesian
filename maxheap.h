#ifndef maxheap_h
#define maxheap_h

#include "maxpq.h"
#define defaultsize 1000

template <class Type>
class maxheap:public maxpq<Type>
{
	public:
		maxheap(int sz=defaultsize);

		bool isfull();
        bool isempty();

		void insert(const data<Type>);
		void deletemax (data<Type>&);

		void heapfull();
		void heapempty();
    	private:
		data<Type> *heap;
		int n;
		int maxsize;
};

#endif
