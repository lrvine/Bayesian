#ifndef maxpq_h
#define maxpq_h

template<class Type>
struct data
{
	double key;
	Type value1;
	Type value2;
};

template<class Type>
class maxpq
{
	public:
		virtual void insert(const data<Type>) = 0;
		virtual void deletemax (data<Type>&) =0;	
};

#endif
