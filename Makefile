CC=g++
#CFLAGS= -std=c++11 -c -g -Wall
#CFLAGS= -c -Ofast -march=native -mavx2 -fslp-vectorize-aggressive -Rpass-analysis=loop-vectorize -Wall
CFLAGS= -c -Ofast -march=native -mavx2 -fslp-vectorize-aggressive -Wall
SOURCE=main.cc bayesian.cc naivebayesian.cc bayesiannetwork.cc
LDFLAGS=
OBJECTS= $(SOURCE:.cc=.o)

EXECUTABLE= bayesian


all:  $(SOURCE) $(EXECUTABLE)


$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cc.o:
	$(CC) $(CFLAGS) $< -o $@


clean: 
	rm -f *.o $(EXECUTABLE)
