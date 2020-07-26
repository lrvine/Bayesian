CC=g++
CFLAGS= -std=c++2a -c -Ofast -march=native -Wall
SOURCE=main.cc machinelearning.cc bayesian.cc naivebayesian.cc bayesiannetwork.cc
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
