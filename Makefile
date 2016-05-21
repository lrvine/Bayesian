CC=g++
CFLAGS=-c -g -Wall
SOURCE=main.cc bayesian.cc naivebayesian.cc bayesiannetwork.cc
LDFLAGS=
OBJECTS= $(SOURCE:.cpp=.o)

EXECUTABLE= bayesian


all:  $(SOURCE) $(EXECUTABLE)


$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@


clean: 
	rm -f *.o $(EXECUTABLE)

