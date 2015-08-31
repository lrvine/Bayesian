CC=g++
CFLAGS=-c -g -Wall
SOURCE=main.cpp bayesian.cpp naivebayesian.cpp bayesiannetwork.cpp maxheap.cpp
LDFLAGS=
OBJECTS= $(SOURCE:.cpp=.o)

EXECUTABLE= bayesian


all:  $(SOURCE) $(EXECUTABLE)


$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
