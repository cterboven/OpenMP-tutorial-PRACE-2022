BIN_DIR=../bin
PROGRAM=$(BIN_DIR)/quicksort

CC=g++
CFLAGS=-c -O3 -fopenmp 
LDFLAGS=-fopenmp

all: $(PROGRAM) 

$(PROGRAM): quicksort.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -o $@ $^ 

sudoku.cpp:
	@cat README.md 
	@false

$(BIN_DIR): 
	mkdir $@

clean:
	rm -rf $(PROGRAM) *.o

wipe: clean
	rm -rf *.out *.err
