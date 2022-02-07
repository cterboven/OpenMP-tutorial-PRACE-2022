BIN_DIR=../bin
PROGRAM=$(BIN_DIR)/jacobi

CC=gcc
CFLAGS=-O3 -std=gnu99 -fopenmp
LDFLAGS=-lm -fopenmp
#VERBOSE=-DVERBOSE

all: $(PROGRAM) 

$(PROGRAM): jacobi.c 
	$(CC) $(CFLAGS) $(VERBOSE) -o $@ $^ $(LDFLAGS)

$(BIN_DIR): 
	mkdir $@

clean:
	rm -rf $(PROGRAM) *.o

wipe: clean
	rm -rf *.out *.err
