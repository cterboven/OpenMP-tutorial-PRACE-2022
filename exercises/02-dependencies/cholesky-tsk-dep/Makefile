BIN_DIR=../bin
PROGRAM=$(BIN_DIR)/cholesky-tsk-dep

MKL_HOME=$(MKLROOT)

CC=gcc
CFLAGS=-O3 -std=gnu99 -I$(MKL_HOME)/include -fopenmp
LDFLAGS=-L$(MKL_HOME)/lib/intel64 -lmkl_sequential -lmkl_core -lmkl_rt -lpthread -lm -fopenmp
#VERBOSE=-DVERBOSE

all: $(PROGRAM) 

$(PROGRAM): cholesky.c 
	$(CC) $(CFLAGS) $(VERBOSE) -o $@ $^ $(LDFLAGS)

$(BIN_DIR): 
	mkdir $@

clean:
	rm -rf $(PROGRAM) *.o

wipe: clean
	rm -rf *.out *.err
