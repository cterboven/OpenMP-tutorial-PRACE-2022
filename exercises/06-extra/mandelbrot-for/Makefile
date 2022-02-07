BIN_DIR=../bin
PROGRAM=$(BIN_DIR)/mandelbrot

CC=gcc
CFLAGS=-O3 -std=gnu99 -fopenmp
LDFLAGS=-lm -fopenmp
#VERBOSE=-DVERBOSE

all: $(PROGRAM) 

$(PROGRAM): mandelbrot.c lodepng.o
	$(CC) $(CFLAGS) $(VERBOSE) -o $@ $^ $(LDFLAGS)

lodepng.o:lib/lodepng.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(BIN_DIR): 
	mkdir $@

clean:
	rm -rf $(PROGRAM) *.o

wipe: clean
	rm -rf *.out *.err
