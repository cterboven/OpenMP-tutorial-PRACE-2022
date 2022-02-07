BIN_DIR=../bin
PROGRAM=$(BIN_DIR)/sudoku-co

CC=g++
CFLAGS=-c -O3 -fopenmp -DUSE_SEQUENTIAL_CUTOFF
LDFLAGS=-fopenmp

OBJS=SudokuBoard.o sudoku.o

all: $(PROGRAM) 

$(PROGRAM): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

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
