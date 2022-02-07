#!/bin/bash
#SBATCH --job-name=sudoku
#SBATCH --workdir=.
#SBATCH --output=sudoku_%j.out
#SBATCH --error=sudoku_%j.err
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --qos=debug

export IFS=";"

THREADS="01;02;04;06;08;10;12;14;16;18;20;22;24;26;28;30;32;34;36;38;40;42;44;46;48"
#THREADS="01;02;04;06;08;10;12"

for threads in $THREADS; do
  #OMP_NUM_THREADS=$threads ./sudoku-co 9 3 sudoku-9x9-1.txt 
  OMP_NUM_THREADS=$threads ./sudoku-co 16 4 sudoku-16x16-1.txt 
  #OMP_NUM_THREADS=$threads ./sudoku-co 25 5 sudoku-25x25-1.txt 
done
