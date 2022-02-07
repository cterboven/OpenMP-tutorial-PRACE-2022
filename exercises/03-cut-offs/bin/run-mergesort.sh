#!/bin/bash
#SBATCH --job-name=mergesort
#SBATCH --workdir=.
#SBATCH --output=mergesort_%j.out
#SBATCH --error=mergesort_%j.err
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --qos=debug

export IFS=";"

THREADS="01;02;04;06;08;10;12;14;16;18;20;22;24;26;28;30;32;34;36;38;40;42;44;46;48"
#THREADS="01;02;04;06;08;10;12"

ASIZES="1000000"

for AS in $ASIZES; do
  for threads in $THREADS; do
    echo Running mergesort-co with $threads
    OMP_NUM_THREADS=$threads ./mergesort-co $AS 
    echo -- 
  done
done
