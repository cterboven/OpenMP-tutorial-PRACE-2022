#!/bin/bash
#SBATCH --job-name=treesearch
#SBATCH --workdir=.
#SBATCH --output=treesearch_%j.out
#SBATCH --error=treesearch_%j.err
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --qos=debug

export IFS=";"

THREADS="01;02;04;06;08;10;12;14;16;18;20;22;24;26;28;30;32;34;36;38;40;42;44;46;48"
#THREADS="01;02;04;06;08;10;12"

export OMP_CANCELLATION=true 

for threads in $THREADS; do
  OMP_NUM_THREADS=$threads ./treesearch-cnl
done
