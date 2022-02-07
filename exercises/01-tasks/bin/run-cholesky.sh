#!/bin/bash
#SBATCH --job-name=cholesky
#SBATCH --workdir=.
#SBATCH --output=cholesky_%j.out
#SBATCH --error=cholesky_%j.err
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --qos=debug

export IFS=";"

THREADS="01;02;04;06;08;10;12;14;16;18;20;22;24;26;28;30;32;34;36;38;40;42;44;46;48"
#THREADS="01;02;04;06;08;10;12"

#MSIZES="2048;4096;8192;16384"
MSIZES="4096"

BSIZES="512"

for MS in $MSIZES; do
  for BS in $BSIZES; do
    for threads in $THREADS; do
      OMP_NUM_THREADS=$threads ./cholesky-tsk $MS $BS 0
    done
  done
done

