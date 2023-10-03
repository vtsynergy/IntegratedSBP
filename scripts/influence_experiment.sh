#!/bin/bash

module load OpenMPI/4.1.1-GCC-10.3.0

vertices=(25 50 75 100)
communities=(3 4 4 5)

for overlap in 1.0 2.0 3.0 4.0 5.0 ; do
  for blocksizevar in 1.0 2.0 3.0 4.0 5.0 ; do
    for size in 0 1 ; do  # 0 1 2 3 ; do
      for algorithm in "async_gibbs" "metropolis_hastings" ; do
        sbatch --job-name="influence_exp_${vertices[$size]}_${overlap}_${blocksizevar}" \
            --export=ALL,alg=${algorithm},overlap=${overlap},blocksizevar=${blocksizevar},vertices=${vertices[$size]},communities=${communities[$size]},run=0 \
            ./influence.sbatch
      done
    done
  done
done
