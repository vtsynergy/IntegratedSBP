#!/bin/bash
module load OpenMPI/4.1.1-GCC-10.3.0

# vertices=(25 50 75 100)
# communities=(3 4 4 5)

size=200000

for communities in 74 ; do
  for overlap in 1.0 5.0 ; do
    for blocksizevar in 1.0 ; do
      for mindegree in 1 10 ; do  # 0 1 2 3 ; do
        for maxdegree in 0.0005 0.05 ; do
          for exponent in -2.5 -2.9 ; do
            graphprops="test_${mindegree}_${maxdegree}_${overlap}_${exponent}"
            for numasync in 0 1 2 4 8 16 32 1000000 ; do
              sbatch --job-name="$graphprops_$numasync_0" \
                     --export=ALL,overlap=${overlap},blocksizevar=${blocksizevar},size=${size},communities=${communities},graphprops=${graphprops},numasync=${numasync},run=0 \
                     ./difficulty_sparsity.sbatch
            done
          done
        done
      done
    done
  done
done
