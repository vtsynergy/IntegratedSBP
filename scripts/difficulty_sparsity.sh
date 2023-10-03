#!/bin/bash

vertices=(1000 5000 20000 50000)

subdir="difficulty_graphs"

args="--threads 32 -d ../../SBMGraphGenerator/${subdir} --transpose"

for i in {0..9} ; do
  size=200000
  outdir="${subdir}/${size}"
  mkdir -p "difficulty_results/$outdir"
  for communities in 74 ; do
    for overlap in 1.0 5.0 ; do
      for blocksizevar in 1.0 ; do
        for mindegree in 1 10 ; do
          for maxdegree in 0.0005 0.05 ; do
            for exponent in -2.5 -2.9 ; do
              graphprops="test_${mindegree}_${maxdegree}_${overlap}_${exponent}"
              echo "========= $graphprops =============="
              for algorithm in "async_gibbs" ; do # "metropolis_hastings" ; do
                for numasync in 0 1 2 4 8 16 32 1000000 ; do
                  /usr/bin/time -v ../build/DistributedSBP -o $overlap -b $blocksizevar -n $size -t $graphprops \
                                -a $algorithm $args --tag "${graphprops}_${numasync}" --asynciterations ${numasync} \
                                &>> "difficulty_results/${outdir}/${graphprops}_${size}.out"
                done
              done
            done
          done
        done
      done
    done
  done
done
