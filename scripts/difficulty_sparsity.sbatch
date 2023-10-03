#!/bin/bash
#SBATCH -p normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:00
#SBATCH --account=wuchun_feng
#SBATCH --mem=MaxMemPerNode

module load OpenMPI/4.1.1-GCC-10.3.0

# get args
algorithm="async_gibbs"

EXE=/home/wanyef/PartiallyAsyncSBP/build/DistributedSBP
DIR="/projects/synergy_lab/wanyef/difficulty_graphs/"

/usr/bin/time -v $EXE --threads 32 -n $size -d $DIR --batches 1 -a $algorithm --transpose -o $overlap \
              -b $blocksizevar --type $graphprops --tag "${graphprops}_${numasync}_${run}" \
              --asynciterations ${numasync}

if [ $run -lt 9 ] ; then
    run=$(( $run + 1 ))
    sbatch --job-name="${graphprops}_${numasync}_${run}" \
           --export=ALL,overlap=${overlap},blocksizevar=${blocksizevar},size=${size},communities=${communities},graphprops=${graphprops},numasync=${numasync},run=${run} \
           ./difficulty_sparsity.sbatch
fi

