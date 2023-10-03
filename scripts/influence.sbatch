#!/bin/bash
#SBATCH -p normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --account=wuchun_feng
#SBATCH --mem=MaxMemPerNode

module load OpenMPI/4.1.1-GCC-10.3.0

# get args
algorithm=$alg
overlap=$overlap
blocksizevar=$blocksizevar
vertices=$vertices
communities=$communities
run=$run

EXE=/home/wanyef/SBPonGPU/build/Influence
DIR="/projects/synergy_lab/wanyef/influence/"
TYPE="test_${communities}_0.25_-2.1_1.0"

/usr/bin/time -v $EXE --threads 32 -n ${vertices} -d ${DIR} --batches 1 -a ${algorithm} --transpose -o ${overlap} -b ${blocksizevar} --type ${TYPE}

if [ $run -lt 10 ] ; then
  run=$(( $run + 1 ))
  sbatch --job-name="influence_exp_${vertices}_${overlap}_${blocksizevar}_${run}" \
      --export=ALL,alg=${algorithm},overlap=${overlap},blocksizevar=${blocksizevar},vertices=${vertices},communities=${communities},run=${run} \
      ./influence.sbatch
fi

