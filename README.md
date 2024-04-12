# Accelerated Stochastic Block Partitioning

This project integrates 3 different heuristics for accelerating Stochastic Block Partitioning (SBP) into a unified implementation.

The three heuristics are:
- sampling via the SamBaS framework [[IEEE HPEC Paper @IEEE Xplore](https://doi-org.ezproxy.lib.vt.edu/10.1109/HPEC.2019.8916542), [IEEE HPEC Paper Preprint](http://synergy.cs.vt.edu/pubs/papers/wanye-fast-block-par-hpec19.pdf), [Journal Paper Preprint](https://arxiv.org/abs/2108.06651)]
- shared-memory parallelization via Hybrid SBP [[ICPP Paper @ACM](https://doi.org/10.1145/3545008.3545058), [ICPP Paper Preprint](http://synergy.cs.vt.edu/pubs/papers/wanye-hybrid-sbp-icpp2022.pdf)]
- distributed-memory parallelization via EDiSt [[IEEE CLUSTER Paper Preprint](https://arxiv.org/abs/2305.18663)]

For a full description of our code, please see our IEEE HPEC 2023 Graph Challenge paper: [[reference pending]()]

Our stochastic block partitioning (SBP) code is based on the reference implementation found in the [Graph Challenge](http://graphchallenge.org)

# Installation

```
git clone --recursive https://github.com/vtsynergy/IntegratedSBP.git
cd IntegratedSBP
mkdir build
cd build
cmake ..
make
```

For testing purposes, you can run the above with 
- `cmake -DSCOREP=ON` for score-p instrumentation
- `cmake -DVALGRIND=ON` for valgrind instrumentation
- `cmake -DDEBUG=ON` for address sanitizer instrumentation

Tests can be run with `make test`, however some of the test build options (in particular, `-DDEBUG=ON`) will cause all tests to fail due to memory leaks.

# Usage

Typical usage of the integrated approach:
```
mpiexec -n <num_cluster_nodes> ./SBP --filepath <path> -a hybrid_mcmc --threads <num_threads> \
--batches <num_batches> --distribute none-edge-balanced --nonparametric -m 0.075 \
--degreeproductsort --samplingalg expansion_snowball --samplesize 0.5
```

The assumption is that the graph file is stored in `<path>.mtx` or `<path>.tsv`. If there is 
ground truth, it is assumed that it is stored in `<path>_truePartition.tsv`. 

We recommend running parallel and distributed versions of SBP with `<num_batches>` >= 2.

Adding the `--evaluate` option will evaluate the results during the run. Excluding it will 
just save the final community detection results, as well as most of the parameters passed to 
the executable, to a json file for later evaluation. The `--tag="some identifier"` option 
can be useful for quickly differentiating between runs.

We recommend running with the new `--nonparametric` option to improve runtime. For best
results, do not use the `--greedy` and `--approximate` options together with this.

The `--greedy` option is a misnomer: excluding it will make the algorithm use a "greedy" 
MCMC technique that does not compute the Hastings Correction. This option should speed up 
the algorithm, but in our testing has often led to a decrease in accuracy, so we do not 
recommend doing so.

The `--approximate` option will run a less involved involved block merge phase for SBP, 
which may speed up the code at the cost of runtime. When using the `--nonparametric`
option, we recommend omitting this parameter.

By specifying `-a async_gibbs`, SBP will run in a fully asynchronous manner, using 
asynchronous Gibbs instead of Hybrid SBP or the Metropolis Hastings algorithm. This will
often be significantly faster than both algorithms, but can severely reduce the quality of
community detection results on some graphs. Note that this option has not been fully tested
with the integrated approach, but should work if you do not use EDiSt (by running the SBP 
executable without MPI).

`--distribute` options other than `none-edge-balanced` are not fully tested with the current
code.

WARNING: the `--degreecorrected` option is not fully implemented. If you use this option,
you will almost definitely get bad results.

Use a combination of `--json` (for the directory) and `--output_file` to control where the
resulting partitions are stored.

Running `./SBP --help` will provide a description of all available command-line options.

# References

If you use this code, please cite our HPEC paper:

```
@INPROCEEDINGS{Wanye2023IntegratedSBP,
  author={Wanye, Frank and Gleyzer, Vitaliy and and Kao, Edward and Feng, Wu-chun},
  booktitle={2023 IEEE High Performance Extreme Computing Conference (HPEC)}, 
  title={An Integrated Approach for Accelerating Stochastic Block Partitioning}, 
  year={2023},
  volume={},
  number={},
  month={9},
  pages={1-7},
  address={Waltham, MA},
  doi={10.1109/HPEC58863.2023.10363599}}
```

# License

&copy; Virginia Polytechnic Institute and State University, 2023.

Please refer to the included [LICENSE](./LICENSE) file.
