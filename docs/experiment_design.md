# Experiment Design

## Goals
For each model we need the following runs:

A. no interference, no rebalancing -> GPipe

B. no interference, with rebalancing -> plain `evaluation.py` run

C. interference, no rebalancing -> GPipe with interference

D. interference with exploration, but only one exploration
- we stop at the first optimum 
- see how the first config is affected by interferences

E. interference with exploration and rebalancing
- interference eval

## Static Baselines
- run A
- baselines that don't change or have interference

## No Interference Experiment
- run B
- use `evaluation.py`
- run it for 10000 requests
- desired results:
	- improved performance since initial seed
	- reaches optimum and stays there
	- does not get worse

## Interference Experiments
- runs C, D and E
- interference script
  - starts and stops benchmark processes
  - benchmark processes:
    - cpu usage: https://github.com/nikela/CPU_stress
    - memory usage: https://github.com/intel/memory-bandwidth-benchmarks
      - set array size to >60MB
  - use taskset to select what cpu cores to use
    - 32 real cores, 64 total virtual cores
    - adaptive pipeline should be on the first 32 cores, while the last 32 cores should have the benchmarks on them
- runs C, D and E should all use a random schedule, but with the same seed for the same experiment run
- desired results:
  - run C: GPipe should be affected by interferences a lot, as it is not made for it
  - run D: massive slowdown due to interference; shows the value of dynamic rebalancing
  - run E: should improve and adapt to the benchmarks

## Bringing it all together
- one script should run all of these
- first run baselines if they don't exist
- then runs with no interference
- then runs with interference
- should be able to be repeated multiple times.
- outputs:
  - each run should have its own data file
  - filename example: `run_A.json`
  - all output to `data/eval/<timestamp>/`
  - if an error happens, capture it and write it to `error.txt` in the same dir 

