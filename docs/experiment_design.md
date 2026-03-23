## TODO
- [ ] finish normal pipeline
	- [ ] find more models -- we should get like 6 or 8
		- [ ] find some transformer models
- [ ] implement interference pipeline
## Meeting notes
- interference experiment
	- have a shell script that starts inference
	- feed 10 minutes worth of requests 
	- then randomly start and stop processes 
	- set stream c array size to 80 MB so that it's bigger than the 30MB L3 cache
	- check mail for benchmarks
	- use different amount of threads
	- have both a random and deterministic benchmark

## Experiment 1: no interference
- should be similar to current evaluation script
- run it for 10000 requests
- desired results:
	- improved performance since initial seed
	- reaches optimum and stays there
	- does not get worse -- prove this using derivative?

## Experiment 2: interference
- running the http server 
- interference script
	- starts and stops benchmark processes
	- benchmark processes:
		- cpu usage: https://github.com/nikela/CPU_stress
		- memory usage: https://github.com/intel/memory-bandwidth-benchmarks
			- set array size to >60MB 
			- and check other options too
	- check on adaptive model runner queue and add to it if it is close to finishing
	- change benchmark every 10 minutes (?)