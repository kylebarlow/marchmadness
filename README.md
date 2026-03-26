marchmadness
============

Python script to simulate march madness brackets from ELO input (as in the format of 538's ELO rankings, but could be from any source.

### Quick pick
When run with default flags, the script generates a "quick-pick" bracket (in the style of a lottery pick) that should be as likely as any other bracket (within the assumptions of the ELO system).

### Monte Carlo optimization
The script can also run a Monte Carlo simulated annealing simulation to try and find an optimal bracket for a particular bracket scoring system. This works by assigning an expected value "score" to each model bracket, based on how a bracket game scores your bracket (more points for upsets, etc.) and the probability of those upsets actually occuring.

Scoring for the CBS Sports bracket game I am personally in is implemented, but this should be relatively easy to modify to change for other games

## Getting started
* install python3
* pip install -r REQUIREMENTS

## Features
Run `./predict.py -h` for a list of help options
* Generate a "quick-pick" bracket
`./predict.py -q`
* Run many simulations to look at summary statistics (useful for comparison with other probabilities to see how well the model is calibrated)
`./predict.py -s`
* Run a Monte Carlo simulation to try and find optimal bracket for scoring scheme
`./predict.py -m 20`

Run `./optimal_8.py -h` for a list of help options for Seed 8 Pools.
* `./optimal_8.py 103` to run ELO simulations to choose the optimal set of 8 teams that will beat a particular score (`103` in this case). Results are output in `cache/optimal_8.csv`
* `./optimal_8.py 103 -b 0 0 5 10 15 20` to run ELO simulations for a per-round bonus structure of 0 points in first two rounds, 5 points in sweet 16, 10 points in elite 8, 15 points for final four and 20 points for winning the championship.
* `./optimal_8.py 103 -p 'Villanova' 'Wichita State' 'Southern Methodist' 'Arizona' 'Oregon' 'Kentucky' 'Duke' 'Michigan'` to run ELO simulations and see how likely a specific set of teams is to exceed the target score. This is convenient for making changes to the proposed best picks.
* `./optimal_8.py 103 -s` to sample the output of the total score distribution from the ELO simulations to gauge what a reasonable target would be.
