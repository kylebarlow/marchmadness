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
Run ./predict.py -h for a list of help options
* Generate a "quick-pick" bracket
`./predict.py -q`
* Run many simulations to look at summary statistics (useful for comparison with other probabilities to see how well the model is calibrated)
`./predict.py -s`
* Run a Monte Carlo simulation to try and find optimal bracket for scoring scheme
`./predict.py -m 20`
