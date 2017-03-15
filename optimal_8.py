#!/usr/bin/env python3
# Requires Python 3

"""
This uses DEE/ELO prediction to simulate brackets to determine the optimal set
of 8 teams to use in a seed 8 pool
"""

import argparse
import copy
import itertools

import numpy as np
import scipy.stats
from tqdm import tqdm

from predict import BracketTree


def optimize(num_trials, target_payout, bonuses=None, n_teams=8):

    # default no bonus payout
    if bonuses is None:
        bonus_payout = dict.fromkeys(range(1, 7), 0)
    else:
        assert len(bonuses) == 6 and isinstance(bonuses, (tuple, list))
        bonus_payout = dict(zip(range(1,7), bonuses))

    # instantiate a blank bracket
    blank_bt = BracketTree.init_starting_bracket()

    # data structure to capture the payouts for each team
    seeds = blank_bt.all_team_seeds()
    payouts = {}
    for team_name in seeds.keys():
        payouts[team_name] = []

    # simulate bracket using ELO and accumulate statistics for each team in
    # terms of the number of points they would have received in this universe
    for trial in range(num_trials):
        bt = blank_bt.copy()
        bt.simulate_fill()

        # this captures the farthest round for this team
        winners_dict = bt.winners_dict()
        for team_name, farthest_round in winners_dict.items():
            payout = 0
            if farthest_round >= 1:
                for round in range(1, farthest_round+1):
                    payout += seeds[team_name] + bonus_payout[round]
            payouts[team_name].append(payout)

    # convert into numpy arrays for fast addition later
    for team_name in payouts:
        payouts[team_name] = np.array(payouts[team_name])

    # go through all combinations of teams that have a reasonable chance of
    # winning
    average_payouts = {}
    for team in payouts.keys():
        average_payouts[team] = np.mean(payouts[team])
    labelled = [(payout, team) for team, payout in average_payouts.items()]
    labelled.sort(reverse=True)
    contender_teams = [team for _, team in labelled[:40]]

    # go through all combinations of teams to identify the set of teams that is
    # the most likely to exceed a pre-defined threshold (for example, the score
    # of last year's winner)
    max_teams, max_total_payouts, max_p = None, None, 0
    iter_teams = itertools.combinations(contender_teams, n_teams)
    n_combinations = scipy.misc.comb(len(contender_teams), n_teams)
    print("finding best combination of %d teams..." % n_teams)
    iter_teams = tqdm(iter_teams, total=n_combinations)
    for teams in iter_teams:
        total_payouts = np.zeros(num_trials)
        for team in teams:
            total_payouts += payouts[team]
        p = 100 - scipy.stats.percentileofscore(total_payouts, target_payout, kind='strict')
        if p > max_p:
            max_p = p
            max_teams = teams
            max_total_payouts = total_payouts

    return max_p, max_teams, max_total_payouts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--monte_carlo',
                        type = int,
                        default = 1000,
                        help = "How many brackets do you want to generate")
    parser.add_argument('-b', '--bonus',
                        type = int,
                        nargs = 6,
                        default = None,
                        help = "Bonuses for winning in each round")
    parser.add_argument('target_score',
                        type = int,
                        help = "Target score you want to beat (like last year's winning score)")

    args = parser.parse_args()

    if args.monte_carlo > 0:
        max_p, max_teams, max_total_payouts = optimize(
            args.monte_carlo,
            args.target_score,
            args.bonus,
        )

        print("The following teams have a %.1f%% chance of winning:" % max_p)
        for team in max_teams:
            print("    %s" % team)
