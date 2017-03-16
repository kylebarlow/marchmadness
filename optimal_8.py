#!/usr/bin/env python3
# Requires Python 3

"""
This uses DEE/ELO prediction to simulate brackets to determine the optimal set
of 8 teams to use in a seed 8 pool
"""

import argparse
import copy
import itertools
import os
import csv
import random

import numpy as np
import scipy.stats
from tqdm import tqdm

from predict import BracketTree


def simulate_payouts(num_trials, bonuses=None):
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

    return payouts


def calculate_probability(payouts, teams, target_payout, num_trials):
    total_payouts = np.zeros(num_trials)
    for team in teams:
        total_payouts += payouts[team]
    p = 100 - scipy.stats.percentileofscore(total_payouts, target_payout, kind='strict')
    return p, total_payouts


def limit_to_contender_teams(payouts, max_contender_teams=40):
    labelled = []
    for team in payouts.keys():
        labelled.append((np.mean(payouts[team]), team))
    labelled.sort(reverse=True)
    contender_teams = [team for _, team in labelled[:max_contender_teams]]
    return contender_teams


def optimize(num_trials, target_payout, bonuses=None, n_teams=8):

    # prepare output file
    output_filename = os.path.join('cache', 'optimal_8.csv')
    if not os.path.isdir( os.path.dirname( output_filename ) ):
        os.makedirs( os.path.dirname( output_filename ) )
    stream = open(output_filename, 'w')
    writer = csv.writer(stream)
    writer.writerow(['p_winning'] + ['team-%d' % i for i in range(1, 9)])

    # calculate the payouts
    payouts = simulate_payouts(num_trials, bonuses=bonuses)

    # go through all combinations of teams that have a reasonable chance of
    # winning
    contender_teams = limit_to_contender_teams(payouts)

    # go through all combinations of teams to identify the set of teams that is
    # the most likely to exceed a pre-defined threshold (for example, the score
    # of last year's winner)
    max_teams, max_total_payouts, max_p = None, None, 0
    iter_teams = itertools.combinations(contender_teams, n_teams)
    n_combinations = scipy.misc.comb(len(contender_teams), n_teams)
    print("finding best combination of %d teams..." % n_teams)
    iter_teams = tqdm(iter_teams, total=n_combinations)
    for teams in iter_teams:
        p, total_payouts = calculate_probability(payouts, teams, target_payout, num_trials)
        if p > max_p:
            max_p = p
            max_teams = teams
            max_total_payouts = total_payouts

        writer.writerow((p, ) + teams)
    stream.close()

    return max_p, max_teams, max_total_payouts


def print_probability(p, teams):
    print("The following teams have a %.1f%% chance of winning:" % p)
    for team in teams:
        print("    %s" % team)


def sample_total_points(payouts, n_samples, num_trials, n_teams=8):

    contender_teams = limit_to_contender_teams(payouts)

    total_points = []
    for i in range(n_samples):
        teams = random.sample(contender_teams, n_teams)
        _, outcomes = calculate_probability(payouts, teams, 0, num_trials)
        total_points.append(random.choice(outcomes))
    return total_points

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
    parser.add_argument('-p', '--probability',
                        type = str,
                        nargs = 8,
                        default = None,
                        metavar = "TEAM",
                        help = "Calculate probability of a given set of teams")
    parser.add_argument('-s', '--sample',
                        type = int,
                        default = 0,
                        help = "Sample candidate combinations of teams to better understand a good target")

    args = parser.parse_args()

    # in --probability mode, just check the probability of the specified set of
    # teams. convenient for trying alternate picks
    if args.probability:
        payouts = simulate_payouts(args.monte_carlo, args.bonus)
        p = calculate_probability(payouts, args.probability, args.target_score, args.monte_carlo)
        print_probability(p, args.probability)

    elif args.sample > 0:
        payouts = simulate_payouts(args.monte_carlo, args.bonus)
        total_scores = sample_total_points(payouts, args.sample, args.monte_carlo)
        total_scores.sort()
        for i, score in enumerate(total_scores):
            print(score, 1.0 - float(i) / args.sample)

    elif args.monte_carlo > 0:
        max_p, max_teams, max_total_payouts = optimize(
            args.monte_carlo,
            args.target_score,
            args.bonus,
        )
        print_probability(max_p, max_teams)
