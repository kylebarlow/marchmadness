#!/usr/bin/env python3
# Requires Python 3

"""
March Madness prediction script
Copyright (C) 2013-2019 Kyle Barlow

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

# Python standard library import statements
import argparse
import time
import os
import sys
import random
import copy
import multiprocessing
import queue
import pickle
import threading
import urllib.request
import itertools
import collections

# NumPy
import numpy as np

import pandas as pd

# Constants
use_multiprocessing = True
program_description = 'Python script to generate march madness brackets from ELO input (as in the format of, but not necessarily, the 538 data)'
default_output_file = 'output.txt'
source_url = 'https://projects.fivethirtyeight.com/march-madness-api/2019/fivethirtyeight_ncaa_forecasts.csv'
default_data_file = 'fivethirtyeight_ncaa_forecasts.csv' # Caches url results

region_pairings = ( ('east', 'west'), ('midwest', 'south') )

# How fast ELO changes
elo_k_factor = 0.00000001 # Arbitrarily small to prevent updates for now # 15.0 / 30.463

# Mapping for strings describing each round to an integer (for indexing)
round_dictionary = {
    0 : 'FIRST FOUR',
    1 : 'ROUND OF 64',
    2 : 'ROUND OF 32',
    3 : 'ROUND OF 16',
    4 : 'ELITE 8',
    5 : 'FINAL 4',
    6 : 'FINALS',
}

seed_pairs_by_round = {
    1 : {
        1:16, 16:1,
        8:9, 9:8,
        5:12, 12:5,
        4:13, 13:4,
        6:11, 11:6,
        3:14, 14:3,
        7:10, 10:7,
        2:15, 15:2,
    },
    2 : {
        1:8, 8:1,
        4:5, 5:4,
        3:6, 6:3,
        2:7, 7:2,
    },
    3 : {
        1:4, 4:1,
        2:3, 3:2,
    },
    4 : {
        1:2, 2:1,
    },
}

class MonteCarloBracketSimulator(object):
    def __init__(self, starting_bt):
        self.highest_bt = starting_bt.copy()
        self.last_bt = starting_bt.copy()
        self.highest_score = starting_bt.expected_score()
        self.last_score = self.highest_score
        self.temperature = 100.0

    def set_last_bt(self, bt):
        self.last_bt = bt.copy()
        self.last_score = bt.expected_score()

    def boltzmann(self, bt):
        bt_score = bt.expected_score()
        score_delta = self.last_score - bt_score
        boltz_factor = ( -1 * score_delta / self.temperature )
        probability = np.exp( min(40.0, max(-40.0, boltz_factor) ) )

        if probability < 1:
            if random.random() > probability:
                # print ( 'reject', probability, self.last_score, bt_score )
                return False # reject
        #     else:
        #         print ( 'MC accept', probability, self.last_score, bt_score )
        # else:
        #     print ( 'accept', probability, self.last_score, bt_score )

        # Accept
        self.last_bt = bt.copy()
        self.last_score = bt_score
        if self.highest_score == None or self.last_score > self.highest_score:
            self.highest_score = self.last_score
            self.highest_bt = bt.copy()

        return True

    def copy(self):
        return pickle.loads( pickle.dumps(self) )

class Team(object):
    def __init__(self, name, region, seed, elo, win_prob_by_round):
        self.region = region.lower()
        self.seed = seed
        self.name = name
        self.starting_elo = elo
        self.elo = elo
        self.win_prob_by_round = win_prob_by_round

        # Keeps track of past ELO changes so we can undo them
        self.elo_history = {}

    @classmethod
    def init_from_row(cls, row, separator_character = ','):
        name = row['team_name']
        region = row['team_region']
        seed = row['team_seed']

        win_prob_by_round = {}
        for round_key in range(0, 7):
            win_prob_by_round[round_key] = float( row[ 'rd%d_win' % (round_key + 1) ] )

        if seed.endswith('a') or seed.endswith('b'):
            seed = seed[:-1]
        try:
            seed = int(seed)
            elo = float(row['team_rating'])
        except ValueError:
            print ('Error parsing this line:')
            print (row)
            raise

        return cls(name, region, seed, elo, win_prob_by_round)

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        # Only check equality based on names
        return self.name == other.name

    def __lt__(self, other):
         return self.elo < other.elo

    def update_elo(self, number_wins, win_prob, round_number):
        elo_change = elo_k_factor * (number_wins - win_prob)
        self.elo += elo_change
        assert( round_number not in self.elo_history ) # We can only have played one match per round
        self.elo_history[round_number] = elo_change

    def undo_elo_update(self, starting_round_number):
        '''
        Undo changes to ELO in self for specific round, and all rounds greater than that round
        '''
        for round_number in range(starting_round_number, max( round_dictionary.keys() ) + 1 ):
            if round_number in self.elo_history:
                # Later round numbers may not be in history if team lost earlier, so we use this if to check
                self.elo -= self.elo_history[round_number]
                del self.elo_history[round_number]

    def probability_of_victory(self, other):
        prob = 1.0 / (1.0 + 10.0 ** ( (other.elo - self.elo) * 30.464 / 400.0) )
        # print( 'prob_v', self, other, other.elo, self.elo, '%.2f' % prob )
        return prob

    def play_match(self, other, round_number, rigged = False, threshold_win_prob = None):
        '''
        Returns true if we beat other team, otherwise false
        Will randomly pick winner based on ELO, unless is rigged (in which case self wins)
        Updates ELOs
        If threshold_win_prob is not None, then team must have at least that chance of winning to win
        '''
        win_prob = self.probability_of_victory(other)
        number_wins = 0
        if rigged:
            number_wins += 1
        elif threshold_win_prob != None and 1.0 - win_prob < threshold_win_prob:
            number_wins += 1
        elif random.random() < win_prob:
            number_wins += 1

        self.update_elo( number_wins, win_prob, round_number )
        other.update_elo( 1 - number_wins, 1.0 - win_prob, round_number )

        if number_wins == 1:
            return True
        else:
            return False

class BracketTree(object):
    def __init__(self, round_number, region_name = None, seeds = None):
        self._children = []
        self._parent = None
        self._round_name = round_dictionary[round_number]
        self._round_number = round_number
        self._region_name = region_name
        self._seeds = seeds

        self._teams = []
        self._winning_team_index = None

    def copy(self):
        # Return fast copy by pickling
        return pickle.loads( pickle.dumps(self) )

    def visualize(self, spacer_len = 0, print_score = True, view_by_round = False, top_level_call = True):
        vis_lines = []
        if print_score:
            vis_lines.append( 'Expected score: %.2f' % self.expected_score() )
        vis_lines.append( '{}{}'.format(spacer_len * '-', self._round_name) )
        if self._winning_team_index == None:
            for team in self._teams:
                vis_lines.append( '{}{}'.format(spacer_len * ' ', team.name) )
        else:
            vis_lines.append( '{}{} ({}) def. {} ({})'.format(spacer_len * ' ', self._teams[self._winning_team_index].name, int(self._teams[self._winning_team_index].seed), self._teams[1-self._winning_team_index].name, int(self._teams[1-self._winning_team_index].seed)) )
        for child in self._children:
            if view_by_round:
                vis_lines.extend( child.visualize( spacer_len = 0, print_score = False, view_by_round = True, top_level_call = False ) )
            else:
                vis_lines.extend( child.visualize( spacer_len = spacer_len + 2, print_score = False, view_by_round = False, top_level_call = False ) )

        if top_level_call and view_by_round:
            score_line = ''
            if print_score:
                score_line = vis_lines[0]
                vis_lines = vis_lines[1:]
            last_round_line = None
            lines_by_round = collections.OrderedDict()
            for i, vis_line in enumerate(vis_lines):
                if i % 2 == 0:
                    last_round_line = vis_line
                    if last_round_line not in lines_by_round:
                        lines_by_round[last_round_line] = []
                else:
                    lines_by_round[last_round_line].append( vis_line )

            return_round_lines = []
            if print_score:
                return_round_lines.append(score_line)
            for round_line in lines_by_round:
                return_round_lines.append(round_line)
                for team_line in lines_by_round[round_line]:
                    return_round_lines.append(team_line)
                return_round_lines.append('')
            return return_round_lines

        return vis_lines

    def add_team(self, team):
        self._teams.append( team )

    def add_child(self, child):
        assert( child._round_number + 1 == self._round_number )
        if self._region_name != None:
            assert( child._region_name == self._region_name )
        child.set_parent( self )
        self._children.append(child)

    def set_parent(self, parent):
        self._parent = parent

    def _init_add_children(self, regional_teams, seeds, cls):
        # Helper function used by init_starting_bracket
        assert( len(seeds) == len(regional_teams) )
        assert( len(seeds) >= 2 and len(seeds) % 2 == 0 )
        if len(seeds) > 2:
            for winning_seed in seeds[:2]:
                child = cls( self._round_number - 1, region_name = self._region_name )
                child_seeds = [winning_seed]
                current_round = self._round_number - 1
                while current_round > 0:
                    new_child_seeds = [ seed_pairs_by_round[current_round][s] for s in child_seeds]
                    child_seeds.extend( new_child_seeds )
                    current_round -= 1
                child_seeds.sort()
                child._init_add_children(
                    { k : regional_teams[k] for k in regional_teams if k in child_seeds },
                    child_seeds, cls,
                )
                self.add_child( child )
        else:
            for seed in seeds:
                if len(regional_teams[seed]) > 1:
                    # First four seed, add one more child
                    child = cls( self._round_number - 1, region_name = self._region_name )
                    for team in regional_teams[seed]:
                        child.add_team(team)
                    self.add_child( child )
                else:
                    # Not a first four seed
                    for team in regional_teams[seed]:
                        self.add_team( team )

    @classmethod
    def init_starting_bracket(cls):
        '''
        Uses round_dictionary to initialize a full bracket. Bracket is filled in according to results so far.
        '''
        teams = {}
        min_seed = None
        max_seed = None

        if not os.path.isfile(default_data_file):
            urllib.request.urlretrieve(source_url, default_data_file)

        df = pd.read_csv(default_data_file)
        df = df.loc[ df['gender'] == 'mens' ].copy().sort_values('forecast_date', ascending = False )
        df = df.loc[ df['forecast_date'] == df.iloc[0]['forecast_date'] ].copy()
        df = df.loc[ df['team_alive'] == 1 ].copy()
        df = df.drop_duplicates( ['team_name'] )

        # Read in team data
        for index, row in df.iterrows():
            team = Team.init_from_row(row)
            if min_seed == None or team.seed < min_seed:
                min_seed = team.seed
            if max_seed == None or team.seed > max_seed:
                max_seed = team.seed

            if team.region not in teams:
                teams[team.region] = {}
            if team.seed not in teams[team.region]:
                teams[team.region][team.seed] = [team]
            else:
                teams[team.region][team.seed].append( team )

        # Initialize root node (finals) and semifinals
        max_round = max(round_dictionary.keys())
        finals = cls(max_round)
        for region_names in region_pairings:
            final_four = cls(max_round-1)
            for region_name in region_names:
                elite_eight = cls(max_round-2, region_name = region_name)
                seeds = list( range(min_seed, max_seed + 1) )
                elite_eight._init_add_children( teams[region_name], seeds, cls )
                final_four.add_child( elite_eight )
            finals.add_child( final_four )

        return finals

    def random_perturb(self, pop_size):
        nodes = random.sample( self.all_nodes(), pop_size )
        for node in nodes:
            node.swap_winner()
        # Run final verification after all swaps are complete
        self.verify_bracket()

    def single_random_perturb(self):
        node = random.choice( self.all_nodes() )
        node.swap_winner()

    def all_nodes(self):
        nodes = [ self ]
        for child in self._children:
            nodes.extend( child.all_nodes() )
        return nodes

    def all_teams(self):
        all_teams = []
        for node in self.all_nodes():
            all_teams.extend( node._teams )
        return all_teams

    def swap_winner(self, threshold_win_prob = None):
        assert( len(self._teams) == 2 )
        current_winner = self._teams[ self._winning_team_index ]
        current_loser = self._teams[ 1 - self._winning_team_index ]
        loser_win_prob = current_loser.probability_of_victory(current_winner)
        if threshold_win_prob != None and loser_win_prob < threshold_win_prob:
            return

        for team in self._teams:
            team.undo_elo_update(self._round_number)

        if self._parent != None:
            self._parent.remove_team_upwards( self._teams[self._winning_team_index], self._teams[ 1 - self._winning_team_index] )

        self._winning_team_index = 1 - self._winning_team_index

        # Update ELOs according to swapped result
        self._teams[self._winning_team_index].play_match( self._teams[ 1 - self._winning_team_index], self._round_number, rigged = True )

    def remove_team_upwards(self, old_winner, new_winner):
        '''
        Removes a team that previously won in a child game
        Resimulates new winner in new random match
        '''
        our_old_winner = self._teams[self._winning_team_index]

        self._teams.remove( old_winner )
        self._teams.append( new_winner )
        assert( len(self._teams) == 2 )

        # Undo ELO updates before new match
        for team in self._teams:
            team.undo_elo_update(self._round_number)

        # Play match
        if self._teams[0].play_match( self._teams[1], self._round_number ):
            self._winning_team_index = 0
        else:
            self._winning_team_index = 1

        # Recursive call upwards
        if self._parent != None:
            self._parent.remove_team_upwards( our_old_winner, self._teams[self._winning_team_index] )

    def verify_bracket(self):
        '''
        Ensures that a bracket is valid and filled
        Checks that if a team won a lower round, is present in the upper round
        '''
        assert( self._winning_team_index != None )
        assert( len(self._teams) == 2 )
        prev_round_winners = sorted( self._teams )
        children_winners = sorted( [ child._teams[child._winning_team_index] for child in self._children ] )
        if len( self._children ) == 2:
            assert( prev_round_winners == children_winners )
        elif len( self._children ) == 1:
            assert( children_winners[0] in prev_round_winners )

        for child in self._children:
            child.verify_bracket()

    def simulate_fill(self):
        # Randomly fills in bracket based on ELO simulation
        # Fills in blanks
        assert( self._winning_team_index == None )
        for child in self._children:
            child.simulate_fill()
            self._teams.append( child._teams[child._winning_team_index] )

        assert( len( self._teams ) == 2 )
        if self._teams[0].play_match( self._teams[1], self._round_number ):
            self._winning_team_index = 0
        else:
            self._winning_team_index = 1

    def all_team_names(self):
        teams = set()
        for child in self._children:
            teams.update( child.all_team_names() )
        teams.update( [team.name for team in self._teams] )
        return teams

    def winners_vector(self):
        '''
        Returns vector representing how far teams advanced
        '''
        winners_dict = self.winners_dict()
        v = np.zeros( (len(winners_dict), len(round_dictionary)) )
        team_names = sorted( winners_dict.keys() )
        for i, team_name in enumerate(team_names):
            if winners_dict[team_name] >= 0:
                for j in range(0, winners_dict[team_name]+1):
                    v[i][j] += 1
        return v

    def team_names(self):
        return sorted( self.winners_dict().keys() )

    def winners_dict(self, furthest_round = None):
        if furthest_round == None:
            min_round = min(round_dictionary.keys())
            furthest_round = {name : min_round - 1 for name in self.all_team_names()}
        for team in self._teams:
            if self._round_number - 1 > furthest_round[team.name]:
                furthest_round[team.name] = self._round_number - 1
        winning_team_name = self._teams[self._winning_team_index].name
        if self._round_number > furthest_round[winning_team_name]:
            furthest_round[winning_team_name] = self._round_number
        for child in self._children:
            child.winners_dict( furthest_round )
        return furthest_round

    def total_probability(self):
        assert( len(self._teams) == 2 )
        winning_team = self._teams[self._winning_team_index]
        losing_team = self._teams[1-self._winning_team_index]

        return_prob = winning_team.probability_of_victory(losing_team)
        if len(self._children) == 2: # Skip first 4
            for child in self._children:
                return_prob = return_prob * child.total_probability()

        if return_prob > 1.0 or return_prob < 0.0:
            print( winning_team, losing_team, self._round_number, winning_team.probability_of_victory(losing_team), child_with_winner.total_probability(), self._children[0].total_probability(), self._children[1].total_probability(), winning_team.elo, losing_team.elo )
            print( return_prob )
            raise Exception()

        return return_prob

        # return 0
        # for child in self._children:
        #     probability_of_victory *= child.total_probability()

        # assert( self._winning_team_index != None )
        # assert( len(self._teams) == 2 )
        # winning_team = self._teams[self._winning_team_index]
        # losing_team = self._teams[1-self._winning_team_index]

        # probability_of_victory *= winning_team.probability_of_victory(losing_team)

        # return probability_of_victory

    def round_cbs_score(self):
        #  This dictionary is used to calculate the expected score of a bracket in leagues where
        #  additional points are awarded for correct picks in later rounds. Each key corresponds
        #  to the number of a round (see round_dictionary) above, and each value corresponds to
        #  the weight for each correct pick in that round. For example, a key/value pair of
        #  3:2 would mean that a correct pick in the third round is worth twice as much as the baseline

        # The seed of winner is also added to score (to give more points for picking low seeds)
        default_cbs_scores = {
            0:0,
            1:1,
            2:2,
            3:3,
            4:4,
            5:6,
            6:8
        }
        assert( self._winning_team_index != None )
        assert( len(self._teams) == 2 )
        winning_team = self._teams[self._winning_team_index]

        return default_cbs_scores[self._round_number] + winning_team.seed

    def round_yahoo_score(self):
        default_yahoo_scores = {
            0:0,
            1:1,
            2:2,
            3:4,
            4:8,
            5:16,
            6:32
        }
        assert( self._winning_team_index != None )
        assert( len(self._teams) == 2 )
        winning_team = self._teams[self._winning_team_index]
        losing_team = self._teams[1-self._winning_team_index]

        return max( [0, winning_team.seed - losing_team.seed] ) + default_yahoo_scores[self._round_number]

    def expected_score(self):
        # Expected value of our winner beating all possible opponents, recursive
        score = 0.0

        winning_team = self._teams[self._winning_team_index]
        losing_team = self._teams[1-self._winning_team_index]
        if len(self._children) == 2:
            # Only recurse if two children (to avoid first four games)
            child_with_loser = None
            if self._children[0]._teams[0].name == losing_team.name or self._children[0]._teams[1].name == losing_team.name:
                child_with_loser = self._children[0]
            if self._children[1]._teams[0].name == losing_team.name or self._children[1]._teams[1].name == losing_team.name:
                assert( child_with_loser == None )
                child_with_loser = self._children[1]
            assert( child_with_loser != None )

            for possible_opponent in child_with_loser.all_teams():
                prob_opponent = possible_opponent.win_prob_by_round[self._round_number-1]
                score += winning_team.probability_of_victory(possible_opponent) * prob_opponent
            for child in self._children:
                score += child.expected_score()
        else:
            score += self.round_score() * winning_team.probability_of_victory(losing_team)

        return score

    def round_score(self):
        # Have to change score function manually below for now
        return self.round_cbs_score()

    def score(self):
        score = self.round_score()
        for child in self._children:
            score += child.round_score()
        return score

def simulate_winners_vector(bt_pickle):
    bt_copy = pickle.loads(bt_pickle)
    bt_copy.simulate_fill()
    return bt_copy.winners_vector()

class CallbackVectorQueue(object):
    def __init__(self, initial_v):
        self.q = queue.Queue()
        self.v = initial_v
        self.trials = 0

        self.thread = threading.Thread(target=self.thread_run)
        self.thread.daemon = True # Daemonize thread
        self.thread.start()

    def thread_run(self):
        while True:
            self.v += self.q.get()
            self.trials += 1

    def callback(self, v):
        self.q.put(v)

    def close(self):
        while not self.q.empty():
            time.sleep(0.001)

def run_stats( number_simulations = 10000 ):
    bt = BracketTree.init_starting_bracket()
    # Initial simulation to initialize vector
    bt_pickle = pickle.dumps( bt )
    initial_v = simulate_winners_vector(bt_pickle)
    v_callback = CallbackVectorQueue(initial_v)

    if use_multiprocessing:
        pool = multiprocessing.Pool()

    for sim_num in range(0, number_simulations):
        if use_multiprocessing:
            pool.apply_async( simulate_winners_vector, args = (bt_pickle,), callback = v_callback.callback )
        else:
            v_callback.callback( simulate_winners_vector(bt_pickle) )

    if use_multiprocessing:
        pool.close()
        pool.join()
    v_callback.close()

    v = v_callback.v
    v /= float( number_simulations )
    print_list = []
    # Run simulation to fill in team names
    bt.simulate_fill()
    for i, team_name in enumerate( bt.team_names() ):
        champion_percentage = v[i][ len(round_dictionary) - 1 ]
        l = list( reversed( v[i] ) )
        l.append( team_name )
        print_list.append( l )
    print_list.sort( reverse = True )
    for row in print_list:
        line = ''
        for x in row:
            if isinstance(x, str):
                line += x
            else:
                line += '%.2f ' % x
        print ( line )
    print ( 'Total trials: %d' % v_callback.trials )

def run_monte_carlo_helper(temp_steps, max_perturbations, mc, blank_bt):
    # chance of fresh bracket start
    if random.random() >= 0.95:
        bt = blank_bt.copy()
        bt.simulate_fill()
        mc.set_last_bt( bt )

    for temperature in temp_steps:
        bt = mc.last_bt.copy()
        # Perturb
        bt.random_perturb( random.randint(1, max_perturbations) )
        # bt.single_random_perturb()

        # Score
        mc.temperature = temperature
        mc.boltzmann( bt )
    return mc

def run_monte_carlo( num_trials = 10000, view_by_round = False ):
    # Parameters for MC simulation
    max_perturbations = 10
    starting_temp = 20.0
    ending_temp = 1.0
    low_temp_final_steps = 500
    # Output parameters
    highest_mc_bt_cache = os.path.join('cache', 'highest_mc_bt.pickle') # Saves best bracket for reloading as starting point in later simulations
    highest_vis_output = os.path.join('cache', 'highest_bracket.txt')

    blank_bt = BracketTree.init_starting_bracket()
    if os.path.isfile( highest_mc_bt_cache ):
        with open(highest_mc_bt_cache, 'rb') as f:
            bt = pickle.load(f)
    else:
        if not os.path.isdir( os.path.dirname( highest_mc_bt_cache ) ):
            os.makedirs( os.path.dirname( highest_mc_bt_cache ) )
        # Initial simulation
        bt = blank_bt.copy()
        bt.simulate_fill()

    mc = MonteCarloBracketSimulator( bt )

    temp_steps = list( np.arange(starting_temp, ending_temp, -0.005) )
    temp_steps.extend( [ending_temp for x in range(low_temp_final_steps) ] )

    def callback(thread_mc):
        nonlocal mc
        if thread_mc.highest_score > mc.highest_score:
            mc = thread_mc

    for trial in range(num_trials):
        if use_multiprocessing:
            pool = multiprocessing.Pool()
            cpu_count = multiprocessing.cpu_count()
        else:
            cpu_count = 1
        for cpu_count in range(cpu_count):
            if use_multiprocessing:
                pool.apply_async( run_monte_carlo_helper, args = (temp_steps, max_perturbations, mc.copy(), blank_bt), callback = callback )
            else:
                callback( run_monte_carlo_helper( temp_steps, max_perturbations, mc.copy(), blank_bt ) )

        if use_multiprocessing:
            pool.close()
            pool.join()

        print ( 'MC simulation complete (round {})'.format(trial) )
        print ( 'Highest score: %.2f' % mc.highest_score )
        print ( 'Last score: %.2f\n' % mc.last_score )

    with open(highest_mc_bt_cache, 'wb') as f:
        pickle.dump(mc.highest_bt, f)

    with open(highest_vis_output, 'w') as f:
        for line in mc.highest_bt.visualize():
            f.write( line + '\n' )

    if view_by_round:
        print( '\n'.join( mc.highest_bt.visualize( view_by_round = True ) ) )

def run_quick_pick( score_thresh, view_by_round = False ):
    while True:
        bt = BracketTree.init_starting_bracket()
        bt.simulate_fill()
        if score_thresh == None or bt.expected_score() >= score_thresh:
            break
    print ( '\n'.join( bt.visualize( view_by_round = view_by_round ) ) )

def predictor():
    # Setup argument parser
    parser = argparse.ArgumentParser(description=program_description)
    parser.add_argument(
        '-s', '--stats',
        type = int,
        default = 0,
        help = "Run many times to get statistics"
    )
    parser.add_argument(
        '-m', '--monte_carlo',
        type = int,
        default = 0,
        help = "How many outer loops of ramping monte carlo simulation"
    )
    parser.add_argument(
        '-q', '--quick_pick',
        default = False,
        action = 'store_true',
        help = 'Generate a "quick pick" style bracket'
    )
    parser.add_argument(
        '--quick_thresh',
        default = None,
        type = float,
        help = 'If running a quick pick, you can specify a minimum expected score threshold here'
    )
    parser.add_argument(
        '--view_by_round',
        default = False,
        action = 'store_true',
        help = 'Print output by round'
    )

    args = parser.parse_args()

    if args.quick_pick:
        run_quick_pick( args.quick_thresh, view_by_round = args.view_by_round )

    if args.stats > 0:
        run_stats( args.stats )

    if args.monte_carlo > 0:
        run_monte_carlo( args.monte_carlo, view_by_round = args.view_by_round )

if __name__ == "__main__":
    predictor()
