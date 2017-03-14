#!/usr/bin/python3
# Requires Python 3

"""
March Madness DEE/ELO prediction script
Copyright (C) 2013-2017 Kyle Barlow

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

# Python import statements
import argparse
import os
import sys
import random

# Constants
program_description = 'Python script to generate march madness brackets from ELO input (as in the format of, but not necessarily, the 538 data)'
default_output_file = 'output.txt'
default_data_file = 'elo.tsv' # Caches url results

region_pairings = ( ('east', 'west'), ('midwest', 'south') )

elo_k_factor = 26 # How fast ELO changes

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

class Team:
    def __init__(self, name, region, seed, elo):
        self.region = region.lower()
        self.seed = seed
        self.name = name
        self.starting_elo = elo
        self.elo = elo

    @classmethod
    def init_from_line(cls, team_line, separator_character = '\t'):
        line_data = team_line.split(separator_character)
        assert( len(line_data) >= 4 )
        name = line_data[0]
        region = line_data[1]
        try:
            seed = int(line_data[2])
            elo = float(line_data[3])
        except ValueError:
            print ('Error parsing this line:')
            print (team_line)
            print (line_data)
            raise

        return cls(name, region, seed, elo)

    def __repr__(self):
        return self.name

    # def __lt__(self, other):
    #      return self.seed < other.seed

    def update_elo(self, number_wins, win_prob):
        self.elo = self.elo + elo_k_factor * (number_wins - win_prob)

    def random_match(self, other):
        # Returns true if we randomly beat other team, false if not
        # Also updates ELOs
        win_prob = 1.0 / (1.0 + 10.0 ** ((other.elo - self.elo) / 400.0) )
        number_wins = 0
        if random.random() < win_prob:
            number_wins += 1
        self.update_elo( number_wins, win_prob )
        other.update_elo( 1 - number_wins, 1.0 - win_prob )

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

    def team_visualize(self, spacer = ''):
        vis_lines = []
        # if self._region_name != None and self._round_number >= 4:
        #     vis_lines.append( '{}Round {} - {} - {}'.format(spacer, self._round_number, self._round_name, self._region_name.capitalize()) )
        # else:
        #     vis_lines.append( '{}Round {} - {}'.format(spacer, self._round_number,self._round_name) )
        for child in self._children:
            vis_lines.extend( child.team_visualize( spacer = spacer + '  ' ) )
        if self._winning_team_index == None:
            for team in self._teams:
                vis_lines.append( ' {}{}'.format(spacer, team.name) )
        else:
            vis_lines.append( ' {}{} def. {}'.format(spacer, self._teams[self._winning_team_index].name, self._teams[1-self._winning_team_index].name) )
        return vis_lines

    def add_team(self, team):
        self._teams.append( team )

    def add_child(self, child):
        assert( child._round_number + 1 == self._round_number )
        if self._region_name != None:
            assert( child._region_name == self._region_name )
        child.set_parent = self
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
        with open(default_data_file, 'r') as f:
            lines = f.readlines()

            # Read in team data
            for line in lines[1:]:
                team = Team.init_from_line(line)
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

    def simulate_fill(self):
        # Randomly fills in bracket based on ELO simulation
        # Must be run on blank bracket
        assert( self._winning_team_index == None )
        for child in self._children:
            child.simulate_fill()
            self._teams.append( child._teams[child._winning_team_index] )

        assert( len( self._teams ) == 2 )
        if self._teams[0].random_match( self._teams[1] ):
            self._winning_team_index = 0
        else:
            self._winning_team_index = 1

def predictor():
    # Setup argument parser
    parser = argparse.ArgumentParser(description=program_description)
    parser.add_argument('-o', '--output',
                        default = default_output_file,
                        help = "File to save output")
    parser.add_argument('-q', '--quiet',
                        action = 'store_true',
                        default = False,
                        help = "Doesn't print bracket output to terminal")
    parser.add_argument('-m', '--maximize_score',
                        action = 'store_true',
                        default = True,
                        help = "Finds bracket with maximum expected score")

    args = parser.parse_args()

    if args.maximize_score:
        bt = BracketTree.init_starting_bracket()
        bt.simulate_fill()
        print ( '\n'.join( bt.team_visualize() ) )
        return 0

if __name__ == "__main__":
    sys.exit( predictor() )
