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

# Mapping for strings describing each round to an integer (for indexing)
round_dictionary = {
    0 : 'FIRST FOUR',
    1 : 'ROUND OF 32',
    2 : 'ROUND OF 16',
    3 : 'ELITE 8',
    4 : 'FINAL 4',
    5 : 'FINALS',
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
            elo = int(line_data[3])
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

class BracketTree(object):
    def __init__(self, round_name, round_number, region_name = None):
        self._children = []
        self._parent = None
        self._round_name = round_name
        self._round_number = round_number
        self._region_name = region_name

        self._teams = []
        self._winning_team_index = None

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

    def _init_add_children(self, regional_teams, min_seed, max_seed, cls):
        # Helper function used by init_starting_bracket
        team1_seed = min_seed
        team2_seed = max_seed
        seed_pairs = []
        while team1_seed < team2_seed:
            seed_pairs.append( (team1_seed, team2_seed) )
            team1_seed += 1
            team2_seed -= 1

        # Make sure there are an even number of pairs
        assert( ( len(seed_pairs) % 2 ) == 0 )

        # Build up region from bottom up
        # TODO: switch this to be top down
        raise Exception('see comment above')
        first_round_games = []
        for seed_pair in seed_pairs:
            first_round_game = cls( round_dictionary[1], 1, region_name = self._region_name )
            for seed in seed_pair:
                if len( regional_teams[seed] ) == 2:
                    zero_round_game = cls( round_dictionary[0], 0, region_name = self._region_name )
                    for team in regional_teams[seed]:
                        zero_round_game.add_team( team )
                    first_round_game.add_child( zero_round_game )
                elif len( regional_teams[seed] ) == 1::
                    first_round_game.add_team( regional_teams[seed][0] )
                else:
                    raise Exception()

        sys.exit()

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
        finals = cls(round_dictionary[max_round], max_round)
        for region_names in region_pairings:
            semifinals = cls(round_dictionary[max_round-1], max_round-1)
            for region_name in region_names:
                final_four = cls(round_dictionary[max_round-2], max_round-2, region_name = region_name)
                final_four._init_add_children( teams[region_name], min_seed, max_seed, cls )
                semifinals.add_child( final_four )
            finals.add_child( semifinals )

        return None

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
        return 0

if __name__ == "__main__":
    sys.exit( predictor() )
