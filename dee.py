#!/usr/bin/python3
# Requires Python 3

"""
March Madness prediction script
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
import operator
import random
import time
from multiprocessing import Pool
import urllib.request, urllib.parse, urllib.error
import re
from datetime import datetime

# Constants
program_description = 'Python script to generate march madness brackets from probability input (as in the format of, but not necessarily, the 538 data)\nEach probability input is assumed to be built up conditionally'
default_input_html = 'https://projects.fivethirtyeight.com/march-madness-api/2017/fivethirtyeight_ncaa_forecasts.csv'
default_output_file = 'output.txt'
default_cache_file = 'data_cache.csv' # Caches url results

# Mapping for strings describing each round to an integer (for indexing)
# Counting starts with 1
round_dictionary = {
    1:'FIRST FOUR',
    2:'ROUND OF 32',
    3:'ROUND OF 16',
    4:'ELITE 8',
    5:'FINAL 4',
    6:'FINALS',
    7:'CHAMPIONS'
}

# Expected header string, to make sure no changes from last script run
expected_header_string = 'gender,forecast_date,playin_flag,rd1_win,rd2_win,rd3_win,rd4_win,rd5_win,rd6_win,rd7_win,team_alive,team_id,team_name,team_rating,team_region,team_seed'

class Team:
    # Stores information on a team's probabilities of advancing to each round
    def __init__(self, name, region, seed, round_odds, conditional_round_odds):
        self.region = region
        self.seed = seed
        self.name = name
        self.round_odds = round_odds
        self.conditional_round_odds = conditional_round_odds

        self.advanced_to_round = 0
        while self.round_odds[self.advanced_to_round] == 1.0:
            self.advanced_to_round += 1

    @classmethod
    def init_from_line(cls, team_line):
        line_data = team_line.split(',')
        region = line_data[14]
        m = re.match('(\d+)(.*?)', line_data[15])
        if m:
            seed = int( m.group(1) )
        else:
            raise Exception( "Couldn't match seed: " + str(line_data[15]) )
        name = line_data[12]
        round_odds = []
        for item in line_data[3:10]:
            item = item.strip()
            if item == '':
                round_odds.append(None)
            elif item == '0':
                round_odds.append(0.000000000000001)
            else:
                x = float(item)
                if x == 0.0:
                    x = 0.000000000000001
                round_odds.append(x)

        # Make the probabilities conditional
        conditional_round_odds = []
        for i,odd in enumerate(round_odds):
            if i == 0:
                conditional_round_odds.append(odd)
            else:
                prev_round = round_odds[i-1]
                if prev_round == None:
                    conditional_round_odds.append(odd)
                else:
                    conditional_round_odds.append(odd/prev_round)

        return cls(name, region, seed, round_odds, conditional_round_odds)

    def copy(self):
        # Returns all duplicate (should be non-mutated) objects, except for resets seed slot
        return Team(self.name, self.region, self.seed, self.round_odds, self.conditional_round_odds)

    def __getitem__(self, i):
        return self.round_odds[i-1] # Rounds are indexed from 1; lists from 0

    def __repr__(self):
        return self.name

    def __lt__(self, other):
         return self.seed < other.seed

class BracketTree(object):
    def __init__(self):
        self.children = (None, None)
        self.parent = None
        self.round_name = None
        self.round_number = None
        self.winner_index = None # Index in children of winner

    @classmethod
    def init_starting_bracket(cls, html_url):
        '''
        Uses round_dictionary to initialize a full bracket. Bracket is filled in according to results so far.
        '''
        if not os.path.isfile( default_cache_file ):
            urllib.urlretrieve (html_url, default_cache_file)

        with open(default_cache_file, 'r') as f:
            lines = f.readlines()

            # Check for correct header line
            header_line = lines[0].strip()
            if header_line != expected_header_string:
                print ( header_line )
                print ( expected_header_string )
                raise Exception("Header line doesn't match expected format")

            # Figure out most recent prediction date
            max_pred_date = datetime.strptime('1900-01-01', '%Y-%m-%d')
            date_format = '%Y-%m-%d'
            for line in lines:
                if line.startswith('mens'):
                    pred_date = datetime.strptime(line.split(',')[1], date_format)
                    if pred_date > max_pred_date:
                        max_pred_date = pred_date

            # Read in team data
            for line in lines[1:]:
                if line.startswith('mens') and datetime.strftime(max_pred_date, date_format) in line:
                    team = Team.init_from_line(line)
                    print (team, team.advanced_to_round, team.round_odds)

        return None

def predictor():
    # Setup argument parser
    parser = argparse.ArgumentParser(description=program_description)
    parser.add_argument('-i', '--input_html',
                        default = default_input_html,
                        help = "Input FiveThirtyEight csv URL to parse")
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
        bt = BracketTree.init_starting_bracket(args.input_html)
        return 0

if __name__ == "__main__":
    sys.exit( predictor() )
