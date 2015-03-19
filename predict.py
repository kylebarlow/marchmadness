#!/usr/bin/python
# Requires Python 2.7

"""
March Madness prediction script
Copyright (C) 2013-2015 Kyle Barlow

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

# Constants
program_description = 'Python script to auto-generate "quick pick" march madness brackets from probability input (as in the format of, but not necessarily, the 538 data from Nate Silver)\nEach probability input is assumed to be built up conditionally'

default_input_file = 'data.csv'
default_output_file = 'output.txt'
default_nyt_file = 'nyt_scoring_data.csv'

# Expected header string
header_string = 'REGION,SEED,TEAM,FIRST FOUR,ROUND OF 32,ROUND OF 16,ELITE 8,FINAL 4,FINALS,CHAMPIONS'

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

# Maps 538 team names to NYT team names (when needed)
map_to_nyt_names = {
    'Mississippi' : 'Ole Miss',
    'Ohio State' : 'Ohio St.',
    'North Carolina' : 'U.N.C.',
    'Georgia State' : 'Georgia St.',
    'Virginia Commonwealth' : 'V.C.U.',
    'Oklahoma State' : 'Okla. St.',
    'North Carolina State' : 'N.C. State',
    'Iowa State' : 'Iowa St.',
    'Michigan State' : 'Mich. St.',
    'Louisiana State' : 'L.S.U.',
    'West Virginia' : 'W. Va.',
    'UCLA' : 'U.C.L.A.',
    'UC Irvine' : 'UC-Irvine',
    'Wichita State' : 'Wichita St.',
    'Boise State' : 'Boise St.',
    'New Mexico State' : 'N.M. State',
    'SMU' : 'S.M.U.',
    'San Diego State' : 'San Diego St.',
    'Eastern Washington' : 'E. Wash.',
    'Stephen F. Austin' : 'S. F. Austin',
    'North Dakota State' : 'N.D. St.',
    'UAB' : 'U.A.B.',
}

max_region_round = 5

num_champion_simulation_runs = 20000
desired_champion_simulation_runs = 1000

championship_string = '\n==========Championship==========\n'
region_string = '\n==========%s==========\n'

view_threshold = 0.01 # Percentages below this value will not be output

default_maximize_score_runs = 5000000

# Classes

# Class to define how many points are earned per correct pick per round
class RoundScores:
    def __init__(self, read_nyt_scores=False, cbs_scoring=True):
        # This dictionary is used to calculate the expected score of a bracket in leagues where
        #  additional points are awarded for correct picks in later rounds. Each key corresponds
        #  to the number of a round (see round_dictionary) above, and each value corresponds to
        #  the weight for each correct pick in that round. For example, a key/value pair of
        #  3:2 would mean that a correct pick in the third round is worth twice as much as the baseline

        # This feature was designed to work well with CBS Sportsline leagues, feel free to add more
        #  support for other sites
        self.default_cbs_scores = {
            2:1,
            3:2,
            4:3,
            5:4,
            6:6,
            7:8
        }

        if read_nyt_scores:
            self.team_scores = {}
            with open(default_nyt_file, 'r') as f:
                for line in f:
                    data = line.strip().split(',')
                    self.team_scores[ data[0] ] = {
                        2 : float(data[1]),
                        3 : float(data[2]),
                        4 : float(data[3]),
                        5 : float(data[4]),
                        6 : float(data[5]),
                        7 : float(data[6]),
                    }
            self.nyt_scores_loaded = True
        else:
            self.nyt_scores_loaded = False

        self.cbs_scoring = cbs_scoring

    def probability_of_victory(self, round_num, this_winner, team1, team2):
        return this_winner[round_num] / (team1[round_num] + team2[round_num])

    def get_score(self, round_num, this_winner, team1, team2):
        if self.nyt_scores_loaded:
            team_name = this_winner.name
            if team_name not in self.team_scores:
                if team_name in map_to_nyt_names:
                    team_name = map_to_nyt_names[team_name]
                else:
                    raise Exception("Couldn't map team name: %s" % team_name)
            return self.team_scores[team_name][round_num]

        elif self.cbs_scoring:
            return self.probability_of_victory(round_num, this_winner, team1, team2) * ( self.default_cbs_scores[round_num] + this_winner.seed )

        else:
            raise Exception("No scoring method set")

    def __getitem__(self, tup):
        raise Exception("Outdated, remove")
        if self.nyt_scores_loaded:
            pass
        elif self.cbs_scoring:
            team_name, round_num = tup
            return self.default_scores[round_num]
        else:
            raise Exception("No scoring method set")

# Output helper class
class Reporter:
    def __init__(self,task):
        self.start = time.time()
        self.lastreport = self.start
        self.task = task
        self.report_interval = 1 # Interval to print progress (seconds)
        print 'Starting ' + task
    def report(self,n):
        t = time.time()
        if self.lastreport < (t-self.report_interval):
            self.lastreport = t
            sys.stdout.write("  Completed: " + str(n) + " simulation runs\r" )
            sys.stdout.flush()
    def done(self):
        print 'Done %s, took %.3f seconds\n' % (self.task, time.time()-self.start)

# Class to generalize multiprocessing pools
class MultiWorker:
    def __init__(self, task, func, custom_cb_func=None):
        self.reporter = Reporter(task)
        self.func = func
        self.pool = Pool()
        self.results_dict = {}
        self.custom_cb_func = custom_cb_func
        if custom_cb_func == None:
            self.custom_cb_enabled = False
        else:
            self.custom_cb_enabled = True
        self.count = 0
    def cb(self, t):
        self.results_dict[ t[0] ] = t[1]
        self.reporter.report( len(self.results_dict) )
    def custom_cb(self, t):
        self.count += 1
        self.reporter.report(self.count)
        self.custom_cb_func(t)
    def addJob(self, argsTuple):
        if self.custom_cb_enabled:
            self.pool.apply_async(self.func, argsTuple, callback=self.custom_cb)
        else:
            self.pool.apply_async(self.func, argsTuple, callback=self.cb)
    def finishJobs(self):
        self.pool.close()
        self.pool.join()
        self.reporter.done()
        return self.results_dict

class Team:
    # Stores information on a team's probabilities of advancing to each round
    def __init__(self, name, region, seed, round_odds, conditional_round_odds, seed_slot):
        self.region = region
        self.seed = seed
        self.name = name
        self.round_odds = round_odds
        self.conditional_round_odds = conditional_round_odds

        # If a team beats a higher seed, this variable is used
        #  to store the higher team's seed
        self.seed_slot = seed_slot

    @classmethod
    def init_from_line(cls, team_line):
        line_data = team_line.split(',')
        region = line_data[0]
        seed = int(line_data[1])
        name = line_data[2]
        round_odds = []
        for item in line_data[3:10]:
            if item == '':
                round_odds.append(None)
            elif item == '<0.1':
                round_odds.append(.00001)
            else:
                round_odds.append(float(item))

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

        return cls(name, region, seed, round_odds, conditional_round_odds, seed)

    def copy(self):
        # Returns all duplicate (should be non-mutated) objects, except for resets seed slot
        return Team(self.name, self.region, self.seed, self.round_odds, self.conditional_round_odds, self.seed)

    def __getitem__(self, i):
        return self.round_odds[i-1] # Rounds are indexed from 1; lists from 0

    def __repr__(self):
        return self.name

    def __lt__(self, other):
         return self.seed < other.seed

    def reset_seed_slot(self):
        self.seed_slot = self.seed

class MaximizeScoreResults:
    def __init__(self):
        self.best_bracket_score = 0.0
        self.best_bracket = None

    def cb(self, tup):
        run_number,bracket = tup
        if bracket.expected_score > self.best_bracket_score:
            with open('tmp-best-bracket.txt', 'w') as f:
                f.write( bracket.simulation_string() )
            self.best_bracket = bracket
            self.best_bracket_score = bracket.expected_score
            print '\nFound new high score %.3f' % (self.best_bracket_score)

class SimulateDesiredChampionResults:
    def __init__(self):
        self.region_counts = {}
        self.finalist_counts = {}

    def cb(self, tup):
        run_number,results_bracket = tup
        for region in results_bracket:
            region_name = str(region)
            if region_name not in self.region_counts:
                self.region_counts[region_name] = {}
            for round_number in region.teams_by_round:
                if round_number not in self.region_counts[region_name]:
                    self.region_counts[region_name][round_number]={}
                for team in region.teams_by_round[round_number]:
                    team = str(team)
                    if team not in self.region_counts[region_name][round_number]:
                        self.region_counts[region_name][round_number][team] = 0
                    self.region_counts[region_name][round_number][team] += 1

        for team in results_bracket.finalists:
            team = str(team)
            if team not in self.finalist_counts:
                self.finalist_counts[team] = 0
            self.finalist_counts[team] += 1

    def __repr__(self):
        return_string = ''
        for region in self.region_counts:
            return_string += region_string % (str(region))
            for round_number in xrange(1, max_region_round+1):
                return_string += '\n%s:\n' % (round_dictionary[round_number])
                team_percentages = []
                for team in self.region_counts[region][round_number]:
                    team_percentages.append( (float(self.region_counts[region][round_number][team]) / float(desired_champion_simulation_runs),team) )
                team_percentages.sort(reverse=True)
                for percentage,team in team_percentages:
                    if percentage >= view_threshold:
                        return_string += '%s: %.1f%%\n'%(team, percentage*100)

        return_string += championship_string
        team_percentages = []
        for team in self.finalist_counts:
            team_percentages.append( (float(self.finalist_counts[team] / float(desired_champion_simulation_runs)),team) )
        team_percentages.sort(reverse=True)
        for percentage,team in team_percentages:
            if percentage >= view_threshold:
                return_string += '%s: %.1f%%\n' % (team, percentage*100)
        return return_string
        

class Region:
    # Stores a region of the bracket and all team data for that region
    def __init__(self, name, teams, teams_by_round, expected_score, round_scores):
        self.name = name
        self.teams = teams
        # After simulation, this dictionary stores the teams that are in each round
        #  Key: round number
        #  Value: list of team objects
        self.teams_by_round = teams_by_round
        self.expected_score = expected_score
        self.round_scores = round_scores

    @classmethod
    def init_empty(cls, name, round_scores):
        return cls(name, [], {}, 0.0, round_scores)

    def copy(self):
        # Does not copy simulation results stored in teams_by_round
        teams = []
        for team in self.teams:
            teams.append(team.copy())
        return Region(self.name, teams, {}, 0.0, self.round_scores)

    def __repr__(self):
        return self.name

    def append(self, team):
        self.teams.append(team)

    def __iter__(self):
        return self.teams.__iter__()

    def sort(self):
        self.teams.sort(key=operator.attrgetter('seed'))

    def reset(self):
        self.expected_score = 0.0
        for team in self.teams:
            team.reset_seed_slot()

    def simulate(self, desired_champion=None):
        self.teams_by_round = {}
        self.reset()
        # Simulate beginning of round of 32 by removing duplicate seeds (first 4)
        round2_teams = {}
        for team in self.teams:
            if team.seed not in round2_teams:
                round2_teams[team.seed] = []
            round2_teams[team.seed].append(team)
        for seed in round2_teams:
            teams = round2_teams[seed]
            if len(teams) == 2:
                round2_teams[seed] = [ pick_winner(teams[0], teams[1], 1) ]
                if desired_champion!=None and (str(teams[0])==desired_champion or str(teams[1])==desired_champion):
                    if str(round2_teams[seed]) != desired_champion:
                        # Abort and restart this run
                        self.simulate()
                        return
            elif len(teams) != 1:
                raise Exception('Incorrect number of teams for seed %d' % (seed))
        self.teams_by_round[1] = [i[0] for i in round2_teams.values()]

        # Now iterate through the rest of the rounds
        for round_number in xrange(2, max_region_round+1):
            this_round_teams = []

            prev_round_teams = self.teams_by_round[round_number-1]
            prev_round_teams.sort( key=operator.attrgetter('seed_slot') )
            if len(prev_round_teams)%2 != 0:
                raise Exception('Not an even number of teams')
            half_num_teams = len(prev_round_teams)/2

            high_seeds = prev_round_teams[:half_num_teams]
            low_seeds = prev_round_teams[half_num_teams:]
            low_seeds.sort(key=operator.attrgetter('seed_slot'), reverse=True)

            for team1, team2 in zip(high_seeds, low_seeds):
                this_winner = pick_winner(team1, team2, round_number)
                if desired_champion != None and (str(team1)==desired_champion or str(team2)==desired_champion):
                    if str(this_winner) != desired_champion:
                        # Abort and restart this run
                        self.simulate()
                        return
                # Calculate the expected value for this victory by using the probability it actually
                # happens and by using a scoring scheme where you get the number of points you get
                # is the same as the seed of the winner + the round multiplier.
                # This is how CBS sportsline leagues I've used work.
                self.expected_score += self.round_scores.get_score(round_number, this_winner, team1, team2)
                this_round_teams.append(this_winner)
            
            this_round_teams.sort(key=operator.attrgetter('seed'))
            self.teams_by_round[round_number] = this_round_teams
            
class Bracket:
    # Represents bracket and stores all region and team data
    def __init__(self, regions, finalists, champion, expected_score, round_scores):
        self.regions = regions
        self.finalists = finalists
        self.champion = champion
        self.expected_score = expected_score

        self.round_scores = round_scores

    @classmethod
    def fromfile(cls, bracket_file, round_scores):
        regions, team_names = read_input(bracket_file, round_scores)

        return cls(regions, None, None, 0.0, round_scores)

    def copy(self):
        regions={}
        for region in self.regions:
            regions[region] = self.regions[region].copy()
        finalists = None
        if self.finalists != None:
            finalists = []
            for finalist in self.finalists:
                finalists.append(finalist.copy())
        champion = None
        if self.champion != None:
            champion = self.champion.copy()
        return Bracket(regions, finalists, champion, 0.0, self.round_scores)

    def __iter__(self):
        return self.regions.values().__iter__()
  
    def simulate_champion(self, desired_champion, strict_mode):
        self.simulate_for_champion(desired_champion, strict_mode)
        while str(self.champion) != desired_champion:
            self.simulate_for_champion(desired_champion, strict_mode)
        return self

    def simulate_for_champion(self, desired_champion, strict_mode):
        midwest = None
        south = None
        east = None
        west = None
        desired_champion_region = None
        desired_champion_team = None
        # Find each region winner
        for region in self.regions.values():
            desired_champion_in_region = False
            for team in region:
                if desired_champion == str(team):
                    desired_champion_in_region = True
                    desired_champion_region = str(region)
                    desired_champion_team = team
                    break

            region.simulate()
            region_winner = region.teams_by_round[5][0]
            if desired_champion_in_region:
                while( str(region_winner)!=desired_champion ):
                    region.simulate(desired_champion)
                    region_winner = region.teams_by_round[5][0]

            if region.name == 'Midwest':
                midwest = region_winner
            elif region.name == 'South':
                south = region_winner
            elif region.name == 'East':
                east = region_winner
            elif region.name == 'West':
                west = region_winner
            else:
                raise Exception ('Region "%s" not recognized' % (region.name) )
        
        # Then matchup region winners
        if desired_champion_region == 'Midwest':
            if not strict_mode:
                finalist_1 = midwest
            else:
                finalist_1 = pick_winner(midwest, west, 6)
                while( str(finalist_1) != desired_champion ):
                    self.regions['West'].simulate()
                    west = self.regions['West'].teams_by_round[5][0]
                    finalist_1 = pick_winner(midwest,west,6)
        elif desired_champion_region == 'West':
            if not strict_mode:
                finalist_1 = west
            else:
                finalist_1 = pick_winner(midwest, west, 6)
                while( str(finalist_1) != desired_champion ):
                    self.regions['Midwest'].simulate()
                    midwest = self.regions['Midwest'].teams_by_round[5][0]
                    finalist_1 = pick_winner(midwest, west, 6)
        else:
            finalist_1 = pick_winner(midwest, west, 6)

        if desired_champion_region == 'South':
            if not strict_mode:
                finalist_2 = south
            else:
                finalist_2 = pick_winner(south, east, 6)
                while( str(finalist_2) != desired_champion ):
                    self.regions['East'].simulate()
                    east = self.regions['East'].teams_by_round[5][0]
                    finalist_2 = pick_winner(south, east, 6)
        elif desired_champion_region == 'East':
            if not strict_mode:
                finalist_2 = east
            else:
                finalist_2 = pick_winner(south, east, 6)
                while( str(finalist_2) != desired_champion ):
                    self.regions['South'].simulate()
                    south = self.regions['South'].teams_by_round[5][0]
                    finalist_2 = pick_winner(south,east,6)
        else:
            finalist_2 = pick_winner(south, east, 6)

        self.finalists = [finalist_1, finalist_2]
        # Now pick a champion
        if strict_mode:
            champion = pick_winner(finalist_1, finalist_2, 7)
            while( str(champion) != desired_champion ):
                if desired_champion == str(finalist_1):
                    self.regions['South'].simulate()
                    south = self.regions['South'].teams_by_round[5][0]
                    self.regions['East'].simulate()
                    east = self.regions['East'].teams_by_round[5][0]
                    finalist_2 = pick_winner(south, east, 6)
                    champion = pick_winner(finalist_1, finalist_2, 7)
                else:
                    self.regions['Midwest'].simulate()
                    midwest = self.regions['Midwest'].teams_by_round[5][0]
                    self.regions['West'].simulate()
                    west = self.regions['West'].teams_by_round[5][0]
                    finalist_1 = pick_winner(midwest, west, 6)
                    champion = pick_winner(finalist_1, finalist_2, 7)
            self.champion = champion
        else:
            self.champion = desired_champion_team
        self.midwest = midwest
        self.south = south
        self.east = east
        self.west = west
        self.set_expected_score()

    def set_expected_score(self):
        self.expected_score = 0.0

        # Finalist 1 winning
        self.expected_score += self.round_scores.get_score(6, self.finalists[0], self.midwest, self.west)
        # self.expected_score += ( self.finalists[0][6] / (self.midwest[6]+self.west[6])) * (self.round_scores[(self.finalists[0].name, 6)] + self.finalists[0].seed)
        # Finalist 2 winning
        self.expected_score += self.round_scores.get_score(6, self.finalists[1], self.south, self.east)
        # self.expected_score += ( self.finalists[1][6] / (self.south[6]+self.east[6])) * (self.round_scores[(self.finalists[1].name, 6)]+self.finalists[1].seed)
        # Champion winning
        self.expected_score += self.round_scores.get_score(7, self.champion, self.finalists[0], self.finalists[1])
        # self.expected_score += ( self.champion[7] / (self.finalists[0][7]+self.finalists[1][7]))*(self.round_scores[(self.champion.name, 7)]+self.champion.seed )
        for region in self.regions.values():
            self.expected_score += region.expected_score
  
    def simulate(self):
        midwest = None
        south = None
        east = None
        west = None
        # Find each region winner
        for region in self.regions.values():
            region.simulate()
            region_winner = region.teams_by_round[5][0]
            if region.name == 'Midwest':
                midwest = region_winner
            elif region.name == 'South':
                south = region_winner
            elif region.name == 'East':
                east = region_winner
            elif region.name == 'West':
                west = region_winner
            else:
                raise Exception( 'Region "%s" not recognized' % (region.name) )
        
        # Then matchup region winners
        finalist_1 = pick_winner(midwest, west, 6)
        finalist_2 = pick_winner(south, east, 6)
        self.finalists = [finalist_1, finalist_2]

        # Now pick a champion
        self.champion = pick_winner(finalist_1, finalist_2, 7)
        
        self.midwest = midwest
        self.south = south
        self.east = east
        self.west = west
        self.set_expected_score()

    def simulation_string(self):
        return_string = 'Score: %f\n' % self.expected_score
        # First, build each region
        for region in self.regions.values():
            return_string += region_string % (str(region))
            for round_number in xrange(1, max_region_round+1):
                return_string += '\n%s:\n' % (round_dictionary[round_number])
                for team in region.teams_by_round[round_number]:
                    return_string += '%s\n' % (str(team))

        # Build up finals
        return_string += championship_string
        for team in self.finalists:
            return_string += '%s\n' % (team)
        return_string += '\nChampion: %s\n' % (self.champion)
        return return_string

    def results(self):
        return (self.regions, self.finalists, self.champion)

# Functions

def read_input(bracket_file, round_scores):
    regions = {}
    team_names = set()
    with open(bracket_file, 'r') as f:
        lines = f.readlines()

        # Check for correct header line
        header_line = lines[0].strip()
        if header_line != header_string:
            print header_line
            print header_string
            raise Exception("Header line doesn't match expected format")

        # Read in team data
        for line in lines[1:]:
            team = Team.init_from_line(line.strip())
            team_names.add(team.name)
            if team.region not in regions:
                regions[team.region] = Region.init_empty(team.region, round_scores)

            regions[team.region].append(team)

        # Sort each region list of teams by seed
        for region in regions.values():
            region.sort()

    return (regions, team_names)

def pick_winner(team1, team2, round_number):
    team1_prob = team1[round_number]
    team2_prob = team2[round_number]
    odds_range = team1_prob+team2_prob
    num = random.uniform(0, odds_range)
    if num <= team1_prob:
        if team2.seed_slot < team1.seed_slot:
            team1.seed_slot = team2.seed_slot
        return team1
    else:
        if team1.seed_slot < team2.seed_slot:
            team2.seed_slot = team1.seed_slot
        return team2

def simulate_desired_champion(run_number, original_bracket, desired_champion, strict_mode):
    bracket = original_bracket.copy()
    return (run_number, bracket.simulate_champion(desired_champion, strict_mode) )

def simulate_max_score(run_number, original_bracket):
    bracket = original_bracket.copy()
    bracket.simulate()
    return (run_number, bracket)

def predictor():
    # Setup argument parser
    parser = argparse.ArgumentParser(description=program_description)
    parser.add_argument('-i', '--input',
                        default = default_input_file,
                        help = "Input data file to read in. Header must be in a specific format")
    parser.add_argument('-o', '--output',
                        default = default_output_file,
                        help = "File to save output")
    parser.add_argument('-c', '--champion_mode',
                        action = 'store_true',
                        default = False,
                        help = "Does many simulations and prints out odds of each team being champion")
    parser.add_argument('-q', '--quiet',
                        action = 'store_true',
                        default = False,
                        help = "Doesn't print bracket output to terminal")
    parser.add_argument('--nyt',
                        action = 'store_true',
                        default = False,
                        help = "Load scores from NYT data")
    parser.add_argument('-m', '--maximize_score',
                        action = 'store_true',
                        default = False,
                        help = "Finds bracket with maximum expected score")
    parser.add_argument('--maximize_score_runs',
                        type = int,
                        default = default_maximize_score_runs,
                        help = "Number of random brackets to generate when attempting to maximize score")

    find_champion_group = parser.add_mutually_exclusive_group()
    find_champion_group.add_argument('-l', '--loose_find_champion',
                        default = None,
                        help = "Runs the simulation until the specified champion is found. Assumes that the desired team will win in the semifinals and championship game.")
    find_champion_group.add_argument('-s', '--strict_find_champion',
                        default = None,
                        help = "Runs the simulation until the specified champion is found. Does not assume that desired team will win in the seminfinals or championship game.")

    args = parser.parse_args()

    if args.nyt:
        round_scores = RoundScores( read_nyt_scores=True )
    else:
        round_scores = RoundScores()

    if args.champion_mode:
        bracket=Bracket.fromfile(args.input, round_scores)
        champions={}
        for i in xrange(0, num_champion_simulation_runs):
            bracket.simulate()
            if bracket.champion not in champions:
                champions[bracket.champion] = 0
            champions[bracket.champion] += 1
            
        output_list = [ (champions[champion], str(champion)) for champion in champions]
        output_list.sort(reverse=True)
        
        print 'Percent chance of winning tournament:'
        for num_wins, name in output_list:
            win_percent = float(num_wins)*100 / float(num_champion_simulation_runs)
            if win_percent >= (view_threshold * 100):
                print '  %s: %.1f%%' % (name, win_percent)
        
        return 0

    if args.loose_find_champion != None or args.strict_find_champion != None:
        strict_mode = False
        desired_champion = args.loose_find_champion
        if args.strict_find_champion != None:
            strict_mode = True
            desired_champion = args.strict_find_champion
        print 'Desired champion: %s' % (desired_champion)
        bracket = Bracket.fromfile(args.input, round_scores)

        results = SimulateDesiredChampionResults()

        print 'Simulation will stop after %d runs generate desired champion' % (desired_champion_simulation_runs)
        w = MultiWorker('running desired champion simulations', simulate_desired_champion, results.cb)

        for x in xrange(1, desired_champion_simulation_runs+1):
            w.addJob( (x, bracket, desired_champion, strict_mode) )

        w.finishJobs()

        result_string = str(results)

        if not args.quiet:
            print result_string

        with open(args.output, 'w') as f:
            f.write(result_string)

        return 0

    if args.maximize_score:
        bracket = Bracket.fromfile(args.input, round_scores)
        print 'Simulation will stop after %d runs' % (args.maximize_score_runs)
        
        results = MaximizeScoreResults()
        
        w = MultiWorker('running maximize score simulations', simulate_max_score, results.cb)

        for x in xrange(1, args.maximize_score_runs+1):
            w.addJob( (x,bracket) )

        w.finishJobs()

        sim_string = results.best_bracket.simulation_string()
        if not args.quiet:
            print sim_string
            print 'Expected score: %.3f' % (results.best_bracket_score)

        with open(args.output, 'w') as f:
            f.write('Expected score: %.3f\n' % (results.best_bracket_score))
            f.write(sim_string)

        return 0

    bracket = Bracket.fromfile(args.input, round_scores)
    bracket.simulate()
    sim_string = bracket.simulation_string()
    if not args.quiet:
        print sim_string

    with open(args.output, 'w') as f:
        f.write(sim_string)

    return 0

# Main function
if __name__ == "__main__":
    sys.exit( predictor() )
