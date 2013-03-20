#!/usr/bin/python
# Requires Python 2.7

"""
March Madness prediction script
Copyright (C) 2013 Kyle Barlow

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

# Constants
program_description='Python script to auto-generate "quick pick" march madness brackets from probability input (as in the format of, but not necessarily, the 538 data from Nate Silver)\nEach probability input is assumed to be built up conditionally'

default_input_file='data.csv'
default_output_file='output.txt'

# Expected header string
header_string='REGION,SEED,TEAM,FIRST FOUR,ROUND OF 32,ROUND OF 16,ELITE 8,FINAL 4,FINALS,CHAMPIONS'

# Mapping for strings describing each round to an integer (for indexing)
# Counting starts with 1
round_dictionary={
1:'FIRST FOUR',
2:'ROUND OF 32',
3:'ROUND OF 16',
4:'ELITE 8',
5:'FINAL 4',
6:'FINALS',
7:'CHAMPIONS'
}

max_region_round=5

num_champion_simulation_runs=20000

# Classes
class Team:
    # Stores information on a team's probabilities of advancing to each round
    def __init__(self,team_line):
        line_data=team_line.split(',')
        self.region=line_data[0]
        self.seed=int(line_data[1])
        self.team=line_data[2]
        round_odds=[]
        for item in line_data[3:10]:
            if item=='':
                round_odds.append(None)
            elif item=='<0.1':
                round_odds.append(.0001)
            else:
                round_odds.append(float(item))

        # Make the probabilities conditional
        self.round_odds=[]
        for i,odd in enumerate(round_odds):
            if i==0:
                self.round_odds.append(odd)
            else:
                prev_round=round_odds[i-1]
                if prev_round==None:
                    self.round_odds.append(odd)
                else:
                    self.round_odds.append(odd/prev_round)

        # If a team beats a higher seed, this variable is used
        #  to store the higher team's seed
        self.seed_slot=self.seed

    def __getitem__(self,i):
        return self.round_odds[i-1] # Rounds are indexed from 1; lists from 0

    def __repr__(self):
        return self.team

    def __lt__(self, other):
         return self.seed < other.seed

    def reset_seed_slot(self):
        self.seed_slot=self.seed

class Region:
    # Stores a region of the bracket and all team data for that region
    def __init__(self,name):
        self.name=name
        self.teams=[]
        # After simulation, this dictionary stores the teams that are in each round
        #  Key: round number
        #  Value: list of team objects
        self.teams_by_round={}

    def __repr__(self):
        return self.name

    def append(self,team):
        self.teams.append(team)

    def __iter__(self):
        return self.teams.__iter__()

    def sort(self):
        self.teams.sort(key=operator.attrgetter('seed'))

    def reset_seed_slots(self):
        for team in self.teams:
            team.reset_seed_slot()

    def simulate(self):
        self.teams_by_round={}
        self.reset_seed_slots()
        # Simulate beginning of round of 32 by removing duplicate seeds (first 4)
        round2_teams={}
        for team in self.teams:
            if team.seed not in round2_teams:
                round2_teams[team.seed]=[]
            round2_teams[team.seed].append(team)
        for seed in round2_teams:
            teams=round2_teams[seed]
            if len(teams)==2:
                round2_teams[seed]=[pick_winner(teams[0],teams[1],1)]
            elif len(teams)!=1:
                raise Exception('Incorrect number of teams for seed %d'%(seed))
        self.teams_by_round[1]=[i[0] for i in round2_teams.values()]

        # Now iterate through the rest of the rounds
        for round_number in xrange(2,max_region_round+1):
            this_round_teams=[]

            prev_round_teams=self.teams_by_round[round_number-1]
            prev_round_teams.sort(key=operator.attrgetter('seed_slot'))
            half_num_teams=len(prev_round_teams)/2

            high_seeds=prev_round_teams[:half_num_teams]
            low_seeds=prev_round_teams[half_num_teams:]
            low_seeds.sort(key=operator.attrgetter('seed_slot'),reverse=True)

            for team1,team2 in zip(high_seeds,low_seeds):
                this_round_teams.append(pick_winner(team1,team2,round_number))
            
            this_round_teams.sort(key=operator.attrgetter('seed'))
            self.teams_by_round[round_number]=this_round_teams
            
class Bracket:
    # Represents bracket and stores all region and team data
    def __init__(self,bracket_file):
        self.regions={}
        with open(bracket_file,'r') as f:
            lines=f.readlines()
            
            # Check for correct header line
            header_line=lines[0].strip()
            if header_line!=header_string:
                print header_line
                print header_string
                raise Exception("Header line doesn't match expected format")

            # Read in team data
            for line in lines[1:]:
                team=Team(line.strip())
                if team.region not in self.regions:
                    self.regions[team.region]=Region(team.region)
                    
                self.regions[team.region].append(team)
            
            # Sort each region list of teams by seed
            for region in self.regions.values():
                region.sort()

            # print self.regions
            # for region in self.regions.values():
            #     print '\n%s:\n'%(region)
            #     for team in region:
            #         print team
  
    def simulate_champion(self):
        self.simulate()
        return self.champion
  
    def simulate(self):
        midwest=None
        south=None
        east=None
        west=None
        # Find each region winner
        for region in self.regions.values():
            region.simulate()
            region_winner=region.teams_by_round[5][0]
            if region.name=='Midwest':
                midwest=region_winner
            elif region.name=='South':
                south=region_winner
            elif region.name=='East':
                east=region_winner
            elif region.name=='West':
                west=region_winner
            else:
                raise Exception('Region "%s" not recognized'%(region.name))
        
        # Then matchup region winners
        finalist_1=pick_winner(midwest,west,6)
        finalist_2=pick_winner(south,east,6)
        self.finalists=[finalist_1,finalist_2]

        # Now pick a champion
        self.champion=pick_winner(finalist_1,finalist_2,7)

    def simulation_string(self):
        return_string=''
        final_four=[]
        # First, build each region
        for region in self.regions.values():
            return_string+='\n==========%s==========\n'%(str(region))
            for round_number in xrange(1,max_region_round+1):
                return_string+='\n%s:\n'%(round_dictionary[round_number])
                for team in region.teams_by_round[round_number]:
                    return_string+='%s\n'%(str(team))

        # Build up finals
        return_string+='\n==========Championship==========\n'
        for team in self.finalists:
            return_string+='%s\n'%(team)
        return_string+='\nChampion: %s\n'%(self.champion)
        return return_string
            

# Functions

def pick_winner(team1,team2,round_number):
    team1_prob=team1[round_number]
    team2_prob=team2[round_number]
    odds_range=team1_prob+team2_prob
    num=random.uniform(0,odds_range)
    if num<=team1_prob:
        if team2.seed<team1.seed_slot:
            team1.seed_slot=team2.seed
        return team1
    else:
        if team1.seed<team2.seed_slot:
            team2.seed_slot=team1.seed
        return team2

def predictor():
    # Setup argument parser
    parser = argparse.ArgumentParser(description=program_description)
    parser.add_argument('-i','--input',
                        default=default_input_file,
                        help="Input data file to read in. Header must be in a specific format")
    parser.add_argument('-o','--output',
                        default=default_output_file,
                        help="File to save output")
    parser.add_argument('-c','--champion_mode',
                        action='store_true',
                        default=False,
                        help="Does many simulations and prints out odds of each team being champion")
    parser.add_argument('-f','--find_champion',
                        default=None,
                        help="Runs the simulation until the specified champion is found")

    args=parser.parse_args()

    if args.champion_mode:
        bracket=Bracket(args.input)
        champions={}
        for i in xrange(0,num_champion_simulation_runs):
            bracket.simulate()
            if bracket.champion not in champions:
                champions[bracket.champion]=0
            champions[bracket.champion]+=1
            
        output_list=[(champions[champion],str(champion)) for champion in champions]
        output_list.sort(reverse=True)
        
        print 'Percent chance of winning tournament:'
        for num_wins,name in output_list:
            win_percent=float(num_wins)*100/float(num_champion_simulation_runs)
            if win_percent>=1:
                print '  %s: %.1f%%'%(name,win_percent)
        
        return 0

    if args.find_champion!=None:
        desired_champion=args.find_champion
        print 'Desired champion: %s'%(desired_champion)
        bracket=Bracket(args.input)
        champion=bracket.simulate_champion()
        
        while (True):
            if str(champion)==desired_champion:
                break
            champion=bracket.simulate_champion()
        
        print bracket.simulation_string()
        return 0

    bracket=Bracket(args.input)
    bracket.simulate()
    print bracket.simulation_string()

    return 0

# Main function
if __name__ == "__main__":
    sys.exit(predictor())
