#!/usr/bin/python
# Requires Python 2.7
# and Beautiful Soup 4

"""
March Madness data parsing script
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

#################
Instructions:
This script expects a file named 'nyt_html.dump' to exist containing the NYT output
The relevant html can be extracted from fivethirtyeight_url using a browser
(e.g. in Chrome select "inspect element" on the data table)

My dump started with this tag:
<table> under "Current points for correct upset picks"

"""

# Python import statements
import argparse
import os
import sys
from bs4 import BeautifulSoup
import urllib
import re
from predict import header_string

# Constants
program_description = 'Python script to update data.csv with updated data from html dump from nytimes.com'

nyt_url = 'http://www.nytimes.com/interactive/2015/03/15/sports/ncaabasketball/an-ncaa-bracket-for-risk-takers.html'

def parse_number(s):
    if s.endswith('k'):
        return float(s[:-1]) * 1000.0
    else:
        return float(s)

def main():
    
    soup = BeautifulSoup(open('nyt_html.dump', 'r'))

    tr_tags = soup.find_all('tr')

    output_data = {}
    for team in tr_tags:
        names = str(team.find_all('td', attrs={'class':'g-team-name'})[0].contents[1]).strip().split('/')

        for name in names:
            points_per_round = {}
            for round_num in xrange(1, 7):
                # NYT round starts from round 2, so we add 1
                points_per_round[round_num+1] = parse_number( team.find_all('td', attrs={'class':'g-num num-%d' % round_num})[0].contents[0] )

            output_data[name] = points_per_round

    with open('nyt_scoring_data.csv', 'w') as f:
        for name in output_data:
            f.write(
                '%s,%f,%f,%f,%f,%f,%f\n' % (
                    name,
                    output_data[name][2],
                    output_data[name][3],
                    output_data[name][4],
                    output_data[name][5],
                    output_data[name][6],
                    output_data[name][7],                    
                )
            )

if __name__ == "__main__":
    main()
