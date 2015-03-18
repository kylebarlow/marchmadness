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
This script expects a file named 'html.dump' to exist containing the 538 output
The relevant html can be extracted from fivethirtyeight_url using a browser
(e.g. in Chrome select "inspect element" on the data table)

My dump started with this tag:
<tbody>

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
program_description = 'Python script to update data.csv with updated data from html dump from fivethirtyeight.com'

fivethirtyeight_url = 'http://fivethirtyeight.com/interactives/march-madness-predictions-2015/#mens'

def main():
    
    soup = BeautifulSoup(open('html.dump', 'r'))

    tr_tags = soup.find_all('tr')

    output_data = []
    for team in tr_tags:
        region = str(team.find_all('td', attrs={'class':'region'})[0].contents[0])
        seed = team.find_all('td', attrs={'class':'seed'})[0].contents[0]
        m = re.match('(\d+)([a-zA-z]+)', seed)
        if m:
            seed = int( m.group(1) )
        else:
            seed = int( seed )

        name = str(team.find_all('td', attrs={'class':'team-name'})[0].contents[0])
        number_wins = len(team.find_all('span', attrs={'class':'win'}))
        
        probability_tags = team.find_all('div', attrs={'data-placement':"bottom", 'class':"", 'title':''})

        dead_tags = team.find_all('div', attrs={'data-placement':"bottom", 'class':"dead", 'title':''})
        if len(dead_tags) == 7:
            # Team already lost
            continue

        # Parsing check - number of win tags plus number of tags with probabilities
        # should equal number of rounds in the tournament (7)
        # print team
        assert( number_wins + len(probability_tags) == 7 )

        # Pull out most precise probability available
        probabilities = [ float(tag['data-original-title'][:-1])*.01 for tag in probability_tags ]
        minned_probabilities = []
        for p in probabilities:
            if p == 0.0:
                minned_probabilities.append( '0.0000001' )
            else:
                minned_probabilities.append( str(p) )
        
        # Add on wins to beginning of list
        for x in xrange(0, number_wins):
            minned_probabilities.insert(0, '')
        assert( len(minned_probabilities) == 7 )

        print region, seed, name
        l = [region, seed, name]
        l.extend( minned_probabilities )
        output_data.append( l )

    with open('data.csv', 'w') as f:
        f.write( header_string + '\n' )
        for d in output_data:
            assert( len(d) == 10 )
            f.write( '%s,%d,%s,%s,%s,%s,%s,%s,%s,%s\n' % tuple(d) )

if __name__ == "__main__":
    main()
