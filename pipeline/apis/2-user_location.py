#!/usr/bin/env python3

"""
This module contains a function that
uses Github API to print location of
specific users"""

import requests
import sys

# get the url from string passed on the terminal


def print_location():
    """print location of user"""
    url = sys.argv[1]
    response = requests.get(url)
    data = response.json()
    if data:
        if 'location' in data and data['location']:
            print(data['location'])
        else:
            print("Not found")
    else:
        print("Not found")


if __name__ == "__main__":
    print_location()
