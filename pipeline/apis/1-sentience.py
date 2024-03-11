#!/usr/bin/env python3

"""
This module contains a function that
uses Swapi API to get the list of names of
all sentinent species in the Star Wars universe"""

import requests


def sentientPlanets():
    """return list of names of all sentient species"""
    url = "https://swapi-api.alx-tools.com/api/species/"
    response = requests.get(url)
    data = response.json()
    species = []
    while response.status_code == 200:
        for result in data['results']:
            if result['designation'] == "sentient" or\
                    result['classification'] == "sentient":
                if result['homeworld'] is None:
                    continue
                planets = requests.get(result['homeworld'])
                name = planets.json()['name']
                species.append(name)
        try:
            response = requests.get(response.json()["next"])
        except Exception:
            break
    return species
