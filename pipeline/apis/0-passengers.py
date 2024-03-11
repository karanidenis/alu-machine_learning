#!/usr/bin/env python3

"""
This module contains a function that
uses Swapi API to get the list of ships that
can hold given number of passengers"""


def availableShips(passengerCount):
    """return empty list if no ships carry that no.
    of passengers else return list of ships"""
    import requests
    url = "https://swapi-api.alx-tools.com/api/starships/"
    response = requests.get(url)
    data = response.json()
    ships = []
    # print(data["results"])
    # for result in data['results']:
    #     if result['passengers'] != "n/a":
    #         passengers_no = int(result['passengers'].replace(',', ''))
    #         # print(passengers_no)
    #         if passengers_no == passengerCount:
    #             ships.append(result['name'])
    while response.status_code == 200:
        # res = res.json()
        for ship in data['results']:
            passengers = ship['passengers'].replace(',', '')
            try:
                if int(passengers) >= passengerCount:
                    ships.append(ship['name'])
            except ValueError:
                pass
        try:
            data = requests.get(data['next'])
        except Exception:
            break

    return ships
