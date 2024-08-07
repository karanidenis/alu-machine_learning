#!/usr/bin/env python3

"""
function that changes all topics of a 
school document based on the name
"""


def update_topics(mongo_collection, name, topics):
    """
    mongo_collection - pymongo collection object
    name - school name to be updated
    topiics - list - list of topics approached in school
    """
    mongo_collection.update_many({'name': name}, {'$set': {'topics': topics}})
