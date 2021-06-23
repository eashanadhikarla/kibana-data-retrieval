'''
**
** Project Lead: Eashan Adhikarla
** Mentor: Ezra Kissel
** 
** Date Created: June 17' 2021
** Last Modified: June 22' 2021 
**
'''

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

import pandas as pd
import requests
import json

# Create the elasticsearch client
HOST = 'nersc-tbn-6.testbed100.es.net'
PORT = 9200

es = Elasticsearch(host=HOST, port=PORT)


class bcolors:
    """
    Defining colors for the print syntax coloring
    """
    HEADER    = '\033[35m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[36m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'


class GETTER:
    """
    Get class to get current information about different Indexs
    """
    def __init__(self, term):
        self.term = term

    def getIndexList(self):
        indices_dict = es.indices.get_alias(self.term)
        if isinstance(indices_dict, dict) and indices_dict is not None:
            print()
            print(f"--------------------\n'{bcolors.OKGREEN}{len(indices_dict)}{bcolors.ENDC}' indexes found!\n--------------------")
            print()
            return indices_dict
        else:
            print (f"{bcolors.FAIL}Empty dict!{bcolors.ENDC}")

    # def 


indexTypes = ["*", "iperf*", "jobmeta*", "bbrmon*"]
term_ = indexTypes[0]

get = GETTER(term_)
indexes = get.getIndexList()
print("Print collected indexes ...")
for i in indexes:
    print(i)


print("\n\n")
response = es.get(index="iperf3-2021.06.06", id="uhKs33kBkImc33Nl6K8P")
print(response)