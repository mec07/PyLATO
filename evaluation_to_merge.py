#!/usr/bin/python
from Verbosity import *
import ipdb

def evaluate_response(verbose,stimulus_store,response):
    # Check to make sure that the response doesn't take you outside bounds
    if response == 0:
        return True
    if len(stimulus_store)>0:
        last_stimulus=stimulus_store[-1]
    else:
        return False
    if (response+1)>len(stimulus_store):
        answer=False
    # Check to see if the response is correct
    elif (stimulus_store[-(response+1)]==last_stimulus):
        answer=True
    else:
        answer=False
        
    verboseprint(verbose,"Answer is: "+str(answer))
    return answer