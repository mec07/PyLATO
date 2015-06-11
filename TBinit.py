"""
Created on Sunday April 19, 2015

@author: Andrew Horsfield and Marc Coury

This module carries out the needed initialisation tasks
"""
#
# Import modules
import TBgeom
import TBH
import TBelec
import commentjson


def init():
    """Initialise the program."""
    #
    # Set up variables that define the job in a dictionary
    with open("JobDef.json",'r') as inputfile:
        JobDef = commentjson.loads(inputfile.read())
        
    # Initialise the geometry
    TBgeom.init(JobDef)
    #
    # Initialise the Hamiltonian module
    TBH.init(JobDef)
    #
    # Initialise the electron module
    TBelec.init(JobDef)
    #
    # Return the job definition
    return JobDef
