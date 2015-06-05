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
#
# Initialise the program
def init():
    #
    # Set up variables that define the job in a dictionary
    JobDef = {# atomic geometry
              'gy_file': 'geom.csv',
              # magnetic field
              'so_eB': (0.0, 0.01, 0.0),
              # electronic temperature in eV
              'el_kT': 0.025,
              # self-consistent field parameters
              'scf_on': 1, 'scf_mix': 0.001, 'scf_tol': 1.0e-10, 'scf_max_loops':500,
              # number of bins for the density of states
              'dos_nbin': 20,
              # Verbosity
              'verbose': 1, 'extraverbose':0}
    #
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

