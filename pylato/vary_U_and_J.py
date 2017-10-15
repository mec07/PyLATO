#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Friday, July 24, 2015

@auther: Marc Coury

My way of generating the results that I want.

What I want:
    * Vary values of U and J and print out the following for each one
        * Occupations written to files
        * Magnetic correlation written to files
    * Make a magnetic correlation phase diagram of the U and J.

To vary U and J I have to mess with the model.

"""
import numpy as np
import commentjson
import graphs
from verbosity import verboseprint
import pdb

# The steps:
#   * Loop over U and J
#       * Alter U and J in the model.
#       * Run the code and store the mag corr value.
#   * Make the plot of the magnetic correlation phase diagram.
for ii in range(3):
    graphs.make_magmomcorr_graphs(ii)
