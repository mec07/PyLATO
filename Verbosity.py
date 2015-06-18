#!/usr/bin/python
"""
Created some time in 2014

@author: Marc Coury

This module prints if verbosity is turned on (i.e. 1) or does not if verbosity
is turned off (i.e. 0).
"""
def verboseprint(verbosity,*args):
	# if the verbosity is 1 then print args
	if verbosity==1:
		# Print each argument separately so caller doesn't need to put everything to be printed into a single string
		for stuff in args:
			print stuff,
		print
	# otherwise, don't print anything
	# else:
	# 	verboseprint = lambda *a: None