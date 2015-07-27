#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Friday, July 24, 2015

@auther: Marc Coury

My quick and dirty way of generating the results that I want.

What I want:
    * Vary values of U and J and print out the following for each one
        * Occupations written to files
        * Magnetic correlation written to files
    * Make a magnetic correlation phase diagram of the U and J.

To vary U and J I have to mess with the model.

"""
import TB
import numpy as np
import commentjson
import shutil, os, sys
import myfunctions
from Verbosity import verboseprint

# The steps:
#   * Loop over U and J
#       * Alter U and J in the model.
#       * Run the code and store the mag corr value.
#   * Make the plot of the magnetic correlation phase diagram.
Verbose = 1
orb_type = "p"
numeperatom = 1
plotname = "Mag_Corr_"+orb_type+"_"+str(numeperatom)
op_sq_name="\\frac{1}{3} \langle :\hat{\mathbf{m}}_1.\hat{\mathbf{m}}_2:\\rangle"

U_min = 0
U_max = 10
U_num_steps = 10

J_min = 0
J_max = 5
J_num_steps = 10

dJ_min = 0
dJ_max = 1
dJ_num_steps = 10

U_array, U_step = np.linspace(U_min, U_max, num=U_num_steps, retstep=True)
# test
U_array = np.append(U_array,U_max+U_step)

if orb_type == "s":
    J_array = [0.0]
    J_step = 0.0
    dJ_array = [0.0]
    dJ_step = 0.0
elif orb_type == "p":
    J_array, J_step = np.linspace(J_min, J_max, num=J_num_steps, retstep=True)
    # test
    J_array = np.append(J_array,J_max+J_step)
    dJ_array = [0.0]
    dJ_step = 0.0
elif orb_type == "d":
    J_array, J_step = np.linspace(J_min, J_max, num=J_num_steps, retstep=True)
    dJ_array, dJ_step = np.linspace(dJ_min, dJ_max, num=dJ_num_steps, retstep=True)
    # test
    J_array = np.append(J_array,J_max+J_step)
    dJ_array = np.append(dJ_array,dJ_max+dJ_step)
else:
    print("ERROR: orb_type must be 's', 'p' or 'd'. Exiting. ")
    sys.exit()


jobdef_file = "JobDef.json"
jobdef_backup = "JobDef_backup.json"
# Make a backup of the JobDef file
shutil.copyfile(jobdef_file, jobdef_backup)
# Read in the JobDef file
with open(jobdef_file, 'r') as f:
    jobdef = commentjson.loads(f.read())


# Read in the model file
modelfile = "models/TBcanonical_"+orb_type+".json"
model_temp = "TBcanonical_"+orb_type+"_temp"
temp_modelfile = "models/"+model_temp+".json"
with open(modelfile, 'r') as f:
    model = commentjson.loads(f.read())
# Copy and paste the regular python model to one with the same temp name
model_python = "models/TBcanonical_"+orb_type+".py"
model_python_temp = "models/"+model_temp+".py"
shutil.copyfile(model_python, model_python_temp)

# change the model to the temp name in jobdef
jobdef["model"] = model_temp
# write jobdef back to file
with open(jobdef_file, 'w') as f:
    commentjson.dump(jobdef, f, sort_keys=True, indent=4, separators=(',', ': '))

# initialise the mag_corr dictionary
mag_corr = {}


for U in U_array:
    for J in J_array:
        for dJ in dJ_array:
            model["U"] = U
            model["NElectrons"] = numeperatom
            if orb_type == "p":
                model["J"] = J
            elif orb_type == "d":
                model["J"] = J
                model["dJ"] = dJ

            # write out the new modelfile
            with open(temp_modelfile, 'w') as f:
                commentjson.dump(model, f, sort_keys=True, indent=4, separators=(',', ': '))

            mag_corr[U, J, dJ] = TB.main()

# clean up temp files
os.remove(temp_modelfile)
os.remove(model_python_temp)
# restore backup of JobDef.json
shutil.copyfile(jobdef_backup, jobdef_file)
os.remove(jobdef_backup)

# Make the plot:
myfunctions.Plot_OpSq_U_J(Verbose,mag_corr,orb_type,plotname,U_min,U_step,U_num_steps,J_min,J_step,J_num_steps,dJ_min,dJ_step,dJ_num_steps,op_sq_name)

