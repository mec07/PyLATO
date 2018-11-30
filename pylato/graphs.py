#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Monday, July 27, 2015

@author: Marc Coury

A few useful functions to make graphs from PyLATO
"""
# Avoid errors when running matplotlib over ssh without X forwarding.
# Must be before importing matplotlib.pyplot or pylab!
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
import shutil, os, sys
import numpy as np
import commentjson

from pylato.main import main


def mag_corr_loop(U_array, J_array, dJ_array, jobdef, jobdef_file, model, temp_modelfile, orb_type, number_decimals):
    """
    Function mag_corr_loop is designed to run over the U, J and dJ values to
    fill up the mag_corr dictionary.

    INPUTS          TYPE        DESCRIPTION

    U_array         nparray     All the U values in a numpy array.

    J_array         nparray     All the J values in a numpy array.

    dJ_array        nparray     All the dJ values in a numpy array.

    jobdef          dict        The dictionary that defines the tight binding
                                job to be run.

    jobdef_file     str         The name of the jobdef file.

    model           dict        The dictionary that contains the model system
                                for the tight binding job to be run.

    temp_modelfile  str         The name of the model file.

    orb_type        str         On-site orbital symmetry. either s, p or d.

    number_decimals int         The number of decimal places to report values
                                of U, J and dJ.


    OUTPUTS         TYPE        DESCRIPTION

    SuccessFlag     bool        If all the tight binding simulations are
                                successful then this is returned as True, if
                                any are not successful the loop is exited and
                                this is returned as False.

    mag_corr        dict        A dictionary containing value of the magnetic
                                correlation at each value of U, J and dJ.


    """
    # initialise the mag_corr dictionary
    mag_corr = {}
    SuccessFlag = True
    for U in U_array:
        for J in J_array:
            print("U = ", U, "\t J = ", J)
            for dJ in dJ_array:
                # if J > U:
                #     mag_corr[U, J, dJ] = 0.0
                # else:
                model['species'][0]["U"] = round(U, number_decimals)
                if orb_type == "p":
                    model['species'][0]["I"] = round(J, number_decimals)
                elif orb_type == "d":
                    model['species'][0]["I"] = round(J, number_decimals)
                    model['species'][0]["dJ"] = round(dJ, number_decimals)

                # write out the new modelfile
                with open(temp_modelfile, 'w') as f:
                    commentjson.dump(model, f, sort_keys=True, indent=4, separators=(',', ': '))
                try:
                    SCFflag, mag_corr[round(U, number_decimals), round(J, number_decimals), round(dJ, number_decimals)] = main()
                # if we end up with a linear algebra error then we can re-run with a different optimisation scheme.
                except numpy.linalg.linalg.LinAlgError:
                    # store original optimisation routine choice
                    old_routine = jobdef['optimisation_routine']
                    if old_routine == 1:
                        # then set it to routine 2
                        jobdef['optimisation_routine'] = 2
                    else:
                        # then set it to routine 1
                        jobdef['optimisation_routine'] = 1
                    # write jobdef back to file
                    with open(jobdef_file, 'w') as f:
                        commentjson.dump(jobdef, f, sort_keys=True, indent=4, separators=(',', ': '))
                    # and run again
                    print("SCF did not converge. Re-running simulation with different optimisation routine. ")
                    SCFflag, mag_corr[round(U, number_decimals), round(J, number_decimals), round(dJ, number_decimals)] = main()
                    # reset optimisation routine
                    jobdef['optimisation_routine'] = old_routine
                    # write jobdef back to file
                    with open(jobdef_file, 'w') as f:
                        commentjson.dump(jobdef, f, sort_keys=True, indent=4, separators=(',', ': '))


                # If the SCF has converged then we can trust the result
                if SCFflag == True:
                    pass
                # If the SCF flag is False and this was an SCF calculation then rerun
                elif jobdef["scf_on"] == 1:
                    # Use a smaller value of A (divide by 10)
                    jobdef["A"] = jobdef["A"]/10.0
                    # Increase number of steps by a factor of 10
                    jobdef["scf_max_loops"] = int(jobdef["scf_max_loops"]*10)
                    # write jobdef back to file
                    with open(jobdef_file, 'w') as f:
                        commentjson.dump(jobdef, f, sort_keys=True, indent=4, separators=(',', ': '))
                    # and run again
                    print("SCF did not converge. Re-running simulation with smaller mixing value. ")
                    SCFflag, mag_corr[round(U, number_decimals), round(J, number_decimals), round(dJ, number_decimals)] = main()
                    
                    # Re-set the jobdef variables:
                    jobdef["A"] = jobdef["A"]*10.0
                    jobdef["scf_max_loops"] = int(jobdef["scf_max_loops"]/10)
                    with open(jobdef_file, 'w') as f:
                        commentjson.dump(jobdef, f, sort_keys=True, indent=4, separators=(',', ': '))
                    # If that still hasn't worked, exit gracefully...
                    if SCFflag == False:
                        SuccessFlag = False
                        print("SCF did not converge for U = ", round(U, number_decimals), "; J = ", round(J, number_decimals), "; dJ = ", round(dJ, number_decimals))
                        print("Exiting.")
                        return SuccessFlag, mag_corr

    return SuccessFlag, mag_corr




def Plot_OpSq_U_J(op_sq_dict, orbtype, plotname, Umin, Ustep, Unumsteps, Jmin, Jstep, Jnumsteps, dJmin, dJstep, dJnumsteps, op_sq_name, number_decimals):
    """
    The function Plot_OpSq_U_J plots the values of an operator squared
    against U and J. This is done for the groundstate eigenvector and will look
    down on the diagram and will use colours, like a colourmap or a phase
    diagram to illustrate the correlation.
    
    INPUTS            TYPE        DESCRIPTION
    
    op_sq_dict        dict        A python dictionary containing the operator
                                  squared values for each value of U, J and dJ.
    
    orbtype           int         The orbital type of the system, 's', 'p' or
                                  'd'.
    
    plotname          str         The name of the plot, if left blank then no
                                  plot is made, remember to not include ".pdf"
                                  as that will be added in the algorithm below.
    
    U/J/dJmin         float       The starting value for U/J/dJ.
    
    U/J/dJstep        float       The stepsize for U/J/dJ.
    
    U/J/dJnumsteps    int         The number of steps for U/J/dJ.
    
    op_sq_name        str         The name of operator squared.

    number_decimals   int         The number of decimal places to read in values
                                  of U, J and dJ.

    
    OUTPUTS           TYPE        DESCRIPTION
    
    plotname          pdf         A 3d graph of the operator squared for many
                                  values of U and J.
    
    """
    # List of symbols in plot
    num_decimal_points = 5
    LineStyle = ["-", "--", ":", "-."]
    PlotSymbols = ["s", "o", "x", "8", "d", "h", "^", "p", "+", "*"]
    PlotColour = ["r", "b", "g", "c", "m", "b", "y"]
    Lz_symbol = ["\Sigma", "\Pi", "\Delta", "\Phi", "\Gamma", "H", "I", "J", "K", "\Lambda", "M", "N", "\Omega", "\Pi"]


    # only make the plot if plotname is not empty
    if plotname:
        CurveDict = {}
        if orbtype == 's':
            plt.figure()
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.ylabel("$"+op_sq_name+"$", fontsize=16)
            plt.xlabel(r"$U/|t|$", fontsize=16)
            Name = plotname+".pdf"
            print("Starting on "+str(Name))
        #     for i in range(NumOpSq):
        #         print("Eigenvalue: "+str(i+1))
            Urange = [Umin+j*Ustep for j in range(Unumsteps+1)]
            Values = [op_sq_dict[round(U, number_decimals), round(Jmin, number_decimals), round(dJmin, number_decimals)] for U in Urange]
        #         #CurveDict[i+1] = plt.plot(Urange,Values,marker=PlotSymbols[len(PlotSymbols)%(i+1)],linestyle=LineStyle[(i+1)%len(LineStyle)],color=PlotColour[(1+i)%len(PlotColour)])
            CurveDict[1] = plt.plot(Urange, Values, linestyle=LineStyle[(1)%len(LineStyle)], color=PlotColour[(1)%len(PlotColour)])
        #     handles = [CurveDict[i+1][0] for i in range(NumOpSq)]
        #     labels = ["Eigenvalue "+str(i+1) for i in range(NumOpSq)]
            plt.xlim([round(min(Urange), 1), round(max(Urange), 1)])
        #     plt.legend(handles,labels,loc="upper left")
            plt.savefig(Name)
            plt.clf()

        elif orbtype=='p' or dJnumsteps==0:
        # -----------------------------------------------------------------------
        # Make a colourmap of the magmom_corr of the GS in U/|t| and J/|t| space.
        # -----------------------------------------------------------------------
            # Urange=np.array([[Umin+j*Ustep for i in range(Jnumsteps+1)] for j in range(Unumsteps+1)])
            # print("U\n")
            # print(Urange)
            # Jrange=np.array([[Jmin+i*Jstep for i in range(Jnumsteps+1)] for j in range(Unumsteps+1)])
            # print("J\n")
            # print(Jrange)
            Umax = (Unumsteps)*Ustep+Umin
            Jmax = (Jnumsteps)*Jstep+Jmin
            Urange,Jrange = np.mgrid[slice(Umin,Umax+Ustep,Ustep),slice(Jmin,Jmax+Jstep,Jstep)]
            Values = np.array([[op_sq_dict[round(Umin+Ustep*k, number_decimals),round(Jmin+Jstep*j, number_decimals), round(dJmin, number_decimals)] for j in range(Jnumsteps+1)] for k in range(Unumsteps+1)])
            fig = plt.figure()
            #NoColourSteps=100
            #ColourStepSize=(Values.max()-Values.min())/(NoColourSteps-1)
            # Create the boundaries of the bins for the colorbar. Round to 2 decimal places. The top and bottom value need special treatment, i.e. floor and ceil, otherwise whitespace is created.
            #levels=[floor(Values.min()*100)/100]+[round(Values.min()+ColourStepSize*i,2) for i in range(1,NoColourSteps)]+[ceil(Values.max()*100)/100]
            #im = plt.contourf(Urange,Jrange,Values,levels=levels,cmap=cm.coolwarm)
            ax = fig.add_subplot(111)
            im = ax.pcolormesh(Urange,Jrange,Values,cmap=cm.gray)
            #im.clim(Values.min(),Values.max())
            cbar = fig.colorbar(im)
            cbar.set_label("$"+op_sq_name+"$",fontsize=24,rotation=270)
            cbar.ax.tick_params(labelsize=18)
            ax.set_xlabel(r"$U/|t|$",fontsize=24)
            ax.set_ylabel(r"$J/|t|$",fontsize=24)
            # plt.yticklabels(fontsize=18)
            # for jj in GS_dict['states']:
            #     mid_x = (GS_dict[jj,"Umin",dJmin]+GS_dict[jj,"Umax",dJmin])*0.5
            #     mid_y = (GS_dict[jj,"Jmin",dJmin]+GS_dict[jj,"Jmax",dJmin])*0.5
            #     print("State "+str(jj)+" has min J value at "+str(GS_dict[jj,"Jmin",dJmin])+" and max J value at "+str(GS_dict[jj,"Jmax",dJmin]))
            #     print("State "+str(jj)+" is located at ("+str(mid_x)+", "+str(mid_y)+")")
            #     S = jj[1]
            #     L_z = jj[2]
            #     ug = jj[3]
            #     pm = jj[4]
            #     if L_z == 0.0:
            #         symbol = "$^"+str(int(2*S+1))+Lz_symbol[int(L_z)]+"_"+ug+"^"+pm+"$"
            #     else:
            #         symbol = "$^"+str(int(2*S+1))+Lz_symbol[int(L_z)]+"_"+ug+"$"
            #         print("L_z = "+str(L_z))
            #     Scheck = checkint(S,0.0000001)
            #     Lzcheck = checkint(L_z,0.0000001)
            #     if not Scheck:
            #         Swarning = "$S$ = "+str(S)
            #         #plt.text(mid_x+0.1*mid_x,mid_y,Swarning,color='red',fontsize=20)
            #     if not Lzcheck:
            #         Lzwarning = "$L_z$ = "+str(L_z)
            #         #plt.text(mid_x+0.1*mid_x,mid_y-0.1*mid_y,Lzwarning,color='red',fontsize=20)
            #     #plt.text(mid_x,mid_y,symbol,fontsize=20)
            ax.set_xlim(Urange.min(),Urange.max())
            ax.set_ylim(Jrange.min(),Jrange.max())
            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(18)      
            Name = plotname+"_nolabels.pdf"
            fig.savefig(Name)
            # fig = plt.figure()
            # ax = fig.gca(projection='3d')
            # Urange=np.array([[Umin+j*Ustep for i in range(Jnumsteps+1)] for j in range(Unumsteps+1)])
            # Jrange=np.array([[Jmin+i*Jstep for i in range(Jnumsteps+1)] for j in range(Unumsteps+1)])
            # # Only consider the groundstate:
            # Values = np.array([[op_sq_dict[j+k*Jnumsteps] for j in range(Jnumsteps+1)] for k in range(Unumsteps+1)])
            # surf = ax.plot_surface(Urange, Jrange, Values, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0)
            # ax.set_xlabel(r"$U/|t|$",fontsize=16,linespacing=3.2)
            # ax.set_ylabel(r"$J/|t|$",fontsize=16,linespacing=4)
            # ax.set_zlabel("$ \\frac{1}{3} N(\hat{M}_1.\hat{M}_2) $",fontsize=16,linespacing=4)
            # ax.dist=10
            # ax.zaxis.set_major_locator(LinearLocator(10))
            # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            # ax.view_init(elev=90., azim=90)
            # #fig.colorbar(surf, shrink=0.5, aspect=5)
            # Name = plotname+".pdf"
            # #plt.legend(handles,labels,loc="upper left")
            # plt.savefig(Name)
        #     if not os.path.exists(moviedir):
        #         os.makedirs(moviedir)

        elif orbtype=='d':
        # -----------------------------------------------------------------------
        # Make a 3d colourmap of the magmom_corr of the GS in U/|t|, J/|t| and dJ/|t| space.
        # Display 2d cross sections for various values of dJ/|t|.
        # -----------------------------------------------------------------------
            digits=3
            Umax = (Unumsteps)*Ustep+Umin
            Jmax = (Jnumsteps)*Jstep+Jmin
            Urange,Jrange = np.mgrid[slice(Umin,Umax+Ustep,Ustep),slice(Jmin,Jmax+Jstep,Jstep)]
            cmax=max(op_sq_dict.values())
            cmin=min(op_sq_dict.values())
            for ii in range(dJnumsteps+1):
                dJ = round(ii*dJstep+dJmin,num_decimal_point)
                Values = np.array([[op_sq_dict[round(Umin+Ustep*k, number_decimals), round(Jmin+Jstep*j, number_decimals), round(dJ, number_decimals)] for j in range(Jnumsteps+1)] for k in range(Unumsteps+1)])
                # plt.figure()
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_xlabel(r"$U/|t|$",fontsize=24)
                ax.set_ylabel(r"$J/|t|$",fontsize=24)
                im = ax.pcolormesh(Urange,Jrange,Values,cmap=cm.gray)
                #im.clim(Values.min(),Values.max())
                cbar = fig.colorbar(im)
                cbar.set_label("$"+op_sq_name+"$",fontsize=24,rotation=270)
                cbar.ax.tick_params(labelsize=18)
                # im = plt.pcolormesh(Urange,Jrange,Values,cmap=cm.gray)
                # plt.clim(cmin,cmax)
                # cbar = plt.colorbar()
                # cbar.set_label("$"+op_sq_name+"$",fontsize=24,rotation=270)
                # cbar.tick_params(labelsize=18)
                # plt.xlabel(r"$U/|t|$",fontsize=24)
                # plt.ylabel(r"$J/|t|$",fontsize=24)
                # plt.xticks(fontsize=18)
                # plt.yticks(fontsize=18)
                # for jj in GS_dict['states']:
                #     if (GS_dict[jj,"dJmin"]<=dJ and GS_dict[jj,"dJmax"]>=dJ):
                #         mid_x = (GS_dict[jj,"Umin",dJ]+GS_dict[jj,"Umax",dJ])*0.5
                #         mid_y = (GS_dict[jj,"Jmin",dJ]+GS_dict[jj,"Jmax",dJ])*0.5
                #         S = jj[1]
                #         L_z = jj[2]
                #         ug = jj[3]
                #         pm = jj[4]
                #         if L_z == 0.0:
                #             symbol = "$^"+str(int(2*S+1))+Lz_symbol[int(L_z)]+"_"+ug+"^"+pm+"$"
                #         else:
                #             symbol = "$^"+str(int(2*S+1))+Lz_symbol[int(L_z)]+"_"+ug+"$"
                #             print("L_z = "+str(L_z))
                #         # removed label for paper
                #         # plt.text(mid_x,mid_y,symbol,fontsize=20)
                #         Scheck = checkint(S,0.0000001)
                #         Lzcheck = checkint(L_z,0.0000001)
                #         if not Scheck:
                #             Swarning = "$S$ = "+str(S)
                #             # removed label for paper
                #             # plt.text(mid_x+0.1*mid_x,mid_y,Swarning,color='red',fontsize=20)
                #         if not Lzcheck:
                #             Lzwarning = "$L_z$ = "+str(L_z)
                #             # removed label for paper
                #             # plt.text(mid_x+0.1*mid_x,mid_y-0.1*mid_y,Lzwarning,color='red',fontsize=20)
                #plt.title(r"$\Delta J/|t| = $"+str(dJ),fontsize=16)
                # plt.axis([Urange.min(),Urange.max(),Jrange.min(),Jrange.max()])
                Name = plotname+"_dJstep_"+str(ii).zfill(digits)+".pdf"
                ax.set_xlim(Urange.min(),Urange.max())
                ax.set_ylim(Jrange.min(),Jrange.max())
                for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(18)  
                fig.savefig(Name)
                # plt.savefig(Name)
                fig.clf()
        # -----------------------------------------------------------------------
        # Construct a video of the 3d phase diagram.
        # -----------------------------------------------------------------------
        # No videos required for paper
            # currentdir=os.getcwd()
            # dirlist=plotname.split("/")[:-1]
            # print(dirlist)
            # moviedir = ""
            # for ii in range(1,len(dirlist)):
            #     moviedir+="/"+dirlist[ii]
            # os.chdir(moviedir)
            # mencommand = "mencoder mf://*.jpg -mf w=800:h=600:fps=1:type=jpg -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o "+ plotname.split("/")[-1]+".avi"
            # os.system(mencommand)
            # os.chdir(currentdir)
        else:
            print("WARNING: Orbital type is not correct for function Plot_OpSq_U_J.")


def make_magmomcorr_graphs(numeperatom):
    number_decimals = 6
    orb_type = "p"
    plotname = "../output_PyLATO/Mag_Corr_"+orb_type+"_"+str(numeperatom)
    op_sq_name="\\frac{1}{3} \langle :\hat{\mathbf{m}}_1.\hat{\mathbf{m}}_2:\\rangle"

    U_min = 0.005
    U_max = 10
    U_num_steps = 100

    J_min = 0.005
    J_max = 5
    J_num_steps = 100

    dJ_min = 0
    dJ_max = 1
    dJ_num_steps = 1

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
    # make sure that the number of electrons is correct.
    model['species'][0]["NElectrons"] = numeperatom

    # change the model and Hamiltonian in jobdef
    jobdef["Hamiltonian"] = orb_type+"case"
    jobdef["model"] = model_temp
    # make sure that the scf is on
    jobdef["scf_on"] = 1
    # write jobdef back to file
    with open(jobdef_file, 'w') as f:
        commentjson.dump(jobdef, f, sort_keys=True, indent=4, separators=(',', ': '))

    #pdb.set_trace()

    magFlag, mag_corr = mag_corr_loop(U_array, J_array, dJ_array, jobdef, jobdef_file, model, temp_modelfile, orb_type, number_decimals)


    # clean up temp files
    os.remove(temp_modelfile)
    os.remove(model_python_temp)
    os.remove(model_python_temp+"c")
    # restore backup of JobDef.json
    shutil.copyfile(jobdef_backup, jobdef_file)
    os.remove(jobdef_backup)

    # Make the plot if the mag_corr_loop was successful
    if magFlag == True:
        Plot_OpSq_U_J(mag_corr,orb_type,plotname,U_min,U_step,U_num_steps,J_min,J_step,J_num_steps,dJ_min,dJ_step,dJ_num_steps,op_sq_name, number_decimals)
    else:
        print("Simulation failed.")


