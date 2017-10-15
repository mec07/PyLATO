#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def save_geom_as_pdf(geomfile, outputfile):
    with open(geomfile,'r') as f:
        # read file
        geom = f.read()
        
    # separate into a list (have to remove new lines and commas)
    geom = geom.split("\n")

    n = int(geom[0])
    x = []
    y = []
    z = []
    for ii in range(n):
        coords = geom[ii+1].split(',')
        x.append(float(coords[1]))
        y.append(float(coords[2]))
        z.append(float(coords[3]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
        ax.scatter(x, y, z, c=c, marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # check to make sure that the outputfile is a pdf.
    if outputfile[-4:] == ".pdf":
        pass
    else:
        outputfile+=".pdf"
    plt.savefig(outputfile)

def make_xyz_file(geomfile, outputfile):
    with open(geomfile,'r') as f:
        # read file
        geom = f.read()
        
    # separate into a list (have to remove new lines and commas)
    geom = geom.split("\n")

    n = int(geom[0])
    x = []
    y = []
    z = []
    for ii in range(n):
        coords = geom[ii+1].split(',')
        x.append(float(coords[1]))
        y.append(float(coords[2]))
        z.append(float(coords[3]))

    # check to make sure that outputfile is xyz format
    if outputfile[-4:] == ".xyz":
        pass
    else:
        outputfile+=".xyz"

    with open(outputfile, 'w') as f:
        f.write(str(n)+"\n")
        f.write("# comment line\n")
        for ii in range(n):
            coords = "\t".join([str(x[ii]),str(y[ii]),str(z[ii])])
            f.write("C\t"+coords+"\n")

