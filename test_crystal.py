#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import crystal
import visualise_geom

geomfile = "geom.csv"
mycry = crystal.Crystal(a=3.0, lattice="cubic")
mycry.populateUnitCell("fcc", geom_filename=geomfile, nx=2, ny=2, nz=2, PBCs=True)

outputfile = "geometry"
visualise_geom.save_geom_as_pdf(geomfile, outputfile)

visualise_geom.make_xyz_file(geomfile,outputfile)