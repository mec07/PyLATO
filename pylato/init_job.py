"""
Created on Sunday April 19, 2015

@author: Andrew Horsfield, Marc Coury and Max Boleininger

This module carries out the needed initialisation tasks
"""
#
# Import modules
import os, sys, importlib
import hamiltonian
import electronic
import pylato_IO
import commentjson
import numpy as np
import crystal
from verbosity import *

class InitJob:
    """Set up the job, build the initial geometry, hamiltonian, and electronic structure."""
    def __init__(self, jobfile):
        """Initialise the job."""

        # Set up variables that define the job in a dictionary
        with open(jobfile, 'r') as inputfile:
            self.Def = commentjson.loads(inputfile.read())

        # Fetch the model name path from the Job file
        if self.Def['model'] == "TBcanonical":
            modelname = self.Def['model']+"_"+self.Def['Hamiltonian'][0]
        else:
            modelname = self.Def['model']
        modelpath = os.path.join("models", modelname + ".py")

        # Catch invalid model path
        if os.path.exists(modelpath) == False:
            print("ERROR: Unable to open tight binding model at %s. ")
            print(modelpath)
            sys.exit()

        # Has a directory for results been specified?
        if "results_dir" in self.Def.keys():
            # check to make sure that it has the final "/"
            if self.Def['results_dir'][-1]=="/":
                self.results_dir = self.Def['results_dir']
            else:
                self.results_dir = self.Def['results_dir']+"/"
            # Make sure that the directory where results will be put exists
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
        # If it hasn't been specified then just set it to the current directory.
        else:
            self.results_dir = "./"

        # Import the module responsible for the tight binding model
        model_module = importlib.import_module("models." + modelname)

        # Initialise the model class
        self.Model = model_module.MatrixElements(os.path.join("models", modelname + ".json"))

        # Initialise the geometry
        self.init_geom(self.Def['gy_file'], self.Def['uc_file'])

        # Initialise the Hamiltonian class
        self.Hamilton = hamiltonian.Hamiltonian(self)

        # Initialise the electron module
        self.Electron = electronic.Electronic(self)



    def init_geom(self, position_file, unitcell_file):
        """initialise the geometry."""
        if self.Def["PBC"]==1:
            PBCs = True
        else:
            PBCs = False
        # If build geometry is turned on then build the geometry
        if self.Def['build_geom'] == 1:
            a, crys_err = self.calculate_crystal_sep()
            # if the calculation of the crystal separation a is successful:
            if crys_err == 0:
                mycry = crystal.Crystal(a=a, lattice="cubic")
                mycry.populateUnitCell(self.Def['crystal'], geom_filename=position_file, uc_filename=unitcell_file, nx=self.Def['nx'], ny=self.Def['ny'], nz=self.Def['nz'], PBCs=PBCs)
        # Read in the geometry from file
        NAtom, Pos, AtomType = pylato_IO.ReadGeom(position_file)
        # If PBCs are turned on then read in the unit cell
        if PBCs:
            a1, a2, a3 = pylato_IO.ReadUnitCell(unitcell_file)
        else:
            a1, a2, a3 = np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.0, 0.0))

        # Write out the geometry
        pylato_IO.WriteXYZ(self, NAtom, '', AtomType, Pos)

        # Transfer geometry to the JobClass
        self.NAtom    = NAtom
        self.Pos      = Pos
        self.UnitCell = [a1, a2, a3]

        self.AtomType = AtomType
        self.NOrb     = [self.Model.atomic[self.AtomType[a]]['NOrbitals'] for a in range(self.NAtom)]

        verboseprint(self.Def['verbose'], "Atom positions:")
        verboseprint(self.Def['verbose'], self.Pos)

    def calculate_crystal_sep(self):
        """
        Calculate the crystal separation from the nearest neigbour separation.
        For cubic they are the same, for fcc and bcc there are simple
        relations that link them.
        """
        if self.Def['crystal'] == "cubic":
            a = self.Def['nearest_neighbour_sep']
            return (a,0)
        elif self.Def['crystal'] == "fcc":
            a = self.Def['nearest_neighbour_sep']*np.sqrt(2)
            return (a,0)
        elif self.Def['crystal'] == "bcc":
            a = 2*self.Def['nearest_neighbour_sep']/np.sqrt(3)
            return (a,0)
        else:
            print("WARNING: To build a crystal, the crystal type inserted must be one of cubic, fcc or bcc.")
            print("Continuing using the geometry file "+position_file+".")
            return (0,1)
