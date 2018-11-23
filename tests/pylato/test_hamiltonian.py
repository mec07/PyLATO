import numpy as np

from pylato.hamiltonian import Hamiltonian
from pylato.init_job import InitJob


class TestHamiltonian:
    def test_total_energy_happy_path(self):
        # Setup
        Job = InitJob("test_data/JobDef.json")

        #####################################
        # create fock matrix & density matrix
        #####################################
        Job.Hamilton.buildHSO(Job)
        Job.e, Job.psi = np.linalg.eigh(Job.Hamilton.HSO)
        Job.Electron.occupy(Job.e, Job.Def['el_kT'], Job.Def['mu_tol'], Job.Def['mu_max_loops'])
        Job.Electron.densitymatrix()
        Job.Hamilton.buildFock(Job)
        print("#######################################")
        print("density matrix = ")
        print(Job.Electron.rho)
        print("#######################################")
        print("fock = ")
        print(Job.Hamilton.fock)
        print("#######################################")

        # Action
        energy = Job.Hamiltonian.total_energy()

        # Result
        assert energy == -1
