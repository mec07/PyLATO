"""
Tests for the genetic module
"""
from pylato.genetic import (Individual,
                            Population,
                            PerformGeneticAlgorithm)
from pylato.hamiltonian import Hamiltonian
from pylato.electronic import Electronic
from tests.utils import GenericClass, return_a_function, approx_equal

import numpy as np
import pytest


@pytest.fixture
def job():
    Job = GenericClass(
        AtomType=['A', 'A'],
        Def={
            'el_kT': 0.0,
            'extraverbose': 0,
            'Hamiltonian': 'scase',
            'mu_tol': 1e-13,
            'mu_max_loops': 5000,
            'num_rho': 1,
            'PBC': 0,
            'population_size': 50,
            'max_num_evolutions': 1000,
            'genetic_tol': 1.0e-8,
            'proportion_to_retain': 0.2,
            'random_select_chance': 0.05,
            'mutation_chance': 0.05,
            'so_eB': [0, 0, 0],
            'spin_orbit': 0,
            'verbose': 1,
        },
        Model=GenericClass(
            atomic={
                'A': {
                    'NOrbitals': 1,
                    'e': [[0]],
                    'l': [0],
                    'NElectrons': 1,
                    'U': 0,
                },
            },
            helements=return_a_function([
                np.array([-1, 0]),
                np.array([[0]]),
                np.array([[1]])
            ]),
        ),
        NAtom=2,
        NOrb=[1, 1],
        Pos=np.array([
            [0, 0, 0],
            [1, 0, 0],
        ]),
        UnitCell=np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
    )

    # generate HSO
    Job.Hamilton = Hamiltonian(Job)
    Job.Hamilton.buildHSO(Job)

    # find eigenvalues & eigenvectors
    Job.e, Job.psi = np.linalg.eigh(Job.Hamilton.HSO)

    # instantiate the Electron object
    Job.Electron = Electronic(Job)

    return Job


def num_values_changed(array1, array2):
    numValuesChanged = 0
    for index in range(len(array1)):
        if not array1[index] == array2[index]:
            numValuesChanged += 1

    return numValuesChanged


class TestGeneticIndividual:
    def test_individual_create_random_DNA(self, job):
        length = 100
        sum_constraint = 10
        min_value = 0.0
        max_value = 1.0

        an_individual = Individual(job, length, sum_constraint)

        assert len(an_individual.DNA) == length
        assert an_individual.DNA.min() >= min_value
        assert an_individual.DNA.max() <= max_value
        assert approx_equal(an_individual.DNA.sum(), sum_constraint)

    def test_individual_provide_DNA(self, job):
        length = 100
        sum_constraint = 10
        min_value = 0.0
        max_value = 1.0

        DNA = np.random.uniform(size=100)

        an_individual = Individual(job, length, sum_constraint, DNA)

        assert len(an_individual.DNA) == length
        assert an_individual.DNA.min() >= min_value
        assert an_individual.DNA.max() <= max_value
        assert approx_equal(an_individual.DNA.sum(), sum_constraint)

    def test_individual_checks(self, job):
        length = 100
        # Raises for a constraint greater than 80% of length
        with pytest.raises(Exception):
            Individual(job, length, 0.9*length)
        # Raises for a constraint less than 1 of length
        with pytest.raises(Exception):
            Individual(job, length, 0.1)
        # Raises if length is not an int
        with pytest.raises(Exception):
            Individual(job, 'hi', 2)
        # Raises if constraint is not a number
        with pytest.raises(Exception):
            Individual(job, length, 'hi')

    def test_normalise(self, job):
        example1 = np.array([0.1, 0.2, 0.3, 0.4])
        example2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        example3 = np.random.uniform(size=200)
        sum_constraint1 = 3
        sum_constraint2 = 1
        sum_constraint3 = 20
        normalised_example1 = Individual(job, 4, sum_constraint1, example1)
        normalised_example2 = Individual(job, 7, sum_constraint2, example2)
        normalised_example3 = Individual(job, 200, sum_constraint3, example3)

        assert approx_equal(normalised_example1.DNA.sum(), sum_constraint1)
        assert approx_equal(normalised_example2.DNA.sum(), sum_constraint2)
        assert approx_equal(normalised_example3.DNA.sum(), sum_constraint3)

    def test_mutate_no_mutation(self, job):
        individual = Individual(job, 100, 10)
        DNA_before = np.copy(individual.DNA)
        individual.mutate(0.0)
        DNA_after = np.copy(individual.DNA)

        assert np.array_equal(DNA_before, DNA_after)

    def test_mutate_definitely(self, job):
        individual = Individual(job, 6, 3)
        DNA_before = np.copy(individual.DNA)
        individual.mutate(1.0)
        DNA_after = np.copy(individual.DNA)

        # Expect that exactly one of the values has changed
        assert num_values_changed(DNA_before, DNA_after) == 1

    def test_fitness_groundstate(self, job):
        # fake a Job and Hamiltonian
        groundstate_occupation = np.array([1, 1, 0, 0])
        individual = Individual(job, 4, 2, groundstate_occupation)

        individual.calculate_fitness()

        # As we've set the density depedent terms to zero, the temperature to
        # zero and chosen the individual to be the groundstate occupation, the
        # the fitness must be zero.
        assert approx_equal(individual.fitness, 0.0)

    def test_fitness_non_groundstate1(self, job):
        # fake a Job and Hamiltonian
        excited_occupation = np.array([0, 1, 0, 1])
        individual = Individual(job, 4, 2, excited_occupation)

        individual.calculate_fitness()

        # As we've set the density depedent terms to zero, the temperature to
        # zero and chosen the individual to be an excited occupation, the
        # the fitness can be worked out to be 0.5 by hand.
        assert approx_equal(individual.fitness, 0.5)

    def test_fitness_non_groundstate2(self, job):
        # fake a Job and Hamiltonian
        excited_occupation = np.array([0, 1, 1, 0])
        individual = Individual(job, 4, 2, excited_occupation)

        individual.calculate_fitness()

        # As we've set the density depedent terms to zero, the temperature to
        # zero and chosen the individual to be an excited occupation, the
        # the fitness can be worked out to be 1.0 by hand.
        assert approx_equal(individual.fitness, 1.0)

    def test_fitness_non_groundstate3(self, job):
        # fake a Job and Hamiltonian
        excited_occupation = np.array([0, 0, 1, 1])
        individual = Individual(job, 4, 2, excited_occupation)

        individual.calculate_fitness()

        # As we've set the density depedent terms to zero, the temperature to
        # zero and chosen the individual to be an excited occupation, the
        # the fitness can be worked out to be 1.0 by hand.
        assert approx_equal(individual.fitness, 1.0)

    def test_mate(self, job):
        female = Individual(job, 4, 1, np.array([0.5, 0.5, 0.0, 0.0]))
        male = Individual(job, 4, 1, np.array([0.0, 0.0, 0.5, 0.5]))

        child = female.mate(male)

        expected_DNA = np.array([0.25, 0.25, 0.25, 0.25])

        assert np.array_equal(child.DNA, expected_DNA)

    def test_mate_odd(self, job):
        female = Individual(job, 5, 1, np.array([0.5, 0.5, 0.0, 0.0, 0.0]))
        male = Individual(job, 5, 1, np.array([0.0, 0.0, 0.5, 0.5, 0.0]))

        child = female.mate(male)

        expected_DNA = np.array([0.25, 0.25, 0.25, 0.25, 0.0])

        assert np.array_equal(child.DNA, expected_DNA)


class TestGeneticPopulation:
    def test_create_population(self, job):
        population_size = 50
        length = 100
        sum_constraint = 10
        min_value = 0.0
        max_value = 1.0

        a_population = Population(job, population_size, length, sum_constraint)

        assert len(a_population.individuals) == population_size
        for ii in range(population_size):
            assert a_population.individuals[ii].DNA.size == length
            assert a_population.individuals[ii].DNA.min() >= min_value
            assert a_population.individuals[ii].DNA.max() <= max_value
            assert approx_equal(a_population.individuals[ii].DNA.sum(), sum_constraint)

    def test_grade(self, job):
        individuals_DNA = np.array([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0]
        ])
        individuals = [Individual(job, 4, 2, individuals_DNA[x]) for x in range(4)]
        population = Population(job, 4, 4, 2)
        population.individuals = individuals

        average_fitness = population.grade()

        assert (average_fitness - 0.625) < 1.0e-8
        assert np.array_equal(population.individuals[0].DNA, individuals_DNA[3])
        assert np.array_equal(population.individuals[1].DNA, individuals_DNA[1])
        # The other two both have a fitness of 1.0 so can be in either order
        assert ((np.array_equal(population.individuals[2].DNA, individuals_DNA[0]) and
                 np.array_equal(population.individuals[3].DNA, individuals_DNA[2])) or
                (np.array_equal(population.individuals[2].DNA, individuals_DNA[2]) and
                 np.array_equal(population.individuals[3].DNA, individuals_DNA[0])))

    def test_survive_no_random_select(self, job):
        sorted_population_DNA = np.array([
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 1]
        ])
        retain = 0.5
        random_select = 0.0
        individuals = [Individual(job, 4, 2, sorted_population_DNA[x]) for x in range(4)]
        population = Population(job, 4, 4, 2)
        population.individuals = individuals

        population.survive(retain, random_select)
        assert len(population.individuals) == 2
        assert np.array_equal(population.individuals[0].DNA, sorted_population_DNA[0])
        assert np.array_equal(population.individuals[1].DNA, sorted_population_DNA[1])

    def test_survive_with_full_select(self, job):
        sorted_population_DNA = np.array([
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 1]
        ])
        retain = 0.5
        random_select = 1.0
        individuals = [Individual(job, 4, 2, sorted_population_DNA[x]) for x in range(4)]
        population = Population(job, 4, 4, 2)
        population.individuals = individuals

        population.survive(retain, random_select)
        assert len(population.individuals) == 4
        assert np.array_equal(population.individuals[0].DNA, sorted_population_DNA[0])
        assert np.array_equal(population.individuals[1].DNA, sorted_population_DNA[1])
        assert np.array_equal(population.individuals[2].DNA, sorted_population_DNA[2])
        assert np.array_equal(population.individuals[3].DNA, sorted_population_DNA[3])

    def test_can_produce_num_children(self, job):
        population = Population(job, 50, 100, 20)
        population.survive(0.2, 0.0)
        assert population.can_produce_num_children() is True

        # Raises an exception when there aren't enough parents to produce the
        # desired number of children.
        population.survive(0.1, 0.0)
        with pytest.raises(Exception):
            population.can_produce_num_children()

    def test_reproduce_no_mutation(self, job):
        parents_DNA = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 0.0, 1.0, 0.0]
        ])
        population_size = 4
        length = 4
        sum_constraint = 2
        individuals = [Individual(job, length, sum_constraint, parents_DNA[x]) for x in range(2)]
        population = Population(job, population_size, length, sum_constraint)
        population.individuals = individuals
        mutation_chance = 0.0

        assert len(population.individuals) == 2
        population.reproduce(mutation_chance)

        assert len(population.individuals) == 4

        # We know that the children must be half of one parent and half of the
        # other (as we chose the parents so that their children would be
        # normalised at birth), so test both options.
        expected_child1 = np.array([1.0, 0.0, 0.5, 0.5])
        expected_child2 = np.array([0.5, 0.5, 1.0, 0.0])
        assert ((np.array_equal(population.individuals[2].DNA, expected_child1) and
                 np.array_equal(population.individuals[3].DNA, expected_child2)) or
                (np.array_equal(population.individuals[2].DNA, expected_child2) and
                 np.array_equal(population.individuals[3].DNA, expected_child1)))

    def test_reproduce_with_mutation(self):
        parents_DNA = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 0.0, 1.0, 0.0]
        ])
        population_size = 4
        length = 4
        sum_constraint = 2
        individuals = [Individual(job, length, sum_constraint, parents_DNA[x]) for x in range(2)]
        population = Population(job, population_size, length, sum_constraint)
        population.individuals = individuals
        mutation_chance = 1.0

        assert len(population.individuals) == 2
        population.reproduce(mutation_chance)

        assert len(population.individuals) == 4

        # We know that the children must be half of one parent and half of the
        # other (as we chose the parents so that their children would be
        # normalised at birth), so test both options.
        expected_child1 = np.array([1.0, 0.0, 0.5, 0.5])
        expected_child2 = np.array([0.5, 0.5, 1.0, 0.0])
        # The mutated children will be different from the expected children. To
        # make sure of that we test to see that they are not the same, but we
        # don't know which way around the children will be, so we check both
        # options.
        assert ((not np.array_equal(population.individuals[2].DNA, expected_child1) and
                 not np.array_equal(population.individuals[3].DNA, expected_child2)) or
                (not np.array_equal(population.individuals[2].DNA, expected_child2) and
                 not np.array_equal(population.individuals[3].DNA, expected_child1)))

        # We also check the normalisation has worked correctly:
        assert approx_equal(population.individuals[2].DNA.sum(), sum_constraint)
        assert approx_equal(population.individuals[3].DNA.sum(), sum_constraint)

    def test_evolve_no_random_select_no_mutation(self, job):
        sorted_population_DNA = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0]
        ])
        retain = 0.5
        random_select = 0.0
        mutation_chance = 0.0
        population_size = 4
        length = 4
        sum_constraint = 2
        individuals = [Individual(job, length, sum_constraint, sorted_population_DNA[x]) for x in range(4)]
        population = Population(job, population_size, length, sum_constraint)
        population.individuals = individuals

        population.evolve(retain, random_select, mutation_chance)
        assert len(population.individuals) == 4
        # We've kept the first two
        assert np.array_equal(population.individuals[0].DNA, sorted_population_DNA[0])
        assert np.array_equal(population.individuals[1].DNA, sorted_population_DNA[1])


        # We know that the children must be half of one parent and half of the
        # other (as we chose the parents so that their children would be
        # normalised at birth), so test both options.
        expected_child1 = np.array([1.0, 0.0, 0.5, 0.5])
        expected_child2 = np.array([0.5, 0.5, 1.0, 0.0])

        assert ((np.array_equal(population.individuals[2].DNA, expected_child1) and
                 np.array_equal(population.individuals[3].DNA, expected_child2)) or
                (np.array_equal(population.individuals[2].DNA, expected_child2) and
                 np.array_equal(population.individuals[3].DNA, expected_child1)))

    def test_perform_genetic_algorithm(self, job, capsys):
        Job = job
        Job.Model.atomic['A']['U'] = 2.0
        Job.Def['el_kT'] = 0.009
        with capsys.disabled():
            Job = PerformGeneticAlgorithm(Job)

        num_electrons_sq = job.Electron.NElectrons*job.Electron.NElectrons
        residue = Job.Electron.rho - Job.Electron.rhotot
        fitness = np.absolute(residue).sum()/num_electrons_sq

        assert approx_equal(fitness, 0.0)
