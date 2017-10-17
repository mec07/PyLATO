"""
Tests for the genetic module
"""
from pylato.genetic import (can_produce_num_children,
                            create_individual,
                            create_population,
                            evolve,
                            fitness,
                            grade,
                            mutate,
                            normalise,
                            reproduce,
                            survive)
from pylato.hamiltonian import Hamiltonian
from pylato.electronic import Electronic
from tests.utils import GenericClass, return_a_function, approx_equal

import pytest
import numpy as np


@pytest.fixture
def job():
    Job = GenericClass(
        AtomType=['A', 'A'],
        Def={
            'el_kT': 0.0,
            'Hamiltonian': 'noncollinear',
            'mu_tol': 1e-13,
            'mu_max_loops': 5000,
            'num_rho': 1,
            'PBC': 0,
            'so_eB': [0, 0, 0],
            'spin_orbit': 0,
        },
        Model=GenericClass(
            atomic={
                'A': {
                    'NOrbitals': 1,
                    'e': [[0]],
                    'l': [0],
                    'NElectrons': 1,
                    'I': 0,
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
    Job.Hamilton.buildHSO()

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


class TestGenetic:
    def test_normalise(self):
        example1 = np.array([0.1, 0.2, 0.3, 0.4])
        example2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        example3 = np.random.uniform(size=200)
        sum_constraint1 = 3
        sum_constraint2 = 1
        sum_constraint3 = 20
        normalised_example1 = normalise(example1, sum_constraint1)
        normalised_example2 = normalise(example2, sum_constraint2)
        normalised_example3 = normalise(example3, sum_constraint3)

        assert approx_equal(normalised_example1.sum(), sum_constraint1)
        assert approx_equal(normalised_example2.sum(), sum_constraint2)
        assert approx_equal(normalised_example3.sum(), sum_constraint3)

    def test_create_individual(self):
        length = 100
        sum_constraint = 10
        min_value = 0.0
        max_value = 1.0
        an_individual = create_individual(length, sum_constraint)

        assert len(an_individual) == length
        assert an_individual.min() >= min_value
        assert an_individual.max() <= max_value
        assert approx_equal(an_individual.sum(), sum_constraint)

    def test_create_individual_checks(self):
        length = 100
        # Raises for a constraint greater than 80% of length
        with pytest.raises(Exception):
            create_individual(length, 0.9*length)
        # Raises for a constraint less than 1 of length
        with pytest.raises(Exception):
            create_individual(length, 0.1)
        # Raises if length is not an int
        with pytest.raises(Exception):
            create_individual('hi', 2)
        # Raises if constraint is not a number
        with pytest.raises(Exception):
            create_individual(length, 'hi')

    def test_create_population(self):
        count = 50
        length = 100
        sum_constraint = 10
        min_value = 0.0
        max_value = 1.0

        a_population = create_population(count, length, sum_constraint)

        assert a_population.shape == (count, length)
        for ii in range(count):
            assert a_population[ii].min() >= min_value
            assert a_population[ii].max() <= max_value
            assert approx_equal(a_population[ii].sum(), sum_constraint)

    def test_fitness_groundstate(self, job):
        # fake a Job and Hamiltonian
        groundstate_occupation = np.array([1, 1, 0, 0])

        the_fitness = fitness(job, groundstate_occupation)

        # As we've set the density depedent terms to zero, the temperature to
        # zero and chosen the individual to be the groundstate occupation, the
        # the fitness must be zero.
        assert approx_equal(the_fitness, 0.0)

    def test_fitness_non_groundstate1(self, job):
        # fake a Job and Hamiltonian
        excited_occupation = np.array([0, 1, 0, 1])

        the_fitness = fitness(job, excited_occupation)

        # As we've set the density depedent terms to zero, the temperature to
        # zero and chosen the individual to be an excited occupation, the
        # the fitness can be worked out to be 0.5 by hand.
        assert approx_equal(the_fitness, 0.5)

    def test_fitness_non_groundstate2(self, job):
        # fake a Job and Hamiltonian
        excited_occupation = np.array([0, 1, 1, 0])

        the_fitness = fitness(job, excited_occupation)

        # As we've set the density depedent terms to zero, the temperature to
        # zero and chosen the individual to be an excited occupation, the
        # the fitness can be worked out to be 1.0 by hand.
        assert approx_equal(the_fitness, 1.0)

    def test_fitness_non_groundstate3(self, job):
        # fake a Job and Hamiltonian
        excited_occupation = np.array([0, 0, 1, 1])

        the_fitness = fitness(job, excited_occupation)

        # As we've set the density depedent terms to zero, the temperature to
        # zero and chosen the individual to be an excited occupation, the
        # the fitness can be worked out to be 1.0 by hand.
        assert approx_equal(the_fitness, 1.0)

    def test_grade(self, job):
        population = np.array([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0]
        ])
        average_fitness, sorted_population = grade(job, population)

        assert (average_fitness - 0.625) < 1e-8
        assert np.array_equal(sorted_population[0], population[3])
        assert np.array_equal(sorted_population[1], population[1])
        # The other two both have a fitness of 1.0 so can be in either order
        assert ((np.array_equal(sorted_population[2], population[0]) and
                 np.array_equal(sorted_population[3], population[2])) or
                (np.array_equal(sorted_population[2], population[2]) and
                 np.array_equal(sorted_population[3], population[0])))

    def test_survive_no_random_select(self):
        sorted_population = np.array([
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 1]
        ])
        retain = 0.5
        random_select = 0.0

        survivors = survive(sorted_population, retain, random_select)
        assert len(survivors) == 2
        assert np.array_equal(survivors[0], sorted_population[0])
        assert np.array_equal(survivors[1], sorted_population[1])

    def test_survive_with_full_select(self):
        sorted_population = np.array([
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 1]
        ])
        retain = 0.5
        random_select = 1.0

        survivors = survive(sorted_population, retain, random_select)
        assert len(survivors) == 4
        assert np.array_equal(survivors[0], sorted_population[0])
        assert np.array_equal(survivors[1], sorted_population[1])
        assert np.array_equal(survivors[2], sorted_population[2])
        assert np.array_equal(survivors[3], sorted_population[3])

    def test_mutate_no_mutation(self):
        individual = create_individual(100, 10)
        individual_after_mutate = mutate(individual, 0.0)

        assert np.array_equal(individual, individual_after_mutate)

    def test_mutate_definitely(self, capsys):
        length = 6
        individual = create_individual(length, 3)
        with capsys.disabled():
            individual_after_mutate = mutate(individual, 1.0)

        # Expect that exactly one of the values has changed
        assert num_values_changed(individual, individual_after_mutate) == 1

    def test_can_produce_num_children(self):
        assert can_produce_num_children(10, 50) is True

        # Raises an exception when there aren't enough parents to produce the
        # desired number of children.
        with pytest.raises(Exception):
            can_produce_num_children(3, 10)

    def test_reproduce_no_mutation(self):
        parents = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 0.0, 1.0, 0.0]
        ])
        num_children = 2
        sum_constraint = 2.0
        half = 2
        mutation_chance = 0.0

        children = reproduce(parents, num_children, sum_constraint, mutation_chance)

        assert children.shape == (num_children, 4)
        # We know that the children must be half of one parent and half of the
        # other, so test both options
        expected_child1 = normalise(np.append(parents[0][:half], parents[1][half:]), sum_constraint)
        expected_child2 = normalise(np.append(parents[1][:half], parents[0][half:]), sum_constraint)
        assert ((np.array_equal(children[0], expected_child1) and
                 np.array_equal(children[1], expected_child2)) or
                (np.array_equal(children[0], expected_child2) and
                 np.array_equal(children[1], expected_child1)))

    def test_reproduce_with_mutation(self):
        parents = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 0.0, 1.0, 0.0]
        ])
        num_children = 2
        sum_constraint = 2.0
        half = 2
        mutation_chance = 1.0

        children = reproduce(parents, num_children, sum_constraint, mutation_chance)

        expected_child1 = np.append(parents[0][:half], parents[1][half:])
        expected_child2 = np.append(parents[1][:half], parents[0][half:])

        assert children.shape == (num_children, 4)
        # The mutated children will be different from the expected children. To
        # make sure of that we test to see that they are not the same, but we
        # don't know which way around the children will be, so we check both
        # options.
        assert ((not np.array_equal(children[0], expected_child1) and
                 not np.array_equal(children[1], expected_child2)) or
                (not np.array_equal(children[0], expected_child2) and
                 not np.array_equal(children[1], expected_child1)))

        # We also check the normalisation has worked correctly:
        assert approx_equal(children[0].sum(), sum_constraint)
        assert approx_equal(children[1].sum(), sum_constraint)

    def test_evolve_no_random_select_no_mutation(self, job, capsys):
        population = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.9, 0.1, 1.0]
        ])
        retain = 0.5
        random_select = 0.0
        mutation_chance = 0.0
        half = 2
        sum_constraint = 2

        expected_child1 = normalise(np.append(population[0][:half], population[1][half:]), sum_constraint)
        expected_child2 = normalise(np.append(population[1][:half], population[0][half:]), sum_constraint)

        with capsys.disabled():
            average_fitness, new_population = evolve(
                job,
                population,
                sum_constraint,
                retain=retain,
                random_select=random_select,
                mutation_chance=mutation_chance
            )
            print("new_population = {}".format(new_population))

        assert np.array_equal(new_population[0], population[0])
        assert np.array_equal(new_population[1], population[1])
        assert ((np.array_equal(new_population[2], expected_child1) and
                 np.array_equal(new_population[3], expected_child2)) or
                (np.array_equal(new_population[2], expected_child2) and
                 np.array_equal(new_population[3], expected_child1)))
