"""
@author: Marc Coury

This module was heavily inspired by this source, although extensively modified:
http://lethain.com/genetic-algorithms-cool-name-damn-simple/

# Example usage
from genetic import population, evolve
p_count = 100
i_length = 200
i_sum = 50
p = population(p_count, i_length, i_sum)
fitness_history = []
for i in xrange(100):
    avg_fitness, p = evolve(p, target, data)
    fitness_history.append(avg_fitness))
"""

import numpy as np
from verbosity import verboseprint


def PerformGeneticAlgorithm(Job):
    population_size = Job.Def['population_size']
    individual_length = Job.Hamilton.HSOsize
    num_electrons = Job.Electron.NElectrons
    max_num_evolutions = Job.Def.get('max_num_evolutions', 50)
    genetic_tol = Job.Def.get('genetic_tol', 1.0e-8)
    retain = Job.Def.get('proportion_to_retain', 0.2)
    random_select = Job.Def.get('random_select_chance', 0.05)
    mutation_chance = Job.Def.get('mutation_chance', 0.05)
    best_individual = None

    population = create_population(population_size,
                                   individual_length,
                                   num_electrons)

    for num_generations in range(max_num_evolutions):
        avg_fitness, population = evolve(
            Job,
            population,
            num_electrons,
            retain=retain,
            random_select=random_select,
            mutation_chance=mutation_chance
        )
        # the best individual from the generation just before the evolution
        best_individual = population[0]
        verboseprint(
            Job.Def['verbose'],
            "Generation {}: Average fitness = {}, best fitness = {}".format(
                num_generations,
                avg_fitness,
                best_individual.fitness,
            )
        )
        if best_individual.fitness < genetic_tol:
            break

    # recreate the best individual as that info will be passed back inside Job
    fitness(Job, best_individual)


def can_normalise(individual, sum_constraint):
    """
    This normalisation routine only works if the sum_constraint is less than
    eighty percent of the length of the individual. It also doesn't work on
    individuals whose sum is zero.

    Either raise an error or return True.
    """
    if sum_constraint > 0.8*len(individual):
        error_message = "Cannot normalise individuals where the sum constraint is greater than 80% of the length"
        error_message += "\nHere sum_constraint = {} and length = {}".format(sum_constraint, len(individual))
        raise Exception(error_message)
    num_non_zero_values = 0
    for val in individual:
        if not abs(val) < 1.0e-13:
            num_non_zero_values += 1
    if num_non_zero_values < sum_constraint:
        error_message = "Cannot normalise individual {} as it does not have enough non-zero elements".format(individual)
        raise Exception(error_message)

    return True


def normalise(individual, sum_constraint):
    """
    Normalise given individual so that the sum is equal to sum_constraint but
    so that each value stays in the interval [0.0, 1.0).

    individual: the individual to normalise
    sum_constraint: the value that the sum has to equal
    """
    if can_normalise(individual, sum_constraint):
        while (abs(individual.sum() - sum_constraint) > 1.0e-13):
            individual = individual*(sum_constraint/individual.sum())
            if individual.max() > 1.0:
                # cut the values that are too large in half
                for index in range(individual.size):
                    if individual[index] > 1.0:
                        individual[index] = individual[index]*0.5

    return individual


def create_individual(length, sum_constraint):
    """
    This function returns an array which represents the individual's DNA.
    Each value will be between 0.0 and 1.0 and the sum of the values will be
    constrained by sum_constraint.

    length: the length of the array
    sum_constraint: the sum of the array must equal this value
    """
    # Checks
    sum_constraint = float(sum_constraint)
    length = int(length)
    if sum_constraint < 1:
        raise Exception("It is not physical to instantiate an individual with the sum constraint less than 1")

    # Randomly create the individual
    an_individual = np.random.uniform(size=length)

    return normalise(an_individual, sum_constraint)


def create_population(count, length, sum_constraint):
    """
    Create a number of individuals (i.e. a population).

    count: the number of individuals in the population
    length: the number of values per individual
    sum_constraint: the sum of each individual must equal this value
    """
    a_population = np.empty([count, length])
    for ii in range(count):
        a_population[ii] = create_individual(length, sum_constraint)

    return a_population


def fitness(Job, individual):
    """
    Determine the fitness of an individual. Lower is better.

    Job: the object containing the Electron object, the Hamilton object, and
         the eigenvalues and eigenvectors to an already solved Fock matrix.
    individual: the individual to evaluate
    """
    # construct the density using individual -- into Job.Electron.rho
    Job.Electron.constructDensityMatrixFromOccupation(individual)

    # construct the fock matrix -- into Job.Hamilton.fock
    Job.Hamilton.buildfock()

    # find the eigenvalues & eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(Job.Hamilton.fock)

    # construct the density using the new eigenvalues and eigenvectors -- into Job.Electron.rhotot
    Job.Electron.constructDensityMatrixFromEigenvaluesAndEigenvectors(eigenvalues, eigenvectors)

    num_electrons_sq = Job.Electron.NElectrons*Job.Electron.NElectrons
    residue = Job.Electron.rho - Job.Electron.rhotot

    # compare the densities
    return np.absolute(residue).sum()/num_electrons_sq


def grade(Job, population):
    """
    Return the average fitness for a population and the ordered array of the
    individuals for that population.

    """
    graded_population = [(fitness(Job, x), x) for x in population]
    sorted_population = np.empty(shape=population.shape)
    for index, graded_individual in enumerate(sorted(graded_population, key=lambda graded: graded[0])):
        sorted_population[index] = graded_individual[1]
    average_fitness = sum([x[0] for x in graded_population])/len(graded_population)
    return average_fitness, sorted_population


def survive(sorted_population, retain, random_select):
    """
    This function takes in a population sorted by their fitness and returns the
    fittest individuals and a few random ones too.

    sorted_population: a population sorted by the fitness of the individuals.
    retain: the proportion of the population that survive to reproduce and live
            on in the next generation.
    random_select: the chance to randomly select additional individuals that
                   survive to reproduce and live on in the next generation, it
                   should lie in the interval [0.0, 1.0).
    """
    retain_length = int(len(sorted_population)*retain)
    survivors = sorted_population[:retain_length]
    # randomly add other individuals to
    # promote genetic diversity
    for individual in sorted_population[retain_length:]:
        if random_select > np.random.uniform():
            survivors = np.append(survivors, [individual], axis=0)

    return survivors


def mutate(individual, mutation_chance):
    """
    If a randomly drawn value between 0.0 and 1.0 is less than the threshold,
    mutation_chance, then mutate a randomly chosen index, to a random value
    between 0.0 and 1.0.

    NOTE THAT THIS FUNCTION DOES NORMALISE THE INDIVIDUAL.

    individual: the individual that may or may not get mutated.
    mutation_chance: the threshold (between 0.0 and 1.0) for mutation of the
                     individual.
    """
    new_individual = np.copy(individual)
    if mutation_chance > np.random.uniform():
        pos_to_mutate = np.random.randint(0, len(individual))
        new_individual[pos_to_mutate] = np.random.uniform()

    return new_individual


def can_produce_num_children(num_parents, num_children):
    """
    Check the maximum limit on the number of children that can be produced from
    the given number of parents is not exceeded.

    Either raise an error or return True.
    """
    if num_children > (num_parents*(num_parents - 1)):
        error_message = "It is not possible to create {} children from {} parents".format(num_children, num_parents)
        raise Exception(error_message)

    return True


def reproduce(parents, num_children, sum_constraint, mutation_chance):
    """
    This routine creates an array of the desired number of children by randomly
    crossing the supplied parents with the chance of a mutation.

    parents: the part of the population that has survived to reproduce.
    num_children: the desired number of children.
    sum_constraint: the value that the sum of any individual has to equal.
    mutation_chance: the chance that a mutation occurs in a child, it should
                     lie in the interval [0.0, 1.0).
    """
    num_parents = len(parents)
    individual_len = len(parents[0])
    half = int(individual_len / 2)
    children = np.empty(shape=(num_children, individual_len))
    if can_produce_num_children(num_parents, num_children):
        child_index = 0
        mated_list = []
        while child_index < num_children:
            male = np.random.randint(0, num_parents)
            female = np.random.randint(0, num_parents)
            can_mate = (male != female and (male, female) not in mated_list)
            if can_mate:
                mated_list.append((male, female))
                male = parents[male]
                female = parents[female]
                child = mutate(np.append(male[:half], female[half:]), mutation_chance)
                child = normalise(child, sum_constraint)
                children[child_index] = child
                child_index += 1

    return children


def evolve(Job, population, sum_constraint, retain=0.2, random_select=0.05, mutation_chance=0.05):
    """
    This function takes in the current population and returns the average
    grade of the inputted population and the new generation.

    Job: the object containing the Electron object, the Hamilton object, and
         the eigenvalues and eigenvectors to an already solved Fock matrix.
    population: an array of indiviuals.
    sum_constraint: the value that the sum of any individual has to equal.
    retain: the proportion of the population that survive to reproduce and live
            on in the next generation.
    random_select: the chance to randomly select additional individuals that
                   survive to reproduce and live on in the next generation, it
                   should lie in the interval [0.0, 1.0).
    mutation_chance: the chance that a mutation occurs in a child, it should
                     lie in the interval [0.0, 1.0).
    """
    average_fitness, sorted_population = grade(Job, population)

    parents = survive(sorted_population, retain, random_select)

    num_children = len(population) - len(parents)
    children = reproduce(parents, num_children, sum_constraint, mutation_chance)

    return average_fitness, np.append(parents, children, axis=0)
