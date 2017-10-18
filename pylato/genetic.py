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
    # Build fock
    Job.Electron.densitymatrix()
    Job.Hamilton.buildfock()

    # Find the eigenvalues & eigenvectors -- these will be used as the basis
    # this genetic algorithm approach
    Job.e, Job.psi = np.linalg.eigh(Job.Hamilton.fock)

    population_size = Job.Def['population_size']
    individual_length = Job.Hamilton.HSOsize
    num_electrons = Job.Electron.NElectrons
    max_num_evolutions = Job.Def.get('max_num_evolutions', 50)
    genetic_tol = Job.Def.get('genetic_tol', 1.0e-8)
    retain = Job.Def.get('proportion_to_retain', 0.2)
    random_select = Job.Def.get('random_select_chance', 0.05)
    mutation_chance = Job.Def.get('mutation_chance', 0.05)
    best_individual = None

    population = Population(population_size,
                            individual_length,
                            num_electrons)

    for num_generations in range(max_num_evolutions):
        avg_fitness = population.evolve(
            Job,
            retain=retain,
            random_select=random_select,
            mutation_chance=mutation_chance
        )
        # the best individual from the generation just before the evolution
        best_individual = population.individuals[0]
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
    best_individual.calculate_fitness(Job)


class Individual(object):
    def __init__(self, length, sum_constraint, DNA=None):
        self.fitness = None
        self.sum_constraint = float(sum_constraint)

        if DNA is None:
            self.create_random_DNA(length)
        else:
            self.DNA = DNA
            self.normalise()

    def can_normalise(self):
        """
        This normalisation routine only works if the sum_constraint is less than
        eighty percent of the length of the individual. It also doesn't work on
        individuals whose sum is zero.

        Either raise an error or return True.
        """
        if self.sum_constraint > 0.8*len(self.DNA):
            error_message = "Cannot normalise individuals where the sum constraint is greater than 80% of the length"
            error_message += "\nHere sum_constraint = {} and length = {}".format(self.sum_constraint, len(self.DNA))
            raise Exception(error_message)
        num_non_zero_values = 0
        for val in self.DNA:
            if not abs(val) < 1.0e-13:
                num_non_zero_values += 1
        if num_non_zero_values < self.sum_constraint:
            error_message = "Cannot normalise individual {} as it does not have enough non-zero elements".format(self.DNA)
            raise Exception(error_message)

        return True

    def normalise(self):
        """
        Normalise given individual so that the sum is equal to sum_constraint but
        so that each value stays in the interval [0.0, 1.0).

        individual: the individual to normalise
        """
        if self.can_normalise():
            while ((abs(self.DNA.sum() - self.sum_constraint) > 1.0e-13) or
                    self.DNA.max() > 1.0):
                scale_factor = self.sum_constraint/self.DNA.sum()
                self.DNA = self.DNA*scale_factor
                if self.DNA.max() > 1.0:
                    # cut the values that are too large in half
                    for index in range(self.DNA.size):
                        if self.DNA[index] > 1.0:
                            self.DNA[index] = self.DNA[index]*0.5

    def create_random_DNA(self, length):
        """
        This function returns an array which represents the individual's DNA.
        Each value will be between 0.0 and 1.0 and the sum of the values will be
        constrained by sum_constraint.

        length: the length of the array
        """
        # Checks
        length = int(length)
        if self.sum_constraint < 1:
            raise Exception("It is not physical to instantiate an individual with the sum constraint less than 1")

        # Randomly create the individual
        self.DNA = np.random.uniform(size=length)
        self.normalise()

    def mutate(self, mutation_chance):
        """
        If a randomly drawn value between 0.0 and 1.0 is less than the threshold,
        mutation_chance, then mutate a randomly chosen index, to a random value
        between 0.0 and 1.0.

        NOTE THAT THIS FUNCTION DOES NORMALISE THE INDIVIDUAL.

        mutation_chance: the threshold (between 0.0 and 1.0) for mutation of the
                         individual.
        """
        new_DNA = np.copy(self.DNA)
        if mutation_chance > np.random.uniform():
            pos_to_mutate = np.random.randint(0, self.DNA.size)
            new_DNA[pos_to_mutate] = np.random.uniform()

        self.DNA = new_DNA

    def calculate_fitness(self, Job):
        """
        Determine the fitness of an individual. Lower is better.

        Job: the object containing the Electron object, the Hamilton object, and
             the eigenvalues and eigenvectors to an already solved Fock matrix.
        """
        if self.fitness is None:
            # construct the density using individual -- into Job.Electron.rho
            Job.Electron.constructDensityMatrixFromOccupation(self.DNA)

            # construct the fock matrix -- into Job.Hamilton.fock
            Job.Hamilton.buildfock()

            # find the eigenvalues & eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(Job.Hamilton.fock)

            # construct the density using the new eigenvalues and eigenvectors -- into Job.Electron.rhotot
            Job.Electron.constructDensityMatrixFromEigenvaluesAndEigenvectors(eigenvalues, eigenvectors)

            num_electrons_sq = Job.Electron.NElectrons*Job.Electron.NElectrons
            residue = Job.Electron.rho - Job.Electron.rhotot

            # compare the densities
            self.fitness = np.absolute(residue).sum()/num_electrons_sq

    def mate(self, another_individual):
        length = self.DNA.size
        half = int(length / 2)
        return Individual(
            length,
            self.sum_constraint,
            DNA=np.append(self.DNA[:half], another_individual.DNA[half:])
        )


class Population(object):
    def __init__(self, population_size, length, sum_constraint):
        """
        Create a number of individuals (i.e. a population).

        population_size: the number of individuals in the population
        length: the number of values in the DNA of the individuals
        sum_constraint: the sum of the DNA of each individual must equal this
                        value
        """
        self.population_size = population_size
        self.individuals = [
            Individual(length, sum_constraint) for x in range(population_size)
        ]

    def grade(self, Job):
        """
        Calculate the fitness of the population and order the population by the
        fitness of the individuals.

        Return the average fitness of the generation.

        Job: the object containing the Electron object, the Hamilton object, and
             the eigenvalues and eigenvectors to an already solved Fock matrix.
        """
        for index, individual in enumerate(self.individuals):
            # The most expensive step of the genetic algorithm:
            individual.calculate_fitness(Job)
            verboseprint(
                Job.Def['extraverbose'],
                "Graded individual {} out of {}".format(index + 1,
                                                        self.population_size)
            )

        self.individuals = sorted(self.individuals, key=lambda x: x.fitness)

        fitness_sum = sum([x.fitness for x in self.individuals])
        average_fitness = fitness_sum / self.population_size
        return average_fitness

    def survive(self, retain, random_select):
        """
        This method performs survival of the fittest on the population. Only
        the fittest survive. A few extra individuals are randomly chosen to
        survive so as to promote genetic diversity.

        retain: the proportion of the population that survive to reproduce and
                live on in the next generation.
        random_select: the chance to randomly select additional individuals
                       that survive to reproduce and live on in the next
                       generation, it should lie in the interval [0.0, 1.0).
        """
        retain_length = int(self.population_size*retain)
        survivors = self.individuals[:retain_length]

        # randomly add other individuals to the survivors to
        # promote genetic diversity
        for individual in self.individuals[retain_length:]:
            if random_select > np.random.uniform():
                survivors = survivors.append(individual)

        self.individuals = survivors

    def can_produce_num_children(self):
        """
        Check the maximum limit on the number of children that can be produced
        from the given number of parents is not exceeded.

        Either raise an error or return True.
        """
        num_parents = len(self.individuals)
        num_children = self.population_size - num_parents

        if num_children > (num_parents*(num_parents - 1)):
            error_message = "It is not possible to create {} children from {} parents".format(num_children, num_parents)
            raise Exception(error_message)

        return True

    def reproduce(self, mutation_chance):
        """
        This routine creates an array of the desired number of children by
        randomly crossing the supplied parents with the chance of a mutation.

        mutation_chance: the chance that a mutation occurs in a child, it
                         should lie in the interval [0.0, 1.0).
        """
        num_parents = len(self.individuals)
        if self.can_produce_num_children():
            mated_list = []
            while len(self.individuals) < self.population_size:
                male = np.random.randint(0, num_parents)
                female = np.random.randint(0, num_parents)
                can_mate = (male != female and (male, female) not in mated_list)
                if can_mate:
                    mated_list.append((male, female))
                    male = self.individuals[male]
                    female = self.individuals[female]
                    child = female.mate(male)
                    child.mutate(mutation_chance)
                    child.normalise()
                    self.individuals.append(child)

    def evolve(self, Job, retain=0.2, random_select=0.05, mutation_chance=0.05):
        """
        This function takes in the current population and returns the average
        grade of the inputted population and the new generation.

        Job: the object containing the Electron object, the Hamilton object, and
             the eigenvalues and eigenvectors to an already solved Fock matrix.
        retain: the proportion of the population that survive to reproduce and
                live on in the next generation.
        random_select: the chance to randomly select additional individuals
                       that survive to reproduce and live on in the next
                       generation, it should lie in the interval [0.0, 1.0).
        mutation_chance: the chance that a mutation occurs in a child, it
                         should lie in the interval [0.0, 1.0).
        """
        average_fitness = self.grade(Job)

        self.survive(retain, random_select)

        self.reproduce(mutation_chance)

        return average_fitness
