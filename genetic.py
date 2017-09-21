"""
A genetic algorithm that I found and modified.

Source: http://lethain.com/genetic-algorithms-cool-name-damn-simple/

# Example usage
from genetic import population, grade, evolve
target = 371
p_count = 100
i_length = 200
p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target, data),]
for i in xrange(100):
    p = evolve(p, target, data)
    fitness_history.append(grade(p, target, data))

for datum in fitness_history:
   print datum
"""
from random import randint, random
from operator import add
from math import copysign

def individual(length, target, data):
    """Create a member of the population. Start from a 0 vector and
    keep adding 1s until the value of the target has been exceeded."""
    individual = [0 for ii in range(length)]
    for ii in range(length):
        pos_to_mutate = randint(0, len(individual)-1)
        individual[pos_to_mutate] = 1
        if fitness(individual, target, data)[1] == -1:
            break

    return individual

def population(count, length, target, data):
    """
    Create a number of individuals (i.e. a population).

    count: the number of individuals in the population
    length: the number of values per individual
    min: the minimum possible value in an individual's list of values
    max: the maximum possible value in an individual's list of values

    """
    return [ individual(length, target, data) for x in xrange(count) ]

def fitness(individual, target, data):
    """
    Determine the fitness of an individual. Lower is better.

    individual: the individual to evaluate
    target: the target number individuals are aiming for
    data: the values that are used to calculate the sum
    """
    sum = 0
    for ii in range(len(individual)):
        sum += individual[ii]*data[ii]
    return abs(target-sum), copysign(1, target-sum)

def best_individual(population, target, data):
    """
    Find the best individual from a population by selecting the one 
    with the smallest value of fitness that also has a negative sign.
    """
    best = [(None, 1e20)] # start best off at a very large value
    for ii in range(len(population)):
        fit, sgn = fitness(population[ii], target, data)
        # print sgn*fit
        if fit < best[0][1]:
            if sgn == -1.0:
                best[0] = (ii, fit)

    try:
        individual = population[best[0][0]]
        err = 0
    except TypeError:
        individual = []
        err = 1
    return individual, err

def grade(pop, target, data):
    'Find average fitness for a population.'
    summed = reduce(add, (fitness(x, target, data)[0] for x in pop))
    return summed / (len(pop) * 1.0)

def evolve(pop, target, data, retain=0.2, random_select=0.05, mutate=0.05):
    graded = [ (fitness(x, target, data)[0], x) for x in pop]
    graded = [ x[1] for x in sorted(graded)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]
    # randomly add other individuals to
    # promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)
    # mutate some individuals
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)
            # this mutation is not ideal, because it
            # restricts the range of possible values,
            # but the function is unaware of the min/max
            # values used to create the individuals,
            individual[pos_to_mutate] = randint(
                min(individual), max(individual))
    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) / 2
            child = male[:half] + female[half:]
            children.append(child)
    parents.extend(children)
    return parents
