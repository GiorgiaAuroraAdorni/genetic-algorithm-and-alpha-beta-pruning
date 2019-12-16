import numpy as np

n = 10

# Each gene P[i] is a real number between 0 and 1 and corresponds to the probability to have 1 in the corresponding position.
# P[i] is initialized to 0.5.
P = np.full(n, 0.5)

print('P: ', P)

# The algorithm in each generation produces a given k number of traditional 0-1 individuals
# (solution_vectors[j] with j=1 to k) each of them of length n by sampling gene by gene with the probabilistic vector.
k = 5
generations = 10
condition = True

while condition:
    for _ in range(generations):
        solution_vector = np.random.binomial(1, P, size=[k, n])

        print(solution_vector)

        new_solution_vector = np.zeros([k, n])

        # Each Individual is evaluated by using the function Evaluate(solution_vectors[j]) that calculates its fitness
        fitness = evaluate(solution_vector)

        for individual in solution_vector:
            
            # Select the best parents in the population for mating
            ind1, ind2 = select_parents(solution_vector, fitness)

            # Generate next generation using crossover
            ch1, ch2 = crossover(ind1, ind2)

            # Adding some variations to the offsrping using mutation.
            ind = mutate(ind1)

            new_solution_vector[individual] = ind

        # At the end of each generation, the best m individuals are used to update the probabilistic vector.

        # The update procedure works like this:
        # for each individual j of the best m chosen,
        # the probabilistic vector is updated using a learning rate LR parameter (that is a real number between 0 and 1)
        # P[i] = P[i] * (1.0 - LR) + solution_vectors[j][i]* (LR); with i=1 to n and j=1 to m.
        LR = np.random.uniform()
        P = P * (1.0 - LR) + new_solution_vector * LR

        if stop:
            condition = False
