import numpy as np

n = 10

# Each gene P[i] is a real number between 0 and 1 and corresponds to the probability to have 1 in the corresponding position.
# P[i] is initialized to 0.5.
P = np.full(n, 0.5)

print('P: ', P)

k = 100
generations = 10
m = 5
condition = True

while condition:
    # In each generation produces a given k number of traditional 0-1 individuals
    # each of them of length n by sampling gene by gene with the probabilistic vector.

    solution_vector = np.random.binomial(1, P, size=[k, n])

    # Each Individual is evaluated by using the function Evaluate(solution_vectors[j]) that calculates its fitness
    fitness = evaluate(solution_vector)

    # Geerates new individuals
    new_solution_vector = np.empty([k, n])

    for i in range(k // 2):
        # Select the best parents in the population for mating
        ind1, ind2 = select_parents(solution_vector, fitness, parents=2)

        # Generate next generation using crossover
        ch1, ch2 = crossover(ind1, ind2)

        # Adding some variations to the offsrping using mutation.
        ch1, ch2 = mutate(ch1), mutate(ch2)

        new_solution_vector[2*i, :] = ch1
        new_solution_vector[2*i+1, :] = ch2

    # At the end of each generation, the best m individuals are used to update the probabilistic vector.
    fitness = evaluate(new_solution_vector)
    best_individuals = np.argsort(fitness)[::-1][:m]

    # The update procedure works like this:
    # for each individual j of the best m chosen,
    # the probabilistic vector is updated using a learning rate LR parameter (that is a real number between 0 and 1)
    # P[i] = P[i] * (1.0 - LR) + solution_vectors[j][i]* (LR); with i=1 to n and j=1 to m.
    LR = np.random.uniform()

    for b in best_individuals:
        P = P * (1.0 - LR) + new_solution_vector[b] * LR

    if stop:
        condition = False
