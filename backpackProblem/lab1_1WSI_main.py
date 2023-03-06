"""
Author: Michał Kwarciński
"""
import random
import time
from lab1_1WSI_solvers import ExhaustingSolver, HeuristicSolver


def generate_parameters(number_to_generate, low_limit=1, high_limit=10):
    weights = []  # weights of items
    for i in range(number_to_generate):
        weights.append(random.randint(low_limit, high_limit))
    values = []  # values of item
    for i in range(number_to_generate):
        values.append(random.randint(low_limit, high_limit))
    capacity = random.randint(int(high_limit / 2)+1, 2*high_limit)  # maximal weights of backpack
    return weights, values, capacity


def print_parameters(weights, values, capacity):
    text = "Parameters: \n"
    text += f'Items weights: {[weight for weight in weights]}\n'
    text += f'Items values: {[value for value in values]}\n'
    text += f'Backpack capacity: {capacity}\n'
    text += f'{25 * "#"}'
    return text


def get_average_time(time_data):
    average_time_exhausting = []
    for i in range(len(time_data[0])):
        average_time_exhausting.append(round(sum([x[i] for x in time_data]) / len(time_data), 2))
    return average_time_exhausting


def get_average_time_different(time_data):
    """
    Returns the average of time ratios between next non-zero times given
    """
    time_different = []
    for i in range(len(time_data) - 1):
        if time_data[i] != 0:
            time_different.append(time_data[i + 1] / time_data[i])
    return round(sum(time_different)/(len(time_different)), 2)


def function_with_given_data(weights, values, capacity):
    """
    Solves the problem for task data using two solvers
    """
    e = ExhaustingSolver(weights, values, capacity)
    h = HeuristicSolver(weights, values, capacity)

    best_option_exhausting = e.find_best_option()
    best_option_heuristic = h.find_best_option()

    print(f'Best option found using exhausting solver: {best_option_exhausting[0]}, weight: {best_option_exhausting[1]}'
          f', value: {best_option_exhausting[2]}')
    print(f'Best option found using heuristic solver: {best_option_heuristic[0]}, weight: {best_option_heuristic[1]}'
          f', value: {best_option_heuristic[2]}')


def function_with_single_random_data(number_of_random_items_generated):
    """
    Solves the problem using two solvers for random items in quantity X
    """
    # random generate parameters
    weights, values, capacity = generate_parameters(number_of_random_items_generated)
    print(print_parameters(weights, values, capacity))

    e = ExhaustingSolver(weights, values, capacity)
    h = HeuristicSolver(weights, values, capacity)

    t_ex_start = time.time()
    best_option_exhausting = e.find_best_option()
    t_ex_end = time.time() - t_ex_start

    t_he_start = time.time()
    best_option_heuristic = h.find_best_option()
    t_he_end = time.time() - t_he_start

    print(f'Best option found using exhausting solver: {best_option_exhausting[0]}, weight: {best_option_exhausting[1]}'
          f', value: {best_option_exhausting[2]}')
    print("Exhausting solver time: ", round(t_ex_end, 2), "[s]")
    print(f'Best option found using heuristic solver: {best_option_heuristic[0]}, weight: {best_option_heuristic[1]}'
          f', value: {best_option_heuristic[2]}')
    print("Heuristic solver time: ", round(t_he_end, 2), "[s]")


def function_with_single_random_data_only_heuristic(number_of_random_items_generated):
    """
    Prints the time of solving the problem using heuristic solver for random items in quantity X
    """
    # random generate parameters
    weights, values, capacity = generate_parameters(number_of_random_items_generated)

    h = HeuristicSolver(weights, values, capacity)

    t_he_start = time.time()
    best_option_heuristic = h.find_best_option()
    t_he_end = time.time() - t_he_start

    print("Heuristic solver time: ", round(t_he_end, 2), "[s]")


def function_with_x_iteration_of_random_data(number_of_iterations, max_time_of_iteration):
    """
    Solves the problem for random items using exhausting solver.
    Number of random items is increasing until the solving time exceeds max_time_of_iteration seconds
    Function iterates number_of_iteration times
    """
    exhausting_time = []
    i = 0
    while i < number_of_iterations:
        print("i: ", i)
        ex = []
        tend = 0
        j = 0
        while tend < max_time_of_iteration:
            print("i: ", i, "j: ", j)
            # random generate parameters
            weights, values, capacity = generate_parameters(j)
            # print(print_parameters(weights, values, capacity))

            e = ExhaustingSolver(weights, values, capacity)

            # exhausting problem
            t1 = time.time()
            best_option_exhausting = e.find_best_option()
            tend = time.time() - t1
            ex.append(tend)
            print("Exhausting solver time: ", round(tend, 2), "s")
            print(f'Best option found using exhausting solver: {best_option_exhausting[0]},'
                  f' weight: {best_option_exhausting[1]}, value: {best_option_exhausting[2]}')
            j += 1
        exhausting_time.append(ex)
        i += 1
    return exhausting_time


if __name__ == "__main__":
    function_with_given_data(weights=[8, 3, 5, 2], values=[16, 8, 9, 6], capacity=9)
    # function_with_single_random_data(number_of_random_items_generated=26)
    # function_with_single_random_data_only_heuristic(37500000)
    exhausting_time = function_with_x_iteration_of_random_data(number_of_iterations=2, max_time_of_iteration=5)
    average_times = get_average_time(exhausting_time)
    average_time_different = get_average_time_different(average_times)
    print(exhausting_time, "\n", average_times, "\n", average_time_different)

