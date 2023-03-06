from autograd import grad
import numpy as np
import matplotlib.pyplot as plt

from cec2017.functions import f1, f2, f3


def starting_point(lower_bound, upper_bound, size):
    return np.random.uniform(lower_bound, upper_bound, size=size)


def algorithm(starting_point, function, step, error, iteration_limit):
    data = []
    x = starting_point
    y = 0
    i = 0
    while i < iteration_limit:
        print("I: ", i, ", X(k): ", x, ", X(k-1): ", y)
        grad_fct = grad(function)
        gradient = grad_fct(x)
        x = x - step*gradient
        diff = x - y
        data.append(x)
        if all(abs(dif) < error for dif in diff):
            print("end")
            break
        y = x
        i += 1
    final_function_value = function(data[-1])
    print(f'"Starting point: {[round(x, 2) for x in data[0]]}\nOptimal point: {[round(x, 2) for x in data[-1]]}\nFunction value: {round(final_function_value, 2)}')
    return data, final_function_value


def print_results(lower_bound, upper_bound, plot_step, function, data, function_value, beta, dimensions=[0, 1]):
    dim1 = dimensions[0]
    dim2 = dimensions[1]
    x_arr = np.arange(lower_bound, upper_bound, plot_step)
    y_arr = np.arange(lower_bound, upper_bound, plot_step)
    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.empty(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = function([X[i, j], Y[i, j]])
    plt.contour(X, Y, Z, 40)
    for i in range(len(data) - 1):
        plt.arrow(data[i][dim1], data[i][dim2], data[i + 1][dim1] - data[i][dim1], data[i + 1][dim2] - data[i][dim2],
                  head_width=0.01*upper_bound, head_length=0.02*upper_bound, fc='k', ec='k')
    plt.title(f'Function value: {round(function_value, 2)}, Beta: {beta}')
    plt.savefig('b1.png')
    plt.show()


if __name__ == "__main__":
    # setup for f1, f2, f3

    LOWER_LIMIT = -100
    UPPER_LIMIT = 100
    SIZE = 10
    FUNCTION = f1  # f1, f2, f3
    #  |2 dimension| f1 -> -7, f2 -> -1, f3 -> -5 |10 dimension| f1 -> -8, f2 -> -18, f3-> -9
    ALGORITHM_STEP = 10**(-8)
    ERROR = ALGORITHM_STEP*10**(-1)
    MAX_NUMBER_ITERATIONS = 1000
    PLOT_STEP = 0.5
    random_starting_point = starting_point(LOWER_LIMIT, UPPER_LIMIT, SIZE)
    data, final_function_value = algorithm(random_starting_point, FUNCTION,
                                           ALGORITHM_STEP, ERROR, MAX_NUMBER_ITERATIONS)
    print_results(LOWER_LIMIT, UPPER_LIMIT, PLOT_STEP, FUNCTION, data, final_function_value, ALGORITHM_STEP, dimensions=[0, 1])

    # setup for booth function

    # def booth(x):
    #     return (x[0]+2*x[1]-7)**2+(2*x[0]+x[1]-5)**2
    # LOWER_LIMIT = -100
    # UPPER_LIMIT = 100
    # SIZE = 2
    # FUNCTION = booth
    # ALGORITHM_STEP = 10**(-2)
    # ERROR = ALGORITHM_STEP*10**(-2)
    # MAX_NUMBER_ITERATIONS = 750
    # PLOT_STEP = 0.1
    # random_starting_point = starting_point(LOWER_LIMIT, UPPER_LIMIT, SIZE)
    # data, final_function_value = algorithm(random_starting_point, FUNCTION,
    #                                        ALGORITHM_STEP, ERROR, MAX_NUMBER_ITERATIONS)
    # print_results(LOWER_LIMIT, UPPER_LIMIT, PLOT_STEP, FUNCTION, data, final_function_value, ALGORITHM_STEP)

