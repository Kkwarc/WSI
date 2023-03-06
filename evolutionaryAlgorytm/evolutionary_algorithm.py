import numpy as np
import random

from cec2017.functions import f4, f5


class Points:
    def __init__(self, dimensions, number_of_points, lower_limit, upper_limit, function, budget):
        self.dimension = dimensions
        self.number_of_points = number_of_points
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.function = function
        self.points = self.generate_points()
        self.sort_points()
        self.budget = budget

    def generate_points(self):
        points = []
        for i in range(self.number_of_points):
            p = Point(np.random.uniform(self.lower_limit, self.upper_limit, size=self.dimension), self.function)
            points.append(p)
        return points

    def sort_points(self):
        return self.points.sort(key=lambda point: point.get_function_value())

    def sort_list_of_points(self, list_of_points):
        return list_of_points.sort(key=lambda point: point.get_function_value())

    def find_the_best(self):
        return self.points[0]

    def find_the_worst(self):
        return self.points[-1]

    def find_sum_of_tick(self):
        sum = 0
        for point in self.points:
            sum += point.get_function_value()
        return sum

    def get_list_of_all_points_value(self):
        points_value = []
        for point in self.points:
            points_value.append(point.get_function_value())
        return points_value

    def reproduction(self):
        rep = []
        self.sort_points()
        while len(rep) < self.number_of_points:
            random_number_1 = random.randint(0, self.number_of_points-1)
            random_number_2 = random.randint(0, self.number_of_points-1)
            winner = min(random_number_1, random_number_2)
            point = self.points[winner]
            p = Point(value=point.get_value(), function=self.function, value_of_function=point.get_function_value())
            rep.append(p)
        return rep

    def mutation(self, mutation_strength):
        rep = self.reproduction()
        mut = []
        for point in rep:
            point.change_value(point.get_value() + mutation_strength * np.random.normal(0, 1, self.dimension))
            mut.append(point)
        self.sort_list_of_points(mut)
        return mut

    def succession(self, mutation_strength, how_many_old_lived):
        suc = self.mutation(mutation_strength)
        if how_many_old_lived != 0:
            suc = suc[:-how_many_old_lived]
            self.sort_points()
            for j in range(0, how_many_old_lived):
                suc.append(self.points[j])
        self.sort_list_of_points(suc)
        self.points = suc

    def find_minimum(self, mutation_strength, how_many_old_lived):
        i = 0
        while i < self.budget/self.number_of_points:
            self.succession(mutation_strength, how_many_old_lived)
            i += 1
        best_function_value = self.find_the_best().get_function_value()
        print(best_function_value)
        return best_function_value


class Point:
    def __init__(self, value, function=None, value_of_function=None):
        self.value = value
        self.function = function
        if value_of_function is None:
            if function is None:
                self.function_value = 0
            else:
                self.function_value = function(value)
        else:
            self.function_value = value_of_function

    def get_value(self):
        return self.value

    def get_function(self):
        return self.function

    def change_function(self, new_function):
        self.function = new_function

    def change_value(self, new_value):
        self.value = new_value
        self.function_value = self.function(new_value)

    def get_function_value(self):
        return self.function_value


if __name__ == "__main__":
    number_of_iteration = 25
    all_iteration_best_function_values = []
    for i in range(number_of_iteration):
        P = Points(dimensions=10, number_of_points=20, lower_limit=-100, upper_limit=100, function=f4, budget=10000)
        best_function_value = P.find_minimum(mutation_strength=2, how_many_old_lived=1)
        all_iteration_best_function_values.append(best_function_value)
        print(f'{25*"#"}')
    average_values = round(sum(all_iteration_best_function_values)/number_of_iteration, 2)
    best_function_values = round(min(all_iteration_best_function_values), 2)
    worst_function_values = round(max(all_iteration_best_function_values), 2)
    standard_deviation = round(np.std(all_iteration_best_function_values), 2)
    print(f'Average values: {average_values}\nBest function value: {best_function_values}\n'
          f'Worst function value: {worst_function_values}\nStandard deviation: {standard_deviation}')

