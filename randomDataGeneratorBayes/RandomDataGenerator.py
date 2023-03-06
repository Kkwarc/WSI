import numpy as np
import itertools
import pandas as pd
from ID3.tree import Tree

data = {}
k = 0


def load_data(file_name):
    nodes = []
    with open(file_name) as file:
        text = file.read()
        lines = text.splitlines()
        for line in lines:
            print(line)
            line_parts = line.split(",")
            print(line_parts)

            # for saving all arguments
            data[line_parts[0]] = []

            if line_parts[1] == 'None':
                inputs_nodes = None
                probabilities = [float(line_parts[2])]
            else:
                index = 1 + int(line_parts[1])
                inputs_nodes_names = line_parts[2:2+int(line_parts[1])]
                probabilities = [float(param) for param in line_parts[1+index:]]
                inputs_nodes = []
                for name in inputs_nodes_names:
                    for node in nodes:
                        if name == node.name:
                            inputs_nodes.append(node)
            n = Node(line_parts[0], inputs_nodes, probabilities)
            nodes.append(n)
    return nodes, data


class Node:
    def __init__(self, name, inputs, probability):
        self.name = name
        self.input = inputs
        self.probability = probability

    def get_value(self):
        if self.input is None:
            x = self.return_value_index(0)
        else:
            inputs = self.input
            values = [input.get_value() for input in inputs]
            all_possibilities = list(itertools.product([False, True], repeat=len(values)))
            all_possibilities.reverse()
            for j, possibility in enumerate(all_possibilities):
                for i in range(len(possibility)):
                    if values[i] is not possibility[i]:
                        break
                    if i == len(possibility)-1:
                        x = self.return_value_index(j)

        print(f'Values of node {self.name}: {x}')
        # saving data
        data[self.name].append(x)
        return x

    def return_value_index(self, index):
        random_int = np.random.randint(0, 100)
        if random_int > self.probability[index] * 100:
            return False
        else:
            return True


if __name__ == "__main__":
    error_data = []
    for time in range(1):
        nodes, data = load_data("settings.txt")
        iters = 1000

        for k in range(iters):
            nodes[-1].get_value()
            print(f'{25*"#"}')
        true_count = data["Ache"].count(True)
        false_count = data["Ache"].count(False)
        print(f'True value: {true_count}')
        print(f'False value: {false_count}')
        print(f'True ratio: {round(true_count/iters, 2)}')
        print(f'{25 * "#"}')
        # writing to txt
        with open('data.data', 'w') as f:
            lines = []
            for l in range(iters):
                lines.append(f'{data["Ache"][l]},{data["Back"][l]},{data["Chair"][l]},{data["Sport"][l]}\n')
            f.writelines(lines)

        # testing tree
        file_name = "data.data"
        values = [True, False]

        data_test = pd.read_csv(file_name, header=None, delimiter=",")
        print(list(data_test.iloc[:, 0]).count(False))
        print(list(data_test.iloc[:, 1]).count(False))
        print(list(data_test.iloc[:, 2]).count(False))
        print(list(data_test.iloc[:, 3]).count(False))

        error = []
        matrix = []
        for _ in range(20):
            train = data_test.sample(frac=0.6)
            test = data_test.drop(train.index)
            t = Tree(train, values)
            error_matrix, error_rate = t.prediction(test)
            print(error_rate)
            error.append(error_rate)
            print(error_matrix)
            matrix.append(error_matrix)
        print(f'{25 * "#"}')
        print(round(1 - (sum(error) / len(error)), 2))
        print([sum([x[i] for x in matrix]) / len(matrix) for i in range(4)])
        error_data.append(round(1 - (sum(error) / len(error)), 2))

    print(f'{25 * "#"}')
    print(f'Average error: {round(sum(error_data)/len(error_data),2)}')
    print(f'Best: {max(error_data)}')
    print(f'Worst: {min(error_data)}')
    print(f'Std: {round(np.std(error_data),2)}')
