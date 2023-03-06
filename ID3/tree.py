import pandas as pd
import numpy as np
from copy import deepcopy


class Tree:
    def __init__(self, data, predictions):
        self.data = data
        self.tree = None
        self.predictions = predictions
        self.exit = {}
        self.index = None
        self.make_tree()

    def get_entropy(self, values):
        _, counts = np.unique(values, return_counts=True)
        entropy = 0
        for count in counts:
            entropy += -count/np.sum(counts)*np.log(count/np.sum(counts))
        return entropy

    def colletion_entropy(self, sorted_params):
        entropy = 0
        val = list(sorted_params.values())
        len = 0
        for tab in val:
            len += sum(tab)
        for key in sorted_params.keys():
            ent = 0
            par = sorted_params[key]
            if par[0] == 0 or par[1] == 0:
                continue
            for count in par:
                ent += count/np.sum(par)*np.log(count/np.sum(par))
            entropy += (sum(par)/len)*ent
        return -entropy

    def sort_params(self, param, values):
        counter = {}
        for i, par in enumerate(param):
            if par in counter.keys():
                if values[i] == self.predictions[0]:
                    counter[par][0] += 1
                else:
                    counter[par][1] += 1
            else:
                if values[i] == self.predictions[1]:
                    counter[par] = [0, 1]
                else:
                    counter[par] = [1, 0]
        return counter

    def sort_data(self, data):
        values = list(data.iloc[:, 0])
        params = [list(data.iloc[:, i+1]) for i in range(len(data.columns)-1)]

        entropy = self.get_entropy(values)
        gain = []
        for param_index, param in enumerate(params):
            sorted_params = self.sort_params(param, values)
            collection_entropy = self.colletion_entropy(sorted_params)
            inf_gain = entropy - collection_entropy
            gain.append([inf_gain, param_index])
        gain.sort(key=lambda x: x[0], reverse=True)
        order = [i[1] + 1 for i in gain]
        order.insert(0, 0)
        return data.iloc[:, order]

    def make_tree(self):
        data = self.sort_data(self.data)
        column_names = list(data.columns)
        column_name = column_names[1]
        self.index = column_name - 1
        params = self.sort_params(list(data.iloc[:, 1]), list(data.iloc[:, 0]))

        if len(column_names) > 2:
            for key in params.keys():
                if params[key][0] == 0:
                    self.exit[key] = 0
                elif params[key][1] == 0:
                    self.exit[key] = 1
                else:
                    next_data = deepcopy(data)
                    next_data = next_data.loc[lambda next_data: next_data[column_name] == key]
                    next_data.pop(column_name)
                    self.exit[key] = Tree(next_data, self.predictions)
        else:
            for key in params.keys():
                if params[key][0] == 0:
                    self.exit[key] = 0
                elif params[key][1] == 0:
                    self.exit[key] = 1
                else:
                    self.exit[key] = 1 if params[key][0] > params[key][1] else 0  ##

    def predict(self, attributes):
        if attributes[self.index] in self.exit.keys():
            pre = self.exit[attributes[self.index]]
        else:
            return None
        if type(pre) is Tree:
            pre = pre.predict(attributes)
        return pre

    def prediction(self, data):
        error_matrix = [0, 0, 0, 0]  # Tp, Fp, FN, TN
        for value in data.values:
            params = list(value[1:])
            values = value[0]
            prediction = self.predict(params)
            predict_val = 1 if self.predictions[0] == values else 0
            if predict_val == prediction == 1:
                error_matrix[0] += 1
            elif predict_val == prediction == 0:
                error_matrix[3] += 1
            elif predict_val != prediction == 1:
                error_matrix[1] += 1
            else:
                error_matrix[2] += 1
        error_rate = round((error_matrix[1] + error_matrix[2]) / len(data), 2)
        return error_matrix, error_rate


def main():
    # grzybki
    file_name = "mushroom/agaricus-lepiota.data"
    values = "p", "e"

    # raki
    file_name = "cancer/breast-cancer.data"
    values = ["recurrence-events", "no-recurrence-events"]

    # proste testowe
    # file_name = "my.data"
    # values = ["Yes", "No"]

    # dane z osttnich labów
    file_name = "data.data"
    values = [True, False]

    data = pd.read_csv(file_name, header=None)

    error = []
    matrix = []
    for _ in range(10):
        train = data.sample(frac=0.6)
        test = data.drop(train.index)
        t = Tree(train, values)
        error_matrix, error_rate = t.prediction(test)
        print(error_rate)
        error.append(error_rate)
        print(error_matrix)
        matrix.append(error_matrix)
    print(f'{25*"#"}')
    print(1-round(sum(error)/len(error), 2))
    print([sum([x[i] for x in matrix])/len(matrix) for i in range(4)])


if __name__ == "__main__":
    main()
