import numpy as np
import time

# parametry
p = [5, 3] # [5, 3], [4, 6], [1, 6]

L_BOUND = -5
U_BOUND = 5


def q(x):
    return np.sin(x*np.sqrt(p[0]+1))+np.cos(x*np.sqrt(p[1]+1))


x = np.linspace(L_BOUND, U_BOUND, 100)
y = q(x)

np.random.seed(1)


# f logistyczna jako przykład sigmoidalej
def sigmoid(x):
    return 1/(1+np.exp(-x))


# pochodna fun. 'sigmoid'
def d_sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s * (1-s)


# f. straty
def nloss(y_out, y):
    return (y_out - y) ** 2


# pochodna f. straty
def d_nloss(y_out, y):
    return 2*(y_out - y)


class Neuron:
    def __init__(self, weights, bias):
        self.bias = bias
        self.weights = weights

    def get_value(self, x):
        if type(x) is not np.float64:
            sume = 0
            for i in range(len(self.weights)):
                sume += self.weights[i] * x[i]
            return self.bias + sume
        else:
            return sigmoid(self.bias + self.weights*x)


class DlNet:
    def __init__(self, x, y, input_neurons=1, output_neurons=1):
        self.x = x
        self.y = y
        self.y_out = 0
        
        self.HIDDEN_L_SIZE = 20
        self.LR = 0.003
        self.neurons_hidden = []
        self.neurons_out = []

        self.make_net()

        self.training_error = []

    def make_net(self):
        # random weights for hidden layout
        rand_weights = np.random.random(self.HIDDEN_L_SIZE)
        rand_bias = np.random.random(self.HIDDEN_L_SIZE)

        # creating network
        for i in range(self.HIDDEN_L_SIZE):
            self.neurons_hidden.append(Neuron(rand_weights[i], rand_bias[i]))
        self.neurons_out.append(Neuron(np.random.random(self.HIDDEN_L_SIZE), np.random.random(1)[0]))

    def forward(self, x):
        hidden_layout_exit = []
        for neuron in self.neurons_hidden:
            hidden_layout_exit.append(neuron.get_value(x))
        return self.neurons_out[0].get_value(hidden_layout_exit)
        
    def backward(self, x, y):
        buffor_b_e = []
        buffor_w_e = []
        buffor_b_h = []
        buffor_w_h = []

        LR = self.LR
        err_diff = d_nloss(self.forward(x), y)
        buffor_b_e.append(LR * err_diff)
        for j, weights in enumerate(self.neurons_out[0].weights):
            buffor_w_e.append(LR * err_diff * self.neurons_hidden[j].get_value(x))
        for j, neuron in enumerate(self.neurons_hidden):
            # neurony wartwy ukrytej
            delta = self.neurons_out[0].weights[j] * d_sigmoid(x * neuron.weights + neuron.bias) * err_diff
            buffor_b_h.append(LR * delta)
            buffor_w_h.append(LR * delta * x)

        self.neurons_out[0].bias -= buffor_b_e[0]
        self.neurons_out[0].weights -= buffor_w_e
        for j, neuron in enumerate(self.neurons_hidden):
            self.neurons_hidden[j].bias -= buffor_b_h[j]
            self.neurons_hidden[j].weights -= buffor_w_h[j]

    def get_error(self, x_set, y_set):
        error = 0
        for i in range(len(x_set)):
            error += (self.forward(x_set[i]) - y_set[i])**2
        return error

    def train(self, x_set, y_set, iters):
        err = 99999999
        for k in range(iters):
            for i in range(len(x_set)):
                self.backward(x_set[i], y_set[i])
            err = self.get_error(x_set, y_set)
            if k % 100 == 0:
                print(f'{k}/{iters}, Error: {err}')
        self.training_error.append(err)

        
nn = DlNet(x, y)
start_time = time.time()
nn.train(x, y, 15000)
print(time.time()-start_time)

yh = [nn.forward(point) for point in x]

import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.title(f'Hidden neurons: {nn.HIDDEN_L_SIZE}, Learning rate: {nn.LR}, Error: {round(nn.training_error[-1], 2)}')
plt.plot(x, y, 'r')
plt.plot(x, yh, 'b')

plt.show()
