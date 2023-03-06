import random

import gym
import json
import numpy as np
import math
import time


class QLearningAlgorythm:
    def __init__(self, weight, height, possible_moves, gamma, beta):
        self.Q = {}
        self.make_Q(weight, height, possible_moves)
        self.gamma = gamma
        self.beta = beta

    def make_Q(self, weight, height, possible_moves):
        Q = {}
        for state in range(weight * height):
            Q[state] = {}
            for move in range(possible_moves):
                Q[state][move] = 0
        self.Q = Q

    def make_move(self, state, epsilon=0):
        if epsilon > 0:
            random_int = random.randint(0, 100)
            if random_int < epsilon:
                moves = [0, 1, 2, 3]
                action = random.choice(moves)
                return action
        best_action = [[], -math.inf]
        for action in self.Q[state].keys():
            value = self.Q[state][action]
            if value > best_action[1]:
                best_action[0].clear()
                best_action[0].append(action)
                best_action[1] = value
            elif value == best_action[1]:
                best_action[0].append(action)
        action = np.random.randint(max(best_action[0])) if len(best_action[0]) > 1 else best_action[0][0]
        return action

    def my_learn(self, next_state, is_terminated, state, action, reward, moves):
        keys = list(self.Q[next_state].keys())
        next_state_values = max([self.Q[next_state][u] for u in keys]) if not is_terminated else 0

        old_q = self.Q[state][action]

        if not is_terminated:
            reward -= self.beta * (0.0025 * (64 - state) + self.gamma * next_state_values - old_q)  # kara
        else:
            if reward == 0:
                self.Q[state][action] += self.beta * -10  # kara
            else:
                for move in moves:
                    self.Q[move[0]][move[1]] += self.beta * reward / 5  # nagroda

        self.Q[state][action] += self.beta * (reward + self.gamma * next_state_values - old_q)  # nagroda

    def default_learning(self, next_state, is_terminated, state, action, reward):
        keys = list(self.Q[next_state].keys())
        next_state_values = max([self.Q[next_state][u] for u in keys]) if not is_terminated else 0

        if state in self.Q.keys():
            if action in self.Q[state].keys():
                self.Q[state][action] += self.beta * (reward + self.gamma * next_state_values - self.Q[state][action])

    def save_Q(self, file_name="sample.json"):
        json_object = json.dumps(self.Q, indent=4)

        with open(file_name, "w") as outfile:
            outfile.write(json_object)


class GameWithoutVisualize:
    def __init__(self):
        self.board = []
        self.make_board()
        self.player_position = 0

    def make_board(self):
        self.board[1:64] = ["F" for _ in range(64)]
        self.board[0] = "S"
        self.board[-1] = "G"
        self.board[19] = "H"
        self.board[29] = "H"
        self.board[35] = "H"
        self.board[41] = "H"
        self.board[42] = "H"
        self.board[46] = "H"
        self.board[49] = "H"
        self.board[52] = "H"
        self.board[54] = "H"
        self.board[59] = "H"

    def move(self, move_direction):
        randint = np.random.randint(3)
        if move_direction == 0:  # left
            if randint == 0:
                self.move_up()
            elif randint == 1:
                self.move_left()
            else:
                self.move_down()
        elif move_direction == 1:  # down
            if randint == 0:
                self.move_left()
            elif randint == 1:
                self.move_down()
            else:
                self.move_right()
        elif move_direction == 2:  # right
            if randint == 0:
                self.move_right()
            elif randint == 1:
                self.move_up()
            else:
                self.move_down()
        else:  # up
            if randint == 0:
                self.move_up()
            elif randint == 1:
                self.move_left()
            else:
                self.move_right()

    def move_right(self):
        if self.player_position % 8 < 7:  # right
            self.player_position += 1

    def move_left(self):
        if self.player_position % 8 > 0:  # left
            self.player_position -= 1

    def move_up(self):
        if self.player_position - 8 >= 0:
            self.player_position -= 8  # up

    def move_down(self):
        if self.player_position + 8 <= 63:
            self.player_position += 8  # down

    def step(self, action):
        self.move(action)
        next_state = self.player_position
        board = self.board[self.player_position]
        reward = 1 if board == "G" else 0
        is_terminated = True if board == "G" or board == "H" else False
        return [next_state, reward, is_terminated]

    def reset_game(self):
        self.player_position = 0


def with_visualize(q, learning_iters, attempt, Eps, win_counter, loose_counter, my_l_function=None, printing=False):
    env = gym.make("FrozenLake-v1", map_name="8x8", render_mode="human")  # "human" / "rgb_array"
    env.reset()
    env.render()

    for i in range(learning_iters):
        if printing:
            print("Attempt: ", i)
        state = 0
        moves = []
        for j in range(attempt):
            action = q.make_move(state, epsilon=Eps)
            return_value = env.step(action)

            next_state = return_value[0]
            reward = return_value[1]
            is_terminated = return_value[2]
            if not (state, action) in moves:
                moves.append((state, action))
            if my_l_function is not None:
                if my_l_function:
                    q.my_learn(next_state, is_terminated, state, action, reward, moves)
                else:
                    q.default_learning(next_state, is_terminated, state, action, reward)

            if reward == 1:
                if printing:
                    print("You won")
                win_counter += 1
                break
            elif is_terminated is True:
                if printing:
                    print("You lose")
                loose_counter += 1
                break
            if j == attempt - 1:
                if printing:
                    print("You lose")
                loose_counter += 1
            state = next_state
        env.reset()
    env.close()
    q.save_Q()
    return q, win_counter, loose_counter


def without_visualize(q, learning_iters, attempt, Eps, win_counter, loose_counter, my_l_function=None, printing=False):
    g = GameWithoutVisualize()

    for i in range(learning_iters):
        if printing:
            print("Attempt: ", i)
        state = 0
        moves = []
        for j in range(attempt):
            action = q.make_move(state, epsilon=Eps)
            return_value = g.step(action)

            next_state = return_value[0]
            reward = return_value[1]
            is_terminated = return_value[2]
            if not (state, action) in moves:
                moves.append((state, action))
            if my_l_function is not None:
                if my_l_function:
                    q.my_learn(next_state, is_terminated, state, action, reward, moves)
                else:
                    q.default_learning(next_state, is_terminated, state, action, reward)

            if reward == 1:
                if printing:
                    print("You won")
                win_counter += 1
                break
            elif is_terminated is True:
                if printing:
                    print("You lose")
                loose_counter += 1
                break
            if j == attempt - 1:
                if printing:
                    print("You lose")
                loose_counter += 1
            state = next_state
        g.reset_game()
    q.save_Q()
    return q, win_counter, loose_counter


def main(gam, bet, eps, my_f):
    print(f'Gamma: {gam}, Beta: {bet}, Epsilon: {eps}, My function: {my_f}')
    LEARNING_WITH_VISUALIZE = 0
    TESTING_WITH_VISUALIZE = 0

    learning_loose_counter = 0
    learning_win_counter = 0

    testing_loose_counter = 0
    testing_win_counter = 0

    learning_iters = 10000
    attempt = 200

    q = QLearningAlgorythm(8, 8, 4, gamma=gam, beta=bet)
    Eps = eps
    print("Start Learning...")
    if LEARNING_WITH_VISUALIZE:
        q, learning_win_counter, learning_loose_counter = with_visualize(q, learning_iters, attempt, Eps,
                                                                         learning_win_counter, learning_loose_counter,
                                                                         my_l_function=my_f)
    else:
        q, learning_win_counter, learning_loose_counter = without_visualize(q, learning_iters, attempt, Eps,
                                                                            learning_win_counter,
                                                                            learning_loose_counter,
                                                                            my_l_function=my_f)
    print("Start Testing...")
    Eps = 0
    if TESTING_WITH_VISUALIZE:
        q, testing_win_counter, testing_loose_counter = with_visualize(q, int(learning_iters/10), attempt, Eps,
                                                                       testing_win_counter, testing_loose_counter,
                                                                       my_l_function=None)
    else:
        q, testing_win_counter, testing_loose_counter = without_visualize(q, int(learning_iters/10), attempt, 0,
                                                                          testing_win_counter, testing_loose_counter,
                                                                          my_l_function=None)
    # print("Learning Looses: ", learning_loose_counter)
    # print("learning Wins: ", learning_win_counter)
    # print("Testing Looses: ", testing_loose_counter)
    # print("Testing Wins: ", testing_win_counter)
    # learning_win_ratio = learning_win_counter / (learning_loose_counter + learning_win_counter)
    # print("Learning win ratio: ", round(learning_win_ratio, 2))
    testing_win_ratio = testing_win_counter / (testing_loose_counter + testing_win_counter)
    # print("Testing win ratio: ", round(testing_win_ratio, 2))
    return testing_win_ratio


if __name__ == "__main__":
    text = ""
    betay = [0.1, 0.1, 0.5, 0.75, 0.9, 0.9]
    gammay = [0.1, 0.9, 0.5, 0.9, 0.1, 0.9]
    epsy = [0, 2.5, 5, 7.5, 10, 20, 30, 40, 50]
    my_fy = [True]
    for f in my_fy:
        for eps in epsy:
            for i, gama in enumerate(gammay):
                beta = betay[i]
                start_time = time.time()
                history = []
                for i in range(10):
                    x = main(gama, beta, eps, f)
                    history.append(x)
                print(f'{25 * "#"}')
                average_values = round(sum(history) / len(history), 2)
                best_function_values = round(max(history), 2)
                worst_function_values = round(min(history), 2)
                standard_deviation = round(np.std(history), 2)
                print(f'Average: {average_values}\nBest value: {best_function_values}\n'
                      f'Worst value: {worst_function_values}\nStandard deviation: {standard_deviation}')
                print(f'Time: {time.time() - start_time}')
                text1 = f'Average,{average_values},Best value,{best_function_values},Worst value,' \
                        f'{worst_function_values},Standard deviation,{standard_deviation}'
                text += f'Gamma,{gama},Beta,{beta},F,{f},Epsilon,{eps},{text1}\n'

        # for gama in gammay:
        #     for beta in betay:
        #         # eps = 0
        #         start_time = time.time()
        #         history = []
        #         for i in range(10):
        #             x = main(gama, beta, eps, f)
        #             history.append(x)
        #         print(f'{25 * "#"}')
        #         average_values = round(sum(history) / len(history), 2)
        #         best_function_values = round(max(history), 2)
        #         worst_function_values = round(min(history), 2)
        #         standard_deviation = round(np.std(history), 2)
        #         print(f'Average: {average_values}\nBest value: {best_function_values}\n'
        #               f'Worst value: {worst_function_values}\nStandard deviation: {standard_deviation}')
        #         print(f'Time: {time.time() - start_time}')
        #         text1 = f'Average,{average_values},Best value,{best_function_values},Worst value,' \
        #                 f'{worst_function_values},Standard deviation,{standard_deviation}'
        #         text += f'Gamma,{gama},Beta,{beta},F,{f},Epsilon,{eps},{text1}\n'
    with open("data2.txt", "w") as file:
        file.write(text)

