#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Rafał Biedrzycki
Kodu tego mogą używać moi studenci na ćwiczeniach z przedmiotu Wstęp do Sztucznej Inteligencji.
Kod ten powstał aby przyspieszyć i ułatwić pracę studentów, aby mogli skupić się na algorytmach sztucznej inteligencji. 
Kod nie jest wzorem dobrej jakości programowania w Pythonie, nie jest również wzorem programowania obiektowego, może zawierać błędy.
Mam świadomość wielu jego braków ale nie mam czasu na jego poprawianie.

Zasady gry: https://en.wikipedia.org/wiki/English_draughts (w skrócie: wszyscy ruszają się po 1 polu. Pionki tylko w kierunku wroga, damki w dowolnym)
  z następującymi modyfikacjami: a) bicie nie jest wymagane,  b) dozwolone jest tylko pojedyncze bicie (bez serii).

Nalezy napisac funkcje minimax_a_b_recurr, minimax_a_b (woła funkcję rekurencyjną) i  evaluate, która ocenia stan gry

Chętni mogą ulepszać mój kod (trzeba oznaczyć komentarzem co zostało zmienione), mogą również dodać obsługę bicia wielokrotnego i wymagania bicia. Mogą również wdrożyć reguły: https://en.wikipedia.org/wiki/Russian_draughts
"""
import sys

import numpy as np
import pygame
from copy import deepcopy
import time

FPS = 20


# PARAMETERS
MINIMAX_DEPTH = 5
PAWN_POINTS = 1
QUEEN_POINTS = 10
STAY_IN_GROUP_DIV = 2


WIN_WIDTH = 800
WIN_HEIGHT = 800


BOARD_WIDTH = 8

FIELD_SIZE = WIN_WIDTH/BOARD_WIDTH
PIECE_SIZE = FIELD_SIZE/2 - 8
MARK_THICK = 2
POS_MOVE_MARK_SIZE = PIECE_SIZE/2


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


class Move:
    def __init__(self, piece, dest_row, dest_col, captures=None):
        self.piece = piece
        self.dest_row = dest_row
        self.dest_col = dest_col
        self.captures = captures


class Field:
    def draw(self):
        pass

    def is_empty(self):
        return True

    def is_white(self):
        return False

    def is_blue(self):
        return False

    def toogle_mark(self):
        pass

    def is_move_mark(self):
        return False

    def is_marked(self):
        return False

    def __str__(self):
        return "."


class PosMoveField(Field):
    def __init__(self, is_white, window, row, col, board, row_from, col_from, pos_move):
        self.__is_white = is_white
        self.__is_marked = False
        self.window = window
        self.row = row
        self.col = col
        self.board = board
        self.row_from = row_from
        self.col_from = col_from
        self.pos_move = pos_move

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        return result

    def draw(self):
        x = self.col*FIELD_SIZE
        y = self.row*FIELD_SIZE
        pygame.draw.circle(self.window, RED, (x+FIELD_SIZE/2, y+FIELD_SIZE/2), POS_MOVE_MARK_SIZE)

    def is_empty(self):
        return True

    def is_move_mark(self):
        return True


class Pawn(Field):
    def __init__(self, is_white, window, row, col, board):
        self.__is_white = is_white
        self.__is_marked = False
        self.window = window
        self.row = row
        self.col = col
        self.board = board

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        return result

    def __str__(self):
        if self.is_white():
            return "w"
        return "b"

    def is_king(self):
        return False

    def is_empty(self):
        return False

    def is_white(self):
        return self.__is_white

    def is_blue(self):
        return not self.__is_white

    def is_marked(self):
        return self.__is_marked

    def toogle_mark(self):
        if self.__is_marked:
            for pos_move in self.pos_moves:  # remove possible moves
                row = pos_move.dest_row
                col = pos_move.dest_col
                self.board.board[row][col] = Field()
            self.pos_moves = []
        else:  # self.is_marked
            self.pos_moves = self.board.get_piece_moves(self)
            for pos_move in self.pos_moves:
                row = pos_move.dest_row
                col = pos_move.dest_col
                self.board.board[row][col] = PosMoveField(False, self.window, row, col, self.board, self.row, self.col, pos_move)

        self.__is_marked = not self.__is_marked

    def draw(self):
        if self.__is_white:
            cur_col = WHITE
        else:
            cur_col = BLUE
        x = self.col*FIELD_SIZE
        y = self.row*FIELD_SIZE
        pygame.draw.circle(self.window, cur_col, (x+FIELD_SIZE/2, y+FIELD_SIZE/2), PIECE_SIZE)

        if self.__is_marked:
            pygame.draw.circle(self.window, RED, (x+FIELD_SIZE/2, y+FIELD_SIZE/2), PIECE_SIZE+MARK_THICK, MARK_THICK)


class King(Pawn):
    def __init__(self, pawn):
        super().__init__(pawn.is_white(), pawn.window, pawn.row, pawn.col, pawn.board)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        return result

    def is_king(self):
        return True

    def __str__(self):
        if self.is_white():
            return "W"
        return "B"

    def draw(self):
        if self.is_white():
            cur_col = WHITE
        else:
            cur_col = BLUE
        x = self.col*FIELD_SIZE
        y = self.row*FIELD_SIZE
        pygame.draw.circle(self.window, cur_col, (x+FIELD_SIZE/2, y+FIELD_SIZE/2), PIECE_SIZE)
        pygame.draw.circle(self.window, GREEN, (x+FIELD_SIZE/2, y+FIELD_SIZE/2), PIECE_SIZE/2)

        if self.is_marked():
            pygame.draw.circle(self.window, RED, (x+FIELD_SIZE/2, y+FIELD_SIZE/2), PIECE_SIZE+MARK_THICK, MARK_THICK)


class Board:
    def __init__(self, window=None):  # row, col
        self.board = []  # np.full((BOARD_WIDTH, BOARD_WIDTH), None)
        self.window = window
        self.marked_piece = None
        self.something_is_marked = False
        self.white_turn = True
        self.white_fig_left = 12
        self.blue_fig_left = 12

        self.__set_pieces()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        result.board = deepcopy(self.board)
        return result

    def __str__(self):
        to_ret = ""
        for row in range(8):
            for col in range(8):
                to_ret += str(self.board[row][col])
            to_ret += "\n"
        return to_ret

    def __set_pieces(self):
        for row in range(8):
            self.board.append([])
            for col in range(8):
                self.board[row].append(Field())

        for row in range(3):
            for col in range((row+1) % 2, BOARD_WIDTH, 2):
                self.board[row][col] = Pawn(False, self.window, row, col, self)

        for row in range(5, 8):
            for col in range((row+1) % 2, BOARD_WIDTH, 2):
                self.board[row][col] = Pawn(True, self.window, row, col, self)

    def get_piece_moves(self, piece):
        pos_moves = []
        row = piece.row
        col = piece.col
        if piece.is_blue():
            enemy_is_white = True
        else:
            enemy_is_white = False

        if piece.is_white() or (piece.is_blue() and piece.is_king()):
            dir_y = -1
            if row > 0:
                new_row = row+dir_y
                if col > 0:
                    new_col = col-1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                        # ruch zwiazany z biciem
                    elif self.board[new_row][new_col].is_white() == enemy_is_white and new_row+dir_y >= 0 and new_col-1 >= 0 and self.board[new_row+dir_y][new_col-1].is_empty():
                        pos_moves.append(Move(piece, new_row+dir_y, new_col-1, self.board[new_row][new_col]))

                if col < BOARD_WIDTH-1:
                    new_col = col+1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                        # ruch zwiazany z biciem
                    elif self.board[new_row][new_col].is_white() == enemy_is_white and new_row+dir_y >= 0 and new_col+1 < BOARD_WIDTH and self.board[new_row+dir_y][new_col+1].is_empty():
                        pos_moves.append(Move(piece, new_row+dir_y, new_col+1, self.board[new_row][new_col]))

        if piece.is_blue() or (piece.is_white() and self.board[row][col].is_king()):
            dir_y = 1
            if row < BOARD_WIDTH-1:
                new_row = row+dir_y
                if col > 0:
                    new_col = col-1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                    elif self.board[new_row][new_col].is_white() == enemy_is_white and new_row+dir_y < BOARD_WIDTH and new_col-1 >= 0 and self.board[new_row+dir_y][new_col-1].is_empty():
                        pos_moves.append(Move(piece, new_row+dir_y, new_col-1, self.board[new_row][new_col]))

                if col < BOARD_WIDTH-1:
                    new_col = col+1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                        # ruch zwiazany z biciem
                    elif self.board[new_row][new_col].is_white() == enemy_is_white and new_row+dir_y < BOARD_WIDTH and new_col+1 < BOARD_WIDTH and self.board[new_row+dir_y][new_col+1].is_empty():
                        pos_moves.append(Move(piece, new_row+dir_y, new_col+1, self.board[new_row][new_col]))
        return pos_moves

    def evaluate_default(self):  # myFunction
        h = 0
        for row in range(BOARD_WIDTH):
            for col in range((row+1) % 2, BOARD_WIDTH, 2):
                pawn = self.board[row][col]
                if type(pawn) is Field:
                    continue
                if pawn.is_white():
                    if not pawn.is_king():
                        h += PAWN_POINTS
                    else:
                        h += QUEEN_POINTS
                else:
                    if not pawn.is_king():
                        h += -PAWN_POINTS
                    else:
                        h += -QUEEN_POINTS
        return h

    def evaluate_grouping(self):  # myFunction
        h = 0
        count_blue_neighbours = 0
        count_white_neighbours = 0
        for row in range(BOARD_WIDTH):
            for col in range((row+1) % 2, BOARD_WIDTH, 2):
                pawn = self.board[row][col]
                if type(pawn) is Field:
                    continue
                if not pawn.is_white():
                    if not pawn.is_king():
                        h += -PAWN_POINTS
                        count_blue_neighbours += self.positive_count_neighbours(pawn)
                    else:
                        h += -QUEEN_POINTS
                else:
                    if not pawn.is_king():
                        h += PAWN_POINTS
                        count_white_neighbours += self.positive_count_neighbours(pawn)
                    else:
                        h += QUEEN_POINTS
        h -= round(count_blue_neighbours / STAY_IN_GROUP_DIV)
        h += round(count_white_neighbours / STAY_IN_GROUP_DIV)
        return h

    def evaluate_going_to_enemy(self):  # myFunction
        h = 0
        for row in range(BOARD_WIDTH):
            for col in range((row+1) % 2, BOARD_WIDTH, 2):
                pawn = self.board[row][col]
                if type(pawn) is Field:
                    continue
                if not pawn.is_white():
                    if not pawn.is_king():
                        h += -5 * PAWN_POINTS
                        if row >= 4:
                            h += -2 * PAWN_POINTS
                    else:
                        h += -QUEEN_POINTS
                else:
                    if not pawn.is_king():
                        h += 5 * PAWN_POINTS
                        if row < 4:
                            h += 2 * PAWN_POINTS
                    else:
                        h += QUEEN_POINTS
        return h

    def evaluate_going_up(self):  # myFunction
        h = 0
        for row in range(BOARD_WIDTH):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                pawn = self.board[row][col]
                if type(pawn) is Field:
                    continue
                if pawn.is_white():
                    if pawn.is_king():
                        h += QUEEN_POINTS
                    else:
                        h += 5 + (7-row)
                else:
                    if pawn.is_king():
                        h -= QUEEN_POINTS
                    else:
                        h -= -5 - row
        return h

    def positive_count_neighbours(self, pawn):  # myFunction
        count = 0
        col = pawn.col
        row = pawn.row
        color = pawn.is_white()
        for i in [-1, 2]:
            for j in [-1, 2]:
                try:
                    if self.board[row + i - 1][col + j - 1].is_white() == color:
                        count += 1
                except IndexError:
                    pass
        return count

    def get_possible_moves(self, is_blue_turn):
        pos_moves = []
        for row in range(BOARD_WIDTH):
            for col in range((row+1) % 2, BOARD_WIDTH, 2):
                if not self.board[row][col].is_empty():
                    if (is_blue_turn and self.board[row][col].is_blue()) or (not is_blue_turn and self.board[row][col].is_white()):
                        pos_moves.extend(self.get_piece_moves(self.board[row][col]))
        return pos_moves

    def draw(self):
        self.window.fill(WHITE)
        for row in range(BOARD_WIDTH):
            for col in range((row+1) % 2, BOARD_WIDTH, 2):
                y = row*FIELD_SIZE
                x = col*FIELD_SIZE
                pygame.draw.rect(self.window, BLACK, (x, y, FIELD_SIZE, FIELD_SIZE))
                self.board[row][col].draw()

    def move(self, field):
        d_row = field.row
        d_col = field.col
        row_from = field.row_from
        col_from = field.col_from
        self.board[row_from][col_from].toogle_mark()
        self.something_is_marked = False
        self.board[d_row][d_col] = self.board[row_from][col_from]
        self.board[d_row][d_col].row = d_row
        self.board[d_row][d_col].col = d_col
        self.board[row_from][col_from] = Field()

        if field.pos_move.captures:
            fig_to_del = field.pos_move.captures

            self.board[fig_to_del.row][fig_to_del.col] = Field()
            if self.white_turn:
                self.blue_fig_left -= 1
            else:
                self.white_fig_left -= 1

        if self.white_turn and d_row == 0:  # damka
            self.board[d_row][d_col] = King(self.board[d_row][d_col])

        if not self.white_turn and d_row == BOARD_WIDTH-1:  # damka
            self.board[d_row][d_col] = King(self.board[d_row][d_col])

        self.white_turn = not self.white_turn

    def end(self):
        return self.white_fig_left == 0 or self.blue_fig_left == 0 or len(self.get_possible_moves(not self.white_turn)) == 0

    def clicked_at(self, row, col):
        field = self.board[row][col]
        if field.is_move_mark():
            self.move(field)
        if (field.is_white() and self.white_turn and not self.something_is_marked) or (field.is_blue() and not self.white_turn and not self.something_is_marked):
            field.toogle_mark()
            self.something_is_marked = True
        elif self.something_is_marked and field.is_marked():
            field.toogle_mark()
            self.something_is_marked = False

    # tu spore powtorzenie kodu z move
    def make_ai_move(self, move):
        d_row = move.dest_row
        d_col = move.dest_col
        row_from = move.piece.row
        col_from = move.piece.col

        self.board[d_row][d_col] = self.board[row_from][col_from]
        self.board[d_row][d_col].row = d_row
        self.board[d_row][d_col].col = d_col
        self.board[row_from][col_from] = Field()

        if move.captures:
            fig_to_del = move.captures

            self.board[fig_to_del.row][fig_to_del.col]=Field()
            if self.white_turn:
                self.blue_fig_left -= 1
            else:
                self.white_fig_left -= 1

        if self.white_turn and d_row == 0:  # damka
            self.board[d_row][d_col] = King(self.board[d_row][d_col])

        if not self.white_turn and d_row == BOARD_WIDTH-1:  # damka
            self.board[d_row][d_col] = King(self.board[d_row][d_col])

        self.white_turn = not self.white_turn


class Game:
    def __init__(self, window):
        self.window = window
        self.board = Board(window)

    def update(self):
        self.board.draw()
        pygame.display.update()

    def mouse_to_indexes(self, pos):
        return (int(pos[0]//FIELD_SIZE), int(pos[1]//FIELD_SIZE))

    def clicked_at(self, pos):
        (col, row) = self.mouse_to_indexes(pos)
        self.board.clicked_at(row, col)


def minimax_a_b(board, depth, max_move, evaluate_function):  # myFunction
    a = -sys.maxsize
    b = sys.maxsize
    best_move = []
    moves = board.get_possible_moves(not max_move)
    eval = 0
    for move in moves:
        board_copy = deepcopy(board)
        board_copy.make_ai_move(move)
        eval = minimax_a_b_recurr(board_copy, depth-1, not max_move, -sys.maxsize, sys.maxsize, evaluate_function)
        if max_move:
            if a < eval:
                a = eval
                best_move.clear()
                best_move.append(move)
            elif a == eval:
                best_move.append(move)
        else:
            if b > eval:
                b = eval
                best_move.clear()
                best_move.append(move)
            elif b == eval:
                best_move.append(move)
    if len(best_move) > 1:
        random_number = np.random.randint(0, len(best_move))
        # print(f'A: {a}, B: {b}, Move max: {max_move}, Function: {evaluate_function}')
        return best_move[random_number]
    else:
        return best_move[0]


def minimax_a_b_recurr(board, depth, move_max, a, b, evaluate_function):  # myFunction
    if depth == 0 or board.end():
        if evaluate_function == "default":
            return board.evaluate_default()  # podstawowy
        elif evaluate_function == "grouping":
            return board.evaluate_grouping()  # premie za grupowanie
        elif evaluate_function == "enemy_half":
            return board.evaluate_going_to_enemy()  # premie za bycie na polowie przeciwnika
        elif evaluate_function == "going_up":
            return board.evaluate_going_up()  # premie za bycie jak najdalej od swojego najnizszego rzedu
    moves = list(set(board.get_possible_moves(not move_max)))
    if move_max:
        for move in moves:
            board_copy = deepcopy(board)
            board_copy.make_ai_move(move)
            a = max(a, minimax_a_b_recurr(board_copy, depth-1, not move_max, a, b, evaluate_function))
            if a >= b:
                return b
        return a
    else:
        for move in moves:
            board_copy = deepcopy(board)
            board_copy.make_ai_move(move)
            b = min(b, minimax_a_b_recurr(board_copy, depth-1, not move_max, a, b, evaluate_function))
            if a >= b:
                return a
        return b


def player_vs_ai(ai_depth, evaluate_function):  # myFunction
    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    is_running = True
    clock = pygame.time.Clock()
    game = Game(window)

    last_move = ""

    while is_running:
        clock.tick(FPS)
        if game.board.end():
            is_running = False
            if last_move == "w":
                print("White won!!!")
            else:
                print("Blue won!!")
            break  # przydalby sie jakiś komunikat kto wygrał zamiast break

        if not game.board.white_turn:
            move = minimax_a_b(deepcopy(game.board), ai_depth, False, evaluate_function)
            game.board.make_ai_move(move)
            last_move = "b"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                game.clicked_at(pos)
                last_move = "w"
        game.update()
    pygame.quit()


def ai_vs_ai_with_visuals(depth_ai_blue, depth_ai_white, evaluation_function_blue, evaluation_function_white):  # myFunction
    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    is_running = True
    clock = pygame.time.Clock()
    game = Game(window)

    last_move = ""

    while is_running:
        clock.tick(FPS)
        if game.board.end():
            is_running = False
            if last_move == "w":
                print("White won!!!")
            else:
                print("Blue won!!")
            data = ""
            with open("data.txt", "r") as file:
                data = file.read()
                data += f"\n{last_move}"
            with open("data.txt", "w") as file:
                file.write(data)
            break

        if not game.board.white_turn:
            move = minimax_a_b(deepcopy(game.board), depth_ai_blue, False, evaluation_function_blue)
            game.board.make_ai_move(move)
            last_move = "b"
        else:
            move = minimax_a_b(deepcopy(game.board), depth_ai_white, True, evaluation_function_white)
            game.board.make_ai_move(move)
            last_move = 'w'
        game.update()
    pygame.quit()


def ai_vs_ai(depth_ai_blue, depth_ai_white, evaluation_function_blue, evaluation_function_white):  # myFunction
    is_running = True
    board = Board()

    last_move = ""

    while is_running:
        if board.end():
            is_running = False
            if last_move == "w":
                print("White won!!!")
            else:
                print("Blue won!!")
            data = ""
            with open("data.txt", "r") as file:
                data = file.read()
                data += f"\n{last_move}"
            with open("data.txt", "w") as file:
                file.write(data)
            break

        if not board.white_turn:
            move = minimax_a_b(deepcopy(board), depth_ai_blue, False, evaluation_function_blue)
            board.make_ai_move(move)
            last_move = "b"
        else:
            move = minimax_a_b(deepcopy(board), depth_ai_white, True, evaluation_function_white)
            board.make_ai_move(move)
            last_move = 'w'


if __name__ == "__main__":  # myFunction
    """
    Evaluation function to choose:
    - default - 1 p for pawn and 10 for queen
    - grouping - default + bonus for staying in group
    - enemy_half - default + bonus for being in enemy's part
    - going_up - default + bonus from higher row
    """
    game_with_player = 1  # 0 -> player vs ai, 1 -> Ai vs Ai with visuals, 2 -> Ai vs Ai without visuals

    # game with player parameters
    depth_ai_vs_player = 5
    evaluation_function_with_player = "default"

    # Ai vs Ai game parameters
    number_of_game_ai_vs_ai = 5

    depth_ai_blue = 2
    evaluation_function_blue = "default"
    # evaluation_function_blue = "grouping"
    # evaluation_function_blue = "enemy_half"
    # evaluation_function_blue = "going_up"

    depth_ai_white = 3
    evaluation_function_white = "default"
    # evaluation_function_white = "grouping"
    # evaluation_function_white = "enemy_half"
    # evaluation_function_white = "going_up"

    if game_with_player == 0:
        # player vs ai
        player_vs_ai(depth_ai_vs_player, evaluation_function_with_player)

    elif game_with_player == 1:
        # ai vs ai with simple data analysts
        with open("data.txt", "w") as file:  # clearing data file
            file.write("")

        for _ in range(number_of_game_ai_vs_ai):
            ai_vs_ai_with_visuals(depth_ai_blue, depth_ai_white, evaluation_function_blue, evaluation_function_white)

        counter = 0
        lines = 0
        with open("data.txt", "r") as file:
            data = file.readlines()
            for line in data:
                if line == "\n":
                    continue
                if line == "w\n" or line == "w":
                    counter += 1
                lines += 1
        win_ratio = round(100*counter/lines, 2) if lines > 0 else 0
        print(f'Average white win ratio: {win_ratio}%')

    elif game_with_player == 2:
        # ai vs ai with simple data analysts
        with open("data.txt", "w") as file:  # clearing data file
            file.write("")

        for _ in range(number_of_game_ai_vs_ai):
            ai_vs_ai(depth_ai_blue, depth_ai_white, evaluation_function_blue, evaluation_function_white)

        counter = 0
        lines = 0
        with open("data.txt", "r") as file:
            data = file.readlines()
            for line in data:
                if line == "\n":
                    continue
                if line == "w\n" or line == "w":
                    counter += 1
                lines += 1
        win_ratio = round(100 * counter / lines, 2) if lines > 0 else 0
        print(f'Average white win ratio: {win_ratio}%')
