import numpy as np


class Board():
    def __init__(self, shape):
        self.shape = shape
        self.board = np.zeros(shape=shape)

    def getWinner(self):
        won = self.check_rows()
        if won > 0:
            return won
        won = self.check_diagonals()
        if won > 0:
            return won
        if len(self.get_possible_moves()) == 0:
            return 0
        else:
            return -1

    def put_move(self, moves, player_num):
        self.board = self.board.reshape((-1, 1))
        for index in moves:
            if self.board[index] == 0:
                self.board[index] = player_num
                break
        self.board = self.board.reshape(self.shape)

    def get_board_for_player2(self):
        board = self.board.copy()
        board[board == 1] = 3
        board[board == 2] = 4
        board[board == 3] = 2
        board[board == 4] = 1
        return board

    def get_possible_moves(self):
        moves = []
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves

    def print_board(self):
        print("%%%%%%%%%%%%%%%%%%%%%%")
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                mark = ' '
                if self.board[i][j] == 1:
                    mark = 'X'
                elif self.board[i][j] == 2:
                    mark = 'O'
                if (j == len(self.board[i]) - 1):
                    print(mark)
                else:
                    print(str(mark) + "|", end='')
            if i < len(self.board) - 1:
                print("-----")

    def check_diagonals(self):
        for i in range(len(self.board) - 2):
            for j in range(len(self.board) - 2):
                if self.board[i][j] == self.board[i + 1][j + 1] == self.board[i + 2][j + 2] and self.board[i][j] == 1:
                    return 1
                elif self.board[i][j] == self.board[i + 1][j + 1] == self.board[i + 2][j + 2] and self.board[i][j] == 2:
                    return 2
                elif self.board[i][j + 2] == self.board[i + 1][j + 1] == self.board[i + 2][j] and self.board[i][j + 2] == 1:
                    return 1
                elif self.board[i][j + 2] == self.board[i + 1][j + 1] == self.board[i + 2][j] and self.board[i][j + 2] == 2:
                    return 2
                else:
                    pass
        return 0

    def check_rows(self):
        for i in self.board:
            for j in range(len(i)-2):
                if i[j] == i[j+1] == i[j+2] == 1:
                    return 1
                elif i[j] == i[j+1] == i[j+2] == 2:
                    return 2
        for i in self.board.T:
            for j in range(len(i)-2):
                if i[j] == i[j+1] == i[j+2] == 1:
                    return 1
                elif i[j] == i[j+1] == i[j+2] == 2:
                    return 2
        return 0