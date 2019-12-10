import numpy as np
from DNN import DNN
import boto3

def initBoardZero(shape):
    board = np.zeros(shape=(shape))
    return board


def printBoard(board):
    print("%%%%%%%%%%%%%%%%%%%%%%")
    for i in range(len(board)):
        for j in range(len(board[i])):
            mark = ' '
            if board[i][j] == 1:
                mark = 'X'
            elif board[i][j] == 2:
                mark = 'O'
            if (j == len(board[i]) - 1):
                print(mark)
            else:
                print(str(mark) + "|", end='')
        if (i < len(board) - 1):
            print("-----")


def getMoves(board):
    moves = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:
                moves.append((i, j))
    return moves


def getWinner(board):
    won = check_rows(board)
    if won > 0:
        return won
    won = check_rows(board.T)
    if won > 0:
        return won
    won = check(board)
    if won > 0:
        return won
    if (len(getMoves(board)) == 0):
        return 0
    else:
        return -1


def put_move_player_1(board, y):
    for index in y:
        if board[index] == 0:
            board[index] = 1
            break
    return board


def put_move_player_2(board, y):
    for index in y:
        if board[index] == 0:
            board[index] = 2
            break
    return board


def exchange_for_player2(board):
    board[board == 1] = 3
    board[board == 2] = 4
    board[board == 3] = 2
    board[board == 4] = 1
    return board


def check_fitness(candidate, currentbest):
    wins = 0
    loses = 0
    for _ in range(games):
        board = initBoardZero(shape)
        status = -1
        while status == -1:
            y = candidate.forward(board.flatten(), True)
            y = y.argsort()[:][::-1]
            board = put_move_player_1(board.flatten(), y).reshape(shape)
            status = getWinner(board)
            if status != -1:
                break
            y = currentbest.forward(exchange_for_player2(board.copy().flatten()), True)
            y = y.argsort()[:][::-1]
            board = put_move_player_2(board.flatten(), y).reshape(shape)
            status = getWinner(board)
            if status != -1:
                break
        if getWinner(board) == 2:
            loses += 1
        elif getWinner(board) == 1:
            wins += 1

    for _ in range(games):
        board = initBoardZero(shape)
        status = -1
        while status == -1:
            y = currentbest.forward(board.flatten(), True)
            y = y.argsort()[:][::-1]
            board = put_move_player_1(board.flatten(), y).reshape(shape)
            status = getWinner(board)
            if status != -1:
                break
            y = candidate.forward(exchange_for_player2(board.copy().flatten()), True)
            y = y.argsort()[:][::-1]
            board = put_move_player_2(board.flatten(), y).reshape(shape)
            status = getWinner(board)
            if status != -1:
                break
    return wins, loses


def objective_function(candidate):
    global currentbest, games
    wins, loses = check_fitness(candidate, currentbest)
    if wins > loses+0.20*games:
        print(wins, loses)
        currentbest = candidate
        candidate.save_network()

def train():
    layers_num = np.random.random_integers(1, 4)
    layers = np.random.randint(5, 125, layers_num).flatten()

    candidatenetwork1 = DNN(inputSize, output, layers, 0.10)
    candidatenetwork2 = currentbest

    for weight in candidatenetwork1.weights:
        weight += np.random.normal(0, 5, np.shape(weight))
    for weight in candidatenetwork2.weights:
        weight += np.random.normal(0, 2, np.shape(weight))

    objective_function(candidatenetwork1)
    objective_function(candidatenetwork2)

def check(grid):
    for i in range(len(grid) - 2):
        for j in range(len(grid) - 2):
            if grid[i][j] == grid[i + 1][j + 1] == grid[i + 2][j + 2] and grid[i][j] == 1:
                return 1
            elif grid[i][j] == grid[i + 1][j + 1] == grid[i + 2][j + 2] and grid[i][j] == 2:
                return 2
            elif grid[i][j + 2] == grid[i + 1][j + 1] == grid[i + 2][j] and grid[i][j + 2] == 1:
                return 1
            elif grid[i][j + 2] == grid[i + 1][j + 1] == grid[i + 2][j] and grid[i][j + 2] == 2:
                return 2
            else:
                pass
    return 0


def check_rows(board):
    for i in board:
        for j in range(len(i)-2):
            if i[j] == i[j+1] == i[j+2] == 1:
                return 1
            elif i[j] == i[j+1] == i[j+2] == 2:
                return 2
    return 0

shape = (3,3)
inputSize = 9
output = 9

# layers_num = np.random.random_integers(1,4)
# layers = np.random.randint(5, 125, layers_num).flatten()

# games = 20
# bestfitness = 0
# currentbest = DNN(inputSize, output, layers, 0.10)
# for _ in range(100000000):
#     train()
dnn = DNN(inputSize, output, [2,2])
dnn.load_network()
board = initBoardZero(shape)
while(getWinner(board) == -1):
    y = dnn.forward(board.flatten())
    y = y.argsort()[:][::-1]
    board = put_move_player_1(board.flatten(), y)
    printBoard(board.reshape(shape))
    userInput = [int(input("Enter move"))]
    board = put_move_player_2(board.flatten(), userInput)
    board = board.reshape(shape)