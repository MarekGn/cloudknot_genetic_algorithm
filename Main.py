import numpy as np
from DNN import DNN


def initBoardZero(shape):
    board = np.zeros(shape=(shape))
    return board


def printBoard(board):
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

    # Still no winner?
    if (len(getMoves(board)) == 0):
        # It's a draw
        return 0
    else:
        # Still more moves to make
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


def make_move(agent1, agent2, board):
    y = agent1.forward(board, True)
    y = y.argsort()[:][::-1]
    board = put_move_player_1(board, y)

    y = agent2.forward(exchange_for_player2(board.copy()), True)
    y = y.argsort()[:][::-1]
    board = put_move_player_2(board, y)
    return board.reshape(shape)


def check_fitness(candidate, currentbest):
    wins = 0
    loses = 0
    for _ in range(games):
        board = initBoardZero(shape)
        while getWinner(board) == -1:
            board = make_move(candidate, currentbest, board.flatten())
        if getWinner(board) == 2:
            loses += 1
        elif getWinner(board) == 1:
            wins += 1

    for _ in range(games):
        board = initBoardZero(shape)
        while getWinner(board) == -1:
            board = make_move(currentbest, candidate, board.flatten())
        if getWinner(board) == 1:
            loses += 1
        elif getWinner(board) == 2:
            wins += 1

    return wins, loses


def objective_function(candidate):
    global currentbest, games
    wins, loses = check_fitness(candidate, currentbest)
    if wins > loses+0.10*games:
        print(wins, loses)
        currentbest = candidate
        candidate.save_network()

def train():
    layers_num = np.random.random_integers(1, 3)
    layers = np.random.randint(5, 25, layers_num).flatten()

    candidatenetwork1 = DNN(inputSize, output, layers, 0.10)
    candidatenetwork2 = currentbest

    candidatenetwork1.weights += np.random.normal(0, 5, np.shape(candidatenetwork1.weights))
    candidatenetwork2.weights += np.random.normal(0, 2, np.shape(candidatenetwork2.weights))
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

def run():
    shape = (5,5)
    inputSize = 25
    output = 25

    layers_num = np.random.random_integers(1,4)
    layers = np.random.randint(14, 124, layers_num).flatten()

    games = 500
    bestfitness = 0
    currentbest = DNN(inputSize, output, layers, 0.10)
    # currentbest.load_network()
    for _ in range(10000):
        train()
    # dnn = DNN(inputSize, output, layers, 0.10)
    # dnn.load_network()
    # board = initBoardZero(shape)
    # while(getWinner(board) == -1):
    #     y = dnn.forward(board.flatten(), True)
    #     y = y.argsort()[:][::-1]
    #     board = put_move_player_1(board.flatten(), y)
    #     printBoard(board.reshape(shape))
    #     userInput = [int(input("Enter move"))]
    #     board = put_move_player_2(board.flatten(), userInput)
    #     board = board.reshape(shape)
