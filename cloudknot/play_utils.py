from cloudknot.Board import Board
from cloudknot.DNN import DNN


def play_agents_tictactoe(players_couple, board_shape):
    player_1 = players_couple[0]
    player_2 = players_couple[1]

    board = Board(board_shape)
    status = -1
    while status == -1:
        player1_moves = player_1.forward(board.board.flatten())
        player1_moves = player1_moves.argsort()[:][::-1]
        board.put_move(moves=player1_moves, player_num=1)
        status = board.getWinner()
        if status != -1:
            break
        player2_moves = player_2.forward(board.get_board_for_player2().flatten())
        player2_moves = player2_moves.argsort()[:][::-1]
        board.put_move(moves=player2_moves, player_num=2)
        status = board.getWinner()
        if status != -1:
            break
    if status == 1:
        player_1.wins += 1
        player_2.loses += 1
    elif status == 2:
        player_2.wins += 1
        player_1.loses += 1
    elif status == 0:
        player_1.draw += 1
        player_2.draw += 1


def play_with_ai(board_shape):
    dnn = DNN(9, 9, [2, 2])
    dnn.load_network()
    board = Board(board_shape)
    while board.getWinner() == -1:
        ai_moves = dnn.forward(board.board.flatten())
        ai_moves = ai_moves.argsort()[:][::-1]
        board.put_move(moves=ai_moves, player_num=1)
        board.print_board()
        userInput = [int(input("Enter move"))]
        board.put_move(userInput, player_num=2)
