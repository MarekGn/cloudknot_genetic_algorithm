import numpy as np
from DNN import DNN
import boto3
import argparse
from Main import *
import itertools
import copy
from random import shuffle
import random

def setup_cmd_parser():
    parser = argparse.ArgumentParser(description='Training Neural Networks By Genetic Algorithm.')
    parser.add_argument('--time', dest='time', default=12, type=int,
                        help='Number of hours that worker train networks default=12')
    parser.add_argument('--min_layers', dest='min_layers', default=1, type=int,
                        help='minimum hidden layers that can contain created networks default=1')
    parser.add_argument('--max_layers', dest='max_layers', default=4, type=int,
                        help='maximum hidden layers that can contain created networks default=4')
    parser.add_argument('--min_neurons', dest='min_neurons', default=5, type=int,
                        help='minimal neuron amount in network hidden layer default=5')
    parser.add_argument('--max_neurons', dest='max_neurons', default=125, type=int,
                        help='maximum neuron amount in network hidden layer default=5')
    parser.add_argument('--s3_bucket', dest='s3_bucket', default="besttictactoe",
                        help='S3 bucket to upload results to')
    parser.add_argument('--best_pop_size', dest='best_pop_size', default=25, type=int,
                        help='Size of population based on best network')
    parser.add_argument('--random_pop_size', dest='random_pop_size', default=25, type=int,
                        help='Size of random generated population')
    parser.add_argument('--games', dest='games', default=10, type=int,
                        help='How many times individuals have to play with whole population before fitness check')
    return parser


NETWORK_HIDDEN_SIZE = [9]
INPUT = 9
OUTPUT = 9
GAMES = 1
POP = 100
BOARD = (3,3)
RANDOMADD = 20
PARENTS_NUM = 5

def get_initial_population():
    pop = []
    for _ in range(POP):
        pop.append(DNN(INPUT, OUTPUT, NETWORK_HIDDEN_SIZE))
    return pop


def cal_fitness(pop, games):
    confrontations = itertools.permutations(pop, 2)
    for _ in range(games):
        for confrontation in confrontations:
            play_agents_tictactoe(confrontation)


def play_agents_tictactoe(players_couple):
    player_1 = players_couple[0]
    player_2 = players_couple[1]

    board = initBoardZero(BOARD)
    status = -1
    while status == -1:
        player1_moves = player_1.forward(board.flatten())
        player1_moves = player1_moves.argsort()[:][::-1]
        board = put_move_player_1(board.flatten(), player1_moves).reshape(BOARD)
        status = getWinner(board)
        if status != -1:
            break
        player2_moves = player_2.forward(exchange_for_player2(board.copy().flatten()))
        player2_moves = player2_moves.argsort()[:][::-1]
        board = put_move_player_2(board.flatten(), player2_moves).reshape(shape)
        status = getWinner(board)
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


def get_individual(layers):
    dnn = DNN(INPUT, INPUT, layers)
    return dnn


def generate_children(parents, pop_size, randomadd):
    children = []
    parents[0].reset_fitness()
    children.append(parents[0])
    while len(children) < pop_size-randomadd:
        bio_parents = itertools.permutations(parents, 2)
        bio_parents = list(bio_parents)
        shuffle(bio_parents)
        for parent_couple in bio_parents:
            if len(children) < pop_size-randomadd:
                children.append(get_child(parent_couple))
            else:
                break
    return children


def add_random(children, RANDOMADD):
    for _ in range(RANDOMADD):
        children.append(DNN(INPUT,INPUT,NETWORK_HIDDEN_SIZE))


def get_child(parent_couple):
    parent1vector = parent_couple[0].toVector()
    parent2vector = parent_couple[1].toVector()
    vectors = [parent1vector, parent2vector]
    childvector = []
    child = DNN(INPUT, INPUT, NETWORK_HIDDEN_SIZE)
    for i in range(len(parent1vector)):
        choice = random.randint(0,1)
        childvector.append(vectors[choice][i])
    childvector = np.reshape(childvector, parent1vector.shape)
    child.fromVector(childvector)
    return child

def get_parents(pop, parents_num):
    pop.sort(key=lambda idv: 2*idv.wins + idv.draw - 4*idv.loses, reverse=True)
    return pop[:parents_num]


if __name__ == "__main__":
    parser = setup_cmd_parser()
    pop = get_initial_population()
    iters = 1000
    while iters:
        print(iters)
        cal_fitness(pop, GAMES)
        parents = get_parents(pop, PARENTS_NUM)
        parents[0].save_network()
        pop = generate_children(parents, POP, RANDOMADD)
        add_random(pop, RANDOMADD)
        iters -= 1

