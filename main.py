from DNN import DNN
import argparse
from utils import *
import itertools
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
    for idv in pop:
        idv.cal_fitness()


def play_agents_tictactoe(players_couple):
    player_1 = players_couple[0]
    player_2 = players_couple[1]

    board = initBoardZero(BOARD)
    status = -1
    while status == -1:
        copy_board = board.copy()
        player1_moves = player_1.forward(board.flatten())
        player1_moves = player1_moves.argsort()[:][::-1]
        board = put_move_player_1(board.flatten(), player1_moves, player_1).reshape(BOARD)
        status = getWinner(board)
        if status != -1:
            break
        elif (board==copy_board).all():
            break
        player2_moves = player_2.forward(exchange_for_player2(board.copy().flatten()))
        player2_moves = player2_moves.argsort()[:][::-1]
        board = put_move_player_2(board.flatten(), player2_moves, player_2).reshape(BOARD)
        status = getWinner(board)
        if status != -1:
            break
        elif (board==copy_board).all():
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
    parents[1].reset_fitness()
    children.append(parents[0])
    children.append(parents[1])
    while len(children) < pop_size-randomadd:
        bio_parents = itertools.permutations(parents, 2)
        bio_parents = list(bio_parents)
        shuffle(bio_parents)
        for parent_couple in bio_parents:
            if len(children) < pop_size-randomadd:
                children.append(get_child_offset(parent_couple))
            else:
                break
    return children


def add_random(children, RANDOMADD):
    for _ in range(RANDOMADD):
        children.append(DNN(INPUT, INPUT, NETWORK_HIDDEN_SIZE))


def cal_probability(pop, alpha):
    pop.sort(key=lambda idv: idv.fitness, reverse=True)
    if pop[-1].fitness < 0:
        for idv in pop:
            idv.fitness += abs(pop[-1].fitness)
    fitness_sum = 0
    for idv in pop:
        fitness_sum += idv.fitness
    fitness_avg = fitness_sum / len(pop)
    normalize_fitness(pop, fitness_avg, alpha)
    fitness_sum = 0
    for idv in pop:
        fitness_sum += idv.fitness
    for idv in pop:
        idv.probability = idv.fitness / fitness_sum


def normalize_fitness(pop, fitness_avg, alpha):
    delta = pop[0].fitness - fitness_avg
    if delta == 0:
        delta = 1
    a = (fitness_avg*(alpha - 1)) / delta
    b = fitness_avg * (1 - a)
    for idv in pop:
        idv.fitness = a*idv.fitness + b


def get_child_random(parent_couple):
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

def get_child_offset(children, parent_couple):
    parent1vector = parent_couple[0].toVector()
    parent2vector = parent_couple[1].toVector()

    child1 = DNN(INPUT, INPUT, NETWORK_HIDDEN_SIZE)
    child2 = DNN(INPUT, INPUT, NETWORK_HIDDEN_SIZE)
    offset = np.random.randint(0, len(parent1vector))
    child1vector = np.concatenate((parent1vector[:offset], parent2vector[offset:]))
    child1vector = np.reshape(child1vector, parent1vector.shape)
    child1.fromVector(child1vector)
    children.append(child1)

    child2vector = np.concatenate((parent2vector[:offset], parent1vector[offset:]))
    child2vector = np.reshape(child2vector, parent1vector.shape)
    child2.fromVector(child2vector)
    children.append(child2)
    return children

def get_parents(pop, parents_num):
    pop.sort(key=lambda idv: 10*idv.wins + idv.draw + idv.good_moves - 2*idv.loses - 3*idv.bad_moves, reverse=True)
    return pop[:parents_num]


def play_with_AI():
    dnn = DNN(9, 9, [2,2])
    dnn.load_network()
    board = initBoardZero(BOARD)
    while(getWinner(board) == -1):
        y = dnn.forward(board.flatten())
        y = y.argsort()[:][::-1]
        board = put_move_player_1(board.flatten(), y, dnn)
        printBoard(board.reshape(BOARD))
        userInput = [int(input("Enter move"))]
        board = put_move_player_2(board.flatten(), userInput, dnn)
        board = board.reshape(BOARD)

def play_with_random(agent):
    win_draw = 0
    for _ in range(10):
        board = initBoardZero(BOARD)
        while(getWinner(board) == -1):
            player1_moves = agent.forward(board.flatten())
            player1_moves = player1_moves.argsort()[:][::-1]
            board = put_move_player_1(board.flatten(), player1_moves, agent).reshape(BOARD)
            status = getWinner(board)
            if status != -1:
                break
            player2_moves = np.arange(0, 9)
            np.random.shuffle(player2_moves)
            board = put_move_player_2(board.flatten(), player2_moves, agent).reshape(BOARD)
            status = getWinner(board)
            if status != -1:
                break
        if status == 1 or status == 0:
            win_draw += 1
        board = initBoardZero(BOARD)
        while(getWinner(board) == -1):
            player1_moves = np.arange(0, 9)
            np.random.shuffle(player1_moves)
            board = put_move_player_1(board.flatten(), player1_moves, agent).reshape(BOARD)
            status = getWinner(board)
            if status != -1:
                break
            player2_moves = agent.forward(exchange_for_player2(board.flatten()))
            player2_moves = player2_moves.argsort()[:][::-1]
            board = put_move_player_2(board.flatten(), player2_moves, agent).reshape(BOARD)
            status = getWinner(board)
            if status != -1:
                break
        if status == 2 or status == 0:
            win_draw += 1
    return win_draw


def mutate_population(pop):
    for idv in pop:
        mutate_idv(idv)

def mutate_idv(idv):
    gene_vector = idv.toVector()
    gene_prob = MUTATION_PROBABILITY / len(gene_vector)
    for i in range(len(gene_vector)):
        chance = np.random.uniform(0, 1)
        if chance <= gene_prob:
            gene_vector[i] += np.random.normal(0,10)
    idv.fromVector(gene_vector)


def get_new_pop(pop):
    children = []
    while len(children) < len(pop):
        choice = np.random.uniform(0, 1, 2)
        parent_1 = None
        parent_1_call = True
        parent_2 = None
        parent_2_call = True
        prob = 0
        for idv in pop:
            prob += idv.probability
            if choice[0] <= prob and parent_1_call:
                parent_1 = idv
                parent_1_call = False
            if choice[1] <= prob and parent_2_call:
                parent_2 = idv
                parent_2_call = False
            if not parent_1_call and not parent_2_call:
                break
        if parent_1 is parent_2:
            continue
        else:
            get_child_offset(children, [parent_1, parent_2])
    return children


NETWORK_HIDDEN_SIZE = [14, 14]
INPUT = 9
OUTPUT = 9
GAMES = 1
POP = 500
BOARD = (3,3)
MUTATION_PROBABILITY = 0.05
ALPHA = 1.5

if __name__ == "__main__":
    parser = setup_cmd_parser()
    pop = get_initial_population()
    iters = 1000
    while iters:
        print("Interation: {}".format(iters))
        random = 0
        cal_fitness(pop, GAMES)
        cal_probability(pop, ALPHA)
        print(pop[0].fitness)
        pop[0].save_network()
        pop = get_new_pop(pop)
        mutate_population(pop)
        iters -= 1
    play_with_AI()
