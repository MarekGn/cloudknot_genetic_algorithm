from geneticknot.Population import Population
from geneticknot.play_utils import play_with_ai
from tqdm import tqdm
from geneticknot.s3_utils import *


def train_networks(iterations, pop_size, input_size, output_size, hidden_layers, board_shape, alpha, mutation_probability):
    pop = Population(pop_size=pop_size, input_size=input_size, output_size=output_size, hidden_layers=hidden_layers)
    for _ in tqdm(range(iterations), desc="Iterations Progress: "):
        pop.cal_fitness(board_shape=board_shape)
        pop.cal_probability(alpha=alpha)
        pop.save_best_network()
        pop.get_new_pop()
        pop.mutate_population(mutation_probability=mutation_probability)


def distributed_training(pop_size, input_size, output_size, hidden_layers, board_shape, workers_num, alpha, mutation_probability, iterations, bucket_name):
    pop = Population(pop_size=pop_size, input_size=input_size, output_size=output_size, hidden_layers=hidden_layers)
    pop.cal_fitness(board_shape=board_shape)
    pop.upload_best_networks_s3(workers_num, bucket_name)
    wait_for_other_workers(workers_num, bucket_name)
    times = init_times(bucket_name)
    population = download_pops(bucket_name)
    pop.save_best_network()
    pop.pop = population
    pop.cal_probability(alpha=alpha)
    pop.get_new_pop()
    pop.mutate_population(mutation_probability=mutation_probability)
    for _ in tqdm(range(iterations), desc="Iterations Progress: "):
        pop.cal_fitness(board_shape=board_shape)
        pop.upload_best_networks_s3(workers_num, bucket_name)
        wait_for_other_workers(workers_num, bucket_name)
        check_modifies(times, bucket_name)
        population = download_pops(bucket_name)
        pop.save_best_network()
        pop.pop = population
        pop.cal_probability(alpha=alpha)
        pop.get_new_pop()
        pop.mutate_population(mutation_probability=mutation_probability)


if __name__ == "__main__":
    iterations = 10
    hidden_layers = [14, 14]
    input_size = 9
    output_size = 9
    pop_size = 50
    board_shape = (3, 3)
    mutation_probability = 0.02
    alpha = 1.3
    workers_num = 1
    bucket_name = 'besttictactoe'

    # distributed_training(
    #     pop_size=pop_size,
    #     input_size=input_size,
    #     output_size=output_size,
    #     hidden_layers=hidden_layers,
    #     board_shape=board_shape,
    #     workers_num=workers_num,
    #     alpha=alpha,
    #     mutation_probability=mutation_probability,
    #     iterations=iterations,
    #     bucket_name=bucket_name
    # )
