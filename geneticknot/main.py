from geneticknot.Population import Population
from tqdm import tqdm
from geneticknot.s3_utils import *


def train_networks(id, iterations, pop_size, input_size, output_size, hidden_layers, board_shape, alpha, mutation_probability):
    pop = Population(id=id, pop_size=pop_size, input_size=input_size, output_size=output_size, hidden_layers=hidden_layers)
    for _ in tqdm(range(iterations), desc="Iterations Progress: "):
        pop.cal_fitness(board_shape=board_shape)
        pop.cal_probability(alpha=alpha)
        pop.save_best_network()
        pop.get_new_pop()
        pop.mutate_population(mutation_probability=mutation_probability)


def distributed_training(id, pop_size, input_size, output_size, hidden_layers, board_shape, workers_num, alpha, mutation_probability, iterations, bucket_name, savebucket):
    best_network = None
    pop = Population(id=id, pop_size=pop_size, input_size=input_size, output_size=output_size, hidden_layers=hidden_layers)
    pop.cal_fitness(board_shape=board_shape)
    pop.upload_best_networks_s3(workers_num, bucket_name)
    wait_for_other_workers(workers_num, bucket_name)
    times = init_times(bucket_name)
    population = download_pops(bucket_name)
    pop.pop = population
    pop.cal_probability(alpha=alpha)
    pop.get_new_pop()
    pop.mutate_population(mutation_probability=mutation_probability)
    for _ in tqdm(range(iterations-1), desc="Iterations Progress: "):
        pop.cal_fitness(board_shape=board_shape)
        pop.upload_best_networks_s3(workers_num, bucket_name)
        wait_for_other_workers(workers_num, bucket_name)
        check_modifies(times, bucket_name)
        population = download_pops(bucket_name)
        pop.pop = population
        pop.cal_probability(alpha=alpha)
        pop.upload_best_network_with_priority_s3(bucket_name=savebucket)
        best_network = pop[0]
        pop.get_new_pop()
        pop.mutate_population(mutation_probability=mutation_probability)
    pop.upload_best_network_with_priority_s3(bucket_name=savebucket)
    return best_network
