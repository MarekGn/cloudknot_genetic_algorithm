from Population import Population
from play_utils import play_with_ai
from tqdm import tqdm


def train_networks(iterations, pop_size, input_size, output_size, hidden_layers, board_shape, alpha, mutation_probability):
    for _ in tqdm(range(iterations), desc="Iterations Progress: "):
        pop = Population(pop_size=pop_size, input_size=input_size, output_size=output_size, hidden_layers=hidden_layers)
        pop.cal_fitness(board_shape=board_shape)
        pop.cal_probability(alpha=alpha)
        pop.save_best_network()
        pop.get_new_pop()
        pop.mutate_population(mutation_probability=mutation_probability)


if __name__ == "__main__":
    iterations = 100
    hidden_layers = [14, 14]
    input_size = 9
    output_size = 9
    pop_size = 666
    board_shape = (3, 3)
    mutation_probability = 0.02
    alpha = 1.3

    # train_networks(
    #     iterations=iterations,
    #     pop_size=pop_size,
    #     input_size=input_size,
    #     output_size=output_size,
    #     hidden_layers=hidden_layers,
    #     board_shape=board_shape,
    #     alpha=alpha,
    #     mutation_probability=mutation_probability
    # )
    play_with_ai(board_shape=board_shape)
