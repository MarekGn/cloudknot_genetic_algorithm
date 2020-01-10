import itertools
import numpy as np
from geneticknot.play_utils import *


class Population():
    def __init__(self, id, pop_size, input_size, hidden_layers, output_size):
        self.id = id
        self.pop = self._get_initial_population(pop_size, input_size, hidden_layers, output_size)
        self.pop_size = pop_size
        self.idv_input_size = input_size
        self.idv_output_size = output_size
        self.idv_hidden_layers = hidden_layers

    def cal_fitness(self, board_shape):
        confrontations = itertools.permutations(self.pop, 2)
        for confrontation in confrontations:
            play_agents_tictactoe(confrontation, board_shape)
        for idv in self.pop:
            idv.cal_fitness()
        self.pop.sort(key=lambda idv: idv.fitness, reverse=True)

    def _get_initial_population(self, pop_size, input_size, hidden_layers, output_size):
        pop = []
        for _ in range(pop_size):
            pop.append(DNN(input_size, output_size, hidden_layers))
        return pop

    def cal_probability(self, alpha):
        self.pop.sort(key=lambda idv: idv.fitness, reverse=True)
        # If lowest fitness is a negative number shift all fitness this difference
        if self.pop[-1].fitness < 0:
            for idv in self.pop:
                idv.fitness += abs(self.pop[-1].fitness)
        fitness_sum = 0
        for idv in self.pop:
            fitness_sum += idv.fitness
        fitness_avg = fitness_sum / len(self.pop)
        self.normalize_fitness(fitness_avg, alpha)
        fitness_sum = 0
        for idv in self.pop:
            fitness_sum += idv.fitness
        if fitness_sum != 0:
            for idv in self.pop:
                idv.probability = idv.fitness / fitness_sum
        else:
            for idv in self.pop:
                idv.probability = 1 / len(self.pop)

    def normalize_fitness(self, fitness_avg, alpha):
        delta = self.pop[0].fitness - fitness_avg
        if delta == 0:
            delta = 1
        a = (fitness_avg*(alpha - 1)) / delta
        b = fitness_avg * (1 - a)
        for idv in self.pop:
            idv.fitness = a*idv.fitness + b

    def get_new_pop(self):
        children = []
        while len(children) < self.pop_size:
            choice = np.random.uniform(0, 1, 2)
            parent_1 = None
            parent_1_call = True
            parent_2 = None
            parent_2_call = True
            prob = 0
            for idv in self.pop:
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
                self._add_child_offset(children, [parent_1, parent_2])
        self.pop = children

    def _add_child_offset(self, children, parent_couple):
        parent1vector = parent_couple[0].toVector()
        parent2vector = parent_couple[1].toVector()

        child1 = DNN(self.idv_input_size, self.idv_output_size, self.idv_hidden_layers)
        child2 = DNN(self.idv_input_size, self.idv_output_size, self.idv_hidden_layers)
        offset = np.random.randint(0, len(parent1vector))
        child1vector = np.concatenate((parent1vector[:offset], parent2vector[offset:]))
        child1vector = np.reshape(child1vector, parent1vector.shape)
        child1.fromVector(child1vector)
        children.append(child1)

        child2vector = np.concatenate((parent2vector[:offset], parent1vector[offset:]))
        child2vector = np.reshape(child2vector, parent1vector.shape)
        child2.fromVector(child2vector)
        children.append(child2)

    def mutate_population(self, mutation_probability):
        for idv in self.pop:
            self._mutate_idv(idv, mutation_probability)

    def _mutate_idv(self, idv, mutation_probability):
        gene_vector = idv.toVector()
        gene_prob = mutation_probability / len(gene_vector)
        for i in range(len(gene_vector)):
            chance = np.random.uniform(0, 1)
            if chance <= gene_prob:
                gene_vector[i] += np.random.normal(0, 10)
        idv.fromVector(gene_vector)

    def save_best_network(self):
        self.pop[0].save_network()

    def upload_best_networks_s3(self, workers_num, bucket_name):
        from botocore.exceptions import ClientError
        import boto3

        best_idvs = self.pop[:int(self.pop_size/workers_num)]
        np.save("best_nets{}.npy".format(self.id), best_idvs, allow_pickle=True)
        s3_client = boto3.client('s3')
        try:
            s3_client.upload_file("best_nets{}.npy".format(self.id), bucket_name, "best_nets{}.npy".format(self.id))
            return True
        except ClientError as e:
            print(e.args)
            return False

    def upload_best_network(self, bucket_name):
        from botocore.exceptions import ClientError
        import boto3

        np.save("BEST_NET.npy", self.pop[0], allow_pickle=True)
        s3_client = boto3.client('s3')
        try:
            s3_client.upload_file("BEST_NET.npy", bucket_name, "BEST_NET.npy")
            return True
        except ClientError as e:
            print(e.args)
            return False

    def upload_best_network_with_priority_s3(self, bucket_name):
        from botocore.exceptions import ClientError
        import boto3
        if self.id == 1:
            np.save("BEST_NET.npy", self.pop[0], allow_pickle=True)
            s3_client = boto3.client('s3')
            try:
                s3_client.upload_file("BEST_NET.npy", bucket_name, "BEST_NET.npy")
                return True
            except ClientError as e:
                print(e.args)
                return False




