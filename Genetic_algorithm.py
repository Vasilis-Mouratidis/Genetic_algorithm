from copy import deepcopy

import numpy as np
from math import sqrt, floor, ceil


class Helper:
    @staticmethod
    def euclidean_distance(coord1: tuple, coord2: tuple) -> float:
        """
        Returns the euclidean distance between 2 coordinates in a 2D plane
        :param coord1: (x1, y1)
        :param coord2: (x2, y2)
        :return:
        """
        return sqrt(pow(coord1[0] - coord2[0], 2) + pow(coord1[1] - coord2[1], 2))

    @staticmethod
    def generate_bit_vector(size: int):
        """
        Generates a random bit vector given the size parameter
        :param size:
        :return:
        """
        bit_vector = []
        for i in range(size):
            bit_vector.append(np.random.randint(0, 2))

        return bit_vector


class WaspExtermination:

    def __init__(self, max_x: int, max_y: int, bomb_set_length: int, random_nests=False):
        self.MAX_X = max_x
        self.MAX_Y = max_y
        self.dmax = self.calculate_max_distance_between_nests()
        self.set_of_bombs = self._generate_random_bomb_triplets(bomb_set_length)
        if random_nests:
            self.nests = self._generate_random_nests(int(0.1 * max(self.MAX_X, self.MAX_Y)))
        else:
            self.nests = self._generate_fixed_nests()

    def _generate_random_nests(self, amount_of_nests):
        """
        Generates random nests by making sure a nest exists only once in the "map"
        :param amount_of_nests:
        :return:
        """
        nests = set()
        while len(nests) != amount_of_nests:
            random_x = np.random.randint(0, self.MAX_X)
            random_y = np.random.randint(0, self.MAX_Y)
            nests.add((random_x, random_y))

        nests = [[x, y] for x, y in list(nests)]
        for nest in nests:
            amount_of_rats = np.random.randint(0, 1000)
            nest.append(amount_of_rats)
        return nests
        #[65,60], [55,50], [45,40]

    def _generate_fixed_nests(self):
        return [[25, 65, 100],
                [23, 8, 200],
                [7, 13, 327],
                [95, 53, 440],
                [3, 3, 450],
                [54, 56, 639],
                [67, 78, 650],
                [32, 4, 678],
                [24, 76, 750],
                [66, 89, 801],
                [94, 4, 945],
                [34, 23, 967]]

    def _generate_random_bomb_triplets(self, amount_of_rounds):
        """
        Generates random bomb triplets. Should be used only at initialization
        :param amount_of_rounds:
        :return:
        """
        return [[[np.random.randint(0, self.MAX_X), np.random.randint(0, self.MAX_Y)] for _ in range(3)]
                for _ in range(amount_of_rounds)]

    def calculate_max_distance_between_nests(self):
        """
        Calculate the max distance that is possible based on MAX_X and MAX_Y
        :return:
        """
        return Helper.euclidean_distance((0, 0), (self.MAX_X, self.MAX_Y))

    def place_bombs(self):
        """
        Drops each bomb triplet, calculates the efficiency and adds it at the end of each bomb set
        :return:
        """
        for bomb_triplet in self.set_of_bombs:
            bomb_triplet_efficiency = self.get_bomb_triplet_efficiency(bomb_triplet)
            bomb_triplet.append(bomb_triplet_efficiency)

    # [[0,0], [1,1], [2,2], 0.6623423]

    def get_bomb_triplet_efficiency(self, bomb_set):
        """
        Drops the bombs of the given bomb_set and calculates the efficiency with the relation:
            efficiency = total_bomb_set_kills / total_wasps_available
        :param bomb_set:
        :return:
        """
        nests_copy = [x[:] for x in self.nests]
        # Drop the bombs of the bomb triplet
        for bomb in bomb_set:
            [bomb_x, bomb_y] = bomb
            for nest in nests_copy:
                [nest_x, nest_y, wasps_available] = nest
                distance = Helper.euclidean_distance((bomb_x, nest_x), (bomb_y, nest_y))
                k = wasps_available * self.dmax / ((20 * distance) + 1e-5)
                nest[2] = max(0, nest[2] - floor(k))

        # Calculate efficiency
        total_killed = 0
        total_wasps = 0
        for [new_nest, old_nest] in zip(nests_copy, self.nests):
            total_killed += old_nest[2] - new_nest[2]
            total_wasps += old_nest[2]

        bomb_set_efficiency = total_killed / total_wasps
        return bomb_set_efficiency

    def sort_based_on_efficiency_descending(self):
        """
        Sorts the set of bombs based on efficiency in descending order
        :return:
        """
        self.set_of_bombs.sort(key=lambda x: x[3], reverse=True)

    def get_top_x_placements(self, x: int):
        """
        Returns a tuple with the best parents and best efficiency and drops the efficiency column for each bomb_set
        :param x:
        :return:
        """
        x = x if x % 2 == 0 else x - 1
        best_efficiency = self.set_of_bombs[0][3]
        best_parents = list(map(lambda y: y[:-1], self.set_of_bombs[:x]))
        return best_parents, best_efficiency

    def generate_offspring(self, parent1, parent2, bit_vector):
        """
        Generate a new offspring which inherits its coordinates based on the bit vector given
        :param parent1:
        :param parent2:
        :param bit_vector:
        :return:
        """
        offspring = []
        for j in range(len(parent1)):
            if bit_vector[j] == 0:
                offspring.append(parent1[j])
            else:
                offspring.append(parent2[j])

        return offspring

    def get_offsprings(self, best_parents):
        """
        Return the offsprings of the best parents, where the first offspring inherits its bomb locations based on a
        randomly generated bit_vector(v), and the second offspring does the same with the inverted bit_vector(~v)
        :param best_parents:
        :return:
        """
        offsprings = []
        for i in range(0, len(best_parents), 2):
            parent1 = best_parents[i]
            parent2 = best_parents[i + 1]
            # offspring 1
            bit_vector = Helper.generate_bit_vector(size=3)
            offsprings.append(self.generate_offspring(parent1, parent2, bit_vector))
            # offspring 2
            inverted_bit_vector = [1 if bit == 0 else 0 for bit in bit_vector]  # invert bits for second offspring
            offsprings.append(self.generate_offspring(parent1, parent2, inverted_bit_vector))
        return offsprings

    def mutate_offsprings(self, offsprings):
        """
        Randomly mutate the bombs in each set of bombs by incrementing or decrementing their coordinates by 10% of the max size
        :param offsprings:
        :return:
        """
        for offspring in offsprings:
            for bomb in offspring:
                # Solution for any size of map
                mutation_based_on_map = ceil(0.1 * self.MAX_X)
                # Randomly decide how much is the mutation going to be
                mutation_step = np.random.randint(0, mutation_based_on_map)
                # Get a random number from [0 - 3]
                move = np.random.randint(0, 4)

                # Increase X axis
                if move == 0:
                    bomb[0] = min(self.MAX_X - 1, bomb[0] + mutation_step)
                # Increase Y axis
                if move == 1:
                    bomb[1] = min(self.MAX_Y - 1, bomb[1] + mutation_step)
                # Decrease X axis
                if move == 2:
                    bomb[0] = max(0, bomb[0] - mutation_step)
                # Decrease Y axis
                if move == 3:
                    bomb[1] = max(0, bomb[1] - mutation_step)


def main():
    max_efficiency = 0
    generation = 0
    generations = {}
    bomb_set_length = 100
    max_x, max_y = 100, 100
    random_nests = False
    max_generations = 10000
    generation_look_back = 1000
    best_triplet = None
    wasp_extermination = WaspExtermination(max_x, max_y, bomb_set_length, random_nests)

    if random_nests:
        print('Initial random map nest:')
        for nest in wasp_extermination.nests:
            print(f'\t\t{nest}')
        print()

    # each loop is a generation
    while True:
        generation += 1

        # Simulate bomb planting
        wasp_extermination.place_bombs()

        # Sort bomb triplets based on efficiency in a descending order
        wasp_extermination.sort_based_on_efficiency_descending()

        # Retrieve the best parents and the best efficiency for this generation
        best_parents, best_efficiency = wasp_extermination.get_top_x_placements(bomb_set_length // 2)   # Get best parents and best efficiency

        if best_efficiency > max_efficiency:
            max_efficiency = best_efficiency
            best_triplet = deepcopy(best_parents[0])

        # Set generation efficiency
        generations[generation] = {
            'best_efficiency': best_efficiency,
            'best_triplet': deepcopy(best_parents[0])
        }

        # Generate their offsprings
        offsprings = wasp_extermination.get_offsprings(best_parents)

        # Mutate the offspringsmutate_offsprings
        wasp_extermination.mutate_offsprings(offsprings)

        # Keep best parents and offsprings together for the next generation
        best_parents.extend(offsprings)

        # Prepare for next generation
        wasp_extermination.set_of_bombs = deepcopy(best_parents)

        # print(generation, max_efficiency)

        # Termination criteria: Passed max generations
        if generation > max_generations:
            break

        # Termination criteria: Current generation is worse than 100 previous
        if generation > generation_look_back:
            # generation = 15
            # lookback_range = [10, 11, 12, 13, 14]
            local_maximum = [generations[gen]['best_efficiency'] > generations[generation]['best_efficiency'] for gen in
                             range(generation - generation_look_back, generation)]
            if all(local_maximum):
                break

    print('Efficiency per generation')
    for generation in generations.keys():
        print(f"\tGeneration {generation}: {generations[generation]['best_efficiency'] * 100}%")
        print(f"\tTriplet: {generations[generation]['best_triplet']}\n")

    print(f'Max efficiency: {max_efficiency* 100}%')
    print(f'Best triplet: {best_triplet}')

if __name__ == '__main__':
    main()
