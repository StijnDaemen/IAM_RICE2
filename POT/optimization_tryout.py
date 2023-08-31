import numpy as np
from POT.tree import PTree
from RICE_model.IAM_RICE import RICE
import random
import time
import copy
import itertools
import pandas as pd
import math

random.seed(0)


class Cluster:
    def __init__(self, num_parents, num_children):
        self.graveyard = {}
        self.VIPs = {}
        self.non_dominated = ()
        self.parents = []
        self.children = []
        self.family = []
        self.num_parents = num_parents
        self.num_children = num_children

        # The center position takes the average dist
        self.center_position = []

        # Model variables
        self.years_10 = []
        for i in range(2005, 2315, 10):
            self.years_10.append(i)

        self.regions = [
            "US",
            "OECD-Europe",
            "Japan",
            "Russia",
            "Non-Russia Eurasia",
            "China",
            "India",
            "Middle East",
            "Africa",
            "Latin America",
            "OHI",
            "Other non-OECD Asia",
        ]
        self.model = RICE(self.years_10, self.regions)
        # Tree variables
        # action_names = ['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05']
        self.action_names = ['miu', 'sr', 'irstp']
        self.action_bounds = [[2100, 2250], [0.2, 0.5], [0.01, 0.1]]
        self.feature_names = ['mat', 'net_output', 'year']
        self.feature_bounds = [[780, 1300], [55, 2300], [2005, 2305]]
        self.max_depth = 4
        self.discrete_actions = False
        self.discrete_features = False
        # Optimization variables
        self.mutation_prob = 0.5
        self.max_nfe = 100000

        # O1 = Organism()
        # O2 = Organism()
        # O1.dna = self.random_tree()
        # O1.fitness = self.policy_tree_RICE_fitness(O1.dna)
        #
        # O2.dna = self.random_tree()
        # O2.fitness = self.policy_tree_RICE_fitness(O2.dna)
        #
        # print(O1.dna)
        # print(O2.dna)

        self.iterate()

        # Create a pandas dataframe out of the graveyard records
        dfs = []
        for i in range(len(self.graveyard.keys())):
            df = pd.DataFrame.from_dict(self.graveyard[i], orient='index', columns=['ofv1', 'ofv2', 'ofv3'])
            df['policy'] = df.index
            df['generation'] = i
            dfs.append(df)
        df = pd.concat(dfs)
        df.reset_index(drop=True, inplace=True)
        print(df.head)
        df.to_excel('generational_test_single_cluster_graveyard_max_nfe_100000.xlsx')

        # Create a pandas dataframe out of the VIPs records
        dfs = []
        for i in range(len(self.VIPs.keys())):
            df = pd.DataFrame.from_dict(self.VIPs[i], orient='index', columns=['ofv1', 'ofv2', 'ofv3'])
            df['policy'] = df.index
            df['generation'] = i
            dfs.append(df)
        df = pd.concat(dfs)
        df.reset_index(drop=True, inplace=True)
        print(df.head)
        df.to_excel('generational_test_single_cluster_VIPs_max_nfe_100000.xlsx')

    def iterate(self):
        nfe = 0
        generation = 0
        while nfe < self.max_nfe:

            # Swipe generational variables
            self.parents = []
            self.children = []
            self.family = []

            self.populate()
            self.natural_selection()

            # determine position of parents in solution space
            self.center_position.append(self.determine_center_position(self.parents[0].fitness, self.parents[1].fitness))

            nfe += len(self.family)

            # Record the non dominated organisms per generation
            graveyard_dict = {}
            for member in self.family:
                graveyard_dict[str(member.dna)] = member.fitness
            self.graveyard[generation] = graveyard_dict

            # Record all organisms per generation
            VIPs_dict = {}
            for member in self.non_dominated:
                VIPs_dict[str(member.dna)] = member.fitness
            self.VIPs[generation] = VIPs_dict

            generation += 1

            # print(f'family: {len(self.family)}')
            # print(f'nfe: {nfe}')
            # print(f'generation: {generation}')
        # print(self.graveyard.keys())
        return

    def random_tree(self, terminal_ratio=0.5,
                    # discrete_actions=True,
                    # discrete_features=None,
                    ):

        num_features = len(self.feature_names)
        num_actions = len(self.action_names)  # SD changed

        depth = np.random.randint(1, self.max_depth + 1)
        L = []
        S = [0]

        while S:
            current_depth = S.pop()

            # action node
            if current_depth == depth or (current_depth > 0 and \
                                          np.random.rand() < terminal_ratio):
                if self.discrete_actions:
                    L.append([str(np.random.choice(self.action_names))])
                else:
                    # TODO:: actions are not mutually exclusive, make it so that multiple actions can be activated by the same leaf node
                    a = np.random.choice(num_actions)  # SD changed
                    action_name = self.action_names[a]
                    action_value = np.random.uniform(*self.action_bounds[a])
                    action_input = f'{action_name}_{action_value}'
                    # L.append([np.random.uniform(*self.action_bounds[a])])  # SD changed
                    L.append([action_input])  # SD changed

            else:
                x = np.random.choice(num_features)
                v = np.random.uniform(*self.feature_bounds[x])
                L.append([x, v])
                S += [current_depth + 1] * 2

        T = PTree(L, self.feature_names, self.discrete_features)
        T.prune()
        return T

    def policy_tree_RICE_fitness(self, T):
        m1, m2, m3 = self.model.POT_control(T)
        # print(m1, m2, m3, T)
        return [m1, m2, m3]

    def populate(self):
        # If there are no parents, create them
        if not self.non_dominated:
            for _ in range(self.num_parents):
                parent = Organism()
                parent.dna = self.random_tree()
                parent.fitness = self.policy_tree_RICE_fitness(parent.dna)
                self.parents.append(parent)

        # If there are possible parents, choose two and let the rest be children
        elif self.non_dominated:
            if len(self.non_dominated) >= 2:
                P1, P2 = np.random.choice(
                    self.non_dominated, 2, replace=False)
                self.parents.append(P1)
                self.parents.append(P2)

                # Take the other non_dominated solutions as children
                self.children = [i for i in self.non_dominated if i not in self.parents]
                # Chuck two(?)/ half out if every solution was non-dominated i.e. len(self.non_dominated) >= 8
                if len(self.non_dominated) >= self.num_children+self.num_parents:  # 8
                    idx = self.num_children-2
                    self.children = self.children[idx:]  # 6:
            # Else if there is only 1 suitable parent, choose it (ofcourse) and create a random other parent
            else:
                self.parents = [self.non_dominated[0]]
                # Create a random other parent
                parent = Organism()
                parent.dna = self.random_tree()
                parent.fitness = self.policy_tree_RICE_fitness(parent.dna)
                self.parents.append(parent)

            # TODO:: In case every family member is non dominated, there is no more progression, what to do?
        self.family.extend(self.parents)
        self.family.extend(self.children)

        # print(f'parents: {len(self.parents)}')
        # print(f'children: {len(self.children)}')
        # print(f'family: {len(self.family)}')

        # for _ in range(self.num_children):
        while len(self.children) < self.num_children:
            # print(len(self.children))
            child = Organism()
            P1, P2 = np.random.choice(
                self.parents, 2, replace=False)
            child.dna = self.crossover(P1.dna, P2.dna)[0]

            # bloat control
            while child.dna.get_depth() > self.max_depth:
                child.dna = self.crossover(P1.dna, P2.dna)[0]

            # Mutate (with probability of mutation accounted for in function)
            child.dna = self.mutate(child.dna)
            child.dna.prune()

            # Welcome child to family
            child.fitness = self.policy_tree_RICE_fitness(child.dna)
            self.children.append(child)
            self.family.append(child)

    def natural_selection(self):
        self.non_dominated = []

        # for member in self.family:
        #     print(member.fitness)
        # Get all possible combinations in family
        for organism_combo in itertools.permutations(self.family, 2):
            # print(organism_combo)
            if self.dominates(organism_combo[0].fitness, organism_combo[1].fitness):
                # print(f'{organism_combo[0]} dominates {organism_combo[1]}')
                self.non_dominated.append(organism_combo[0])
        return

    def dominates(self, a, b):
        # assumes minimization
        # a dominates b if it is <= in all objectives and < in at least one
        # Note SD: somehow the logic with np.all() breaks down if there are positive and negative numbers in the array
        # So to circumvent this but still allow multiobjective optimisation in different directions under the
        # constraint that every number is positive, just add a large number to every index.
        large_number = 1000000000

        a = np.array(a)
        a = a + large_number

        b = np.array(b)
        b = b + large_number
        # print(f'a: {a}')
        # print(f'b: {b}')
        return (np.all(a <= b) and np.any(a < b))

    def crossover(self, P1, P2):
        P1, P2 = [copy.deepcopy(P) for P in (P1, P2)]
        # should use indices of ONLY feature nodes
        feature_ix1 = [i for i in range(P1.N) if P1.L[i].is_feature]
        feature_ix2 = [i for i in range(P2.N) if P2.L[i].is_feature]
        index1 = np.random.choice(feature_ix1)
        index2 = np.random.choice(feature_ix2)
        slice1 = P1.get_subtree(index1)
        slice2 = P2.get_subtree(index2)
        P1.L[slice1], P2.L[slice2] = P2.L[slice2], P1.L[slice1]
        P1.build()
        P2.build()
        return (P1, P2)

    def mutate(self, P, mutate_actions=True):
        P = copy.deepcopy(P)

        for item in P.L:
            if np.random.rand() < self.mutation_prob:
                if item.is_feature:
                    low, high = self.feature_bounds[item.index]
                    if item.is_discrete:
                        item.threshold = np.random.randint(low, high + 1)
                    else:
                        item.threshold = self.bounded_gaussian(
                            item.threshold, [low, high])
                elif mutate_actions:
                    if self.discrete_actions:
                        item.value = str(np.random.choice(self.action_names))
                    else:
                        # print(item)
                        # print(self.action_bounds)
                        # print(item.value)
                        # item.value = self.bounded_gaussian(
                        #     item.value, self.action_bounds)

                        a = np.random.choice(len(self.action_names))  # SD changed
                        action_name = self.action_names[a]
                        action_value = np.random.uniform(*self.action_bounds[a])
                        action_input = f'{action_name}_{action_value}'
                        # print(action_input)
                        item.value = action_input

        return P

    def bounded_gaussian(self, x, bounds):
        # do mutation in normalized [0,1] to avoid sigma scaling issues
        lb, ub = bounds
        xnorm = (x - lb) / (ub - lb)
        x_trial = np.clip(xnorm + np.random.normal(0, scale=0.1), 0, 1)

        return lb + x_trial * (ub - lb)

    def determine_center_position(self, P1, P2):
        num_dimensions = len(P1)
        center_position = []
        for dimension in range(num_dimensions):
            avg_pos = P1[dimension] + ((P2[dimension]-P1[dimension])/2)
            center_position.append(avg_pos)
        return center_position


        # Step 1: select best performing parent -> create dominates() function
        # Step 2: let other parents mutate -> if mutation dominates over

        # -----------
        # create children by cross-over
        # let all within a population mutate -> keep the better performing version
        # Choose the best number of organisms equal to num_parents and choose them as parent clusters -> repeat process

        # ----------
        # When a cluster is not performing better, keep its solutions and create a new (random) cluster.
        # When the fitness of two clusters starts to approach each other, freeze one and let the other grow


class Organism:
    def __init__(self):
        self.dna = None
        self.fitness = None


Cluster(2, 8)


class ClusterOpt:
    def __init__(self):
        # np.random.seed(time.perf_counter())
        self.C1 = Cluster(2, 8)
        # np.random.seed(time.perf_counter())
        self.C2 = Cluster(2, 8)

        print(self.C1.center_position)
        print(self.C1.center_position)

        print(self.distance(self.C1.center_position[0], self.C2.center_position[0]))

    def distance(self, P1, P2):
        # Input is list
        num_dimensions = len(P1)
        dist = []
        for dimension in range(num_dimensions):
            dist_ = (P2[dimension] - P1[dimension]) ** 2
            dist.append(dist_)
        distance = math.sqrt(sum(dist))
        return distance
        # return math.sqrt(((P2[0] - P1[0]) ** 2) + ((P2[1] - P1[1]) ** 2) + ((P2[2] - P1[2]) ** 2))


# ClusterOpt()


# -------------------------------------------------------------------------------------------------------
# O1 = Organism()
# O2 = Organism()
#
# print(O1.dna)
# print(O2.dna)

    # def create(self):
    #     self.dna = self.genetic_blueprint.random_tree()
    #     pass
    #
    # def live_and_die(self):
    #     self.fitness = self.genetic_blueprint.policy_tree_RICE_fitness(self.dna)
    #     pass


# class Parent(Organism):
#     pass
#
#
# class Child(Organism):
#     pass
#
#
# def random_tree(self, terminal_ratio=0.5,
#                 # discrete_actions=True,
#                 # discrete_features=None,
#                 ):
#
#     num_features = len(feature_names)
#     num_actions = len(action_names) # SD changed
#
#     depth = np.random.randint(1, max_depth + 1)
#     L = []
#     S = [0]
#
#     while S:
#         current_depth = S.pop()
#
#         # action node
#         if current_depth == depth or (current_depth > 0 and \
#                                       np.random.rand() < terminal_ratio):
#             if self.discrete_actions:
#                 L.append([str(np.random.choice(action_names))])
#             else:
#                 # TODO:: actions are not mutually exclusive, make it so that multiple actions can be activated by the same leaf node
#                 a = np.random.choice(num_actions)  # SD changed
#                 action_name = action_names[a]
#                 action_value = np.random.uniform(*action_bounds[a])
#                 action_input = f'{action_name}_{action_value}'
#                 # L.append([np.random.uniform(*self.action_bounds[a])])  # SD changed
#                 L.append([action_input])  # SD changed
#
#         else:
#             x = np.random.choice(num_features)
#             v = np.random.uniform(*feature_bounds[x])
#             L.append([x, v])
#             S += [current_depth + 1] * 2
#
#     T = PTree(L, feature_names, discrete_features)
#     T.prune()
#     return T
#
#
# class PolicyTree:
#     def __init__(self, model, action_names,  action_bounds, discrete_actions, feature_names, feature_bounds, discrete_features, epsilon=0.1, max_nfe=1000, max_depth=4, population_size=100):
#         self.model = model
#         # Tree variables
#         self.action_names = action_names
#         self.action_bounds = action_bounds
#         self.discrete_actions = discrete_actions
#         self.feature_names = feature_names
#         self.feature_bounds = feature_bounds
#         self.discrete_features = discrete_features
#         self.max_depth = max_depth
#         # Optimization variables
#         self.epsilon = epsilon
#         self.max_nfe = max_nfe
#         self.population_size = population_size
#
#         T1 = self.random_tree()
#         T2 = self.random_tree()
#         print(T1)
#         print(T2)
#
#     def random_tree(self, terminal_ratio=0.5,
#                     # discrete_actions=True,
#                     # discrete_features=None,
#                     ):
#
#         num_features = len(self.feature_names)
#         num_actions = len(self.action_names) # SD changed
#
#         depth = np.random.randint(1, self.max_depth + 1)
#         L = []
#         S = [0]
#
#         while S:
#             current_depth = S.pop()
#
#             # action node
#             if current_depth == depth or (current_depth > 0 and \
#                                           np.random.rand() < terminal_ratio):
#                 if self.discrete_actions:
#                     L.append([str(np.random.choice(self.action_names))])
#                 else:
#                     # TODO:: actions are not mutually exclusive, make it so that multiple actions can be activated by the same leaf node
#                     a = np.random.choice(num_actions)  # SD changed
#                     action_name = self.action_names[a]
#                     action_value = np.random.uniform(*self.action_bounds[a])
#                     action_input = f'{action_name}_{action_value}'
#                     # L.append([np.random.uniform(*self.action_bounds[a])])  # SD changed
#                     L.append([action_input])  # SD changed
#
#             else:
#                 x = np.random.choice(num_features)
#                 v = np.random.uniform(*self.feature_bounds[x])
#                 L.append([x, v])
#                 S += [current_depth + 1] * 2
#
#         T = PTree(L, self.feature_names, self.discrete_features)
#         T.prune()
#         return T
#
#     def policy_tree_RICE_fitness(self, T):
#         m1, m2, m3 = self.model.POT_control(T)
#         # print(m1, m2, m3, T)
#         return m1, m2, m3
#
#
# # Cluster(num_parents=2, num_children=5).populate()
#
# # Model variables
# years_10 = []
# for i in range(2005, 2315, 10):
#     years_10.append(i)
#
# regions = [
#     "US",
#     "OECD-Europe",
#     "Japan",
#     "Russia",
#     "Non-Russia Eurasia",
#     "China",
#     "India",
#     "Middle East",
#     "Africa",
#     "Latin America",
#     "OHI",
#     "Other non-OECD Asia",
# ]
# # Tree variables
# # action_names = ['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05']
# action_names = ['miu', 'sr', 'irstp']
# action_bounds = [[2100, 2250], [0.2, 0.5], [0.01, 0.1]]
# feature_names = ['mat', 'net_output', 'year']
# feature_bounds = [[780, 1300], [55, 2300], [2005, 2305]]
# # Save variables
# # database_POT = 'C:/Users/Stijn Daemen/Documents/master thesis TU Delft/code/IAM_RICE2/jupyter notebooks/Tests_Borg.db'
# # table_name_POT = 'Test1_couplingborg_not_edited_borg'
#
# PolicyTree(model=RICE(years_10, regions),
#                 # model=RICE(years_10, regions, database_POT=database_POT, table_name_POT=table_name_POT),
#                 action_names=action_names,
#                 action_bounds=action_bounds,
#                 discrete_actions=False,
#                 feature_names=feature_names,
#                 feature_bounds=feature_bounds,
#                 discrete_features=False,
#                 epsilon=0.1,
#                 max_nfe=6,
#                 max_depth=4,
#                 population_size=3)
#
# # T1 = PolicyTree(model=RICE(years_10, regions),
# #                 # model=RICE(years_10, regions, database_POT=database_POT, table_name_POT=table_name_POT),
# #                 action_names=action_names,
# #                 action_bounds=action_bounds,
# #                 discrete_actions=False,
# #                 feature_names=feature_names,
# #                 feature_bounds=feature_bounds,
# #                 discrete_features=False,
# #                 epsilon=0.1,
# #                 max_nfe=6,
# #                 max_depth=4,
# #                 population_size=3).random_tree()
# # T2 = PolicyTree(model=RICE(years_10, regions),
# #                 # model=RICE(years_10, regions, database_POT=database_POT, table_name_POT=table_name_POT),
# #                 action_names=action_names,
# #                 action_bounds=action_bounds,
# #                 discrete_actions=False,
# #                 feature_names=feature_names,
# #                 feature_bounds=feature_bounds,
# #                 discrete_features=False,
# #                 epsilon=0.1,
# #                 max_nfe=6,
# #                 max_depth=4,
# #                 population_size=3).random_tree()
# #
# # print(T1)
# # print(T2)

