# import numpy as np
#
#
# class Node(object):
#     def __init__(self, value):
#         self.left = None
#         self.right = None
#         self.value = value
#
#     def __str__(self):
#         raise NotImplementedError('Node object -- Must be defined in a child class')
#
#
# class BST:
#     def __init__(self):
#         self.root = None
#
#     def add(self, current, value):
#         if self.root == None:
#             self.root = Node(value)
#         else:
#             if value < current.value:
#                 if current.left == None:
#                     current.left = Node(value)
#                 else:
#                     self.add(current.left, value)
#
#             else:
#                 if current.right == None:
#                     current.right = Node(value)
#                 else:
#                     self.add(current.right, value)

### MADE THE TREE IN SUCH A WAY THAT EVALUATION IS BUILT IN THE ADDIN GOF A NODE #######################################
# import random
#
#
# class Node(object):
#     def __init__(self, indicator, threshold):
#         self.indicator = indicator
#         self.threshold = threshold
#         self.left = None
#         self.right = None
#
#
# class BinaryTree(object):
#     def __init__(self, indicator_dict, max_depth):
#         self.indicator_dict = indicator_dict
#         indicator_threshold = self.pick_random_indicator_threshold()
#         self.root = Node(indicator_threshold[0], indicator_threshold[1])
#         self.add_node(self.root.indicator, self.root.threshold)
#         if self.root.left:
#             print('root left is true')
#         if self.root.right:
#             print('root right is true')
#         # i = 0
#         # while i < max_depth:
#         #     self.add_node(self.root.indicator, self.root.threshold)
#
#     def evaluate(self, indicator, threshold):
#         if indicator < threshold:
#             return True
#         else:
#             return False
#
#     def add_node(self, indicator_parent, threshold_parent):
#         indicator_threshold = self.pick_random_indicator_threshold()
#         if self.evaluate(indicator_parent, threshold_parent):
#             self.root.left = Node(indicator_threshold[0], indicator_threshold[1])
#         else:
#             self.root.right = Node(indicator_threshold[0], indicator_threshold[1])
#
#     def pick_random_indicator_threshold(self):
#         indicator = random.choice(list(self.indicator_dict.keys()))
#         threshold = random.randint(self.indicator_dict[indicator][0], self.indicator_dict[indicator][1])
#         return indicator, threshold
#
#     def print_tree(self, traversal_type):
#         if traversal_type == "preorder":
#             return self.preorder_print(self.root, "")
#         else:
#             print("Traversal type " + traversal_type + " not supported.")
#             return False
#
#     def preorder_print(self, start, traversal):
#         '''Root -> Left -> Right'''
#         if start:
#             traversal += (str(start.indicator) + "/" + str(start.threshold) + "-")
#             traversal = self.preorder_print(start.left, traversal)
#             traversal = self.preorder_print(start.right, traversal)
#         return traversal
#
#
# # indicator_dict = {'indicator_1': [0, 5],
# #          'indicator_2': [6,10]}
# indicator_dict = {2: [0, 5],
#          4: [6,10]}
#
#
# # print(list(indicator_dict.keys()))
# # print(random.choice(list(indicator_dict.keys())))
# # print(random.randint(indicator_dict['indicator_1'][0], indicator_dict['indicator_1'][1]))
#
# tree = BinaryTree(indicator_dict=indicator_dict, max_depth=4)
# # print(tree.root.indicator)
# # print(tree.root.threshold)
# print(tree.print_tree("preorder"))
# tree.root.left = Node(2)
# tree.root.right = Node(3)
# tree.root.left.left = Node(4)
#
# print(tree.root)

### REDO action and indicator classes ######################################################################################

# import random
#
# import matplotlib.pyplot as plt
# import networkx as nx
# from networkx.drawing.nx_agraph import graphviz_layout
#
#
# class Node(object):
#     def __init__(self, indicator, threshold):
#         self.indicator = indicator
#         self.threshold = threshold
#         self.left = None
#         self.right = None
#
#
# class ActionNode(object):
#     def __init__(self, action):
#         self.action = action
#
#
# class BinaryTree(object):
#     def __init__(self, indicator_dict, action_list, max_depth):
#         self.indicator_dict = indicator_dict
#         self.action_list = action_list
#         indicator_threshold = self.pick_random_indicator_threshold()
#         self.root = Node(indicator_threshold[0], indicator_threshold[1])
#         # self.root.left = Node(2,3)
#         # self.root.left.left = Node(4,2)
#         # self.root.left.right = Node(7,9)
#         # self.root.right = Node(1,3)
#
#         # self.try_action_node = ActionNode('a2')
#         # print(type(self.try_action_node))
#         # print(f"is action node:  {isinstance(ActionNode, type(self.try_action_node))} ")
#         # print(f"is action node:  {isinstance(self.try_action_node, ActionNode)} ")
#
#         self.leaf_nodes = []
#         self.find_leaf_nodes(self.root, '')
#         # print(self.leaf_nodes)
#
#         # self.add_node(self.leaf_nodes[0])
#
#         depth = 0
#         while depth <= max_depth:
#             self.find_leaf_nodes(self.root, '')
#             random_leaf_node = random.choice(self.leaf_nodes)
#             # randomly assign or another indicator node or an action node
#             random_bit = random.getrandbits(1)
#             if random_bit == 0:
#                 self.add_action_node(random_leaf_node)
#             elif random_bit == 1:
#                 self.add_node(random_leaf_node)
#                 depth = self.maxDepth(self.root)
#
#             # print(depth)
#
#
#
#         # iterator = self.root
#         # while (not iterator.left) & (not iterator.right):
#         #     if iterator.left:
#         #         iterator = iterator.left
#
#         # # Randomly create nodes
#         # i = 0
#         # node = self.root
#         # self.add_node(parent_node=node)
#         # while i < max_depth:
#         #     print(node.indicator)
#         #     if node.left:
#         #         print('made left node')
#         #         node = node.left
#         #     else:
#         #         print('made right node')
#         #         node = node.right
#         #     self.add_node(parent_node=node)
#         #     i += 1
#
#     def find_leaf_nodes(self, start, iterator):
#         # if isinstance(ActionNode, type(start)):
#         #     return iterator
#         # elif start:
#         #     if (not start.left) & (not start.right):
#         #         # self.leaf_nodes.append(str(start.indicator)+"/"+str(start.threshold))
#         #         self.leaf_nodes.append(start)
#         #     iterator = self.find_leaf_nodes(start.left, iterator)
#         #     iterator = self.find_leaf_nodes(start.right, iterator)
#         #     return iterator
#         if not isinstance(start, ActionNode):
#             if start:
#                 if (not start.left) & (not start.right):
#                     # self.leaf_nodes.append(str(start.indicator)+"/"+str(start.threshold))
#                     self.leaf_nodes.append(start)
#                 iterator = self.find_leaf_nodes(start.left, iterator)
#                 iterator = self.find_leaf_nodes(start.right, iterator)
#         return iterator
#
#     def maxDepth(self, node):
#         if node is None:
#             return 0
#         else:
#             if not isinstance(node, ActionNode):
#                 # Compute the depth of each subtree
#                 lDepth = self.maxDepth(node.left)
#                 rDepth = self.maxDepth(node.right)
#                 # Use the larger one
#                 if (lDepth > rDepth):
#                     return lDepth + 1
#                 else:
#                     return rDepth + 1
#             else:
#                 return 0
#
#     # def evaluate(self, indicator, threshold):
#     #     if indicator < threshold:
#     #         return True
#     #     else:
#     #         return False
#
#     def add_node(self, parent_node):
#         # Randomly choose indicator and threshold pair
#         indicator_threshold = self.pick_random_indicator_threshold()
#         # Randomly choose if you add a node on the left or the right
#         random_bit = random.getrandbits(1)
#         if random_bit == 0:
#             parent_node.left = Node(indicator_threshold[0], indicator_threshold[1])
#         elif random_bit == 1:
#             parent_node.right = Node(indicator_threshold[0], indicator_threshold[1])
#
#     def add_action_node(self, parent_node):
#         # Randomly choose action
#         action = random.choice(self.action_list)
#         # Randomly choose if you add a node on the left or the right
#         random_bit = random.getrandbits(1)
#         if random_bit == 0:
#             parent_node.left = ActionNode(action)
#         elif random_bit == 1:
#             parent_node.right = ActionNode(action)
#
#     def pick_random_indicator_threshold(self):
#         indicator = random.choice(list(self.indicator_dict.keys()))
#         threshold = random.randint(self.indicator_dict[indicator][0], self.indicator_dict[indicator][1])
#         return indicator, threshold
#
#     def print_tree(self, traversal_type):
#         if traversal_type == "preorder":
#             return self.preorder_print(self.root, "")
#         else:
#             print("Traversal type " + traversal_type + " not supported.")
#             return False
#
#     def preorder_print(self, start, traversal):
#         '''Root -> Left -> Right'''
#         if start:
#             traversal += (str(start.indicator) + "<" + str(start.threshold) + "-")
#             traversal = self.preorder_print(start.left, traversal)
#             traversal = self.preorder_print(start.right, traversal)
#         return traversal
#
#     # def add_edges(self, graph, node):
#     #     if node is not None:
#     #         if node.left is not None:
#     #             graph.add_edge(node.indicator, node.left.indicator)
#     #             self.add_edges(graph, node.left)
#     #         if node.right is not None:
#     #             graph.add_edge(node.indicator, node.right.indicator)
#     #             self.add_edges(graph, node.right)
#     #
#     # def plot_tree(self, root):
#     #     graph = nx.DiGraph()
#     #     self.add_edges(graph, root)
#     #     pos = graphviz_layout(graph, prog='dot')
#     #     nx.draw(graph, pos, with_labels=True, arrows=False)
#     #     plt.show()
#
#
# indicator_dict = {'indicator_1': [0, 5],
#          'indicator_2': [6,10]}
# # indicator_dict = {2: [0, 5],
# #          4: [6,10]}
# action_list = ['a1', 'a2', 'a3']
#
# # print(list(indicator_dict.keys()))
# # print(random.choice(list(indicator_dict.keys())))
# # print(random.randint(indicator_dict['indicator_1'][0], indicator_dict['indicator_1'][1]))
#
# tree = BinaryTree(indicator_dict=indicator_dict, action_list=action_list, max_depth=3)
# # print(tree.root.indicator)
# # print(tree.root.threshold)
# print(tree.print_tree("preorder"))
#
#
# # print(random.getrandbits(1))


import random

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


class Node(object):
    def __init__(self, indicator, threshold):
        self.value = (indicator, threshold)
        self.left = None
        self.right = None


class ActionNode(object):
    def __init__(self, action):
        self.value = action
        self.left = None
        self.right = None



class BinaryTree(object):
    def __init__(self, indicator_dict, action_list, max_depth):
        self.indicator_dict = indicator_dict
        self.action_list = action_list
        indicator_threshold = self.pick_random_indicator_threshold()
        self.root = Node(indicator_threshold[0], indicator_threshold[1])
        # self.root.left = Node(2,3)
        # self.root.left.left = Node(4,2)
        # self.root.left.right = Node(7,9)
        # self.root.right = Node(1,3)

        # self.try_action_node = ActionNode('a2')
        # print(type(self.try_action_node))
        # print(f"is action node:  {isinstance(ActionNode, type(self.try_action_node))} ")
        # print(f"is action node:  {isinstance(self.try_action_node, ActionNode)} ")

        self.leaf_nodes = []
        self.find_leaf_nodes(self.root, '')
        # print(self.leaf_nodes)

        # self.add_node(self.leaf_nodes[0])

        depth = 0
        while depth <= max_depth:
            self.find_leaf_nodes(self.root, '')
            random_leaf_node = random.choice(self.leaf_nodes)
            # randomly assign or another indicator node or an action node
            random_bit = random.getrandbits(1)
            if random_bit == 0:
                self.add_action_node(random_leaf_node)
            elif random_bit == 1:
                self.add_node(random_leaf_node)
                depth = self.maxDepth(self.root)
        # Assign an action node to every leaf node that is not an action already
        leaf_nodes = self.find_leaf_nodes(self.root, '')
        for leaf_node in leaf_nodes:
            self.add_action_node(leaf_node)

            # print(depth)



        # iterator = self.root
        # while (not iterator.left) & (not iterator.right):
        #     if iterator.left:
        #         iterator = iterator.left

        # # Randomly create nodes
        # i = 0
        # node = self.root
        # self.add_node(parent_node=node)
        # while i < max_depth:
        #     print(node.indicator)
        #     if node.left:
        #         print('made left node')
        #         node = node.left
        #     else:
        #         print('made right node')
        #         node = node.right
        #     self.add_node(parent_node=node)
        #     i += 1

    def find_leaf_nodes(self, start, iterator):
        # if isinstance(ActionNode, type(start)):
        #     return iterator
        # elif start:
        #     if (not start.left) & (not start.right):
        #         # self.leaf_nodes.append(str(start.indicator)+"/"+str(start.threshold))
        #         self.leaf_nodes.append(start)
        #     iterator = self.find_leaf_nodes(start.left, iterator)
        #     iterator = self.find_leaf_nodes(start.right, iterator)
        #     return iterator
        if not isinstance(start, ActionNode):
            if start:
                if (not start.left) & (not start.right):
                    # self.leaf_nodes.append(str(start.indicator)+"/"+str(start.threshold))
                    self.leaf_nodes.append(start)
                iterator = self.find_leaf_nodes(start.left, iterator)
                iterator = self.find_leaf_nodes(start.right, iterator)
        return iterator

    def maxDepth(self, node):
        if node is None:
            return 0
        else:
            if not isinstance(node, ActionNode):
                # Compute the depth of each subtree
                lDepth = self.maxDepth(node.left)
                rDepth = self.maxDepth(node.right)
                # Use the larger one
                if (lDepth > rDepth):
                    return lDepth + 1
                else:
                    return rDepth + 1
            else:
                return 0

    # def evaluate(self, indicator, threshold):
    #     if indicator < threshold:
    #         return True
    #     else:
    #         return False

    def add_node(self, parent_node):
        # Randomly choose indicator and threshold pair
        indicator_threshold = self.pick_random_indicator_threshold()
        # Randomly choose if you add a node on the left or the right
        random_bit = random.getrandbits(1)
        if random_bit == 0:
            parent_node.left = Node(indicator_threshold[0], indicator_threshold[1])
        elif random_bit == 1:
            parent_node.right = Node(indicator_threshold[0], indicator_threshold[1])

    def add_action_node(self, parent_node):
        # Randomly choose action
        action = random.choice(self.action_list)
        # Randomly choose if you add a node on the left or the right
        random_bit = random.getrandbits(1)
        if random_bit == 0:
            parent_node.left = ActionNode(action)
        elif random_bit == 1:
            parent_node.right = ActionNode(action)

    def pick_random_indicator_threshold(self):
        indicator = random.choice(list(self.indicator_dict.keys()))
        threshold = random.randint(self.indicator_dict[indicator][0], self.indicator_dict[indicator][1])
        return indicator, threshold

    def print_tree(self, traversal_type):
        if traversal_type == "preorder":
            return self.preorder_print(self.root, "")
        else:
            print("Traversal type " + traversal_type + " not supported.")
            return False

    def preorder_print(self, start, traversal):
        '''Root -> Left -> Right'''
        if start:
            traversal += (str(start.value) + "-")
            traversal = self.preorder_print(start.left, traversal)
            traversal = self.preorder_print(start.right, traversal)
        return traversal

    # def add_edges(self, graph, node):
    #     if node is not None:
    #         if node.left is not None:
    #             graph.add_edge(node.indicator, node.left.indicator)
    #             self.add_edges(graph, node.left)
    #         if node.right is not None:
    #             graph.add_edge(node.indicator, node.right.indicator)
    #             self.add_edges(graph, node.right)
    #
    # def plot_tree(self, root):
    #     graph = nx.DiGraph()
    #     self.add_edges(graph, root)
    #     pos = graphviz_layout(graph, prog='dot')
    #     nx.draw(graph, pos, with_labels=True, arrows=False)
    #     plt.show()


indicator_dict = {'indicator_1': [0, 5],
         'indicator_2': [6,10]}
# indicator_dict = {2: [0, 5],
#          4: [6,10]}
action_list = ['a1', 'a2', 'a3']

# print(list(indicator_dict.keys()))
# print(random.choice(list(indicator_dict.keys())))
# print(random.randint(indicator_dict['indicator_1'][0], indicator_dict['indicator_1'][1]))

tree = BinaryTree(indicator_dict=indicator_dict, action_list=action_list, max_depth=7)
# print(tree.root.indicator)
# print(tree.root.threshold)
print(tree.print_tree("preorder"))


# print(random.getrandbits(1))