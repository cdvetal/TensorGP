#MIT License

#Copyright (c) 2020-2021 Francisco Baeta (University of Coimbra)
#Copyright (c) 2020-2021 João Correia (University of Coimbra)
#Copyright (c) 2020-2021 Tiago Martins (University of Coimbra)
#Copyright (c) 2020-2021 Penousal Machado (University of Coimbra)

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


import csv
import sys
from collections import Counter
import copy
import datetime
import math
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pylab
import time
import random
from heapq import nsmallest, nlargest
from operator import itemgetter
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

delimiter = os.path.sep
subdir = "runs"

# evil global variables
_min_domain = -1
_max_domain = 1
_domain_delta = _max_domain - _min_domain

# np debug options
#np.set_printoptions(linewidth = 1000)
#np.set_printoptions(precision = 3)

#taken from https://svn.blender.org/svnroot/bf-blender/trunk/blender/build_files/scons/tools/bcolors.py

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

## ====================== tensorflow operators ====================== ##
def resolve_var_node(dimensions, n):
    temp = np.ones(len(dimensions), dtype=int)
    temp[n] = -1
    res = tf.reshape(tf.range(dimensions[n], dtype=tf.float32), temp)
    _resolution = dimensions[n]
    dimensions[n] = 1
    res = tf.scalar_mul(((1.0 / (_resolution - 1)) * _domain_delta), res)
    res = tf.math.add(res, tf.constant(_min_domain, tf.float32, res.shape))
    res = tf.tile(res, dimensions)
    return res

def resolve_abs_node(child1, dims=[]):
    return tf.math.abs(child1)

def resolve_add_node(child1, child2, dims=[]):
    return tf.math.add(child1, child2)

def resolve_and_node(child1, child2, dims=[]):
    left_child_tensor = tf.cast(tf.scalar_mul(1e6, child1), tf.int32)
    right_child_tensor = tf.cast(tf.scalar_mul(1e6, child2), tf.int32)
    return tf.scalar_mul(1e-6, tf.cast(tf.bitwise.bitwise_and(left_child_tensor, right_child_tensor), tf.float32))

def resolve_xor_node(child1, child2, dims=[]):
    left_child_tensor = tf.cast(tf.scalar_mul(1e6, child1), tf.int32)
    right_child_tensor = tf.cast(tf.scalar_mul(1e6, child2), tf.int32)
    return tf.scalar_mul(1e-6, tf.cast(tf.bitwise.bitwise_xor(left_child_tensor, right_child_tensor), tf.float32))

def resolve_or_node(child1, child2, dims=[]):
    left_child_tensor = tf.cast(tf.scalar_mul(1e6, child1), tf.int32)
    right_child_tensor = tf.cast(tf.scalar_mul(1e6, child2), tf.int32)
    return tf.scalar_mul(1e-6, tf.cast(tf.bitwise.bitwise_or(left_child_tensor, right_child_tensor), tf.float32))

def resolve_cos_node(child1, dims=[]):
    return tf.math.cos(tf.scalar_mul(math.pi, child1))

def resolve_div_node(child1, child2, dims=[]):
    left_child_tensor = tf.cast(child1, tf.float32)
    right_child_tensor = tf.cast(child2, tf.float32)
    return tf.math.divide_no_nan(left_child_tensor, right_child_tensor)

def resolve_exp_node(child1, dims=[]):
    return tf.math.exp(child1)

def resolve_if_node(child1, child2, child3, dims=[]):
    return tf.where(child3 < 0, child1, child2)

def resolve_log_node(child1, dims=[]):
    res = tf.where(child1 > 0, tf.math.log(child1), tf.constant(-1, tf.float32, dims))
    return res

def resolve_max_node(child1, child2, dims=[]):
    return tf.math.maximum(child1, child2)

def resolve_mdist_node(child1, child2, dims=[]):
    return tf.scalar_mul(0.5, tf.add(child1, child2))

def resolve_min_node(child1, child2, dims=[]):
    return tf.math.minimum(child1, child2)

def resolve_mod_node(child1, child2, dims=[]):
    return tf.math.mod(child1, child2)

def resolve_mult_node(child1, child2, dims=[]):
    return tf.math.multiply(child1, child2)

def resolve_neg_node(child1, dims=[]):
    return tf.math.negative(child1)


def resolve_pow_node(child1, child2, dims=[]):
    return tf.where(child1 == 0, tf.constant(0, tf.float32, dims), tf.math.pow(tf.math.abs(child1), tf.math.abs(child2)));

def resolve_sign_node(child1, dims=[]):
    #return tf.math.divide_no_nan(tf.math.abs(child1),child1)
    return tf.math.sign(child1)

def resolve_sin_node(child1, dims=[]):
    return tf.math.sin(tf.scalar_mul(math.pi, child1))

def resolve_sqrt_node(child1, dims=[]):
    return tf.where(child1 > 0, tf.math.sqrt(child1), tf.constant(0, tf.float32, dims))

def resolve_sub_node(child1, child2, dims=[]):
    return tf.math.subtract(child1, child2)

def resolve_tan_node(child1, dims=[]):
    return tf.where(child1 != (math.pi * 0.5), tf.math.tan(tf.scalar_mul(math.pi, child1)), tf.constant(0, tf.float32, dims))

def resolve_warp_node(tensors, image, dimensions):
    n = len(dimensions)
    #print("[DEBUG]:\tWarp dimensions: " + str(dimensions))
    #print("[DEBUG]:\tWarp log(y): ")
    #print(tensors[1].numpy())

    tensors = [tf.where(tf.math.is_nan(t), tf.zeros_like(t), t) for t in tensors]


    indices = tf.stack([
        tf.clip_by_value(
            tf.round(tf.multiply(
                tf.constant((dimensions[k] - 1) * 0.5, tf.float32, shape=dimensions),
                tf.math.add(tensors[k], tf.constant(1.0, tf.float32, shape=dimensions))
            )),
            clip_value_min=0.0,
            clip_value_max=(dimensions[k] - 1)
        ) for k in range(n)],
        axis=n
    )

    indices = tf.cast(indices, tf.int32)
    #print("[DEBUG]:\tWarp Indices: ")
    #print(indices.numpy())
    return tf.gather_nd(image, indices)

# return x * x * x * (x * (x * 6 - 15) + 10);
# simplified to x2*(6x3 - (15x2 - 10x)) to minimize TF operations
def resolve_sstepp_node(child1, dims=[]):
    x = resolve_clamp(child1)
    x2 = tf.square(x)
    return tf.add(tf.multiply(x2,
                              tf.subtract(tf.scalar_mul(6.0 * _domain_delta, tf.multiply(x, x2)),
                                          tf.subtract(tf.scalar_mul(15.0 * _domain_delta, x2),
                                                      tf.scalar_mul(10.0 * _domain_delta, x)))),
                  tf.constant(_min_domain, dtype=tf.float32, shape=dims))

# return x * x * (3 - 2 * x);
# simplified to (6x2 - 4x3) to minimize TF operations
def resolve_sstep_node(child1, dims=[]):
    x = resolve_clamp(child1)
    x2 = tf.square(x)
    return tf.add(tf.subtract(tf.scalar_mul(3.0 * _domain_delta, x2),
                              tf.scalar_mul(2.0 * _domain_delta, tf.multiply(x2, x))),
                  tf.constant(_min_domain, dtype=tf.float32, shape=dims))

def resolve_step_node(child1, dims=[]):
    return tf.where(child1 < 0.0,
                    tf.constant(-1.0, dtype=tf.float32, shape=dims),
                    tf.constant(1.0, dtype=tf.float32, shape=dims))

def resolve_clamp(tensor):
    return tf.clip_by_value(tensor, clip_value_min=_min_domain, clip_value_max=_max_domain)
    #return tf.cast(tf.cast(tensor, tf.uint8), tf.float32)

def resolve_frac_node(child1, dims=[]):
    return tf.subtract(child1, tf.floor(child1))

def resolve_clip_node(child1, child2, child3, dims=[]): # a < n < b
    return tf.minimum(tf.maximum(child1, child2), child3)

def resolve_len_node(child1, child2, dims=[]):
    return tf.sqrt(tf.add(tf.square(child1), tf.square(child2)))

def resolve_lerp_node(child1, child2, child3, dims=[]):
    child3 = resolve_frac_node(child3, dims)
    t_dist = tf.subtract(child2, child1)
    t_dist = tf.multiply(child3, t_dist)
    return tf.math.add(child1, t_dist)

def old_tf_rmse(child1, child2):
    child1 = tf.cast(child1, tf.float32)
    child2 = tf.cast(child2, tf.float32)
    elements = np.prod(child1.shape.as_list())
    sdiff = tf.math.squared_difference(child1, child2)
    mse = tf.math.reduce_sum(sdiff).numpy() / elements
    mse = math.sqrt(mse)
    return mse

@tf.function
def tf_rmse(child1, child2):
    child1 = tf.scalar_mul(1/127.5, child1)
    child2 = tf.scalar_mul(1/127.5, child2)
    return tf.sqrt(tf.reduce_mean(tf.square(child1 - child2)))

## ====================== tensorflow node class ====================== ##

class Node:
    def __init__(self, value, children, terminal):
        self.value = value
        self.children = children
        self.terminal = terminal

    def node_tensor(self, tens, engref):

        arity = engref.function.set[self.value][0]
        if self.value != 'warp':
            arg_list = tens[:arity]
        else:
            arg_list = [tens[1:] + list(engref.terminal.latentset.values()), tens[0]]

        arg_list += [engref.target_dims]

        return engref.function.set[self.value][1](*arg_list)


    def get_tensor(self, engref):

        if self.terminal:
            if self.value == 'scalar':
                args = len(self.children)


                if args == 1:
                    return tf.constant(self.children[0], tf.float32, engref.target_dims)
                else:
                    #print("we should not be here!")
                    #last_dim = engref.target_dims[-1]
                    last_dim = engref.terminal.dimension
                    extend_children = self.children + ([self.children[-1]] * (last_dim - args))
                    return tf.stack(
                        [tf.constant(float(c), tf.float32, engref.target_dims[:engref.effective_dims]) for c in extend_children[:last_dim]],
                        axis = engref.effective_dims
                    )
                # both cases do the same if args == 1, this condition is here for speed concerns if
                # the last dimension is very big (e.g. 2d(1024, 1024) we don't want color in that case)

            else:
                return engref.terminal.set[self.value]
        else:
            tens_list = []
            for c in self.children:
                tens_list.append(c.get_tensor(engref))

            return self.node_tensor(tens_list, engref)

    def get_str(self):
        if self.terminal and self.value != 'scalar':
            return str(self.value)
        else:
            string_to_use = self.value
            strings_to_differ = ['and', 'or', 'if']
            if self.value in strings_to_differ:
                string_to_use = '_' + self.value

            string_to_use += '('
            c = 0
            size = len(self.children)
            while True:
                if self.terminal:
                    string_to_use += str(self.children[c])
                else:
                    string_to_use += self.children[c].get_str()
                c += 1
                if c > size - 1: break
                string_to_use += ', '

            return string_to_use + ')'

    def get_depth(self, depth=0):
        n_childs = 1
        if self.terminal:
            return depth, 1
        else:
            max_d = 0
            for i in self.children:
                child_depth, tmp = i.get_depth(depth + 1)
                n_childs += tmp
                if max_d < child_depth:
                    max_d = child_depth
            return max_d, n_childs

def constrain(n, a, b):
    return min(max(n, a), b)

def save_image(tensor, index, fn, dims, addon=''):
    path = fn + addon + "_indiv" + str(index).zfill(5) + ".png"
    # force output to 0...255
    final_tensor = tf.math.subtract(tensor, tf.constant(_min_domain, tf.float32, dims))
    final_tensor = tf.scalar_mul(255 / _domain_delta, final_tensor)

    aux = np.array(tensor, dtype='uint8')
    try:
        if len(dims) == 2:
            Image.fromarray(aux, mode="L").save(path) # no color
        elif len(dims) == 3:
            Image.fromarray(aux, mode="RGB").save(path) # color
        else:
            print("Attempting to save tensor with rank ", len(dims), " as an image, must be rank 2 (grayscale) or 3 (RGB).")
    except ValueError:
        print()
    return path

# just a wrapper for different expression types and strip preprocessing
def str_to_tree(stree, terminal_set, constrain_domain=True):
    return str_to_tree_normal(stree.replace(" ", "") + "", terminal_set, 0, constrain_domain)


def str_to_tree_normal(stree, terminal_set, number_nodes=0, constrain_domain = True):
    if stree in terminal_set:
        return number_nodes, Node(value=stree, terminal=True, children=[])
    elif stree[:6] == 'scalar':
        numbers = [(constrain(float(x), _min_domain, _max_domain) if constrain_domain else float(x)) for x in re.split('\(|\)|,', stree)[1:-1]]

        return number_nodes, Node(value='scalar', terminal=True, children=numbers)
    else:
        x = stree[:-1].split("(", 1)
        #print("x:", x)
        primitive = x[0]
        #print("x0:", x[0])
        if x[0][0] == '_':
            primitive = x[0][1::]
        args = x[1]
        pc = 0
        last_pos = 0
        children = []

        for i in range(len(args)):
            c = args[i]
            if c == '(':
                pc += 1
            elif c == ')':
                pc -= 1
            elif c == ',' and pc == 0:
                number_nodes, tree = str_to_tree_normal(args[last_pos:i], terminal_set, number_nodes, constrain_domain)
                children.append(tree)
                last_pos = i + 1

        number_nodes, tree = str_to_tree_normal(args[last_pos:], terminal_set, number_nodes, constrain_domain)
        children.append(tree)
        if primitive == "if":
            children = [children[1], children[2], children[0]]
        return number_nodes + 1, Node(value=primitive, terminal=False, children=children)


def set_device(device = '/gpu:0', debug_lvl = 1):
    cuda_build = tf.test.is_built_with_cuda()
    gpus_available = len(tf.config.list_physical_devices('GPU'))

    # because these won't give an error but will be unspecified
    if (device is None) or (device == '') or (device == ':'):
        device = '0'
    try:
        result_device = device

        # just to verify errors
        with(tf.device(device)):
            a = tf.constant(2, dtype=tf.float32)
            if debug_lvl > 0:
                if a == 2:
                    print(bcolors.OKGREEN + "Device " + device + " successfully tested, using this device. " , bcolors.ENDC)
                else:
                    print(bcolors.FAIL + "Device " + device + " not working." , bcolors.ENDC)
    except RuntimeError or ValueError:
        if cuda_build and gpus_available > 0:
            result_device = '/gpu:0'
            print(bcolors.WARNING + "[WARNING]:\tCould not find the specified device, reverting to GPU." , bcolors.ENDC)
        else:
            result_device = '/cpu:0'
            print(bcolors.WARNING + "[WARNING]:\tCould not find the specified device, reverting to CPU." , bcolors.ENDC)
    return result_device


# TODO: should this be an inner class of Engine()?
class Experiment:

    def get_experiment_filename(self):
        date = datetime.datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S_%f')[:-3]
        return "run__" + date + "__" + str(self.ID)

    def set_generation_directory(self, generation):
        try:
            self.current_directory = self.working_directory + "generation_" + str(generation).zfill(5) + delimiter
            os.makedirs(self.current_directory)
            #print("[DEBUG]:\tSet current directory to: " + self.current_directory)
        except FileExistsError:
            self.current_directory = self.working_directory
            print("Could not create directory for generation" + str(generation) + " as it already exists")

    def set_experiment_ID(self):
        return int(time.time() * 1000.0) << 16

    def __init__(self, previous_state, seed = None, wd = delimiter + subdir + delimiter):

        self.ID = self.set_experiment_ID() if (previous_state is None) else previous_state['ID']
        self.seed = self.ID if (seed is None) else seed
        self.filename = self.get_experiment_filename() + "__images"

        try:
            self.working_directory = os.getcwd() + wd + self.filename + delimiter
            os.makedirs(self.working_directory)
        except FileExistsError:
            print(bcolors.WARNING + "[WARNING]:\tExperiment directory already exists, saving files to current directory." , bcolors.ENDC)
            print(bcolors.WARNING + "[WARNING]:\tFilename: " + self.working_directory , bcolors.ENDC)
            self.working_directory = ''

        self.current_directory = self.working_directory
        self.generations_directory = self.working_directory + "generations" + delimiter
        self.immigration_directory = self.generations_directory + "immigration" + delimiter
        self.logging_diredctory = self.working_directory + "logs" + delimiter
        self.best_directory = self.working_directory + "best" + delimiter
        try:
            os.makedirs(self.generations_directory)
            os.makedirs(self.immigration_directory)
            os.makedirs(self.logging_diredctory)
            os.makedirs(self.best_directory)
        except FileExistsError:
            print(bcolors.WARNING + "[WARNING]:\tTried to create a directory that already exists" , bcolors.ENDC)

class Engine:

    ## ====================== genetic operators ====================== ##

    def crossover(self, parent_1, parent_2):
        crossover_node = None

        if self.engine_rng.random() < self.koza_rule_prob and not parent_1.terminal:  # TODO: review, this is Koza's rule
            # function crossover
            parent_1_candidates = self.list_nodes(parent_1, True, add_funcs=True, add_terms=False, add_root=True)
            parent_1_chosen_node, _ = self.engine_rng.choice(parent_1_candidates)
            possible_children = []
            for i in range(len(parent_1_chosen_node.children)):
                if not parent_1_chosen_node.children[i].terminal:
                    possible_children.append(i)
            if possible_children != []:
                crossover_node = copy.deepcopy(parent_1_chosen_node.children[self.engine_rng.choice(possible_children)])
            else:
                crossover_node = copy.deepcopy(parent_1_chosen_node)
        else:
            parent_1_terminals = self.get_terminals(parent_1)
            #print("size of crossover node list: ", len(list(parent_1_terminals.elements())))
            crossover_node = copy.deepcopy(self.engine_rng.choice(list(parent_1_terminals.elements())))

        #print("Crossover node in parent 1: ", crossover_node.get_str())

        if crossover_node is None:
            print("[ERROR]: Did not select a crossover node.")
        new_individual = copy.deepcopy(parent_2)
        parent_2_candidates = self.list_nodes(new_individual, root=True, add_funcs=True, add_terms=False, add_root=True)

        if len(parent_2_candidates) > 0:
            parent_2_chosen_node, _ = self.engine_rng.choice(parent_2_candidates)
            # second part of if, because in future there can be funcs with no arguments
            if not parent_2_chosen_node.terminal and len(parent_2_chosen_node.children) > 0:
                rand_child = self.engine_rng.randint(0, len(parent_2_chosen_node.children) - 1)
                parent_2_chosen_node.children[rand_child] = crossover_node
            else:
                new_individual = crossover_node
        else:
            new_individual = crossover_node

        #print("Indiv cross: ", new_individual.get_str())

        return new_individual

    def get_candidates(self, node, root):
        candidates = Counter()
        if not node.terminal:
            for i in node.children:
                if (i is not None) and (not i.terminal):
                    candidates.update([node])
                    candidates.update(self.get_candidates(i, False))
        if root and candidates == Counter():
            candidates.update([node])
        return candidates

    def tournament_selection(self):
        if self.objective == 'minimizing':
            st = float('inf')
        else:
            st = -float('inf')

        winner = {'fitness': st}
        while winner['fitness'] == st:
            tournament_population = self.engine_rng.sample(self.population, self.tournament_size)
            for i in tournament_population:
                if ((self.objective == 'minimizing' and (i['fitness'] < winner['fitness'])) or
                        (self.objective != 'minimizing' and (i['fitness'] > winner['fitness']))):
                    winner = i
        return winner

    def mutation(self, parent):
        random_n = self.engine_rng.random()
        for k in range(len(self.mutation_funcs) - 1, -1, -1):
            if random_n > self.mutation_probs[k]:
                if callable(getattr(self, self.mutation_funcs[k].__name__, None)):
                    return self.mutation_funcs[k](self, parent)
                else:
                    return self.mutation_funcs[k](parent)


    def old_random_terminal(self):
        _l = []
        if self.engine_rng.random() < self.scalar_prob:
            _v = 'scalar'

            #color
            if self.engine_rng.random() < self.uniform_scalar_prob:
                _l = [self.engine_rng.uniform(self.erc_min, self.erc_max)] * self.terminal.dimension
            else:
                _l = [self.engine_rng.uniform(self.erc_min, self.erc_max) for i in range(self.terminal.dimension)]

        else:
            _v = self.engine_rng.sample(list(self.terminal.set), 1)[0]
        return Node(terminal = True, children = _l, value = _v)

    def random_terminal(self):
        _, node = self.generate_program(method='full', max_nodes = 0, max_depth = 0)
        return node

    def copy_node(self, n):
        return Node(value=n.value, terminal=n.terminal, children=n.children)

    def delete_mutation(self, parent):
        new_individual = copy.deepcopy(parent)

        # every node except last depth (terminals)
        candidates = self.list_nodes(new_individual, True, True, False, False)

        if len(candidates) > 0:
            chosen_node, _ = self.engine_rng.choice(candidates) # parent = root

            # random child of chosen
            chosen_child = chosen_node.children[self.engine_rng.randint(0, len(chosen_node.children) - 1)]
            chosen_node.value =  chosen_child.value
            chosen_node.children = chosen_child.children
            chosen_node.terminal = chosen_child.terminal

        return new_individual


    def insert_mutation(self, parent):
        new_individual = copy.deepcopy(parent)
        #print("[DEBUG D] Before:\t" + new_individual.get_str())

        # every node except last depth (terminals)
        candidates = self.list_nodes(new_individual, root=True, add_funcs=True, add_terms=False, add_root=True)

        if len(candidates) > 1 or (len(candidates) == 1 and not candidates[0][0].terminal):
            chosen_node, _ = self.engine_rng.choice(candidates)
            #print(new_individual.get_str())

            #insert node between choosen and choosen's child
            #random child of chosen
            # second part of if, because in future there can be funcs with no arguments
            if not chosen_node.terminal and len(chosen_node.children) > 0:
                chosen_child = chosen_node.children[self.engine_rng.randint(0, len(chosen_node.children) - 1)]
            else:
                chosen_child = new_individual
        else:
            chosen_child = new_individual

        _v = chosen_child.value
        _c = chosen_child.children
        _t = chosen_child.terminal
        child_temp = Node(value=_v, children=_c, terminal=_t)

        chosen_child.value = self.engine_rng.choice(list(self.function.set))
        chosen_child.terminal = False
        chosen_child.children = []

        nchildren = self.function.set[chosen_child.value][0]
        new_child_position = self.engine_rng.randint(0, nchildren - 1)

        for i in range(nchildren):
            if i == new_child_position:
                chosen_child.children.append(child_temp)
            else:
                chosen_child.children.append(self.random_terminal())
        return new_individual

    def list_nodes(self, node, dep=0, root=False, add_funcs=True, add_terms=True, add_root=False):
        res = []
        if (node.terminal and (add_terms or (root and add_root))) or ((not node.terminal) and add_funcs and ((not root) or add_root)):
            res.append((node, dep))
        if not node.terminal:
            for c in node.children:
                res += self.list_nodes(c, dep + 1, False, add_funcs, add_terms, add_root)
        return res

    # this is the same , except it does not go back to the loop
    def hacky_subtree_mutation(self, parent):
        new_individual = copy.deepcopy(parent)
        #print("\nindiv, ", new_individual.get_str())
        candidates = self.list_nodes(new_individual, root=True, add_funcs = True, add_terms=False, add_root=True)

        chosen_node, chosen_dep = self.engine_rng.choice(candidates)
        _max_dep = self.max_tree_depth - chosen_dep
        _max_dep = min(_max_dep, self.max_subtree_dep)
        _min_dep = min(_max_dep, self.min_subtree_dep)

        _, mutation_node = self.generate_program('grow', -1, max_depth=_max_dep, min_depth=_min_dep, root=True)

        if not chosen_node.terminal:
            chosen_node.children[self.engine_rng.randint(0, len(chosen_node.children) - 1)] = mutation_node
        else: # means its root in this case
            new_individual = mutation_node
        return new_individual

    def subtree_mutation(self, parent):
        new_individual = copy.deepcopy(parent)
        # candidates = self.get_candidates(new_individual, True)
        # print("\nindiv, ", new_individual.get_str())
        candidates = self.list_nodes(new_individual, root=True, add_funcs=True, add_terms=False, add_root=True)

        chosen_node, _ = self.engine_rng.choice(candidates)
        _, mutation_node = self.generate_program('grow', -1, max_depth = self.max_subtree_dep, min_depth = self.min_subtree_dep, root=True)

        if not chosen_node.terminal:
            chosen_node.children[self.engine_rng.randint(0, len(chosen_node.children) - 1)] = mutation_node
        else:  # means its root in this case
            new_individual = mutation_node
        return new_individual

    def replace_nodes(self, node):
        if node.terminal:
            if self.engine_rng.random() < self.scalar_prob:
                node.value = 'scalar'
                # node.children = [self.engine_rng.uniform(0, 1)] # no color
                # color
                if self.engine_rng.random() < self.uniform_scalar_prob:
                    node.children = [self.engine_rng.uniform(self.erc_min, self.erc_max)] * self.terminal.dimension
                else:
                    node.children = [self.engine_rng.uniform(self.erc_min, self.erc_max) for i in range(self.terminal.dimension)]
            else:
                if node.value == 'scalar':
                    node.value = self.engine_rng.choice(list(self.terminal.set))
                else:
                    temp_tset = self.terminal.set.copy()
                    del temp_tset[node.value]
                    node.value = self.engine_rng.choice(list(temp_tset))
        else:

            this_arity = self.function.set[node.value][0]
            arity_to_search = random.choice(list(self.function.arity.keys())) if self.replace_mode is 'dynamic_arities' else this_arity

            set_of_same_arities = self.function.arity[arity_to_search][:]
            if this_arity == arity_to_search:
                set_of_same_arities.remove(node.value)

            if len(set_of_same_arities) > 0:
                node.value = self.engine_rng.choice(set_of_same_arities)
                if this_arity < arity_to_search:
                    for i in range(this_arity, arity_to_search):
                        node.children.append(self.random_terminal())
                elif this_arity > arity_to_search:
                    node.children = node.children[:arity_to_search]

        if not node.terminal and self.replace_prob > 0:
            for i in node.children:
                if self.engine_rng.random() < self.replace_prob:
                    self.replace_nodes(i)


    def point_mutation(self, parent):
        new_individual = copy.deepcopy(parent)
        candidates = self.list_nodes(new_individual, True, True, True, True)
        chosen_node, _ = self.engine_rng.choice(candidates)
        self.replace_nodes(chosen_node)
        return new_individual

    def get_terminals(self, node):
        candidates = Counter()
        if node.terminal:
            candidates.update([node])
        else:
            for i in node.children:
                if i is not None:
                    candidates.update(self.get_terminals(i))
        return candidates

    ## ====================== init class ====================== ##



    def __init__(self,
                 fitness_func = None,
                 population_size = 100,
                 tournament_size = 3,
                 mutation_rate = 0.15,
                 mutation_funcs = None,
                 mutation_probs = None,
                 crossover_rate = 0.9,
                 elitism = 1,
                 min_tree_depth = -1,
                 max_tree_depth = 8,
                 max_init_depth = None,
                 min_init_depth = None,
                 max_subtree_dep = None,
                 min_subtree_dep = None,
                 method = 'ramped half-and-half',
                 terminal_prob = 0.2,
                 scalar_prob = 0.55,
                 uniform_scalar_prob = 0.5,
                 koza_rule_prob = 0.9,
                 stop_criteria = 'generation',
                 stop_value = 10,
                 objective = 'minimizing', # TODO: this feature will be removed soon...on second thought maybe not
                 min_domain = -1,
                 max_domain = 1,
                 bloat_control = 'full_dynamic_dep',
                 domain_mode = 'clip',
                 replace_mode = 'dynamic_arities',
                 replace_prob = 0.05,
                 const_range = None,
                 effective_dims = None,
                 operators = None,
                 function_set = None,
                 terminal_set = None,
                 immigration = float('inf'),
                 target_dims = None,
                 target = None,
                 max_nodes = -1,
                 seed = None,
                 debug = 0,
                 save_graphics = True,
                 show_graphics = True,
                 save_best = True,
                 device = '/cpu:0',
                 save_to_file = 10,
                 write_log = True,
                 write_gen_stats = True,
                 write_final_pop = False,
                 path_to_final_pop = None,
                 initial_test_device = True,
                 previous_state = None,
                 var_func = None,
                 stats_file_path = None,
                 pop_file_path = None,
                 read_init_pop_from_file = None):

        # start timers
        self.last_engine_time = time.time()
        start_init = self.last_engine_time
        self.elapsed_init_time = 0
        self.elapsed_fitness_time = 0
        self.elapsed_tensor_time = 0
        self.elapsed_engine_time = 0

        # check for fitness func
        self.fitness_func = fitness_func

        # optional vars
        self.recent_fitness_time = 0
        self.recent_tensor_time = 0
        self.recent_engine_time = 0
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = constrain(0, elitism, self.population_size)
        self.koza_rule_prob = koza_rule_prob
        self.stop_criteria = stop_criteria
        self.save_graphics = save_graphics
        self.show_graphics = show_graphics
        if bloat_control not in ['full_dynamic_dep', 'dynamic_dep']: # add full_dynamic_size, dynamic_size
            bloat_control = 'off'
        self.bloat_control = bloat_control
        if domain_mode not in ['log', 'dynamic', 'mod']: # add full_dynamic_size, dynamic_size
            domain_mode = 'clip'
        self.domain_mode = domain_mode
        self.immigration = immigration
        self.debug = debug
        self.save_to_file = save_to_file
        self.max_nodes = max_nodes
        self.objective = objective
        self.terminal_prob = terminal_prob
        self.scalar_prob = scalar_prob
        self.uniform_scalar_prob = uniform_scalar_prob

        if self.bloat_control in ['full_dynamic_dep', 'dynamic_dep']:
            self.max_init_depth = 5 if (max_init_depth is None) else max_init_depth
            self.min_init_depth = 2 if (min_init_depth is None) else min_init_depth
            if self.max_init_depth < self.min_init_depth: self.max_init_depth, self.min_init_depth = self.min_init_depth, self.max_init_depth
            self.max_tree_depth = self.max_init_depth
            self.min_tree_depth = self.min_init_depth
        else:
            self.min_tree_depth = min_tree_depth
            self.max_tree_depth = max_tree_depth
            if self.max_tree_depth < self.min_tree_depth: self.max_tree_depth, self.min_tree_depth = self.min_tree_depth, self.max_tree_depth
            self.max_init_depth = self.max_tree_depth if (max_init_depth is None) else max_init_depth
            self.min_init_depth = self.min_tree_depth if (min_init_depth is None) else min_init_depth
        self.max_subtree_dep = self.max_tree_depth if (max_subtree_dep is None) else max_subtree_dep
        self.min_subtree_dep = self.min_tree_depth if (min_subtree_dep is None) else min_subtree_dep
        if self.max_subtree_dep < self.min_subtree_dep: self.max_subtree_dep, self.min_subtree_dep = self.min_subtree_dep, self.max_subtree_dep

        self.target_dims = [128, 128] if (target_dims is None) else target_dims
        self.dimensionality = len(self.target_dims)
        self.effective_dims = self.dimensionality if effective_dims is None else effective_dims
        self.device = set_device(device=device) if initial_test_device else device # Check for available devices
        self.previous_state = previous_state
        self.experiment = Experiment(self.previous_state, seed)
        self.engine_rng = random.Random(self.experiment.seed)
        self.method = method if (method in ['ramped half-and-half', 'grow', 'full']) else 'ramped half-and-half'
        self.replace_mode = replace_mode if replace_mode is 'dynamic_arities' else 'same_arity'
        self.replace_prob = max(0.0, min(1.0, replace_prob))
        self.pop_file = read_init_pop_from_file
        self.write_log = write_log
        self.write_gen_stats = write_gen_stats
        self.write_final_pop = write_final_pop
        self.save_best = save_best
        self.path_to_final_pop = self.experiment.current_directory if path_to_final_pop is None else path_to_final_pop
        #self.overall_stats_filename = 'overall_stats.csv'
        self.save_state = 0
        self.last_stop = 0
        self.overall_stats_filename = "tensorgp_" + str(self.target_dims[0]) + "_" + str(self.experiment.seed) + '.csv'
        self.stats_file_path = stats_file_path
        self.pop_file_path = pop_file_path


        if mutation_funcs is None or mutation_funcs == []:
            mut_funcs_implemented = 4
            self.mutation_funcs = [Engine.subtree_mutation, Engine.point_mutation, Engine.delete_mutation, Engine.insert_mutation]
            self.mutation_probs = np.linspace(0, 1, mut_funcs_implemented + 1)[:-1]
        else:
            self.mutation_funcs = []
            temp_probs = []
            for k in range(len(mutation_funcs)):
                if callable(mutation_funcs[k]) and k < len(mutation_probs) and mutation_probs[k] > 0:
                    self.mutation_funcs.append(mutation_funcs[k])
                    temp_probs.append(mutation_probs[k])

            self.mutation_probs = np.copy(temp_probs)
            self.mutation_probs[0] = 0
            for k in range(len(self.mutation_funcs) - 1):
                self.mutation_probs[k + 1] = min(self.mutation_probs[k] + mutation_probs[k], 1.0)

        # technically globals are not needed but it helps with external operators
        global _domain_delta, _min_domain, _max_domain
        if min_domain is not None:
            _min_domain = min_domain
        if max_domain is not None:
            _max_domain = max_domain
        _domain_delta = max_domain - min_domain

        if const_range is None or len(const_range) < 1:
            self.erc_min = _min_domain
            self.erc_max = _max_domain
        else:
            self.erc_min = max(const_range)
            self.erc_max = min(const_range)

        if isinstance(function_set, Function_Set):
            self.function = function_set
        else:
            self.function = Function_Set(operators, self.effective_dims, debug=self.debug)
        if isinstance(terminal_set, Terminal_Set):
            self.terminal = terminal_set
        else:
            #self.terminal = Terminal_Set(self.effective_dims, self.target_dims, engref=self)
            if var_func is None or not callable(var_func):
                self.var_func = resolve_var_node
            else:
                self.var_func = var_func
            self.terminal = Terminal_Set(self.effective_dims, self.target_dims, engref=self, function_ptr_to_var_node=self.var_func)


        #print("x tensor: ", self.terminal.set['x'])
        #print("y tensor: ", self.terminal.set['y'])


        if isinstance(target, str):
            target = 'mult(scalar(127.5), ' + target + ')'
            _, tree = str_to_tree(target, self.terminal.set, constrain_domain=False)

            with tf.device(self.device):
                #self.target = self.final_transform_domain(tree.get_tensor(self))
                self.target = tf.cast(tree.get_tensor(self), tf.float32) # cast to an int tensor
                #self.target = pagie_poly(self.terminal.set, self.target_dims)
        else:
            self.target = target

        self.current_generation = 0

        # for now:
        # fitness - stop evolution if best individual is close enought to target (or given value)
        # generation - stop evolution after a given number of generations
        if self.objective == 'minimizing':
            self.objective_condition = lambda x: (x < self.best_overall['fitness'])
        else:
            self.objective_condition = lambda x: (x > self.best_overall['fitness'])

        if stop_criteria == 'fitness':  # if fitness then stop value means error
            self.stop_value = stop_value
            if self.objective == 'minimizing':
                self.condition = lambda: (self.best > self.stop_value)
            else:
                self.condition = lambda: (self.best < self.stop_value)
        else: # if generations then stop_value menas number of generations to evaluate
            self.stop_value = int(stop_value)
            self.condition = lambda: (self.current_generation <= self.stop_value)

        # update timers
        self.elapsed_init_time += time.time() - start_init
        if self.debug > 0: print("Elapsed init time: ", self.elapsed_init_time)
        print(bcolors.OKGREEN + "Engine seed:", self.experiment.seed, bcolors.ENDC)
        self.update_engine_time()

    def restart(self, new_stop = 10):
        self.last_stop = self.stop_value
        self.stop_value = self.stop_value + new_stop

    def generate_program(self, method, max_nodes, max_depth, min_depth = -1, root=True):
        terminal = False
        children = []

        # set primitive node
        # (max_depth * max_nodes) == 0 means if we either achieved maximum depth ( = 0) or there are no more
        # nodes allowed on this tree then we have to force the terminal placement
        # to ignore max_nodes, we can set this to -1
        #if (max_depth * max_nodes) == 0 or (
        #        (not root) and method == 'grow' and self.engine_rng.random() < (len(terminal_set) / (len(terminal_set) + len(function_set)))):
        # TODO: review ((self.max_init_depth - max_depth) >= min_depth), works if we call initialize pop with max_init_depth

        if (max_depth * max_nodes) == 0 or (
                #(not root) and
                (method == 'grow') and
                ((self.max_init_depth - max_depth) >= min_depth) and
                (self.engine_rng.random() < self.terminal_prob)
        ):
            if self.engine_rng.random() < self.scalar_prob:
                primitive = 'scalar'
                if self.engine_rng.random() < self.uniform_scalar_prob:
                    children = [self.engine_rng.uniform(self.erc_min, self.erc_max)]
                else:
                    children = [self.engine_rng.uniform(self.erc_min, self.erc_max) for i in range(self.terminal.dimension)]
            else:
                primitive = self.engine_rng.choice(list(self.terminal.set)) # TODO: what is this???
            terminal = True
        else:
            primitive = self.engine_rng.choice(list(self.function.set))
            if max_nodes > 0: max_nodes -= 1

        # set children
        if not terminal:
            for i in range(self.function.set[primitive][0]):
                max_nodes, child = self.generate_program(method, max_nodes, max_depth - 1, min_depth = min_depth, root=False)
                children.append(child)

        return max_nodes, Node(value=primitive, terminal=terminal, children=children)

    def generate_population(self, individuals, method, max_nodes, max_depth, min_depth = -1):
        if max_depth < 2:
            if max_depth <= 0:
                return 0, []
            #if max_depth == 1:
            #    method = 'full'

        population = []
        pop_nodes = 0


        if method == 'full' or method == 'grow':
            for i in range(individuals):

                # TODO: why was this without min_depth? TEST
                _n, t = self.generate_program(method, max_nodes, max_depth, min_depth=min_depth)

                tree_nodes = (max_nodes - _n)
                pop_nodes += tree_nodes
                dep, nod = t.get_depth()
                population.append({'tree': t, 'fitness': 0, 'depth': dep, 'nodes': nod})
        else:
            # -1 means no minimun depth, but the ramped 5050 default should be 2 levels
            #_min_depth = 2 if (min_depth <= 0) else min_depth  # TODO: (commented)
            _min_depth = min_depth


            divisions = max_depth - (_min_depth - 1)
            parts = math.floor(individuals / divisions)
            last_part = individuals - (divisions - 1) * parts
            #load_balance_index = (max_depth - 1) - (last_part - parts)
            load_balance_index = (max_depth + 1) - (last_part - parts)
            part_array = []

            num_parts = parts
            mfull = math.floor(num_parts / 2)
            for i in range(_min_depth, max_depth + 1):

                # TODO: shouldn't i - 2 be i - min_depth? TEST or just i
                #if i - 2 == load_balance_index:
                if i == load_balance_index:
                    num_parts += 1
                    mfull += 1
                part_array.append(num_parts)
                met = 'full'
                for j in range(num_parts):

                    if j >= mfull:
                        met = 'grow'
                    #print("i: ", i, "min dep: ", min_depth)

                    _n, t = self.generate_program(met, max_nodes, i, min_depth = min_depth)
                    tree_nodes = (max_nodes - _n)
                    #print("Tree nodes: " + str(tree_nodes))
                    pop_nodes += tree_nodes
                    dep, nod = t.get_depth()
                    population.append({'tree': t, 'fitness': 0, 'depth': dep, 'nodes': nod})

        if len(population) != individuals:
            print(bcolors.FAIL + "[ERROR]:\tWrong number of individuals generated: " + str(len(population)) + "/" + str(individuals) , bcolors.ENDC)

        return pop_nodes, population

    def domain_range(self, final_tensor):
        if self.domain_mode == 'log':
            final_tensor = tf.math.abs(final_tensor)
            final_tensor = tf.clip_by_value(tf.math.log(final_tensor), clip_value_min=_min_domain, clip_value_max=_max_domain)
            return tf.clip_by_value(final_tensor, clip_value_min=_min_domain, clip_value_max=_max_domain)
        elif self.domain_mode == 'dynamic':
            final_tensor = tf.math.abs(final_tensor) # TODO: between 0,+inf -> 0,1
            return (final_tensor / (1 + final_tensor)) * _domain_delta + _min_domain
        elif self.domain_mode == 'mod':
            return tf.math.floormod(final_tensor, _domain_delta) + _min_domain
        else:
            return tf.clip_by_value(final_tensor, clip_value_min=_min_domain, clip_value_max=_max_domain)

    #@tf.function
    def final_transform_domain(self, final_tensor):
        #global _min_domain, _max_domain, _domain_delta
        final_tensor = tf.where(tf.math.is_nan(final_tensor), 0.0, final_tensor)
        final_tensor = self.domain_range(final_tensor)

        #final_tensor = tf.cast(final_tensor, tf.uint8)
        return final_tensor

    def calculate_tensors(self, population):
        tensors = []
        #with tf.device(self.device):
        start = time.time()

        for p in population:

            #print("Evaluating ind: ", p['tree'].get_str())
            #_start = time.time()
            test_tens = p['tree'].get_tensor(self)
            tens = self.final_transform_domain(test_tens)
            tensors.append(tens)

            #dur = time.time() - _start
            #print("Took", dur, "s")


        time_tensor = time.time() - start

        self.elapsed_tensor_time += time_tensor
        self.recent_tensor_time = time_tensor
        return tensors, time_tensor

    def fitness_func_wrap(self, population, f_path):

        # calculate tensors
        #with tf.device(self.device):
        if self.debug > 4: print("\nEvaluating generation: " + str(self.current_generation))
        with tf.device(self.device):
            tensors, time_taken = self.calculate_tensors(population)
        if self.debug > 4: print("Calculated " + str(len(tensors)) + " tensors in (s): " + str(time_taken))

        # calculate fitness
        if self.debug > 4: print("Assessing fitness of individuals...")
        _s = time.time()
        # Notes: measuring time should not be up to the fit function writer. We should provide as much info as possible
        # Maybe best pop shouldnt be required

        population, best_ind = self.fitness_func(generation = self.current_generation,
                                                 population = population,
                                                 tensors = tensors,
                                                 f_path = f_path,
                                                 rng = self.engine_rng,
                                                 objective = self.objective,
                                                 resolution = self.target_dims,
                                                 stf = self.save_to_file,
                                                 target = self.target,
                                                 dim = self.dimensionality,
                                                 debug = False if (self.debug == 0) else True)
        fitness_time = time.time() - _s

        # save best
        if self.save_best:
            fn = self.experiment.best_directory + "best_gen" + str(self.current_generation).zfill(5) + "_"
            save_image(tensors[best_ind], best_ind, fn, self.target_dims, addon='_best')

        self.elapsed_fitness_time += fitness_time
        self.recent_fitness_time = fitness_time
        if self.debug > 4: print("Assessed " + str(len(tensors)) + " fitness tensors in (s): " + str(fitness_time))

        return population, population[best_ind], tensors

    def generate_pop_from_expr(self, strs):
        population = []
        nodes_generated = 0
        maxpopd = -1

        for p in strs:
            t, node = str_to_tree(p, self.terminal.set)

            thisdep, t = node.get_depth()
            if thisdep > maxpopd:
                maxpopd = thisdep

            population.append({'tree': node, 'fitness': 0, 'depth': thisdep, 'nodes': t})
            if self.debug > 0:
                print("Number of nodes:\t:" + str(t))
                print(node.get_str())
            nodes_generated += t
        if self.debug > 0:
            print("Total number of nodes:\t" + str(nodes_generated))

        return population, nodes_generated, maxpopd


    def generate_pop_from_file(self, read_from_file, pop_size = float('inf')):
        # open population files
        strs = []
        with open(read_from_file) as fp:
            line = fp.readline()
            cnt = 0
            while line and cnt < pop_size:
                #line = line[:-1]
                strs.append(line)
                line = fp.readline()
                cnt += 1
            if cnt < pop_size < float('inf'):
                print(bcolors.WARNING + "[WARNING]:\tCould only read " + str(cnt) + " expressions from population file " +
                      str(read_from_file) + " instead of specified population size of " + str(pop_size) , bcolors.ENDC)

        # convert expressions to trees
        return self.generate_pop_from_expr(strs)

    def initialize_population(self, max_depth, min_depth, individuals, method, max_nodes, immigration = False, read_from_file = None):
        #generate trees

        start_init_population = time.time()

        if read_from_file is None:
            nodes_generated, population = self.generate_population(individuals, method, max_nodes, max_depth, min_depth)
        else:
            if ".txt" not in read_from_file:
                print(bcolors.FAIL + "[ERROR]:\tCould not read from file: " + str(read_from_file) + ", randomly generating population instead." , bcolors.ENDC)
                read_from_file = None
                nodes_generated, population = self.generate_population(individuals, method, max_nodes, max_depth, min_depth)
            else: # read from population files


                population, nodes_generated, maxpopd = self.generate_pop_from_file(read_from_file, pop_size=self.population_size)
                # open population files

                if maxpopd > self.max_tree_depth:
                    newmaxpopd = max(maxpopd, self.max_tree_depth)
                    print(bcolors.WARNING + "[WARNING]:\tMax depth of input trees (" + str(maxpopd) + ") is higher than defined max tree depth (" +
                          str(self.max_tree_depth) + "), clipping max tree depth value to " + str(newmaxpopd) , bcolors.ENDC)
                    self.max_tree_depth = newmaxpopd

                if self.debug > 0:
                    for p in population:
                        print(p['tree'].get_str())

        tree_generation_time = time.time() - start_init_population

        if self.debug > 0: # 1
            print("Generated Population: ")
            self.print_population(population)

        #calculate fitness
        f_fitness_path = self.experiment.immigration_directory if immigration else self.experiment.current_directory
        population, best_pop, tensors = self.fitness_func_wrap(population=population,
                                                                f_path=f_fitness_path)

        total_time = self.recent_fitness_time + self.recent_tensor_time

        if self.debug > 0:  # print info
            print("\nInitial Population: ")
            print("Generation method: " + ("Ramdom expression" if (read_from_file is None) else "Read from file"))
            print("Generated trees in: " + str(tree_generation_time) + " (s)")
            print("Evaluated Individuals in: " + str(total_time) + " (s)")
            print("\nIndividual(s): ")
            print("Nodes generated: " + str(nodes_generated))
            for i in range(individuals):
                print("\nIndiv " + str(i) + ":\nExpr: " + population[i]['tree'].get_str())
                print("Fitness: " + str(population[i]['fitness']))
                print("Depth: " + str(population[i]['depth']))
                print("Depth: " + str(population[i]['nodes']))

        return population, best_pop, tensors


    def write_pop_to_csv(self, fp = None):
        if self.write_gen_stats:
            genstr = "gen_" + str(self.current_generation).zfill(5)
            fn = self.experiment.logging_diredctory if fp is None else fp
            fn += genstr + "_stats.csv"
            with open(fn, mode='w', newline='') as file:
                fwriter = csv.writer(file, delimiter=',')
                ind = 0
                for p in self.population:
                    fwriter.writerow([str(ind), genstr + "_indiv_" + str(ind).zfill(5), p['fitness'], p['depth'], p['nodes'], p['tree'].get_str()])
                    ind += 1


    def population_stats(self, population, fitness = True, depth = True, nodes = True):
        keys = []
        res = {}
        if fitness: keys.append('fitness')
        if depth: keys.append('depth')
        if nodes: keys.append('nodes')
        for k in keys:
            res[k] = []
            for p in population:
                res[k].append(p[k])
            _avg = np.average(res[k])
            _std = np.std(res[k])
            _best = self.best[k]
            _best_all = self.best_overall[k]
            res[k] = [_avg, _std, _best, _best_all]
        return res

    def generate_pop_images(self, expressions, fpath = None):
        fp = self.experiment.current_directory if fpath is None else fpath
        tensors = []
        if isinstance(expressions, str):
            pop, number_nodes, max_dep = self.generate_pop_from_file(expressions)
            tensors, time_taken = self.calculate_tensors(pop)
        elif isinstance(expressions, list):
            #for p in expressions:
            #    print(p)
            #    _, node = str_to_tree(p, self.terminal.set)
            #    tensors.append(node)
            pop, number_nodes, max_dep = self.generate_pop_from_expr(expressions)
            tensors, time_taken = self.calculate_tensors(pop)
        else:
            print(bcolors.FAIL + "[ERROR]:\tTo generate images from a population please enter either"
                                 " a file or a list with the corresponding expressions." , bcolors.ENDC)
            return
        index = 0
        for t in tensors:
            save_image(t, index, fp, self.target_dims)
            index += 1

    def run(self, stop_value = 10):

        print(bcolors.OKGREEN + "\n\n" +  "=" * 84, bcolors.ENDC)
        if self.save_state > 0:
            self.restart(stop_value)

        if self.fitness_func is None:
            print(bcolors.FAIL + "[ERROR]:\tFitness function must be defined to run evolutionary process." , bcolors.ENDC)
            return

        # either generate initial pop randomly or read fromfile along with remaining experiment data
        tensors = []
        if self.previous_state is not None:
            self.current_generation = self.previous_state['generations']
            self.experiment.seed += self.current_generation

            self.experiment.set_generation_directory(self.current_generation)

            # time counters
            self.elapsed_init_time = self.previous_state['elapsed_init_time']
            self.elapsed_fitness_time = self.previous_state['elapsed_fitness_time']
            self.elapsed_tensor_time = self.previous_state['elapsed_tensor_time']
            self.elapsed_engine_time = self.previous_state['elapsed_engine_time']

            self.population = self.previous_state['population']
            self.best = self.previous_state['best']
            self.best_overall = self.previous_state['best_overall']

        else:
            self.experiment.set_generation_directory(self.current_generation)
            self.population, self.best, tensors = self.initialize_population(self.max_init_depth,
                                                                            self.min_init_depth,
                                                                            self.population_size,
                                                                            self.method,
                                                                            self.max_nodes,
                                                                            immigration=False,
                                                                            read_from_file=self.pop_file)
            self.best_overall = self.best

        # write first gen data
        self.write_pop_to_csv(self.pop_file_path)
        self.save_state_to_file(self.experiment.logging_diredctory)
        if self.debug > 2: self.print_engine_state(force_print=True)

        # display gen statistics
        data = []
        pops = self.population_stats(self.population)
        data.append([pops['fitness'][0], pops['fitness'][1], pops['fitness'][2], pops['fitness'][3],
                     pops['depth'][0], pops['depth'][1], pops['depth'][2], pops['depth'][3],
                     pops['nodes'][0], pops['nodes'][1], pops['nodes'][2], pops['nodes'][3], tensors])
        if self.save_state == 0:
            print(bcolors.BOLD + bcolors.OKCYAN + "\n[       |                    FITNESS                    |                     DEPTH                     |                     NODES                     |              TIMINGS              ]" , bcolors.ENDC)
            print(bcolors.BOLD + bcolors.OKCYAN +   "[  gen  |    avg    ,    std    , best(gen) , best(all) |    avg    ,    std    , best(gen) , best(all) |    avg    ,    std    , best(gen) , best(all) | generation,  fit eval ,tensor eval]\n" , bcolors.ENDC)
        #print(bcolors.BOLD + bcolors.OKCYAN + "\n[  gen  ,     fit avg,   std,  best (gen),   dep avg,   dep std,  dep best,   nod avg,   nod std,  nod best, gen time, fit time,tensor time]\n" , bcolors.ENDC)
        #print("[", self.current_generation, ",", data[-1][0], ",", data[-1][1], ",", data[-1][2], ",", data[-1][3], ",", data[-1][4], ",", data[-1][5], ",", data[-1][6], ",", data[-1][7], ",", data[-1][8], "]")
        print(bcolors.OKBLUE + "[%7d, %10.6f, %10.6f, %10.6f, %10.6f, %10.3f, %10.6f, %10d, %10d, %10.3f, %10.6f, %10d, %10d, %10.6f, %10.6f, %10.6f]"%((self.current_generation,) + tuple(data[-1][:-1]) + (self.recent_engine_time, self.recent_fitness_time, self.recent_tensor_time)) , bcolors.ENDC)


        self.current_generation += 1

        while self.condition():

            # Update seed according to generation
            self.engine_rng = random.Random(self.experiment.seed)

            # Set directory to save engine state in this generation
            self.experiment.set_generation_directory(self.current_generation)

            # immigrate individuals
            if self.current_generation % self.immigration == 0:
                immigrants, _, _ = self.initialize_population(self.max_init_depth,
                                                           self.min_init_depth,
                                                           self.population_size,
                                                           self.method,
                                                           self.max_nodes,
                                                           immigration = True, read_from_file = None)

                self.population.extend(immigrants)
                self.population = self.engine_rng.sample(self.population, self.population_size)

            # create new population of individuals
            if self.objective == 'minimizing':
                new_population = nsmallest(self.elitism, self.population, key=itemgetter('fitness'))
            else:
                new_population = nlargest(self.elitism, self.population, key=itemgetter('fitness'))

            # TODO: this is to monitor number of retries, it will be changed
            retrie_cnt = []

            # for each individual, build new population
            for current_individual in range(self.population_size - self.elitism):

                member_depth = float('inf')

                # generate new individual with acceptable depth

                rcnt = 0
                while member_depth > self.max_tree_depth:

                    parent = self.tournament_selection()
                    random_n = self.engine_rng.random()
                    if random_n < self.crossover_rate:
                        parent_2 = self.tournament_selection()
                        indiv_temp = self.crossover(parent['tree'], parent_2['tree'])
                        #print("cross")
                    elif (random_n >= self.crossover_rate) and (random_n < self.crossover_rate + self.mutation_rate):
                        indiv_temp = self.mutation(parent['tree'])
                        #print("mut")
                    else:
                        indiv_temp = parent['tree']

                    member_depth, member_nodes = indiv_temp.get_depth()
                    #print("Eval ind: ", indiv_temp.get_str())

                    rcnt+=1
                retrie_cnt.append(rcnt)

                # add newly formed child
                new_population.append({'tree': indiv_temp, 'fitness': 0, 'depth': member_depth, 'nodes':member_nodes})
                #if member_depth > max_tree_depth:
                #    max_tree_depth = member_depth

                # print individual
                if self.debug > 10: print("Individual " + str(current_individual) + ": " + indiv_temp.get_str())

            if self.debug >= 4:
                rstd = np.average(np.array(retrie_cnt))
                print("[DEBUG]:\tAverage evolutionary ops retries for generation " + str(self.current_generation) + ": " + str(rstd))


            # calculate fitness of the new population
            new_population, self.best, tensors = self.fitness_func_wrap(population=new_population,
                                                               f_path=self.experiment.current_directory)

            # bloat_control
            # if self.bloat_control == 'full_dynamic_dep':
            #    self.max_tree_depth = min(self.best['depth'], self.min_tree_depth)
            # elif self.bloat_control == 'dynamic_dep':
            #    self.max_tree_depth = max(self.best['depth'], self.max_tree_depth)
            # print("max tree depth: ", self.max_tree_depth)


            # update best and population
            if self.objective_condition(self.best['fitness']): self.best_overall = self.best
            self.population = new_population

            # update engine time
            self.update_engine_time()

            # save engine state
            #if self.save_to_file != 0 and (self.current_generation % self.save_to_file) == 0:
            self.save_state_to_file(self.experiment.logging_diredctory)

            # add population data to statistics and display gen statistics
            pops = self.population_stats(self.population)
            data.append([pops['fitness'][0], pops['fitness'][1], pops['fitness'][2], pops['fitness'][3],
                         pops['depth'][0], pops['depth'][1], pops['depth'][2], pops['depth'][3],
                         pops['nodes'][0], pops['nodes'][1], pops['nodes'][2], pops['nodes'][3], tensors])
            print(bcolors.OKBLUE + "[%7d, %10.6f, %10.6f, %10.6f, %10.6f, %10.3f, %10.6f, %10d, %10d, %10.3f, %10.6f, %10d, %10d, %10.6f, %10.6f, %10.6f]" %((self.current_generation,) + tuple(data[-1][:-1]) + (self.recent_engine_time, self.recent_fitness_time, self.recent_tensor_time)), bcolors.ENDC)

            self.write_pop_to_csv(self.pop_file_path)

            # print engine state
            self.print_engine_state(force_print=False)

            # advance generation
            self.current_generation += 1
            self.experiment.seed += 1

        # write statistics(data) to csv
        self.write_stats_to_csv(data, self.stats_file_path)

        # print final stats
        if self.debug < 0:
            self.print_engine_state(force_print=True)
        else:
            print(bcolors.OKGREEN + "\nElapsed Engine Time: \t" + str(self.elapsed_engine_time) + " sec.")
            print("\nElapsed Init Time: \t\t" + str(self.elapsed_init_time) + " sec.")
            print("Elapsed Tensor Time: \t" + str(self.elapsed_tensor_time) + " sec.")
            print("Elapsed Fitness Time:\t" + str(self.elapsed_fitness_time) + " sec.")
            print("\nBest individual (generation):\n" + bcolors.OKCYAN + self.best['tree'].get_str())
            print("\nBest individual (overall):\n" + bcolors.OKCYAN + self.best_overall['tree'].get_str(), bcolors.ENDC)

        if self.save_graphics: self.graph_statistics()
        print(bcolors.BOLD + bcolors.OKGREEN +  "=" * 84, "\n\n", bcolors.ENDC)

        self.save_state += 1
        return data, tensors

    def graph_statistics(self):

        if not self.show_graphics:
            matplotlib.use('Agg')


        matplotlib.rcParams.update({'font.size': 16})
        line_start = 2
        line_end = self.stop_value + 1 + line_start

        avg_fit = []
        std_fit = []
        best_fit = []
        avg_dep = []
        std_dep = []
        best_dep = []
        lcnt = 1

        fn = self.experiment.working_directory if self.stats_file_path is None else self.stats_file_path
        with open(fn + self.overall_stats_filename, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if self.debug > 10: print(row)
                if line_start <= lcnt < line_end:
                    avg_fit.append(float(row[1]))
                    std_fit.append(float(row[2]))
                    best_fit.append(float(row[4]))
                    avg_dep.append(float(row[5]))
                    std_dep.append(float(row[6]))
                    best_dep.append(float(row[8]))
                lcnt += 1

        # showing best overall
        fig, ax = plt.subplots(1, 1)
        ax.plot(range(self.stop_value + 1), avg_fit, linestyle='-', label="AVG")
        ax.plot(range(self.stop_value + 1), best_fit, linestyle='-', label="BEST")
        pylab.legend(loc='upper left')
        ax.set_xlabel('Generations')
        ax.set_ylabel('Fitness')
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_title('Fitness across generations')
        fig.set_size_inches(12, 8)
        plt.savefig(self.experiment.working_directory + 'Fitness.png')
        if self.show_graphics: plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        ax.plot(range(self.stop_value + 1), avg_dep, linestyle='-', label="AVG")
        ax.plot(range(self.stop_value + 1), best_dep, linestyle='-', label="BEST")
        ax.set_xlabel('Generations')
        ax.set_ylabel('Depth')
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_title('Avg depth across generations')
        fig.set_size_inches(12, 8)
        plt.savefig(self.experiment.working_directory + 'Depth.png')
        if self.show_graphics: plt.show()
        plt.close(fig)


    def write_stats_to_csv(self, data, fp = None):
        # evolutionary stats across generations
        fn = self.experiment.working_directory if fp is None else fp
        fn += self.overall_stats_filename
        with open(fn, mode='a', newline='') as file:
            fwriter = csv.writer(file, delimiter=',')
            ind = 0
            for d in data:
                if ind == 0 and self.save_state == 0:
                    file.write("[generation, fitness avg, fitness std, fitness generational best, fitness overall best, depth avg, depth std, depth generational best, depth overall best, generation time, fitness time, tensor time]\n")
                fwriter.writerow([ind] + d[:-1])
                ind += 1

        # overall engine stats
        fn = self.experiment.working_directory + "tensorgp_" + str(self.target_dims[0]) + "_" + str(self.experiment.seed) + '_timings.csv'
        with open(fn, mode='a', newline='') as file:
            fwriter = csv.writer(file, delimiter=',')
            if self.save_state == 0:
                file.write("[resolution, seed, initialization time, tensor time, fitness time, total, engine time]\n")
            fwriter.writerow([self.target_dims[0], self.experiment.seed, self.elapsed_init_time,
                              self.elapsed_tensor_time, self.elapsed_fitness_time, self.elapsed_engine_time])


    def update_engine_time(self):
        t_ = time.time()
        self.elapsed_engine_time += t_ - self.last_engine_time
        self.recent_engine_time = t_ - self.last_engine_time
        self.last_engine_time = t_



    def save_state_to_file(self, filename):
        #print("[DEBUG]:\t" + filename)
        if self.write_final_pop and not self.condition():
            #print("printing last stats")
            with open(self.path_to_final_pop + self.experiment.get_experiment_filename() + "_final_pop.txt", "w") as text_file:
                try:
                    for i in range(self.population_size):
                        ind = self.population[i]
                        text_file.write(str(ind['tree'].get_str()) + "\n")
                except IOError as e:
                    print(bcolors.FAIL + "[ERROR]:\tI/O error while writing engine state ({0}): {1}".format(e.errno, e.strerror) , bcolors.ENDC)

        if self.write_log or not self.condition():
            with open(filename + "log.txt", "w") as text_file:
                try:
                    # general info

                    text_file.write("Engine state information:\n")
                    text_file.write("Population Size: " + str(self.population_size) + "\n")
                    text_file.write("Tournament Size: " + str(self.tournament_size) + "\n")
                    text_file.write("Mutation Rate: " + str(self.mutation_rate) + "\n")
                    text_file.write("Crossover Rate: " + str(self.crossover_rate) + "\n")
                    text_file.write("Minimun Tree Depth: " + str(self.max_tree_depth) + "\n")
                    text_file.write("Maximun Tree Depth: " + str(self.min_tree_depth) + "\n")
                    text_file.write("Initial Tree Depth: " + str(self.max_init_depth) + "\n")
                    text_file.write("Population method: " + str(self.method) + "\n")
                    text_file.write("Terminal Probability: " + str(self.terminal_prob) + "\n")
                    text_file.write("Scalar Probability (from terminals): " + str(self.scalar_prob) + "\n")
                    text_file.write("Uniform Scalar (scalarT) Probability (from terminals): " + str(self.uniform_scalar_prob) + "\n")
                    text_file.write("Stop Criteria: " + str(self.stop_criteria) + "\n")
                    text_file.write("Stop Value: " + str(self.stop_value) + "\n")
                    text_file.write("Objective: " + str(self.objective) + "\n")
                    text_file.write("Generations per immigration: " + str(self.immigration) + "\n")
                    text_file.write("Dimensions: " + str(self.target_dims) + "\n")
                    text_file.write("Max nodes: " + str(self.max_nodes) + "\n")
                    text_file.write("Debug Level: " + str(self.debug) + "\n")
                    text_file.write("Device: " + str(self.device) + "\n")
                    text_file.write("Save to file: " + str(self.save_to_file) + "\n")
                    # previous state and others
                    text_file.write("Generation: " + str(self.current_generation) + "\n")
                    text_file.write("Engine Seed : " + str(self.experiment.seed) + "\n") # redundancy
                    text_file.write("Engine ID : " + str(self.experiment.ID) + "\n")     # to check while loading
                    text_file.write("Elapse Engine Time: " + str(self.elapsed_engine_time) + "\n")
                    text_file.write("Elapse Initiation Time: " + str(self.elapsed_init_time) + "\n")
                    text_file.write("Elapse Tensor Time: " + str(self.elapsed_tensor_time) + "\n")
                    text_file.write("Elapse Fitness Time: " + str(self.elapsed_fitness_time) + "\n")

                    # population
                    text_file.write("\n\nPopulation: \n")

                    text_file.write("\nBest individual (generation):")
                    text_file.write("\nExpression: " + str(self.best['tree'].get_str()))
                    text_file.write("\nFitness: " + str(self.best['fitness']))
                    text_file.write("\nDepth: " + str(self.best['depth']) + "\n")

                    text_file.write("\nBest individual (overall):")
                    text_file.write("\nExpression: " + str(self.best_overall['tree'].get_str()))
                    text_file.write("\nFitness: " + str(self.best_overall['fitness']))
                    text_file.write("\nDepth: " + str(self.best_overall['depth']) + "\n")

                    # 4 lines per indiv
                    text_file.write("\nCurrent Population:\n")
                    for i in range(self.population_size):
                        ind = self.population[i]
                        text_file.write("\n\nIndividual " + str(i) + ":")
                        text_file.write("\nExpression: " + str(ind['tree'].get_str()))
                        text_file.write("\nFitness: " + str(ind['fitness']))
                        text_file.write("\nDepth: " + str(ind['depth']))


                except IOError as e:
                    print(bcolors.FAIL + "[ERROR]:\tI/O error while writing engine state ({0}): {1}".format(e.errno, e.strerror) , bcolors.ENDC)

    def print_population(self, population):
        for i in range(len(population)):
            p = population[i]
            print("\nIndividual " + str(i) + ":")
            print("Fitness:\t" + str(p['fitness']))
            print("Depth:\t" + str(p['depth']))
            print("Expression:\t" + str(p['tree'].get_str()))

    def print_engine_state(self, force_print = False):
        if force_print and self.debug > 0:
            print("\n____________________Engine state____________________")
            if not self.condition(): print("The run is over!")

            print("\nGeneral Info:")
            print("Engine Seed:\t" + str(self.experiment.seed))
            print("Engine ID:\t" + str(self.experiment.ID))
            print("Generation:\t" + str(self.current_generation))

            print("\nBest Individual (generation):")
            print("Fitness:\t" + str(self.best['fitness']))
            print("Depth:\t" + str(self.best['depth']))
            if self.debug > 2:
                print("Expression:\t" + str(self.best['tree'].get_str()))

            print("\nBest Individual (overall):")
            print("Fitness:\t" + str(self.best_overall['fitness']))
            print("Depth:\t" + str(self.best_overall['depth']))
            if self.debug > 2:
                print("Expression:\t" + str(self.best_overall['tree'].get_str()))

            #print population
            if self.debug > 10:
                print("\nPopulation:")
                self.print_population(self.population)

            print("\nTimers:")
            print("Elapsed initial time:\t" + str(round(self.elapsed_init_time, 6)) + " s")
            print("Elapsed fitness time:\t" + str(round(self.elapsed_fitness_time, 6)) + " s")
            print("Elapsed tensor time:\t" + str(round(self.elapsed_tensor_time, 6)) + " s")
            print("Elapsed engine time:\t" + str(round(self.elapsed_engine_time, 6)) + " s")
            print("\n____________________________________________________\n")


class Function_Set:

    def __init__(self, fset, edim, debug = 0):

        self.debug = debug
        self.set = {}
        self.arity = {}
        self.min_arity = float('inf')
        self.max_arity = 0

        self.operators_def = {  # arity, function
            'abs': [1, resolve_abs_node],
            'add': [2, resolve_add_node],
            'and': [2, resolve_and_node],
            'clip': [3, resolve_clip_node],
            'cos': [1, resolve_cos_node],
            'div': [2, resolve_div_node],
            'exp': [1, resolve_exp_node],
            'frac': [1, resolve_frac_node],
            'if': [3, resolve_if_node],
            'len': [2, resolve_len_node],
            'lerp': [3, resolve_lerp_node],
            'log': [1, resolve_log_node],
            'max': [2, resolve_max_node],
            'mdist': [2, resolve_mdist_node],
            'min': [2, resolve_min_node],
            'mod': [2, resolve_mod_node],
            'mult': [2, resolve_mult_node],
            'neg': [1, resolve_neg_node],
            'or': [2, resolve_or_node],
            'pow': [2, resolve_pow_node],
            'sign': [1, resolve_sign_node],
            'sin': [1, resolve_sin_node],
            'sqrt': [1, resolve_sqrt_node],
            'sstep': [1, resolve_sstep_node],
            'sstepp': [1, resolve_sstepp_node],
            'step': [1, resolve_step_node],
            'sub': [2, resolve_sub_node],
            'tan': [1, resolve_tan_node],
            'warp': [edim + 1, resolve_warp_node],
            'xor': [2, resolve_xor_node],
        }

        if fset is None:
            fset = set(self.operators_def.keys())
        fset = sorted(fset)

        for e in fset:
            fset_values = self.operators_def.get(e)
            self.set[e] = fset_values

            arity_val = fset_values[0]
            self.min_arity = min(self.min_arity, arity_val)
            self.max_arity = max(self.max_arity, arity_val)
            if arity_val not in self.arity:
                self.arity[arity_val] = []
            self.arity[arity_val].append(e)

    def add_to_set(self, operator_name, number_of_args, function_pointer):
        self.min_arity = min(self.min_arity, number_of_args)
        self.max_arity = max(self.max_arity, number_of_args)
        if operator_name in self.set and self.debug > 0:
            print(bcolors.WARNING + "[WARNING]:\tOperator already existing in current function_set, overriding..." , bcolors.ENDC)
            if operator_name in self.operators_def and self.debug > 1:
                print(bcolors.WARNING + "[WARNING]:\tOverriding operator", operator_name, ", which is an engine defined operator. This is not recommended." , bcolors.ENDC)
            self.remove_from_set(operator_name)

        self.set[operator_name] = [number_of_args, function_pointer]
        if number_of_args not in self.arity:
            self.arity[number_of_args] = []
        self.arity[number_of_args].append(operator_name)

    def remove_from_set(self, operator_name):
        if operator_name not in self.set:
            print(bcolors.FAIL + "[ERROR]:\tOperator", operator_name, "not present in function set, failed to remove." , bcolors.ENDC)
        else:
            if operator_name in self.operators_def and self.debug > 0:
                print(bcolors.WARNING + "[WARNING]:\tRemoving operator", operator_name, ", which is an engine defined operator. I hope you know what you are doing." , bcolors.ENDC)
            if operator_name != 'mult': # we wont actually remove some operators because we might need them for scaling in the begginning
                ari = self.set[operator_name][0]
                del self.set[operator_name]
                self.arity[ari].remove(operator_name)
                if len(self.arity[ari]) == 0:
                    del self.arity[ari]
                    self.min_arity = int('inf')
                    self.max_arity = 0
                    if len(self.arity.keys()) > 0:
                        for a in self.arity:
                            self.min_arity = min(self.min_arity, a)
                            self.max_arity = max(self.max_arity, a)

    def __str__(self):
        res = "\nFunction Set:\n"
        for s in self.set:
            res += str(s) + ", " + str(self.set[s][0]) + "\n"
        res += "\nArity sorted:\n"
        for s in self.arity:
            res += str(s) + ", " + str(self.arity[s]) + "\n"
        return res


def uniform_sampling(res, minval = _min_domain, maxval = _max_domain):
    return tf.random.uniform(res, minval=minval, maxval=maxval)

# engref optional to allow for independent set creation
# when calling inside the engine, we will provide the engine ref itself to allow terminal definition by expressions
class Terminal_Set:

    def __init__(self, effective_dim, resolution, function_ptr_to_var_node = resolve_var_node, debug = 0, engref = None):
        self.debug = debug
        self.engref = engref
        self.dimension = resolution[effective_dim] if (effective_dim < len(resolution)) else 1
        self.set = self.make_term_variables(0, effective_dim, resolution, function_ptr_to_var_node) # x, y
        self.latentset = self.make_term_variables(effective_dim, len(resolution), resolution, function_ptr_to_var_node) # z for terminal

    def make_term_variables(self, start, end, resolution, fptr):
        res = {}
        for i in range(start, end): # TODO, what does this dim - 1 do? Is it only to do with warp?
            digit = i
            name = ""

            while True:
                n = digit % 26
                val = n if n <= 2 else n - 26
                name = chr(ord('x') + val) + name
                digit //= 26
                if digit <= 0:
                    break

            if self.debug > 2: print("[DEBUG]:\tAdded terminal " + str(name))

            vari = i # TODO: ye, this is because of images, right?....
            if i < 2: vari = 1 - i
            res[name] = fptr(np.copy(resolution), vari)
            #res[name] = fptr(np.copy(resolution), minval = _min_domain, maxval = _max_domain)
        return res

    def add_to_set(self, name, t, engref = None):
        if name in self.set and self.debug > 0:
            print(bcolors.WARNING + "[WARNING]:\tOperator already existing in current terminal set. Be careful not to redefine 'variables' like x, y, z, etc..." , bcolors.ENDC)

        tensor = t
        if isinstance(t, str):
            if engref is None:
                print(bcolors.FAIL + "[ERROR]:\tIf you wish to generate a terminal by an expression, pass an engine instance when initializing the Terminal set." , bcolors.ENDC)
            else:
                _, tree = str_to_tree(t, self.set)
                tensor = tree.get_tensor(engref)
        else:
            tensor = t

        self.set[name] = tensor

    def remove_from_set(self, name):
        if name not in self.set:
            print(bcolors.FAIL + "[ERROR]:\tOperator", name, "not present in terminal set, failed to remove." , bcolors.ENDC)
        else:
            if self.debug:
                print(bcolors.WARNING + "[WARNING]:\tBe careful while removing terminals, your trees must be expressed accordingly. I REALLY hope you know what you are doing..." , bcolors.ENDC)
            del self.set[name]

    def __str__(self):
        res = "\nTerminal Set:\n"
        for s in self.set:
            res += s + "\n"
        return res
