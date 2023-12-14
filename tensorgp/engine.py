# MIT License

# Copyright (c) 2020-2022 Francisco Baeta (University of Coimbra)
# Copyright (c) 2020-2022 João Correia (University of Coimbra)
# Copyright (c) 2020-2022 Tiago Martins (University of Coimbra)
# Copyright (c) 2020-2022 Penousal Machado (University of Coimbra)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import csv
import pickle
import subprocess
from collections import Counter
import copy
import datetime
import math
import os
import re

import keyboard as keyboard
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pylab
import time
import random
import json
from heapq import nsmallest, nlargest
from operator import itemgetter
from PIL import Image
import configparser
import ast
import torch

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import tensorflow as tf
import numpy as np

from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray

# Evil global variables
_tgp_np_debug = 1
_tgp_delimiter = os.path.sep
_tgp_subdir = "runs"

_domain = [-1.0, 1.0]
_domain_delta = _domain[1] - _domain[0]

_codomain = [0.0, 1.0]
_codomain_delta = _codomain[1] - _codomain[0]

_final_transform = [0.0, 255.0]
_final_transform_delta = _final_transform[1] - _final_transform[0]
_do_final_transform = False

#_tf_type = tf.float32
dtype = torch.float32
torch_dtype_min = torch.finfo(dtype).min
torch_dtype_max = torch.finfo(dtype).max
cur_dev = 'cpu'

warp_batch = 4
save_seed_file = "seed.obj"

# np debug options
if _tgp_np_debug:
    np.set_printoptions(linewidth=1000)
    np.set_printoptions(precision=3)

# torch has no astype
TORCH_DTYPES = {
    'torch.float32': torch.float32,
    'torch.float64': torch.float64,
    'torch.float': torch.float,
    'torch.int': torch.int,
    'torch.int32': torch.int32,
    'torch.int64': torch.int64,
    'torch.double': torch.double,
    'torch.cdouble': torch.cdouble,
}


# taken from https://svn.blender.org/svnroot/bf-blender/trunk/blender/build_files/scons/tools/bcolors.py
class Bcolors:
    def __init__(self):
        self.HEADER = '\033[95m'
        self.OKBLUE = '\033[94m'
        self.OKCYAN = '\033[96m'
        self.OKGREEN = '\033[92m'
        self.WARNING = '\033[93m'
        self.FAIL = '\033[91m'
        self.ENDC = '\033[0m'
        self.BOLD = '\033[1m'
        self.UNDERLINE = '\033[4m'
    def removecolor(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKCYAN = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''
        self.BOLD = ''
        self.UNDERLINE = ''

bcolors = Bcolors()


#============================== torch operators #==============================
def node_var(dimensions, n):
    temp = np.ones(len(dimensions), dtype=int)
    temp[n] = -1
    res = torch.reshape(torch.arange(0, dimensions[n], dtype=dtype), tuple(temp))
    resolution = dimensions[n]
    dimensions[n] = 1
    res = torch.add(torch.full(res.shape, _domain[0]), res, alpha=((1.0 / (resolution - 1)) * _domain_delta))
    return res.repeat(tuple(dimensions))

def node_abs(x1, dims=[]):
    return torch.abs(x1)

def node_add(x1, x2, dims=[]):
    return torch.add(x1, x2)

def node_sub(x1, x2, dims=[]):
    return torch.sub(x1, x2)

def node_mul(x1, x2, dims=[]):
    return torch.mul(x1, x2)

def node_div(x1, x2, dims=[]):
    return torch.where(x2 != 0.0, torch.div(x1, x2), torch.tensor(0.0, dtype=dtype, device=cur_dev))

def node_bit_and(x1, x2, dims=[]):
    left_child = torch.mul(x1, 1e6).int()
    right_child = torch.mul(x2, 1e6).int()
    return torch.mul(torch.bitwise_and(left_child, right_child), 1e-6)

def node_bit_or(x1, x2, dims=[]):
    left_child = torch.mul(x1, 1e6).int()
    right_child = torch.mul(x2, 1e6).int()
    return torch.mul(torch.bitwise_or(left_child, right_child), 1e-6)

def node_bit_xor(x1, x2, dims=[]):
    left_child = torch.mul(x1, 1e6).int()
    right_child = torch.mul(x2, 1e6).int()
    return torch.mul(torch.bitwise_xor(left_child, right_child), 1e-6)

def node_cos(x1, dims=[]):
    return torch.cos(torch.mul(x1, math.pi))

def node_sin(x1, dims=[]):
    return torch.sin(torch.mul(x1, math.pi))

def node_tan(x1, dims=[]):
    return torch.tan(torch.mul(x1, math.pi))

def node_exp(x1, dims=[]):
    return torch.exp(x1)

def node_if(x1, x2, x3, dims=[]):
    return torch.where(x3 < 0, x1, x2)

def node_log(x1, dims=[]):
    return torch.where(x1 > 0.0, torch.log(x1), torch.tensor(-1.0, dtype=dtype, device=cur_dev))

def node_max(x1, x2, dims=[]):
    return torch.max(x1, x2)

def node_min(x1, x2, dims=[]):
    return torch.min(x1, x2)

def node_mdist(x1, x2, dims=[]):
    return torch.mul(torch.add(x1, x2), 0.5)

def node_mod(x1, x2, dims=[]):
    return torch.where(x2 != 0.0, torch.fmod(x1, x2), torch.tensor(0.0, dtype=dtype, device=cur_dev))
    #return torch.fmod(x1, x2)

def node_neg(x1, dims=[]):
    return torch.neg(x1)

def node_pow(x1, x2, dims=[]):
    return torch.where(x1 != 0, torch.pow(torch.abs(x1), torch.abs(x2)), torch.tensor(0.0, dtype=dtype, device=cur_dev))


def node_sign(x1, dims=[]):
    return torch.sign(x1)

def node_sqrt(child1, dims=[]):
    return torch.where(child1 > 0, torch.sqrt(child1), torch.tensor(0.0, dtype=dtype, device=cur_dev))

def tensor_rmse(x1, x2):
    x1 = torch.mul(x1, 1/127.5)
    x2 = torch.mul(x2, 1/127.5)
    return torch.sqrt(torch.mean(torch.square(torch.sub(x1, x2))))

def node_clamp(tensor):
    return torch.clip(tensor, _codomain[0], _codomain[1])

def node_clip(child1, child2, child3, dims=[]):  # a < n < b
    return torch.min(torch.max(child1, child2), child3)

# return x * x * x * (x * (x * 6 - 15) + 10);
# simplified to x2*(6x3 - (15x2 - 10x)) to minimize operations
def node_sstepp(x1, dims=[]):
    x = node_clamp(x1)
    x2 = torch.square(x)

    return torch.add(torch.mul(x2,
                              torch.sub(torch.mul(torch.mul(x, x2), 6.0 * _domain_delta),
                                        torch.sub(torch.mul(x2, 15.0 * _domain_delta),
                                                  torch.mul(x, 10.0 * _domain_delta)))),
                     _codomain[0])

# return x * x * (3 - 2 * x);
# simplified to (6x2 - 4x3) to minimize TF operations
def node_sstep(x1, dims=[]):
    x = node_clamp(x1)
    x2 = torch.square(x)
    return torch.add(torch.sub(torch.mul(3.0 * _domain_delta, x2),
                               torch.mul(2.0 * _domain_delta, torch.mul(x2, x))),
                  torch.tensor(_codomain[0], dtype=dtype, device=cur_dev))

def node_step(x1, dims=[]):
    return torch.where(x1 < 0.0,
                    torch.tensor(-1.0, dtype=dtype, device=cur_dev),
                    torch.tensor(1.0, dtype=dtype, device=cur_dev))

def node_frac(x1, dims=[]):
    return torch.frac(x1)

def node_len(x1, x2, dims=[]):
    return torch.hypot(x1, x2)

def node_lerp(x1, x2, x3, dims=[]):
    return torch.lerp(x1, x2, node_frac(node_abs(x3)))

def node_stack(nums, dimensions, edims):
    return torch.stack([torch.full(dimensions[:edims], float(carvar), dtype=dtype, device=cur_dev) for carvar in nums], dim = edims)



## ====================== TensorFlow Node Class ====================== ##

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

    def get_tensor(self, engref, dynamic_lerp=None):
        if self.terminal:
            if self.value == 'scalar':
                args = len(self.children)
                last_dim = engref.terminal.dimension
                if args == 1 or last_dim <= 1:
                    #return tf.constant(self.children[0], tf.float32, engref.target_dims)
                    return torch.full(engref.target_dims, self.children[0], dtype=dtype)
                else:
                    extend_children = self.children + ([self.children[-1]] * (last_dim - args))
                    return node_stack(extend_children[:last_dim], engref.target_dims, engref.effective_dims)
                # both cases do the same if args == 1, this condition is here for speed concerns if
                # the last dimension is very big (e.g. 2d(1024, 1024) we don't want color in that case)

            else:
                return engref.terminal.set[self.value]
        else:
            tens_list = []
            if (self.value == 'lerp') and (dynamic_lerp is not None):
                self.children[2].children = [dynamic_lerp]
            for c in self.children:
                tens_list.append(c.get_tensor(engref, dynamic_lerp))

            return self.node_tensor(tens_list, engref)

    def fancy_print(self, acc='', dep=0):
        acc += "." * 2 * dep
        acc += str(self.value) + "\n"
        if not self.terminal:
            for c in self.children:
                acc = c.fancy_print(acc, dep + 1)
        elif self.value == 'scalar':
            for c in self.children:
                acc += "." * (2 * (dep + 1)) + str(c) + "\n"
        return acc

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

    def get_node_c(self, n, count_terms=True, count_funcs=True):
        if n == 0:
            return self, 0
        else:
            node = None
            if not self.terminal:
                i = 0
                for c in self.children:
                    node, t = c.get_node_c(n - i - count_funcs)
                    i += t
                    if node is not None:
                        break
                return node, i + count_funcs
            else:
                return None, count_terms

    def get_node_c1(self, n, count_terms=True, count_funcs=True):
        if n == 0:
            return self, 0
        else:
            node = None
            if not self.terminal:
                i = 0
                for c in self.children:
                    node, t = c.get_node_c(n - i - count_funcs)
                    i += t
                    if node is not None:
                        break
                return node, i + count_funcs
            else:
                return None, count_terms

    def debug_print_node(self):
        print("\nFancy node print")
        print(self.fancy_print())
        d, n = self.get_depth()
        for i in range(n):
            res_node, _ = self.get_node_c(i)
            print("subtree w/ node count " + str(i) + " :", res_node.get_str())


## ====================== Utility methods ====================== ##

def interactive_pause(key_str='space'):
    while True:
        if keyboard.read_key() == key_str:
            break

def constrain(a, n, b):
    return min(max(n, a), b)


def get_func_name(func, default="Not callable"):
    return func.__name__ if callable(func) else default


# Map from codomain range to a specified final transform range
def get_final_transform(tensor, ft_delta, ft_min):
    return (tensor - _codomain[0]) * (ft_delta / _codomain_delta) + ft_min


def get_np_array(tensor):
    # check if tensor is in image range (0..255)
    if ((not ((not _do_final_transform) and _codomain[0] == 0.0 and _codomain[1] == 255.0)) and (not (
            _do_final_transform and _final_transform[0] == 0.0 and _final_transform[1] == 255.0))):
        tensor = get_final_transform(tensor, 255.0, 0.0)
    return np.array(tensor.cpu(), dtype='uint8')


def save_image(tensor, index, fn, dims, sufix='', extension=".png", BGR=False):  # expects [min_domain, max_domain]

    if extension.strip(".") not in ["png", "jpg", "jpeg"]:
        extension = ".png"
    path = fn + "_ind" + str(index).zfill(5) + sufix + extension
    # print(path)
    aux = get_np_array(tensor)

    # print()
    # print("ft0", _final_transform[0])
    # print("ft1", _final_transform[1])

    # aux = np.array(tensor, dtype='uint8')

    try:
        if len(dims) == 2:
            #os.makedirs(path, exist_ok=True)
            Image.fromarray(aux, mode="L").save(path)  # no color
        elif len(dims) == 3:
            if BGR: aux = aux[:, :, ::-1]
            #os.makedirs(path, exist_ok=True)
            Image.fromarray(aux, mode="RGB").save(path)  # color
        else:
            print("Attempting to save tensor with rank ", len(dims),
                  " as an image, must be rank 2 (grayscale) or 3 (RGB).")
    except ValueError:
        print(bcolors.FAIL + "[ERROR]:\tWrong rank in tensor" + bcolors.ENDC)
    return path


# Wrapper for different expression types and strip preprocessing
def str_to_tree(stree, terminal_set, constrain_domain=True):
    return str_to_tree_normal(stree.replace(" ", "") + "", terminal_set, 0, constrain_domain)


def str_to_tree_normal(stree, terminal_set, number_nodes=0, constrain_domain=True):
    if stree in terminal_set:
        return number_nodes, Node(value=stree, terminal=True, children=[])
    elif stree[:6] == 'scalar':
        numbers = [(constrain(_codomain[0], float(x), _codomain[1]) if constrain_domain else float(x)) for x in
                   re.split('\(|\)|,', stree)[1:-1]]
        # numbers = [float(x) for x in re.split('\(|\)|,', stree)[1:-1]]
        return number_nodes, Node(value='scalar', terminal=True, children=numbers)
    else:
        x = stree[:-1].split("(", 1)
        primitive = x[0]
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
        # if primitive == "if":
        #    children = [children[1], children[2], children[0]]
        return number_nodes + 1, Node(value=primitive, terminal=False, children=children)

def set_device(device = 'gpu', debug_lvl = 1):
    cuda_build = torch.has_cuda
    gpus_available = []
    if cuda_build:
        cuda_build = cuda_build and torch.cuda.is_available()
        gpus_available = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

    # because these won't give an error but will be unspecified
    if (device is None) or (device == '') or (device == ':'):
        device = 'cpu'
        torch.set_default_device('cpu')
    try:
        result_device = torch.device('cuda' if (device == 'gpu' or device == 'cuda') and cuda_build else 'cpu')

        #print("Device to try: ", result_device)
        a = torch.full((), 2, dtype=dtype, device=result_device)
        if debug_lvl > 0:
            if a == 2:
                print(bcolors.OKGREEN + "Device " + result_device.__str__() + " successfully tested, using this device. " , bcolors.ENDC)
            else:
                print(bcolors.FAIL + "Device " + result_device.__str__() + " not working." , bcolors.ENDC)
        torch.set_default_device('cuda')
    except RuntimeError or ValueError:
        if cuda_build and gpus_available > 0:
            result_device = 'cuda'
            print(bcolors.WARNING + "[WARNING]:\tCould not find the specified device, reverting to GPU." , bcolors.ENDC)
        else:
            result_device = 'cpu'
            print(bcolors.WARNING + "[WARNING]:\tCould not find the specified device, reverting to CPU." , bcolors.ENDC)
    return result_device


# debug help for the config file
def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, "__module__"):
        return str(obj)
    else:
        raise TypeError("Unserializable object {} of type {}".format(obj, type(obj)))


def load_engine(fitness_func=None, var_func=None, mutation_funcs=None, pop_source=None,
                file_path='', file_name='state.log', seed_file_name = None):
    config = configparser.RawConfigParser()
    fn = file_path + file_name
    config.read(fn)
    conf_dict = dict(config.items('state'))

    # print(conf_dict)

    kwargs = {}
    for k, v in conf_dict.items():

        if k[0] != '_':
            if k == 'tf_type':
                #kwargs[k] = tf.as_dtype(v.split("'")[1])
                kwargs[k] = TORCH_DTYPES[v.split("'")[0]]
            elif v[0] == "{":
                kwargs[k] = set(v[1:-1].replace("'", "").replace(" ", "").split(","))
            else:
                try:
                    kwargs[k] = ast.literal_eval(v)

                except (ValueError, SyntaxError):
                    kwargs[k] = v
                except:
                    print(bcolors.FAIL + "[ERROR]:\tCould not read argument with key, value: " + str(k) + ", " + str(
                        v) + "." + bcolors.END)
                    raise


    gen_num =  int(conf_dict['_current_generation'])
    print(bcolors.WARNING + "Loading engine with timestamp: " +
          datetime.datetime.fromtimestamp(int(float(conf_dict['_last_engine_time']))).strftime(
              '%Y-%m-%d %H:%M:%S') + bcolors.ENDC)
    print(bcolors.WARNING + "Starting from generation: " + str(gen_num) + bcolors.ENDC)

    #gen_str = str(conf_dict['_last_pop']).replace('\\', "\\\\")
    gen_str = str(conf_dict['_last_pop'])
    work_dir = str(conf_dict['_work_dir'])
    #print("Gen str", gen_str)
    #print("os spe", os.path.sep)
    if pop_source is None:
        if gen_str is not None:
            kwargs['read_init_pop_from_source'] = gen_str
            print(bcolors.WARNING + "Starting from population found in file: " + gen_str + bcolors.ENDC)
        else:
            last_source = kwargs['read_init_pop_from_source']
            print("[WARNING]:\tNew population source is None, reading population from last engine source: " +
                  ("Random inititialization" if last_source is None else str(last_source)) + ".")
    else: kwargs['read_init_pop_from_source'] = pop_source

    # load seed
    flag_seed = True
    ss = None
    try:
        sfp = (work_dir + save_seed_file) if seed_file_name is None else seed_file_name
        with open(sfp, 'rb') as pickle_seed_file:
            ss = pickle.load(pickle_seed_file)
            print(bcolors.WARNING + "Loading seed from file: " + sfp + bcolors.ENDC)

    except (FileNotFoundError, TypeError):
        #kwargs['seed_state'] = None
        flag_seed = False
    if not flag_seed:
        print(bcolors.WARNING + "Did not find suitable seed file, randomizing seed: " + bcolors.ENDC)
        kwargs['seed'] = random.randint(0, 0x7fffffff)

    # print(kwargs)
    engine = Engine(fitness_func=fitness_func,
                    var_func=var_func,
                    mutation_funcs=mutation_funcs,
                    seed_state=ss,
                    **kwargs)
    engine.current_generation = gen_num
    engine.save_state = int(conf_dict['_save_state']) + 1

    return engine


# TODO: should this be an inner class of Engine()?
class Experiment:

    def set_experiment_filename(self, fixed_path=False, addon=None):
        if fixed_path is None:
            date = datetime.datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S_%f')[:-3]
            experiment_filename = (addon if addon is not None else "") + "__run__" + date + "__" + str(self.ID)
        else:
            experiment_filename = str(fixed_path)

        return experiment_filename

    def set_generation_directory(self, generation, can_save_image_pop):
        try:
            self.cur_image_directory = self.all_directory + "generation_" + str(generation).zfill(5) + _tgp_delimiter
            if can_save_image_pop():
                os.makedirs(self.cur_image_directory, exist_ok=True)
            # print("[DEBUG]:\tSet current directory to: " + self.current_directory)
        except OSError as error:
            if error is FileExistsError:
                print(bcolors.WARNING + "[WARNING]:\tExperiment directory of generation " + str(
                    generation) + " already exists, saving files to current directory." + bcolors.ENDC)
            elif error is PermissionError:
                print(bcolors.WARNING + "[WARNING]:\tPermission denied while creating directory of generation: " + str(
                    generation) + "." + bcolors.ENDC)
            else:
                print(bcolors.WARNING + "[WARNING]:\tOSError while creating directory of generation: " + str(
                    generation) + "." + bcolors.ENDC)
            print(
                bcolors.WARNING + "[WARNING]:\tReverting current directory to general image directory." + bcolors.ENDC)
            self.cur_image_directory = self.image_directory

    def set_experiment_ID(self):
        return int(time.time() * 1000.0) << 16

    def summary(self):
        summary_str = "Experiment ID: " + str(self.ID) + "\n"
        summary_str += "Experiment filename: " + self.filename + "\n"
        summary_str += "Experiment directories: \n"
        summary_str += "Current dir: " + str(self.current_directory) + "\n"
        summary_str += "Image dir: " + str(self.image_directory) + "\n"
        summary_str += "Current image dir: " + str(self.cur_image_directory) + "\n"
        summary_str += "Bests dir: " + str(self.bests_directory) + "\n"
        summary_str += "Bests dir: " + str(self.best_overall_directory) + "\n"
        summary_str += "All dir: " + str(self.all_directory) + "\n"
        summary_str += "Immigration dir: " + str(self.immigration_directory) + "\n"
        summary_str += "Logging dir: " + str(self.logging_directory) + "\n"
        summary_str += "Generations dir: " + str(self.generations_directory) + "\n"
        summary_str += "Graphics dir: " + str(self.graphs_directory) + "\n"
        summary_str += "Experiment file pointers: \n"
        summary_str += "Overall fp: " + str(self.overall_fp) + "\n"
        summary_str += "Timings fp: " + str(self.timings_fp) + "\n"
        summary_str += "Setup fp: " + str(self.setup_fp) + "\n"
        summary_str += "Best (gen) fp: " + str(self.bests_fp) + "\n"
        summary_str += "Best (overall) fp: " + str(self.bests_overall_fp) + "\n"
        return summary_str

    def __init__(self,
                 sub_wd=_tgp_delimiter + _tgp_subdir + _tgp_delimiter,
                 immigration=None,
                 seed=None,
                 wd=None,
                 fixed = False,
                 addon=None,
                 best_overall_dir=False):

        self.ID = self.set_experiment_ID()
        self.seed = self.ID if (seed is None) else seed
        self.filename = self.set_experiment_filename(addon=addon, fixed_path=fixed)
        self.prefix = addon

        try:
            self.working_directory = (os.getcwd() + sub_wd + addon + _tgp_delimiter + self.filename + _tgp_delimiter) if wd is None else wd
            os.makedirs(self.working_directory, exist_ok=True)

        except OSError as error:
            if error is FileExistsError:
                print(
                    bcolors.WARNING + "[WARNING]:\tExperiment directory already exists, saving files to current directory.",
                    bcolors.ENDC)
            elif error is PermissionError:
                print(bcolors.WARNING + "[WARNING]:\tPermission denied while creating directory" + bcolors.ENDC)
            else:
                print(bcolors.WARNING + "[WARNING]:\tOSError while creating directory" + bcolors.ENDC)
            print(bcolors.WARNING + "[WARNING]:\tFilename: " + self.working_directory + bcolors.ENDC)
            self.working_directory = os.getcwd()

        # New Filesystem
        self.current_directory = self.working_directory
        self.image_directory = self.working_directory + "images" + _tgp_delimiter
        self.cur_image_directory = self.image_directory
        self.bests_directory = self.image_directory + "bests" + _tgp_delimiter
        self.best_overall_directory = self.working_directory if not best_overall_dir else (self.image_directory + "bests_overall" + _tgp_delimiter) # for many runs, e.g. tgpgan

        self.all_directory = self.image_directory + "all" + _tgp_delimiter
        # TODO: transform immigration into archive
        self.immigration_directory = (
                self.image_directory + "immigration" + _tgp_delimiter) if immigration is not None else os.getcwd()
        self.logging_directory = self.working_directory + "logs" + _tgp_delimiter
        self.generations_directory = self.logging_directory + "generations" + _tgp_delimiter
        self.graphs_directory = self.working_directory
        try:

            os.makedirs(self.current_directory, exist_ok=True)
            os.makedirs(self.image_directory, exist_ok=True)
            if best_overall_dir:
                os.makedirs(self.best_overall_directory, exist_ok=True)
            os.makedirs(self.cur_image_directory, exist_ok=True)
            os.makedirs(self.bests_directory, exist_ok=True)
            os.makedirs(self.all_directory, exist_ok=True)
            if immigration is not None:
                os.makedirs(self.immigration_directory)
            os.makedirs(self.logging_directory, exist_ok=True)
            os.makedirs(self.generations_directory, exist_ok=True)
            os.makedirs(self.graphs_directory, exist_ok=True)

        except OSError as error:
            if error is PermissionError:
                print(bcolors.WARNING + "[WARNING]:\tPermission denied while creating experiment subdirectories.",
                      bcolors.ENDC)
            else:
                print(bcolors.WARNING + "[WARNING]:\tOSError while creating experiment subdirectories." + bcolors.ENDC)

        overall_extension = "csv"
        overall_suffix = str(self.filename) + "." + overall_extension
        self.overall_fp = self.working_directory + "evolution_" + overall_suffix
        self.timings_fp = self.logging_directory + "timings_" + overall_suffix
        self.setup_fp = self.working_directory + "seput.txt"
        self.bests_fp = self.logging_directory + "bests.csv"
        self.bests_overall_fp = self.logging_directory + "bests_overall.csv"


## ====================== Methods of an individual ====================== ##

def has_illegal_parents(tree):
    for p in tree['parents']:
        if not p['valid']:
            return False
    return True


def get_largest_parent(tree, depth=True):
    sizep = -1
    for p in tree['parents']:
        tsize = p['depth'] if depth else p['nodes']
        sizep = max(sizep, tsize)
    return sizep


# NOTE: Placeholder for weights for later implementation with Adam Optimmizer
def new_individual(tree, fitness=0, depth=0, nodes=0, tensor=[], valid=True, parents=[], weights=[]):
    return {'tree': tree, 'fitness': fitness, 'depth': depth, 'nodes': nodes, 'tensor': tensor, 'valid': valid,
            'parents': parents, 'weights': weights}


def get_ind_str(ind, fancy_print=False, stats=False, limit_fancy_print=1000):
    res = ind['tree'].fancy_print() if (fancy_print and ind['nodes'] < limit_fancy_print) else ind['tree'].get_str()
    if stats:
        res += "\n"
        res += "Fitness: " + str(ind['fitness']) + "\n"
        res += "Depth: " + str(ind['depth']) + "\n"
        res += "Nodes: " + str(ind['nodes']) + "\n"
        res += "Valid: " + str(ind['valid']) + "\n"
    return res + "\n"


def g_population(population, fancy_print=False, stats=False, limit_fancy_print=1000):
    res = ''
    for p in population:
        res += get_ind_str(p, fancy_print=fancy_print, stats=stats, limit_fancy_print=limit_fancy_print)
    return res


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            print("What the here? ", obj.tolist())
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def default_json(obj):
    if isinstance(obj, list):
        return [str(a) for a in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj.__dict__
    # return json.JSONEncoder.default(self, obj)




#TODO: check rest of code to use this function where it fits
def print_population(population, best=None, print_expr=False, msg=""):
    print("\nPopulation print", msg)
    for i in range(len(population)):
        to_p = "Fitness of ind " + str(i) + ": " + str(population[i]['fitness'])
        if print_expr:
            to_p += ", expr:  " + population[i]['tree'].get_str()
        print(to_p)
    if best is not None:
        to_p = "Fitness of best: " + str(best['fitness'])
        if print_expr:
            to_p += ", expr:  " + best['tree'].get_str()
        print(to_p)

## ====================== Engine ====================== ##

class Engine:

    ## ====================== genetic operators ====================== ##
    def crossover_debug(self, parent_1, parent_2, koza_rule_val=None, koza_child=None,
                        parent_1_node=None, parent_2_node=None, parent_2_child=None):
        crossover_node = None

        koza_rule_val = self.engine_rng.random() if koza_rule_val is None else koza_rule_val

        if koza_rule_val < self.koza_rule_prob and not parent_1.terminal:
            # function crossover
            if parent_1_node is None:
                parent_1_candidates = self.list_nodes(parent_1, True, add_funcs=True, add_terms=False, add_root=True)
                parent_1_chosen_node, _ = self.engine_rng.choice(parent_1_candidates)
            else:
                parent_1_chosen_node, _ = parent_1.get_node_c(parent_1_node)

            possible_children = []
            for i in range(len(parent_1_chosen_node.children)):
                if not parent_1_chosen_node.children[i].terminal:
                    possible_children.append(i)
            if possible_children != []:
                if koza_child is None:
                    crossover_node = copy.deepcopy(
                        parent_1_chosen_node.children[self.engine_rng.choice(possible_children)])
                else:
                    if len(possible_children) <= koza_child:
                        print(bcolors.FAIL + "[ERROR]: Out of bondaries for second parent 1." + bcolors.ENC)
                        parent_2_child = clamp(0, koza_child, len(possible_children) - 1)
                    crossover_node = copy.deepcopy(parent_1_chosen_node.children[koza_child])
            else:
                crossover_node = copy.deepcopy(parent_1_chosen_node)

        else:
            if parent_1_node is None:
                parent_1_terminals = self.get_terminals(parent_1)
                crossover_node = copy.deepcopy(self.engine_rng.choice(list(parent_1_terminals.elements())))
            else:
                parent_1_chosen_node, _ = parent_1.get_node_c(parent_1_node)
                crossover_node = copy.deepcopy(parent_1_chosen_node)

        if crossover_node is None:
            print(bcolors.FAIL + "[ERROR]: Did not select a crossover node." + bcolors.ENC)
        new_ind = copy.deepcopy(parent_2)

        if parent_2_node is None:
            parent_2_candidates = self.list_nodes(new_ind, root=True, add_funcs=True, add_terms=False, add_root=True)

            if len(parent_2_candidates) > 0:
                parent_2_chosen_node, _ = self.engine_rng.choice(parent_2_candidates)
                len_parent_2_children = len(parent_2_chosen_node.children)
                if not parent_2_chosen_node.terminal and len_parent_2_children > 0:
                    rand_child = self.engine_rng.randint(0, len_parent_2_children - 1)
                    parent_2_chosen_node.children[rand_child] = crossover_node
                else:
                    new_ind = crossover_node
            else:
                new_ind = crossover_node
        else:
            parent_2_chosen_node, _ = parent_2.get_node_c(parent_2_node)
            len_parent_2_children = len(parent_2_chosen_node.children)
            if not parent_2_chosen_node.terminal and len_parent_2_children > 0:
                if parent_2_child is None:
                    rand_child = self.engine_rng.randint(0, len_parent_2_children - 1)
                    parent_2_chosen_node.children[rand_child] = crossover_node
                else:
                    if len_parent_2_children <= parent_2_child:
                        print(bcolors.FAIL + "[ERROR]: Out of bondaries for second parent 2." + bcolors.ENC)
                        parent_2_child = clamp(0, parent_2_child, len_parent_2_children - 1)
                    parent_2_chosen_node.children[parent_2_child] = crossover_node
            else:
                new_ind = crossover_node

        return new_ind


    def crossover(self, parent_1, parent_2):
        crossover_node = None

        if self.engine_rng.random() < self.koza_rule_prob and not parent_1.terminal:
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
            crossover_node = copy.deepcopy(self.engine_rng.choice(list(parent_1_terminals.elements())))

        if crossover_node is None:
            print(bcolors.FAIL + "[ERROR]: Did not select a crossover node." + bcolors.ENC)
        new_ind = copy.deepcopy(parent_2)
        parent_2_candidates = self.list_nodes(new_ind, root=True, add_funcs=True, add_terms=False, add_root=True)

        if len(parent_2_candidates) > 0:
            parent_2_chosen_node, _ = self.engine_rng.choice(parent_2_candidates)
            if not parent_2_chosen_node.terminal and len(parent_2_chosen_node.children) > 0:
                rand_child = self.engine_rng.randint(0, len(parent_2_chosen_node.children) - 1)
                parent_2_chosen_node.children[rand_child] = crossover_node
            else:
                new_ind = crossover_node
        else:
            new_ind = crossover_node

        return new_ind


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


    def random_terminal(self):
        _, node = self.generate_program(method='full', max_nodes=0, max_depth=0)
        return node


    def copy_node(self, n):
        return Node(value=n.value, terminal=n.terminal, children=[] + n.children)


    def delete_mutation(self, parent):
        new_ind = copy.deepcopy(parent)

        # every node except last depth (terminals)
        candidates = self.list_nodes(new_ind, root=True, add_funcs=True, add_terms=False, add_root=False)
        if not new_ind.terminal: candidates.append((new_ind, 0))

        if len(candidates) > 0:
            chosen_node, _ = self.engine_rng.choice(candidates)  # parent = root
            # chosen_node = new_ind

            # random child of chosen
            chosen_child = chosen_node.children[self.engine_rng.randint(0, len(chosen_node.children) - 1)]
            # new_ind = self.copy_node(chosen_child)

            chosen_node.value = chosen_child.value
            chosen_node.terminal = chosen_child.terminal
            chosen_node.children = chosen_child.children  # does not need []

        return new_ind


    def insert_mutation(self, parent):
        new_ind = copy.deepcopy(parent)
        # print("[DEBUG D] Before:\t" + new_ind.get_str())

        # every node except last depth (terminals)
        candidates = self.list_nodes(new_ind, root=True, add_funcs=True, add_terms=False, add_root=True)

        if len(candidates) > 1 or (len(candidates) == 1 and not candidates[0][0].terminal):
            chosen_node, _ = self.engine_rng.choice(candidates)
            # print(new_ind.get_str())

            # Insert node between choosen and choosen's child
            # random child of chosen
            # second part of if, because in future there can be funcs with no arguments
            if not chosen_node.terminal and len(chosen_node.children) > 0:
                chosen_child = chosen_node.children[self.engine_rng.randint(0, len(chosen_node.children) - 1)]
            else:
                chosen_child = new_ind
        else:
            chosen_child = new_ind

        _v = chosen_child.value
        _t = chosen_child.terminal
        _c = chosen_child.children
        child_temp = self.copy_node(chosen_child)

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
        return new_ind


    def list_nodes(self, node, dep=0, root=False, add_funcs=True, add_terms=True, add_root=False):
        res = []
        if (node.terminal and (add_terms or (root and add_root))) or (
                (not node.terminal) and add_funcs and ((not root) or add_root)):
            res.append((node, dep))
        if not node.terminal:
            for c in node.children:
                res += self.list_nodes(c, dep + 1, False, add_funcs, add_terms, add_root)
        return res


    # This is the same , except it does not go back to the loop
    def hacky_subtree_mutation(self, parent):
        new_ind = copy.deepcopy(parent)
        # print("\nindiv, ", new_ind.get_str())
        candidates = self.list_nodes(new_ind, root=True, add_funcs=True, add_terms=False, add_root=True)

        chosen_node, chosen_dep = self.engine_rng.choice(candidates)
        _max_dep = max((self.max_tree_depth - chosen_dep) - 1, 0)
        _max_dep = min(_max_dep, self.max_subtree_dep)
        _min_dep = min(_max_dep, self.min_subtree_dep)

        _, mutation_node = self.generate_program('grow', -1, max_depth=_max_dep, min_depth=_min_dep, root=True)
        # print("mutation_node \n", mutation_node.fancy_print())
        # print()

        if not chosen_node.terminal:
            chosen_node.children[self.engine_rng.randint(0, len(chosen_node.children) - 1)] = mutation_node
        else:  # means its root in this case
            new_ind = mutation_node
        return new_ind


    def subtree_mutation(self, parent):
        new_ind = copy.deepcopy(parent)
        # candidates = self.get_candidates(new_ind, True)
        # print("\nindiv, ", new_ind.get_str())
        candidates = self.list_nodes(new_ind, root=True, add_funcs=True, add_terms=False, add_root=True)

        chosen_node, _ = self.engine_rng.choice(candidates)
        _, mutation_node = self.generate_program('grow', -1, max_depth=self.max_subtree_dep,
                                                 min_depth=self.min_subtree_dep, root=True)

        if not chosen_node.terminal:
            chosen_node.children[self.engine_rng.randint(0, len(chosen_node.children) - 1)] = mutation_node
        else:  # means its root in this case
            new_ind = mutation_node
        return new_ind


    # TODO: Support for genetic operators per node
    def replace_nodes(self, node):
        if node.terminal:
            if self.engine_rng.random() < self.scalar_prob:
                node.value = 'scalar'
                # node.children = [self.engine_rng.uniform(0, 1)] # no color
                # color
                if self.engine_rng.random() < self.uniform_scalar_prob:
                    node.children = [self.engine_rng.uniform(self.erc_min, self.erc_max)] * self.terminal.dimension
                else:
                    node.children = [self.engine_rng.uniform(self.erc_min, self.erc_max) for i in
                                     range(self.terminal.dimension)]
            else:
                if node.value == 'scalar':
                    node.value = self.engine_rng.choice(list(self.terminal.set))
                else:
                    temp_tset = self.terminal.set.copy()
                    del temp_tset[node.value]
                    node.value = self.engine_rng.choice(list(temp_tset))
        else:

            this_arity = self.function.set[node.value][0]
            arity_to_search = self.engine_rng.choice(
                list(self.function.arity.keys())) if self.replace_mode == 'dynamic_arities' else this_arity

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
        new_ind = copy.deepcopy(parent)
        candidates = self.list_nodes(new_ind, root=True, add_funcs=True, add_terms=True, add_root=True)
        chosen_node, _ = self.engine_rng.choice(candidates)
        self.replace_nodes(chosen_node)
        return new_ind


    ### needs to be passed
    def __init__(self,
                 fitness_func=None,  ###
                 population_size=100,
                 tournament_size=3,
                 mutation_rate=0.15,
                 mutation_funcs=None,
                 mutation_probs=None,
                 crossover_rate=0.9,
                 elitism=1,
                 min_tree_depth=-1,
                 max_tree_depth=8,
                 max_init_depth=None,
                 min_init_depth=None,
                 max_subtree_dep=None,
                 min_subtree_dep=None,
                 method='ramped half-and-half',
                 terminal_prob=0.2,
                 scalar_prob=0.55,
                 uniform_scalar_prob=0.5,
                 max_retries=5,
                 koza_rule_prob=0.9,
                 stop_criteria='generation',
                 stop_value=10,
                 objective='minimizing',
                 domain=None,
                 codomain=None,
                 final_transform=None,
                 do_final_transform=False,
                 fixed_path=None,

                 bloat_control='off',
                 bloat_mode='depth',
                 dynamic_limit=8,
                 min_overall_size=None,
                 max_overall_size=None,
                 lock_dynamic_limit=False,

                 domain_mode='clip',
                 replace_mode='dynamic_arities',
                 replace_prob=0.05,
                 const_range=None,
                 effective_dims=None,
                 operators=None,
                 function_set=None,  ###
                 terminal_set=None,  ###
                 immigration=float('inf'),
                 target_dims=None,
                 target=None,
                 max_nodes=-1,
                 seed=None,
                 seed_state=None,
                 debug=0,
                 save_graphics=True,
                 show_graphics=False,
                 save_image_best=True,
                 save_image_pop=True,
                 save_to_file=10,
                 save_to_file_image=None,
                 save_to_file_log=None,
                 save_to_file_state=None,
                 save_bests=True,
                 save_bests_overall=True,

                 exp_prefix='',
                 device='/cpu:0',
                 do_bgr=False,
                 interface=False,

                 polar_coordinates=False,
                 do_polar_mask=True,
                 polar_mask_value=None,

                 image_extension=None,
                 graphic_extension=None,
                 minimal_print=False,

                 save_log=True,
                 write_engine_state=True,

                 last_init_time=None,
                 last_fitness_time=None,
                 last_tensor_time=None,
                 last_engine_time=None,
                 tf_type=torch.float32,

                 color_print=True,
                 initial_test_device=True,
                 var_func=None, #
                 reeval_elite = False,
                 best_overall_dir = False,
                 stats_file_path=None,
                 graphics_file_path=None,
                 pop_file_path=None,
                 run_dir_path=None,
                 reeval_fitness_start=True,
                 start_fitness_file=None,
                 read_init_pop_from_file=None,  # to be deprecated
                 read_init_pop_from_source=None):

        # start timers
        self.last_engine_time = time.time()
        start_init = self.last_engine_time
        self.elapsed_init_time = 0 if last_init_time is None else last_init_time
        self.elapsed_fitness_time = 0 if last_fitness_time is None else last_fitness_time
        self.elapsed_tensor_time = 0 if last_tensor_time is None else last_tensor_time
        self.elapsed_engine_time = 0 if last_engine_time is None else last_engine_time

        # define style
        self.colored_print = color_print
        if not color_print: bcolors.removecolor()

        # check for fitness func
        self.fitness_func = fitness_func
        if debug > 0:
            if self.fitness_func is None:
                print(bcolors.WARNING + "[WARNING]:\tFitness function not defined!" + bcolors.ENDC)
            elif not callable(self.fitness_func):
                print(bcolors.WARNING + "[WARNING]:\tFitness function is not callable!" + bcolors.ENDC)

        # TODO: read configuration if present

        # optional vars
        self.minimal_print = minimal_print
        self.recent_fitness_time = 0
        self.recent_tensor_time = 0
        self.recent_engine_time = 0
        self.population_size = population_size
        self.tournament_size = min(tournament_size, self.population_size)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = constrain(0, elitism, self.population_size)
        self.koza_rule_prob = koza_rule_prob
        self.stop_criteria = stop_criteria
        self.save_graphics = save_graphics
        self.show_graphics = show_graphics
        self.do_bgr = do_bgr
        self.reeval_elite = reeval_elite

        if bloat_control not in ['very heavy', 'heavy', 'weak']:  # add full_dynamic_size, dynamic_size
            bloat_control = 'off'
        self.bloat_control = bloat_control

        self.bloat_mode = 'size' if bloat_mode == 'size' else 'depth'
        if domain_mode not in ['log', 'dynamic', 'mod']:  # add full_dynamic_size, dynamic_size
            domain_mode = 'clip'
        self.domain_mode = domain_mode
        self.immigration = immigration
        self.debug = debug
        self.save_to_file = save_to_file
        self.save_to_file_image = save_to_file if save_to_file_image is None else save_to_file_image
        self.save_to_file_log = save_to_file if save_to_file_log is None else save_to_file_log
        self.save_to_file_state = save_to_file if save_to_file_state is None else save_to_file_state
        self.max_nodes = max_nodes
        self.objective = objective
        self.terminal_prob = terminal_prob
        self.scalar_prob = scalar_prob
        self.uniform_scalar_prob = uniform_scalar_prob
        self.max_retries = max_retries if max_retries != 0 else 10

        if self.bloat_control in ['very heavy', 'heavy', 'weak']:
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

        self.dynamic_limit = min(dynamic_limit, self.max_tree_depth)
        self.initial_dynamic_limit = self.dynamic_limit
        max_overall_dynamic_limit = 50 if self.bloat_mode == 'depth' else 2147483647
        self.min_overall_size = self.min_tree_depth if min_overall_size is None else clamp(0, min_overall_size,
                                                                                           max_overall_dynamic_limit)
        self.max_overall_size = self.max_tree_depth if max_overall_size is None else clamp(0, max_overall_size,
                                                                                           max_overall_dynamic_limit)
        if self.min_overall_size > self.max_overall_size: self.min_overall_size, self.max_overall_size = self.max_overall_size, self.min_overall_size
        self.lock_dynamic_limit = lock_dynamic_limit

        self.stats_file_path = stats_file_path
        self.graphics_file_path = graphics_file_path
        self.pop_file_path = pop_file_path
        self.run_dir_path = run_dir_path
        self.target_dims = [128, 128] if (target_dims is None) else target_dims
        self.dimensionality = len(self.target_dims)
        self.effective_dims = self.dimensionality if effective_dims is None else effective_dims
        self.initial_test_device = initial_test_device
        self.device = set_device(device=device) if self.initial_test_device else device  # Check for available devices

        self.fixed_path = fixed_path
        self.experiment = Experiment(seed=seed, wd=self.run_dir_path, addon=str(exp_prefix), fixed=self.fixed_path, best_overall_dir=best_overall_dir)
        self.flag_file =  self.experiment.all_directory + "_flag_to_evolve"
        self.interface = interface
        self.active_interface = False

        # self.engine_rng = random.Random(self.experiment.seed) if seed_state is None else random.Random().setstate(seed_state)
        # random.Random().setstate(seed_state) does not work
        self.engine_rng = random.Random(self.experiment.seed)
        if seed_state is not None:
            self.engine_rng.setstate(seed_state)

        #print("setting seed state: ", seed_state)
        #print("Type of engine RNG: ", type(self.engine_rng))

        self.method = method if (method in ['ramped half-and-half', 'grow', 'full']) else 'ramped half-and-half'
        self.replace_mode = replace_mode if replace_mode == 'dynamic_arities' else 'same_arity'
        self.image_extension = '.' + (image_extension if (image_extension in ['png', 'jpeg', 'bmp', 'jpg']) else 'png')
        self.graphic_extension = '.' + (
            graphic_extension if (graphic_extension in ['pdf', 'png', 'jpg', 'jpeg']) else 'pdf')
        self.replace_prob = max(0.0, min(1.0, replace_prob))
        self.pop_source = read_init_pop_from_file if read_init_pop_from_file is not None else read_init_pop_from_source
        self.start_fitness_file = start_fitness_file
        self.reeval_fitness_start = reeval_fitness_start
        self.save_log = save_log
        self.write_engine_state = write_engine_state
        self.save_image_best = save_image_best
        self.save_bests = save_bests
        self.save_bests_overall = save_bests_overall
        self.save_image_pop = save_image_pop
        self.save_state = 0
        self.last_stop = 0

        if mutation_funcs is None or mutation_funcs == []:
            mut_funcs_implemented = 4
            self.mutation_funcs = [Engine.subtree_mutation, Engine.point_mutation, Engine.delete_mutation,
                                   Engine.insert_mutation]
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
        global _domain_delta, _domain
        global _codomain_delta, _codomain
        global _final_transform, _final_transform_delta, _do_final_transform
        global _tf_type

        # check for tf data type validity
        self.tf_type = tf_type if (hasattr(tf_type, "__module__") and (
                ('torch' and 'dtypes') in tf_type.__module__.split("."))) else torch.float32
        _tf_type = self.tf_type

        # set domain
        if (domain is not None) and isinstance(domain, list):
            _domain = [float(domain[0]), float(domain[1])]
        _domain_delta = _domain[1] - _domain[0]

        # set codomain
        if (codomain is not None) and isinstance(codomain, list):
            _codomain = [float(codomain[0]), float(codomain[1])]
        _codomain_delta = _codomain[1] - _codomain[0]

        self.do_final_transform = do_final_transform
        _do_final_transform = self.do_final_transform

        if self.do_final_transform:
            if (final_transform is not None) and isinstance(final_transform, list):
                _final_transform = [float(final_transform[0]), float(final_transform[1])]
            _final_transform_delta = _final_transform[1] - _final_transform[0]

        # Polar coordinate stuff
        self.polar_coordinates = polar_coordinates
        self.polar_mask_value_expr = 'undefined'
        self.do_polar_mask = do_polar_mask
        self.polar_mask_value = polar_mask_value
        self.polar_mask = torch.ones(self.target_dims, dtype=torch.float32)

        if const_range is None or len(const_range) < 1:
            self.erc_min = _domain[0]
            self.erc_max = _domain[1]
        else:
            self.erc_min = max(const_range)
            self.erc_max = min(const_range)

        # define function set
        if isinstance(function_set, Function_Set):
            self.function = function_set
        else:
            self.function = Function_Set(operators, self.effective_dims, debug=self.debug)

        # define terminal set
        if var_func is None or not callable(var_func):
            self.var_func = node_var
        else:
            self.var_func = var_func
        if isinstance(terminal_set, Terminal_Set):
            self.terminal = terminal_set
            self.terminal.set = self.terminal.make_term_variables(0, self.effective_dims, self.target_dims,
                                                                  fptr=self.var_func)
        else:
            # self.terminal = Terminal_Set(self.effective_dims, self.target_dims, engref=self)
            self.terminal = Terminal_Set(self.effective_dims, self.target_dims, engref=self,
                                         function_ptr_to_var_node=self.var_func)

        # print("x tensor: ", self.terminal.set['x'])
        # print("y tensor: ", self.terminal.set['y'])
        self.set_polar_mask_value()

        self.target_expr = 'undefined'
        if isinstance(target, str):
            self.target_expr = target
            # target = 'mult(scalar(127.5), ' + target + ')'
            _, tree = str_to_tree(target, self.terminal.set, constrain_domain=False)

            #with tf.device(self.device):
            final_temp = get_final_transform(tree.get_tensor(self), _final_transform_delta, _final_transform[0])
            self.target = final_temp.type(torch.float32)
                # self.target = tf.cast(tree.get_tensor(self) * 127.5, tf.float32) # cast to an int tensor
        else:
            self.target = target

        self.current_generation = 0

        # for now:
        # fitness - stop evolution if best individual is close enought to target (or given value)
        # generation - stop evolution after a given number of generations
        if self.objective == 'minimizing':
            self.condition_overall = lambda x: (x < self.best_overall['fitness'])
            self.condition_local = lambda x: (x < self.best['fitness'])
        else:
            self.condition_overall = lambda x: (x > self.best_overall['fitness'])
            self.condition_local = lambda x: (x > self.best['fitness'])

        if stop_criteria == 'fitness':  # if fitness then stop value means error
            self.stop_value = stop_value
            if self.objective == 'minimizing':
                self.condition = lambda: (self.best['fitness'] > self.stop_value)
            else:
                self.condition = lambda: (self.best['fitness'] < self.stop_value)
            self.next_condition = self.condition
        else:  # if generations then stop_value menas number of generations to evaluate
            self.stop_value = int(stop_value)
            self.condition = lambda: (self.current_generation <= self.stop_value)
            self.next_condition = lambda: (self.current_generation + 1 <= self.stop_value)

        # update timers
        self.elapsed_init_time += time.time() - start_init
        if self.debug > 0: print("Elapsed init time: ", self.elapsed_init_time)
        print(bcolors.OKGREEN + "Engine seed:" + str(self.experiment.seed) + bcolors.ENDC)
        self.update_engine_time()

        self.population = []
        self.best = {}
        self.best_overall = {}
        # print(self.get_json())

    def get_working_dir(self):
        return self.experiment.working_directory

    ## ====================== End init class ====================== ##

    def summary(self, force_print=False, print_prints=False, ind_fancy_print=False, ind_stats=False,
                bloat=False, trees=False, timers=False, general=False, probs=False, domain=False, graphics=False,
                extra=False, images=False, logs=False, paths=False, experiment=False, terminals=False,
                functions=False, population=False, log_format=False, write_file=False, file_path=None,
                file_name='state.log', pop_path = None):


        summary_str = ""
        if force_print:
            if log_format:
                summary_str += "[state]\n"
            else:
                summary_str += "Engine summary\n"
        if print_prints and not log_format:
            summary_str += "\n############# Begin summary information #############\n"
            summary_str += "Force print: " + str(force_print) + "\n"
            if not force_print:
                summary_str += "Print bloat: " + str(bloat) + "\n"
                summary_str += "Print trees: " + str(trees) + "\n"
                summary_str += "Print timers: " + str(timers) + "\n"
                summary_str += "Print general: " + str(general) + "\n"
                summary_str += "Print probs: " + str(probs) + "\n"
                summary_str += "Print domain: " + str(domain) + "\n"
                summary_str += "Print saves: " + str(graphics) + "\n"
                summary_str += "Print extra: " + str(extra) + "\n"
                summary_str += "Print images: " + str(images) + "\n"
                summary_str += "Print logs: " + str(logs) + "\n"
                summary_str += "Print paths: " + str(paths) + "\n"
                summary_str += "Print experiment: " + str(experiment) + "\n"
                summary_str += "Print terminals: " + str(terminals) + "\n"
                summary_str += "Print functions: " + str(functions) + "\n"
            summary_str += "\n############# End summary information #############\n"

        if log_format:
            summary_str += "# Comments start with '#'\n"
            summary_str += "# Check the README for documentation on the arguments: https://github.com/AwardOfSky/TensorGP\n"

        if general or force_print:
            summary_str += "\n############# General Information #############\n"
            if not log_format: summary_str += "Fitness function: " + get_func_name(self.fitness_func) + "\n"
            summary_str += ("_current_generation = " if log_format else "Current generation: ") + str(
                self.current_generation) + "\n"
            summary_str += ("seed = " if log_format else "Seed: ") + str(self.experiment.seed) + "\n"
            if not self.condition() and not log_format: summary_str += "The run is over!\n"
            summary_str += ("population_size = " if log_format else "Population size: ") + str(
                self.population_size) + "\n"
            summary_str += ("tournament_size = " if log_format else "Tournament size: ") + str(
                self.tournament_size) + "\n"
            summary_str += ("elitism = " if log_format else "Elite size: ") + str(self.elitism) + "\n"
            summary_str += ("mutation_rate = " if log_format else "Mutation rate: ") + str(self.mutation_rate) + "\n"
            summary_str += ("crossover_rate = " if log_format else "Crossover rate: ") + str(self.crossover_rate) + "\n"
            summary_str += ("method = " if log_format else "Tree generation: ") + str(self.method) + "\n"
            summary_str += ("stop_criteria = " if log_format else "Stop criteria: ") + self.stop_criteria + "\n"
            summary_str += ("stop_value = " if log_format else "Stop value: ") + str(self.stop_value) + "\n"
            summary_str += ("objective = " if log_format else "Objective: ") + str(self.objective) + "\n"
            summary_str += ("target_dims = " if log_format else "Resolution: ") + str(self.target_dims) + "\n"
            summary_str += ("device = " if log_format else "Device: ") + str(self.device) + "\n"

        if trees or force_print:
            summary_str += "\n############# Tree Information #############\n"
            summary_str += ("min_init_depth = " if log_format else "Min init: ") + str(self.min_init_depth) + "\n"
            summary_str += ("max_init_depth = " if log_format else "Max init: ") + str(self.max_init_depth) + "\n"
            summary_str += ("min_tree_depth = " if log_format else "Min overall: ") + str(self.min_tree_depth) + "\n"
            summary_str += ("max_tree_depth = " if log_format else "Max overall: ") + str(self.max_tree_depth) + "\n"
            summary_str += ("min_subtree_dep = " if log_format else "Min subtree depth: ") + str(
                self.min_subtree_dep) + "\n"
            summary_str += ("max_subtree_dep = " if log_format else "Max subtree depth: ") + str(
                self.max_subtree_dep) + "\n"
            summary_str += ("max_nodes = " if log_format else "Max nodes: ") + str(self.max_nodes) + "\n"
            if log_format:
                summary_str += "const_range = [" + str(self.erc_min) + ", " + str(self.erc_max) + "]\n"
            else:
                summary_str += "ERC min value: " + str(self.erc_min) + "\n"
                summary_str += "ERC max value: " + str(self.erc_max) + "\n"

        if probs or force_print:
            summary_str += "\n############# Probabilities Information #############\n"
            summary_str += ("koza_rule_prob = " if log_format else "Koza probability rule: ") + str(
                self.koza_rule_prob) + "\n"
            summary_str += ("terminal_prob = " if log_format else "Terminal prob: ") + str(self.terminal_prob) + "\n"
            summary_str += ("scalar_prob = " if log_format else "Scalar prob: ") + str(self.scalar_prob) + "\n"
            summary_str += ("uniform_scalar_prob = " if log_format else "Uniform scalar prob: ") + str(
                self.uniform_scalar_prob) + "\n"
            summary_str += ("_mutation_funcs = " if log_format else "Mutations functions: ") + "["
            lim = len(self.mutation_funcs)
            for m in range(lim):
                summary_str += get_func_name(self.mutation_funcs[m])
                if m < lim - 1:
                    summary_str += ", "
            summary_str += "]\n"
            summary_str += ("mutation_probs = " if log_format else "Mutations probabilities: ") + "["
            lim = len(self.mutation_probs)
            for m in range(lim):
                prob_sub = 1 if m == (lim - 1) else self.mutation_probs[m + 1]
                summary_str += str(prob_sub - self.mutation_probs[m])
                #summary_str += str(self.mutation_probs[m])
                if m < lim - 1:
                    summary_str += ", "
            summary_str += "]\n"
            summary_str += ("replace_prob = " if log_format else "Replace prob (point mut): ") + str(
                self.replace_prob) + "\n"

        if domain or force_print:
            summary_str += "\n############# Inputs/Outputs/Mapping Information #############\n"
            summary_str += ("domain = " if log_format else "Domain: ") + str(_domain) + "\n"
            summary_str += ("codomain = " if log_format else "Codomain: ") + str(_codomain) + "\n"
            summary_str += ("domain_mode = " if log_format else "Domain Mapping: ") + str(self.domain_mode) + "\n"
            summary_str += ("do_final_transform = " if log_format else "Do final transform: ") + str(
                self.do_final_transform) + "\n"
            summary_str += ("final_transform = " if log_format else "Final transform: ") + str(_final_transform) + "\n"
            summary_str += ("polar_coordinates = " if log_format else "Polar coordinates: ") + str(
                self.polar_coordinates) + "\n"
            if self.polar_coordinates:
                summary_str += ("do_polar_mask = " if log_format else "Mask polar coordinates: ") + str(
                    self.do_polar_mask) + "\n"
                if not log_format: summary_str += "Polar mask" + str(self.polar_mask.numpy()) + "\n"
                if self.polar_mask_value_expr != 'undefined':  # write the expression instead
                    summary_str += ("polar_mask_value = " if log_format else "Polar mask value: ") + str(
                        self.polar_mask_value_expr) + "\n"
                else:  # write the tensor
                    summary_str += ("polar_mask_value = " if log_format else "Polar mask value: ") + str(
                        self.polar_mask_value.numpy()) + "\n"

        if terminals or force_print:
            summary_str += "\n############# Terminal set Information #############\n"
            summary_str += self.terminal.summary(log_format=log_format)

        if functions or force_print:
            summary_str += "\n############# Function set Information #############\n"
            summary_str += self.function.summary(log_format=log_format)

        if bloat or force_print:
            summary_str += "\n############# Bloat control Information #############\n"
            summary_str += ("bloat_control = " if log_format else "Bloat control: ") + self.bloat_control + "\n"
            if self.bloat_control != 'off':
                summary_str += ("bloat_mode = " if log_format else "Bloat mode: ") + self.bloat_mode + "\n"
                summary_str += ("dynamic_limit = " if log_format else "Initial dynamic limit: ") + str(
                    self.initial_dynamic_limit) + "\n"
                summary_str += ("min_overall_size = " if log_format else "Overall lower limit: ") + str(
                    self.min_overall_size) + "\n"
                summary_str += ("max_overall_size = " if log_format else "Overall upper limit: ") + str(
                    self.max_overall_size) + "\n"
                summary_str += ("lock_dynamic_limit = " if log_format else "Dynamic limit locked: ") + str(
                    self.lock_dynamic_limit) + "\n"
                if not log_format: summary_str += "Dynamic limit: " + str(self.dynamic_limit) + "\n"

        if population or (force_print and not log_format):
            summary_str += "\n############# Population Information #############\n"
            summary_str += "Best individual: \n"
            summary_str += ("Not defined" if ('tree' not in self.best) else get_ind_str(self.best,
                                                                                        fancy_print=ind_fancy_print,
                                                                                        stats=ind_stats)) + "\n"
            summary_str += "Best overall individual: \n"
            summary_str += ("Not defined" if ('tree' not in self.best_overall) else get_ind_str(self.best_overall,
                                                                                                fancy_print=ind_fancy_print,
                                                                                                stats=ind_stats)) + "\n"
            summary_str += "Population: \n"
            summary_str += g_population(self.population, fancy_print=ind_fancy_print,
                                        stats=ind_stats, limit_fancy_print=int(max(1.0, 10000 * self.population_size)))

        if graphics or force_print:
            summary_str += "\n############# Saves Information #############\n"
            summary_str += ("save_graphics = " if log_format else "Save graphics: ") + str(self.save_graphics) + "\n"
            summary_str += ("show_graphics = " if log_format else "Show graphics: ") + str(self.show_graphics) + "\n"
            summary_str += ("graphic_extension = " if log_format else "Graphics save extension: ") + str(
                self.graphic_extension) + "\n"
            summary_str += ("save_to_file = " if log_format else "Overall save state to file: ") + str(
                self.save_to_file) + "\n"

        if images or force_print:
            summary_str += "\n############# Images Information #############\n"
            summary_str += ("save_image_pop = " if log_format else "Save Population: ") + str(
                self.save_image_pop) + "\n"
            summary_str += ("save_image_best = " if log_format else "Save Bests: ") + str(self.save_image_best) + "\n"
            summary_str += ("save_to_file_image = " if log_format else "Save images n generations: ") + str(
                self.save_to_file_image) + "\n"
            summary_str += ("do_bgr = " if log_format else "Invert RGB: ") + str(self.do_bgr) + "\n"
            summary_str += ("image_extension = " if log_format else "Image save extension: ") + str(
                self.image_extension) + "\n"

        if logs or force_print:
            summary_str += "\n############# Logs Information #############\n"
            summary_str += ("save_log = " if log_format else "Save Logs: ") + str(self.save_log) + "\n"
            summary_str += ("save_to_file_log = " if log_format else "Save logs n generations: ") + str(
                self.save_to_file_log) + "\n"
            summary_str += ("write_engine_state = " if log_format else "Save state: ") + str(
                self.write_engine_state) + "\n"
            summary_str += ("save_to_file_state = " if log_format else "Save state n generations: ") + str(
                self.save_to_file_state) + "\n"
            summary_str += ("save_bests = " if log_format else "Save Bests (gen): ") + str(self.save_bests) + "\n"
            summary_str += ("save_bests_overall = " if log_format else "Save Bests (overall): ") + str(
                self.save_bests_overall) + "\n"

        if extra or force_print:
            summary_str += "\n############# Extra Information #############\n"
            summary_str += ("debug = " if log_format else "Debug level: ") + str(self.debug) + "\n"
            summary_str += ("immigration = " if log_format else "Immigration: ") + str(self.immigration) + "\n"
            summary_str += ("max_retries = " if log_format else "Max individual retries: ") + str(
                self.max_retries) + "\n"
            if not log_format: summary_str += "Dimensionality: " + str(self.dimensionality) + "\n"
            summary_str += ("effective_dims = " if log_format else "Indexable dimensions: ") + str(
                self.effective_dims) + "\n"
            summary_str += ("initial_test_device = " if log_format else "Initial device test: ") + str(
                self.initial_test_device) + "\n"
            summary_str += ("replace_mode = " if log_format else "Replace mode (point mut): ") + str(
                self.replace_mode) + "\n"
            if not log_format:
                summary_str += "Save state (first gen): " + str(self.save_state) + "\n"
                summary_str += "Last saved population: " + str(self.last_stop) + "\n"
                summary_str += "Function to generate terminal vars: " + get_func_name(self.var_func) + "\n"
            if self.target_expr != 'undefined':  # write the expression instead
                #summary_str += (("target = " + '"') if log_format else "Target tensor: ") + str(self.target_expr) + ('"' if log_format else "") + "\n"
                summary_str += ("target = " if log_format else "Target tensor: ") + str(self.target_expr) + "\n"
            else:  # write the tensor
                if hasattr(self.target, "__module__") and type(self.target).__module__.split(".", 1)[0] == 'tensorflow':
                    summary_str += ("target = " if log_format else "Target tensor: ") + str(self.target) + "\n"
                else:
                    summary_str += ("target = " if log_format else "Target tensor: ") + str(self.target) + "\n"
            summary_str += ("minimal_print = " if log_format else "Mininal printing: ") + str(self.minimal_print) + "\n"
            summary_str += ("tf_type = " if log_format else "Internal data type (tensorflow type): ") + str(
                self.tf_type) + "\n"
            summary_str += ("reeval_fitness_start = " if log_format else "Reeval Fitness Start: ") + str(self.reeval_fitness_start) + "\n"
            summary_str += ("reeval_elite = " if log_format else "Reeval Elite: ") + str(self.reeval_elite) + "\n"
            summary_str += ("_seed_state = " if log_format else "Seed State: ") + str(self.engine_rng.getstate()) + "\n"
            summary_str += ("interface = " if log_format else "Interface: ") + str(self.interface) + "\n"
            summary_str += ("fixed_path = " if log_format else "Fixed Path: ") + str(self.fixed_path) + "\n"


        if log_format: summary_str += "_save_state = " + str(self.save_state) + "\n"

        if timers or force_print:
            summary_str += "\n############# Timers Information #############\n"
            summary_str += "# You should not change this as it is information for the next run"
            summary_str += ("last_init_time = " if log_format else "Elapsed initialization time: ") + str(
                self.elapsed_init_time) + "\n"
            summary_str += ("last_fitness_time = " if log_format else "Elapsed fitness time: ") + str(
                self.elapsed_fitness_time) + "\n"
            summary_str += ("last_tensor_time = " if log_format else "Elapsed tensor time: ") + str(
                self.elapsed_tensor_time) + "\n"
            summary_str += ("last_engine_time = " if log_format else "Total engine time: ") + str(
                self.elapsed_engine_time) + "\n"
            if not log_format: summary_str += "Last engine update time: " + str(self.last_engine_time) + "\n"

        if log_format: summary_str += "_last_engine_time = " + str(self.last_engine_time) + "\n"

        summary_str += ("_work_dir = " if log_format else "Working directory: ") + str(self.experiment.working_directory) + "\n"

        if pop_path is not None:
            summary_str += ("_last_pop = " if log_format else "Current generation: ") + str(pop_path) + "\n"

        if paths or force_print:
            summary_str += "\n############# Paths Information #############\n"
            summary_str += ("run_dir_path = " if log_format else "Run directory: ") + str(self.run_dir_path) + "\n"
            summary_str += ("pop_file_path = " if log_format else "Population file: ") + str(self.pop_file_path) + "\n"
            summary_str += ("stats_file_path = " if log_format else "Stats file: ") + str(self.stats_file_path) + "\n"
            summary_str += ("graphics_file_path = " if log_format else "Graphics file: ") + str(
                self.graphics_file_path) + "\n"
            summary_str += ("read_init_pop_from_source = " if log_format else "Initial pop source: ") + str(
                self.pop_source) + "\n"
            summary_str += ("exp_prefix = " if log_format else "Experimence prefix: ") + '"' + str(
                self.experiment.prefix) + '"' + "\n"
            summary_str += ("start_fitness_file = " if log_format else "Starting Fitness File: ") + str(
                self.start_fitness_file) + "\n"

        if (experiment or force_print) and not log_format:
            summary_str += "\n############# Experiment Information #############\n"
            summary_str += self.experiment.summary()

        if write_file:
            fn = self.experiment.working_directory if file_path is None else file_path
            fn += file_name
            with open(fn, "w") as text_file:
                try:
                    text_file.write(summary_str)
                except IOError as error:
                    print(bcolors.FAIL + "[ERROR]:\tI/O error while writing engine state ({0}): {1}"
                          .format(error.errno, error.strerror), bcolors.ENDC)

        return summary_str

    def load_from_file(self, load_timers=True):
        pass

    def get_json(self):
        return json.dumps(self, default=default_json, cls=NumpyEncoder, sort_keys=True, indent=4)
        # return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def get_terminals(self, node):
        candidates = Counter()
        if node.terminal:
            candidates.update([node])
        else:
            for i in node.children:
                if i is not None:
                    candidates.update(self.get_terminals(i))
        return candidates

    def get_terminal_set(self):
        return self.terminal

    def get_function_set(self):
        return self.function

    def can_save_image_pop(self):
        return self.save_image_pop and (
                ((self.current_generation % self.save_to_file_image) == 0) or not self.next_condition())

    def can_save_image_best(self):
        return self.save_image_best and (
                ((self.current_generation % self.save_to_file_image) == 0) or not self.next_condition())

    def can_save_log(self):
        return self.save_log and ((self.current_generation % self.save_to_file_log) == 0) or not self.next_condition()

    def can_save_state(self):
        return self.write_engine_state and (
                (self.current_generation % self.save_to_file_state) == 0) or not self.next_condition()

    def restart(self, new_stop=10):
        self.last_stop = self.stop_value
        self.stop_value = self.stop_value + new_stop

    def generate_program(self, method, max_nodes, max_depth, min_depth=-1, root=True):
        terminal = False
        children = []
        # print(max_depth)

        # set primitive node
        # (max_depth * max_nodes) == 0 means if we either achieved maximum depth ( = 0) or there are no more
        # nodes allowed on this tree then we have to force the terminal placement
        # to ignore max_nodes, we can set this to -1
        # if (max_depth * max_nodes) == 0 or (
        #        (not root) and method == 'grow' and self.engine_rng.random() < (len(terminal_set) / (len(terminal_set) + len(function_set)))):
        # TODO: review ((self.max_init_depth - max_depth) >= min_depth), works if we call initialize pop with max_init_depth

        if (max_depth * max_nodes) == 0 or (
                # (not root) and
                (method == 'grow') and
                ((self.max_init_depth - max_depth) >= min_depth) and
                (self.engine_rng.random() < self.terminal_prob)
        ):
            if self.engine_rng.random() < self.scalar_prob:
                primitive = 'scalar'
                if self.engine_rng.random() < self.uniform_scalar_prob:
                    children = [self.engine_rng.uniform(self.erc_min, self.erc_max)]
                else:
                    children = [self.engine_rng.uniform(self.erc_min, self.erc_max) for i in
                                range(self.terminal.dimension)]
            else:
                # print("Lets print the current terminal set list:")
                # print(list(self.terminal.set))
                # print("")

                primitive = self.engine_rng.choice(list(self.terminal.set))

            terminal = True
        else:
            primitive = self.engine_rng.choice(list(self.function.set))
            if max_nodes > 0: max_nodes -= 1

        # set children
        if not terminal:
            for i in range(self.function.set[primitive][0]):
                max_nodes, child = self.generate_program(method, max_nodes, max_depth - 1, min_depth=min_depth,
                                                         root=False)
                children.append(child)

        return max_nodes, Node(value=primitive, terminal=terminal, children=children)


    def generate_population(self, individuals, method, max_nodes, max_depth, min_depth=-1):
        # print("Entering generate program (min, max)", min_depth, max_depth)

        if max_depth <= min_depth:
            max_depth, min_depth = min_depth, max_depth

        population = []
        pop_nodes = 0

        if not ((method == 'ramped half-and-half') and (max_depth >= 1)):
            for i in range(individuals):
                # TODO: why was this without min_depth? TEST
                _n, t = self.generate_program(method, max_nodes, max_depth, min_depth=min_depth)

                tree_nodes = (max_nodes - _n)
                pop_nodes += tree_nodes
                dep, nod = t.get_depth()
                # population.append({'tree': t, 'fitness': 0, 'depth': dep, 'nodes': nod, 'tensor': [], 'valid': True, 'parents':[]})
                population.append(new_individual(t, fitness=0, depth=dep, nodes=nod))
        else:
            # -1 means no minimun depth, but the ramped 5050 default should be 2 levels
            _min_depth = 0 if (min_depth <= -1) else min_depth

            # divisions = max_depth - (_min_depth - 1)
            divisions = max_depth - (_min_depth) + 1
            parts = math.floor(individuals / divisions)
            last_part = individuals - (divisions - 1) * parts
            # load_balance_index = (max_depth - 1) - (last_part - parts)
            load_balance_index = (max_depth + 1) - (last_part - parts)
            part_array = []

            num_parts = parts
            mfull = math.floor(num_parts / 2)

            # print("generate program (min, max)", _min_depth, max_depth + 1)
            for i in range(_min_depth, max_depth + 1):
                # print("This is i", i)

                # TODO: shouldn't i - 2 be i - min_depth? TEST or just i
                # if i - 2 == load_balance_index:
                if i == load_balance_index:
                    num_parts += 1
                    mfull += 1
                part_array.append(num_parts)
                met = 'full'
                for j in range(num_parts):

                    if j >= mfull:
                        met = 'grow'
                    # print("i: ", i, "min dep: ", _min_depth)

                    _n, t = self.generate_program(met, max_nodes, i, min_depth=min_depth)
                    tree_nodes = (max_nodes - _n)
                    # print("Tree nodes: " + str(tree_nodes))
                    pop_nodes += tree_nodes
                    dep, nod = t.get_depth()
                    # population.append({'tree': t, 'fitness': 0, 'depth': dep, 'nodes': nod, 'tensor': [], 'valid': True, 'parents':[]})
                    population.append(new_individual(t, fitness=0, depth=dep, nodes=nod))

        if len(population) != individuals:
            print(bcolors.FAIL + "[ERROR]:\tWrong number of individuals generated: " + str(len(population)) + "/" + str(
                individuals) + bcolors.ENDC)

        return pop_nodes, population


    def codomain_range(self, final_tensor):
        # 'dynamic' and 'mod' modes normmalize to 0..1
        if self.domain_mode == 'log':
            final_tensor = torch.log(torch.abs(final_tensor)) / torch.log(torch.full_like(final_tensor, 10.0, dtype=torch.float32))
        elif self.domain_mode == 'dynamic':
            final_tensor = torch.abs(final_tensor)
            final_tensor = (final_tensor / (1 + final_tensor))
        elif self.domain_mode == 'mod':
            final_tensor = torch.abs(final_tensor)
            final_tensor = final_tensor - torch.floor(final_tensor)
        return torch.clip(final_tensor, min=_codomain[0], max=_codomain[1])


    def domain_mapping(self, tensor):
        final_tensor = torch.where(torch.isnan(tensor), torch.tensor(_domain[0], dtype=dtype, device=cur_dev), tensor)
        final_tensor = torch.clip(final_tensor, min=torch_dtype_min, max=torch_dtype_max)
        # ".cuda()" is "temporary fix
        final_tensor = self.codomain_range(final_tensor).cuda()

        if self.do_polar_mask:
            final_tensor = torch.where(self.polar_mask == 1, final_tensor, self.polar_mask_value)

        if self.do_final_transform:
            final_tensor = get_final_transform(final_tensor, _final_transform_delta, _final_transform[0])

        return final_tensor


    def calculate_tensors(self, population):
        tensors = []
        start = time.time()
        for p in population:
            # print("Evaluating ind: ", p['tree'].get_str())
            # _start = time.time()
            test_tens = p['tree'].get_tensor(self)
            # tens = self.final_transform_domain(test_tens)

            tens = self.domain_mapping(test_tens)

            # tens = test_tens
            p['tensor'] = tens
            tensors.append(tens)

        time_tensor = time.time() - start

        self.elapsed_tensor_time += time_tensor
        self.recent_tensor_time = time_tensor
        return tensors, time_tensor


    # TODO: test
    def evaluate_from_expr(self, expr, resolution):
        _, node = str_to_tree(expr, self.terminal.set)
        return self.evaluate_from_tree(node, resolution)


    def evaluate_from_tree(self, ind, resolution):
        original_res = self.target_dims
        self.target_dims = resolution
        final_tensor = ind.get_tensor(self)
        self.target_dims = original_res
        return self.domain_mapping(final_tensor)


    def fitness_func_wrap(self, population, f_path):

        # calculate tensors
        if self.debug > 4: print("\nEvaluating generation: " + str(self.current_generation))
        #with tf.device(self.device):
        tensors, time_taken = self.calculate_tensors(population)
        if self.debug > 4: print("Calculated " + str(len(population)) + " tensors in (s): " + str(time_taken))

        # save pop and bests
        self.save_pop(population=population)

        # launch interface
        self.launch_interface()

        # calculate fitness
        if self.debug > 4: print("Assessing fitness of individuals...")
        _s = time.time()

        # Notes: measuring time should not be up to the fit function writer. We should provide as much info as possible
        # Maybe best pop shouldn't be required
        population, best_ind = self.fitness_func(generation=self.current_generation,
                                                 population=population,
                                                 tensors=tensors,
                                                 f_path=f_path,
                                                 image_extension=self.image_extension,
                                                 work_dir = self.experiment.working_directory,
                                                 polar_mask=self.polar_mask,
                                                 rng=self.engine_rng,
                                                 objective=self.objective,
                                                 resolution=self.target_dims,
                                                 stf=self.save_to_file_image,
                                                 target=self.target,
                                                 dim=self.dimensionality,
                                                 best_o=self.best_overall,
                                                 debug=False if (self.debug == 0) else True)
        fitness_time = time.time() - _s

        # Timers
        self.elapsed_fitness_time += fitness_time
        self.recent_fitness_time = fitness_time
        if self.debug > 4: print("Assessed " + str(len(population)) + " tensors in (s): " + str(fitness_time))

        #print_population(population, best=population[best_ind], print_expr=True, msg="after fit")

        return population, population[best_ind]


    def generate_pop_from_expr(self, strs):
        population = []
        nodes_generated = 0
        maxpopd = -1

        for p in strs:
            t, node = str_to_tree(p, self.terminal.set, constrain_domain=False)

            thisdep, t = node.get_depth()
            if thisdep > maxpopd:
                maxpopd = thisdep

            population.append(new_individual(node, fitness=0, depth=thisdep, nodes=t))
            if self.debug > 0:
                print("Number of nodes:\t:" + str(t))
                print(node.get_str())
            nodes_generated += t
        if self.debug > 0:
            print("Total number of nodes:\t" + str(nodes_generated))

        return population, nodes_generated, maxpopd


    def generate_pop_from_file(self, read_from_file, pop_size=float('inf')):
        # open population files
        strs = []
        with open(read_from_file) as fp:
            line = fp.readline().replace("\n", "").replace('"', "")
            cnt = 0
            while line and cnt < pop_size:
                strs.append(line)
                line = fp.readline().replace("\n", "").replace('"', "")
                cnt += 1
            if cnt < pop_size < float('inf'):
                print(
                    bcolors.WARNING + "[WARNING]:\tCould only read " + str(cnt) + " expressions from population file " +
                    str(read_from_file) + " instead of specified population size of " + str(pop_size) + bcolors.ENDC)

        # convert expressions to trees
        return self.generate_pop_from_expr(strs)


    def read_fitness_from_file(self, population, sl=1, fitness_row = 2):
        line_start = sl
        line_end = len(population) + line_start
        lcnt = 0
        best_ind = 0

        # set objective function according to min/max
        fit = 0
        if self.objective == 'minimizing':
            condition = lambda: (fit < max_fit)  # minimizing
            max_fit = float('inf')
        else:
            condition = lambda: (fit > max_fit)  # maximizing
            max_fit = float('-inf')

        with open(self.start_fitness_file, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if line_start <= lcnt < line_end:
                    fit = float(row[fitness_row])
                    population[lcnt - 1]['fitness'] = fit
                    if condition():
                        max_fit = fit
                        best_ind = lcnt - 1
                lcnt += 1

        return population, population[best_ind]


    def launch_interface(self):
        if self.interface and (not self.active_interface):
            self.active_interface = True
            cmd_list = ["processing-java", "--sketch=" + os.getcwd() + _tgp_delimiter + "evolver" + _tgp_delimiter, "--run", self.experiment.all_directory]
            subprocess.Popen(cmd_list)


    def initialize_population(self, max_depth=8, min_depth=-1, individuals=100, method='ramped half-and-half',
                              max_nodes=-1, read_from=None):
        start_init_population = time.time()
        if read_from is None:  # generate randomly
            nodes_generated, population = self.generate_population(individuals, method, max_nodes, max_depth, min_depth)
        else:

            maxpopd = -1
            if isinstance(read_from, str) and (".txt"  in read_from) or (".csv" in read_from):  # read from file
                population, nodes_generated, maxpopd = self.generate_pop_from_file(read_from_file=read_from,
                                                                                   pop_size=self.population_size)
            elif isinstance(read_from, list):  # generate from list of strs

                population, nodes_generated, maxpopd = self.generate_pop_from_expr(read_from)

            else:  # give warning generate randomly
                print(bcolors.FAIL + "[ERROR]:\tCould not read from source: " + str(
                    read_from) + ", not a list of strings and not a file, randomly generating population instead.",
                      bcolors.ENDC)
                read_from = None
                nodes_generated, population = self.generate_population(individuals, method, max_nodes, max_depth,
                                                                       min_depth)
            # ajust depth if needed
            if maxpopd > self.max_tree_depth:
                newmaxpopd = max(maxpopd, self.max_tree_depth)
                print(bcolors.WARNING + "[WARNING]:\tMax depth of input trees (" + str(
                    maxpopd) + ") is higher than defined max tree depth (" +
                      str(self.max_tree_depth) + "), clipping max tree depth value to " + str(newmaxpopd),
                      bcolors.ENDC)
                self.max_tree_depth = newmaxpopd

        tree_generation_time = time.time() - start_init_population

        if self.debug > 1:
            print("Generated Population: ")
            self.print_population(population)

        #print("Evaluating ", str(len(population)), " individuals in initialization pop gen.", str(self.current_generation))

        flag_fitness = False
        if (self.start_fitness_file is not None) and self.reeval_fitness_start:
            print(bcolors.WARNING + "Reading initial population fitness from file: " + str(self.start_fitness_file) + bcolors.ENDC)
            try:
                population, best_pop = self.read_fitness_from_file(population)
                flag_fitness = True
            except FileNotFoundError:
                print(bcolors.FAIL + "[ERROR]:\tStarting fitness file not found: " + str(self.start_fitness_file) + bcolors.ENDC)
                flag_fitness = False # technically not needed
        if not flag_fitness: population, best_pop = self.fitness_func_wrap(population=population, f_path=self.experiment.cur_image_directory)


        total_time = self.recent_fitness_time + self.recent_tensor_time

        if self.debug > 1:  # print detailed method info
            print("\nInitial Population: ")
            print("Generation method: " + ("Ramdom expression" if (read_from is None) else "Read from file"))
            print("Generated trees in: " + str(tree_generation_time) + " (s)")
            print("Evaluated Individuals in: " + str(total_time) + " (s)")
            print("\nIndividual(s): ")
            print("Nodes generated: " + str(nodes_generated))
            for i in range(individuals):
                print("\nIndiv " + str(i) + ":\nExpr: " + population[i]['tree'].get_str())
                print("Fitness: " + str(population[i]['fitness']))
                print("Depth: " + str(population[i]['depth']))
                print("Depth: " + str(population[i]['nodes']))

        return population, best_pop


    def write_pop_to_csv(self, fp=None):
        if self.can_save_log():
            genstr = "gen" + str(self.current_generation).zfill(5)
            popstr = "pop" + str(self.current_generation).zfill(5)
            fn = self.experiment.generations_directory if fp is None else fp

            fg = fn + genstr + ".csv"
            fnp = fn + popstr + ".csv"

            # write engine summary
            self.summary(force_print=True, log_format=True, write_file=True,
                         file_path=self.experiment.working_directory,
                         file_name='state.log',
                         pop_path=fnp)

            # save seed state
            state = self.engine_rng.getstate()
            #print("Kwargs seed state ", state)
            state_fp = self.experiment.current_directory + save_seed_file
            with open(state_fp, 'wb') as seed_file:
                pickle.dump(state, seed_file)

            # write information of generation
            with open(fg, mode='w', newline='') as g_file, open(fnp, mode='w', newline='') as p_file:
                g_fwriter = csv.writer(g_file, delimiter=',')
                p_fwriter = csv.writer(p_file, delimiter = ',')
                ind = 0
                for p in self.population:
                    if ind == 0:
                        g_file.write("individual_number,individual_name,fitness,depth,number_nodes,expression\n")
                    g_fwriter.writerow(
                        [str(ind), genstr + "_indiv_" + str(ind).zfill(5), p['fitness'], p['depth'], p['nodes']])
                    p_fwriter.writerow([p['tree'].get_str()])
                    ind += 1

    def population_stats(self, population, fitness=True, depth=True, nodes=True):
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

    def generate_pop_images(self, expressions, fpath=None):
        fp = self.experiment.current_directory if fpath is None else fpath

        if isinstance(expressions, str):
            pop, number_nodes, max_dep = self.generate_pop_from_file(expressions)
            tensors, time_taken = self.calculate_tensors(pop)
        elif isinstance(expressions, list):
            # for p in expressions:
            #    print(p)
            #    _, node = str_to_tree(p, self.terminal.set)
            #    tensors.append(node)
            pop, number_nodes, max_dep = self.generate_pop_from_expr(expressions)
            tensors, time_taken = self.calculate_tensors(pop)
        else:
            print(bcolors.FAIL + "[ERROR]:\tTo generate images from a population please enter either"
                                 " a file or a list with the corresponding expressions." + bcolors.ENDC)
            return None
        index = 0
        for p in pop:
            t = p['tensor']
            save_image(t, index, fp, self.target_dims, BGR=self.do_bgr, extension=self.image_extension)
            index += 1
        return tensors


    # and ((node_p1.value != 'scalar') or (node_p1.children == node_p2.children)):

    def generate_aligned(self, node_p1, node_p2):
        if (node_p1.value != 'scalar') and (node_p1.value == node_p2.value):
            children = [self.generate_aligned(c1, c2) for c1, c2 in zip(node_p1.children, node_p2.children)]
            return Node(node_p1.value, children, node_p1.terminal)
        else:
            children = [copy.deepcopy(node_p1), copy.deepcopy(node_p2), Node('scalar', [0.0], True)]
            return Node('lerp', children, False)

    def deep_shallow_copy(self, ind):
        # tensors already make a copy so technically copy.copy() is not needed
        return new_individual(
            copy.deepcopy(ind['tree']), fitness=ind['fitness'], depth=ind['depth'], nodes=ind['nodes'],
            tensor=copy.copy(ind['tensor']), weights=ind['weights']
        )

    def run(self, stop_value=10, start_from_last_pop=True):

        if not self.minimal_print:
            print(bcolors.OKGREEN + "\n\n" + "=" * 84, bcolors.ENDC)

        if self.save_state > 0:
            self.restart(stop_value)

        if self.fitness_func is None and ((self.pop_source is None) or (self.stop_value > 0)):
            print(bcolors.FAIL + "[ERROR]:\tFitness function must be defined to run evolutionary process.",
                  bcolors.ENDC)
            return

        if (start_from_last_pop is False) or (start_from_last_pop is True):
            start_from_last_pop *= self.population_size
        start_from_last_pop = clamp(0, start_from_last_pop, self.population_size)

        if (self.save_state == 0) or (len(self.population) == 0):
            start_from_last_pop = 0

        #print(start_from_last_pop)
        # either generate initial pop randomly or read fromfile along with remaining experiment data
        self.data = []

        if start_from_last_pop < self.population_size or self.save_state == 0:
            self.experiment.set_generation_directory(self.current_generation, self.can_save_image_pop)

            population, best = self.initialize_population(max_depth=self.max_init_depth,
                                                          min_depth=self.min_init_depth,
                                                          individuals=self.population_size - start_from_last_pop,
                                                          method=self.method,
                                                          max_nodes=self.max_nodes,
                                                          read_from=self.pop_source)

            if start_from_last_pop > 0:
                meta_elite = self.get_n_best_from_pop(population=self.population, n=start_from_last_pop)
                if self.condition_local(meta_elite[0]['fitness']):
                    best = meta_elite[0]
                population += meta_elite

            self.population = population
            self.best = best
            self.best_overall = self.deep_shallow_copy(self.best)

            # Print initial population
            if self.debug > 1:
                self.print_population(population=population, minimal=False)
            # print("tournament size: " + str(self.tournament_size))

            # save bests
            self.save_best()


            # write first gen data
            self.write_pop_to_csv(self.pop_file_path)
            self.save_bests_log()

            if self.debug > 2:
                self.summary(force_print=True)

            # display gen statistics
            pops = self.population_stats(self.population)
            self.data.append([self.current_generation,
                              pops['fitness'][0], pops['fitness'][1], pops['fitness'][2], pops['fitness'][3],
                              pops['depth'][0], pops['depth'][1], pops['depth'][2], pops['depth'][3],
                              pops['nodes'][0], pops['nodes'][1], pops['nodes'][2], pops['nodes'][3],
                              self.recent_engine_time, self.recent_fitness_time, self.recent_tensor_time])
            if self.save_state == 0:
                print(
                    bcolors.BOLD + bcolors.OKCYAN + "\n[       |                    FITNESS                    |                     DEPTH                     |                     NODES                     |              TIMINGS              ]",
                    bcolors.ENDC)
                print(
                    bcolors.BOLD + bcolors.OKCYAN + "[  gen  |    avg    ,    std    , best(gen) , best(all) |    avg    ,    std    , best(gen) , best(all) |    avg    ,    std    , best(gen) , best(all) | generation,  fit eval ,tensor eval]\n",
                    bcolors.ENDC)
            print(
                bcolors.OKBLUE + "[%7d, %10.6f, %10.6f, %10.6f, %10.6f, %10.3f, %10.6f, %10d, %10d, %10.3f, %10.6f, %10d, %10d, %10.6f, %10.6f, %10.6f]" % tuple(
                    self.data[-1]), bcolors.ENDC)

            self.current_generation += 1


        # Save engine state to file
        #self.save_engine_state()

        while self.condition():

            # Set directory to save engine state in this generation
            self.experiment.set_generation_directory(self.current_generation, self.can_save_image_pop)

            # TODO: immigrate individuals (archive)

            # Create new population of individuals

            new_population = self.get_n_best_from_pop(population=self.population, n=self.elitism)

            #print_population(new_population, best=None, print_expr=True, msg="after new_pop")

            temp_population = []
            retrie_cnt = []

            #print_population(self.population, best=self.best, print_expr=True, msg="before torunament")

            for current_individual in range(self.population_size - self.elitism):

                rcnt = 0
                if self.bloat_control == "off":
                    member_depth = float('inf')

                    while member_depth > self.max_tree_depth and rcnt < self.max_retries:
                        indiv_temp, parent, plist = self.selection()
                        member_depth, member_nodes = indiv_temp.get_depth()

                        rcnt += 1
                    if member_depth > self.max_tree_depth:
                        indiv_temp = parent['tree']
                        member_depth = parent['depth']
                        member_nodes = parent['nodes']

                    # print("retries: ", rcnt)

                    retrie_cnt.append(rcnt)
                else:
                    indiv_temp, _, plist = self.selection()
                    member_depth, member_nodes = indiv_temp.get_depth()
                temp_population.append(
                    new_individual(indiv_temp, fitness=0, depth=member_depth, nodes=member_nodes, valid=False,
                                   parents=plist))
                if self.debug > 10: print("Individual " + str(indiv_temp) + ": " + indiv_temp.get_str())

            # Print average retrie count
            if self.debug >= 4:
                rstd = np.average(np.array(retrie_cnt))
                print("[DEBUG]:\tAverage evolutionary ops retries for generation " + str(
                    self.current_generation) + ": " + str(rstd))

            # bloat control:
            #
            # off        - min_tree_depth < depth < max_tree_depth
            # weak       - dynamic_limit can only increase (until max_overall_size)
            # heavy      - dynamic_limit can increase (until max_overall_size) and decrease (until initial dynamic_limit value)
            # very heavy - dynamic_limit can increase (until max_overall_size) and decrease (until min_overall_size)
            #
            # bloat control modes:
            #
            # depth - dynamic_limit refers to tree depth
            # size  - dynamic_limit refers to number of nodes in a tree
            #
            # https://www.researchgate.net/publication/220286086_Dynamic_limits_for_bloat_control_in_genetic_programming_and_a_review_of_past_and_current_bloat_theories


            if self.bloat_control == "off":
                # for current_individual in range(self.population_size - self.elitism):
                #    ind = temp_population[current_individual]
                #    new_population.append(ind)

                if self.reeval_elite:
                    new_population += temp_population
                    if len(new_population) > 0:
                        new_population, _ = self.fitness_func_wrap(population=new_population, f_path=self.experiment.current_directory)
                else:
                    if len(temp_population) > 0:
                        temp_population, _ = self.fitness_func_wrap(population=temp_population, f_path=self.experiment.current_directory)
                    new_population += temp_population

            else:

                if len(temp_population) > 0:
                    temp_population, _ = self.fitness_func_wrap(population=temp_population, f_path=self.experiment.current_directory)

                # force reevaluation of elite (for example when fitness func is dynamic)
                if self.reeval_elite and len(new_population) > 0:
                    new_population, _ = self.fitness_func_wrap(population=new_population, f_path=self.experiment.current_directory)


                accepted = 0
                depth_mode = self.bloat_mode == 'depth'
                best_fit = self.best_overall['fitness']
                temp_limit = self.dynamic_limit

                for current_individual in range(self.population_size - self.elitism):
                    ind = temp_population[current_individual]
                    my_limit = get_largest_parent(ind, depth=depth_mode) if has_illegal_parents(
                        ind) else self.dynamic_limit

                    sizeind = ind['depth'] if depth_mode else ind['nodes']

                    if self.min_overall_size <= sizeind <= self.max_overall_size:  # verify overall min and max sizes
                        fitnessind = ind['fitness']

                        if sizeind <= my_limit:
                            ind['valid'] = True
                            accepted += 1
                            if ((self.objective == 'minimizing') and (fitnessind < best_fit)) or (
                                    (self.objective != 'minimizing') and (fitnessind > best_fit)):
                                best_fit = fitnessind

                            if (self.bloat_control == 'very heavy') or (
                                    (self.bloat_control == 'heavy') and (sizeind >= self.initial_dynamic_limit)):

                                if self.lock_dynamic_limit:
                                    temp_limit = sizeind
                                else:
                                    self.dynamic_limit = sizeind

                        if sizeind > self.dynamic_limit and (
                                (self.objective == 'minimizing') and (fitnessind < best_fit)) or (
                                (self.objective != 'minimizing') and (fitnessind > best_fit)):
                            ind['valid'] = True
                            accepted += 1

                            best_fit = fitnessind

                            if self.lock_dynamic_limit:
                                temp_limit = sizeind
                            else:
                                self.dynamic_limit = sizeind

                    # print("my limit: ", my_limit, "ind dep", ind['depth'], "is legal", ind['valid'])
                    # print("dynamic limit: ", self.dynamic_limit)

                if self.lock_dynamic_limit: self.dynamic_limit = temp_limit

                # print("dynamic limit: ", self.dynamic_limit)

                # build new_pop
                illegals = 0
                passp = 0
                for current_individual in range(self.population_size - self.elitism):
                    ind = temp_population[current_individual]
                    sizeind = ind['depth'] if depth_mode else ind['nodes']
                    # build new pop according to the ones that passed last loop
                    if ind['valid']:
                        new_population.append(ind)
                    else:
                        passp += 1
                        new_population.append(self.engine_rng.choice(ind['parents']))
                    ind['valid'] = sizeind < self.dynamic_limit  # update illegals according to final limits
                    if not ind['valid']:
                        illegals += 1


            # update population
            self.population = new_population

            # update best gen and overall
            self.best = self.get_n_best_from_pop(population=self.population, n=1)[0]
            #print_population([self.best], best=None, print_expr=True, msg="saving best")

            if self.condition_overall(self.best['fitness']):
                self.best_overall = self.deep_shallow_copy(self.best)

            self.save_best()

            # update engine time
            self.update_engine_time()

            if self.debug > 10 and self.bloat_control != 'off':
                print("\ngenerated, passed, illegals, len(pop), passp", self.population_size - self.elitism, accepted,
                      illegals, len(new_population), passp)
                print("Pop")
                i = 0
                for ind in new_population:
                    print("ind, fit, exp", i, ind['fitness'], ind['tree'].get_str())
                    i += 1
                print("Best (gen    ), fit, exp", self.best['fitness'], self.best['tree'].get_str())
                print("Best (overall), fit, exp", self.best_overall['fitness'], self.best_overall['tree'].get_str())

            # if self.save_state == 0: self.print_population(self.population, minimal = True)


            # add population data to statistics and display gen statistics
            pops = self.population_stats(self.population)
            self.data.append([self.current_generation,
                              pops['fitness'][0], pops['fitness'][1], pops['fitness'][2], pops['fitness'][3],
                              pops['depth'][0], pops['depth'][1], pops['depth'][2], pops['depth'][3],
                              pops['nodes'][0], pops['nodes'][1], pops['nodes'][2], pops['nodes'][3],
                              self.recent_engine_time, self.recent_fitness_time, self.recent_tensor_time])
            print(
                bcolors.OKBLUE + "[%7d, %10.6f, %10.6f, %10.6f, %10.6f, %10.3f, %10.6f, %10d, %10d, %10.3f, %10.6f, %10d, %10d, %10.6f, %10.6f, %10.6f]" % tuple(
                    self.data[-1]), bcolors.ENDC)

            self.write_pop_to_csv(self.pop_file_path)
            self.save_bests_log()

            # print engine state
            self.summary(force_print=False)

            # advance generation
            self.current_generation += 1
            #self.experiment.seed += 1

            # save engine state
            # if self.save_to_file != 0 and (self.current_generation % self.save_to_file) == 0:
            # self.save_state_to_file(self.experiment.logging_directory)
            #self.save_engine_state()

        # write statistics(data) to csv
        self.write_overall_to_csv(self.data)

        # Write final enggine state to file
        # self.save_engine_state()

        # save best overall image to top level
        if self.can_save_image_best():
            # Save Best Image
            fn = self.experiment.best_overall_directory + str(self.current_generation).zfill(5)
            save_image(self.best_overall['tensor'], 0, fn, self.target_dims, BGR=self.do_bgr,
                       extension=self.image_extension, sufix="_best_overall")

        # print final stats
        if self.debug > 0:
            self.summary(force_print=True)
        elif not self.minimal_print:
            # TODO: make this an engine property for displaying to console (not to saved files)
            places_to_round = 3
            print(bcolors.OKGREEN + "\nElapsed Engine Time: \t" + str(
                round(self.elapsed_engine_time, places_to_round)) + " sec.")
            print("\nElapsed Init Time   : \t" + str(round(self.elapsed_init_time, places_to_round)) + " sec.")
            print("Elapsed Tensor Time : \t" + str(round(self.elapsed_tensor_time, places_to_round)) + " sec.")
            print("Elapsed Fitness Time:\t" + str(round(self.elapsed_fitness_time, places_to_round)) + " sec.")
            print("\nBest individual (generation):\n" + bcolors.OKCYAN + self.best['tree'].get_str())
            print(
                "\nBest individual (overall):\n" + bcolors.OKCYAN + self.best_overall['tree'].get_str() + bcolors.ENDC)

        if self.save_graphics: self.graph_statistics(extension=self.graphic_extension)
        if not self.minimal_print: print(bcolors.BOLD + bcolors.OKGREEN + "=" * 84, "\n\n" + bcolors.ENDC)

        self.save_state += 1
        tensors = [p['tensor'] for p in self.population]
        return self.data, tensors


    def set_polar_mask_value(self):
        if self.do_polar_mask:
            polar_mask_value = self.polar_mask_value
            if hasattr(polar_mask_value, "__module__"):
                if type(polar_mask_value).__module__ == 'numpy':
                    self.polar_mask_value = torch.Tensor(polar_mask_value)
            else:
                if isinstance(polar_mask_value, float):
                    mask_expr = "scalar(" + str(clamp(_codomain[0], polar_mask_value, _codomain[1])) + ")"
                elif isinstance(polar_mask_value, int):
                    mask_expr = "scalar(" + str(clamp(_codomain[0], float(polar_mask_value), _codomain[1])) + ")"
                elif isinstance(polar_mask_value, list):
                    mask_expr = "scalar("
                    mask_args = min(self.dimensionality, len(polar_mask_value))
                    for i in range(mask_args):
                        mask_expr += str(polar_mask_value[i])
                        if i < mask_args - 1:
                            mask_expr += ","
                    mask_expr += ")"
                elif isinstance(polar_mask_value, str):
                    mask_expr = polar_mask_value
                else:
                    mask_expr = "scalar(" + str(_codomain[0]) + ")"

                _, tt = str_to_tree(mask_expr, self.terminal.set, constrain_domain=False)
                tt_temp = tt.get_tensor(self)
                self.polar_mask_value = tt_temp.type(torch.float32)
                self.polar_mask_value_expr = mask_expr


    def selection(self):
        parent = self.tournament_selection()
        plist = []
        if self.bloat_control != "off":
            plist = [parent]
        indiv_temp = parent['tree']
        random_n1 = self.engine_rng.random()
        random_n2 = self.engine_rng.random()
        if random_n1 < self.crossover_rate:
            parent_2 = self.tournament_selection()
            if self.bloat_control != "off":
                plist.append(parent_2)
            indiv_temp = self.crossover(parent['tree'], parent_2['tree'])
            # print("cross")
        if random_n2 < self.mutation_rate:
            indiv_temp = self.mutation(parent['tree'])

        return indiv_temp, parent, plist


    def get_n_best_from_pop(self, population, n):
        if self.objective == 'minimizing':
            elite = nsmallest(n, population, key=itemgetter('fitness'))
        else:
            elite = nlargest(n, population, key=itemgetter('fitness'))
        return elite

    def save_best(self):
        if self.can_save_image_best():
            # Save Best Image
            fn = self.experiment.bests_directory + "best_gen" + str(self.current_generation).zfill(5)
            save_image(self.best['tensor'], 0, fn, self.target_dims, BGR=self.do_bgr,
                       extension=self.image_extension)

    def save_pop(self, population):

        if self.can_save_image_pop():
            # Save Population Images
            if self.save_image_pop:
                for i in range(len(population)):
                    fn = self.experiment.cur_image_directory + "gen" + str(self.current_generation).zfill(5)
                    save_image(population[i]['tensor'], i, fn, self.target_dims, BGR=self.do_bgr,
                               extension=self.image_extension)

            # Remove file if it exists
            if os.path.exists(self.flag_file):
                os.remove(self.flag_file)


    def graph_statistics(self, extension=".pdf"):

        extension.strip(".")
        if extension not in ["pdf", "svg"]:
            extension = "pdf"
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

        with open(self.experiment.overall_fp, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if self.debug > 10: print(row)
                if line_start <= lcnt < line_end:
                    avg_fit.append(float(row[1]))
                    std_fit.append(float(row[2]))
                    best_fit.append(float(row[4]))  # overall
                    avg_dep.append(float(row[5]))
                    std_dep.append(float(row[6]))
                    best_dep.append(float(row[8]))  # overall
                lcnt += 1

        # showing best overall
        fig, ax = plt.subplots(1, 1)
        start_gen = (self.stop_value + 1) - len(avg_fit)
        stop_gen = self.stop_value + 1

        ax.plot(range(start_gen, stop_gen), avg_fit, linestyle='-', label="AVG")
        ax.plot(range(start_gen, stop_gen), best_fit, linestyle='-', label="BEST (overall)")
        pylab.legend(loc='upper left')
        ax.set_xlabel('Generations')
        ax.set_ylabel('Fitness')
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_title('Fitness across generations')
        fig.set_size_inches(12, 8)
        plt.savefig(fname=self.experiment.graphs_directory + 'Fitness_ ' + self.experiment.filename + '.' + extension,
                    format=extension)
        if self.show_graphics: plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        ax.plot(range(start_gen, stop_gen), avg_dep, linestyle='-', label="AVG")
        ax.plot(range(start_gen, stop_gen), best_dep, linestyle='-', label="BEST (overall)")
        pylab.legend(loc='upper left')
        ax.set_xlabel('Generations')
        ax.set_ylabel('Depth')
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_title('Avg depth across generations')
        fig.set_size_inches(12, 8)
        plt.savefig(fname=self.experiment.graphs_directory + 'Depth_' + self.experiment.filename + '.' + extension,
                    format=extension)
        if self.show_graphics: plt.show()
        plt.close(fig)


    def write_overall_to_csv(self, data):
        # evolutionary stats across generations
        fn = (
                self.experiment.working_directory + "evolution_" + self.experiment.filename + ".csv") if self.experiment.overall_fp is None else self.experiment.overall_fp
        with open(fn, mode='a', newline='') as file:
            fwriter = csv.writer(file, delimiter=',')
            ind = 0
            for d in data:
                if ind == 0 and self.save_state == 0:
                    file.write(
                        "generation,fitness_avg,fitness_std,fitness_generational_best,fitness_overall_best," \
                        "depth_avg,depth_std,depth_generational_best,depth_overall_best," \
                        "node_avg,node_std,node_generational_best,node_overall_best," \
                        "generation_time,fitness_time,tensor_time\n")
                fwriter.writerow(d)
                ind += 1

        fn = (
                self.experiment.working_directory + "timings_" + self.experiment.filename + ".csv") if self.experiment.timings_fp is None else self.experiment.timings_fp
        with open(fn, mode='a', newline='') as file:
            fwriter = csv.writer(file, delimiter=',')
            if self.save_state == 0:
                file.write("resolution,seed,initialization_time,tensor_time,fitness_time,total,engine_time\n")
            fwriter.writerow([self.target_dims[0], self.experiment.seed, self.elapsed_init_time,
                              self.elapsed_tensor_time, self.elapsed_fitness_time, self.elapsed_engine_time])

    def update_engine_time(self):
        t_ = time.time()
        self.elapsed_engine_time += t_ - self.last_engine_time
        self.recent_engine_time = t_ - self.last_engine_time
        self.last_engine_time = t_

    def save_engine_state(self):
        if self.can_save_state():
            with open(self.experiment.setup_fp, "w") as text_file:
                try:
                    text_file.write(self.summary(force_print=True))
                except IOError as error:
                    print(bcolors.FAIL + "[ERROR]:\tI/O error while writing engine state ({0}): {1}".format(error.errno,
                                                                                                            error.strerror),
                          bcolors.ENDC)

    def save_bests_log(self):
        if self.save_bests:
            with open(self.experiment.bests_fp, "a") as csv_file:
                try:
                    if self.current_generation == 0: csv_file.write("generation,index,expression\n")
                    writer = csv.writer(csv_file)
                    writer.writerow([self.current_generation, self.best['fitness'], self.best['tree'].get_str()])
                except IOError as error:
                    print(bcolors.FAIL + "[ERROR]:\tI/O error while writing best overall individuals ({0}): {1}".format(
                        error.errno, error.strerror), bcolors.ENDC)

        if self.save_bests_overall:
            with open(self.experiment.bests_overall_fp, "a") as csv_file:
                try:
                    if self.current_generation == 0: csv_file.write("generation,index,expression\n")
                    writer = csv.writer(csv_file)
                    writer.writerow(
                        [self.current_generation, self.best_overall['fitness'], self.best_overall['tree'].get_str()])
                except IOError as error:
                    print(bcolors.FAIL + "[ERROR]:\tI/O error while writing best individuals ({0}): {1}".format(
                        error.errno, error.strerror), bcolors.ENDC)

    def print_population(self, population, minimal=False):
        if not minimal:
            for i in range(len(population)):
                p = population[i]
                print("\nIndividual " + str(i) + ":")
                print("Fitness:\t" + str(p['fitness']))
                print("Depth:\t" + str(p['depth']))
                print("Expression:\t" + str(p['tree'].get_str()))
        else:
            for i in range(len(population)):
                p = population[i]
                print(str(p['tree'].get_str()))

    def print_engine_sta(self, force_print=False):
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

            # print population
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

    def __init__(self, fset, edim, debug=0):

        self.debug = debug
        self.set = {}
        self.arity = {}
        self.min_arity = float('inf')
        self.max_arity = 0

        self.operators_def = {  # arity, function
            'abs': [1, node_abs],
            'add': [2, node_add],
            'and': [2, node_bit_and],
            'clip': [3, node_clip],
            'cos': [1, node_cos],
            'div': [2, node_div],
            'exp': [1, node_exp],
            'frac': [1, node_frac],
            'if': [3, node_if],
            'len': [2, node_len],
            'lerp': [3, node_lerp],
            'log': [1, node_log],
            'max': [2, node_max],
            'mdist': [2, node_mdist],
            'min': [2, node_min],
            'mod': [2, node_mod],
            'mult': [2, node_mul],
            'neg': [1, node_neg],
            'or': [2, node_bit_or],
            'pow': [2, node_pow],
            'sign': [1, node_sign],
            'sin': [1, node_sin],
            'sqrt': [1, node_sqrt],
            'sstep': [1, node_sstep],
            'sstepp': [1, node_sstepp],
            'step': [1, node_step],
            'sub': [2, node_sub],
            'tan': [1, node_tan],
            # 'warp': [edim + 1, node_warp],
            'xor': [2, node_bit_xor],
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
        if operator_name in self.set:
            if self.debug > 0:
                print(bcolors.WARNING + "[WARNING]:\tOperator already existing in current function_set, overriding...",
                      bcolors.ENDC)
            if operator_name in self.operators_def and self.debug > 1:
                print(bcolors.WARNING + "[WARNING]:\tOverriding operator", operator_name,
                      ", which is an engine defined operator. This is not recommended." + bcolors.ENDC)
            self.remove_from_set(operator_name)

        self.set[operator_name] = [number_of_args, function_pointer]
        if number_of_args not in self.arity:
            self.arity[number_of_args] = []
        self.arity[number_of_args].append(operator_name)

    def remove_from_set(self, operator_name):
        if operator_name not in self.set:
            print(bcolors.FAIL + "[ERROR]:\tOperator", operator_name, "not present in function set, failed to remove.",
                  bcolors.ENDC)
        else:
            if operator_name in self.operators_def and self.debug > 0:
                print(bcolors.WARNING + "[WARNING]:\tRemoving operator", operator_name,
                      ", which is an engine defined operator. I hope you know what you are doing." + bcolors.ENDC)
            if operator_name != 'mult':  # we won't actually remove some operators because we might need them for scaling in the begginning
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

    def __str__(self, log_format=False):
        res = "operators = {" if log_format else "\nOperators:\n"
        i = 0
        for s, v in self.set.items():
            if log_format:
                res += str(s)
            else:
                res += str(s) + ": [" + str(v[0]) + ", " + str(get_func_name(v[1])) + "]"
            if i < len(self.set) - 1:
                res += ", "
                if not log_format:
                    res += "\n"
            i += 1
        if log_format:
            res += "}"
        else:
            res += "\nArity sorted:\n"
            for s in self.arity:
                res += str(s) + ", " + str(self.arity[s]) + "\n"
        return res

    def summary(self, log_format=False):
        summary_str = ''
        if not log_format:
            summary_str += "Debug: " + str(self.debug) + "\n"
            summary_str += "Min arity: " + str(self.min_arity) + "\n"
            summary_str += "Max arity: " + str(self.max_arity) + "\n"
        summary_str += self.__str__(log_format=log_format) + "\n"
        return summary_str


def clamp(x, n, y):
    return max(min(n, y), x)


def uniform_sampling(res, minval = _domain[0], maxval = _domain[1]):
    delta = maxval - minval
    return torch.add(torch.mul(torch.rand(res), delta), minval)


# engref optional to allow for independent set creation
# when calling inside the engine, we will provide the engine ref itself to allow terminal definition by expressions
class Terminal_Set:

    def __init__(self, effective_dim, resolution, function_ptr_to_var_node=node_var, debug=0, engref=None):
        self.debug = debug
        self.engref = engref
        self.dimension = resolution[effective_dim] if (effective_dim < len(resolution)) else 1

        # if engine ref is None then we also did not define the domain and codomain so make an empty set
        if self.engref is not None:
            self.set = self.make_term_variables(0, effective_dim, resolution, function_ptr_to_var_node)  # x, y
            if effective_dim >= 2 and self.engref.polar_coordinates:
                x = self.set['x']
                y = self.set['y']
                y = y * - 1
                xy_dist = x ** 2 + y ** 2
                self.set['x'] = torch.atan2(x, y) / math.pi
                self.set['y'] = torch.sqrt(xy_dist) * 2 - 1
                if self.engref.do_polar_mask:
                    self.engref.polar_mask = torch.where(xy_dist > 1, 0, 1)
        else:
            self.set = {}

        self.latentset = self.make_term_variables(effective_dim, len(resolution), resolution,
                                                  function_ptr_to_var_node)  # z for terminal

    def make_term_variables(self, start, end, resolution, fptr):
        res = {}
        for i in range(start, end):  # TODO, what does this dim - 1 do? Is it only to do with warp?
            digit = i
            name = ""

            while True:
                n = digit % 26
                val = n if n <= 2 else n - 26
                name = chr(ord('x') + val) + name
                digit //= 26
                if digit <= 0:
                    break

            if self.debug > 2:
                print("[DEBUG]:\tAdded terminal " + str(name))

            vari = i  # TODO: ye, this is because of images, right?....
            if i < 2:
                vari = 1 - i
            res[name] = fptr(np.copy(resolution), vari)

        return res

    def add_to_set(self, name, t, engref=None):
        if name in self.set and self.debug > 0:
            print(
                bcolors.WARNING + "[WARNING]:\tOperator already existing in current terminal set. Be careful not to redefine 'variables' like x, y, z, etc...",
                bcolors.ENDC)

        tensor = t
        if isinstance(t, str):
            if engref is None:
                print(
                    bcolors.FAIL + "[ERROR]:\tIf you wish to generate a terminal by an expression, pass an engine instance when initializing the Terminal set.",
                    bcolors.ENDC)
            else:
                _, tree = str_to_tree(t, self.set)
                tensor = tree.get_tensor(engref)
        elif hasattr(t, "__module__") and type(t).__module__ == 'numpy':
            tensor = torch.Tensor(t)
        else:
            tensor = t

        self.set[name] = tensor

    def remove_from_set(self, name):
        if name not in self.set:
            print(bcolors.FAIL + "[ERROR]:\tOperator", name, "not present in terminal set, failed to remove.",
                  bcolors.ENDC)
        else:
            if self.debug:
                print(
                    bcolors.WARNING + "[WARNING]:\tBe careful while removing terminals, your trees must be expressed accordingly. I REALLY hope you know what you are doing...",
                    bcolors.ENDC)
            del self.set[name]

    def __str__(self):
        res = ''
        for s in self.set:
            res += s + "\n"
        return res

    def summary(self, log_format=False):
        summary_str = ''
        if not log_format:
            summary_str += "Debug: " + str(self.debug) + "\n"
            summary_str += "Dimension: " + str(self.dimension) + "\n"
            summary_str += "Engine reference: " + ("exists" if self.engref is not None else "None") + "\n"
            summary_str += "Latent set: " + "\n"
            for s in self.latentset:
                summary_str += s + "\n"
            summary_str += "Variables: \n" + self.__str__() + "\n"

        return summary_str
