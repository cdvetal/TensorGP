from tensorgp.engine import *
import random

# NIMA classifier imports
from keras.models import Model
from keras.layers import Dense, Dropout

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as preprocess_input_mob

from utils.score_utils import mean_score, std_score

def teste(**kwargs):
    # read parameters
    population = kwargs.get('population')
    generation = kwargs.get('generation')
    tensors = kwargs.get('tensors')
    f_path = kwargs.get('f_path')
    objective = kwargs.get('objective')
    _resolution = kwargs.get('resolution')
    _stf = kwargs.get('stf')

    images = True
    # set objective function according to min/max
    fit = 0
    if objective == 'minimizing':
        condition = lambda: (fit < max_fit)  # minimizing
        max_fit = float('inf')
    else:
        condition = lambda: (fit > max_fit) # maximizing
        max_fit = float('-inf')

    fn = f_path + "gen" + str(generation).zfill(5)
    fitness = []
    best_ind = 0

    # scores
    for index in range(len(tensors)):
        if generation % _stf == 0:
            save_image(tensors[index], index, fn, _resolution) # save image
        #print(tensors[index])
        # fit = mean - std
        fit = random.random() ## random!

        if condition():
            max_fit = fit
            best_ind = index
        fitness.append(fit)
        population[index]['fitness'] = fit

    # save best indiv
    if images:
        save_image(tensors[best_ind], best_ind, fn, _resolution, addon='_best')
    return population, best_ind, tensors

# if no function set is provided, the engine will use all internally available operators:
#fset = {'abs', 'add', 'and', 'clip', 'cos', 'div', 'exp', 'frac', 'if', 'len', 'lerp', 'log',
#        'max', 'mdist', 'min', 'mod', 'mult', 'neg', 'or', 'pow', 'sign', 'sin', 'sqrt', 'sstep',
#        'sstepp', 'step', 'sub', 'tan', 'warp', 'xor'}


if __name__ == "__main__":

    # NIMA likes 224 by 224 pixel images, the remaining 3 are the RBG color channels
    #resolution = [28, 28, 1]
    resolution = [28, 28]

    # GP params (teste super simples)
    dev = '/cpu:0'  # device to run, write '/cpu_0' to tun on cpu
    number_generations = 10
    pop_size = 10
    tour_size = 3
    mut_prob = 0.1
    cross_prob = 0.95
    max_tree_dep = 3
    # tell the engine that the RGB does not explicitly make part of the terminal set
    edims = 2

    # Initialize NIMA classifier
    #base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    #x = Dropout(0)(base_model.output)
    #x = Dense(10, activation='softmax')(x)
    #model = Model(base_model.input, x)
    #model.load_weights('weights/weights_mobilenet_aesthetic_0.07.hdf5')

    #seed = random.randint(0, 0x7fffffff)
    seed = 2020  # reproducibility

    # create engine
    engine = Engine(fitness_func=teste,
                    population_size=pop_size,
                    tournament_size=tour_size,
                    mutation_rate=mut_prob,
                    crossover_rate=cross_prob,
                    max_tree_depth = max_tree_dep,
                    target_dims=resolution,
                    #method='grow',
                    method='ramped half-and-half',
                    objective='minimizing',
                    device=dev,
                    stop_criteria='generation',
                    min_init_depth=0,
                    max_init_depth=10,
                    bloat_control='dynamic_dep',
                    elitism=0,
                    stop_value=number_generations,
                    effective_dims = edims,
                    seed = seed,
                    debug=0,
                    save_to_file=3, # save all images from each 10 generations
                    save_graphics=True,
                    show_graphics=False,
                    write_gen_stats=False,
                    write_log = False,
                    write_final_pop = True,
                    read_init_pop_from_file = None)
                    #read_init_pop_from_file = "/home/scarlett/Documents/TensorGP/TensorGP-master/runs/run__2021_11_19__18_30_27_780__107305148598124544__images/run__2021_11_19__18_30_27_830__107305148598124544_final_pop.txt") # read predefined pop

    #str_tree1 = "add(x, y)"  log(x), y, escolhe x dá x; x, y escolhe x dá x; x, log(y), escolhe x dá log(x)
    #str_tree2 = "scalar(1.0)"
    #str_tree1 = "x"
    #_, tree1 = str_to_tree(str_tree1, engine.terminal.set, constrain_domain=False)
    #print("Tree1: ", tree1.get_str())
    #_, tree2 = str_to_tree(str_tree2, engine.terminal.set, constrain_domain=False)
    #print("Tree2: ", tree2.get_str())
    #indiv = engine.crossover(tree1, tree2)
    #print("Res: ", indiv.get_str())


    # This experiment is comparatively slower, but bear inmind that the NIMA classifier takes
    # a considerable amount of time
    # run evolutionary process
    engine.run()
    print("First run over!")
    _, tensors = engine.run(5)
    print("Done!")
    for t in tensors:
        print(t.shape)
    print(tensors[0])


    
