from tensorgp.engine import *

# NIMA classifier imports
from keras.models import Model
from keras.layers import Dense, Dropout

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as preprocess_input_mob

from utils.score_utils import mean_score, std_score


# Fitness assessment with the NIMA classifier
# https://github.com/idealo/image-quality-assessment
def nima_classifier(**kwargs):
    # read parameters
    population = kwargs.get('population')
    generation = kwargs.get('generation')
    tensors = kwargs.get('tensors')
    f_path = kwargs.get('f_path')
    objective = kwargs.get('objective')
    _resolution = kwargs.get('resolution')
    _stf = kwargs.get('stf')

    images = True

    fn = f_path + "gen" + str(generation).zfill(5)
    fitness = []
    best_ind = 0

    # set objective function according to min/max
    fit = 0
    if objective == 'minimizing':
        condition = lambda: (fit < max_fit)  # minimizing
        max_fit = float('inf')
    else:
        condition = lambda: (fit > max_fit) # maximizing
        max_fit = float('-inf')


    number_tensors = len(tensors)
    with tf.device('/CPU:0'):

        # NIMA classifier
        x = np.stack([tensors[index].numpy() for index in range(number_tensors)], axis = 0)
        x = preprocess_input_mob(x)
        scores = model.predict(x, batch_size = number_tensors, verbose=0)

        # scores
        for index in range(number_tensors):

            mean = mean_score(scores[index])
            std = std_score(scores[index])
            # fit = mean - std
            fit = mean

            if condition():
                max_fit = fit
                best_ind = index
            fitness.append(fit)
            population[index]['fitness'] = fit

    return population, best_ind
    # return population, population[best_ind]


# if no function set is provided, the engine will use all internally available operators:
#fset = {'abs', 'add', 'and', 'clip', 'cos', 'div', 'exp', 'frac', 'if', 'len', 'lerp', 'log',
#        'max', 'mdist', 'min', 'mod', 'mult', 'neg', 'or', 'pow', 'sign', 'sin', 'sqrt', 'sstep',
#        'sstepp', 'step', 'sub', 'tan', 'warp', 'xor'}


if __name__ == "__main__":

    # NIMA likes 224 by 224 pixel images, the remaining 3 are the RBG color channels
    resolution = [224, 224, 3]

    # GP params
    dev = '/gpu:0'  # device to run, write '/cpu_0' to tun on cpu
    number_generations = 10
    pop_size = 50
    tour_size = 3
    mut_prob = 0.1
    cross_prob = 0.9
    max_tree_dep = 10
    # tell the engine that the RGB does not explicitly make part of the terminal set
    edims = 2

    # Initialize NIMA classifier
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)
    model = Model(base_model.input, x)
    model.load_weights('weights/weights_mobilenet_aesthetic_0.07.hdf5')

    #seed = random.randint(0, 0x7fffffff)
    # seed = 39485793482  # reproducibility
    seed = 2020

    # create engine
    engine = Engine(fitness_func=nima_classifier,
                    population_size=pop_size,
                    tournament_size=tour_size,
                    mutation_rate=mut_prob,
                    crossover_rate=cross_prob,
                    max_tree_depth = max_tree_dep,
                    target_dims=resolution,
                    method='ramped half-and-half',
                    objective='maximizing',
                    device=dev,
                    stop_criteria='generation',
                    codomain = [0, 1],
                    domain=[0, 1],
                    do_final_transform = True,
                    final_transform = [0, 255],
                    stop_value=number_generations,
                    effective_dims = edims,
                    seed = seed,
                    debug=0,
                    save_to_file=1, # save all images from each 10 generations
                    save_graphics=True,
                    show_graphics=False,
                    read_init_pop_from_file=None)

    # This experiment is comparatively slower, but bear inmind that the NIMA classifier takes
    # a considerable amount of time
    # run evolutionary process
    engine.run()
