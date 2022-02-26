from tensorgp.diff_engine import *

def fit_teste(**kwargs):
    # read parameters
    population = kwargs.get('population')
    target = kwargs.get('target')
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
    tensors = [p['tensor'] for p in population]

    # scores
    for index in range(len(tensors)):
        #fit = random.random() ## random!
        fit = tf_rmse(tensors[index], target).numpy()

        if condition():
            max_fit = fit
            best_ind = index
        fitness.append(fit)
        population[index]['fitness'] = fit

    # save best indiv
    #if images: save_image(tensors[best_ind], best_ind, fn, _resolution, addon='_best')
    return population, best_ind

# if no function set is provided, the engine will use all internally available operators:
#fset = {'abs', 'add', 'and', 'clip', 'cos', 'div', 'exp', 'frac', 'if', 'len', 'lerp', 'log',
#        'max', 'mdist', 'min', 'mod', 'mult', 'neg', 'or', 'pow', 'sign', 'sin', 'sqrt', 'sstep',
#        'sstepp', 'step', 'sub', 'tan', 'warp', 'xor'}


if __name__ == "__main__":

    #resolution = [28, 28, 1]
    resolution = [28, 28]

    # GP params (teste super simples)
    dev = '/cpu:0'  # device to run, write '/cpu_0' to tun on cpu
    number_generations = 10
    pop_size = 10
    tour_size = 3
    mut_prob = 0.1
    cross_prob = 0.95
    max_tree_dep = 10
    # tell the engine that the RGB does not explicitly make part of the terminal set
    edims = 2

    seed = 2020  # reproducibility
    pagie = "add(div(scalar(1.0), add(scalar(1.0), div(scalar(1.0), mult(mult(x, x), mult(x, x))))), div(scalar(1.0), add(scalar(1.0), div(scalar(1.0), mult(mult(y, y), mult(y, y))))))"

    # create engine
    engine = Engine(fitness_func=fit_teste,
                    population_size=pop_size,
                    tournament_size=tour_size,
                    mutation_rate=mut_prob,
                    crossover_rate=cross_prob,
                    max_tree_depth = max_tree_dep,
                    target_dims=resolution,
                    target = pagie,
                    #method='grow',
                    method='ramped half-and-half',
                    objective='minimizing',
                    device=dev,
                    stop_criteria='generation',
                    min_init_depth=0,
                    max_init_depth=10,
                    bloat_control='dynamic_dep',
                    bloat_mode='depth',
                    dynamic_limit = 10,
                    elitism=1,
                    stop_value=number_generations,
                    effective_dims = edims,
                    seed = seed,
                    debug=0,
                    differentiable=True,
                    save_to_file=3, # save all images from each 10 generations
                    save_graphics=True,
                    show_graphics=False,
                    write_gen_stats=False,
                    write_log = False,
                    write_final_pop = True,
                    read_init_pop_from_file = None)
    #read_init_pop_from_file = "/home/scarlett/Documents/TensorGP/TensorGP-master/runs/run__2021_11_19__18_30_27_780__107305148598124544__images/run__2021_11_19__18_30_27_830__107305148598124544_final_pop.txt") # read predefined pop


    data, last_tensors = engine.run()
    print("First run over!")
    #_, _tensors = engine.run(number_generations)
    #print("Second run over!")

    # TODO: in-depth backpropagation test
