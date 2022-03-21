from tensorgp.pytorch_diff_engine import *

# Fitness function to calculate RMSE from target (Pagie Polynomial)
def fit_test(**kwargs):
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
    condition = lambda: (fit < max_fit)  # minimizing
    max_fit = float('inf')

    fn = f_path + "gen" + str(generation).zfill(5)
    fitness = []
    best_ind = 0
    tensors = [p['tensor'] for p in population]

    # scores
    for index in range(len(tensors)):
        #if generation % _stf == 0: save_image(tensors[index], index, fn, _resolution) # save image
        #print(tensors[index])
        # fit = mean - std
        #fit = random.random() ## random!
        fit = tensor_rmse(tensors[index], target).data


        if condition():
            max_fit = fit
            best_ind = index
        fitness.append(fit)
        population[index]['fitness'] = fit

    # save best indiv
    #if images: save_image(tensors[best_ind], best_ind, fn, _resolution, addon='_best')
    return population, best_ind


# Different types of function sets
extended_fset = {'max', 'min', 'abs', 'add', 'and', 'or', 'mult', 'sub', 'xor', 'neg', 'cos', 'sin', 'tan', 'sqrt', 'div', 'exp', 'log'}
simple_set = {'add', 'sub', 'mult', 'div', 'sin', 'tan', 'cos'}
normal_set = {'add', 'mult', 'sub', 'div', 'cos', 'sin', 'tan', 'abs', 'sign', 'pow'}


if __name__ == "__main__":

    # GP params
    dev = 'cpu'  # device to run, write '/cpu_0' to tun on cpu
    gens = 5  # 50
    pop_size = 50  # 50
    tour_size = 3
    mut_rate = 0.1
    cross_rate = 0.9

    minid = 2
    maxid = 6
    mintd = 2
    maxtd = 15

    elite_size = 1 # 0 to turn off

    # problems
    pagie = "add(div(scalar(1.0), add(scalar(1.0), div(scalar(1.0), mult(mult(x, x), mult(x, x))))), div(scalar(1.0), add(scalar(1.0), div(scalar(1.0), mult(mult(y, y), mult(y, y))))))"
    target= 'add(div(scalar(1.0), add(scalar(1.0), mult(mult(x, x), mult(x, x)))), div(scalar(1.0), add(scalar(1.0), mult(mult(y, y), mult(y, y)))))'
    keijzer11 = "add(mult(x, y), sin(mult(sub(x, scalar(1.0), sub(y, scalar(1.0)))))"
    korns3 = "add(scalar(-5.41), mult(scalar(4.9), div(sub(v, add(x, div(y, w))), mult(scalar(3.0, w)))))"

    problem = pagie  # Add to run more problems

    # Domains dimensions
    res = [1024, 1024]
    fset = extended_fset

    #seed = random.randint(0, 0x7fffffff)
    seed = 39485793482 # reproducibility
    engine = Engine(fitness_func=fit_test,
                    population_size=pop_size,
                    tournament_size=tour_size,
                    mutation_rate=mut_rate,
                    crossover_rate=cross_rate,
                    min_init_depth=minid,
                    max_init_depth=maxid,
                    min_tree_depth=mintd,
                    max_tree_depth=maxtd,
                    target_dims=res,
                    target=pagie,
                    #elitism=elite_size,
                    method='ramped half-and-half',
                    objective='minimizing',
                    bloat_control="off",
                    device=dev,
                    stop_criteria='generation',
                    stop_value=gens,
                    effective_dims=2,
                    min_domain=-5,
                    max_domain=5,
                    operators=fset,
                    seed=seed,
                    debug=0,
                    save_to_file=10,
                    save_graphics=True,
                    show_graphics=False,
                    write_gen_stats=False,
                    write_log=False,
                    write_final_pop=True,
                    read_init_pop_from_file=None)

    # create engine

    # run evolutionary process
    engine.run()