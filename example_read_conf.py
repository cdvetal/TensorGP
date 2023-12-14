from tensorgp.engine import *

# random fitness to exemplify
def rand_fit(**kwargs):
    population = kwargs.get('population')
    for i in range(len(population)):
        population[i]['fitness'] = random.random()
    return population, 0


exp_prefix = ""
fixed_path = "fixed_example"


def initial_engine():
    # create the initial engine with  folder structure
    engine = Engine(fitness_func=rand_fit,
                    population_size=10,
                    tournament_size=3,
                    mutation_rate=0.1,
                    crossover_rate=0.9,
                    max_tree_depth=14,
                    target_dims=[128, 128],
                    method='ramped half-and-half',
                    max_init_depth=6,
                    objective='minimizing',
                    device='cuda',
                    stop_criteria='generation',
                    stop_value=gens,
                    effective_dims=2,
                    domain = [-5, 5],
                    seed=seed,

                    exp_prefix=exp_prefix,
                    fixed_path=fixed_path,

                    save_graphics=False,
                    read_init_pop_from_file=None)

    # run the engine
    engine.run()

if __name__ == "__main__":

    # load state.log file, start with population stored in "test_default.txt"
    #engine = load_engine(fitness_func = rand_fit, pop_source="test_default.txt", file_name = 'state.log')
    seed = 39485793482  # reproducibility
    gens = 5

    # experiment with fixed path (replace)
    exp_path = "C://Users//fjrba//OneDrive//Documentos//TensorGP-dev//runs//" + exp_prefix + "//" + fixed_path + "//"

    print(exp_path)

    # comment to restart run!
    initial_engine()

    # load state.log file
    engine1 = load_engine(fitness_func=rand_fit,
                          pop_source=None,
                          #file_path=engine.experiment.working_directory,
                          file_path=exp_path,
                          file_name="state.log")

    # write the engine summary after loading, should be the same
    engine1.run(stop_value=gens)

