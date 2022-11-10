from tensorgp.engine_scarlett import *

# random fitness to exemplify
def rand_fit(**kwargs):
    population = kwargs.get('population')
    for i in range(len(population)):
        population[i]['fitness'] = random.random()
    return population, 0

if __name__ == "__main__":

    # load state.log file, start with population stored in "test_default.txt"
    engine = load_engine(fitness_func = rand_fit, pop_source="test_default.txt", file_name = 'state.log')

    # write the engine summary after loading, should be the same
    engine.summary(force_print=True, log_format=True, write_file=True, file_path='', file_name = 'new_state.log')

    # run the engine
    engine.run()
