# TensorGP
**A data parallel approach to Genetic Programming**

TensorGP is a general-purpose Genetic Programming engine that uses TensorFlow with Python to accelerate fitness evaluation through operator vectorization.

Even though this framework was designed with speed and efficiency in mind, the simplistic design and flexibility allow for the quick prototyping of controlled environments and experimental setups.

## Installation

Import the engine with:

```python
from tensorgp.engine import *
```

You can use the [pip](https://pip.pypa.io/en/stable/) package manager to install the list of dependencies provided in the "requirements.txt" file:
```bash
pip install -r requirements.txt
```

## Getting Started

To complement this startup guide, we encourage you to check the Python files starting with **"example_"** as they can be used as templates for your experiments. These files exemplify different TensorGP use cases:

 - **"example_symreg.py"** - A typical symbolic regression scenario, probably one of the most common use cases of GP. This example is written in a test suite style that shows how easy it is to define different problems and scenarios.
 - **"example_evoart.py"** - TensorGP is not restricted to symbolic regression problems. This example uses a classifier to produce visually appealing images. Additionally, the fitness function in this example demonstrates how you save an individual as an image.
 -  **"example_images.py"** - You can also use TensorGP as an evaluation engine and generate a set of image phenotypes for a set of individuals represented as expressions.

The implementation of any evolutionary experimentation with TensorGP can be summarized in 3 steps:

### 1 - Define the engine call

Here you will provide a function set for your experiment as well as other GP parameters needed. 
To define a costum function set we need to define the operators that your setup requires:

```python
my_fset = {'add', 'sub', 'mult', 'div', 'tan', 'sin', 'cos'}
```
... then you can pass the enumeration to the initial engine call as so:

```python
engine = Engine(operators = m_fset,
		... # other parameters,
		)
```
If you don't provide your own function set, the engine will default to using every operator that is implemented internally.
To see a full list with all the operators that are supported out of the box or if you wish to write your own custom operator, refer to the "Features" section.
You may also check the Parameterization section to see all the available parameters available for the initial engine call.

### 2 - Write your fitness function

This function will assess the quality of every individual in your population for every generation of the evolutionary process.
For performance and flexibility concerns, you will have to loop through all individuals and calculate the fitness of each one, rather than defining how to assess each solution individually.
This function provides you with a bunch of engine variables that you may access through kwargs:

```python
def my_fitness_function(**kwargs):  
	# read parameters  
	population = kwargs.get('population')  
    	generation = kwargs.get('generation')
	
	# loop through the population and assess fitness
```

Check the files starting with **"example_"** to adapt the fitness function to your particular use case.
You may also check the Parameterization section for a list of all parameters that you can access through `kwargs`.

After this function is defined, you simply pass it to the initial engine call as we did for the remaining parameters.
```python
engine = Engine(fitness_func=my_fitness_function, ...)
```

### 3 - Run the evolution
Now we are ready to start the evolutionary process by simply calling:

```python
engine.run()
```

## Features

This section details some of TensorGPs' core features and capabilities that you might find useful.

### Defining a target

If you are working with a problem where the optimal is known (such as in Symbolic regression), you must define a target to pass to the engine.

You can define this target by defining a Tensor in TensorFlow:

```python
import tensorflow as tf
import numpy as np

numpy_target = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
optimal_solution = tf.convert_to_tensor(numpy_target)

engine = Engine(target=optimal_solution ,  ...)
```

...or simply as an expression,  if that solution can be represented by the primitives defined by the primitibe set you are using:


```python
# approximate the function: f(x, y) = (x + y) / 2 for x, y defined in the problem domain
optimal_solution = 'div(add(x, y), scalar(2))'

engine = Engine(target=optimal_solution ,  ...)
```
Where ```x``` is the first variable of the terminal set, ```y```  the second, ```z``` the third, and so on.
This makes the integration of new problems with TensorGP, oftentimes, as trivial as adding one line of code (especially for Symbolic Regression problems)!

### File System and Logging
Every time ```Engine()``` is called, a new folder will be created in the **"runs"** directory that will contain information pertaining to that experiment.
In this directory, a subdirectory will be created for each generation where information regarding engine state and population statistics may be saved.

You may tell the engine to save its state in each generation by setting the ```write_log = True``` parameter and log population statistics (including information for individual) with ```write_gen_stats = True```.

Basic generational statistics are printed to the console for each generation in the experiment,
as well as information regarding elapsed engine time:

<img src="/images_dir/gitrun.png" alt="alt text" width="800">

This basic information is saved as a CSV file in the main experiment directory.
You may increase the verbosity of the information printed to the console with the ```debug``` parameter, which defaults to 0.

Additionally, you can also automatically generate graphics representing fitness and depth evolution across generations by setting ```save_graphics = True```.

<img src="/images_dir/gitgraficos.png" alt="alt text">

Again, refer to the Parameterization section for a full list of parameters.

### Custom Initial population

TensorGP implements the traditional stochastic methods for generating the initial population, such as Ramped-half-and-half, grow, and full. However, you can also write your own initial population in a file and pass it to the engine:

```python
engine = Engine(read_init_pop_from_file = 'pop.txt' ,  ...)
```
with **"pop.txt"** containing the individuals to evolve represented as expressions, one per line:

```txt
add(x, sub(y, scalar(1.5))
sqrt(add(mult(x, x), mult(y, y)))
scalar(1.5)
```
Be aware that variables defined in these expressions must be part of the terminal set defined by the engine.
If you are having trouble while reading the expressions, insert an additional newline at the end of the text file.

### Terminal set
If you don't define the terminal set yourself, a default one will be generated according to the dimensionality of your problems' domain.
As an example, if you have a two-dimensional domain, a terminal set containing the variables ```x``` and ```y``` for the first and second dimensions, respectively. This nomenclature follows alphabetical order ( ```x```,  ```y```,  ```z```,  ```a```, ```b```,  ```c```, ...) for higher dimensions.

You may define your own terminal set by calling the  ```Terminal_Set()``` constructor and passing it to the engine:

As a default, the variables needed to index a point within the problem domain are always defined, although you can add or delete every variable you want from the terminal set.

```python
indexing_variables = 2
domain_range = [256, 256] # two-dimensions
my_terminal_set = Terminal_set(indexing_variables, domain_range)

# Add an element to the set called var that is filled with ones (same thing as scalar(1))
my_terminal_set.add_to_set("var", tf.constant(1.0, shape=resolution))

# Remove the 'y' from the set (the tensor that indexes into the second dimension)
my_terminal_set.remove_from_set("y")

engine = Engine(terminal_set=my_terminal_set,  ...)
```
Where ```domain_range ``` defines the shape and size of the problem domain and ```indexing_variables``` defines how many dimensions you want to explicitly be able to index within your domain. If that sounded complicated don't worry, for most use cases just pass the number of dimensions of your domain. This can be useful if your domain has extra dimensionality for things like RGB colors channels, defined for example as  ```domain_range = [256, 256, 3]```, but you don't really want to have a ```z``` variable to index the color channels as part of the terminal set.  This is used in **"example_evoart.py"**, the only difference being that in that example the effective dimension is passed directly to the engine call using the ```effective_dims``` parameter.

Another important aspect to point out is that scalars/constants are not really part of the terminal set, even though they are defined as terminals by the engine. Instead, TensorGP resorts to user-defined probabilities for the frequency of generated scalars (see ```terminal_prob```, ```scalar_prob``` and ```uniform_scalar_prob``` in the parameters section). 

 
### Custom operators

You can define custom operators in TensorGP by defining a function as a composition of existing TensorFlow primitives:

```python
import tensorflow as tf

def rmse_node(child1, child2, dims=[])
	return tf.sqrt(tf.reduce_mean(tf.square(child1 - child2)))
```

The output should be a tf.float32 tensor (in this case there is no need to cast, the inputs are floats themselves).
Besides, `dims=[]` should be placed at the end of the argument list. The engine passes the domain range to this parameter in case it is needed while writing the operator (useful if you want to write define a scalar/constant: `tf.constant(1.0, shape=dims, dtype=tf.float32)`).

After this, the only thing to do is to define the new operator arity and add in to the function set:

```python
# Define subset of internally implemented operators
primitives = {'add', 'sub', 'mult', 'div'}
my_function_set = Function_Set(primitives)

# Add rmse_node, which has 2 arguments with name "rmse"
my_function_set.add_to_set("rmse", 2, rmse_node)

# You can also remove operators from the set
my_function_set-remove_from_set("mult")

engine = Engine(operators=my_function_set,  ...)
```


For a full list of TensorFlow primitives, check the official [website](https://www.tensorflow.org/api_docs/python/tf/all_symbols).

## Documentation
This section documents implemented GP operators, recombination methods as well as parameterizations available for both the fitness function and the engine itself.

###  Internal Operators
TensorGP provides an extensive set of operators that are implemented out of box, along with the necessary protection mechanism. Not to be restricted to Symbolic Regression and Classification applications, several image specific operators such as step functions are also provided. 

| Operator Name | Function | Arity | Type | Subtype | Functionality 
|--|--|--|--|--|:--:|
| add | Addition| 2 | Mathematical | Arithmetic | `x + y`
| sub| Subtraction| 2 | Mathematical | Arithmetic | `x - y`
| mult | Multiplication | 2 | Mathematical | Arithmetic | `x * y`
| div | Division | 2 | Mathematical | Arithmetic | `x / y` <br> `0 if denominator is 0`
| cos | Cosine | 1 | Mathematical | Trigonometric | `cos(pi * x)`
| sin | Sine | 1 | Mathematical | Trigonometric | `sin(pi * x)`
| tan | Tangent | 1 | Mathematical | Trigonometric | `tan(pi * x)`
| exp | Exponential | 1 | Mathematical | Others | `e ^ x`
| log | Logarithm | 1 | Mathematical | Others | `log(x)` <br> `-1 if x < 0`
| abs | Abs | 1 | Mathematical | Others | `-x if x < 0` <br> `x if x > 0`
| pow | Exponentiation | 2 | Mathematical | Others | `abs(x) ^ abs(y)` <br> `0 if x == 0`
| min | Minimum | 2 | Mathematical | Others | `min(x, y)`
| max | Maximum | 2 | Mathematical | Others | `max(x, y)`
| mdist | Average | 2 | Mathematical | Others | `(x + y) / 2`
| neg | Negative | 1 | Mathematical | Others | `-x`
| sqrt | Square Root | 1 | Mathematical | Others | `sqrt(x)`
| sign | Sign | 1 | Mathematical | Others | `-1 if x < 0` <br> `0 if x == 0` <br> `1 if x > 0`
| clip | Constrain | 3 | Mathematical | Others | ensure `y <= x <= z` <br> or `max(min(z, x), y)`
| mod | Modulo | 2 | Mathematical | Others | remainder of division <br> `x % y` 
| frac | Fractional part | 1 | Mathematical | Others | `x - floor(x)`
| if | Condition | 2 | Logic | Conditional | `if x then y else z`
| or | OR | 2 | Logic | Bitwise | logic value of `x | y` <br> for all bits
| xor | Exclusive OR | 2 | Logic | Bitwise | logic value of `x ^ y` <br> for all bits
| and | AND | 2 | Logic | Bitwise | logic value of `x & y` <br> for all bits
| warp | Warp | n | Image | Transform | Map data given input tensors [1]
| step | Nomral | 1 | Image | Step | `-1 if x < 0` <br> `1 if x >= 0`
| stepp | Smooth | 1 | Image | Step | `x^2(3-2*x)`
| sstepp | Perlin Smooth | 1 | Image | Step | `x^3(x(6x - 15) + 10)`
| len | Euclidean distance | 2 | Image | Color | `sqrt(x^2 + y^2)`
| lerp | Linear Interpolation | 3 | Image | Color | `x + (y - z) * frac(z)`

 [1] The warp operator is commonly used to deform images and is defined as a transformation that maps every element of a tensor to a different coordinate (two-dimensional warp is commonly to distort image shapes, see [Wikipedia](https://en.wikipedia.org/wiki/Image_warping)). This mapping is done according to an array of tensors with a size equal to the number of dimensions of the problem. Each of these tensors dictates where elements will end up within a given dimension. TensorGP implements a generalization of the warp operator that enables the mapping of any set of dimensions.

###  Fitness arguments
List of engine variables that can be accessed by the fitness function through Python `**kwargs`:

  >**generation**: integer
  >
  > Number of current generations.
  > Note: The generation counter starts at 0.

  >**population**: list of dictionaries
  >
  > Current population being evolved.
  > `population[i]['fitness']` accesses the fitness of the `i`th individual. Valid options for keys on each indivual are `depth`, `tree`, `fitness` and `nodes`.
  > For instance, if you want to print the expression of the `n`th individual, you can write:
  > ```python
  > population = kwargs.get('population')
  > 
  > print(population[n]['tree'].get_str())
  > ```

  >**tensors**: list of TensorFlow tensors
  >
  > These contain the tensor phenotypes of each individual (i.e. the `n`th tensor will be the result of evaluating the `n`th individual in `population` across the whole problem domain).

  >**f_path**: string
  >
  > Path to the current generation directory.
  > Note: This is useful if you wish to save some information from that generation.

  >**rng**: Random()
  >
  > A reference to the random number generator used by TensorGP.
  > This ensures reproducibility throughout the whole run, even if you wish to use randomness inside the fitness function.
  > All methods from the [Python random module](https://docs.python.org/3/library/random.html) can be accessed through `rng`:
  > ```python
  > rng = kwargs.get('rng')
  > 
  > k = rng.randint(0, 5)
  > ```

  >**objective**: string
  >
  > `minimization` if we are dealing with a minimization problem, `maximization` otherwise.
  > _Note_: this might get deprecated.

  >**resolution**: list of integers
  >
  > Shape and size of the problem domain (this is also the shape and size of the `tensors`).

  >**stf**: integer
  >
  > Generational interval that TensorGP uses to save information (see `save_to_file`). Can be coupled with `generation` to make a condition true every `stf` generations:
  > ```python
  > generation= kwargs.get('generation')
  > stf = kwargs.get('stf')
  > 
  > if generation % stf == 0:
  > 	# print or save some information
  > ```

  >**target**: TensorFlow tensor
  >
  > The target to approximate in tensor form.
  > This will be `None` if the problem has no target.

  >**debug**: integer
  >
  > The level of debugging used in the engine. See `debug` in the GP Parameters subsection;


### Evolutionary methods

The following is a list of mutation and crossover operators currently available. These methods are part of TensorGP and are not supposed to be used outside of an `Engine()` object. To see how you can use different mutation methods and control different parameters, refer to the `mutation_funcs` and `mutation_probs` in the "GP parameters" subsection.

  >`subtree_mutation`(_self_, _parent_):
  >
  > : A random subtree is select in the parent to be replaced by a randomly generated one with the same depth using the `grow` method. The root is a valid node to be selected, meaning that the whole tree can get replaced.
  > 
  > : **Parameters:**
  > A parent, represented by its tree-graph genotype. 
  > 
  > : **Returns:**
  > A mutated offspring, represented by its tree-graph genotype.

  >`point_mutation`(_self_, _parent_):
  >
  > : Selects a random node in the parent tree, replacing it by a primitive of the same arity.
  > Recursively, each child of the replaced node can also be replaced with a probability of 5%; 
  > If no primitive of the same arity exists in the corresponding set, then no modification is made.
  > If a terminal is chosen, it can be replaced by any other terminals including scalar/constants according to the `scalar_prob` parameter.
  > If a scalar/constant is selected to be generated, its values can either be random or not, according to the `uniform_scalar_prob` parameter.
  > 
  > : **Parameters:**
  > A parent, represented by its tree-graph genotype. 
  > 
  > : **Returns:**
  > A mutated offspring, represented by its tree-graph genotype.

  >`promotion_mutation`(_self_, _parent_):
  >
  > : Selects a random non-terminal node from the parent tree and replaces it with one of its children. This will also erase any other children of the older node.
  > This method never deletes terminals as terminals do not have children.
  > On average, this method decreases the number of nodes and possibly the depth when compared to the parent tree.
  > This is the opposite method to `demotion_mutation`.
  > 
  > : **Parameters:**
  > A parent, represented by its tree-graph genotype. 
  > 
  > : **Returns:**
  > A mutated offspring, represented by its tree-graph genotype.

  >`demotion_mutation`(_self_, _parent_):
  >
  > : Selects a random node (either terminal or non-terminal), replacing it with a randomly selected primitive from the function set. One of the new nodes' children will be the originally selected node, while the remaining children will be randomly generated terminals.
  > On average, this method increases the number of nodes and possibly the depth when compared to the parent tree.
  > This is the opposite method to `promotion_mutation`.
  > 
  > : **Parameters:**
  > A parent, represented by its tree-graph genotype. 
  > 
  > : **Returns:**
  > A mutated offspring, represented by its tree-graph genotype.

  >`crossover`(_self_, _parent1_, _parent2_):
  >
  > : This method implements the common subtree crossover algorithm.
  > Select a random node from _parent1_. This first selection has a 90% probability of choosing a non-terminal node and a 10% probability of choosing a terminal. Then the algorithm randomly selects a function node from _parent2_ and replaces the subtree rooted in this node with the one rooted in the previously selected node of _parent1_.
  > 
  > 
  > : **Parameters:**
  > Two parents, represented by their tree-graph genotype. 
  > 
  > : **Returns:**
  > An offspring that is the crossover between both parents, represented by its tree-graph genotype.

  >`tournament`(_self_, _parent1_, _parent2_):
  >
  > : Traditional tournament selection method.
  > Select a sample of size _k_ from the current population. Retrieve the best-fitted individual within that sample.
  > Here _k_ is the `tournament_size` parameter.
  > 
  > : **Parameters:**
  > 
  > : **Returns:**
  > The best-fitted individual of a randomly chosen sample of size `tournament_size` retrieved from the current population, represented by its tree-graph genotype.

###  GP parameters

These are the possible parameters for the `Engine()` initialization.

  >**fitness_func** (default=`None`): pointer to function or `None`, optional
  >
  > Pointer to the user-defined fitness function. 
  > Although this parameter is optional if you plan on running evolution by calling `run()`, you need to set this parameter.

  >**population_size** (default=`100`): integer, optional
  >
  > Number of individuals in the population at each generation.
 
  >**tournament_size** (default=`3`): integer, optional
  >
  > Number of individuals that participate in tournament selection.
  > The larger this value the greater the evolutionary pressure.

  >**mutation_rate** (default=`0.15`): float, optional
  >
  > Probability of applying the selected mutation operators to an individual once a generation.
 
  >**mutation_funcs** (default=`None`): list of function pointers or `None`, optional
  >
  > Non-empty list of mutation methods to select from while mutation an individual.
  > If `None`, the set of methods to choose from will be the ones internally implemented by the engine

  >**mutation_probs** (default=`None`): list of floats or `None`, optional
  >
  > Non-empty list of floats relating to the frequency of choosing the mutation method of the same index defined in `mutation_funcs`.
  > The sum of all values should amount to 1. The engine computes the reduced sum for the whole array, capping the sum at 1.0, meaning that any methods whose corresponding cumulative sum surpass this value are ignored.
  > If `None`, each method will be assigned the same probability such that the random selection is uniform amongst all methods defined in  `mutation_funcs`.
  > If the probability is equal or lesser than 0, the corresponding mutation method is ignored.
  > If the list is shorter than the one defined in `mutation_funcs`, the methods corresponding to the higher indices are ignored.
  > If the list is larger than the one defined in `mutation_funcs`, the remaining probabilities are ignored.

  >**crossover_rate** (default=`0.9`): float, optional
  >
  > Probability of applying crossover to an individual once a generation.

  >**elitism** (default=`1`): integer, optional
  >
  > Number of best-fitted individuals to automatically pass to the next generation.
  > This value is clamped between `0` and `population_size`. 
  > The higher this value, the greater the evolutionary pressure.

  >**max_tree_depth** (default=`8`): integer, optional
  >
  > Maximum allowed depth for the tree representation of an individual.

  >**min_tree_depth** (default=`-1`): integer, optional
  >
  > Minimum allowed depth for the tree representation of an individual.
  > -1 is the same as 0, meaning that there is no lower limit (depth is counted as edges, not nodes).


  >**max_init_depth** (default=`None`): integer or `None`, optional
  >
  > Maximum allowed depth for the tree representation of any individual in the initial population.
  > If `None`, the parameter is set to the same value as `max_tree_depth`.


  >**min_init_depth** (default=`None`): integer or `None`, optional
  >
  > Maximum allowed depth for the tree representation of any individual in the initial population.
  >  If `None`, the parameter is set to the same value as `min_tree_depth`.

  >**method** (default=`'ramped half-and-half'`): default, `'full'` or `'grow'`, optional
  > 
  > Method used to generate the initial population.
  > This parameter is ignored if reading the population from a file.
  > *Note*: Typically, there are two ways of implementing `'ramped half-and-half'`: one that works over a single tree, generating a root node and creating half of the tree with the grow method using full for the other half, and one that works over the entire population, dividing it into blocks of different depths splitting the number of trees in each block to use either the full or grow method. TensorGP uses the second approach.

  >**terminal_prob** (default=`0.2`): float, optional
  > 
  > Probability of choosing any terminal from the function set. All scalars/constants are counted as only one terminal.

  >**scalar_prob** (default=`0.55`): float, optional
  > 
  > Probability of generating a scalar/constant while selecting a terminal. All the remaining terminals have uniform probabilities to be chosen.

  >**uniform_scalar_prob** (default=`0.7`): float, optional
  > 
  > Probability that the scalar/constant generated is constant across the shole domain.
  > *Note* A scalar/constant may not be constant through the whole domain if the `effective_dims` parameter is lower than the dimensionality of the problem. For instance, following the example of the `Terminal` subsection of the `Features` section, `scalar(255.0, 0.0, 0.0)` may correspond to a constant that translates to the color red, but is not a "uniform scalar" because it is defined as having the value 255 throughout the first index of the last dimension and 0 for the remaining values.  In this case, a uniform scalar would be `scalar(255.0, 255.0, 255.0)`, or just `scalar(255.0)` (the last value gets replicated to meet the arity of the scalar).
  
  >**stop_criteria** (default=`'generation'`): default or `'fitness'`, optional
  > 
  > If `'generation'` is chosen, evolution will stop after a number of generations determined by `stop_value`.
  > If `'fitness'` is chosen, evolution will stop if the error from the best-fitted to the target is less than the one determined by `stop_value`. This is ignored if no target is provided.

  >**stop_value** (default=`10`): integer, optional
  > 
  > Values that stop the evolutionary process according to the `stop_criteria` 

  >**objective** (default=`'minimizing'`): default or `maximizing`, optional
  > 
  > This is really just a commodity feature and might get deprecated in the future, as the fitness function can be modified as easily as changing `fitness` to be `-fitness` in the fitness function.

  >**min_domain** (default=`-1`): integer, optional
  > 
  > Lower bound for values in the problem domain.

  >**max_domain** (default=`1`): integer, optional
  > 
  > Upper bound for values in the problem domain.

  >**const_range** (default=`None`): list of floats or `None`, optional
  > 
  > Non-empty list of floats that dictates a maximum and minimum that the scalar/constants of the terminal can take. The minimum value is set to be the minimum value existing in the list (same idea for the maximum).
  > If `None` The maximum and minimum values default to `min_domain` and `max_domain`, respectively.

  >**effective_dims** (default=`None`): integer or `None`, optional
  > 
  > Number of dimensions to explicitly index within the problem domain. In most cases, this will be equal to the dimensionality of the domain (i.e. the length of the `target_dims` parameter). Refer to the `Terminal` subsection of the `Features` section to see use cases for this parameter.
  > If `None` this will be equal to the problems' dimensionality.
 
  >**operators** (default=`None`): list or `None`, optional
  > 
  > Non-empty list of operators to be drawn from the function set while evolving candidate solutions.
  > If `None`, the engine will define a function set with all the operators that are internally implemented by TensorGP (check the "Internal Operators" subsection).
 
  >**function_set** (default=`None`): Function_Set or `None`, optional
  > 
  > Function set to be used by the engine.
  > If not `None`, the `operators` parameter will be ignored.

   >**terminal_set** (default=`None`): Terminal_Set or `None`, optional
  > 
  > Terminal set to be used by the engine.
  > If `None`, a terminal set will be created with the variables needed to index all dimensions of the problem domain, unless dictated otherwise by the `effective_dims` parameter.

  >**immigration** (default=`float('inf')`): integer, optional
  > 
  > Insert random individuals into the population every `n`th generation, `n` being  `immigration`.
  > This can be an efficient way of escaping fitness plateaus.

  >**target_dims** (default=`None`): list of integers or `None`, optional
  > 
  > Non-empty list of integers defining the size and shape of the problem domain.
  > If `None`, TensorGP defaults to a two-dimensional domain of 128 by 128 (`[128, 128]`).
 
  >**target** (default=`None`): string, TensorFlow tensor or `None`, optional
  > 
  > Defines a target for candidate solutions to approximate. Typically this is the optimal solution to your problem. Refer to the "Defining a target" subsection of the "Features" section.

  >**max_nodes** (default=`-1`): integer, optional
  > 
  > Maximum number of nodes for generated trees.
  > If `-1` then there is no limit.

  >**debug** (default=`0`): integer, optional
  > 
  > Verbosity level of console output. `0` print basic statistics starting with few messages regarding recognized devices and engine seed as well as engine timers at the end.
  > The basic statistics also prints the generation number as well as the best, average, and deviation of fitness, depth, and the number of nodes for each individual at each generation.

  >**show_graphics** (default=`True`): boolean, optional
  > 
  > Show a graphical representation for the depth and fitness evolution of individuals throughout across generations.

  >**save_graphics** (default=`True`): boolean, optional
  > 
  > Save a graphical representation for the depth and fitness evolution of individuals throughout across generations.

  >**device** (default=`/cpu:0`): string, optional
  > 
  > Device with which to run the evaluation phase of individuals.
  > To use your main GPU define this parameter as `/gpu:0`.
  > Refer to the [TensorFlow list_physical_devices function](https://www.tensorflow.org/api_docs/python/tf/config/list_physical_devices) to check the devices recognized by TensorFlow in your machine.

  >**initial_test_device** (default=`True`): boolean, optional
  > 
  > Whether or not to test the device specified in `device`.
 
  >**save_to_file** (default=`10`): integer, optional
  > 
  > Generational Interval to save information provided by `write_log` and `write_gen_stats`.

  >**write_log** (default=`True`): boolean, optional
  > 
  > If `True`, the engine will save information regarding the engine state every `n`th generation, `n` being the `save_to_file` parameter.

  >**write_gen_stats** (default=`True`): boolean, optional
  > 
  > If `True`, the engine will save information regarding every individual in the population every `n`th generation, `n` being the `save_to_file` parameter.

  >**previous_state** (default=`None`): dictionary or None, optional
  > 
  > If not `None`, the engine will override the current engine state with the state provided.
  > WARNING: This feature is a Work In Progress and is still not fully implemented.

  >**var_func** (default=`None`): function pointer or None, optional
  > 
  > Pointer to the function used to initialize the index variables of the terminal set. 
  > If `None`, TensorGP defaults to a function that defines a linearly spaced grid of values across the defined domain.
  > Instead, if you wish to define a uniform random selection of points within the problem domain, you may reference the `uniform_sampling`, already implemented by TensorGP.

  >**read_init_pop_from_file** (default=`None`): string or None, optional
  > 
  > Defines a text file containing expressions used to generate the individuals of the initial population.
  > If `None`, then the initial population will be randomly generated using the algorithm selected by the `method` parameter.
  
## Known Issues

(as of 14/02/21)
Be aware that TensorFlow is still in development and, as so, some issues are still not solved:

- If your tensors are large or if you are evaluating a large number of individuals while executing in a GPU, it might happen that there TensorFlow raises an OOM error while trying to fit the entire tensor data in the GPU VRAM. Unfortunately, TensorFlow does not provide an effective way to batch the information to be sent to the GPU, limit VRAM usage nor clear the GPU VRAM with TF 2.x. For this reason, if you have this issue, your best bet is to use the CPU instead (device = `/cpu:0`).
- Some machines might get an error while reading expressions from a file, a case in which you should add a blank space at the end of the file.

## Citing this project
Authors of academic papers that use TensorGP for their experimenation are encouraged to cite the following paper:
```
@inproceedings{baeta2021tensorgp,
	title = {TensorGP -- Genetic Programming Engine in TensorFlow},
	author = {Baeta, Francisco and Correia, Jo{\~{a}}o and Martins, Tiago and Machado, Penousal},
	booktitle = {Applications of Evolutionary Computation - 24th International Conference, EvoApplications 2021},
	year = {2021},
	organization={Springer}
}
```

## Contact

In case you are having trouble with a specific experimental setup (and already read the documentation), or if you have any suggestion/feedback about TensorGP you may contact:

**fjrbaeta@student.dei.uc.pt**
