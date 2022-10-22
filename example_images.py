from tensorgp.engine import *

if __name__ == "__main__":

    # This example uses TensorGP not as an evolutionary tool but to evaluate expressions
    # converting them into images

    resolution = [512, 512, 3]

    # Generate from pop file
    Engine(target_dims = resolution, effective_dims=2,).generate_pop_images('test_default.txt', 'images_dir/')

    # Generate from List of strs
    # expr = "tan(add(max(log(xor(log(add(pow(mult(exp(exp(y)), pow(sub(x, mult(y, scalar(-0.6347885638222202, 0.6011813921240905, 0.5137609112895838))), tan(x))), scalar(0.7453392097176601)), neg(x))), _and(cos(scalar(-0.6343768495230879)), y))), xor(abs(warp(y, x, scalar(-0.8560255538223858, 0.8518172740714811, -0.09226456331173094))), sin(scalar(-0.7513214358991382, 0.5814230533953741, 0.7537424409867375)))), y))"
    # Engine(target_dims = resolution, effective_dims=2).generate_pop_images([expr], 'images_dir/')

    """
    Notes:
    Terminals variables in the expression should be "x", "y", "z" ,... and so on
    You should input a resolution for the generated images (else a default of 128 by 128 is used)
    The number of variables that the expression contains should be in line with the dimensionality of the domain
    The function defines an input to read the expressions, you can also write expressions directly as a list pf strings:
    
    Engine().generate_pop_images(['add(sub(x, y), scalar(1.0))', 'cos(sin(x))'], 'images_dir/')
    
    If you wish to save the images inside the run directory itself, just drop the file argument as so:
    
    Engine().generate_pop_images('images_file.txt')
    
    """
