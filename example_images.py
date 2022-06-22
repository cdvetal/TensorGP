from tensorgp.tensorgp_v2 import *

if __name__ == "__main__":

    # This example uses TensorGP not as an evolutionary tool but to evaluate expressions
    # converting them into images

    resolution = [256, 256]
    Engine(target_dims = resolution).generate_pop_images('images_file.txt')


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
