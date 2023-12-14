import tensorflow as tf

from tensorgp.engine_2 import *

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import PIL
from heapq import nsmallest, nlargest
from tensorflow.keras import layers
import time
from keras.models import load_model
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray

gen_image_cnt = 0
fake_image_cnt = 0

# function sets available
full_set = {'abs', 'add', 'and', 'clip', 'cos', 'div', 'exp', 'frac', 'if', 'len', 'lerp', 'log', 'max', 'mdist',
            'min', 'mod', 'mult', 'neg', 'or', 'pow', 'sign', 'sin', 'sqrt', 'sstep', 'sstepp', 'step', 'sub', 'tan',
            'warp', 'xor'}
extended_set = {'max', 'min', 'abs', 'add', 'and', 'or', 'mult', 'sub', 'xor', 'neg', 'cos', 'sin', 'tan', 'sqrt',
                'div', 'exp', 'log', 'warp'}
simple_set = {'add', 'sub', 'mult', 'div', 'sin', 'tan', 'cos'}
normal_set = {'add', 'mult', 'sub', 'div', 'cos', 'sin', 'tan', 'abs', 'sign', 'pow'}
# custom_set = {'sstep', 'add', 'sub', 'mult', 'div', 'sin', 'tan', 'cos', 'log', 'warp'}
custom_set = {'add', 'cos', 'div', 'if', 'min', 'mult', 'sin', 'sub', 'tan', 'warp'}
#Function set +, −,  * , /, min, max, abs, neg, warp, sign, sqrt, pow, mdist, sin, cos, if
std_set = {'add', 'sub', 'mult', 'div', 'sin', 'cos', 'min', 'max', 'abs', 'neg', 'warp', 'sign', 'sqrt', 'pow', 'mdist', 'if'}

cnn_model = load_model('MNIST_keras_CNN.h5')



class dcgan(object):

    def __init__(self,
                 batch_size=32,
                 gens_per_batch=100,
                 archive_size = 100,
                 archive_stf = 1,
                 starchive = 1,
                 do_archive = False,
                 digits_to_train=None,
                 run_from_last_pop=True,
                 linear_gens_per_batch=False,
                 log_losses=True,
                 seed=202020212022,
                 log_digits_class=True,
                 fset=None,
                 run_dir=None,
                 gp_fp=None,
                 archive_dir=None):

        self.seed = seed
        tf.random.set_seed(self.seed)

        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.input_shape = [self.img_rows, self.img_cols, self.channels]

        self.do_archive = do_archive
        self.archive = []
        self.starchive = starchive
        self.archive_size = archive_size
        self.archive_stf = archive_stf
        self.log_losses = log_losses
        self.log_digits_class = log_digits_class

        temp = sys.argv[1] if len(sys.argv) > 1 else ""
        pref = datetime.datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S_%f')[:-3] + "_" + temp + "_" + str(digits_to_train)
        # print(date)
        self.run_dir = os.getcwd() + delimiter + "gp_dcgan_results" + delimiter + "run__" + pref + delimiter if run_dir is None else run_dir
        self.gp_fp = self.run_dir + "gp" + delimiter if gp_fp is None else gp_fp
        self.gan_images = self.run_dir + "dcgan_images" + delimiter
        self.archive_dir = self.run_dir + "archive" + delimiter if archive_dir is None else archive_dir
        self.run_from_last_pop = run_from_last_pop
        self.linear_gens_per_batch = linear_gens_per_batch

        # os.makedirs(self.run_dir)
        # print("Created dir: ", self.run_dir)

        self.batch_size = batch_size
        self.gens_per_batch = gens_per_batch
        self.last_gen_imgs = []

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.discriminator = self.make_discriminator_model()
        self.disc_optimizer = tf.keras.optimizers.Adam(1e-4)
        resolution = [self.img_rows, self.img_cols]
        self.fset = normal_set if fset is None else fset
        stop_value = self.gens_per_batch - 1 if self.linear_gens_per_batch else 4
        self.generator = Engine(fitness_func=self.disc_forward_pass,
                                population_size=self.batch_size,
                                tournament_size=2,
                                mutation_rate=0.3,
                                crossover_rate=0.8,
                                max_tree_depth=14,
                                target_dims=resolution,
                                method='ramped half-and-half',
                                objective='maximizing',
                                device='/gpu:0',
                                stop_criteria='generation',
                                domain_mode='log',
                                operators=self.fset,
                                min_init_depth=3,
                                max_init_depth=6,
                                terminal_prob=0.5,
                                min_domain=-1,
                                max_domain=1,
                                bloat_control='off',
                                elitism=1,
                                stop_value=stop_value,
                                effective_dims=2,
                                seed=self.seed,
                                debug=0,
                                save_to_file=10000,
                                gen_display_step=10,
                                minimal_print=True, # True
                                save_graphics=False, # True
                                show_graphics=False, #False
                                write_gen_stats=True, # True
                                write_log=True, # True
                                write_final_pop=False, # True
                                stats_file_path=self.gp_fp,
                                graphics_file_path=self.run_dir,
                                run_dir_path=self.gp_fp,
                                read_init_pop_from_file=None,
                                mutation_funcs=[Engine.subtree_mutation, Engine.point_mutation,
                                                Engine.delete_mutation, Engine.insert_mutation],
                                mutation_probs=[0.6, 0.2, 0.1, 0.1]
                                )


        os.makedirs(self.gan_images)
        if self.do_archive:
            os.makedirs(self.archive_dir)

        self.gloss = 0
        self.dloss = 0
        self.training_time = 0
        self.loss_hist = []

        self.digits_to_train = digits_to_train if digits_to_train is not None else [0 for i in range(10)]
        (self.x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
        train_mask = np.isin(y_train, self.digits_to_train)
        self.x_train = self.x_train[train_mask]
        #self.x_train = self.x_train[:64] # testing purposes

        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, self.channels).astype(
            'float32')
        self.x_train = (self.x_train - 127.5) / 127.5  # Normalize the images to [-1, 1]
        print("Len of selected dataset: ", len(self.x_train))
        self.x_train = tf.data.Dataset.from_tensor_slices(self.x_train).shuffle(len(self.x_train)).batch(self.batch_size)
        #print(self.x_train.shape)


    def disc_forward_pass(self, **kwargs):
        population = kwargs.get('population')
        #generation = kwargs.get('generation')
        #tensors = kwargs.get('tensors')
        _resolution = kwargs.get('resolution')

        fit = 0
        max_fit = float('-inf')

        fitness = []
        best_ind = 0
        tensors = [p['tensor'] for p in population]

        # TODO: is predict okay here?
        fit_array = self.discriminator(np.array(np.expand_dims(tensors, axis=3)), training=False)
        # scores
        for index in range(len(tensors)):
            fit = float(fit_array[index][0])

            if fit > max_fit:
                max_fit = fit
                best_ind = index
            fitness.append(fit)
            population[index]['fitness'] = fit

        return population, best_ind

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=self.input_shape))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        return model

    def compute_losses(self, gen_output, real_output):
        gen_loss = self.cross_entropy(tf.zeros_like(gen_output), gen_output)
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        self.dloss = gen_loss + real_loss
        self.gloss = -self.dloss
        self.loss_hist.append([self.dloss.numpy(), self.gloss.numpy()])

    def print_training_hist(self):
        for h in self.loss_hist:
            print(h)

    def train_step(self, images, step):

        #index = np.random.randint(0, self.x_train.shape[0], self.batch_size)
        #images = self.x_train[index]
        global gen_image_cnt, fake_image_cnt
        fake_image_cnt += len(images)

        with tf.GradientTape() as disc_tape:
            ep = self.gens_per_batch if self.linear_gens_per_batch else round(step / 10) + 5
            starchive = self.starchive if step == 0 else 0
            gen_image_cnt += self.batch_size * ep
            _, generated_images = self.generator.run(stop_value=ep,
                                                    start_from_last_pop=self.run_from_last_pop,
                                                    start_from_archive = starchive,
                                                    archive = self.archive)


            # rollling archive
            if self.do_archive:
                get_pop = [copy.deepcopy(p) for p in self.generator.population]
                self.archive = nlargest(min(self.archive_size, self.generator.population_size + len(self.archive)), self.archive + get_pop, key=itemgetter('fitness'))

            # tf.debugging.assert_greater_equal(generated_images, -1.0, message="Less than min domain!")
            # tf.debugging.assert_less_equal(generated_images, 1.0, message="Grater than max domain!")

            self.last_gen_imgs = np.expand_dims(generated_images, axis=3)
            classify_digits(self.last_gen_imgs)

            #(self.last_gen_imgs.shape)
            gen_output = self.discriminator(self.last_gen_imgs, training=True)
            real_output = self.discriminator(images, training=True)

            self.compute_losses(gen_output, real_output)

            gradients_of_discriminator = disc_tape.gradient(self.dloss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, epochs = 1):
        start = time.time()

        for epoch in range(epochs):
            step = 0
            for images in self.x_train:
                self.train_step(images, step)
                if self.log_losses: self.write_losses_epochs(step, epoch)
                if self.log_digits_class:
                    self.write_digits_classifications(step, epoch, self.last_gen_imgs)

                # for image_batch in self.dataset:

                # Save the model every 15 epochs
                #self.generate_and_save_images(step + 1, epoch + 1)
                step += 1

                print('[DCGAN - step {}/{} of epoch {}/{}]:\t[Gloss, Dloss]: [{}, {}]\tTime so far: {} sec'.format(step, len(self.x_train),
                                                                                                                   epoch + 1, epochs, self.gloss,
                                                                                                                   self.dloss, time.time() - start))
            # Generate after the final epoch
            #self.generate_and_save_images(step + 1, epoch + 1)

            if self.do_archive and ((epoch + 1) % self.archive_stf) == 0:
                print("Saving archive...")
                self.write_archive(epoch)

        self.training_time = time.time() - start
        #self.plot_losses()
        return self.training_time, self.loss_hist


    def print_archive(self):
        for p in self.archive:
            print(p['fitness'])


    def generate_and_save_images(self, s, e):
        self.last_gen_imgs = np.array(self.last_gen_imgs)
        #self.last_gen_imgs = 0.5 * self.last_gen_imgs + 0.5  # .... [-1, 1] to [0, 1]

        fig = plt.figure(figsize=(8, 4))
        for i in range(self.last_gen_imgs.shape[0]):
            plt.subplot(4, 8, i + 1)
            #print("Tensor for image: ", "_indiv" + str(i).zfill(5) + ".png")
            #print("Original tensor:")
            #print(self.last_gen_imgs[i, :, :, 0])
            #print("aftyer transform:")
            #print(self.last_gen_imgs[i, :, :, 0] * 127.5 + 127.5)
            #print("\n\n\n")
            plt.imshow(self.last_gen_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(self.gan_images + 'image_at_epoch{:04d}_step{:04d}.png'.format(e, s))
        plt.close()


    def write_losses_epochs(self, step, epoch):
        fn = self.run_dir + "dcgan_losses.txt"
        with open(fn, mode='a', newline='') as file:
            fwriter = csv.writer(file, delimiter=',')
            if epoch == 0 and step == 0:
                file.write("[d_loss, g_loss]\n")
            fwriter.writerow([self.dloss.numpy(), self.gloss.numpy()])


    def write_digits_classifications(self, step, epoch, digits, classifications = True, path = None):
        if path is None:
            path = self.run_dir
        fn = path + "digit_max.txt"
        header = epoch == 0 and step == 0

        with open(fn, mode='a', newline='') as file:
            fwriter = csv.writer(file, delimiter=',')
            if header:
                file.write("[step, epoch, max]\n")
            fwriter.writerow([step, epoch] + list(np.argmax(classify_digits(digits), axis=1)))

        if classifications:
            fn = path + "digit_classifications.txt"
            with open(fn, mode='a', newline='') as file:
                fwriter = csv.writer(file, delimiter=',')
                if header:
                    file.write("[step, epoch, classifications]\n")
                fwriter.writerow([step, epoch] + [list(p) for p in classify_digits(digits).numpy()])


    def plot_losses(self, show_graphics = False):
        fig, ax = plt.subplots(1, 1)
        ax.plot(range(len(self.loss_hist)), np.asarray(self.loss_hist)[:, 0], linestyle='-', label="D loss")
        pylab.legend(loc='upper left')
        ax.set_xlabel('Training steps')
        ax.set_ylabel('Loss')
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_title('Discriminator loss across training steps')
        fig.set_size_inches(12, 8)
        plt.savefig(fname=self.run_dir + 'Losses.svg', format="svg")
        if show_graphics: plt.show()
        plt.close(fig)

    def write_archive(self, epoch, save_class=True):
        fn = self.archive_dir + delimiter + "epoch_" + str(epoch).zfill(4) + delimiter
        os.makedirs(fn)
        with open(fn + "expressions.txt", mode='a', newline='') as file:
            fwriter = csv.writer(file, delimiter=',')
            file.write("[indiv, fitness, expression]\n")
            c = 0
            for ind in self.archive:
                fwriter.writerow([str(c), ind['fitness'], ind['tree'].get_str()])
                save_image(ind['tensor'], c, fn, self.generator.target_dims, addon='_archive_best')
                c += 1
        with open(fn + "tensors.txt", mode='a', newline='') as file:
            fwriter = csv.writer(file, delimiter=',')
            file.write("[indiv, expression]\n")
            c = 0
            for ind in self.archive:
                fwriter.writerow([str(c), ind['tensor'].numpy()])
                c += 1
        if save_class:
            archive_digits = np.expand_dims([p['tensor'] for p in self.archive], axis=3)
            self.write_digits_classifications(0, 0, archive_digits, classifications = True, path = fn)



def classify_from_name(imname='test_im.png', invert=True):
    x = io.imread(imname)
    # compute a bit-wise inversion so black becomes white and vice versa
    if invert:
        np.invert(x)
    x = rgb2gray(x)
    # make it the right size
    x = resize(x, (28, 28))
    # print(x)
    # convert to a 4D tensor to feed into our model
    x = x.reshape(1, 28, 28, 1)
    x = x.astype('float32')
    classify_digits(x)


def classify_digits(digits):
    return cnn_model(digits, training=False)
    #return


    #print(out.shape)
    #print("Output:", out)
    #print("Argmax: ", np.argmax(out, axis=1))


if __name__ == '__main__':

    gen_pop = 32
    digits = [1]
    #if len(sys.argv) > 1:
    #    print("Going for digit: ", sys.argv[1])
    #    digits = [int(sys.argv[1])]

    gens = [50] # 50 # 1- teste maluco (tem de ser pelo menos 2)
    epochs = 1
    fsets = [std_set]
    runs = 1 # 15
    digits = range(4, 5) # 10

    seeds = [random.randint(0, 0x7fffffff) for i in range(runs)]
    #cnn_model.summary()

    for r in range(runs):
        for d in digits:
            for g in gens:
                for cur_set in fsets:
                    mnist_dcgan = dcgan(batch_size=gen_pop, gens_per_batch=g, fset=std_set, digits_to_train=d,
                                        run_from_last_pop=1,
                                        linear_gens_per_batch=False,
                                        do_archive=True,
                                        starchive=1,
                                        seed = seeds[r],
                                        log_losses = False,
                                        log_digits_class = False)
                    train_time, train_hist = mnist_dcgan.train(epochs = epochs)
                    print("Elapsed training time (s): ", train_time)
                    #mnist_dcgan.print_training_hist()
    print("Number of gen image: ", gen_image_cnt)
    print("Number of fake images: ", fake_image_cnt)

    """
    epochs = 100
    gen_pop = 32
    #run_from_last_pop = True
    #linear_gens_per_batch = True

    gens = 100
    fsets = extended_set

    print("\n\nCurrent number of gens: ", gens)
    print("Current set: ", str(fsets))
    print("CRun from last pop?: ", False)
    print("Linear gens per batch?: ", True)
    mnist_dcgan = dcgan(batch_size=gen_pop, gens_per_batch=100, fset=fsets,
                    run_from_last_pop=False, linear_gens_per_batch=True)
    train_time, train_hist = mnist_dcgan.train(epochs = epochs)
    print("Elapsed training time (s): ", train_time)
    mnist_dcgan.print_training_hist()
    """