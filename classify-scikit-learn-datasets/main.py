# Sardesai, Chaitanya
# 2018-11-26
import data_generation
import sys
import os
import random
import math
import copy
import tensorflow as tf

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.backends.tkagg as tkagg
import matplotlib.colors as colr

class MainWindow(tk.Tk):
    """
    This class creates and controls the main window frames and widgets
    """

    def __init__(self, debug_print_flag=False):
        tk.Tk.__init__(self)
        self.debug_print_flag = debug_print_flag
        self.master_frame = tk.Frame(self)
        self.master_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.rowconfigure(0, weight=1, minsize=500)
        self.columnconfigure(0, weight=1, minsize=500)
        # set the properties of the row and columns in the master frame
        self.master_frame.rowconfigure(0, weight=1, minsize=10, uniform='xx')
        self.master_frame.columnconfigure(0, weight=1, minsize=200, uniform='xx')

        self.left_frame = tk.Frame(self.master_frame)
        self.left_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        # Create an object for plotting graphs in the left frame
        self.display_activation_functions = MainFrame(self, self.left_frame, debug_print_flag=self.debug_print_flag)

class MainFrame:
    """
        This class creates and controls the widgets and figures in the Main frame which
        are used to display the activation functions.
        """

    def __init__(self, root, master, debug_print_flag=False):
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.xmin = 0
        self.xmax = 100
        self.ymin = 0
        self.ymax = 2

		# Epoc variables
        self.no_of_epocs = 10
        self.max_no_epocs = 100
        self.total_epocs = 0
		
		# Create default values for hyper-parameters 
        self.no_of_classes = 4
        self.no_of_samples = 200
        self.hidden_layer_nodes = 100
        self.output_layer_nodes = self.no_of_classes
        self.regularization_lambda = 0.01
        self.learning_rate = 0.1
        self.activation_function = 'Sigmoid'

		# Create empty objects for weights, biases and input data, targets of the input data
        self.input = np.array([[]])
        self.target = np.array([[]])
        self.weight_matrix = np.array([[]])
        self.weight_matrix2 = np.array([[]])
        self.bias = np.array([])
        self.bias2 = np.array([])
        self.dataset_type = 's_curve'

		# Set range of values which will be assigned to weight matrices as initialization process 
        self.weight_lower_limit_val = -0.001
        self.weight_higher_limit_val = 0.001
        self.no_weights_for_a_neuron = 2

		# Create variables required in network (outputs, errors)
        self.lr = 0
        self.p = None
        self.W = None
        self.b = None
        self.net_value = 0
        self.op_hidden_layer = 0
        self.cross_entropy_error = 0
        self.dW = 0
        self.db = 0
        self.update_W = None
        self.update_b = None

        self.t = None
        self.p2 = None
        self.W2 = None
        self.b2 = 0
        self.op_layer_op = None
        self.class_op = 0
        self.dW2 = 0
        self.db2 = 0
        self.update_W2 = None
        self.update_b2 = None
        self.sample_no = 0
        self.net_value_2 = 0
        self.prob_class = 0

        #########################################################################
        #  Set up the plotting frame and controls frame
        #########################################################################
        master.rowconfigure(0, weight=10, minsize=200)
        master.columnconfigure(0, weight=1)
        self.plot_frame = tk.Frame(self.master, borderwidth=10, relief=tk.SUNKEN)
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.figure = plt.figure("")
        #self.axes = self.figure.add_axes([0.15, 0.15, 0.6, 0.8])
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Input')
        self.axes.set_ylabel('Output')
        self.axes.set_title("")
        #plt.xlim(self.xmin, self.xmax)
        #plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.pack(expand=True, fill=tk.BOTH)
        # Create a frame to contain all the controls such as sliders, buttons, ...
        self.controls_frame = tk.Frame(self.master)
        self.controls_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the control widgets such as sliders and selection boxes
        #########################################################################
        self.learning_rate_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=0.001, to_=1, resolution=0.001, bg="#DDDDDD",
                                             activebackground="#FF0000", highlightcolor="#00FFFF",
                                             label="Learning Rate",
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # lambda slider
        self.lambda_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_=0, to_=1, resolution=0.01, bg="#DDDDDD",
                                     activebackground="#FF0000", highlightcolor="#00FFFF",
                                     label="Weight Regularization", length = 130,
                                     command=lambda event: self.lambda_slider_callback())
        self.lambda_slider.set(self.regularization_lambda)
        self.lambda_slider.bind("<ButtonRelease-1>", lambda event: self.lambda_slider_callback())
        self.lambda_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        # number of samples slider
        self.training_size_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=4, to_=1000, resolution=1, bg="#DDDDDD",
                                             activebackground="#FF0000", highlightcolor="#00FFFF",
                                             label="Number of Samples", length = 120,
                                             command=lambda event: self.training_size_slider_callback())
        self.training_size_slider.set(self.no_of_samples)
        self.training_size_slider.bind("<ButtonRelease-1>", lambda event: self.training_size_slider_callback())
        self.training_size_slider.grid(row=0, column=2, sticky=tk.N + tk.E + tk.S + tk.W)

        # no of classes slider
        self.classes_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                      from_=2, to_=10, resolution=1, bg="#DDDDDD",
                                      activebackground="#FF0000", highlightcolor="#00FFFF",
                                      label="Number of Classes",length = 120,
                                      command=lambda event: self.classes_slider_callback())
        self.classes_slider.set(self.no_of_classes)
        self.classes_slider.bind("<ButtonRelease-1>", lambda event: self.classes_slider_callback())
        self.classes_slider.grid(row=0, column=3, sticky=tk.N + tk.E + tk.S + tk.W)

        # number of nodes in hidden layer slider
        self.hidden_layer_nodes_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_=1, to_=500, resolution=1, bg="#DDDDDD",
                                     activebackground="#FF0000", highlightcolor="#00FFFF",
                                     label="Hidden Layer Nodes", length = 120,
                                     command=lambda event: self.hidden_layer_nodes_callback())
        self.hidden_layer_nodes_slider.set(self.hidden_layer_nodes)
        self.hidden_layer_nodes_slider.bind("<ButtonRelease-1>", lambda event: self.hidden_layer_nodes_callback())
        self.hidden_layer_nodes_slider.grid(row=0, column=4, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for buttons
        #########################################################################
        self.adjust_weights_train = tk.Button(self.controls_frame, text='Adjust Weights (Train)',
                                           command=self.train)
        self.adjust_weights_train.grid(row=0, column=5, sticky=tk.N + tk.E + tk.S + tk.W)

        self.randomize_weights_button = tk.Button(self.controls_frame, text='Randomize Weights',
                                             command=self.gen_random_weight_matrix)
        self.randomize_weights_button.grid(row=0, column=6, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for drop down selection
        #########################################################################
        self.label_for_activation_function = tk.Label(self.controls_frame, text="Hidden Layer Transfer Function",
                                                      justify="center")
        self.label_for_activation_function.grid(row=0, column=7, sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.controls_frame, self.activation_function_variable,
                                                          "Relu", "Sigmoid",
                                                          command=lambda
                                                              event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set(self.activation_function)
        self.activation_function_dropdown.grid(row=0, column=8, sticky=tk.N + tk.E + tk.S + tk.W)

        self.label_for_dataset_type = tk.Label(self.controls_frame, text="Type of generated data",
                                                      justify="center")
        self.label_for_dataset_type.grid(row=0, column=9, sticky=tk.N + tk.E + tk.S + tk.W)
        self.dataset_type_variable = tk.StringVar()
        self.dataset_type_dropdown = tk.OptionMenu(self.controls_frame, self.dataset_type_variable,
                                                          "s_curve", "blobs", "swiss_roll", "moons",
                                                          command=lambda event: self.dataset_type_dropdown_callback())
        self.dataset_type_variable.set(self.dataset_type)
        self.dataset_type_dropdown.grid(row=0, column=10, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Call required functions
        #########################################################################
        self.generate_data()
        #self.gen_random_weight_matrix()
        #self.train()

    def learning_rate_slider_callback(self):
        self.learning_rate = np.float(self.learning_rate_slider.get())
        self.gen_random_weight_matrix()

    def lambda_slider_callback(self):
        self.regularization_lambda = np.float(self.lambda_slider.get())
        self.gen_random_weight_matrix()

    def training_size_slider_callback(self):
        self.reset_all()
        self.no_of_samples = self.training_size_slider.get()
        self.gen_data_and_plot_samples_points()

    def classes_slider_callback(self):
        self.reset_all()
        self.no_of_classes = np.int(self.classes_slider.get())
        self.gen_data_and_plot_samples_points()

    def hidden_layer_nodes_callback(self):
        self.hidden_layer_nodes = self.hidden_layer_nodes_slider.get()
        self.gen_random_weight_matrix()

    def activation_function_dropdown_callback(self):
        self.activation_function = self.activation_function_variable.get()
        self.gen_random_weight_matrix()

    def dataset_type_dropdown_callback(self):
        self.reset_all()
        self.dataset_type = self.dataset_type_variable.get()
        self.gen_data_and_plot_samples_points()

    def generate_data(self):
        self.input, self.target = data_generation.generate_data(self.dataset_type, self.no_of_samples, self.no_of_classes)
        #print(self.input, self.target)

    def gen_random_weight_matrix(self):
        self.delete_weight_matrix_and_bias()
        self.set_weight_matrix_and_bias()
        self.weight_matrix = np.random.uniform(low=self.weight_lower_limit_val, high=self.weight_higher_limit_val,
                                       size=(self.hidden_layer_nodes, self.no_weights_for_a_neuron))
        self.weight_matrix2 = np.random.uniform(low=self.weight_lower_limit_val, high=self.weight_higher_limit_val,
                                               size=(self.no_of_classes, self.hidden_layer_nodes))
        self.bias = np.random.uniform(low=self.weight_lower_limit_val, high=self.weight_higher_limit_val,
                                           size=(self.hidden_layer_nodes, 1))
        self.bias2 = np.random.uniform(low=self.weight_lower_limit_val, high=self.weight_higher_limit_val,
                                      size=(self.no_of_classes, 1))
        self.axes.cla()
        data_generation.plot_sample_points(self.dataset_type, self.input, self.target, self.canvas, self.axes,
                                          self.no_of_classes,
                                          self.no_of_samples)
        self.canvas.draw()

    def define_neurons(self):
        #hidden layer
        self.lr = tf.placeholder(dtype=tf.float32, name='learning_rate')
        self.p = tf.placeholder(dtype=tf.float32, name='input')
        self.W = tf.Variable(self.weight_matrix, dtype=tf.float32, name='weight_matrix')
        self.b = tf.Variable(self.bias, dtype=tf.float32, name='bias')

        #hidden layer operations
        self.net_value = tf.matmul(self.W, self.p) + self.b

        if self.activation_function == 'Sigmoid':
            #print('here sigmoid')
            self.op_hidden_layer = tf.nn.sigmoid(self.net_value)
        else:
            #print('here relu')
            self.op_hidden_layer = tf.nn.relu(self.net_value)

        #output layer
        #self.p2 = self.op_hidden_layer
        self.W2 = tf.Variable(self.weight_matrix2, dtype=tf.float32, name='op_layer_weight_matrix')
        self.b2 = tf.Variable(self.bias2, dtype=tf.float32, name='op_layer_bias')
        self.labels = tf.placeholder(tf.int32)
        self.sample_no = tf.placeholder(dtype=tf.int32, name='sample_number')
        self.t = tf.placeholder(dtype=tf.float32, name='target')

        #output layer operations
        self.net_value_2 = tf.matmul(self.W2, self.op_hidden_layer) + self.b2
        self.op_layer_op = tf.nn.softmax(self.net_value_2, axis=0)
        self.class_op = tf.cast(self.t[self.sample_no], dtype=tf.int32)
        self.prob_class = self.op_layer_op[self.class_op]
        self.cross_entropy_error = tf.add((tf.log(self.prob_class) * (-1)), tf.multiply(self.regularization_lambda, tf.nn.l2_loss(self.W2)))

        #calculate gradients
        self.dW, self.db = tf.gradients(self.cross_entropy_error, [self.W, self.b])
        #print(self.dW)
        self.dW2, self.db2 = tf.gradients(self.cross_entropy_error, [self.W2, self.b2])

        #update weights and biases of both the layers
        self.update_W = tf.assign_sub(self.W, tf.multiply(self.lr, self.dW))
        self.update_b = tf.assign_sub(self.b, self.lr*self.db)
        self.update_W2 = tf.assign_sub(self.W2, self.lr * self.dW2)
        self.update_b2 = tf.assign_sub(self.b2, self.lr * self.db2)

        #self.train_obj = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cross_entropy_error)

    def plot_samples_points_and_boundries(self, xx, yy, zz):
        self.axes.cla()
        self.axes.pcolormesh(xx, yy, zz, cmap='nipy_spectral')
        #self.axes.pcolormesh(xx, yy, zz, cmap='terrain')
        Sardesai_05_02.plot_sample_points(self.dataset_type, self.input, self.target, self.canvas, self.axes, self.no_of_classes,
                                          self.no_of_samples)

        self.canvas.draw()

    def gen_data_and_plot_samples_points(self):
        self.generate_data()
        self.axes.cla()
        Sardesai_05_02.plot_sample_points(self.dataset_type, self.input, self.target, self.canvas, self.axes, self.no_of_classes,
                                          self.no_of_samples)
        self.canvas.draw()

    def train(self):
        if self.weight_matrix.size == 0:
            self.gen_random_weight_matrix()
		# Create tensors and graph.
        self.define_neurons()
        with tf.Session() as s:
            s.run(tf.global_variables_initializer())
            errors = []
            current_error = 0
            # update_W = 0
            # update_b = 0
            # update_W2 = 0
            # update_b2 = 0
			# Train the network
            for epoc in range(0, self.no_of_epocs, 1):
                for sample_no, sample in enumerate(self.input):
                    sample = np.reshape(sample, (-1, 1))
                    current_error, update_W, update_b, update_W2, update_b2 = s.run([self.cross_entropy_error,
                                                                                     self.update_W, self.update_b,
                                                                                    self.update_W2, self.update_b2],
                                                                    feed_dict={self.p: sample, self.t: self.target, self.lr: self.learning_rate,
                                                                                self.sample_no: sample_no})
                errors.append(current_error)

            # self.final_weight_matrix = copy.deepcopy(update_W)
            # self.final_weight_matrix2 = copy.deepcopy(update_W2)
            # self.final_b = copy.deepcopy(update_b)
            # self.final_b2 = copy.deepcopy(update_b2)

			# Plot the class regions using trained weights of hidden and output layers
            resolution = 100
            self.xmin = np.amin(self.input[:,0])
            self.xmax = np.amax(self.input[:,0])
            self.ymax = np.amax(self.input[:,1])
            self.ymin = np.amin(self.input[:,1])
            xs = np.linspace(self.xmin, self.xmax, resolution)
            ys = np.linspace(self.ymin, self.ymax, resolution)
            xx, yy = np.meshgrid(xs, ys)
            zz = []
            w = np.array([[]])
            w2 = np.array([[]])
            sample_no = 0
            for outer_index, x_list in enumerate(xx):
                for inner_index, x_point in enumerate(x_list):
                    sample = np.array([x_point, yy[outer_index][inner_index]]).reshape(-1,1)
                    output_class, w, w2 = s.run([self.net_value_2, self.W, self.W2], feed_dict={self.p: sample,
                                                                                                self.lr: self.learning_rate})
                    zz.append(np.argmax(output_class))
            np.set_printoptions(threshold=np.inf)
            zz = np.array(zz).reshape(xx.shape)
            self.plot_samples_points_and_boundries(xx, yy, zz)
        return

	# Delete weight matrices, bias 
    def delete_weight_matrix_and_bias(self):
        del self.weight_matrix
        del self.weight_matrix2
        del self.bias
        del self.bias2

	# Initialize weight matrix, bias: empty objects created
    def set_weight_matrix_and_bias(self):
        self.weight_matrix = np.array([[]])
        self.weight_matrix2 = np.array([[]])
        self.bias = np.array([])
        self.bias2 = np.array([])

	# Delete all weight matrices, bias, input data, target 
    def delete_all(self):
        self.delete_weight_matrix_and_bias()
        del self.input
        del self.target

	# reset all variables weight matrix, data input, targets: empty objects created 
    def reset_all(self):
        #delete weight matrix
        self.delete_all()
        self.set_weight_matrix_and_bias()
        self.input = np.array([[]])
        self.target = np.array([])
        #Sardesai_05_02.plot_sample_points(self.dataset_type, self.canvas, self.axes, self.no_of_classes, self.no_of_samples)
        return


def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

main_window = MainWindow(debug_print_flag=False)
# main_window.geometry("500x500")
main_window.wm_state('zoomed')
main_window.title('Training window')
main_window.minsize(800, 600)
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
main_window.mainloop()