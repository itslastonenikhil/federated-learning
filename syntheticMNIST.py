#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This code to create a custom MNIST dataset was made possible thanks to
 https://github.com/LaRiffle/collateral-learning . 
 
Important to know that aside the tampering I did on the build_dataset function
for my own application, I also had to change rgba_to_rgb. Indeed, the function
was working as desired on Jupyter but not on Spyder. Do not ask me why !
"""



import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import pickle
import torch
import math
import os
from IPython.display import clear_output
import random


import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader


"""PLOT FUNCTIONS TO VISUALIZE THE FONTS AND DATASETS"""
def show_original_font(family:str):
    """Plot the original numbers used to create the dataset"""
    
    plt.figure()
    plt.title(family)
    plt.text(0, 0.4, '1234567890', size=50, family=family)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"plots/{family}_original.png") 
    
    
def convert_to_rgb(data):
    
    def rgba_to_rgb(rgba):
        return rgba[1:]

    return np.apply_along_axis(rgba_to_rgb, 2, data) 



def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = np.array([28, 28, 3], dtype =int)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    #print(x.shape, y.shape, z.shape)
    #print(dx.shape, dy.shape)
    #x, y, z = x[:28, :28, :3], y[:28, :28, :3], z[:28, :28, :3]
    #dx, dy = dx[:28, :28, :3], dy[:28, :28, :3]
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(shape)



def center(data):
    # Inverse black and white
    wb_data = np.ones(data.shape) * 255 - data
    
    # normalize
    prob_data = wb_data / np.sum(wb_data)
    
    # marginal distributions
    dx = np.sum(prob_data, (1, 2))
    dy = np.sum(prob_data, (0, 2))

    # expected values
    (X, Y, Z) = prob_data.shape
    cx = np.sum(dx * np.arange(X))
    cy = np.sum(dy * np.arange(Y))
    
    # Check bounds
    assert cx > X/4 and cx < 3 * X/4, f"ERROR: {cx} > {X/4} and {cx} < {3 * X/4}"
    assert cy > Y/4 and cy < 3 * Y/4, f"ERROR: {cy} > {Y/4} and {cy} < {3 * Y/4}"
    
    # print('Center', cx, cy)
    
    x_min = int(round(cx - X/4))
    x_max = int(round(cx + X/4))
    y_min = int(round(cy - Y/4))
    y_max = int(round(cy + Y/4))
    
    return data[x_min:x_max, y_min:y_max, :]
   


def create_transformed_digit(digit:int, size:float, rotation:float, family:str):
    
    fig = plt.figure(figsize=(2,2), dpi=28)
    fig.text(0.4, 0.4, str(digit), size=size, rotation=rotation, family=family)

    # Rm axes, draw and get the rgba shape of the digit
    plt.axis('off')
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    # Convert to rgb
    data = convert_to_rgb(data)

    # Center the data
    data = center(data)

    # Apply an elastic deformation
    data = elastic_transform(data, alpha=991, sigma=9)

    # Free memory space
    plt.close(fig)
    
    return data

    

def save_dataset(dataset_name:str, array_X:np.array, array_y:np.array):
    
    with open(f'{dataset_name}.pkl', 'wb') as output:
        dataset = array_X, array_y
        pickle.dump(dataset, output)
        
        
        
def build_dataset(C:dict, std_size=2.5):
    """build a dataset with `dataset_size` according to the chosen font
    and deformation. Only digits in `datasets_digits` are in the created 
    dataset."""
    
    numbers_str="".join([str(n) for n in C['numbers']])
    file_name=f"{C['font']}_{numbers_str}_{C['n_samples']}_{C['tilt']}_{C['seed']}"    
    
    if os.path.isfile(f"{file_name}.pkl"):
        return pickle.load(open(f"{file_name}.pkl", "rb"))
    
    
    if C['seed']: np.random.seed(C['seed'])
    
    #Make a plot of each original digit to know what they look like
#    show_original_font(C['font'])
    
    list_X = []
    list_y= []
    print("")
    for i in range(C['n_samples']):

        if i%(C['n_samples']/10) == 0: 
            print("\b|", round(i / C['n_samples'] * 100), '%')
    
        
        X = np.zeros((1, 28, 28 ))
        #Choosing a number at this step and its transformation characteristics
        digit = C["numbers"][np.random.randint(len(C["numbers"]))]

        # for j, tilt in enumerate(C['tilt']):
        # 	rotation = tilt + np.random.normal(0, C['std_tilt'])
        # 	size = 60 + np.random.normal(0, std_size)         	

        # 	X_tilt=create_transformed_digit(digit, size, rotation, C['font'])

        # 	X[j] = X_tilt[:, :, j]
        
        tilt = random.choice(C['tilt'])
        rotation = tilt + np.random.normal(0, C['std_tilt'])
        size = 60 + np.random.normal(0, std_size)         	

        X_tilt=create_transformed_digit(digit, size, rotation, C['font'])

        X[0] = X_tilt[:, :, 0]

        # Append data to the datasets
        #list_X.append(X[:,:,0])
        list_X.append(X)
        list_y.append(digit)
    
    #save the dataset
    dataset = (np.array(list_X), np.array(list_y))
    pickle.dump(dataset, open(f'{file_name}.pkl', 'wb'))
    
    return np.array(list_X), np.array(list_y)

 
class Ds_MNIST_modified(Dataset):
    """Creation of the dataset used to create the clients' dataloader"""
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self): return len(self.features)

    def __getitem__(self,idx):
        
        #3D input 1x28x28
        sample_x = torch.Tensor(self.features[idx])
        sample_y = self.labels[idx]
        
        return sample_x, sample_y


    def plot_samples(self, channel:int, title=None, plot_name="", 
        n_examples =20):
    
        n_rows = int(n_examples / 5)
        plt.figure(figsize=(1* n_rows, 1*n_rows))
        if title: plt.suptitle(title)
            
        for idx in range(n_examples):
            
            X, y = self[idx]

            ax = plt.subplot(n_rows, 5, idx + 1)

            image = 255 - X.view((-1, 28, 28))[channel]
            ax.imshow(image, cmap='gist_gray')
            ax.axis("off")

        if plot_name!="":plt.savefig(f"plots/"+plot_name+".png")

        plt.tight_layout()

    
    

def get_synth_MNIST(clients, batch_size:int, shuffle=True):
    """function returning a list of training and testing dls."""
    
    list_train, list_test = [], []
    
    for C in clients:
        X, y = build_dataset(C)
        X = (255 - X) /255

        X_train, y_train = X[:C['n_samples_train']], y[:C['n_samples_train']]
        X_test, y_test = X[C['n_samples_train']:], y[C['n_samples_train']:]
            
        train_ds = Ds_MNIST_modified(X_train, y_train)         
        train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = shuffle)
        list_train.append(train_dl)
         
        test_ds = Ds_MNIST_modified(X_test, y_test)         
        test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle = shuffle)  
        list_test.append(test_dl)
        
    return list_train, list_test
    
