import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import torchvision

def compute_output_shape(current_shape, kernel_size, stride, padding):
    """
    :param tuple current_shape:  The current shape of the data before a convolution is applied.
    :param tuple kernel_size:    The kernel size of the current convolution operation.
    :param tuple stride:         The stride of the current convolution operation.
    :param tuple padding:        The padding of the current convolution operation.

    :return:  The shape after a convolution operation with the above parameters is applied.
    :rtype:   tuple

        The formula used to compute the final shape is

        component[i] = floor((N[i] - K[i] + 2 * P[i]) / S[i]) + 1

        where, N = current shape of the data
               K = kernel size
               P = padding
               S = stride
    """
    # get the dimension of the data compute each component using the above formula
    dimensions = len(current_shape)
    return tuple((current_shape[i] - kernel_size[i] + 2 * padding[i]) // stride[i] + 1
                 for i in range(dimensions))

def compute_transpose_output_shape(current_shape, kernel_size, stride, padding):
    """
    :param tuple current_shape:  The current shape of the data before a transpose convolution is
                                   applied.
    :param tuple kernel_size:    The kernel size of the current transpose convolution operation.
    :param tuple stride:         The stride of the current transpose convolution operation.
    :param tuple padding:        The padding of the current transpose convolution operation.

    :return:  The shape after a transpose convolution operation with the above parameters is
                applied.
    :rtype:   tuple

            The formula used to compute the final shape is

        component[i] = (N[i] - 1) * S[i] - 2 * P[i] + (K[i] - 1) + 1

        where, N = current shape of the data
               K = kernel size
               P = padding
               S = stride
    """
    # get the dimension of the data compute each component using the above formula
    dimensions = len(current_shape)
    return tuple((current_shape[i] - 1) * stride[i] - 2 * padding[i] + (kernel_size[i] - 1) + 1
                 for i in range(dimensions))

def invalid_shape(current_shape):
    for component in current_shape:
        if component <= 0:
            return True
    return False

def compute_output_padding(current_shape, target_shape):
    """
    :param tuple current_shape:  The shape of the data after a transpose convolution operation
                                   takes place.
    :param tuple target_shape:   The target shape that we would like our data to have after the
                                   transpose convolution operation takes place.

    :return:  The output padding needed so that the shape of the image after a transpose
                convolution is applied, is the same as the target shape.
    :rtype:   tuple
    """
    # basically subtract each term to get the difference which will be the output padding
    dimensions = len(current_shape)
    return tuple(target_shape[i] - current_shape[i] for i in range(dimensions))


def plot_multiple(images, n, dim, cmap):

    # unpack the image dimensions
    z_dim, x_dim, y_dim = dim

    # if image is grayscale
    if (z_dim == 1):
        # initialize some limits on x&y
        x_limit = np.linspace(-2, 2, n)
        y_limit = np.linspace(-2, 2, n)

        # initialize the final combined image
        empty = np.empty((x_dim*n, y_dim*n))

        current = 0
        for i, zi in enumerate(x_limit):
            for j, pi in enumerate(y_limit):
                # each image insert it into a subsection of the final image
                empty[(n-i-1)*x_dim:(n-i)*x_dim, j*y_dim:(j+1)*y_dim] = images[current][0]
                current+=1

        plt.figure(figsize=(8, 10))

        x,y = np.meshgrid(x_limit, y_limit)
        plt.imshow(empty, origin="upper", cmap=cmap)
        plt.grid(False)
        plt.show()

    # if the image is rgb
    elif (z_dim == 3):
        # initialize some limits on x&y
        x_limit = np.linspace(-2, 2, n)
        y_limit = np.linspace(-2, 2, n)

        # initialize the final combined image (now with one more dim)
        empty = np.empty((x_dim*n, y_dim*n, 3))

        current = 0
        for i, zi in enumerate(x_limit):
            for j, pi in enumerate(y_limit):
                # flatten the image
                curr_img = images[current].ravel()
                # reshape it into the correct shape for pyplot
                curr_img = np.reshape(curr_img, (x_dim, y_dim, z_dim), order='F')
                # rotate it by 270 degrees
                curr_img = np.rot90(curr_img, 3)

                # insert it into a subsection of the final image
                empty[(n-i-1)*x_dim:(n-i)*x_dim, j*y_dim:(j+1)*y_dim] = curr_img
                current+=1

        plt.figure(figsize=(8, 10))

        x,y = np.meshgrid(x_limit, y_limit)
        plt.imshow(empty, origin="upper", cmap=cmap)
        plt.grid(False)
        plt.show()


def filepath_is_not_valid(filepath):
    """
    :param str filepath:   The path of the given file to check

    :return:  True if the the path passed as an arguments is invalid; Else False.
    :rtype:   bool

    Function used to check whether a filepath containing information is valid.
    """

    # check if the path leads to a file
    if not os.path.isfile(filepath):
        # return false
        return True

    # return false since the path is valid
    return False


def prepare_dataset(configuration):
    """
    :param dict configuration:  The configuration dictionary returned by parse_config_file.

    :return:  A dictionary containing information about the dataset used.
    :rtype:   dict

    Function used to set some values used by the model based on the dataset selected
    """
    dataset_info = {}
    if (configuration["dataset"] == "MNIST"):
        dataset_info["ds_method"] = torchvision.datasets.MNIST
        dataset_info["ds_shape"] = (1, 28, 28)
        dataset_info["ds_path"] = configuration["path"]
    elif (configuration["dataset"] == "CIFAR10"):
        dataset_info["ds_method"] = torchvision.datasets.CIFAR10
        dataset_info["ds_shape"] = (3, 32, 32)
        dataset_info["ds_path"] = configuration["path"]
    elif (configuration["dataset"] == "FashionMNIST"):
        dataset_info["ds_method"] = torchvision.datasets.FashionMNIST
        dataset_info["ds_shape"] = (1, 28, 28)
        dataset_info["ds_path"] = configuration["path"]
    else:
        print("Currently only MNIST & CIFAR10 datasets are supported")
        return None

    return dataset_info


def str_to_int_list(string):
    """
    :param str string:  A string read by the config file.

    :return:  A list of integers.
    :rtype:   list

    Utility function used to convert a string to a list of integers
    """
    list = []
    parts = string.split(',')

    for part in parts:
        part = part.replace('[', '')
        part = part.replace(']', '')
        part = part.strip()

        number = int(part)
        list.append(number)

    return list


def str_to_tuple_list(string):
    """
    :param str string:  A string read by the config file.

    :return:  A list of tuples of integers.
    :rtype:   list

    Utility function used to convert a string to a list of tuples of integers.
    """
    list = []
    parts = string.split(')')

    for part in parts:
        part = part.replace('[', '')
        part = part.replace(']', '')
        part = part.replace('(', '')
        part = part.strip()

        inner_parts = part.split(',')

        inner_list = []
        for inner_part in inner_parts:
            if (inner_part == ''):
                continue
            inner_part = inner_part.strip()
            number = int(inner_part)
            inner_list.append(number)

        inner_tuple = tuple(inner_list)
        if (len(inner_tuple) == 2):
            list.append(inner_tuple)

    return list

