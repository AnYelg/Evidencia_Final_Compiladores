import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(image_path):
    path = image_path.strip()
    return cv2.imread(path)

def load_image_gray(image_path):
    path = image_path.strip()
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def show_histogram(image):
    cv2.imshow('window',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.hist(image.ravel(), 256, [0, 256])
    plt.title("Histogram for gray scale picture")
    plt.xlabel("Pixel Value")
    plt.ylabel("Number of Pixels")
    plt.show()

def show_image(image):
    cv2.imshow('window',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

def search_cv2(function_name):
    try:
        return getattr(cv2, function_name)
    except:
        pass
    return None

def gen_vector(*args):
    return np.array(args)

def print_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            print(line.strip())