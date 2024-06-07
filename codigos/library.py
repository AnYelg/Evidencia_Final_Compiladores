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

def search_numpy(function_name):
    try:
        return getattr(np, function_name)
    except:
        pass
    return None

def gen_vector(*args):
    return np.array(args)

def print_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            print(line.strip())

def apply_watershed(image_path):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert img is not None, "file could not be read, check with os.path.exists()"
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 0] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    show_image(img)