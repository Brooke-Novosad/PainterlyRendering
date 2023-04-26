import cairo
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import shutil
from scipy import signal
import os

strokes = {
    'minimalist': {
                'threshold': 150,
                'brush_radii': [16, 8, 4],
                'curvature_filter': 1.,
                'blur_factor': 0.1,
                'opacity': 1,
                'gridsize': 1.,
                'smalleststrokelength': 2,
                'largeststrokelength': 8
                },
    'cubism': {
                'threshold': 0,
                'brush_radii': [0.1, 0.2, 0.3],
                'curvature_filter': 0,
                'blur_factor': 0,
                'opacity': 0,
                'gridsize': 0,
                'smalleststrokelength': 0,
                'largeststrokelength': 0
            }
    }

class PaintRender:

    def __init__(self, filename):
        self.image = self.read_image(filename) 
        self.initialize_canvas(self.image.shape[0], self.image.shape[1])  

    def read_image(self, filename):
        image = cv2.imread(filename)
        return image

    def initialize_canvas(self, width, height):
        self.canvas = cairo.ImageSurface(cairo.FORMAT_RGB24, width, height)
        self.context = cairo.Context(self.canvas)
        self.context.scale(width, height)
        self.context.set_line_cap(cairo.LINE_CAP_ROUND)
        
    def write_image(self, output_dir, filename, image):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        os.chmod(output_dir, 0o777)
        fullpath = os.path.join(output_dir, filename)
        cv2.imwrite(fullpath, image)

    def paint(self):
        stroke = strokes['minimalist'] # TODO: Get from cmd line args
        brush_radii = stroke['brush_radii']
        for radius in brush_radii:
            # colored_im = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            sigma_low = stroke['blur_factor']*radius
            blur = self.gaussian_kernel(sigma_low, int(3*sigma_low))
            blurred_im = cv2.filter2D(self.image, -1, blur)
            paint_render.write_image(output_dir, "dog_output" + str(radius) + ".jpg", blurred_im) 

    def gaussian_kernel(self, sigma, kernel_half_size):
        # gotten from project 1 utils file
        '''
        Inputs:
            sigma = standard deviation for the gaussian kernel
            kernel_half_size = recommended to be at least 3*sigma
        
        Output:
            Returns a 2D Gaussian kernel matrix
        '''
        window_size = kernel_half_size*2+1
        gaussian_kernel_1d = signal.gaussian(window_size, std=sigma).reshape(window_size, 1)
        gaussian_kernel_2d = np.outer(gaussian_kernel_1d, gaussian_kernel_1d)
        gaussian_kernel_2d /= np.sum(gaussian_kernel_2d) # make sure it sums to one

        return gaussian_kernel_2d

if __name__ == '__main__':
    output_dir = "./output_images"
    filename = "src_images/dog.jpg"
    paint_render = PaintRender(filename) 
    paint_render.paint()

