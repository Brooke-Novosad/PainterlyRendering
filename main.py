import cairo
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import shutil
from scipy import signal
import math
import os
from scipy import ndimage
import random
import sys

strokes = {
    'minimalist': {
                'threshold': 150,
                'brush_radii': 8,
                'curvature_filter': 1.,
                'blur_factor': 0.1,
                'opacity': 1,
                'gridsize': 1.,
                'smalleststrokelength': 2,
                'largeststrokelength': 8
                },
    'cubism': {
                'threshold': 28,
                'brush_radii': 8,
                'curvature_filter': 1.,
                'blur_factor': 0.1,
                'opacity': 1,
                'gridsize': 1.,
                'smalleststrokelength': 2,
                'largeststrokelength': 8
            },
    'popart': {
                'threshold': 70,
                'brush_radii': 4,
                'curvature_filter': 1.,
                'blur_factor': 0.1,
                'opacity': 1,
                'gridsize': 1.,
                'smalleststrokelength': 6,
                'largeststrokelength': 12
            },
    'photorealistic': {
                'threshold': 28,
                'brush_radii': 4,
                'curvature_filter': 1.,
                'blur_factor': 0.1,
                'opacity': 1,
                'gridsize': 1.,
                'smalleststrokelength': 2,
                'largeststrokelength': 8
            }
    }

class PaintRender:

    # our errors are around 28-30 
    # could be a problem with the diff (getting the error)
    # out of bounds errors, changes when we change threshold and blur factor

    def __init__(self, filename, style):
        self.image = self.read_image(filename) 
        print("style = " + str(style))
        if style not in strokes:
            print("here")
            style = 'minimalist'
        self.style = strokes[style]
        print("self.style = " + str(self.style))
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
        stroke = self.style
        brush_radii = stroke['brush_radii']
        # colored_im = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        sigma_low = stroke['blur_factor']*brush_radii
        blur = self.gaussian_kernel(sigma_low, int(3*sigma_low))
        blurred_im = cv2.filter2D(self.image, -1, blur)
        self.apply_brush_strokes(blurred_im)
        # self.apply_brush_strokes(blurred_im, brush_radii, stroke)
        # paint_render.write_image(output_dir, "dog_output" + str(brush_radii) + ".jpg", self.canvas) 
        self.canvas.write_to_png("dog_output" + str(brush_radii) + ".png")


    def img_diff(self, img1, img2):
        #might be this to change
        #should be sum
        return np.sqrt((img1 - img2)**2).astype(dtype=np.int32)

    
    # From project 1 utils.py
    def gaussian_kernel(self, sigma, kernel_half_size):
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

    def apply_brush_strokes(self, image):
        radius = self.style['brush_radii']
        lines = []
        width, height = image.shape[0], image.shape[1]
        buf = self.canvas.get_data()
        array = np.ndarray (shape=(width, height, 3), dtype=np.uint8, buffer=buf)
        diff = self.img_diff(array, image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)
        grid_length = int(self.style['gridsize']*radius)
        
        threshold = self.style['threshold']
        rounded_width = (width // grid_length) * grid_length
        rounded_height = (height // grid_length) * grid_length
        for x in range(0, rounded_width, grid_length):
            for y in range(0, rounded_height, grid_length):

                curr_diff = diff[x:x+grid_length, y:y+grid_length]
                err = curr_diff.sum()/math.pow(grid_length, 2)
                # print("err = " + str(err))
                if err > threshold:
                    max_idx = np.argmax(curr_diff)
                    bad_pixel_x = max_idx // curr_diff.shape[1]
                    bad_pixel_y = max_idx % curr_diff.shape[1]
                    #make strokes
                    # lines.append(stroke)
        
        # randomize if it's bad

        random.shuffle(lines)
            

if __name__ == '__main__':
    output_dir = "./output_images"
    filename = "src_images/dog.jpg"
    style = str(sys.argv[1])
    paint_render = PaintRender(filename=filename, style=style) 
    paint_render.paint()

