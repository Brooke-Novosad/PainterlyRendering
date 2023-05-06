import numpy as np
import cv2
import random
import sys
import os
from scipy import signal
import math

# Styles from Painterly Rendering with Curved Brush Strokes of Multiple Sizes; Aaron Hertzmann
styles = {
    'futurist': {
                'threshold': 60,
                'brush_radii': [64, 32, 16],
                'curvature_filter': 0.8,
                'blur_factor': 0.4,
                'gridsize': 1.,
                'smalleststrokelength': 6,
                'largeststrokelength': 12
            },
    'colorist_wash': {
                'threshold': 200,
                'brush_radii': [8, 4, 2],
                'curvature_filter': 1.,
                'blur_factor': 0.5,
                'gridsize': 1.,
                'smalleststrokelength': 4,
                'largeststrokelength': 16
            },
    'impressionist': {
                'threshold': 100,
                'brush_radii': [8, 4, 2],
                'curvature_filter': 1.,
                'blur_factor': 0.8,
                'gridsize': 1.,
                'smalleststrokelength': 4,
                'largeststrokelength': 16
            },
    'pop_art': {
                'threshold': 70,
                'brush_radii': [32, 16, 8],
                'curvature_filter': 1.,
                'blur_factor': 0.5,
                'gridsize': 1.,
                'smalleststrokelength': 6,
                'largeststrokelength': 12
            },
  }

class PaintRender:
    def __init__(self, filename, style):
        self.results = {}
        if style not in styles:
            style = 'minimalist'
        self.style = styles[style]
        self.image = self.read_image(filename)
    
    def read_image(self, filename):
        image = cv2.imread(filename)
        return image
    
    # canvas = a new constant color image
    def initialize_canvas(self, image):
        self.canvas = np.zeros(image.shape, dtype=int)
        self.canvas.fill(4531783356.)   # using this value as a placeholder to compare with our threshold
        
    # function from pseudo code from Painterly Rendering with Curved Brush Strokes of Multiple Sizes; Aaron Hertzmann pg 2
    def paint(self, filename):
        brush_radii = self.style['brush_radii']
        self.initialize_canvas(self.image)
        # for each brush size
        for r in brush_radii:
            # compute a blurred reference image
            sigma_low = self.style['blur_factor']*r
            blur = self.gaussian_kernel(sigma_low, int(3*sigma_low))
            blurred_image = cv2.filter2D(self.image, -1, blur)
            self.apply_brush_strokes(blurred_image, r)
        output_dir = "./output_images/"
        self.write_image(output_dir, filename, self.canvas)

    # From Project 1 utils.py
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
    
    def write_image(self, output_dir, filename, image):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        os.chmod(output_dir, 0o777)
        fullpath = output_dir + filename
        cv2.imwrite(fullpath, image)

    def img_diff(self, blurred_image):
        return np.sqrt(np.sum((self.canvas-blurred_image)**2, axis=2))
    
    # function from pseudo code from Stroke Based Painterly Rendering; David Vanderhaeghe, John Collomosse pg 10
    def apply_brush_strokes(self, blurred_image, brush_radii):
        lines = []
        
        width, height = blurred_image.shape[0], blurred_image.shape[1]
        diff = self.img_diff(blurred_image)
        gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
        
        # gradient direction calculation from papers
        self.sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
        self.sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
        mag = np.abs(np.sqrt(self.sobelx**2 + self.sobely**2))
        grid_length = int(self.style['gridsize'] * brush_radii)
        
        rounded_width = (width // grid_length) * grid_length
        rounded_height = (height // grid_length) * grid_length
        # foreach position (x; y) on a grid with spacing grid_length
        for x in range(0, rounded_width, grid_length):
            for y in range(0, rounded_height, grid_length):     
                # curr_diff is the region [x : grid/2...x + grid_length; y : grid/2...y+grid_length]
                curr_diff = diff[x:x+grid_length, y:y+grid_length]
                # err is Sigma curr_diff / grid_length squared
                err = curr_diff.sum() / math.pow(grid_length, 2)


                # if refresh or err > threshold then 
                # (x1; y1) = arg max(i; j)2* curr_diff
                # paintStroke(x1; y1; Ip; Ri)
                if err > self.style['threshold']:
                    # argmax difference from Painterly Rendering for Video and Interaction; Aaron Hertzmann, Ken Perlin; pg 8
                    # find the largest error point
                    bad_pixel_x, bad_pixel_y= np.unravel_index(np.argmax(curr_diff, axis=None), curr_diff.shape)
                    bad_pixel_x += x
                    bad_pixel_y += y
                    stroke = self.make_stroke(bad_pixel_x, bad_pixel_y, brush_radii, blurred_image, mag)
                    lines.append(stroke)

        # paint all strokes in lines on the canvas, in random order
        randlines = random.sample(list(range(len(lines))), len(lines))
        self.canvas = (self.canvas).astype(np.uint8)
        for i in randlines:
            line = lines[i]
            stroke_color = (blurred_image[line[0]]/255)*255
            self.draw_line(line, brush_radii, stroke_color)
        
        sigma = brush_radii 
        blur = self.gaussian_kernel(sigma, 1)
        self.canvas = cv2.filter2D(self.canvas, -1, blur)
        self.canvas = (self.canvas).astype(int)

    def draw_line(self, line, brush_radii, stroke_color):
        for i in range(1, len(line)):
            coords = line[i]
            cv2.circle(self.canvas,(coords[1], coords[0]), brush_radii, stroke_color, -1)
            
    # function from pseudo code from Painterly Rendering with Curved Brush Strokes of Multiple Sizes; Aaron Hertzmann pg 5
    def make_stroke(self, x, y, r, blurred_image, mag):
        width, height = blurred_image.shape[0], blurred_image.shape[1]
        if (x >= width) or (y >= height):
            return
        color = blurred_image[x,y]
        all_pixels = [(x, y)]
        x1, y1 = x, y
        px, py = 0, 0
        maxlen = self.style['largeststrokelength']
        minlen = self.style['smalleststrokelength']
        curv_filter = self.style['curvature_filter']
        for i in range(1, maxlen):
            diff = np.sqrt((float(blurred_image[x1,y1,0]) - float(self.canvas[x1,y1,0]))**2 + (float(blurred_image[x1,y1,1]) - float(self.canvas[x1,y1,1]))**2 + (float(blurred_image[x1,y1,2]) - float(self.canvas[x1,y1,2]))**2)
            diff2 = np.sqrt((float(blurred_image[x1,y1,0]) - float(color[0]))**2 + (float(blurred_image[x1,y1,1]) - float(color[1]))**2 + (float(blurred_image[x1,y1,2]) - float(color[2]))**2)
            if (i > minlen and diff < diff2) or (mag[x1, y1] == 0):
                break
            gy, gx = self.sobelx[x1,y1], self.sobely[x1,y1]
            
            dx, dy = -gy, gx

            if(px * dx + py * dy < 0):
                dx, dy = -dx, -dy
            
            dx = curv_filter * dx + (1-curv_filter) * (px)
            dy = curv_filter * dy + (1-curv_filter) * (py)

            dx = dx / np.sqrt(dx**2 + dy**2)
            dy = dy / np.sqrt(dx**2 + dy**2)

            x1, y1 = (int(x1+r*dx), int(y1+r*dy))
            
            if (x1 >= width) or (y1 >= height):
                return all_pixels
            px, py = dx, dy
            all_pixels.append((int(x1), int(y1)))
    
        return all_pixels 

if __name__ == '__main__':
    
    input_sourcefile = str(sys.argv[1])
    input_style = str(sys.argv[2])
 
    painter = PaintRender(
        "input_images/"+input_sourcefile, style=input_style)
    input_file_without_ext, extension = input_sourcefile.split(".")
    output_filename = input_file_without_ext + "_" + input_style + "." + extension
    painter.paint(output_filename)

    # for style in styles:
    #     input_style = style
    #     painter = PaintRender("input_images/"+input_sourcefile, style=input_style)
    #     input_file_without_ext, extension = input_sourcefile.split(".")
    #     output_filename = input_file_without_ext + "_" + input_style + "." + extension
    #     painter.paint(output_filename)
  