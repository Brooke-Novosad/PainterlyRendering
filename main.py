import cairo
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import shutil
import os
# import Painter

class PaintRender:

    def __init__(self):
        print("here")
    

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

    def draw_circle(self, x, y, radius):
        self.context.arc(x, y, radius, 0, 2*np.pi)
        self.context.stroke()
        # canvas_data = np.ndarray(shape=(image.shape[0], image.shape[1], 4), buffer=self.canvas.get_data())
        # canvas_data = canvas_data[:, :, :3]
        # self.write_image(output_dir, "dog_output.jpg", canvas_data) 
    

if __name__ == '__main__':
    output_dir = "./output_images"
    filename = "src_images/dog.jpg"
    paint_render = PaintRender()
    image = paint_render.read_image(filename)    
    paint_render.initialize_canvas(image.shape[0],100)
    paint_render.draw_circle(0.5, 0.5, 0.2)
    paint_render.write_image(output_dir, "dog_output.jpg", image) 

# class PaintRender:

#     def __init__(self):
#         print("here")


#     def read_image(self, filename):
#         image = cv2.imread(filename)
#         return image

#     def initialize_canvas(self, width, height):
#         self.canvas = cairo.ImageSurface(cairo.FORMAT_RGB24, width, height)
#         self.context = cairo.Context(self.canvas)
#         self.context.scale(width, height)
#         self.context.set_line_cap(cairo.LINE_CAP_ROUND)
        
#     def write_image(self, output_dir, filename, image):
#         if not os.path.exists(output_dir):
#             os.mkdir(output_dir)
#         os.chmod(output_dir, 0o777)
#         fullpath = os.path.join(output_dir, filename)
#         cv2.imwrite(fullpath, image)

#     def draw_circle(self, x, y, radius):
#         self.context.arc(x, y, radius, 0, 2*np.pi)
#         self.context.stroke()

#     if __name__ == '__main__':
#         output_dir = "./output_images"
#         filename = "src_images/dog.jpg"
#         paint_render = PaintRender()
#         image = paint_render.read_image(filename)    
#         paint_render.initialize_canvas(image.shape[1], image.shape[0])
#         paint_render.draw_circle(0.5, 0.5, 0.2)
#         canvas_data = np.ndarray(shape=(image.shape[0], image.shape[1], 4), buffer=self.canvas.get_data())
#         canvas_data = canvas_data[:, :, :3]
#         paint_render.write_image(output_dir, "dog_output.jpg", canvas_data) 
