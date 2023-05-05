# PaintRender - Artistic Image Renderer

PaintRender is a Python-based artistic image renderer that turns input images into paintings using different styles. The code uses the OpenCV and NumPy libraries to process the images and apply artistic effects.

## Dependencies

* Python
* OpenCV (cv2)
* NumPy

## Styles available

There are two styles available in the current implementation:

1. Minimalist
2. Cubism
3. Futurist
4. Colorist Wash
5. Expressionist
6. Impressionist

## Brief overview

The code uses a custom PaintRender class to perform the artistic rendering on an input image. It applies brush strokes of varying radii to create the desired painting effect. The algorithm follows these steps:

1. Read the input image
2. Initialize a canvas
3. Apply brush strokes of different radii
4. Write the output image

## How to run

1. Install Python, OpenCV (cv2), and NumPy.
2. Save the input image in the "input_images" folder.
3. Choose the desired painting style (either 'minimalist' or 'cubism').
4. Run the code using the following command:

   `python <executable> <path-to-input image> <chosen-style>`

5. The rendered image will be saved to the output_images directory.

## Citations

This code is on the research outlined in [this](https://dl.acm.org/doi/pdf/10.1145/340916.340917 ) paper by Aaron Hertzmann Ken Perlin. The implementation relies on OpenCV for image processing and NumPy for numerical operations.
