########################################################################################
### Heavily inspired by: https://github.com/rmislam/PythonSIFT/blob/master/pysift.py ###
########################################################################################

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math



###################################################################################
######             1 - Scale-space extrema detection - Section 3              #####
# The first stage of computation searches over all scales and image locations.    #
# It is implemented efficiently by using a difference-of-Gaussian function to     #
# identify potential interest points that are invariant to scale and orientation. #
###################################################################################

def image_preprocessing(image: np.ndarray, sigma=1.6, assumed_sigma=0.5) -> np.ndarray:
    # Based on section 3.3
    # - sigma (desired blur) in paper = 1.6
    # - assumed sigma (assumed blur of original image) in paper = 0.5
    
    # Double image in size, increasing number of stable keypoints
    # - In the paper resizing is done using linear interpolation, which is the cv2 default
    image = cv2.resize(image, (0,0), fx=2, fy=2)
    
    # Blurring an input image by kernel size σ₁ and then blurring the resulting image by σ₂ 
    #   is equivalent to blurring the input image just once by σ, where σ² = σ₁² + σ₂².
    
    # In this case, we want to calculate how much additional blur is needed, given that 
    #   the image already has some assumed blur. To find the difference, we subtract 
    #   the existing blur from the desired blur.
    # Can be seen in Equation (1) of the paper (i think, but maybe not, probably maybe not)
    
    # Additionally, given the doubled image size, the assumed sigma must also be doubled, given its new pixel spacing.
    doubled_assumed_sigma = assumed_sigma * 2
    sigma_difference = np.sqrt(sigma**2 - doubled_assumed_sigma**2)
    
    # Blurring the image
    blurred_image = cv2.GaussianBlur(image, (0,0), sigmaX=sigma_difference, sigmaY=sigma_difference)
    
    return blurred_image


def get_num_octaves(image_shape: tuple, min_allowed_dim=3) -> int:
    """Calculate the maximum allowed number of octaves for the image

    Args:
        image_shape (tuple): Height and width shape of the input image
        min_allowed_dim (int, optional): The smallest allowed image dimension of an ocatave image. Defaults to 3, based on Section 3.1 and Figure 2.

    Returns:
        num_octaves (int): Number of octaves for the input image
    """
    # min_allowed_dim defaults to 3, given extrema are compared in a 3x3 neighborhood, section 3.1
    
    assert len(image_shape) == 2, "Image shape passed to function must have lenght = 2"
    
    height, width = image_shape
    smallest_image_dim = min(height, width)
    
    # Calculates the maximum number of octaves possible, uextremasing log2 to take 
    #   into account that the image is halved in size for each octave
    num_octaves = np.log2(smallest_image_dim / min_allowed_dim) + 1 # +1 to account for initial starting octave
    
    return int(num_octaves)


def get_gaussian_kernels(sigma=1.6, s=3):
    """
    Computes a list of Gaussian kernel sigma values for each interval within an octave.

    Args:
        sigma (float): The initial sigma for the base level in the octave (typically 1.6, as suggested in section 3.3).
        s (int, optional): Number of intervals per octave (default 3, as suggested in section 3.2).

    Returns:
        kernels_diff (list): A list of sigma difference values representing the additional blur applied at each interval.
    """
    
    # Section 3.2: "Highest repeatability is obtained when sampling 3 scales per octave"
    
    # Calculate the number of intervals per octave
    octave_stack_size = s + 3 # "We must produce s+3 images in the stack of blurred images for each octave..."
    
    
    # k is the scale factor between consecutive intervals within an octave
    k = 2 ** (1. / s)
    
    # Compute the sigma values for the Gaussian kernels at each interval
    kernels = [(k**kernel_idx) * sigma for kernel_idx in range(octave_stack_size)] # corresponding to σ,kσ,k²σ,k³σ,...
    
    # Compute the sigma difference for the Gaussian kernels at each interval
    # The first value remains the same since there's no previous sigma to subtract from.
    kernels_diff = [np.sqrt(kernels[k_idx]**2 - kernels[k_idx-1]**2) for k_idx in range(1, octave_stack_size)] # Math seen in image_preprocessing function aswell
    kernels_diff.insert(0, sigma)
    
    return kernels_diff

    
    

def construct_gaussian_pyramid(image: np.ndarray, num_octaves: int, gaussian_kernels_diff: list):
    """Generate scale-space pyramid of Gaussian images

    Args:
        image (np.ndarray): The pre-processed base octave image
        num_octaves (int): Number of octaves for the input image
        gaussian_kernels_diff (list): A list of sigma difference values representing the additional blur applied at each interval.

    Returns:
        gaussian_pyramid (np.ndarray): 2D array containing the gaussian blurred images of each octave. 
            Example: ```gaussian_pyramid[octave][interval]```
    """
    
    gaussian_pyramid = []
    
    # Go through each octave
    for _ in range(num_octaves):
        octave_stack = []
        octave_stack.append(image) # The first image has already been blurred (see "image_preprocessing()"")
        
        # Go through each kernel
        for kernel_diff in gaussian_kernels_diff[1:]:
            # Apply additional blur difference
            image = cv2.GaussianBlur(image, (0,0), sigmaX=kernel_diff, sigmaY=kernel_diff)
            
            # Save blurred image into stack
            octave_stack.append(image)
        
        # Append octave stack to gaussian pyramid
        gaussian_pyramid.append(octave_stack)
        
        ### Define the base for the next octave #############################################
        # In the paper, bottom of section 3, it is written that each new octave must start
        #   with a resampled base image with a sigma twice the previous octave's base sigma
        # When observing the kernels, with sigma=1.6 and s = 3:
        #   [1.6, 2.015, 2.539, 3.2, 4.031, 5.079]
        #   It can be seen that the kernel has doubled by index 3 or from the back, index -3.
        #   Thus, the base of the next octave will be approximated to octave_stack[-3]
        next_octave_base = octave_stack[-3]
        
        # Downsample - Resize to half size for next octave
        image = cv2.resize(next_octave_base, 
                           (int(next_octave_base.shape[1]/2) , int(next_octave_base.shape[0]/2)), # (resize uses (x,y), while shape uses (y,x))
                           interpolation=cv2.INTER_LINEAR)
    
    # Return the gaussian pyramid
    return np.array(gaussian_pyramid, dtype=object)



def construct_DoG_pyramid(gaussian_pyramid: np.ndarray):
    """Calculates the Difference-of-Gaussians (DoG), as described in section 3 of the paper.
    Functionally it identifies potential interest points that are invariant to 
    scale and orientation by searching over all scales and image locations.
    
    Args:
        gaussian_pyramid (np.ndarray): 2D array containing the gaussian blurred images of each octave. 
            Example: ```gaussian_pyramid[octave][interval]```
    
    Returns:
        dog_pyramid (np.ndarray): 2D array containing the difference of gaussian images of each octave. 
            Example: ```dog_pyramid[octave][interval]```
    """
    
    dog_pyramid = []
    
    # Go through each octave of the pyramid
    for octave_gaussian_images in gaussian_pyramid:
        
        # Create list for the octave DoG images
        octave_dog_images = []
        
        # Match layer 'n' with layer 'n+1', as seen in Figure 1
        for layer1, layer2 in zip(octave_gaussian_images[:-1], octave_gaussian_images[1:]):
            
            # Debug: Make sure layer shapes match
            assert layer1.shape == layer2.shape, "Layers did not have same shape when computing DoG pyramid"
            
            # Subtract layers to get difference of gaussian
            dog_image = cv2.subtract(layer2, layer1) # Using cv2.subtract because it uses saturated subtraction, meaning it clips the result to the range [0,255]
            
            # Save current DoG image
            octave_dog_images.append(dog_image)
        
        # Save DoG images from current octave
        dog_pyramid.append(octave_dog_images)
            
    # Return DoG pyramid
    return np.array(dog_pyramid, dtype=object)




########################################################################################
#####                   2 - Keypoint localization - Section 3/4                    #####
# At each candidate location, a detailed model is fit to determine location and scale. #
# Keypoints are selected based on measures of their stability.                         #
########################################################################################

def is_pixel_extremum(first_subimage: np.ndarray, second_subimage: np.ndarray, third_subimage: np.ndarray, threshold: float) -> bool:
    """Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors, False otherwise

    Args:
        first_subimage (np.ndarray): First subimage
        second_subimage (np.ndarray): Second subimage
        third_subimage (np.ndarray): Third subimage
        threshold (float): Threshold value for pixel to be considered an extremum

    Returns:
        bool: Whether the pixel was strictly greater or less than all its neighbors.
    """
    # Set the center pixel value
    center_pixel_value = second_subimage[1, 1]
    
    # Check if the center pixel value is greater than the threshold
    if abs(center_pixel_value) > threshold:
        
        # Check if the center pixel value is greater than than all its neighbors
        if center_pixel_value > 0:
            return np.all(center_pixel_value >= first_subimage) and \
                   np.all(center_pixel_value >= third_subimage) and \
                   np.all(center_pixel_value >= second_subimage[0, :]) and \
                   np.all(center_pixel_value >= second_subimage[2, :]) and \
                   center_pixel_value >= second_subimage[1, 0] and \
                   center_pixel_value >= second_subimage[1, 2]
                   
        # Check if the center pixel value is less than all its neighbors
        elif center_pixel_value < 0:
            return np.all(center_pixel_value <= first_subimage) and \
                   np.all(center_pixel_value <= third_subimage) and \
                   np.all(center_pixel_value <= second_subimage[0, :]) and \
                   np.all(center_pixel_value <= second_subimage[2, :]) and \
                   center_pixel_value <= second_subimage[1, 0] and \
                   center_pixel_value <= second_subimage[1, 2]
    return False


def get_center_pixel_gradient(pixel_array: np.ndarray) -> np.ndarray:
    """Compute the gradient at the center pixel of a 3x3x3 array using the central difference formula of order O(h^2).
    
    Args:
        pixel_array (np.ndarray): A 3x3x3 array representing pixel values.
    
    Returns:
        np.ndarray: A 1D array containing the gradients [dx, dy, ds] along the x, y, s axes, respectively.
        
    Notes:
        Taken from https://github.com/rmislam/PythonSIFT/blob/master/pysift.py
    """
    
    # Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
    # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return np.array([dx, dy, ds])


def get_center_pixel_hessian(pixel_array: np.ndarray) -> np.ndarray:
    """
    Compute the Hessian matrix at the center pixel of a 3x3x3 array using the central difference formula of order O(h^2).
    
    Args:
        pixel_array (np.ndarray): A 3x3x3 array representing pixel values.
    
    Returns:
        np.ndarray: A 3x3 Hessian matrix computed at the center pixel [1, 1, 1].
        
    Notes:
        Taken from https://github.com/rmislam/PythonSIFT/blob/master/pysift.py
    
        The Hessian matrix is computed as follows:
        [dxx, dxy, dxs]
        [dxy, dyy, dys]
        [dxs, dys, dss]
    """
    
    # Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    
    # Compute the second-order partial derivatives
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    
    return np.array([[dxx, dxy, dxs], 
                     [dxy, dyy, dys],
                     [dxs, dys, dss]])


def accurate_keypoint_localization(): # FIXME:
    # Secton 4 - Accurate keypoint localization
    # Fit a 3D quadratic function to the local sample points to determine the interpolated location of the extremum
    # Enables algorithm to localize sub-pixel extremum locations, which can then be broadcast back to the discrete image space.
    
    
    return None


def get_scale_space_extrema(s=3, contrast_threshold=0.04): # FIXME:
    # Accurate keypoint localization, section 4 of the paper
    # s -> number of intervals
    
    threshold = np.floor(0.5 * contrast_threshold / s * 255)
    keypoints = []
    
    return keypoints













if __name__ == "__main__":
    # box = cv2.imread('/home/dreezy/azureDev/repos/ipcv-mini-project/box.png', 0)
    
    # image = box.astype('float32')
    # base_image = image_preprocessing(box)
    
    
    # n_octaves = get_num_octaves(base_image.shape[:2])
    # print(f"n_octaves {n_octaves}")
    
    # g_kernels = get_gaussian_kernels()
    # print(f"gaussian kernels: {g_kernels}")
    
    # g_pyramid = construct_gaussian_pyramid(base_image, n_octaves, g_kernels)
    # print(f"pyramid lengths: {len(g_pyramid)}, {len(g_pyramid[0])}")
    # print(g_pyramid[7][4].shape)
    
    # dog_pyramid = construct_DoG_pyramid(g_pyramid)
    # print(f"dog lengths: {len(dog_pyramid)}, {len(dog_pyramid[0])}")
    # print(dog_pyramid[5][2].shape)
    
    list = [1,2,3,4,5]
    for x,y,z in zip(list[:-2], list[1:-1], list[2:], strict=True):
        print(x,y,z)
    
    