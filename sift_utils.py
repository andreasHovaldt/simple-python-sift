########################################################################################
### Original paper: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf                    ###
### Inspired by:    https://github.com/rmislam/PythonSIFT/blob/master/pysift.py      ###
########################################################################################

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

float_tolerance = 1e-7
counter = 0

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
#####                  2 - Keypoint localization - Section 3/4/5                   #####
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
    
    **Notes:**
        Taken from: https://github.com/rmislam/PythonSIFT/blob/master/pysift.py
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


def get_center_difference_gradient(neighbor_array: np.ndarray) -> np.ndarray:
    """Compute the gradient at the center pixel of a 3x3x3 array using the central difference formula of order O(h^2).
        
    Args:
        neighbor_array (np.ndarray): A 3x3x3 array representing pixel values. 1 center pixel with 28 neighbors.
    
    Returns:
        np.ndarray: A 1D array containing the gradients [dx, dy, ds] along the x, y, s axes, respectively.
        
    **Notes**:
        *Finite (Discrete) differences formula taken from:* 
        https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf 
        https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences 

        *Inspired by the implementation of:*
        https://github.com/rmislam/PythonSIFT/blob/master/pysift.py
    """
    
    # Centered difference approximation for the gradient, f'(x), is given by:
    #   f′(x) : {f (x + ∆x) − f (x − ∆x)} / (2∆x)
    #   where ∆x is the step size, given we work in discrete space, ∆x = 1
    #   f′(x) : {f (x + 1) − f (x − 1)} / 2
    
    # Remember image cube is created by stacking the image slices on axis=0 (first axis),
    #  leading to the shape of image cube is (channel,height,width) = (s,y,x)  (s = interval)
    
    
    # Calculating the gradients:
    # TODO: Test if multiplying by 0.5 instead of dividing by 2 makes a difference
    # First axis: s
    ds = (neighbor_array[2,1,1] - neighbor_array[0,1,1]) / 2
    
    # Second axis: y
    dy = (neighbor_array[1,2,1] - neighbor_array[1,0,1]) / 2
    
    # Third axis: x
    dx = (neighbor_array[1,1,2] - neighbor_array[1,1,0]) / 2
    
    # Return the gradient
    return np.array([dx, dy, ds])
    

def get_center_difference_hessian(neighbor_array: np.ndarray) -> np.ndarray:
    """
    Compute the Hessian matrix at the center pixel of a 3x3x3 array using the central difference formula of order O(h^2).
    
    Args:
        neighbor_array (np.ndarray): A 3x3x3 array representing pixel values. 1 center pixel with 28 neighbors.
    
    Returns:
        np.ndarray: A 3x3 Hessian matrix computed at the center pixel [1, 1, 1].
        
    **Notes:**
        The Hessian matrix is computed as follows:
        [dxx, dxy, dxs]
        [dxy, dyy, dys]
        [dxs, dys, dss]
        
        *Finite (Discrete) differences formulas taken from:* 
        https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf 
        https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences 

        *Inspired by the implementation of:*
        https://github.com/rmislam/PythonSIFT/blob/master/pysift.py
    """
    
    # Centered difference approximation for the second-order derivative, f''(x) is given by:
    #   f′′(x) : {f(x + ∆x) − 2*f(x) + f(x − ∆x)} / ∆x^2
    #   where ∆x is the step size, given we work in discrete space, ∆x = 1,
    #   f′′(x) : {f(x + 1) − 2*f(x) + f(x − 1)}

    # Centered difference approximation for the multivariate second-order derivative, f''(x,y) is given by:
    #   f′′(x,y) : {f(x + ∆x, y + ∆y) - f(x + ∆x, y - ∆y) - f(x - ∆x, y + ∆y) + f(x - ∆x, y - ∆y) } / 4 * ∆x * ∆y
    #   where ∆x and ∆y is the step size, given we work in discrete space, ∆x = 1 and ∆y = 1,
    #   f′′(x,y) : {f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1) } / 4
    
    # Remember image cube is created by stacking the image slices on axis=0 (first axis),
    #  leading to the shape of image cube is (channel,height,width) = (s,y,x)  (s = interval)
    
    
    # Calculating the second-order partial derivatives for the hessian:
    # Same variable second-order partial derivatives
    dss = neighbor_array[2, 1, 1] - 2*neighbor_array[1, 1, 1] + neighbor_array[0, 1, 1]
    dyy = neighbor_array[1, 2, 1] - 2*neighbor_array[1, 1, 1] + neighbor_array[1, 0, 1]
    dxx = neighbor_array[1, 1, 2] - 2*neighbor_array[1, 1, 1] + neighbor_array[1, 1, 0]
    
    # Mixed variable second-order partial derivatives
    # TODO: Test if multiplying by 0.25 makes a difference
    dxy = (neighbor_array[1, 2, 2] - neighbor_array[1, 2, 0] - neighbor_array[1, 0, 2] + neighbor_array[1, 0, 0]) / 4
    dxs = (neighbor_array[2, 1, 2] - neighbor_array[2, 1, 0] - neighbor_array[0, 1, 2] + neighbor_array[0, 1, 0]) / 4
    dys = (neighbor_array[2, 2, 1] - neighbor_array[2, 0, 1] - neighbor_array[0, 2, 1] + neighbor_array[0, 0, 1]) / 4
    
    # Return the hessian matrix
    return np.array([[dxx, dxy, dxs], [dxy, dyy, dys],[dxs, dys, dss]])


def accurate_keypoint_localization(center_y: int, center_x: int, image_idx: int, octave_idx: int, s: int, dog_images_in_octave: list[np.ndarray], sigma: float, contrast_threshold: float, image_border_width: int, r=10, num_attempts_until_convergence=5) -> tuple[cv2.KeyPoint, int] | None:
    """Localizes keypoints accurately by fitting a 3D quadratic function to the local sample points.
    Args:
        center_y (int): The y-coordinate of the center pixel.
        center_x (int): The x-coordinate of the center pixel.
        image_idx (int): The index of the image in the octave.
        octave_idx (int): The index of the octave.
        s (int): The number of scale levels per octave.
        dog_images_in_octave (list of np.ndarray): The Difference of Gaussian images in the current octave.
        sigma (float): The initial sigma value used in the scale-space.
        contrast_threshold (float): The threshold for contrast.
        image_border_width (int): The width of the image border.
        r (int, optional): The eigenvalue ratio used to eliminate edge responses. Defaults to 10.
        num_attempts_until_convergence (int, optional): The number of attempts to try for convergence. Defaults to 5.
    
    Returns:
        tuple: A tuple containing the keypoint (cv2.KeyPoint) and the image index (int) if successful, otherwise None.
    
    **Notes:**
        Secton 4 - Accurate keypoint localization
        Fit a 3D quadratic function to the local sample points to determine the interpolated location of the extremum
        Enables algorithm to localize sub-pixel extremum locations, which can then be broadcast back to the discrete image space.
    """
    
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    
    # Define number of convergence attempts, this convergence process is described on page 11 of the paper
    for attempt_idx in range(num_attempts_until_convergence):
        # Construct the image stack as pictured in Figure 2 in the paper
        # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds (following the equation on page 11)
        layer1, layer2, layer3 = dog_images_in_octave[image_idx-1:image_idx+2]
        center_pixel_stack = np.stack([layer1[center_y-1:center_y+2, center_x-1:center_x+2], layer2[center_y-1:center_y+2, center_x-1:center_x+2], layer3[center_y-1:center_y+2, center_x-1:center_x+2]]).astype('float32') / 255.
        
        # Compute the gradiant and the hessian used for the Taylor series expansion when fitting the 3D quadratic function
        gradient = get_center_difference_gradient(center_pixel_stack)
        hessian = get_center_difference_hessian(center_pixel_stack)
        
        # Calculate the solution to the problem of the quadratic function, the solution is then the subpixel extremum
        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0] 
        # Returns the solution (extremum_update) in a [x,y,s] array, because of the hessian and gradient construction method
        
        # Check if the extremum lies closest to the current center pixel
        # From paper: "If the offset x hat is larger than 0.5 in any dimension, then it means that the extremum lies closer to a different sample point."
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            # Extremum lies closest to the current center pixel, thus we have converged and break the convergence loop 
            break
        
        # Extremum doesn't lie closest to the current center pixel, thus we update the center pixel for the next iteration
        center_x, center_y, image_idx = int(round(extremum_update[0])), int(round(extremum_update[1])), int(round(extremum_update[2]))
        
        # Check if the new center pixel is within the image
        if (center_y < image_border_width or center_y >= image_shape[0] - image_border_width 
         or center_x < image_border_width or center_x >= image_shape[1] - image_border_width 
         or image_idx < 1 or image_idx > s):
            
            # New center pixel is outside the image, meaning the extremum also is,
            #   thus we break the convergence loop and set flag to true.
            extremum_is_outside_image = True
            break
    
    # Following the convergence loop, if convergence wasn't possible, return None
    if extremum_is_outside_image: return None
    if attempt_idx >= num_attempts_until_convergence - 1: return None
    
    # Now that we have the gradient and the cenverged updated extremum location
    #   the combined equation of Equation (2) and (3), on page 11, can be applied
    #   where, D is the center pixel, and x hat is the updated extremum location
    function_value_at_extremum = center_pixel_stack[1,1,1] + 0.5 * np.dot(gradient, extremum_update)
    
    # Check if the extremum upholds the contrast threshold
    if abs(function_value_at_extremum) * s >= contrast_threshold:
        
        ### Section 4.1: Eliminating edge responses ###
        H = hessian[:2, :2] # Equation (4), get only the x and y part of the hessian
        trace_H = np.trace(H)
        determinant_H = np.linalg.det(H)
        
        # From paper:
        # "The experiments in this paper use a value of r = 10, which eliminates
        # keypoints that have a ratio between the principal curvatures greater than 10."
        if determinant_H > 0 and r * (trace_H ** 2) < ((r + 1) ** 2) * determinant_H:
            
            # Taken from https://github.com/rmislam/PythonSIFT/blob/master/pysift.py#L177, used to make it compatible with cv2
            
            # Contrast check passed -- construct and return OpenCV KeyPoint object
            keypoint = cv2.KeyPoint()
            
            # opencv docs: Coordinates of the keypoints 
            keypoint.pt = ((center_x + extremum_update[0]) * (2 ** octave_idx), (center_y + extremum_update[1]) * (2 ** octave_idx))
            
            # opencv docs: Octave (pyramid layer) from which the keypoint has been extracted 
            keypoint.octave = octave_idx + image_idx * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            
            # opencv docs: Diameter of the meaningful keypoint neighborhood 
            # Computing the absolute scale of the keypoint
            keypoint.size = sigma * (2 ** ((image_idx + extremum_update[2]) / np.float32(s))) * (2 ** (octave_idx + 1))  # octave_index + 1 because the input image was doubled
            
            # opencv docs: The response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling 
            keypoint.response = abs(function_value_at_extremum)
            
            # Return the keypoint and the image index
            return keypoint, image_idx
        
    return None


def get_scale_space_extrema(gaussian_images, dog_images, s, sigma, image_border_width, contrast_threshold=0.04):
    # TODO: Add docstring
    
    # Accurate keypoint localization, section 4+5 of the paper
    # s -> number of intervals
    
    
    # TODO: Where is this threshold from?
    threshold = np.floor(0.5 * contrast_threshold / s * 255)  # from OpenCV implementation
    
    # List for saving the returned final keypoints
    keypoints = []
    
    # Go through each octave
    for octave_idx, dog_images_in_octave in enumerate(dog_images):
        # Go through each image in the current octave, the images are stacked following Figure 2 in the paper
        for image_idx, (layer1, layer2, layer3) in enumerate(zip(dog_images_in_octave[:-2], dog_images_in_octave[1:-1], dog_images_in_octave[2:])):
            assert layer1.shape == layer2.shape == layer3.shape, "Layers of DoG stack did not have same shape" # FIXME: If it works, delete this safety check
            # Go through each pixel (x,y) of the images, the current pixel is considered the "center pixel" 
            # Each center pixel is tested for its possiblity of being an extrema
            for center_y in range(image_border_width, layer1.shape[0] - image_border_width):
                for center_x in range(image_border_width, layer1.shape[1] - image_border_width):
                    # Perform naive check of pixel's possiblity of being a local extremum
                    if is_pixel_extremum(layer1[center_y-1:center_y+2, center_x-1:center_x+2], layer2[center_y-1:center_y+2, center_x-1:center_x+2], layer3[center_y-1:center_y+2, center_x-1:center_x+2], threshold=threshold):
                        
                        # NOTE: Due to slicing doing an [inclusive:exclusive]
                        #       operation we need to do [center -1 : center + 2]
                        #       to get the 3x3 array slice
                        
                        
                        
                        # Perform subpixel approximation of the extremum to estimate the best possible keypoint localization
                        accurate_keypoint_localization_result = accurate_keypoint_localization(center_y, center_x, image_idx + 1, # +1 to get the center image in the stack
                                                                                               octave_idx, s, dog_images_in_octave, sigma, 
                                                                                               contrast_threshold, image_border_width)
                        
                        
                        
                        # If the approximation of the extremum is within the image
                        if accurate_keypoint_localization_result is not None:
                            keypoint, scale_idx = accurate_keypoint_localization_result
                            
                            global counter
                            counter += 1
                            
                            
                            # Section 5: Orientation assignment
                            keypoints_with_orientations = orientation_assignment(keypoint, octave_idx, gaussian_images[octave_idx][scale_idx])
                            
                            # Save the keypoints with orientations to the final keypoints list
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)

    return keypoints


def orientation_assignment(keypoint: cv2.KeyPoint, octave_idx: int, gaussian_image: np.ndarray, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """Assigns orientation(s) to a keypoint based on the local image gradient directions.
    Parameters:
        keypoint (cv2.KeyPoint): The keypoint for which the orientation is being assigned.
        octave_idx (int): The index of the octave in which the keypoint was detected.
        gaussian_image (np.ndarray): The Gaussian-blurred image in which the keypoint was detected.
        radius_factor (int, optional): The factor used to determine the radius of the region around the keypoint. Default is 3.
        num_bins (int, optional): The number of bins in the orientation histogram. Default is 36.
        peak_ratio (float, optional): The ratio used to determine the threshold for peak detection in the histogram. Default is 0.8.
        scale_factor (float, optional): The factor used to scale the keypoint size. Default is 1.5.
    Returns:
        keypoints_with_orientations (list[cv2.KeyPoint]): A list of keypoints with assigned orientations.
    """
    
    # Section 5: Orientation assignment
    # As described in Section 5 of the SIFT paper (Lowe, 2004), 
    # this step assigns an orientation to each keypoint based on the local image gradient.
    # The goal is to make keypoints invariant to rotation, ensuring the descriptor remains robust
    # under different orientations of the image.

    keypoints_with_orientations = []  # List to store keypoints with their assigned orientations.
    image_shape = gaussian_image.shape  # Get the shape of the Gaussian-blurred image.

    # The scale of the keypoint is computed based on the size and the octave index.
    # This ensures scale invariance. The scale factor is typically 1.5 (Section 5 of the SIFT paper).
    scale = scale_factor * keypoint.size / np.float32(2 ** (octave_idx + 1))

    # The radius determines the region around the keypoint to be considered for orientation calculation.
    # The radius is based on the scale of the keypoint and a factor (typically 3, but not described in the paper).
    radius = int(round(radius_factor * scale))

    # The weight factor is used to compute a Gaussian weight for each pixel in the region,
    # prioritizing pixels closer to the keypoint center. This helps give more importance to gradients near the keypoint.
    weight_factor = -0.5 / (scale ** 2) # The gaussian function is e^(-r^2/2*sigma^2), thus -1/(2*sigma^2) is the weight factor

    # The raw histogram will store gradient magnitudes for different orientation bins.
    # The number of bins is typically 36 (covering 360 degrees), and the orientation will be assigned 
    # based on the peak in this histogram. (See Section 5 of the SIFT paper).
    raw_histogram = np.zeros(num_bins)

    # A smoothed histogram will later be used to perform peak detection.
    smooth_histogram = np.zeros(num_bins)

    # Loop over the neighborhood around the keypoint (based on the radius) to compute gradient magnitudes and orientations.
    for i in range(-radius, radius + 1):
        # Calculate the y-coordinate in the scale space (adjust for the current octave).
        region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_idx))) + i

        # Ensure the y-coordinate is within the image bounds.
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                # Calculate the x-coordinate in the scale space (adjust for the current octave).
                region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_idx))) + j
                
                # Ensure the x-coordinate is within the image bounds.
                if region_x > 0 and region_x < image_shape[1] - 1:
                    # Compute the image gradients in the x and y directions (dx, dy).
                    # Gradient magnitudes and orientations are calculated from these values.
                    # The use of gradients is directly mentioned in Section 5 (Lowe, 2004).
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    
                    # Compute the gradient magnitude.
                    gradient_magnitude = np.sqrt(dx**2 + dy**2)
                    
                    # Compute the gradient orientation in degrees.
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                    
                    # Weight the gradient based on its distance from the keypoint.
                    # The weight is computed using a Gaussian function, emphasizing closer pixels.
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                    
                    # Map the gradient orientation into one of the histogram bins (36 bins for 360 degrees).
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    
                    # Add the weighted gradient magnitude to the corresponding histogram bin.
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    # Smooth the histogram to suppress noise and make the orientation assignment more robust.
    # As described in the SIFT paper, Lowe suggests smoothing the histogram before peak detection.
    for n in range(num_bins):
        # A simple smoothing operation that takes neighboring bin values into account.
        # The smoothing formula is y
        smooth_histogram[n] = (6 * raw_histogram[n] + 
                            4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + 
                            raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.



    # Find the peak of the smoothed histogram, which represents the dominant orientation.
    # This peak corresponds to the most frequent gradient direction in the keypoint's neighborhood.
    orientation_max = max(smooth_histogram)

    # Identify all local peaks in the histogram that are above 80% of the highest peak.
    # These peaks represent significant orientations in the keypoint's neighborhood.
    # This process allows for assigning multiple orientations to a keypoint, improving robustness to image rotation.
    orientation_peaks = np.where(np.logical_and(
        smooth_histogram > np.roll(smooth_histogram, 1),
        smooth_histogram > np.roll(smooth_histogram, -1)))[0]
    

    # Loop over all significant peaks to assign orientations to the keypoint.
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        
        # Only consider peaks that are within a certain threshold of the dominant peak (peak_ratio is typically 0.8).
        if peak_value >= peak_ratio * orientation_max:
            # Perform quadratic interpolation to refine the orientation.
            # This helps improve the accuracy of the peak detection (see equation (6.30) in the Stanford resource).
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            
            # Interpolated peak index for higher precision.
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            
            # Convert the peak index back to an orientation in degrees.
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            
            # Ensure the orientation is within [0, 360) degrees.
            if abs(orientation - 360.) < float_tolerance:
                orientation = 0
            
            # Create a new keypoint with the assigned orientation.
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            
            # Add the keypoint with orientation to the list.
            keypoints_with_orientations.append(new_keypoint)

    # Return the list of keypoints with assigned orientations.
    return keypoints_with_orientations







if __name__ == "__main__":
    s = numinterval = 3
    sigma = 1.6
    
    box = cv2.imread('/home/dreezy/azureDev/repos/ipcv-mini-project/resources/box.png', 0)
    
    image = box.astype('float32')
    base_image = image_preprocessing(image, sigma=sigma)
    
    
    print("base_image.type: ", type(base_image[0][0]))
    
    n_octaves = get_num_octaves(base_image.shape[:2])
    print(f"n_octaves {n_octaves}")
    
    g_kernels = get_gaussian_kernels(sigma=sigma, s=s)
    print(f"gaussian kernels: {np.round(g_kernels,7)}")
    print("g_kern.type: ", type(g_kernels[1]))
    
    g_pyramid = construct_gaussian_pyramid(base_image, n_octaves, g_kernels)
    print(f"pyramid lengths: {len(g_pyramid)}, {len(g_pyramid[0])}")
    print(g_pyramid[7][4].shape)
    
    # print("g_pyramid ", g_pyramid[0][1][2])
    # print("g_pyramid ", g_pyramid[0][1].shape)
    # print("g_pyramid ", g_pyramid[0][1][2].shape)
    # print("g_pyramid.type ", type(g_pyramid[0][1][2][0]))
    
    dog_pyramid = construct_DoG_pyramid(g_pyramid)
    print(f"dog lengths: {len(dog_pyramid)}, {len(dog_pyramid[0])}")
    print(dog_pyramid[5][2].shape)
    
    # print("dog_image ", dog_pyramid[0][1][2])
    # print("dog_image ", dog_pyramid[0][1].shape)
    # print("dog_image ", dog_pyramid[0][1][2].shape)
    # print("dog_image.type ", type(dog_pyramid[0][1][2][0]))
    
    keys_w_orients = get_scale_space_extrema(g_pyramid, dog_pyramid, s, sigma=sigma, image_border_width=3)
    print(len(keys_w_orients))
    print(keys_w_orients[0])
    
    box_copy = cv2.cvtColor(box.copy(), cv2.COLOR_GRAY2BGR)
    
    cv2.drawKeypoints(box_copy, keys_w_orients, outImage=box_copy)
    cv2.imshow("box", box_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("counter: ", counter)
    
    
    # list = [1,2,3,4,5]
    # for x,y,z in zip(list[:-2], list[1:-1], list[2:], strict=True):
    #     print(x,y,z)
    
    pass