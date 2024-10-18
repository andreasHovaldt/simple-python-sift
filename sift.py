import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

import sift_utils



def sift(image: np.ndarray, sigma: float = 1.6, s: int = 3, image_border_width: int = 5):
    """Compute keypoints and keypoint descriptors using Scale-invariant Feature Transform

    Args:
        img (np.ndarray): Image
        

    Returns:
        _type_: _description_
    """
    
    # Image preprocessing
    # - Double image in size, increasing number of stable keypoints
    image = image.astype(np.float32)
    base_image = sift_utils.image_preprocessing(image, sigma=sigma)
    
    
    ### 1 - Scale-space extrema detection
    # The first stage of computation searches over all scales and image locations. 
    # It is implemented efficiently by using a difference-of-Gaussian function to 
    # identify potential interest points that are invariant to scale and orientation.
    
    n_octaves = sift_utils.get_num_octaves(image_shape=base_image.shape, min_allowed_dim=3)
    gaussian_kernel_list = sift_utils.get_gaussian_kernels(sigma=sigma, s=s)
    gaussian_pyramid = sift_utils.construct_gaussian_pyramid(base_image, n_octaves, gaussian_kernel_list)
    dog_pyramid = sift_utils.construct_DoG_pyramid(gaussian_pyramid)
    
    
    ### 2/3 - Keypoint localization and Orientation assignment
    # At each candidate location, a detailed model is fit to determine location and scale. 
    # Keypoints are selected based on measures of their stability.
    
    # One or more orientations are assigned to each keypoint location based on local image
    # gradient directions. All future operations are performed on image data that has been
    # transformed relative to the assigned orientation, scale, and location for each feature,
    # thereby providing invariance to these transformations.
    
    keypoints_with_orientations = sift_utils.get_scale_space_extrema(gaussian_pyramid, dog_pyramid, s, sigma, image_border_width)
    
    ## Remove duplicate keypoints and convert to input image size
    keypoints = sift_utils.removeDuplicateKeypoints(keypoints_with_orientations)
    keypoints = sift_utils.convertKeypointsToInputImageSize(keypoints)
    
    ### 4 - Keypoint descriptor
    descriptors = sift_utils.generateDescriptors(keypoints, gaussian_pyramid)
    
    return keypoints, descriptors



if __name__ == "__main__":
    print(__name__)
    
    box = cv2.imread('/home/dreezy/azureDev/repos/ipcv-mini-project/resources/box.png', 0)
    
    kps, dcrpts = sift(image=box)
    print(dcrpts[0])
    
    
    # Show image
    box_copy = cv2.cvtColor(box.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawKeypoints(box_copy, kps, outImage=box_copy)
    cv2.imshow("box", box_copy)
    cv2.imwrite("resources/box_keypoints.png", box_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()