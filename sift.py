import numpy as np
import cv2
import sift_utils



def sift(image: np.ndarray, sigma: float = 1.6, s: int = 3, image_border_width: int = 5):
    """Compute keypoints and keypoint descriptors using Scale-invariant Feature Transform

    Args:
        image (np.ndarray): The input image in greyscale
        sigma (float, optional): The default blurring value used. Defaults to 1.6.
        s (int, optional): The number of intervals per octave. Defaults to 3.
        image_border_width (int, optional): The border width of the image. Pixels on the border do not get processed. Defaults to 5.

    Returns:
        keypoints (list[cv2.keyPoint]): List of keypoints
        descriptors (np.ndarray): Array of descriptors
    """
    
    # Image preprocessing
    # - Double image in size, increasing number of stable keypoints
    image = image.astype(np.float32)
    base_image = sift_utils.image_preprocessing(image, sigma=sigma)
    
    
    ### 1 - Scale-space extrema detection
    # The first stage of computation searches over all scales and image locations. 
    # It is implemented efficiently by using a difference-of-Gaussian function to 
    # identify potential interest points that are invariant to scale and orientation.
    
    def scale_space_construction(base_image: np.ndarray, min_allowed_dim=3, sigma=sigma, s=s):
        n_octaves = sift_utils.get_num_octaves(image_shape=base_image.shape, min_allowed_dim=min_allowed_dim)
        gaussian_kernel_list = sift_utils.get_gaussian_kernels(sigma=sigma, s=s)
        gaussian_pyramid = sift_utils.construct_gaussian_pyramid(base_image, n_octaves, gaussian_kernel_list)
        
        return gaussian_pyramid
    
    gaussian_pyramid = scale_space_construction(base_image)
    
    dog_pyramid = sift_utils.construct_DoG_pyramid(gaussian_pyramid)
    
    # For plotting/saving the pyramids, set True if desired
    if False:
        sift_utils.plot_pyramid(gaussian_pyramid, 'resources/gaussian_pyramid.png')
        sift_utils.plot_pyramid(dog_pyramid, 'resources/dog_pyramid.png')
    
    
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
    
    file_path = 'resources/cup2.JPG'
    image = cv2.imread(file_path)
    image = cv2.resize(image, (0,0), fx=0.3, fy=0.3)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    kps, dcrpts = sift(image=image_gray)
    
    # Show image
    cv2.drawKeypoints(image, kps, outImage=image, color=(255,0,0))
    cv2.imshow("Image with keypoints!", image)
    cv2.imwrite("resources/cup2_keypoints.png", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()