# Introduction 
Basic implementation of Scale-invariant Feature Transform (SiFT), created based on the original 2004 paper, "Distinctive Image Features from Scale-Invariant Keypoints" by David G. Lowe [[1](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)]. Implementation method inspired by Russ Islam [[2](https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-1-306a99b50aa5)] [[3](https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-2-c4350274be2b)].

# Getting Started

### Dependencies
```pip install -r requirements.txt```

### Usage
```py
from sift import sift
kpts1, dscpts1 = sift(greyscale_image)
```

An example of how to use the sift function can also be seen in the ```sift_matcher.py``` script, which matches keypoints and descriptors between two images.