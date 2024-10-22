import os
import cv2
import time
import matplotlib.pyplot as plt

######################
USE_CV2_SIFT = False
######################


# Loading images
CWD = os.path.dirname(__file__)

cup1 = cv2.imread(os.path.join(CWD, 'resources/cup1.JPG'))
cup1 = cv2.resize(cup1, (0,0), fx=0.3, fy=0.3) # Resize to make processing faster
cup1_gray = cv2.cvtColor(cup1, cv2.COLOR_BGR2GRAY)

cup2 = cv2.imread(os.path.join(CWD, 'resources/cup2.JPG'))
cup2 = cv2.resize(cup2, (0,0), fx=0.3, fy=0.3)
cup2_gray = cv2.cvtColor(cup2, cv2.COLOR_BGR2GRAY)


# Run the SiFT algorithm on the images
print("Sift is running...")
if not USE_CV2_SIFT:
    from sift import sift
    
    t = time.time()
    kpts1, dscpts1 = sift(cup1_gray)
    kpts2, dscpts2 = sift(cup2_gray)
    print(f"SiFT execution time: {time.time() - t}")

elif USE_CV2_SIFT:
    t = time.time()
    cv2_sift: cv2.SIFT = cv2.SIFT_create()
    kpts1, dscpts1 = cv2_sift.detectAndCompute(cup1_gray, None)
    kpts2, dscpts2 = cv2_sift.detectAndCompute(cup2_gray, None)
    print(f"SiFT execution time: {time.time() - t}")


# Match the found keypoints and descriptors pairs
print("Matching keypoints and descriptors...")
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(dscpts1, dscpts2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])


# Draw matches and show figure
matched_img = cv2.drawMatchesKnn(cup1,kpts1,cup2,kpts2,good,None,flags=2)

plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.show()