import cv2
import matplotlib.pyplot as plt
import time


######################
USE_CV2_SIFT = False #
######################

box = cv2.imread('resources/box.png', 0)
box_in_scene = cv2.imread('resources/box_in_scene.png', 0)

print("Sift is running...")
if not USE_CV2_SIFT:
    from sift import sift
    
    t = time.time()
    kpts1, dscpts1 = sift(box)
    kpts2, dscpts2 = sift(box_in_scene)
    print(f"SiFT execution time: {time.time() - t}")

if USE_CV2_SIFT:
    t = time.time()
    cv2_sift: cv2.SIFT = cv2.SIFT_create()
    kpts1, dscpts1 = cv2_sift.detectAndCompute(box, None)
    kpts2, dscpts2 = cv2_sift.detectAndCompute(box_in_scene, None)
    print(f"SiFT execution time: {time.time() - t}")



# Init matcher
print("Matching keypoints and descriptors...")
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(dscpts1, dscpts2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        
matched_img = cv2.drawMatchesKnn(box,kpts1,box_in_scene,kpts2,good,None,flags=2)

plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.show()