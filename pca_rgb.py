from PIL import Image
import numpy as np

img_fn = "test_2.png"
img = Image.open(img_fn)

# convert to RGB mode
rgb_img = img.convert('RGB')

# convert to numpy array
im = np.array(rgb_img) 

# simply combine the three (R,G,B) channels
im1 = np.hstack((im[:,:,0], im[:,:,1], im[:,:,2]))

## normalization. For each column, mean=0, sd=1
#means = np.mean(im1, axis=0).reshape(1, -1)
#sds = np.std(im1, axis=0).reshape(1, -1)
#im2 = (im1 - means) / sds
im2 = im1

# compute the eigenvalues and eigenvectors of {A^T}A
S = np.matmul(im2.T, im2)
W, Q = np.linalg.eig(S)

# sort the eigenvalues and corresponding eigenvectors
# from largest to smallest
w_args = np.flip(np.argsort(W))
print(w_args)
Q = Q[:, w_args]
W = W[w_args]

# calculate new scores (coordinates)
C = np.matmul(im2, Q)

k = 800   # CHANGE ME! number of PCs to keep

# reconstruct the image data with k PCs
im3 = np.matmul(C[:, :k], Q.T[:k, :])
#im3 = im3 * sds + means
im3 = im3.astype('uint8')

# reconstruct the three (R,G,B) channels
im3_channels = np.hsplit(im3, 3)
im4 = np.zeros_like(im)
for i in range(3):
    im4[:,:,i] = im3_channels[i]
Image_show = Image.fromarray(im4)
Image_show.show()