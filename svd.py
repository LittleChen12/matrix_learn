import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

def svd(A):
    # 特征值，特征向量
    val, U = np.linalg.eigh(A.dot(A.T))

    # 特征值由大到小排序
    val_idx = np.argsort(val)[::-1]
    val = val[val_idx]
    Sigma = np.sqrt(np.abs(val))  #

    # 特征向量由大到小排序
    U = U[:, val_idx]

    # A = U * Sigma * VT --->  VT = inv_(U * Sigma) * A = inv(Sigma) * UT * A
    VT = np.linalg.inv(np.diag(Sigma)).dot(U.T).dot(A)
    return U, Sigma, VT


def draw(img1, img2, img3):
    fig, ax = plt.subplots(1, 3, figsize=(25, 30))
    ax[0].imshow(img1)
    ax[0].set(title='src')
    ax[1].imshow(img2)
    ax[1].set(title='nums of sigma = 1')
    ax[2].imshow(img3)
    ax[2].set(title='nums of sigma = 120')

    plt.show()


if __name__ == '__main__':
    # img_0 = "test_3.png"  # (154, 208, 3)
    # img = Image.open(img_0)
    # img = img.convert('RGB')
    # img.save('rgb_test.png')
    # print(img.shape)
    img = mpimg.imread("./rgb_test.png")  # (521, 396, 3)
    print(img.shape)
    A = img.reshape(4096, 3072 * 3)
    U, Sigma, VT = svd(A)
    print(U.shape, Sigma.shape, VT.shape)

    nums = 1
    img2 = U[:, 0:nums].dot(np.diag(Sigma[0:nums])).dot(VT[0:nums, :])
    img2 = img2.reshape(4096, 3072, 3)

    nums = 10
    img3 = U[:, 0:nums].dot(np.diag(Sigma[0:nums])).dot(VT[0:nums, :])
    img3 = img3.reshape(4096, 3072, 3)

    draw(img, img2, img3)