import numpy as np
from PIL import Image

img_fn = "test_2.png"
img = Image.open(img_fn)

# 将图片处理为灰度图片
gray_img = img.convert('L')

# 将图片处理为数组
im1 = np.array(gray_img)

# 去中心化
# 计算每一列的均值，并将结果重塑为一个行向量
means = np.mean(im1, axis=0).reshape(1, -1)
# 计算标准差
sds = np.std(im1, axis=0).reshape(1, -1)
# 标准化数据
im2 = (im1 - means) / sds

# 计算 A^T A,将结果储存在S中
S = np.matmul(im2.T, im2)/1.0

# 求出特征向量和特征值 W为特征值 Q为特征向量
W, Q = np.linalg.eig(S)

# 对特征值和特征向量进行排序
w_args = np.flip(np.argsort(W))
Q = Q[:, w_args]
W = W[w_args]

# 主成分选取
k = 30

# 新的数据
C = np.matmul(im2, Q[:, :k])

# 求矩阵的转置
Q_T = Q.T[:k, :]

# # 获取矩阵的行数和列数
# rows, cols = Q_T.shape
#
# # 输出行数和列数
# print(f"行数: {rows}")
# print(f"列数: {cols}")

# 查看不同的主成分
# 选择要保留的行索引
row_index = 26
# 创建一个与原矩阵相同形状的零矩阵
result = np.zeros_like(C)
# 将选定行的元素复制到结果矩阵中
result[:, row_index] = C[:, row_index]

# 数据还原到原本的空间
#im3 = np.matmul(C, Q_T)
im3 = np.matmul(result, Q_T)
im3 = im3 * sds + means
im3 = im3.astype('uint8')
image_show = Image.fromarray(im3)
image_show.show()