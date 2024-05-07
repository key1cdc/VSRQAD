import cv2
import time
import numpy as np

path = '/media/luo/data/data/NTIRE2021/test/img/001-001.png'


start = time.time()
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
end = time.time()
print(end - start)

np.save(path[:-4]+'.npy', img)

start = time.time()
img1 = np.load(path[:-4]+'.npy')
end = time.time()

print(end - start)

