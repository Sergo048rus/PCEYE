from unittest import result
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import sinc
import scipy.signal as s
import math
import os
import sys
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter

# np.set_printoptions(threshold=sys.maxsize)

os.system('cls')
# 0. Параметры:

# IMG_FILE =  "image.png"
IMG_FILE =  "image1.png"
PIXEL_SIZE_MM = 1.7 * 1.7 * 10 ** -3
DISTANCE_MM = 200
FOCUS_MM = 5
# LowPassCore = np.array(
#     [
#         [1, 1, 1, 1, 1, 1, 1],
#         [1, 2, 2, 2, 2, 2, 1],
#         [1, 2, 3, 3, 3, 2, 1],
#         [1, 2, 3, 4, 3, 2, 1],
#         [1, 2, 3, 3, 3, 2, 1],
#         [1, 2, 2, 2, 2, 2, 1],
#         [1, 1, 1, 1, 1, 1, 1]
#     ]
# )
# LowPassCore = LowPassCore/np.sum(np.sum(LowPassCore))


def LP_filter(x, y):
    return np.sinc(x)*np.sinc(y)


# 1. Считываем файл:
imgi = np.uint8(cv2.imread(IMG_FILE, cv2.IMREAD_GRAYSCALE))

# 2. Отображаем файл:
fig0 = plt.figure('Оригинал')
plt.imshow(imgi, cmap='gray', vmin=np.min(0), vmax=np.max(imgi))
plt.draw()
plt.show(block=False)
fig0.canvas.draw()
fig0.canvas.flush_events()

result = gaussian_filter(imgi, sigma=2)
(thresh, result) = cv2.threshold(result, 107, 255, cv2.THRESH_BINARY)
plt.imshow(result)
imgi = result
# # 2.1 ФНЧ
# a = 1
# n = 51
# x = np.linspace(-a, a, n)
# y = np.linspace(-a, a, n)
# LowPassCore = LP_filter(x[:, None], y[None, :])
# print('Изображение фильтруется с помощью ФНЧ', n, 'x', n)
# # imgi = np.array(s.convolve2d(np.double(imgi), LowPassCore,
# #                 mode='same', boundary='wrap'))
# imgi = s.fftconvolve(np.double(imgi), LowPassCore, mode='same')
# imgi = 255 * (imgi/imgi.max())
# # imgi = cv2.GaussianBlur(imgi, (n, n), 0)
fig_ = plt.figure('Бинаризация')
plt.imshow(imgi, cmap='gray', vmin=np.min(0), vmax=np.max(imgi))
plt.draw()
plt.show(block=False)
# fig = plt.figure('11')
# fig.canvas.draw()
# fig.canvas.flush_events()

# 3. Формируем ядра свертки для поиска точек:
hsize = 30
width = 11
# hsize = 50
# width = 25

border = (hsize-width)//2
centr = (hsize)//2
h = np.array([[-1] * hsize] * hsize)
h[0:width, :] = 1
h[hsize-width:hsize, :] = 1
h[:, centr-width:centr+width] = 1

# hsize = 31
# h = np.array([[1] * hsize] * hsize)
# h[12:22, :12] = -1
# h[12:22, 22:] = -1
# # h = h - 2
# h = h / h.sum()

# hsize = 21
# h = np.array([[1] * hsize] * hsize)
# h[6:16, :6] = -1
# h[6:16, 14:] = -1
# h = h - 2
# h = h / h.sum()

p = []
print('Производится двумерная свёртка с окном "звезда"', hsize, 'x', hsize)
# imgo = np.array(s.convolve2d(np.double(imgi), np.flip(h),
#                              mode='same', boundary='wrap'))
imgo = s.fftconvolve(np.double(imgi), h, mode='same')
imgo = 255 * (imgo / imgo.max())
imgo[imgo < 0] = 0
imgo[imgo > 255] = 255
fig1 = plt.figure('Свёртка со звездой')
plt.imshow(imgo, cmap='gray', vmin=np.min(0), vmax=np.max(imgo))
plt.draw()
plt.show(block=False)
fig1.canvas.draw()
fig1.canvas.flush_events()

hsize = 50
a = hsize//2
b = hsize//2
radius = 25
h = np.array([[-1] * hsize] * hsize)
for y in range(hsize):
    for x in range(hsize):
        if (x-a)**2 + (y-b)**2 < radius**2:
            h[x][y] = 1

print('Производится двумерная свёртка с окном "круг"', hsize, 'x', hsize)
# imgo = np.array(s.convolve2d(np.double(imgi), np.flip(h),
#                              mode='same', boundary='wrap'))
imgo = s.fftconvolve(np.double(imgo), h, mode='same')
imgo = 255 * (imgo / imgo.max())
imgo[imgo < 0] = 0
imgo[imgo > 255] = 255
fig2 = plt.figure('Свёртка с кругом')
plt.imshow(imgo, cmap='gray', vmin=np.min(0), vmax=np.max(imgo))
plt.draw()
plt.show(block=False)
fig2.canvas.draw()
fig2.canvas.flush_events()



hsize = 40
a = hsize//2
b = hsize//2
radius = 15
h = np.array([[-1] * hsize] * hsize)
for y in range(hsize):
    for x in range(hsize):
        if (x-a)**2 + (y-b)**2 < radius**2:
            h[x][y] = 1

print('Производится двумерная свёртка с окном "круг"', hsize, 'x', hsize)
# imgo = np.array(s.convolve2d(np.double(imgi), np.flip(h),
#                              mode='same', boundary='wrap'))
imgo = s.fftconvolve(np.double(imgo), h, mode='same')
imgo = 255 * (imgo / imgo.max())
imgo[imgo < 0] = 0
imgo[imgo > 255] = 255
fig2 = plt.figure('Свёртка с кругом')
plt.imshow(imgo, cmap='gray', vmin=np.min(0), vmax=np.max(imgo))
plt.draw()
plt.show(block=False)
fig2.canvas.draw()
fig2.canvas.flush_events()



hsize = 30
a = hsize//2
b = hsize//2
radius = 10
h = np.array([[-1] * hsize] * hsize)
for y in range(hsize):
    for x in range(hsize):
        if (x-a)**2 + (y-b)**2 < radius**2:
            h[x][y] = 1

print('Производится двумерная свёртка с окном "круг"', hsize, 'x', hsize)
# imgo = np.array(s.convolve2d(np.double(imgi), np.flip(h),
#                              mode='same', boundary='wrap'))
imgo = s.fftconvolve(np.double(imgo), h, mode='same')
imgo = 255 * (imgo / imgo.max())
imgo[imgo < 0] = 0
imgo[imgo > 255] = 255
fig2 = plt.figure('Свёртка с кругом')
plt.imshow(imgo, cmap='gray', vmin=np.min(0), vmax=np.max(imgo))
plt.draw()
plt.show(block=False)
fig2.canvas.draw()
fig2.canvas.flush_events()



hsize = 5
h = np.array([[-1] * hsize] * hsize)
h[(hsize-1)//2, (hsize-1)//2] = 255




print('Производится двумерная свёртка с окном "точка"', hsize, 'x', hsize)
# imgo = np.array(s.convolve2d(np.double(imgi), np.flip(h),
#                              mode='same', boundary='wrap'))
imgo = s.fftconvolve(np.double(imgo), h, mode='same')
imgo = 255 * (imgo / imgo.max())
imgo[imgo < 0] = 0
imgo[imgo > 255] = 255
fig2 = plt.figure('Свёртка с точкой')
plt.imshow(imgo, cmap='gray', vmin=np.min(0), vmax=np.max(imgo))
plt.draw()
plt.show(block=False)
fig2.canvas.draw()
fig2.canvas.flush_events()
# p = np.argwhere(imgo > 130)
p = np.argwhere(imgo > 140)

points = []

flag = 0
next_i = 0
for i in range(0, len(p)):
    index = []
    for j in range(0, len(p)):
        if (p[i][0]-p[j][0])**2 + (p[i][1]-p[j][1])**2 < radius**2:
            index.append([True, True])
        else:
            if flag == 0:
                next_i = j
                flag = 1
            index.append([False, False])
    i = j
    flag = 0
    tmp = np.reshape(p[index], (-1, 2))
    points.append(np.floor(np.mean(tmp, axis=0)))
points = np.unique(points, axis=0)
print('\nНайдено', len(points), 'звёзд:')
for i in range(0, len(points)):
    print('Точка', i, ':', points[i])

# 4 Поиск попарного пути

print('\n')
SUM_res = 0

for i in range(len(points)):
    for j in range(i, len(points)):
        if i != j:
            resultLength = np.sqrt((points[i][0] - points[j][0]) **
                    2 + (points[i][1] - points[j][1])**2)
            res_mm = int((resultLength * PIXEL_SIZE_MM) * DISTANCE_MM / FOCUS_MM)
            print(f"Расстояние в мм между 'Объектом {i}' и 'Объектом {j}' равно {res_mm} мм")
            SUM_res = SUM_res + res_mm
        

print(f'\nПолная длина равна: {SUM_res}мм')










# len0to2 = np.sqrt((points[0][0] - points[2][0]) **
#                   2 + (points[0][1] - points[2][1])**2)

# len2to5 = np.sqrt((points[2][0] - points[5][0]) **
#                   2 + (points[2][1] - points[5][1])**2)

# len5to3 = np.sqrt((points[5][0] - points[3][0]) **
#                   2 + (points[5][1] - points[3][1])**2)

# len3to1 = np.sqrt((points[3][0] - points[1][0]) **
#                   2 + (points[3][1] - points[1][1])**2)

# len1to4 = np.sqrt((points[1][0] - points[4][0]) **
#                   2 + (points[1][1] - points[4][1])**2)

# len4to6 = np.sqrt((points[4][0] - points[6][0]) **
#                   2 + (points[4][1] - points[6][1])**2)

# resultLength = len0to2+len2to5+len5to3+len3to1+len1to4+len4to6
# resultLength_mm = (resultLength * PIXEL_SIZE_MM) * DISTANCE_MM / FOCUS_MM

# print('\nДлина пути в пикселях:', resultLength)
# print('Длина пути в мм:', resultLength_mm)

input("\nPress ENTER...")
