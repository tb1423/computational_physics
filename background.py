import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np

from hermite import hermite_poly

n = 30

arr_r = np.array(hermite_poly(n))
arr_r = np.abs(arr_r)
arr_r *= 255/np.max(np.concatenate(arr_r)).T

arr_g = np.array(hermite_poly(n))
arr_g = np.abs(arr_g)
arr_g *= 255/np.max(np.concatenate(arr_g))

arr_b = np.array(hermite_poly(n))
arr_b = np.abs(arr_b)
arr_b *= 100/np.max(np.concatenate(arr_b))

arr = np.stack([arr_r, arr_g, arr_b], axis=0).T

#print(arr_b)

#print(arr)

im = Image.fromarray(arr, mode='RGBA')
im = im.filter(ImageFilter.GaussianBlur)
im = im.resize((1920,1080))

im = im.filter(ImageFilter.SMOOTH_MORE)
im = im.filter(ImageFilter.GaussianBlur)

im.save('out.png')
#im.show()
im.close()
