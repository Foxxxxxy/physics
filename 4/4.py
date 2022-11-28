import numpy as np
import cv2
from matplotlib import pyplot as plot

SHIFT = 50
N = 121

image = cv2.imread("./img.jpg", cv2.IMREAD_GRAYSCALE)
f_shift = np.fft.fftshift(np.fft.fft2(image))


def draw_plot(i, title, cmap='gray'):
    plot.subplot(draw_plot.sub_plot)
    draw_plot.sub_plot += 1
    plot.imshow(i, cmap)
    plot.title(title)
    plot.xticks([])
    plot.yticks([])


draw_plot.sub_plot = N
draw_plot(image, "Исходная картинка")
draw_plot(np.log(np.abs(f_shift)), 'Спектр')

r, c = image.shape
cr, cc = r // 2, c // 2
f_shift[cr - SHIFT:cr + SHIFT, cc - SHIFT:cc + SHIFT] = 0

draw_plot.sub_plot = N
draw_plot(np.abs(f_shift), "Спектр без основных частот")

mod = np.fft.ifftshift(f_shift)
transformed_image = np.abs(np.fft.ifft2(mod))

draw_plot.sub_plot = N
draw_plot(image, "Исходная картинка")
draw_plot(transformed_image, "Контур")
