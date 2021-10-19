from pathlib import Path

import numpy as np
from skimage.io import imread, imsave


def dft(f):
    M, N = f.shape
    F = np.zeros_like(f, dtype=np.complex128)
    for u in range(M):
        for v in range(N):
            for x in range(M):
                for y in range(N):
                    F[u, v] += f[x, y] * np.exp(-1j * 2 * np.pi * (((u * x) / M) + ((v * y) / N)))
    return F


def idft(F):
    M, N = F.shape
    f = np.zeros_like(F, dtype=np.complex128)
    for x in range(M):
        for y in range(N):
            for u in range(M):
                for v in range(N):
                    f[x, y] += F[u, v] * np.exp(1j * 2 * np.pi * (((u * x) / M) + ((v * y) / N)))
    return f * (1 / (M * N))


def main():
    filename = "cameraman.tif"
    filename_small = "cameraman_small.tif"

    Path("output").mkdir(exist_ok=True)


    ##
    ## Exercise 1
    ##
    f = imread(filename)

    # 1)
    F = np.fft.fft2(f)
    real = np.fft.ifft2(F.real).astype(np.uint8)
    imsave("output/01_1_reconstructed_real.png", real)

    # 2)
    imaginary = np.fft.ifft2(F.imag * 1j)
    imaginary = np.clip(imaginary.real, 0, 255).astype(np.uint8)
    imsave("output/01_2_reconstructed_imaginary.png", imaginary)


    ##
    ## Exercise 2
    ##
    f = imread(filename_small)

    # a)
    F = np.fft.fft2(f)
    reconstructed = idft(F)
    imsave("output/02_1_np_manual.png", reconstructed.real)

    # b)
    F = dft(f)
    reconstructed = np.fft.ifft2(F)
    imsave("output/02_2_manual_np.png", reconstructed.real)

    # c)
    F = dft(f)
    reconstructed = idft(F)
    imsave("output/02_3_manual_manual.png", reconstructed.real)


    ####
    #### Exercise 3
    ####
    f = imread(filename)

    # a)
    F = np.fft.fft2(f)
    F[0, 0] = 0
    reconstructed = np.clip(np.fft.ifft2(F).real, 0, 255).astype(np.uint8)
    imsave("output/03_1_reconstructed_dc_zeroed.png", reconstructed.real)

    # b)
    imsave("output/03_2_mean_subtracted.png", np.clip(f - f.mean(), 0, 255).astype(np.uint8))


    ####
    #### Exercise 4
    ####

    # a)
    F = np.fft.fft2(f)
    spectrum = np.abs(F)
    spectrum = np.log(spectrum + 1)
    imsave("output/04_1_spectrum.png", spectrum)

    shifted_spectrum = np.fft.fftshift(F)
    shifted_spectrum = np.abs(shifted_spectrum)
    shifted_spectrum = np.log(shifted_spectrum + 1)
    imsave("output/04_2_shifted_spectrum.png", shifted_spectrum)

    shifted_spectrum = np.fft.fftshift(F)
    reconstructed = np.fft.ifft2(shifted_spectrum).real
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    imsave("output/04_3_reconstructed.png", reconstructed)

    # b)
    g = np.empty(f.shape)

    for x in range(f.shape[0]):
        for y in range(f.shape[1]):
            g[x, y] = (-1) ** (x + y) * f[x, y]

    g = np.clip(g, 0, 255).astype(np.uint8)
    imsave("output/04_4_g.png", g)

    # c)
    shifted_space = np.fft.fftshift(f)
    imsave("output/04_4_shifted_space.png", shifted_space)

    spectrum = np.fft.fft2(shifted_space)
    spectrum = np.abs(spectrum)
    spectrum = np.log(spectrum + 1)
    imsave("output/04_5_shifted_space_spectrum.png", spectrum)


    ####
    #### Exercise 4
    ####

    # a)
    F = np.fft.fft2(f)
    F = np.abs(F)
    reconstructed = np.fft.ifft2(F)
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    imsave("output/05_1_reconstructed.png", reconstructed)

    # b)
    F = np.fft.fft2(f)
    F = np.exp(1j * np.angle(F))
    reconstructed = np.fft.ifft2(F).real
    reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())
    imsave("output/05_2_reconstructed.png", reconstructed)


if __name__ == "__main__":
    main()




####
#### Exercise 1
####

# image = imread("cameraman.tif")

# image_dft = np.fft.fft2(image)
# real = np.fft.ifft2(image_dft.real).astype(np.uint8)
# imsave("output/01_1_reconstructed_real.png", real)

# imaginary = np.fft.ifft2(image_dft.imag * 1j)
# imaginary = np.clip(imaginary.real, 0, 255).astype(np.uint8)
# imsave("output/01_2_reconstructed_imaginary.png", imaginary)

####
#### Exercise 2
####

# image = imread("cameraman_small.tif")

# # a)
# image_dft = np.fft.fft2(image)
# reconstructed = idft(image_dft)
# imsave("output/02_1_np_manual.png", reconstructed.real)

# # b)
# image_dft = dft(image)
# reconstructed = np.fft.ifft2(image_dft)
# imsave("output/02_2_manual_np.png", reconstructed.real)

# # c)
# image_dft = dft(image)
# reconstructed = idft(image_dft)
# imsave("output/02_3_manual_manual.png", reconstructed.real)

####
#### Exercise 3
####

# image = imread("cameraman.tif")

# image_dft = np.fft.fft2(image)
# image_dft[0, 0] = 0
# reconstructed = np.clip(np.fft.ifft2(image_dft).real, 0, 255).astype(np.uint8)
# imsave("output/03_1_reconstructed_dc_zeroed.png", reconstructed.real)

# imsave("output/03_2_mean_subtracted.png", np.clip(image - image.mean(), 0, 255).astype(np.uint8))

####
#### Exercise 4
####

# # a)
# image = imread("cameraman.tif")

# image_dft = np.fft.fft2(image)

# spectrum = np.abs(image_dft)
# spectrum = np.log(spectrum + 1)
# imsave("output/04_1_spectrum.png", spectrum)

# shifted_spectrum = np.fft.fftshift(image_dft)
# shifted_spectrum = np.abs(shifted_spectrum)
# shifted_spectrum = np.log(shifted_spectrum + 1)
# imsave("output/04_2_shifted_spectrum.png", shifted_spectrum)

# shifted_spectrum = np.fft.fftshift(image_dft)
# reconstructed = np.fft.ifft2(shifted_spectrum).real
# reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
# imsave("output/04_3_reconstructed.png", reconstructed)

# # # b)
# image = imread("cameraman.tif")

# g = np.empty(image.shape)

# for x in range(image.shape[0]):
#     for y in range(image.shape[1]):
#         g[x, y] = (-1) ** (x + y) * image[x, y]

# g = np.clip(g, 0, 255).astype(np.uint8)
# imsave("output/04_4_g.png", g)

# # c)
# image = imread("cameraman.tif")

# shifted_space = np.fft.fftshift(image)
# imsave("output/04_4_shifted_space.png", shifted_space)

# spectrum - np.fft.fft2(shifted_space)
# spectrum = np.abs(spectrum)
# spectrum = np.log(spectrum + 1)
# imsave("output/04_5_shifted_space_spectrum.png", spectrum)


####
#### Exercise 5
####

# # a)
# image = imread("cameraman.tif")
# image_dft = np.fft.fft2(image)

# image_dft = np.abs(image_dft)
# reconstructed = np.fft.ifft2(image_dft)
# reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
# imsave("output/05_1_reconstructed.png", reconstructed)


# # b)

# image = imread("cameraman.tif")
# image_dft = np.fft.fft2(image)

# image_dft = np.exp(1j * np.angle(image_dft))
# reconstructed = np.fft.ifft2(image_dft).real
# reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())
# imsave("output/05_2_reconstructed.png", reconstructed)
