import timeit
from pathlib import Path

import numpy as np
from skimage.io import imread, imsave

from dft import dft, idft


def fft(f):
    N = len(f)

    if N == 1:
        return f

    f_even = fft(f[::2])
    f_odd = fft(f[1::2])
    w = np.exp(-2j * np.pi * np.arange(N) / N)

    F = np.concatenate([
        f_even + w[:N//2] * f_odd,
        f_even + w[N//2:] * f_odd
    ])

    return F


def ifft(F):
    N = len(F)

    if N == 1:
        return F

    F_even = ifft(F[::2])
    F_odd = ifft(F[1::2])
    w = np.exp(2j * np.pi * np.arange(N) / N)

    f = np.concatenate([
        F_even + w[:N//2] * F_odd,
        F_even + w[N//2:] * F_odd
    ])

    return f


def fft2(f):
    rows = []
    F = []

    for row in f:
        rows.append(fft(row))

    rows = np.stack(rows, axis=1)

    for column in rows:
        F.append(fft(column))
    F = np.stack(F, axis=1)

    return F


def ifft2(F):
    rows = []
    f = []

    for row in F:
        rows.append(ifft(row))

    rows = np.stack(rows, axis=1)

    for column in rows:
        f.append(ifft(column))
    f = np.stack(f, axis=1)

    return f.real


def composite(original, compressed, output_path):
    output_path = Path(output_path)
    composite_image = str(output_path.parent.joinpath(f"{output_path.stem}_composite.jpg"))
    diff_image = str(output_path.parent.joinpath(f"{output_path.stem}_diff.jpg"))

    original = imread(original)
    compressed = imread(compressed)
    width = original.shape[1]

    compressed[:, :width//2] = original[:, :width//2].copy()
    imsave(composite_image, compressed)
    compressed_reloaded = imread(composite_image)

    diff = np.abs(compressed - compressed_reloaded)
    imsave(diff_image, diff)


def main():
    filename = "cameraman.tif"
    filename_small = "cameraman_small.tif"

    Path("output").mkdir(exist_ok=True)

    ##
    ## Exercise 1
    ##

    # b)
    f = imread(filename, as_gray=True)

    F = fft2(f)
    shifted_spectrum = np.fft.fftshift(F)
    shifted_spectrum = np.abs(shifted_spectrum)
    shifted_spectrum = np.log(shifted_spectrum + 1)
    imsave("output/shifted_spectrum.png", shifted_spectrum)

    F_numpy = np.fft.fft2(f)
    shifted_spectrum = np.fft.fftshift(F_numpy)
    shifted_spectrum = np.abs(shifted_spectrum)
    shifted_spectrum = np.log(shifted_spectrum + 1)
    imsave("output/shifted_spectrum_numpy.png", shifted_spectrum)

    f = ifft2(F)
    imsave("output/reconstructed.png", f)

    f_numpy = np.fft.ifft2(F_numpy).real
    imsave("output/reconstructed_numpy.png", f_numpy)


    # c)
    f = imread(filename_small, as_gray=True)

    n = 10
    fft_timer = timeit.Timer(lambda: fft2(f))
    fft_times = fft_timer.repeat(repeat=n, number=1)

    ifft_timer = timeit.Timer(lambda: ifft2(f))
    ifft_times = ifft_timer.repeat(repeat=n, number=1)

    dft_timer = timeit.Timer(lambda: dft(f))
    dft_times = dft_timer.repeat(repeat=n, number=1)

    idft_timer = timeit.Timer(lambda: idft(f))
    idft_times = idft_timer.repeat(repeat=n, number=1)

    print(f"Execution time with {n} repetitions:")
    print(f" - FFT: {min(fft_times)} seconds")
    print(f" - IFFT: {min(ifft_times)} seconds")
    print(f" - DFT: {min(dft_times)} seconds")
    print(f" - IDFT: {min(idft_times)} seconds")


    ##
    ## Exercise 2
    ##
    composite("error_level_analysis/IMG_2960 - original.jpg", "error_level_analysis/IMG_2960 - 90.jpg", "output/IMG_2960_90.jpg")
    composite("error_level_analysis/IMG_2960 - original.jpg", "error_level_analysis/IMG_2960 - 70.jpg", "output/IMG_2960_70.jpg")
    composite("error_level_analysis/IMG_2960 - original.jpg", "error_level_analysis/IMG_2960 - 50.jpg", "output/IMG_2960_50.jpg")

    composite("error_level_analysis/IMG_7867 - original.jpg", "error_level_analysis/IMG_7867 - 90.jpg", "output/IMG_7867_90.jpg")
    composite("error_level_analysis/IMG_7867 - original.jpg", "error_level_analysis/IMG_7867 - 70.jpg", "output/IMG_7867_70.jpg")
    composite("error_level_analysis/IMG_7867 - original.jpg", "error_level_analysis/IMG_7867 - 50.jpg", "output/IMG_7867_50.jpg")


if __name__ == "__main__":
    main()
