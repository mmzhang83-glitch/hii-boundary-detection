"""Toy model radial profiles and 2D image generation for testing."""

import numpy as np
from typing import Callable, Tuple


def crater_sharp(r, R0=60.0, A=0.0, B=1.0):
    """Sharp step: f(r) = A for r ≤ R0, B for r > R0."""
    return np.where(r <= R0, A, B)


def crater_linear_ramp(r, R1=55.0, R2=65.0, A=0.0, B=1.0):
    """Linear ramp: f(r) transitions linearly from A to B between R1 and R2."""
    result = np.full_like(r, A, dtype=float)
    mask_ramp = (r > R1) & (r < R2)
    result[mask_ramp] = A + (B - A) * (r[mask_ramp] - R1) / (R2 - R1)
    result[r >= R2] = B
    return result


def crater_sigmoid(r, R0=60.0, k=3.0, A=0.0, B=1.0):
    """Sigmoid transition: f(r) = A + (B-A) / (1 + exp(-(r-R0)/k))."""
    return A + (B - A) / (1.0 + np.exp(-(r - R0) / k))


def gaussian_ring(r, R0=60.0, sigma=10.0, B_max=1.0):
    """Gaussian ring: f(r) = B_max * exp(-(r-R0)^2 / (2*sigma^2))."""
    return B_max * np.exp(-((r - R0) ** 2) / (2 * sigma ** 2))


def make_radial_image(
    shape: Tuple[int, int],
    xc: float,
    yc: float,
    radial_func: Callable
) -> np.ndarray:
    """
    Build a 2D image from a radial profile function.

    Parameters:
        shape: (height, width) in pixels.
        xc, yc: Center coordinates (pixels).
        radial_func: f(r) → value, the radial profile.

    Returns:
        2D numpy array of shape `shape`.
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    return radial_func(r)


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int = None) -> np.ndarray:
    """Add Gaussian noise N(0, sigma^2) per pixel. Returns a new array."""
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(0.0, sigma, image.shape)
    return image + noise


def add_poisson_noise(image: np.ndarray, peak_count: float, seed: int = None) -> np.ndarray:
    """
    Apply Poisson noise. Scales the image so max = peak_count,
    draws from Poisson, and returns as float.
    """
    if seed is not None:
        np.random.seed(seed)
    imax = np.max(image)
    if imax <= 0:
        return image.astype(float)
    scaled = image * peak_count / imax
    noisy = np.random.poisson(np.maximum(scaled, 0))
    return noisy.astype(float)


def make_elliptical_image(
    shape: tuple,
    xc: float,
    yc: float,
    a: float,
    b: float,
    phi: float,
    radial_func: Callable,
) -> np.ndarray:
    """
    Build a 2D image where pixel values follow an elliptical radial profile.

    r_norm(x, y) = sqrt((dx'/a)^2 + (dy'/b)^2)
    where dx', dy' are coordinates rotated by -phi to align with ellipse axes.
    r_norm = 0 at center, 1 on ellipse edge, >1 outside.

    image[x, y] = radial_func(r_norm)
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    dx = x - xc
    dy = y - yc
    if phi != 0:
        cos_p = np.cos(-phi)
        sin_p = np.sin(-phi)
        dx_rot = dx * cos_p - dy * sin_p
        dy_rot = dx * sin_p + dy * cos_p
    else:
        dx_rot, dy_rot = dx, dy
    r_norm = np.sqrt((dx_rot / a) ** 2 + (dy_rot / b) ** 2)
    return radial_func(r_norm)
