"""Model image generators for testing.

Each function returns a ModelImage dataclass containing a clean (noise-free)
image and all metadata needed by the test runner.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable

from test_generators import (
    crater_sharp,
    crater_linear_ramp,
    crater_sigmoid,
    gaussian_ring,
    make_radial_image,
    make_elliptical_image,
)


@dataclass
class ModelImage:
    """Clean test image + metadata for boundary detection testing."""
    name: str               # e.g. "Sharp Step (R0=40)"
    clean_image: np.ndarray
    radial_func: Callable   # r → pixel value
    rmax: float
    expected_radius: float
    xc: float
    yc: float
    params: dict            # model parameters for reporting


def make_sharp_step(shape, config):
    """Sharp step (crater) model."""
    R0 = config["crater_r0"]
    A = config["crater_a"]
    B = config["crater_b"]
    xc = config["xc"]
    yc = config["yc"]
    rmax = R0 * config["model_rmax_factor"]

    def rf(r):
        return crater_sharp(r, R0=R0, A=A, B=B)

    clean = make_radial_image(shape, xc, yc, rf)
    return ModelImage(
        name=f"Sharp Step (R0={R0:.0f})",
        clean_image=clean,
        radial_func=rf,
        rmax=rmax,
        expected_radius=R0,
        xc=xc,
        yc=yc,
        params={"R0": R0, "A": A, "B": B},
    )


def make_gaussian_ring(shape, config, sigma):
    """Gaussian ring model with given sigma width."""
    R0 = config["ring_r0"]
    xc = config["xc"]
    yc = config["yc"]
    rmax = R0 * config["model_rmax_factor"]

    def rf(r):
        return gaussian_ring(r, R0=R0, sigma=sigma, B_max=1.0)

    clean = make_radial_image(shape, xc, yc, rf)
    return ModelImage(
        name=f"Gaussian Ring (sigma={sigma:.0f})",
        clean_image=clean,
        radial_func=rf,
        rmax=rmax,
        expected_radius=R0,
        xc=xc,
        yc=yc,
        params={"R0": R0, "sigma": sigma},
    )


def make_sigmoid(shape, config, k):
    """Sigmoid transition model with given steepness k."""
    R0 = config["crater_r0"]
    A = config["crater_a"]
    B = config["crater_b"]
    xc = config["xc"]
    yc = config["yc"]
    rmax = R0 * config["model_rmax_factor"]

    def rf(r):
        return crater_sigmoid(r, R0=R0, k=k, A=A, B=B)

    clean = make_radial_image(shape, xc, yc, rf)
    return ModelImage(
        name=f"Sigmoid (k={k:.0f})",
        clean_image=clean,
        radial_func=rf,
        rmax=rmax,
        expected_radius=R0,
        xc=xc,
        yc=yc,
        params={"R0": R0, "k": k},
    )


def make_sigmoid_elliptical(shape, config, k, a, b, phi):
    """Sigmoid transition on elliptical profile."""
    xc = config["xc"]
    yc = config["yc"]
    A = config["crater_a"]
    B = config["crater_b"]

    def rf(r_norm):
        return crater_sigmoid(r_norm * a, R0=a, k=k, A=A, B=B)

    clean = make_elliptical_image(shape, xc, yc, a, b, phi, rf)
    return ModelImage(
        name=f"Sigmoid Elliptical (a={a:.0f}, b={b:.0f}, k={k:.0f})",
        clean_image=clean,
        radial_func=rf,
        rmax=a,
        expected_radius=a,
        xc=xc,
        yc=yc,
        params={"a": a, "b": b, "phi": phi, "k": k},
    )


def make_linear_ramp(shape, config, half_width=5.0):
    """Linear ramp model between R0-half_width and R0+half_width."""
    R0 = config["crater_r0"]
    R1, R2 = R0 - half_width, R0 + half_width
    xc = config["xc"]
    yc = config["yc"]
    rmax = R2 * config["model_rmax_factor"]

    def rf(r):
        return crater_linear_ramp(r, R1=R1, R2=R2, A=0.0, B=1.0)

    clean = make_radial_image(shape, xc, yc, rf)
    return ModelImage(
        name=f"Linear Ramp (hw={half_width:.0f})",
        clean_image=clean,
        radial_func=rf,
        rmax=rmax,
        expected_radius=(R1 + R2) / 2,
        xc=xc,
        yc=yc,
        params={"R1": R1, "R2": R2, "half_width": half_width},
    )


def build_model_list(config):
    """Build all model instances from config.

    Respects config['ring_sigmas'] and config['sigmoid_ks'] for parameter grids.
    """
    models = []
    shape = tuple(config["shape"])

    models.append(make_sharp_step(shape, config))

    for hw in config.get("ramp_half_widths", [5.0]):
        models.append(make_linear_ramp(shape, config, half_width=hw))

    for sigma in config.get("ring_sigmas", [3.0, 5.0, 10.0, 20.0]):
        models.append(make_gaussian_ring(shape, config, sigma=sigma))

    for k in config.get("sigmoid_ks", [1.0, 3.0, 5.0, 10.0]):
        models.append(make_sigmoid(shape, config, k=k))

    return models
