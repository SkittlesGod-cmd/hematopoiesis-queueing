"""Distribution fitting: MLE fitting of gamma and exponential distributions.

Fits residence time distributions to gamma and exponential models using
maximum likelihood estimation (scipy.stats). Provides AIC/BIC for model
comparison and likelihood ratio test for nested model comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class FitResult:
    """Result of fitting a distribution to data.

    Attributes
    ----------
    distribution : str
        Name of the fitted distribution ('gamma' or 'exponential').
    params : dict[str, float]
        Fitted parameters. For gamma: {'shape': k, 'loc': loc, 'scale': theta}.
        For exponential: {'loc': loc, 'scale': theta}.
    n_params : int
        Number of free parameters (used for AIC/BIC).
    loglik : float
        Log-likelihood at MLE.
    aic : float
        Akaike Information Criterion: 2*k - 2*loglik.
    bic : float
        Bayesian Information Criterion: k*ln(n) - 2*loglik.
    n_samples : int
        Number of data points used in fitting.
    mean : float
        Mean of the fitted distribution (hours).
    variance : float
        Variance of the fitted distribution.
    """

    distribution: str
    params: dict[str, float]
    n_params: int
    loglik: float
    aic: float
    bic: float
    n_samples: int
    mean: float
    variance: float


def fit_gamma(data: np.ndarray, floc: float = 0.0) -> FitResult:
    """Fit a gamma distribution to residence time data via MLE.

    Parameters
    ----------
    data : ndarray
        Positive residence times in hours. Must have len >= 3.
    floc : float, default 0.0
        Fixed location parameter. Residence times are non-negative
        with natural origin at 0.

    Returns
    -------
    FitResult
        Fitted gamma distribution parameters and diagnostics.

    Notes
    -----
    Gamma parameterization: shape k, scale theta.
    Mean = k * theta, Variance = k * theta^2.
    When k=1, gamma reduces to exponential (the nested model).
    """
    data = np.asarray(data, dtype=np.float64)
    _validate_data(data)

    # MLE fit with fixed location
    shape, loc, scale = stats.gamma.fit(data, floc=floc)
    n = len(data)

    # Log-likelihood
    loglik = float(np.sum(stats.gamma.logpdf(data, a=shape, loc=loc, scale=scale)))

    # Number of free parameters: shape + scale (loc is fixed)
    n_params = 2
    aic = 2 * n_params - 2 * loglik
    bic = n_params * np.log(n) - 2 * loglik

    mean = shape * scale
    variance = shape * scale**2

    return FitResult(
        distribution="gamma",
        params={"shape": float(shape), "loc": float(loc), "scale": float(scale)},
        n_params=n_params,
        loglik=loglik,
        aic=float(aic),
        bic=float(bic),
        n_samples=n,
        mean=float(mean),
        variance=float(variance),
    )


def fit_exponential(data: np.ndarray, floc: float = 0.0) -> FitResult:
    """Fit an exponential distribution to residence time data via MLE.

    Parameters
    ----------
    data : ndarray
        Positive residence times in hours. Must have len >= 2.
    floc : float, default 0.0
        Fixed location parameter.

    Returns
    -------
    FitResult
        Fitted exponential distribution parameters and diagnostics.

    Notes
    -----
    Exponential is gamma with shape=1.
    Mean = scale (= 1/rate), Variance = scale^2.
    """
    data = np.asarray(data, dtype=np.float64)
    _validate_data(data)

    # MLE fit with fixed location
    loc, scale = stats.expon.fit(data, floc=floc)
    n = len(data)

    # Log-likelihood
    loglik = float(np.sum(stats.expon.logpdf(data, loc=loc, scale=scale)))

    # Number of free parameters: scale only (loc is fixed)
    n_params = 1
    aic = 2 * n_params - 2 * loglik
    bic = n_params * np.log(n) - 2 * loglik

    mean = scale
    variance = scale**2

    return FitResult(
        distribution="exponential",
        params={"loc": float(loc), "scale": float(scale)},
        n_params=n_params,
        loglik=loglik,
        aic=float(aic),
        bic=float(bic),
        n_samples=n,
        mean=float(mean),
        variance=float(variance),
    )


def likelihood_ratio_test(
    fit_null: FitResult,
    fit_alt: FitResult,
) -> tuple[float, float]:
    """Likelihood ratio test comparing nested models.

    Tests H0: exponential (null, restricted) vs H1: gamma (alternative, general).
    Gamma nests exponential at shape=1.

    Parameters
    ----------
    fit_null : FitResult
        Null model fit (exponential, fewer parameters).
    fit_alt : FitResult
        Alternative model fit (gamma, more parameters).

    Returns
    -------
    tuple[float, float]
        (test_statistic, p_value). Test statistic is -2*(loglik_null - loglik_alt).
        P-value from chi-squared distribution with df = difference in parameters.

    Notes
    -----
    Valid when models are nested and sample sizes are the same.
    The test statistic follows chi-squared under H0 (Wilks' theorem).
    """
    assert fit_null.n_samples == fit_alt.n_samples, (
        "Both fits must use the same data (same n_samples)."
    )
    assert fit_null.n_params < fit_alt.n_params, (
        "Null model must have fewer parameters than alternative."
    )

    lr_stat = -2.0 * (fit_null.loglik - fit_alt.loglik)
    # Clamp to 0 (can be slightly negative due to numerical issues)
    lr_stat = max(0.0, lr_stat)

    df = fit_alt.n_params - fit_null.n_params
    p_value = float(stats.chi2.sf(lr_stat, df=df))

    return float(lr_stat), p_value


# ── Private helpers ───────────────────────────────────────────────────────────


def _validate_data(data: np.ndarray) -> None:
    """Validate input data for distribution fitting."""
    if len(data) < 2:
        raise ValueError(f"Need at least 2 data points for fitting, got {len(data)}.")
    if np.any(data <= 0):
        raise ValueError("All residence times must be strictly positive.")
    if np.any(~np.isfinite(data)):
        raise ValueError("Data contains non-finite values (inf or nan).")
