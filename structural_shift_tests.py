"""
Structural-shift significance tests for efficient-frontier analysis.

Tests implemented:
1) Block-bootstrap test for tangency Sharpe-ratio difference (post - pre)
2) Reduced spanning/alpha proxy in post period using pre-period basis portfolios:
   - pre tangency
   - pre GMV
   - pre equal-weight

Usage:
    venv/bin/python structural_shift_tests.py
    venv/bin/python structural_shift_tests.py --frequency weekly --bootstraps 3000
"""

import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from utils import ANNUALIZATION_FACTORS, compute_tangency_portfolio


@dataclass
class TestResults:
    frequency: str
    t_pre: int
    t_post: int
    n_assets: int
    rf_pre: float
    rf_post: float
    sr_pre_tangency: float
    sr_post_tangency: float
    sr_diff_post_minus_pre: float
    sr_diff_ci_low: float
    sr_diff_ci_high: float
    sr_diff_p_one_sided: float
    sr_diff_p_two_sided: float
    alpha_daily: float
    alpha_annualized: float
    alpha_ci_low_daily: float
    alpha_ci_high_daily: float
    alpha_p_one_sided: float
    sr_best_pre_basis_in_post: float
    sr_gap_full_minus_basis: float
    sr_gap_ci_low: float
    sr_gap_ci_high: float
    sr_gap_p_one_sided: float


def load_returns_csv(path: str) -> np.ndarray:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        rows = [row for row in reader if row]
    return np.array([[float(x) for x in row[1:]] for row in rows], dtype=float)


def sharpe_annualized(returns: np.ndarray, rf_annual: float, periods_per_year: int) -> float:
    rf_period = rf_annual / periods_per_year
    excess = returns - rf_period
    sigma = excess.std(ddof=1)
    if sigma <= 0:
        return np.nan
    return float(np.sqrt(periods_per_year) * excess.mean() / sigma)


def block_bootstrap_indices(length: int, block_size: int) -> np.ndarray:
    out = []
    while len(out) < length:
        start = np.random.randint(0, length)
        out.extend((start + i) % length for i in range(block_size))
    return np.array(out[:length], dtype=int)


def compute_gmv_weights(returns: np.ndarray) -> np.ndarray:
    cov = np.cov(returns, rowvar=False, ddof=1)
    n = returns.shape[1]
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def portfolio_variance(w):
        return float(w @ cov @ w)

    res = minimize(
        portfolio_variance,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"GMV optimization failed: {res.message}")
    return res.x


def optimize_tangency_on_series(
    basis_returns: np.ndarray, rf_annual: float, periods_per_year: int
) -> np.ndarray:
    mu = basis_returns.mean(axis=0)
    cov = np.cov(basis_returns, rowvar=False, ddof=1)
    n = basis_returns.shape[1]
    rf_period = rf_annual / periods_per_year

    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def neg_sharpe(w):
        m = float(w @ mu)
        v = float(np.sqrt(max(w @ cov @ w, 1e-18)))
        return -((m - rf_period) / v)

    res = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-10},
    )
    if not res.success:
        raise RuntimeError(f"Basis tangency optimization failed: {res.message}")
    return res.x


def run_tests(
    frequency: str,
    bootstraps: int,
    block_size: int,
    rf_pre: float,
    rf_post: float,
) -> TestResults:
    periods = ANNUALIZATION_FACTORS[frequency]
    pre_path = f"Data/{frequency}/pre-covid/returns_pre_{frequency}.csv"
    post_path = f"Data/{frequency}/post-covid/returns_post_{frequency}.csv"

    pre_returns = load_returns_csv(pre_path)
    post_returns = load_returns_csv(post_path)

    pre_mu_ann = pre_returns.mean(axis=0) * periods
    pre_cov_ann = np.cov(pre_returns, rowvar=False, ddof=1) * periods
    post_mu_ann = post_returns.mean(axis=0) * periods
    post_cov_ann = np.cov(post_returns, rowvar=False, ddof=1) * periods

    w_pre_tan = compute_tangency_portfolio(pre_mu_ann, pre_cov_ann, rf_pre)["weights"]
    w_post_tan = compute_tangency_portfolio(post_mu_ann, post_cov_ann, rf_post)["weights"]
    w_pre_gmv = compute_gmv_weights(pre_returns)
    w_pre_eq = np.ones(pre_returns.shape[1]) / pre_returns.shape[1]

    r_pre_tan = pre_returns @ w_pre_tan
    r_post_tan = post_returns @ w_post_tan

    sr_pre = sharpe_annualized(r_pre_tan, rf_pre, periods)
    sr_post = sharpe_annualized(r_post_tan, rf_post, periods)
    sr_diff = sr_post - sr_pre

    boot_sr_diff = np.empty(bootstraps, dtype=float)
    for b in range(bootstraps):
        ipre = block_bootstrap_indices(len(r_pre_tan), block_size)
        ipost = block_bootstrap_indices(len(r_post_tan), block_size)
        boot_sr_diff[b] = (
            sharpe_annualized(r_post_tan[ipost], rf_post, periods)
            - sharpe_annualized(r_pre_tan[ipre], rf_pre, periods)
        )
    sr_ci_low, sr_ci_high = np.percentile(boot_sr_diff, [2.5, 97.5])
    p_one_sr = float((boot_sr_diff <= 0).mean())
    p_two_sr = float(2 * min((boot_sr_diff <= 0).mean(), (boot_sr_diff >= 0).mean()))

    basis_post = np.column_stack(
        [
            post_returns @ w_pre_tan,
            post_returns @ w_pre_gmv,
            post_returns @ w_pre_eq,
        ]
    )
    y_full = r_post_tan
    rf_period_post = rf_post / periods
    y_excess = y_full - rf_period_post
    x_excess = basis_post - rf_period_post
    x_reg = np.column_stack([np.ones(len(y_excess)), x_excess])

    beta = np.linalg.lstsq(x_reg, y_excess, rcond=None)[0]
    alpha_daily = float(beta[0])
    alpha_annualized = alpha_daily * periods

    w_basis = optimize_tangency_on_series(basis_post, rf_post, periods)
    r_basis_best = basis_post @ w_basis
    sr_basis_best = sharpe_annualized(r_basis_best, rf_post, periods)
    sr_gap = sr_post - sr_basis_best

    boot_alpha = np.empty(bootstraps, dtype=float)
    boot_gap = np.empty(bootstraps, dtype=float)
    for b in range(bootstraps):
        idx = block_bootstrap_indices(len(y_full), block_size)
        xb = x_reg[idx]
        yb = y_excess[idx]
        bb = np.linalg.lstsq(xb, yb, rcond=None)[0]
        boot_alpha[b] = bb[0]
        boot_gap[b] = (
            sharpe_annualized(y_full[idx], rf_post, periods)
            - sharpe_annualized(r_basis_best[idx], rf_post, periods)
        )
    alpha_ci_low, alpha_ci_high = np.percentile(boot_alpha, [2.5, 97.5])
    p_one_alpha = float((boot_alpha <= 0).mean())
    gap_ci_low, gap_ci_high = np.percentile(boot_gap, [2.5, 97.5])
    p_one_gap = float((boot_gap <= 0).mean())

    return TestResults(
        frequency=frequency,
        t_pre=pre_returns.shape[0],
        t_post=post_returns.shape[0],
        n_assets=pre_returns.shape[1],
        rf_pre=rf_pre,
        rf_post=rf_post,
        sr_pre_tangency=sr_pre,
        sr_post_tangency=sr_post,
        sr_diff_post_minus_pre=sr_diff,
        sr_diff_ci_low=float(sr_ci_low),
        sr_diff_ci_high=float(sr_ci_high),
        sr_diff_p_one_sided=p_one_sr,
        sr_diff_p_two_sided=p_two_sr,
        alpha_daily=alpha_daily,
        alpha_annualized=alpha_annualized,
        alpha_ci_low_daily=float(alpha_ci_low),
        alpha_ci_high_daily=float(alpha_ci_high),
        alpha_p_one_sided=p_one_alpha,
        sr_best_pre_basis_in_post=sr_basis_best,
        sr_gap_full_minus_basis=sr_gap,
        sr_gap_ci_low=float(gap_ci_low),
        sr_gap_ci_high=float(gap_ci_high),
        sr_gap_p_one_sided=p_one_gap,
    )


def save_results(results: TestResults) -> str:
    os.makedirs("Figures", exist_ok=True)
    out_path = f"Figures/structural_shift_tests_{results.frequency}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in results.__dict__.items():
            writer.writerow([key, value])
    return out_path


def print_summary(results: TestResults, out_path: str) -> None:
    print("=" * 72)
    print("STRUCTURAL SHIFT TESTS")
    print("=" * 72)
    print(
        f"Frequency: {results.frequency} | T_pre={results.t_pre}, "
        f"T_post={results.t_post}, N={results.n_assets}"
    )
    print()
    print("[1] Block-bootstrap Sharpe difference (post tangency - pre tangency)")
    print(f"  SR_pre_tangency   = {results.sr_pre_tangency:.4f}")
    print(f"  SR_post_tangency  = {results.sr_post_tangency:.4f}")
    print(f"  Difference        = {results.sr_diff_post_minus_pre:.4f}")
    print(
        "  95% CI            = "
        f"[{results.sr_diff_ci_low:.4f}, {results.sr_diff_ci_high:.4f}]"
    )
    print(
        "  p-value (one-sided, H1: post>pre) = "
        f"{results.sr_diff_p_one_sided:.4f}"
    )
    print(f"  p-value (two-sided)               = {results.sr_diff_p_two_sided:.4f}")
    print()
    print("[2] Reduced spanning/alpha proxy in post period")
    print("  Basis portfolios from pre period: tangency, GMV, equal-weight")
    print(f"  Alpha (daily/period)  = {results.alpha_daily:.6f}")
    print(f"  Alpha (annualized)    = {results.alpha_annualized:.2%}")
    print(
        "  Alpha 95% CI          = "
        f"[{results.alpha_ci_low_daily:.6f}, {results.alpha_ci_high_daily:.6f}]"
    )
    print(
        "  p-value (one-sided, H1: alpha>0) = "
        f"{results.alpha_p_one_sided:.4f}"
    )
    print(f"  SR_post_full_tangency = {results.sr_post_tangency:.4f}")
    print(f"  SR_best_pre_basis     = {results.sr_best_pre_basis_in_post:.4f}")
    print(f"  SR gap (full-basis)   = {results.sr_gap_full_minus_basis:.4f}")
    print(
        "  SR gap 95% CI         = "
        f"[{results.sr_gap_ci_low:.4f}, {results.sr_gap_ci_high:.4f}]"
    )
    print(
        "  p-value (one-sided, H1: gap>0)   = "
        f"{results.sr_gap_p_one_sided:.4f}"
    )
    print()
    print(f"Saved detailed metrics to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run structural-shift significance tests.")
    parser.add_argument(
        "--frequency",
        choices=sorted(ANNUALIZATION_FACTORS.keys()),
        default="daily",
        help="Return frequency to test (default: daily)",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=1500,
        help="Number of bootstrap replications (default: 1500)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=20,
        help="Moving block size for bootstrap (default: 20)",
    )
    parser.add_argument(
        "--rf-pre",
        type=float,
        default=0.013995071193866374,
        help="Annualized risk-free rate for pre period (default from existing summary)",
    )
    parser.add_argument(
        "--rf-post",
        type=float,
        default=0.02779529282977559,
        help="Annualized risk-free rate for post period (default from existing summary)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    results = run_tests(
        frequency=args.frequency,
        bootstraps=args.bootstraps,
        block_size=args.block_size,
        rf_pre=args.rf_pre,
        rf_post=args.rf_post,
    )
    out_path = save_results(results)
    print_summary(results, out_path)


if __name__ == "__main__":
    main()
