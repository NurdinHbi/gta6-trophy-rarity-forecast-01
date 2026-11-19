"""GTA VI trophy rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Reads one case at a time from the input directory (each case: every read the
house made of the game's Exophase trophy page before the case's checkpoint)
and writes, for each of the game's trophies, five quantiles (5/25/50/75/95%)
of its earn rate at the case's target, as /output/<id>.out.

Method, per trophy, all in logit space (rates are percents, clamped to
[0.01, 99.99] before the logit):

  1. Trend in the rate itself: a least-squares line of logit(rate) on
     ln(1 + hours since release), anchored at the last read, so the forecast
     is the last observed rate moved along the trophy's own recent slope.
  2. Earners-vs-tracked split: Exophase's rate is earners / players tracked,
     and the two move for different reasons (players progress through the
     game; new players join and dilute every rate). So earners are forecast
     on their own (log earners on ln(1 + hours)) and players-tracked on its
     own (one global line), and the ratio gives a second rate forecast,
     trusted only while the rate is low (its multiplicative growth has no
     saturation, so for a common trophy the trend stands alone) and never
     allowed far from the trend forecast. The median is the rate-weighted
     blend of the two forecasts in logit space, so a launch week where the
     tracked base doubles does not fool the trend.
  3. Hierarchical shrinkage: each trophy's slopes are shrunk toward its
     tier's mean slope (platinum / gold / silver / bronze), because one
     trophy's early reads are noisy and the tier's trophies move together.
     With flat early data every slope shrinks to ~0 and the forecast is
     persistence, which is the honest answer then.
  4. Coherence: the platinum (earn every trophy) can never be more common
     than the rarest other trophy, so its median is capped at the smallest
     other median.
  5. Spread: the quantile spread combines the fit residual, the slope's
     standard error stretched over the horizon, a random-walk term that
     grows with the square root of the horizon in days, and half the
     disagreement between the two forecasts above. Quantiles are
     inv_logit(median + z_tau * sigma) with z from the standard normal.

Python 3.13, standard library only, no network, deterministic: the same
case always writes the same bytes. A case that fails to parse still gets a
valid persistence forecast, never a crash, so a run always exits 0.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Shrinkage strength: a trophy's own slope counts as this many extra reads
# against its tier mean.
SHRINK_READS = 6.0
# Random-walk spread, in logit units per square-root day of horizon.
HORIZON_SIGMA = 0.5
# Floor on the total spread, in logit units: even a flat history can jump
# when the page refreshes.
SIGMA_FLOOR = 0.30
# The earners-vs-tracked split is only trusted while the rate is low: its
# multiplicative growth has no saturation, so above this rate (percent) the
# trend forecast stands alone.
COUNTS_TRUST_BELOW = 25.0
# The counts forecast may not stray this far (logit units) from the trend
# forecast: a guardrail against either model running away.
MAX_MODEL_GAP = 1.0


def clamp_rate(pct):
    """A percent into [0.01, 99.99], the range the logit can take."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit_pct(y):
    """A logit back to a percent in (0, 100)."""
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up; whole numbers come back as ints."""
    value = float(Decimal(repr(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
    return int(value) if value.is_integer() else value


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("norm_ppf needs 0 < p < 1")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - p_low:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_std, se_slope, n)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return my, 0.0, 0.0, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    se_b = resid / math.sqrt(sxx) if n > 2 else 0.0
    return a, b, resid, se_b, n


def shrink(value, toward, n, strength=SHRINK_READS):
    """Pull value toward a group mean, weighted by how much data backs it."""
    return (n * value + strength * toward) / (n + strength)


def trophy_series(case, index):
    """(hours, x, y, earners, tracked) points for one trophy, oldest first.

    x is ln(1 + hours since release); y is the logit of the rate; earners
    is the implied headcount behind the rate. Reads with a missing rate or
    a missing tracked count are skipped for that trophy.
    """
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    hours, xs, ys, es, ns = [], [], [], [], []
    for read in case["history"]:
        rate = read["rates"][index]
        tracked = read.get("players_tracked")
        if rate is None or tracked is None or tracked <= 0:
            continue
        h = (parse_time(read["slot"]) - release).total_seconds() / 3600.0
        hours.append(h)
        xs.append(math.log1p(max(h, -0.99)))
        ys.append(logit(rate))
        es.append(math.log(max(rate / 100.0 * tracked, 0.5)))
        ns.append(tracked)
    return hours, xs, ys, es, ns


def recent_window(hours, xs, ys, es, min_points=4, window_hours=24.0):
    """The last window_hours of a series, when it holds enough points.

    Early launch-night reads are flat and noisy (the page barely moves
    before its first refresh); the recent window is what the slope should
    come from. Falls back to the whole series when the window is too thin.
    """
    if not hours:
        return xs, ys, es
    cut = hours[-1] - window_hours
    keep = [k for k, h in enumerate(hours) if h >= cut]
    if len(keep) >= min_points:
        return [xs[k] for k in keep], [ys[k] for k in keep], [es[k] for k in keep]
    return xs, ys, es


def tracked_growth(case):
    """Global slope of ln(players tracked) on ln(1 + hours), anchored last."""
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    xs, ys = [], []
    for read in case["history"]:
        tracked = read.get("players_tracked")
        if tracked is None or tracked <= 0:
            continue
        hours = (parse_time(read["slot"]) - release).total_seconds() / 3600.0
        xs.append(math.log1p(max(hours, -0.99)))
        ys.append(math.log(tracked))
    if len(xs) < 2:
        return 0.0, xs[-1] if xs else 0.0, ys[-1] if ys else 0.0
    _, slope, _, _, _ = ols(xs, ys)
    return min(max(slope, 0.0), 2.0), xs[-1], ys[-1]


def forecast_trophy(fit, x_target, days, z_values):
    """Five quantiles for one trophy from its fitted pieces."""
    y_med = fit["y_med"]
    sigma = fit["sigma"]
    qs = []
    for z in z_values:
        qs.append(round4(clamp_rate(inv_logit_pct(y_med + z * sigma))))
    # Rounding can break monotonicity by a hair; enforce it.
    for k in range(1, len(qs)):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return qs


def forecast_case(case):
    levels = case.get("levels", LEVELS)
    z_values = [norm_ppf(level) for level in levels]
    history = case.get("history", [])
    if not history:
        raise ValueError("case has no history")
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target = parse_time(case["target"])
    hours_target = (target - release).total_seconds() / 3600.0
    x_target = math.log1p(max(hours_target, -0.99))
    days = max(case.get("horizon_minutes", 0) / 1440.0, 0.0)

    growth_slope, x_last_n, log_n_last = tracked_growth(case)

    # First pass: per-trophy fits, to pool the tier means.
    fits = []
    tier_slopes = {}
    tier_earner_slopes = {}
    tier_resids = {}
    for index, trophy in enumerate(case["trophies"]):
        hours, xs, ys, es, _ = trophy_series(case, index)
        xs, ys, es = recent_window(hours, xs, ys, es)
        slope = earner_slope = 0.0
        resid = se_b = 0.0
        if len(xs) >= 2:
            _, slope, resid, se_b, _ = ols(xs, ys)
            _, earner_slope, _, _, _ = ols(xs, es)
        fits.append({
            "index": index, "tier": trophy.get("tier", "bronze"),
            "xs": xs, "ys": ys, "es": es,
            "slope": slope, "earner_slope": earner_slope,
            "resid": resid, "se_b": se_b, "has_data": len(xs) > 0,
        })
        tier_slopes.setdefault(trophy.get("tier", "bronze"), []).append(slope)
        tier_earner_slopes.setdefault(trophy.get("tier", "bronze"), []).append(earner_slope)
        tier_resids.setdefault(trophy.get("tier", "bronze"), []).append(resid)
    all_slopes = [f["slope"] for f in fits]
    all_earner = [f["earner_slope"] for f in fits]
    all_resids = [f["resid"] for f in fits]
    global_slope = sum(all_slopes) / len(all_slopes) if all_slopes else 0.0
    global_earner = sum(all_earner) / len(all_earner) if all_earner else 0.0
    global_resid = sorted(all_resids)[len(all_resids) // 2] if all_resids else 0.0
    tier_mean = {t: sum(v) / len(v) for t, v in tier_slopes.items()}
    tier_earner_mean = {t: sum(v) / len(v) for t, v in tier_earner_slopes.items()}
    tier_resid_med = {}
    for t, v in tier_resids.items():
        ordered = sorted(v)
        tier_resid_med[t] = ordered[len(ordered) // 2]

    # Second pass: shrunk forecasts.
    rows = []
    medians = {}
    for fit in fits:
        xs, ys, es = fit["xs"], fit["ys"], fit["es"]
        tier = fit["tier"]
        n = len(xs)
        if n == 0:
            # No read ever carried this trophy: a 0.5% prior with a wide
            # spread, and no counts model (its earner anchor would be made up).
            y_last, e_last = logit(0.5), math.log(0.5)
            x_last = xs[-1] if xs else x_target
            n_last = math.exp(log_n_last) if log_n_last else 1.0
        else:
            y_last = ys[-1]
            e_last = es[-1]
            x_last = xs[-1]
            n_last = math.exp(log_n_last)
        slope = shrink(fit["slope"], tier_mean.get(tier, global_slope), n)
        slope = min(max(slope, -1.0), 3.0)
        earner = shrink(fit["earner_slope"], tier_earner_mean.get(tier, global_earner), n)
        earner = min(max(earner, -0.5), 2.5)
        dx = x_target - x_last
        y_trend = y_last + slope * dx
        n_hat = n_last * math.exp(growth_slope * dx)
        e_hat = math.exp(e_last + earner * dx)
        y_counts = logit(100.0 * e_hat / max(n_hat, 1.0))
        # The counts model grows multiplicatively and has no saturation, so
        # trust it only while the rate is low, and never let it stray far
        # from the trend forecast: for a common trophy the trend stands.
        y_counts = min(max(y_counts, y_trend - MAX_MODEL_GAP),
                       y_trend + MAX_MODEL_GAP)
        r_last = inv_logit_pct(y_last)
        w_counts = 0.5 * max(0.0, 1.0 - r_last / COUNTS_TRUST_BELOW)
        if not fit["has_data"]:
            w_counts = 0.0
        y_med = (1.0 - w_counts) * y_trend + w_counts * y_counts
        resid = shrink(fit["resid"], tier_resid_med.get(tier, global_resid), n)
        sigma = math.sqrt(
            SIGMA_FLOOR ** 2
            + (HORIZON_SIGMA * math.sqrt(days)) ** 2
            + resid ** 2
            + (fit["se_b"] * dx) ** 2
            + (0.5 * abs(y_trend - y_counts)) ** 2
        )
        sigma = min(max(sigma, SIGMA_FLOOR), 3.0)
        if not fit["has_data"]:
            # Nothing was ever observed: say so with a wide spread.
            sigma = max(sigma, 1.5)
        medians[fit["index"]] = y_med
        rows.append({"index": fit["index"], "id": case["trophies"][fit["index"]]["id"],
                     "tier": tier, "y_med": y_med, "sigma": sigma,
                     "has_data": fit["has_data"]})

    # Platinum can never be more common than the rarest other trophy that
    # was actually observed; a no-data prior must not drag it down.
    others = [r["y_med"] for r in rows if r["tier"] != "platinum" and r["has_data"]]
    for row in rows:
        if row["tier"] == "platinum" and others:
            row["y_med"] = min(row["y_med"], min(others))

    out_rows = []
    for row in rows:
        out_rows.append({
            "id": row["id"],
            "q": forecast_trophy(row, x_target, days, z_values),
        })
    return {"trophies": out_rows}


def fallback_forecast(case):
    """Persistence with a wide spread: always valid, never crashes."""
    levels = case.get("levels", LEVELS)
    z_values = [norm_ppf(level) for level in levels]
    days = max(case.get("horizon_minutes", 0) / 1440.0, 0.0)
    sigma = 0.5 * math.sqrt(days) if days > 0 else 0.5
    last_rates = []
    history = case.get("history", [])
    if history:
        last_rates = history[-1].get("rates", [])
    rows = []
    for index, trophy in enumerate(case["trophies"]):
        rate = last_rates[index] if index < len(last_rates) and last_rates[index] is not None else 0.5
        y = logit(rate)
        qs = [round4(clamp_rate(inv_logit_pct(y + z * sigma))) for z in z_values]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback_forecast(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
