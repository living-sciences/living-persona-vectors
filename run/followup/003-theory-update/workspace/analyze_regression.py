"""Analyses 1 & 4: regressions, era-invariance of depth, parametric fits.
Low power is intrinsic (5 models = 5 eras, ~7-8B all). Everything is flagged.
"""
import json, os, numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

RES = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/003-theory-update/results"
FU002 = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/002-living-update"
df = pd.read_csv(os.path.join(RES, "theory_table.csv"))
out = {}


def ols(X, y, names):
    """OLS with SE/t/p. X already includes intercept column."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj_r2 = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else float("nan")
    out_rows = {}
    if dof > 0:
        sigma2 = ss_res / dof
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * XtX_inv))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = beta / se
        p = 2 * stats.t.sf(np.abs(t), dof)
        for nm, b, s, tt, pp in zip(names, beta, se, t, p):
            out_rows[nm] = dict(coef=round(float(b), 4), se=round(float(s), 4),
                                t=round(float(tt), 3), p=round(float(pp), 4))
    else:
        for nm, b in zip(names, beta):
            out_rows[nm] = dict(coef=round(float(b), 4), se=None, t=None, p=None)
    cond = float(np.linalg.cond(X))
    return dict(coefficients=out_rows, r2=round(r2, 4), adj_r2=round(adj_r2, 4),
                n=n, dof=dof, cond_number=round(cond, 1))


# ---- design matrices ----
# alignment dummies: reference = base-RLHF
df["is_newer"] = (df["alignment"] == "newer-RLHF").astype(float)
df["is_reason"] = (df["alignment"] == "reasoning-distill").astype(float)
one = np.ones(len(df))

# 1a. Univariate slopes (continuous predictors) for both outcomes
uni = {}
for outcome in ["steer_delta", "monitor_r"]:
    uni[outcome] = {}
    for pred in ["log10_params", "era", "depth_frac"]:
        lr = stats.linregress(df[pred], df[outcome])
        uni[outcome][pred] = dict(slope=round(float(lr.slope), 4),
                                  intercept=round(float(lr.intercept), 4),
                                  r=round(float(lr.rvalue), 3),
                                  r2=round(float(lr.rvalue ** 2), 4),
                                  p=round(float(lr.pvalue), 4))
    # alignment as one-way ANOVA (3 groups)
    groups = [df[df.alignment == a][outcome].values
              for a in ["base-RLHF", "newer-RLHF", "reasoning-distill"]]
    F, p = stats.f_oneway(*groups)
    uni[outcome]["alignment_anova"] = dict(
        F=round(float(F), 3), p=round(float(p), 4),
        group_means={a: round(float(df[df.alignment == a][outcome].mean()), 3)
                     for a in ["base-RLHF", "newer-RLHF", "reasoning-distill"]})
out["univariate"] = uni

# 1b. Full multivariate model (flag collinearity via cond_number)
Xfull = np.column_stack([one, df.log10_params, df.era, df.depth_frac,
                         df.is_newer, df.is_reason])
names = ["intercept", "log10_params", "era", "depth_frac",
         "align_newer", "align_reason"]
out["multivariate"] = {
    "steer_delta": ols(Xfull, df.steer_delta, names),
    "monitor_r": ols(Xfull, df.monitor_r, names),
    "_note": "n=15 obs cluster into 5 models; era near-collinear with alignment; params near-constant. Treat as descriptive.",
}

# ---- Analysis 1 hypothesis: is selected-layer-fraction era-invariant? ----
lr = stats.linregress(df.era, df.depth_frac)
out["depth_frac_vs_era"] = dict(
    slope=round(float(lr.slope), 4), p=round(float(lr.pvalue), 4),
    r2=round(float(lr.rvalue ** 2), 4),
    mean_depth_frac=round(float(df.depth_frac.mean()), 4),
    std_depth_frac=round(float(df.depth_frac.std(ddof=1)), 4),
    min=round(float(df.depth_frac.min()), 4), max=round(float(df.depth_frac.max()), 4),
    interpretation="slope~0 & CI spans 0 => era-invariant late-layer locus")

# ---- Analysis 4: parametric summaries ----
# (a) monitoring r ~ a + b*era + c*depth_frac
Xp = np.column_stack([one, df.era, df.depth_frac])
out["parametric_monitor_r"] = ols(Xp, df.monitor_r, ["a_intercept", "b_era", "c_depth_frac"])

# (b) steering effect: logistic in coef, per-era midpoint, from C1 curve data
agg = json.load(open(os.path.join(FU002, "results/aggregate.json")))
order = ["Mistral-7B-Instruct-v0.2", "Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct",
         "DeepSeek-R1-Distill-Llama-8B", "Qwen3-8B-Nonthinking"]


def logistic(x, L, k, x0):
    return L / (1.0 + np.exp(-k * (x - x0)))


era_fits = {}
for model in order:
    era = agg["models"][model]["_meta"]["era"]
    xs, ys = [], []
    for trait, cd in agg["models"][model]["C1_curve"].items():
        for coef, val in cd["curve"].items():
            xs.append(float(coef)); ys.append(float(val))
    xs = np.array(xs); ys = np.array(ys)
    try:
        popt, _ = curve_fit(logistic, xs, ys, p0=[100, 3, 1.5],
                            maxfev=20000, bounds=([50, 0.1, 0], [110, 20, 4]))
        pred = logistic(xs, *popt)
        ss_res = float(((ys - pred) ** 2).sum())
        ss_tot = float(((ys - ys.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot
        era_fits[model] = dict(era=era, L=round(float(popt[0]), 2),
                               k_slope=round(float(popt[1]), 3),
                               x0_midpoint=round(float(popt[2]), 3),
                               r2=round(r2, 4), n_points=len(xs))
    except Exception as e:
        era_fits[model] = dict(era=era, error=str(e))
out["parametric_steer_logistic_per_era"] = era_fits

# regress midpoint x0 on era
eras = np.array([era_fits[m]["era"] for m in order])
x0s = np.array([era_fits[m]["x0_midpoint"] for m in order])
lr = stats.linregress(eras, x0s)
out["steer_midpoint_vs_era"] = dict(
    slope=round(float(lr.slope), 4), intercept=round(float(lr.intercept), 4),
    r2=round(float(lr.rvalue ** 2), 4), p=round(float(lr.pvalue), 4),
    x0_by_era={int(e): float(x) for e, x in zip(eras, x0s)})

json.dump(out, open(os.path.join(RES, "regression_results.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
