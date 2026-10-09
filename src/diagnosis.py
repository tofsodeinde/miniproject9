# Diagnosis: split SPE and the ridge score into one term per channel (the "contributions"),
# average them over the alarmed faulty samples, and write results/contributions.csv
# (top five channels per fault and detector). Run from src/, like the other scripts.

# Import Functions
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from preprocessing import * # tep_fault_free, tep_faulty and the shared preprocessing functions

# Create global variables
FAULT = "faultNumber"
SAMPLE = "sample"
lst_channels = channels(tep_fault_free)
VAR_KEPT = 0.90 # PCA keeps the smallest k reaching 90 % of the variance
ONSET = 20      # samples 1-20 are normal, the fault is in from sample 21
TOP = 5         # channels kept per fault and detector

# What each channel is (teprob.f / Downs and Vogel 1993), for reading the contributions
NAMES = {
    "xmeas_1": "A feed (stream 1)", "xmeas_2": "D feed (stream 2)", "xmeas_3": "E feed (stream 3)",
    "xmeas_4": "A and C feed (stream 4)", "xmeas_5": "recycle flow", "xmeas_6": "reactor feed rate",
    "xmeas_7": "reactor pressure", "xmeas_8": "reactor level", "xmeas_9": "reactor temperature",
    "xmeas_10": "purge rate", "xmeas_11": "separator temperature", "xmeas_12": "separator level",
    "xmeas_13": "separator pressure", "xmeas_14": "separator underflow", "xmeas_15": "stripper level",
    "xmeas_16": "stripper pressure", "xmeas_17": "stripper underflow",
    "xmeas_18": "stripper temperature", "xmeas_19": "stripper steam flow",
    "xmeas_20": "compressor work", "xmeas_21": "reactor cooling water outlet T",
    "xmeas_22": "separator cooling water outlet T",
    **{f"xmeas_{22 + i}": f"reactor feed {c}" for i, c in enumerate("ABCDEF", 1)},
    **{f"xmeas_{28 + i}": f"purge {c}" for i, c in enumerate("ABCDEFGH", 1)},
    **{f"xmeas_{36 + i}": f"product {c}" for i, c in enumerate("DEFGH", 1)},
    "xmv_1": "D feed flow valve", "xmv_2": "E feed flow valve", "xmv_3": "A feed flow valve",
    "xmv_4": "A and C feed flow valve", "xmv_5": "compressor recycle valve", "xmv_6": "purge valve",
    "xmv_7": "separator liquid flow valve", "xmv_8": "stripper product flow valve",
    "xmv_9": "stripper steam valve", "xmv_10": "reactor cooling water flow",
    "xmv_11": "condenser cooling water flow",
}


# Step 1: Per-channel contributions for both detectors
def previous_two(keys): # keys sorted by fault, run, sample
    '''True where the two rows above are samples t-1 and t-2 of the same run'''
    f, r, s = (keys[c].to_numpy() for c in (FAULT, RUN, SAMPLE))
    ok = np.zeros(len(keys), bool)
    ok[2:] = (f[2:] == f[:-2]) & (r[2:] == r[:-2]) & (s[2:] == s[:-2] + 2)
    return ok

def lagged(keys, Z):
    '''Row indices that have two lags, features [z(t-1), z(t-2)] (104 columns) and target z(t)'''
    idx = np.flatnonzero(previous_two(keys))
    return idx, np.hstack([Z[idx - 1], Z[idx - 2]]), Z[idx]

def spe_contributions(Z_train, Z):
    '''SPE: contribution of channel j is (z_j - (P P^T z)_j)^2'''
    lam, vecs = np.linalg.eigh(np.cov(Z_train, rowvar=False)) # ascending order
    lam, vecs = lam[::-1], vecs[:, ::-1]                      # largest first
    k = int(np.searchsorted(np.cumsum(lam) / lam.sum(), VAR_KEPT) + 1)
    P = vecs[:, :k]
    return (Z - Z @ P @ P.T) ** 2

def ridge_contributions(train_keys, Z_train, keys, Z):
    '''Ridge: contribution of channel j is its squared standardized residual (NaN for samples 1-2)'''
    _, Xtr, ytr = lagged(train_keys, Z_train)
    model = Ridge(alpha=1.0).fit(Xtr, ytr)
    res_std_dev = (ytr - model.predict(Xtr)).std(axis=0, ddof=1)

    idx, X, y = lagged(keys, Z)
    C = np.full(Z.shape, np.nan)
    C[idx] = ((y - model.predict(X)) / res_std_dev) ** 2
    return C


# Step 2: Thresholds and alarms
def alarms(keys, score, threshold):
    '''In alarm when this sample and the two before it, in the same run, all exceed the threshold'''
    above = np.nan_to_num(score, nan=-np.inf) > threshold
    alarm = np.zeros(len(score), bool)
    alarm[2:] = above[2:] & above[1:-1] & above[:-2]
    return alarm & previous_two(keys)

def thresholds(keys, scores):
    '''99th percentile of the validation scores; compared with results/thresholds.csv if it exists'''
    val = (keys[FAULT].to_numpy() == 0) & keys[RUN].is_between(301, 400).to_numpy()
    ours = {d: float(np.quantile(s[val & ~np.isnan(s)], 0.99)) for d, s in scores.items()}
    try:
        table = pl.read_csv("../results/thresholds.csv")
    except FileNotFoundError:
        print("no results/thresholds.csv yet; using the thresholds computed here")
        return ours
    theirs = {str(d): float(t) for d, t in table.select("detector", "threshold").iter_rows()}
    for d in scores:
        if d not in theirs:
            print(f"WARNING: thresholds.csv has no detector named exactly '{d}' "
                  f"(the evidence script needs T2, SPE and ridge); found {list(theirs)}")
        else:
            print(f"threshold {d}: thresholds.csv {theirs[d]:.6g}, recomputed here {ours[d]:.6g}")
    return ours

def cross_check(keys, scores):
    '''Contributions must add up to the team's scores, if those files are in results/'''
    for d, path, col in [("SPE", "../results/scores_pca.parquet", "SPE"),
                         ("ridge", "../results/scores_ridge.parquet", "score")]:
        try:
            theirs = pl.read_parquet(path).select(FAULT, RUN, SAMPLE, col)
        except FileNotFoundError:
            print(f"no {path[3:]} yet; skipping the {d} cross-check")
            continue
        theirs = (keys.join(theirs, on=[FAULT, RUN, SAMPLE], how="left", maintain_order="left")[col]
                  .cast(pl.Float64).fill_null(np.nan).to_numpy())
        both = ~np.isnan(theirs) & ~np.isnan(scores[d])
        rel = np.abs(scores[d][both] - theirs[both]) / np.abs(theirs[both])
        print(f"{d}: sum of contributions vs {path[3:]}: max relative difference {rel.max():.1e}"
              f" over {both.sum():,} rows")


# Step 3: Average the contributions over every alarmed sample after sample 20, per fault
def top_channels(keys, contrib, scores, limit):
    fault = keys[FAULT].to_numpy()
    after_onset = (fault > 0) & (keys[SAMPLE].to_numpy() > ONSET)
    rows = []
    for d, C in contrib.items():
        alarm = alarms(keys, scores[d], limit[d])
        for f in range(1, 21):
            m = after_onset & (fault == f) & alarm
            if not m.any(): # faults 3, 9 and 15: never in alarm, nothing to average
                print(f"{d:5s} fault {f:2d}: no alarmed samples after onset")
                continue
            mean = C[m].mean(axis=0)
            for rank, j in enumerate(np.argsort(mean)[::-1][:TOP], 1):
                rows.append({"fault": f, "detector": d, "rank": rank, "channel": lst_channels[j],
                             "contribution": float(mean[j]), "n_alarmed": int(m.sum())})
    return pl.DataFrame(rows)

#---------------------------------------------------------------------------------------------------------------
def main():
    mean, std_dev = standardization(tep_fault_free, lst_channels)
    train = training_runs(tep_fault_free).sort(FAULT, RUN, SAMPLE)
    scored = pl.concat([validation_runs(tep_fault_free), tep_faulty]).sort(FAULT, RUN, SAMPLE)

    Z_train = apply_standard(train, lst_channels, mean, std_dev).select(lst_channels).to_numpy()
    Z = apply_standard(scored, lst_channels, mean, std_dev).select(lst_channels).to_numpy()
    train_keys, keys = train.select(FAULT, RUN, SAMPLE), scored.select(FAULT, RUN, SAMPLE)

    contrib = {"SPE": spe_contributions(Z_train, Z),
               "ridge": ridge_contributions(train_keys, Z_train, keys, Z)}
    scores = {d: C.sum(axis=1) for d, C in contrib.items()} # NaN where ridge has no score
    cross_check(keys, scores)
    limit = thresholds(keys, scores)

    table = top_channels(keys, contrib, scores, limit)
    table.drop("n_alarmed").write_csv("../results/contributions.csv")
    print(f"wrote {table.height} rows to results/contributions.csv\n")

    # Top channel per fault, with what it is, for the report
    top1 = (table.filter(pl.col("rank") == 1)
            .with_columns(pl.col("channel").replace_strict(NAMES).alias("what"))
            .pivot(on="detector", index="fault", values=["channel", "what", "n_alarmed"])
            .sort("fault"))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, fmt_str_lengths=40):
        print(top1)


if __name__ == "__main__":
    main()
