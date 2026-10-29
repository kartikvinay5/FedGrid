"""Train and evaluate the FedGrid models on real ICED Southern Region data.

Models (same features, same time-based split):
  central_local   centralized, pooled data, local features only
  central_graph   centralized, pooled data, + one-hop neighbour feature
  fed_local       FedAvg over 6 state clients, local features only
  fed_graph       FedAvg over 6 state clients, + one-hop neighbour feature
  solo_graph      extra: each state trains alone on its own data (no federation)

FedAvg: sample-count-weighted averaging of SGDRegressor parameters,
ROUNDS communication rounds x LOCAL_EPOCHS local epochs per round.

Usage: python train.py   -> outputs/results.json, outputs/fed_graph_model.json, outputs/tg_week.json
"""
import json
import os
import warnings

import numpy as np
from sklearn.linear_model import SGDRegressor

from config import EXTRA_TEST, LOCAL_EPOCHS, PEAK_QUANTILE, ROUNDS, SEEDS, TEST, TRAIN, ZONES
from features import GRAPH_FEATURE, build, load_clean, split

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
ETA0, ALPHA = 0.002, 1e-5


def new_model(seed):
    return SGDRegressor(loss="squared_error", penalty="l2", alpha=ALPHA,
                        learning_rate="constant", eta0=ETA0, random_state=seed)


def local_update(w, b, X, y, epochs, rng, seed):
    m = new_model(seed)
    m.partial_fit(X[:1], y[:1])               # initialise internal state
    m.coef_, m.intercept_ = w.copy(), b.copy()
    for _ in range(epochs):
        idx = rng.permutation(len(X))
        m.partial_fit(X[idx], y[idx])
    return m.coef_.copy(), m.intercept_.copy()


def fedavg(clients, n_feat, seed):
    """clients: list of (X, y). Returns global (w, b)."""
    rng = np.random.default_rng(seed)
    w, b = np.zeros(n_feat), np.zeros(1)
    n = np.array([len(y) for _, y in clients], dtype=float)
    for _ in range(ROUNDS):
        ups = [local_update(w, b, X, y, LOCAL_EPOCHS, rng, seed) for X, y in clients]
        w = sum(ni * u[0] for ni, u in zip(n, ups)) / n.sum()
        b = sum(ni * u[1] for ni, u in zip(n, ups)) / n.sum()
    return w, b


def centralized(clients, n_feat, seed):
    X = np.vstack([c[0] for c in clients])
    y = np.concatenate([c[1] for c in clients])
    rng = np.random.default_rng(seed)
    return local_update(np.zeros(n_feat), np.zeros(1), X, y, ROUNDS * LOCAL_EPOCHS, rng, seed)


def metrics(actual, pred, peak_mask):
    e = pred - actual
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mape": float(np.mean(np.abs(e) / actual) * 100),
        "peak_mae": float(np.mean(np.abs(e[peak_mask]))),
    }


def evaluate(params, data, cols, period, per_zone_params=None):
    per, A, P, M = {}, [], [], []
    for z, info in data.items():
        f = split(info["frame"], period)
        w, b = per_zone_params[z] if per_zone_params else params
        pred = (f[cols].values @ w + b[0]) * info["sd"] + info["mu"]
        act = f["load_mw"].values
        peak = act >= np.quantile(act, PEAK_QUANTILE)
        per[z] = metrics(act, pred, peak)
        A.append(act); P.append(pred); M.append(peak)
    pooled = metrics(np.concatenate(A), np.concatenate(P), np.concatenate(M))
    return pooled, per


def main():
    d, has_temp = load_clean()
    data, local_feats = build(d, has_temp)
    graph_feats = local_feats + [GRAPH_FEATURE]
    zones = list(data)

    def clients(cols):
        return [(split(data[z]["frame"], TRAIN)[cols].values,
                 split(data[z]["frame"], TRAIN)["y"].values) for z in zones]

    specs = {
        "central_local": (local_feats, centralized),
        "central_graph": (graph_feats, centralized),
        "fed_local": (local_feats, fedavg),
        "fed_graph": (graph_feats, fedavg),
    }
    runs = {k: [] for k in list(specs) + ["fed_local_ft", "fed_graph_ft", "solo_graph"]}
    saved = None
    for seed in SEEDS:
        for name, (cols, fn) in specs.items():
            cl = clients(cols)
            w, b = fn(cl, len(cols), seed)
            test = evaluate((w, b), data, cols, TEST)
            extra = evaluate((w, b), data, cols, EXTRA_TEST)
            runs[name].append({"test": test, "extra": extra})
            if name == "fed_graph" and seed == SEEDS[0]:
                saved = {"features": cols, "coef": w.tolist(), "intercept": b.tolist()}
            if fn is fedavg:
                # personalised FL: each state fine-tunes the global model locally
                rng = np.random.default_rng(seed + 100)
                pz = {z: local_update(w, b, X, y, LOCAL_EPOCHS, rng, seed)
                      for z, (X, y) in zip(zones, cl)}
                runs[name + "_ft"].append({"test": evaluate(None, data, cols, TEST, pz),
                                           "extra": evaluate(None, data, cols, EXTRA_TEST, pz)})
        # solo: one model per state, its own data only
        cl = clients(graph_feats)
        pz = {z: centralized([cl[i]], len(graph_feats), seed) for i, z in enumerate(zones)}
        runs["solo_graph"].append({"test": evaluate(None, data, graph_feats, TEST, pz),
                                   "extra": evaluate(None, data, graph_feats, EXTRA_TEST, pz)})
        print("seed", seed, "done")

    def summarise(rs, key):
        keys = ["mae", "rmse", "mape", "peak_mae"]
        pooled = {k: {"mean": float(np.mean([r[key][0][k] for r in rs])),
                      "std": float(np.std([r[key][0][k] for r in rs]))} for k in keys}
        per = {z: {k: float(np.mean([r[key][1][z][k] for r in rs])) for k in keys} for z in zones}
        return {"pooled": pooled, "per_zone": per}

    results = {
        "dataset": "NITI Aayog ICED hourly state demand met (source: Grid-India), Southern Region",
        "zones": ZONES, "train": TRAIN, "test": TEST, "extra_test": EXTRA_TEST,
        "horizon_h": 3, "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "seeds": SEEDS,
        "temperature": has_temp,
        "rows": {z: {"train": int(len(split(data[z]["frame"], TRAIN))),
                     "test": int(len(split(data[z]["frame"], TEST))),
                     "extra": int(len(split(data[z]["frame"], EXTRA_TEST)))} for z in zones},
        "features_local": local_feats, "features_graph": graph_feats,
        "models": {k: {"test": summarise(v, "test"), "extra": summarise(v, "extra")} for k, v in runs.items()},
    }
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    json.dump(saved, open(os.path.join(OUT, "fed_graph_model.json"), "w"), indent=2)

    # Telangana: the week of its highest 2025 test-period load, actual vs both federated models
    tg = split(data["TG"]["frame"], TEST)
    peak_day = tg["load_mw"].idxmax().normalize()
    wk = tg.loc[peak_day - np.timedelta64(3, "D"): peak_day + np.timedelta64(4, "D") - np.timedelta64(1, "h")]
    wf = fedavg(clients(local_feats), len(local_feats), SEEDS[0])
    def p(params, cols):
        return ((wk[cols].values @ params[0] + params[1][0]) * data["TG"]["sd"] + data["TG"]["mu"]).round(1).tolist()
    week = {"ts": [t.strftime("%Y-%m-%d %H:00") for t in wk.index],
            "actual": wk["load_mw"].round(1).tolist(),
            "fed_graph": p((np.array(saved["coef"]), np.array(saved["intercept"])), graph_feats),
            "fed_local": p(wf, local_feats)}
    json.dump(week, open(os.path.join(OUT, "tg_week.json"), "w"))

    for k, v in results["models"].items():
        pm = v["test"]["pooled"]
        print(f"{k:14s} " + "  ".join(f"{m}={pm[m]['mean']:.3f}±{pm[m]['std']:.3f}" for m in pm))


if __name__ == "__main__":
    main()
