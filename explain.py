"""SHAP explanations for the federated graph-aware models.

1. Global FedAvg graph-aware model (seed 0): mean |SHAP| over all six states' 2025 test hours,
   in normalised-load units (standard deviations of each state's own load).
2. Telangana's personalised model (global model + 5 local fine-tuning epochs): mean |SHAP| in MW,
   overall and in peak windows, plus a worked explanation of Telangana's highest-load 2025 hour.

Usage: python explain.py  (after train.py)
Writes outputs/shap_results.json, outputs/example_explanation.json, updates outputs/tg_week.json
"""
import json
import os

import numpy as np
import shap

from config import LOCAL_EPOCHS, PEAK_QUANTILE, SEEDS, TEST, TRAIN
from features import build, load_clean, split
from train import local_update

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")


class _Lin:  # minimal linear model object for shap.LinearExplainer
    def __init__(self, w, b):
        self.coef_, self.intercept_ = np.asarray(w), float(np.asarray(b).ravel()[0])


def mean_abs(sv, cols, mask=None):
    s = np.abs(sv if mask is None else sv[mask]).mean(axis=0)
    return {c: float(v) for c, v in sorted(zip(cols, s), key=lambda t: -t[1])}


def main():
    d, has_temp = load_clean()
    data, _ = build(d, has_temp)
    g = json.load(open(os.path.join(OUT, "fed_graph_model.json")))
    cols, w, b = g["features"], np.array(g["coef"]), np.array(g["intercept"])

    # 1. global model across all states
    bg = np.vstack([split(data[z]["frame"], TRAIN)[cols].values for z in data])
    Xt = np.vstack([split(data[z]["frame"], TEST)[cols].values for z in data])
    ex = shap.LinearExplainer(_Lin(w, b), bg)
    sv_global = ex.shap_values(Xt)

    # 2. Telangana personalised model, attributions in MW
    tr, te = split(data["TG"]["frame"], TRAIN), split(data["TG"]["frame"], TEST)
    rng = np.random.default_rng(SEEDS[0] + 100)
    wt, bt = local_update(w, b, tr[cols].values, tr["y"].values, LOCAL_EPOCHS, rng, SEEDS[0])
    sd, mu = data["TG"]["sd"], data["TG"]["mu"]
    ex_tg = shap.LinearExplainer(_Lin(wt, bt), tr[cols].values)
    sv_tg = ex_tg.shap_values(te[cols].values) * sd
    peak = te["load_mw"].values >= np.quantile(te["load_mw"].values, PEAK_QUANTILE)

    shap_results = {
        "global_fed_graph_sigma_units": mean_abs(sv_global, cols),
        "telangana_personalised_mw": {"all_hours": mean_abs(sv_tg, cols),
                                      "peak_windows": mean_abs(sv_tg, cols, peak)},
        "global_coefficients": dict(zip(cols, map(float, w))),
        "telangana_coefficients": dict(zip(cols, map(float, wt))),
    }
    json.dump(shap_results, open(os.path.join(OUT, "shap_results.json"), "w"), indent=2)

    # worked example: Telangana's highest-load hour in 2025
    i = int(np.argmax(te["load_mw"].values))
    base = float(ex_tg.expected_value) * sd + mu
    contrib = {c: float(v) for c, v in zip(cols, sv_tg[i])}
    pred = base + sum(contrib.values())
    example = {"zone": "TG", "ts": te.index[i].strftime("%Y-%m-%d %H:00"),
               "actual_mw": float(te["load_mw"].values[i]), "predicted_mw": pred,
               "baseline_mw": base,
               "contributions_mw": dict(sorted(contrib.items(), key=lambda t: -abs(t[1])))}
    json.dump(example, open(os.path.join(OUT, "example_explanation.json"), "w"), indent=2)

    # add personalised predictions to the Telangana peak week
    wk = json.load(open(os.path.join(OUT, "tg_week.json")))
    f = te.loc[wk["ts"][0]:wk["ts"][-1]]
    wk["fed_graph_ft"] = ((f[cols].values @ wt + bt[0]) * sd + mu).round(1).tolist()
    json.dump(wk, open(os.path.join(OUT, "tg_week.json"), "w"))

    print(json.dumps(shap_results, indent=1)[:2500])
    print(json.dumps(example, indent=1))


if __name__ == "__main__":
    main()
