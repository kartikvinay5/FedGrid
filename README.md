# FedGrid

**An Explainable Federated Graph Neural Network Framework for Privacy-Preserving Short-Term Electricity Load Forecasting in Smart Grids**

Federated, graph-aware and explainable 3-hour-ahead load forecasting on real hourly demand for the six
Southern Region states of India: Telangana, Andhra Pradesh, Karnataka, Tamil Nadu, Kerala and Puducherry.
Each state is a federated client and keeps its own data. The graph links states that share a border (8 edges).

Project website: `index.html` (served by GitHub Pages, see below).

## Results (2025 test year, all six states, mean of 5 seeds)

| Model | MAE (MW) | RMSE (MW) | MAPE % | Peak-hour MAE (MW) |
|---|---|---|---|---|
| Centralized, local | 351.7 | 534.0 | 4.59 | 383.1 |
| Centralized, graph-aware | 351.2 | 533.8 | 4.59 | 384.6 |
| FedAvg, local | 358.2 | 543.6 | 4.65 | 389.6 |
| FedAvg, graph-aware | 359.3 | 544.6 | 4.66 | 390.1 |
| Each state alone | 316.6 | 487.8 | 4.21 | 340.3 |
| Personalised FedAvg, local | 317.4 | 488.0 | 4.21 | 335.6 |
| **Personalised FedAvg, graph-aware** | **316.3** | **487.1** | **4.21** | **332.8** |

- Plain FedAvg underfits because the states' load patterns differ.
- Personalised federation (5 local fine-tuning epochs) cuts RMSE by 10.5% vs plain FedAvg.
- The neighbour feature helps a little: about 0.2% RMSE and 0.8% peak-hour MAE.

## Data
- Hourly demand met: Grid-India, via [NITI Aayog ICED](https://iced.niti.gov.in/energy/electricity/distribution/national-level-consumption/load-curve), Jan 2023 to Mar 2026 (`data/raw/`).
- Hourly temperature: [Open-Meteo historical weather](https://open-meteo.com/en/docs/historical-weather-api) at each state's main load centre (`data/weather.csv`).
- Holidays: the `holidays` Python package, state calendars.
- `data/clean.csv`: merged, cleaned file. 413 days that the source copies from other periods are blank.

## Run
```
pip install pandas numpy scikit-learn shap holidays openpyxl
python data_prep.py     # data/clean.csv, outputs/data_quality.json
python train.py         # outputs/results.json, fed_graph_model.json, tg_week.json (~2 min)
python explain.py       # outputs/shap_results.json, example_explanation.json
python build_site.py    # index.html (root) and site/index.html
```

## Repository layout
| Path | Contents |
|---|---|
| `config.py` | States, border graph, periods, rounds, seeds |
| `data_prep.py` | Merge and clean ICED files, add holidays and temperature |
| `features.py` | Per-state normalisation, lag, calendar and neighbour features |
| `train.py` | Centralized, FedAvg, personalised and per-state models; metrics |
| `explain.py` | SHAP attributions and worked example |
| `build_site.py` | Builds the website from `site/` templates and `outputs/` |
| `index.html` | The website |
| `docs/FedGrid_Website_Guide.pdf` | How each part of the website and pipeline works |

## Publish the website (GitHub Pages)
Settings → Pages → Source: *Deploy from a branch* → Branch: `main`, folder `/ (root)` → Save.
The site appears at `https://kartikvinay5.github.io/FedGrid/` within a minute or two.

## Team
B. Bethi Ikshitha, G. Srivardhan Rao, B. Bala Karthik Vinay. Department of CSE, Koneru Lakshmaiah Education Foundation, Hyderabad.

Check ICED and Open-Meteo terms before reusing the data elsewhere.
