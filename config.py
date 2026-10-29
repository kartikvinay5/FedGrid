"""Shared configuration for the FedGrid real-data pipeline (Southern Region, India)."""

# Grid zones = the six Southern Region states/UTs. Codes are ISO-style short names.
ZONES = {
    "TG": "Telangana",
    "AP": "Andhra Pradesh",
    "KA": "Karnataka",
    "TN": "Tamil Nadu",
    "KL": "Kerala",
    "PY": "Puducherry",
}

# Adjacency graph: states that share a land border (Puducherry linked to Tamil Nadu,
# which surrounds its main district). 6 nodes, 8 edges.
EDGES = [
    ("TG", "AP"), ("TG", "KA"), ("AP", "KA"), ("AP", "TN"),
    ("KA", "TN"), ("KA", "KL"), ("TN", "KL"), ("TN", "PY"),
]

# State codes used by the `holidays` package for each zone's public-holiday calendar.
HOLIDAY_SUBDIV = {"TG": "TS", "AP": "AP", "KA": "KA", "TN": "TN", "KL": "KL", "PY": "PY"}

# Weather point per zone (main load centre) — used only if data/weather.csv is present.
WEATHER_POINTS = {
    "TG": (17.3850, 78.4867),   # Hyderabad
    "AP": (16.5062, 80.6480),   # Vijayawada
    "KA": (12.9716, 77.5946),   # Bengaluru
    "TN": (13.0827, 80.2707),   # Chennai
    "KL": (9.9312, 76.2673),    # Kochi
    "PY": (11.9416, 79.8083),   # Puducherry
}

HORIZON = 3                     # forecast hours ahead
TRAIN = ("2023-01-01", "2024-12-31")
TEST = ("2025-01-01", "2025-12-31")
EXTRA_TEST = ("2026-01-01", "2026-02-28")   # Mar 2026 is a copy of Jan in the source

ROUNDS = 40
LOCAL_EPOCHS = 5
SEEDS = [0, 1, 2, 3, 4]
PEAK_QUANTILE = 0.90            # peak window = hours at or above a state's 90th percentile load


def neighbours():
    nb = {z: [] for z in ZONES}
    for a, b in EDGES:
        nb[a].append(b)
        nb[b].append(a)
    return nb
