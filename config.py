"""All project constants in one place.

Every module imports its settings from here, so a parameter is changed in
exactly one location and every result can be traced back to this file.
Nothing in this file performs computation or touches the network.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
LOG_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = ROOT / "figures"
GRID_LOG = RESULTS_DIR / "grid_log.csv"

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
# name: (US ticker, HK ticker, HK shares per ADR)
# All six ratios verified 2026-10-07 against the Deutsche Bank DR directory
# (adr.db.com, DR details pages); YUMC confirmed separately as 1:1.
# These are CURRENT ratios; the Session 1 split check guards against a change
# inside the sample (which would go in RATIO_CHANGES).
PAIRS: dict[str, tuple[str, str, int]] = {
    "Alibaba":  ("BABA", "9988.HK", 8),
    "JD":       ("JD",   "9618.HK", 2),
    "NetEase":  ("NTES", "9999.HK", 5),
    "YumChina": ("YUMC", "9987.HK", 1),
    "Baidu":    ("BIDU", "9888.HK", 8),
    "TripCom":  ("TCOM", "9961.HK", 1),
}

# Date-dependent ADR ratios, for any pair whose ratio changed mid-sample.
# Format: name -> list of (effective_date, HK shares per ADR), sorted by date.
# The ratio applies from effective_date (inclusive) until the next entry;
# before the first entry the ratio in PAIRS applies.
# Pairs not listed here use the constant ratio in PAIRS for the whole sample.
RATIO_CHANGES: dict[str, list[tuple[str, int]]] = {}

FX_TICKER = "HKD=X"  # HKD per 1 USD

# ---------------------------------------------------------------------------
# Sample periods and the out-of-sample firewall
# ---------------------------------------------------------------------------
START, END = "2021-04-19", "2026-09-30"
IS_START = START
IS_END = "2023-12-31"
OOS_START = "2024-01-01"
OOS_END = END

# Download window for the raw snapshot (wider than the backtest window).
# - Start: 2019-01-01 predates every HK listing (Alibaba HK, Nov 2019), so the
#   snapshot holds all HK history; pre-START rows serve only as rolling-window
#   warm-up. The backtest itself still starts at START for every pair.
# - End: yfinance treats `end` as EXCLUSIVE, and the HK leg executes at the HK
#   open on day t+1, so we fetch a few sessions past END.
DOWNLOAD_START = "2019-01-01"
DOWNLOAD_END = "2026-10-08"

# Must stay False until Session 6. Data loaders raise an error if asked for
# any date >= OOS_START while this is False, so in-sample work cannot peek.
ALLOW_OOS = False

# Dates each HK line became eligible for Southbound Stock Connect.
CONNECT_EVENTS: dict[str, str] = {"Alibaba": "2024-09-10", "Baidu": "2026-09-07"}
CONNECT_WINDOW = 120  # trading days before/after the event

# ---------------------------------------------------------------------------
# Data-quality thresholds
# ---------------------------------------------------------------------------
FX_MAX_FFILL_DAYS = 1      # forward-fill FX at most this many days (logged)
SPREAD_FLAG_ABS = 0.10     # flag |spread| above 10% for manual review

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
STOP_Z = 4.0  # fixed, not tuned

# In-sample grid: 3 x 3 x 2 x 2 = 36 combinations.
GRID: dict[str, list] = {
    "L":      [20, 60, 120],    # rolling window (trading days)
    "k":      [1.5, 2.0, 2.5],  # entry threshold |z| > k
    "exit_z": [0.0, 0.5],       # exit when |z| < exit_z
    "H":      [5, 10],          # max holding days
}

# Frozen parameters — chosen 2026-10-09 at the end of Session 5, BEFORE OOS is run.
# Do not change after this point: the OOS period is run once with these values.
# Choice (student, from the 72-run in-sample grid at costs 1x, results/grid_plateau.csv):
#   - Net Sharpe improves steadily towards L=120, k=2.5 in both signals and all four
#     (exit_z, H) slices: fewer, larger-deviation trades beat a fixed cost per trade.
#   - Within that corner, exit_z=0.5, H=10 is plateau-like for the two-reading signal
#     (own net Sharpe 0.08 vs neighbourhood mean 0.07, best worst-neighbour +0.01);
#     the higher H=5 cell (0.15) is a spike driven by a single JD trade.
#   - H=10 rarely binds (average holding 1.1-2.7 days), so the result does not hinge on it.
#   - One setting for both signals, so OOS differences come from the signal, not tuning.
# Caveats: the setting sits at the edge of the pre-registered grid (not extended), and
# in-sample net Sharpe (morning 0.06, two-reading 0.08) is well within one standard
# error (~0.61 over 2.69 years) of zero.
FROZEN_PARAMS: dict | None = {"L": 120, "k": 2.5, "exit_z": 0.5, "H": 10, "stop_z": STOP_Z}

# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
PAIR_WEIGHT = 1 / len(PAIRS)  # each pair gets an equal share of capital
LEG_WEIGHT = 0.5              # each leg = 50% of pair capital (gross 1x)

# ---------------------------------------------------------------------------
# Costs (per side, fraction of traded notional)
# ---------------------------------------------------------------------------
HK_STAMP_DUTY = 0.0010
HK_LEVIES = 0.0001
COMMISSION = 0.0003  # both legs

# Half bid-ask spread per pair, applied to both legs.
HALF_SPREAD: dict[str, float] = {
    "Alibaba":  0.0005,
    "JD":       0.0005,
    "NetEase":  0.0010,
    "YumChina": 0.0010,
    "Baidu":    0.0010,
    "TripCom":  0.0010,
}

BORROW_RATE_ANNUAL = 0.01  # short leg, accrued daily
TRADING_DAYS = 252         # annualisation (Sharpe, borrow accrual)

COST_MULTIPLIERS = [0.0, 0.5, 1.0, 2.0]

# ---------------------------------------------------------------------------
# Download behaviour (Session 1)
# ---------------------------------------------------------------------------
DOWNLOAD_RETRIES = 3        # attempts per ticker before giving up
DOWNLOAD_BACKOFF_SEC = 5.0  # wait 5s, 10s, ... between attempts
DOWNLOAD_PAUSE_SEC = 1.0    # polite pause between tickers


def all_tickers() -> list[str]:
    """Every series in the snapshot: US and HK leg of each pair, then FX."""
    tickers = [t for us, hk, _ in PAIRS.values() for t in (us, hk)]
    return tickers + [FX_TICKER]
