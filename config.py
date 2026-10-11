# ============================================================
# config.py — All project parameters in one place
# Change anything here and it flows through the whole project
# ============================================================

# --- Data Settings ---
TICKER = "SPY"           # Asset to simulate market making on
PERIOD = "5years"           # How much historical data to pull
START_DATE = "2021-05-01"
END_DATE = "2026-05-01"
DATA_PATH = "data/prices.csv"

# --- Statistical Settings ---
ROLLING_WINDOW = 20      # Days for rolling volatility (1 trading month)

# --- Model Parameters ---
GAMMA = 0.1             # Risk aversion (higher = more conservative quoting)
                         # Kept as a design choice rather than calibrated from market data

KAPPA = 33.05159978243325  # Calibrated SPY IEX fill-decay proxy from Alpaca microstructure data
                           # Larger kappa means fills decay faster as quotes move away from mid

# --- Simulation Settings ---
INITIAL_CASH = 100_000   # Starting cash in simulation
OBSERVATION_DT = 1.0      # One row = one trading day, independent of dataset length
FILL_REFERENCE_DT = 1.0   # Fixed trading-day reference for the illustrative arrival proxy
TRADING_DAYS_PER_YEAR = 252.0

# --- Market Microstructure Proxy ---
MARKET_SPREAD = 0.09    # Calibrated SPY IEX median spread proxy from Alpaca quote data
                        # Used in the paper-style arrival-scale calibration

# --- Baseline Settings ---
FIXED_SPREAD = MARKET_SPREAD  # Passive baseline uses the configured market spread

# --- Monte Carlo (for final version) ---
N_SIMULATIONS = 100      # Number of Monte Carlo runs
MC_RANDOM_SEED = 55      # Base seed for reproducible Monte Carlo paths and fills
MC_MIN_PRICE = 0.01      # Safety floor to keep arithmetic Brownian paths positive enough for log returns

# --- Output Paths ---
PLOTS_DIR = "outputs/plots"
RESULTS_DIR = "outputs/results"
