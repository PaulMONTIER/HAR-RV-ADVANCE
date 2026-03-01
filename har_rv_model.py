#!/usr/bin/env python3
"""
================================================================================
                    MODELE HAR-RV OPTIMISE
               Prediction de la Volatilite Realisee
================================================================================

Auteur: Paul MONTIER
Date: Decembre 2024

--------------------------------------------------------------------------------
CHANGELOG
--------------------------------------------------------------------------------

2026-03-01 (v6 — features leverage):
  - Ret_w : rendement cumule 5j. Seule feature directionnelle (leverage effect).
  - Leverage_22d : corr rolling 22j entre return et delta_RV. Force du leverage.
  - 18 → 20 features.

2026-03-01 (v5 — fenetre adaptative VIX):
  - Fenetre d'entrainement adaptative basee sur le VIX (interpolation lineaire).
    VIX <= 18 → 504j (regime calme, plus de donnees utiles).
    VIX >= 22 → 189j (regime crise, donnees recentes plus pertinentes).
    Transition continue entre les deux (pas de saut discret).
  - Applique dans backtest(), run_benchmark() et tune_xgboost().

2026-02-28 (v4 — moments superieurs intraday):
  - Realized Skewness (RSkew) : sqrt(n)*sum(r^3)/RV^(3/2), asymetrie crash risk.
  - Realized Kurtosis (RKurt) : n*sum(r^4)/RV^2, epaisseur des queues.
  - Jump Ratio : max(0, RV-BPV)/RV, decomposition BPV (Barndorff-Nielsen &
    Shephard 2004) remplace le J_w arbitraire (seuil 95e percentile).
  - 15 → 18 features.

2026-02-28 (v3 — corrections methodologiques):
  - Purging walk-forward : gap de horizon-1=4 samples entre train et test
    pour eliminer le chevauchement des targets (data leakage).
  - Target decouple : log(RV_{t+5} / RV_m) au lieu de log(RV_{t+5} / RV_w).
    RV_w etait feature ET denominateur → correlation mecanique exploitee.
    RV_m (22j) comme denominateur preserve la stationnarisation sans fuite.
    Winsorisation a [-3, 3] contre les outliers.
  - Vraie RV intraday : sqrt(sum(r_1min^2)) par jour au lieu de mean(|r_daily|).
    Utilise pour RV_w, RV_m, RV_q features + target. Fallback MAD si absent.
  - VIX_regime retire des features (redondant pour XGBoost). 16 → 15 features.
  - CV purge dans tune_xgboost() : gap=4 entre folds.

2026-02-28 (v2):
  - Suppression de RV_intra et Max_hourly (rho=0.97 et 0.90 avec RV_1min).
  - XGBoost regularise : max_depth=3, min_child_weight=5, subsample=0.8,
    reg_lambda=1.5. Tune via Spearman IC multi-stocks.
  - Stacking Ridge+XGBoost teste et ecarte (IC -18.6% vs XGBoost seul
    malgre rho=0.72, Ridge trop faible en IC pour enrichir le signal).

--------------------------------------------------------------------------------
DESCRIPTION
--------------------------------------------------------------------------------

Ce modele predit le log-ratio de volatilite realisee future d'une action
sur un horizon de 5 jours : log(RV_{t+5} / RV_m_t).
Extension du modele HAR-RV (Heterogeneous Autoregressive model of Realized
Volatility) de Corsi (2009).

FORMULE ETENDUE (20 features):
    log(RV_{t+5} / RV_m) = f(RV_w, RV_m, RV_q, RV_neg_w, J_w, VIX,
                              RV_overnight, Parkinson, Volume_ratio,
                              RV_1min, Vol_ratio_1m5m, Vol_AM_PM, Autocorr,
                              RSkew, RKurt, Jump_ratio,
                              Ret_w, Leverage_22d,
                              RV_w_zscore, RV_w_rank_delta)

MODELES DISPONIBLES:
    - ridge        : Regression Ridge (L2)
    - lasso        : Regression Lasso (L1)
    - elasticnet   : Regression Elastic Net (L1+L2)
    - logistic     : Regression Logistique (classification direction)
    - random_forest: Random Forest
    - xgboost      : XGBoost (recommande — IC=0.61, HR=71%)
    - mlp          : Reseau de Neurones (MLP)
    - stacking     : Ensemble Stacking (Ridge + Lasso + XGBoost)

--------------------------------------------------------------------------------
REFERENCES
--------------------------------------------------------------------------------

- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility.
  Journal of Financial Econometrics, 7(2), 174-196.

- Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2007). Roughing it up:
  Including jump components in the measurement of realized volatility.
  Review of Economics and Statistics, 89(4), 701-720.

- Barndorff-Nielsen, O. E., & Shephard, N. (2004). Power and bipower variation
  with stochastic volatility and jumps. Journal of Financial Econometrics, 2(1), 1-37.

- Amaya, D., Christoffersen, P., Jacobs, K., & Vasquez, A. (2015). Does realized
  skewness and kurtosis predict the cross-section of equity returns?
  Journal of Financial Economics, 118(1), 135-167.

- Black, F. (1976). Studies of stock market volatility changes. Proceedings of the
  Business and Economic Statistics Section, American Statistical Association, 177-181.

- Christie, A. A. (1982). The stochastic behavior of common stock variances: Value,
  leverage and interest rate effects. Journal of Financial Economics, 10(4), 407-432.

================================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ==============================================================================
# CONFIGURATION
# ==============================================================================

HORIZON = 5
TRAIN_WINDOW = 252
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Fenetre adaptative : interpolation lineaire entre VIX anchors
# VIX <= 18 → 504j (2 ans, regime calme), VIX >= 22 → 189j (regime crise)
# Interpolation lineaire entre les deux
ADAPTIVE_WINDOW_ANCHORS = [(18, 504), (22, 189)]
ADAPTIVE_WINDOW_MIN = 126  # minimum absolu

# Modeles disponibles
AVAILABLE_MODELS = {
    'ridge': 'Ridge (L2)',
    'lasso': 'Lasso (L1)',
    'elasticnet': 'Elastic Net (L1+L2)',
    'logistic': 'Logistique (direction)',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'mlp': 'MLP Neural Network',
    'stacking': 'Stacking Ensemble',
}

REGIME_NAMES = {0: 'Low', 1: 'Medium', 2: 'High'}


def adaptive_train_window(vix_value: float,
                          anchors=ADAPTIVE_WINDOW_ANCHORS,
                          minimum: int = ADAPTIVE_WINDOW_MIN) -> int:
    """Fenetre d'entrainement adaptative basee sur le VIX.

    Interpolation lineaire entre les points d'ancrage.
    VIX bas → fenetre longue (marche stable, plus de donnees utile).
    VIX haut → fenetre courte (regime change, donnees recentes plus pertinentes).

    Args:
        vix_value: VIX au jour t (pas de look-ahead)
        anchors: liste de (vix_threshold, window_size) triee par VIX croissant
        minimum: taille minimum de la fenetre

    Returns:
        Taille de la fenetre en jours (int)
    """
    if np.isnan(vix_value):
        return TRAIN_WINDOW  # fallback 252

    vix_lo, win_lo = anchors[0]
    vix_hi, win_hi = anchors[-1]

    if vix_value <= vix_lo:
        return win_lo
    if vix_value >= vix_hi:
        return max(minimum, win_hi)

    # Interpolation lineaire
    t = (vix_value - vix_lo) / (vix_hi - vix_lo)
    window = int(win_lo + t * (win_hi - win_lo))
    return max(minimum, window)


# ==============================================================================
# FEATURES CROSS-SECTIONNELLES
# ==============================================================================

def build_cross_sectional_features(all_data: dict) -> dict:
    """
    Calcule les features cross-sectionnelles pour chaque stock de l'univers.

    Features calculees:
      - RV_w_zscore : z-score de RV_w du stock vs l'univers au jour t
        (RV_w_stock - mean) / std, calcule sur les stocks disponibles ce jour.
      - RV_w_rank_delta : variation du rang percentile sur 5 jours
        (rank_t - rank_{t-5}). Positif = acceleration, negatif = deceleration.

    Pas de look-ahead: chaque jour t utilise uniquement les donnees a t.

    Args:
        all_data: Dict {ticker: DataFrame} avec colonne 'returns'

    Returns:
        Dict {ticker: {'RV_w_zscore': Series, 'RV_w_rank_delta': Series}}
    """
    loader = HARRVModel()

    rv_w_dict = {}
    for symbol, df in all_data.items():
        # Utilise la vraie RV intraday si disponible
        if 'rv_intraday' in df.columns and df['rv_intraday'].notna().sum() > 60:
            rv_w = df['rv_intraday'].rolling(5).mean()
            rv_w.name = symbol
        else:
            returns = df['returns'].values.flatten()
            rv_w = pd.Series(
                loader.compute_realized_volatility(returns, 5),
                index=df.index
            )
        rv_w_dict[symbol] = rv_w

    if len(rv_w_dict) < 2:
        return {}

    rv_w_df = pd.DataFrame(rv_w_dict)

    # Z-score cross-sectionnel par jour
    cs_mean = rv_w_df.mean(axis=1)
    cs_std = rv_w_df.std(axis=1)
    zscore_df = rv_w_df.sub(cs_mean, axis=0).div(cs_std.replace(0, np.nan), axis=0)

    # Rang percentile + delta lag-5 (momentum du rang)
    rank_df = rv_w_df.rank(axis=1, pct=True)
    rank_delta_df = rank_df - rank_df.shift(5)

    result = {}
    for symbol in rv_w_dict:
        result[symbol] = {
            'RV_w_zscore': zscore_df[symbol],
            'RV_w_rank_delta': rank_delta_df[symbol],
        }

    return result


# ==============================================================================
# REGIME VIX
# ==============================================================================

def build_vix_regime(vix_series: pd.Series, window: int = 252) -> pd.Series:
    """
    Classifie le VIX en regimes {0=Low, 1=Medium, 2=High} a partir
    de percentiles rolling p33/p66 calcules sur les `window` jours passes.

    Pas de look-ahead: chaque jour t utilise uniquement les valeurs [t-window+1, t].

    Args:
        vix_series: Serie VIX (deja decalee si necessaire)
        window: Fenetre rolling pour le calcul des percentiles

    Returns:
        Serie ordinale {0, 1, 2} alignee sur l'index d'entree
    """
    p33 = vix_series.rolling(window).quantile(0.33)
    p66 = vix_series.rolling(window).quantile(0.66)

    regime = pd.Series(np.nan, index=vix_series.index)
    regime[vix_series <= p33] = 0   # Low
    regime[(vix_series > p33) & (vix_series <= p66)] = 1  # Medium
    regime[vix_series > p66] = 2    # High

    return regime


# ==============================================================================
# CLASSE PRINCIPALE DU MODELE
# ==============================================================================

class HARRVModel:
    """
    Modele HAR-RV avec modeles pluggables et features multi-timeframe.

    Args:
        model_type: Type de modele ('ridge', 'lasso', 'elasticnet', 'logistic',
                    'random_forest', 'xgboost', 'mlp', 'stacking')
        horizon: Horizon de prediction en jours
        train_window: Fenetre d'entrainement rolling
        **model_params: Hyperparametres specifiques au modele
    """

    def __init__(self,
                 model_type: str = 'ridge',
                 horizon: int = HORIZON,
                 train_window: int = TRAIN_WINDOW,
                 **model_params):
        self.model_type = model_type
        self.horizon = horizon
        self.train_window = train_window
        self.model_params = model_params
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._base_models = None  # Pour stacking

    # ==========================================================================
    # FACTORY MODELE
    # ==========================================================================

    def _create_model(self):
        """Cree le modele sklearn/xgboost selon model_type."""
        p = self.model_params

        if self.model_type == 'ridge':
            return Ridge(alpha=p.get('alpha', 1.0))

        elif self.model_type == 'lasso':
            return Lasso(alpha=p.get('alpha', 0.01), max_iter=5000)

        elif self.model_type == 'elasticnet':
            return ElasticNet(
                alpha=p.get('alpha', 0.1),
                l1_ratio=p.get('l1_ratio', 0.5),
                max_iter=5000
            )

        elif self.model_type == 'logistic':
            return LogisticRegression(max_iter=1000, C=p.get('C', 1.0))

        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=p.get('n_estimators', 100),
                max_depth=p.get('max_depth', 8),
                random_state=42
            )

        elif self.model_type == 'xgboost':
            if not HAS_XGBOOST:
                raise ImportError("xgboost non installe: pip install xgboost")
            return XGBRegressor(
                n_estimators=p.get('n_estimators', 200),
                max_depth=p.get('max_depth', 3),
                min_child_weight=p.get('min_child_weight', 5),
                subsample=p.get('subsample', 0.8),
                colsample_bytree=p.get('colsample_bytree', 0.8),
                learning_rate=p.get('learning_rate', 0.05),
                reg_alpha=p.get('reg_alpha', 0),
                reg_lambda=p.get('reg_lambda', 1.5),
                verbosity=0
            )

        elif self.model_type == 'mlp':
            return MLPRegressor(
                hidden_layer_sizes=p.get('hidden_layers', (64, 32)),
                max_iter=p.get('max_iter', 500),
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15
            )

        elif self.model_type == 'stacking':
            return None  # Gere separement dans fit()

        else:
            raise ValueError(f"model_type inconnu: {self.model_type}. "
                           f"Disponibles: {list(AVAILABLE_MODELS.keys())}")

    def _needs_scaling(self) -> bool:
        """Les tree-based n'ont pas besoin de normalisation."""
        return self.model_type not in ('random_forest', 'xgboost')

    def _is_classifier(self) -> bool:
        """Logistic = classification (target binaire)."""
        return self.model_type == 'logistic'

    # ==========================================================================
    # RECUPERATION DES DONNEES
    # ==========================================================================

    def get_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Charge les donnees d'une action depuis le CSV local.
        Extrait les barres 1Day + calcule des metriques intraday depuis 5Min, 1Min, 1Hour.
        """
        csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")

        if not os.path.exists(csv_path):
            print(f"  {symbol}: Fichier introuvable. Lancez download_data.py.")
            return None

        try:
            df_all = pd.read_csv(csv_path)

            # --- Barres journalieres ---
            df = df_all[df_all['timeframe'] == '1Day'].copy()
            df['Time'] = pd.to_datetime(df['Time'], utc=True)
            df.index = df['Time'].dt.tz_localize(None).dt.normalize()
            df.index.name = 'Date'
            df = df.sort_index()

            if len(df) < self.train_window + 100:
                print(f"  {symbol}: Donnees insuffisantes ({len(df)})")
                return None

            df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

            # --- Vol intraday depuis barres 5Min ---
            df_5min = df_all[df_all['timeframe'] == '5Min'].copy()
            if len(df_5min) > 0:
                df_5min['Time'] = pd.to_datetime(df_5min['Time'], utc=True)
                df_5min['date'] = df_5min['Time'].dt.tz_localize(None).dt.normalize()
                df_5min = df_5min.sort_values('Time')
                df_5min['ret'] = df_5min.groupby('date')['Close'].transform(
                    lambda x: np.log(x / x.shift(1))
                )
                intraday_vol = df_5min.groupby('date')['ret'].std()
                intraday_vol.index.name = 'Date'
                df['intraday_vol'] = intraday_vol.reindex(df.index)

            # --- Vol intraday depuis barres 1Min (plus granulaire) ---
            df_1min = df_all[df_all['timeframe'] == '1Min'].copy()
            if len(df_1min) > 0:
                df_1min['Time'] = pd.to_datetime(df_1min['Time'], utc=True)
                df_1min['date'] = df_1min['Time'].dt.tz_localize(None).dt.normalize()
                df_1min['hour'] = df_1min['Time'].dt.hour  # UTC
                df_1min = df_1min.sort_values('Time')
                df_1min['ret'] = df_1min.groupby('date')['Close'].transform(
                    lambda x: np.log(x / x.shift(1))
                )

                # Vol 1min par jour (std pour feature RV_1min)
                vol_1min = df_1min.groupby('date')['ret'].std()
                vol_1min.index.name = 'Date'
                df['vol_1min'] = vol_1min.reindex(df.index)

                # Vraie RV intraday : sqrt(sum(r_1min^2)) par jour
                rv_intraday = df_1min.groupby('date')['ret'].apply(
                    lambda x: np.sqrt((x.dropna()**2).sum())
                )
                rv_intraday.index.name = 'Date'
                df['rv_intraday'] = rv_intraday.reindex(df.index)

                # Realized Skewness : sqrt(n) * sum(r^3) / RV^(3/2) par jour
                # Asymetrie intraday, capte le crash risk (Amaya et al., 2015)
                rskew = df_1min.groupby('date')['ret'].apply(
                    lambda x: x.dropna().pipe(
                        lambda r: np.sqrt(len(r)) * (r**3).sum() / max(1e-20, ((r**2).sum())**(3/2))
                    ) if len(x.dropna()) >= 10 else np.nan
                )
                rskew.index.name = 'Date'
                df['rskew_intraday'] = rskew.reindex(df.index)

                # Realized Kurtosis : n * sum(r^4) / RV^2 par jour
                # Epaisseur des queues, capte les regimes extremes
                rkurt = df_1min.groupby('date')['ret'].apply(
                    lambda x: x.dropna().pipe(
                        lambda r: len(r) * (r**4).sum() / max(1e-20, ((r**2).sum())**2)
                    ) if len(x.dropna()) >= 10 else np.nan
                )
                rkurt.index.name = 'Date'
                df['rkurt_intraday'] = rkurt.reindex(df.index)

                # Bipower Variation : (pi/2) * sum(|r_t| * |r_{t-1}|) par jour
                # Composante continue de la vol sans les jumps (Barndorff-Nielsen & Shephard, 2004)
                bpv = df_1min.groupby('date')['ret'].apply(
                    lambda x: x.dropna().pipe(
                        lambda r: (np.pi / 2) * (r.abs() * r.abs().shift(1)).dropna().sum()
                    ) if len(x.dropna()) >= 10 else np.nan
                )
                bpv.index.name = 'Date'
                df['bpv_intraday'] = bpv.reindex(df.index)

                # Autocorrelation des returns 1min (mean-reversion / momentum intraday)
                autocorr_1min = df_1min.groupby('date')['ret'].apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 10 else np.nan
                )
                autocorr_1min.index.name = 'Date'
                df['autocorr_1min'] = autocorr_1min.reindex(df.index)

                # Volume AM (12-16h UTC = 8h-12h ET) vs PM (16-20h UTC = 12h-16h ET)
                vol_am = df_1min[df_1min['hour'].between(12, 15)].groupby('date')['Volume'].sum()
                vol_pm = df_1min[df_1min['hour'].between(16, 19)].groupby('date')['Volume'].sum()
                vol_am.index.name = 'Date'
                vol_pm.index.name = 'Date'
                vol_am = vol_am.reindex(df.index, fill_value=0)
                vol_pm = vol_pm.reindex(df.index, fill_value=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    df['volume_am_pm'] = np.where(vol_pm > 0, vol_am / vol_pm, 1.0)

            # --- Max range horaire depuis barres 1Hour ---
            df_1h = df_all[df_all['timeframe'] == '1Hour'].copy()
            if len(df_1h) > 0:
                df_1h['Time'] = pd.to_datetime(df_1h['Time'], utc=True)
                df_1h['date'] = df_1h['Time'].dt.tz_localize(None).dt.normalize()
                df_1h = df_1h.sort_values('Time')
                # Range horaire = (High - Low) / Low
                with np.errstate(divide='ignore', invalid='ignore'):
                    df_1h['hourly_range'] = np.where(
                        df_1h['Low'] > 0,
                        (df_1h['High'] - df_1h['Low']) / df_1h['Low'],
                        0
                    )
                max_hourly = df_1h.groupby('date')['hourly_range'].max()
                max_hourly.index.name = 'Date'
                df['max_hourly_range'] = max_hourly.reindex(df.index)

            df = df.dropna(subset=['returns'])
            return df

        except Exception as e:
            print(f"  {symbol}: Erreur lecture - {e}")
            return None

    def get_vix(self) -> Optional[pd.Series]:
        """Charge le VIX depuis le CSV local."""
        csv_path = os.path.join(DATA_DIR, "VIX.csv")

        if not os.path.exists(csv_path):
            return None

        try:
            df = pd.read_csv(csv_path)
            df = df[df['timeframe'] == '1Day'].copy()
            df['Time'] = pd.to_datetime(df['Time'], utc=True)
            df.index = df['Time'].dt.tz_localize(None).dt.normalize()
            df.index.name = 'Date'
            df = df.sort_index()
            return df['Close']
        except:
            return None

    # ==========================================================================
    # CALCUL DES FEATURES
    # ==========================================================================

    def compute_realized_volatility(self, returns: np.ndarray, window: int) -> np.ndarray:
        """RV = moyenne glissante des |returns| sur 'window' jours (fallback daily)."""
        return pd.Series(np.abs(returns)).rolling(window).mean().values

    def compute_rv_intraday(self, rv_daily: np.ndarray, window: int) -> np.ndarray:
        """Moyenne glissante de la RV intraday journaliere sur 'window' jours.

        rv_daily doit contenir sqrt(sum(r_1min^2)) pour chaque jour.
        """
        return pd.Series(rv_daily).rolling(window, min_periods=max(1, window // 2)).mean().values

    def compute_semivariance(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Semi-variance negative: moyenne des |returns| quand r < 0."""
        rv_neg = np.where(returns < 0, np.abs(returns), 0)
        return pd.Series(rv_neg).rolling(window).mean().values

    def compute_jumps(self, returns: np.ndarray, window: int = 5,
                      threshold_window: int = 252) -> np.ndarray:
        """Composante jumps: mouvements > 95eme percentile glissant."""
        rv_daily = np.abs(returns)
        rolling_threshold = pd.Series(rv_daily).rolling(threshold_window).quantile(0.95).values
        is_jump = rv_daily > rolling_threshold
        jump_rv = np.where(is_jump, rv_daily, 0)
        return pd.Series(jump_rv).rolling(window).mean().values

    def create_features(self, df: pd.DataFrame, vix: Optional[pd.Series] = None,
                        cs_features: Optional[Dict[str, pd.Series]] = None) -> pd.DataFrame:
        """
        Cree les 20 features du modele HAR-RV etendu.

        Features originales (6):
            RV_w, RV_m, RV_q, RV_neg_w, J_w, VIX

        Features multi-timeframe daily (3):
            RV_overnight, Parkinson_vol, Volume_ratio

        Features intraday 1Min (4):
            RV_1min, Vol_1min_5min_ratio, Volume_am_pm, Autocorr_intra

        Features moments superieurs intraday (3):
            RSkew (realized skewness — crash risk)
            RKurt (realized kurtosis — fat tails)
            Jump_ratio (proportion jumps via BPV decomposition)

        Features leverage (2):
            Ret_w (rendement cumule 5j — leverage effect directionnel)
            Leverage_22d (corr rolling 22j ret vs delta_RV — force du leverage)

        Features cross-sectionnelles (2):
            RV_w_zscore (z-score de RV_w vs univers au jour t)
            RV_w_rank_delta (momentum du rang: rank_t - rank_{t-5})
        """
        returns = df['returns'].values.flatten()
        features = pd.DataFrame(index=df.index)

        # === FEATURES ORIGINALES ===
        # Utilise la vraie RV intraday (sqrt(sum(r_1min^2))) si disponible,
        # sinon fallback sur MAD journalier (mean(|daily_returns|))
        has_intraday = 'rv_intraday' in df.columns and df['rv_intraday'].notna().sum() > 60

        if has_intraday:
            rv_daily = df['rv_intraday'].values
            # 1. RV_w - Volatilite Hebdomadaire (5 jours) - Court terme
            features['RV_w'] = self.compute_rv_intraday(rv_daily, 5)
            # 2. RV_m - Volatilite Mensuelle (22 jours) - Moyen terme
            features['RV_m'] = self.compute_rv_intraday(rv_daily, 22)
            # 3. RV_q - Volatilite Trimestrielle (60 jours) - Memoire longue
            features['RV_q'] = self.compute_rv_intraday(rv_daily, 60)
        else:
            # Fallback : MAD journalier
            features['RV_w'] = self.compute_realized_volatility(returns, 5)
            features['RV_m'] = self.compute_realized_volatility(returns, 22)
            features['RV_q'] = self.compute_realized_volatility(returns, 60)

        # 4. RV_neg_w - Semi-variance Negative (5 jours) - Asymetrie
        features['RV_neg_w'] = self.compute_semivariance(returns, 5)

        # 5. J_w - Composante Jumps (5 jours)
        features['J_w'] = self.compute_jumps(returns, window=5)

        # 6. VIX - Volatilite Implicite (decale +1j)
        if vix is not None:
            vix_aligned = vix.reindex(df.index, method='ffill').values.flatten()
            features['VIX'] = pd.Series(vix_aligned).shift(1).values

        # VIX_regime : calcule pour l'evaluation par regime (pas comme feature)
        # Redondant pour les tree-based : XGBoost cree ses propres seuils sur VIX continu
        if 'VIX' in features.columns:
            self._vix_regime = build_vix_regime(
                pd.Series(features['VIX'].values, index=features.index)
            )

        # === FEATURES MULTI-TIMEFRAME (DAILY) ===

        # 7. RV_overnight - Gap overnight (|Open_t - Close_{t-1}| / Close_{t-1})
        if 'Open' in df.columns:
            overnight = np.abs(df['Open'].values - df['Close'].shift(1).values) / df['Close'].shift(1).values
            features['RV_overnight'] = pd.Series(overnight).rolling(5).mean().values

        # 9. Parkinson_vol - Volatilite range-based sqrt(ln(H/L)^2 / (4*ln2))
        if 'High' in df.columns and 'Low' in df.columns:
            hl_ratio = np.log(df['High'].values / df['Low'].values)
            parkinson_daily = np.sqrt(hl_ratio ** 2 / (4 * np.log(2)))
            features['Parkinson'] = pd.Series(parkinson_daily).rolling(5).mean().values

        # 10. Volume_ratio - Volume / Volume moyen 22j
        if 'Volume' in df.columns:
            vol_mean = pd.Series(df['Volume'].values, dtype=float).rolling(22).mean().values
            with np.errstate(divide='ignore', invalid='ignore'):
                features['Vol_ratio'] = np.where(vol_mean > 0, df['Volume'].values / vol_mean, 1.0)

        # === FEATURES INTRADAY 1Min / 1Hour ===

        # 11. RV_1min - Vol depuis barres 1min (plus granulaire, moyenne 5j)
        if 'vol_1min' in df.columns:
            features['RV_1min'] = df['vol_1min'].rolling(5).mean().values

        # 12. Vol_1min_5min_ratio - Ratio vol 1min / vol 5min (proxy microstructure)
        #     Un ratio eleve = bruit microstructure / activite HFT
        if 'vol_1min' in df.columns and 'intraday_vol' in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(
                    df['intraday_vol'].values > 0,
                    df['vol_1min'].values / df['intraday_vol'].values,
                    1.0
                )
            features['Vol_ratio_1m5m'] = pd.Series(ratio).rolling(5).mean().values

        # 13. Volume AM/PM ratio - Volume matin / apres-midi (smart money, moyenne 5j)
        #     Les institutionnels tradent plus le matin
        if 'volume_am_pm' in df.columns:
            features['Vol_AM_PM'] = df['volume_am_pm'].rolling(5).mean().values

        # 14. Autocorr intraday - Autocorrelation lag-1 des returns 1min (moyenne 5j)
        #     Negatif = mean-reversion, Positif = momentum intraday
        if 'autocorr_1min' in df.columns:
            features['Autocorr'] = df['autocorr_1min'].rolling(5).mean().values

        # === FEATURES MOMENTS SUPERIEURS INTRADAY ===

        # 15. RSkew - Realized Skewness (asymetrie intraday, rolling 5j)
        #     Negatif = crash risk (queues gauches), Positif = rallye asymetrique
        if 'rskew_intraday' in df.columns:
            features['RSkew'] = df['rskew_intraday'].rolling(5).mean().values

        # 16. RKurt - Realized Kurtosis (epaisseur des queues, rolling 5j)
        #     > 3 = queues epaisses (fat tails), = 3 si gaussien
        if 'rkurt_intraday' in df.columns:
            features['RKurt'] = df['rkurt_intraday'].rolling(5).mean().values

        # 17. Jump_ratio - Proportion de la vol due aux sauts (rolling 5j)
        #     max(0, RV - BPV) / RV : 0 = pas de jumps, 1 = que des jumps
        #     Remplace le J_w arbitraire (seuil 95e percentile) par la decomposition
        #     BPV standard (Barndorff-Nielsen & Shephard, 2004)
        if 'bpv_intraday' in df.columns and has_intraday:
            rv_sq = df['rv_intraday'].values ** 2  # sum(r^2) = RV en variance
            bpv_vals = df['bpv_intraday'].values
            with np.errstate(divide='ignore', invalid='ignore'):
                jump_ratio = np.where(
                    (rv_sq > 1e-20) & np.isfinite(bpv_vals),
                    np.maximum(0, rv_sq - bpv_vals) / rv_sq,
                    0
                )
            features['Jump_ratio'] = pd.Series(jump_ratio).rolling(5).mean().values

        # === FEATURES LEVERAGE (Black 1976, Christie 1982) ===

        # 18. Ret_w - Rendement cumule 5 jours
        #     Leverage effect : rendements negatifs → hausse de vol future.
        #     Seule feature directionnelle du modele (les autres mesurent la magnitude).
        features['Ret_w'] = pd.Series(returns).rolling(5).sum().values

        # 19. Leverage_22d - Correlation rolling 22j entre return et delta_RV
        #     Mesure la force du leverage effect au jour t.
        #     Tres negatif = regime ou les baisses amplifient fortement la vol.
        if has_intraday:
            rv_daily_s = pd.Series(df['rv_intraday'].values, index=df.index)
            delta_rv = rv_daily_s.diff()
            ret_s = pd.Series(returns, index=df.index)
            features['Leverage_22d'] = ret_s.rolling(22).corr(delta_rv).values

        # === FEATURES CROSS-SECTIONNELLES ===

        # 20. RV_w_zscore - Z-score de RV_w vs l'univers au jour t
        #     Positif = stock anormalement volatile, negatif = calme
        # 21. RV_w_rank_delta - Momentum du rang (rank_t - rank_{t-5})
        #     Positif = monte dans le classement vol (acceleration)
        if cs_features is not None:
            for feat_name, feat_series in cs_features.items():
                features[feat_name] = feat_series.reindex(df.index).values

        self.feature_names = list(features.columns)
        return features

    def create_target(self, df: pd.DataFrame) -> np.ndarray:
        """Target = log(RV_{t+h} / RV_m_t) — log-ratio normalise par RV mensuelle.

        Denominateur = RV_m (22 jours) au lieu de RV_w (5 jours) pour eviter
        la fuite : RV_w est aussi une feature, utiliser le meme signal des
        deux cotes cree une correlation mecanique negative que XGBoost exploite.
        RV_m (22j) a peu de chevauchement avec les features 5j et preserve
        la stationnarisation cross-sectionnelle.
        """
        returns = df['returns'].values.flatten()
        n = len(returns)
        has_intraday = 'rv_intraday' in df.columns and df['rv_intraday'].notna().sum() > 60

        # RV future : moyenne de RV intraday sur les prochains horizon jours
        rv_future = np.full(n, np.nan)
        if has_intraday:
            rv_intra = df['rv_intraday'].values
            for i in range(n - self.horizon):
                future_rv = rv_intra[i+1:i+1+self.horizon]
                if np.all(np.isfinite(future_rv)) and np.all(future_rv > 0):
                    rv_future[i] = np.mean(future_rv)
                else:
                    future_returns = returns[i+1:i+1+self.horizon]
                    rv_future[i] = np.std(future_returns, ddof=1)
        else:
            for i in range(n - self.horizon):
                future_returns = returns[i+1:i+1+self.horizon]
                rv_future[i] = np.std(future_returns, ddof=1)

        # Denominateur : RV_m (22 jours) — decouple de la feature RV_w
        if has_intraday:
            rv_m = self.compute_rv_intraday(df['rv_intraday'].values, 22)
        else:
            rv_m = self.compute_realized_volatility(returns, 22)

        # Log-ratio : log(RV_future / RV_m) avec protection division par zero
        with np.errstate(divide='ignore', invalid='ignore'):
            target = np.where(
                (rv_m > 1e-12) & (rv_future > 0) & np.isfinite(rv_future),
                np.log(rv_future / rv_m),
                np.nan
            )

        # Winsorisation : clip les outliers extremes
        target = np.where(np.isnan(target), np.nan, np.clip(target, -3.0, 3.0))

        return target

    # ==========================================================================
    # ENTRAINEMENT ET PREDICTION
    # ==========================================================================

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HARRVModel':
        """Entraine le modele sur les donnees."""
        # Scaling conditionnel
        if self._needs_scaling():
            self.scaler = StandardScaler()
            X_proc = self.scaler.fit_transform(X)
        else:
            self.scaler = None
            X_proc = X

        # Classification: convertir target en binaire
        if self._is_classifier():
            y_proc = (y > np.median(y)).astype(int)
        else:
            y_proc = y

        # Stacking: entrainer plusieurs modeles de base
        if self.model_type == 'stacking':
            self._base_models = [
                Ridge(alpha=1.0),
                Lasso(alpha=0.01, max_iter=5000),
            ]
            if HAS_XGBOOST:
                self._base_models.append(
                    XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, verbosity=0)
                )
            for m in self._base_models:
                m.fit(X_proc, y_proc)
            self.model = True  # Marqueur "entraine"
        else:
            self.model = self._create_model()
            self.model.fit(X_proc, y_proc)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predit la volatilite future (ou probabilite pour logistic)."""
        if self.model is None:
            raise ValueError("Modele non entraine")

        X_proc = self.scaler.transform(X) if self.scaler else X

        if self.model_type == 'stacking':
            preds = np.column_stack([m.predict(X_proc) for m in self._base_models])
            return preds.mean(axis=1)

        if self._is_classifier():
            return self.model.predict_proba(X_proc)[:, 1]

        return self.model.predict(X_proc)

    def get_feature_importance(self) -> Dict[str, float]:
        """Retourne l'importance des features (adapte a chaque type de modele)."""
        if self.model is None or self.feature_names is None:
            return {}

        if self.model_type == 'stacking':
            # Moyenne des coefficients Ridge et Lasso
            coefs = np.zeros(len(self.feature_names))
            count = 0
            for m in self._base_models:
                if hasattr(m, 'coef_'):
                    coefs += np.abs(m.coef_)
                    count += 1
            if count > 0:
                coefs /= count
            return dict(zip(self.feature_names, coefs))

        if hasattr(self.model, 'coef_'):
            coefs = self.model.coef_
            if coefs.ndim > 1:
                coefs = coefs[0]
            return dict(zip(self.feature_names, coefs))

        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))

        return {}

    # ==========================================================================
    # BACKTEST
    # ==========================================================================

    def backtest(self, symbol: str, vix: Optional[pd.Series] = None,
                 cs_features: Optional[Dict[str, pd.Series]] = None) -> Optional[Dict]:
        """
        Backtest rolling sur une action.

        Returns:
            Dict avec hit_rate, ic, avg_gain_bps, predictions, actuals, dates,
            regime_metrics
        """
        df = self.get_stock_data(symbol)
        if df is None:
            return None

        features = self.create_features(df, vix, cs_features)
        target = self.create_target(df)

        # VIX regime pour evaluation conditionnelle (calcule dans create_features)
        vix_regime_col = getattr(self, '_vix_regime', None)

        # Nettoyer les NaN
        valid_idx = ~(features.isna().any(axis=1) | np.isnan(target))
        X = features[valid_idx].values
        y = target[valid_idx]
        dates = df.index[valid_idx]

        # Fenetre adaptative : VIX aligne sur les indices valides
        vix_vals = None
        if 'VIX' in features.columns:
            vix_vals = features['VIX'][valid_idx].values

        # Demarrer apres la plus grande fenetre possible pour eviter les
        # predictions avec training set tronque
        max_anchor_window = max(w for _, w in ADAPTIVE_WINDOW_ANCHORS)
        min_start = max_anchor_window
        if len(y) < min_start + 50:
            return None

        # Rolling backtest avec purging (evite le chevauchement des targets)
        predictions = np.full(len(y), np.nan)
        purge_gap = self.horizon - 1  # 4 samples exclus pour horizon=5

        for i in range(min_start, len(y)):
            # Fenetre adaptative basee sur VIX au jour t
            if vix_vals is not None and not np.isnan(vix_vals[i]):
                window = adaptive_train_window(vix_vals[i])
            else:
                window = self.train_window
            # Ne pas depasser les donnees disponibles
            window = min(window, i - purge_gap)
            train_start = max(0, i - window)

            X_train = X[train_start:i - purge_gap]
            y_train = y[train_start:i - purge_gap]
            X_test = X[i:i+1]

            try:
                self.fit(X_train, y_train)
                predictions[i] = self.predict(X_test)[0]
            except Exception:
                continue

        # Retirer les NaN
        valid_pred = ~np.isnan(predictions)
        if valid_pred.sum() < 50:
            return None

        predictions = predictions[valid_pred]
        actuals = y[valid_pred]
        dates = dates[valid_pred]

        # Metriques
        ic, _ = spearmanr(predictions, actuals)

        pred_median = np.median(predictions)
        actual_median = np.median(actuals)
        pred_high = predictions > pred_median
        actual_high = actuals > actual_median
        hit_rate = (pred_high == actual_high).mean()

        gains = np.where(pred_high == actual_high, actuals, -actuals)
        avg_gain = gains.mean() * 100

        # Metriques par regime VIX
        regime_metrics = {}
        if vix_regime_col is not None:
            regimes = vix_regime_col[valid_idx].values[valid_pred]
            for r_val, r_name in REGIME_NAMES.items():
                mask = regimes == r_val
                n = int(mask.sum())
                if n >= 20:
                    r_ic, _ = spearmanr(predictions[mask], actuals[mask])
                    r_pred_high = predictions[mask] > pred_median
                    r_actual_high = actuals[mask] > actual_median
                    r_hr = (r_pred_high == r_actual_high).mean()
                    regime_metrics[r_name] = {'hit_rate': r_hr, 'ic': r_ic, 'n': n}
                else:
                    regime_metrics[r_name] = {'hit_rate': np.nan, 'ic': np.nan, 'n': n}

        return {
            'symbol': symbol,
            'hit_rate': hit_rate,
            'ic': ic,
            'avg_gain_bps': avg_gain,
            'n_predictions': len(predictions),
            'predictions': predictions,
            'actuals': actuals,
            'dates': dates,
            'regime_metrics': regime_metrics,
        }

# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

def main():
    """Demo du modele HAR-RV (Ridge par defaut, backward compatible)."""
    print("=" * 70)
    print("        MODELE HAR-RV - PREDICTION DE VOLATILITE")
    print("=" * 70)

    model = HARRVModel(model_type='ridge', horizon=5, train_window=252)

    print("\nChargement du VIX...", end=" ")
    vix = model.get_vix()
    print("OK" if vix is not None else "Non disponible")

    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'MA']

    print("Chargement des donnees...", end=" ")
    all_data = {}
    for s in stocks:
        d = model.get_stock_data(s)
        if d is not None:
            all_data[s] = d
    print(f"{len(all_data)} stocks")

    print("Calcul des features cross-sectionnelles...", end=" ")
    cs_features = build_cross_sectional_features(all_data)
    print(f"OK ({len(cs_features)} stocks)")

    print(f"\nBacktest sur {len(stocks)} actions (Ridge):")
    print("-" * 70)

    results = []
    for symbol in stocks:
        print(f"\n{symbol}...", end=" ")
        result = model.backtest(symbol, vix, cs_features.get(symbol))

        if result is not None:
            results.append(result)
            print(f"Hit Rate={result['hit_rate']:.1%} | IC={result['ic']:.3f} | Gain={result['avg_gain_bps']:+.2f}bps")
            rm = result.get('regime_metrics', {})
            if rm:
                parts = []
                for rname in ('Low', 'Medium', 'High'):
                    if rname in rm and not np.isnan(rm[rname]['ic']):
                        parts.append(f"{rname}: HR={rm[rname]['hit_rate']:.0%} IC={rm[rname]['ic']:.2f} (n={rm[rname]['n']})")
                    elif rname in rm:
                        parts.append(f"{rname}: n={rm[rname]['n']} (insuf.)")
                if parts:
                    print(f"         Regimes VIX: {' | '.join(parts)}")
        else:
            print("Echec")

    if results:
        print("\n" + "=" * 70)
        print("RESUME")
        print("=" * 70)

        avg_hr = np.mean([r['hit_rate'] for r in results])
        avg_ic = np.mean([r['ic'] for r in results])
        avg_gain = np.mean([r['avg_gain_bps'] for r in results])

        print(f"\n  Hit Rate moyen:  {avg_hr:.1%}")
        print(f"  IC moyen:        {avg_ic:.3f}")
        print(f"  Gain moyen:      {avg_gain:+.2f} bps")

        importance = model.get_feature_importance()
        if importance:
            print(f"\n  Features importance:")
            for name, val in importance.items():
                print(f"    {name:14s}: {val:+.4f}")

    return results


def run_benchmark():
    """
    Compare v1 (6 features originales HAR-RV) vs v2 (14 features etendues)
    sur XGBoost avec les memes hyperparametres et le meme walk-forward.

    Evalue sur la periode 2024-01-01 a aujourd'hui.
    Sauvegarde dans results/benchmark_v1_v2.csv.
    """
    STOCKS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
              "JPM", "MA", "V", "UNH", "JNJ", "XOM", "BRK-B", "COST"]

    V1_FEATURES = ["RV_w", "RV_m", "RV_q", "RV_neg_w", "J_w", "VIX"]

    TEST_START = pd.Timestamp("2024-01-01")

    print("=" * 110)
    print("          BENCHMARK v1 (6 features HAR-RV) vs v2 (16 features etendues) — XGBoost")
    print("=" * 110)

    model = HARRVModel(model_type='xgboost', horizon=5, train_window=252)
    vix = model.get_vix()
    print(f"VIX: {'OK' if vix is not None else 'Non disponible'}")
    print(f"Univers: {len(STOCKS)} stocks | Periode test: {TEST_START.date()} -> aujourd'hui")
    print(f"Hyperparametres XGBoost: max_depth=3, min_child_weight=5, subsample=0.7")

    print("Chargement des donnees...", end=" ")
    all_data = {}
    for s in STOCKS:
        d = model.get_stock_data(s)
        if d is not None:
            all_data[s] = d
    print(f"{len(all_data)} stocks")

    print("Calcul des features cross-sectionnelles...", end=" ")
    cs_features = build_cross_sectional_features(all_data)
    print(f"OK ({len(cs_features)} stocks)")
    print("=" * 110)

    rows = []

    for symbol in STOCKS:
        print(f"\n  {symbol}...", end=" ")

        if symbol not in all_data:
            continue
        df = all_data[symbol]

        features_all = model.create_features(df, vix, cs_features.get(symbol))
        target = model.create_target(df)

        # VIX regime pour evaluation conditionnelle (calcule dans create_features)
        vix_regime_eval = getattr(model, '_vix_regime', None)

        for version, feat_cols in [('v1', V1_FEATURES), ('v2', list(features_all.columns))]:
            available_cols = [c for c in feat_cols if c in features_all.columns]
            features = features_all[available_cols]

            # Nettoyer les NaN
            valid_idx = ~(features.isna().any(axis=1) | np.isnan(target))
            X = features[valid_idx].values
            y = target[valid_idx]
            dates = df.index[valid_idx]

            # VIX pour fenetre adaptative
            vix_vals = None
            if 'VIX' in features_all.columns:
                vix_vals = features_all['VIX'][valid_idx].values

            max_anchor_window = max(w for _, w in ADAPTIVE_WINDOW_ANCHORS)
            min_start = max_anchor_window
            if len(y) < min_start + 50:
                continue

            # Walk-forward backtest avec purging + fenetre adaptative
            predictions = np.full(len(y), np.nan)
            purge_gap = model.horizon - 1
            for i in range(min_start, len(y)):
                if vix_vals is not None and not np.isnan(vix_vals[i]):
                    window = adaptive_train_window(vix_vals[i])
                else:
                    window = model.train_window
                window = min(window, i - purge_gap)
                train_start = max(0, i - window)

                X_train = X[train_start:i - purge_gap]
                y_train = y[train_start:i - purge_gap]
                X_test = X[i:i+1]
                try:
                    model.fit(X_train, y_train)
                    predictions[i] = model.predict(X_test)[0]
                except Exception:
                    continue

            # Filtrer: predictions valides ET periode de test
            valid_pred = ~np.isnan(predictions)
            test_mask = dates >= TEST_START
            combined = valid_pred & test_mask

            if combined.sum() < 30:
                continue

            preds = predictions[combined]
            acts = y[combined]

            # Metriques globales
            ic, _ = spearmanr(preds, acts)
            pred_med = np.median(preds)
            act_med = np.median(acts)
            pred_high = preds > pred_med
            act_high = acts > act_med
            hr = (pred_high == act_high).mean()

            row = {
                'symbol': symbol,
                'version': version,
                'IC': round(ic, 4),
                'HitRate': round(hr, 4),
                'n': int(combined.sum()),
            }

            # Metriques par regime
            if vix_regime_eval is not None:
                regimes = vix_regime_eval[valid_idx].values[combined]
                for r_val, r_name in REGIME_NAMES.items():
                    mask = regimes == r_val
                    n_r = int(mask.sum())
                    if n_r >= 15:
                        r_ic, _ = spearmanr(preds[mask], acts[mask])
                        r_pred_high = preds[mask] > pred_med
                        r_act_high = acts[mask] > act_med
                        r_hr = (r_pred_high == r_act_high).mean()
                        row[f'IC_{r_name}'] = round(r_ic, 4)
                        row[f'HR_{r_name}'] = round(r_hr, 4)
                        row[f'n_{r_name}'] = n_r
                    else:
                        row[f'IC_{r_name}'] = np.nan
                        row[f'HR_{r_name}'] = np.nan
                        row[f'n_{r_name}'] = n_r

            rows.append(row)

        # Affichage inline
        v1_row = next((r for r in rows if r['symbol'] == symbol and r['version'] == 'v1'), None)
        v2_row = next((r for r in rows if r['symbol'] == symbol and r['version'] == 'v2'), None)
        if v1_row and v2_row:
            d_ic = v2_row['IC'] - v1_row['IC']
            d_hr = v2_row['HitRate'] - v1_row['HitRate']
            print(f"v1: IC={v1_row['IC']:.3f} HR={v1_row['HitRate']:.1%}  |  "
                  f"v2: IC={v2_row['IC']:.3f} HR={v2_row['HitRate']:.1%}  |  "
                  f"delta: IC={d_ic:+.3f} HR={d_hr:+.1%}")
        elif v1_row:
            print(f"v1: IC={v1_row['IC']:.3f} HR={v1_row['HitRate']:.1%}  |  v2: echec")
        elif v2_row:
            print(f"v1: echec  |  v2: IC={v2_row['IC']:.3f} HR={v2_row['HitRate']:.1%}")

    if not rows:
        print("\nAucun resultat.")
        return None

    df_out = pd.DataFrame(rows)

    # --- Tableau detaille ---
    regime_cols = ['IC_Low', 'HR_Low', 'IC_Medium', 'HR_Medium', 'IC_High', 'HR_High']
    has_regime = any(c in df_out.columns for c in regime_cols)

    print("\n\n" + "=" * 110)
    print("                              RESULTATS DETAILLES")
    print("=" * 110)

    header = f"{'Ticker':<8} {'Ver':>3} {'IC':>7} {'HR':>7} {'n':>5}"
    if has_regime:
        header += f"  {'IC_Low':>7} {'HR_Low':>7} {'IC_Med':>7} {'HR_Med':>7} {'IC_Hi':>7} {'HR_Hi':>7}"
    print(header)
    print("-" * len(header))

    for symbol in STOCKS:
        for ver in ('v1', 'v2'):
            r = df_out[(df_out['symbol'] == symbol) & (df_out['version'] == ver)]
            if r.empty:
                continue
            r = r.iloc[0]
            line = f"{r['symbol']:<8} {r['version']:>3} {r['IC']:>7.3f} {r['HitRate']:>6.1%} {r['n']:>5}"
            if has_regime:
                for reg in ('Low', 'Medium', 'High'):
                    ic_val = r.get(f'IC_{reg}', np.nan)
                    hr_val = r.get(f'HR_{reg}', np.nan)
                    ic_s = f"{ic_val:>7.3f}" if not np.isnan(ic_val) else "    n/a"
                    hr_s = f"{hr_val:>6.1%}" if not np.isnan(hr_val) else "    n/a"
                    line += f"  {ic_s} {hr_s}"
            print(line)

    # --- Moyennes cross-sectionnelles ---
    print("-" * len(header))

    for ver in ('v1', 'v2'):
        sub = df_out[df_out['version'] == ver]
        if sub.empty:
            continue
        line = f"{'MEAN':<8} {ver:>3} {sub['IC'].mean():>7.3f} {sub['HitRate'].mean():>6.1%} {int(sub['n'].mean()):>5}"
        if has_regime:
            for reg in ('Low', 'Medium', 'High'):
                ic_col = f'IC_{reg}'
                hr_col = f'HR_{reg}'
                ic_m = sub[ic_col].mean() if ic_col in sub.columns else np.nan
                hr_m = sub[hr_col].mean() if hr_col in sub.columns else np.nan
                ic_s = f"{ic_m:>7.3f}" if not np.isnan(ic_m) else "    n/a"
                hr_s = f"{hr_m:>6.1%}" if not np.isnan(hr_m) else "    n/a"
                line += f"  {ic_s} {hr_s}"
        print(line)

    # Delta
    v1_mean = df_out[df_out['version'] == 'v1']
    v2_mean = df_out[df_out['version'] == 'v2']
    if not v1_mean.empty and not v2_mean.empty:
        d_ic = v2_mean['IC'].mean() - v1_mean['IC'].mean()
        d_hr = v2_mean['HitRate'].mean() - v1_mean['HitRate'].mean()
        line = f"{'DELTA':<8} {'':>3} {d_ic:>+7.3f} {d_hr:>+6.1%} {'':>5}"
        if has_regime:
            for reg in ('Low', 'Medium', 'High'):
                ic1 = v1_mean[f'IC_{reg}'].mean() if f'IC_{reg}' in v1_mean.columns else np.nan
                ic2 = v2_mean[f'IC_{reg}'].mean() if f'IC_{reg}' in v2_mean.columns else np.nan
                hr1 = v1_mean[f'HR_{reg}'].mean() if f'HR_{reg}' in v1_mean.columns else np.nan
                hr2 = v2_mean[f'HR_{reg}'].mean() if f'HR_{reg}' in v2_mean.columns else np.nan
                d_ic_r = ic2 - ic1 if not (np.isnan(ic1) or np.isnan(ic2)) else np.nan
                d_hr_r = hr2 - hr1 if not (np.isnan(hr1) or np.isnan(hr2)) else np.nan
                ic_s = f"{d_ic_r:>+7.3f}" if not np.isnan(d_ic_r) else "    n/a"
                hr_s = f"{d_hr_r:>+6.1%}" if not np.isnan(d_hr_r) else "    n/a"
                line += f"  {ic_s} {hr_s}"
        print(line)

    # --- Sauvegarde CSV ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "benchmark_v1_v2.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\n  Resultats sauvegardes: {csv_path}")

    return df_out


def _update_xgboost_defaults(best_params):
    """Met a jour les defaults XGBoost dans _create_model() du fichier source.

    Ne modifie que les lignes p.get('param', default) situees dans le bloc
    ``elif self.model_type == 'xgboost'`` pour eviter de toucher Random Forest
    ou le Stacking.
    """
    import re as _re

    filepath = os.path.abspath(__file__)
    with open(filepath, 'r') as f:
        lines = f.readlines()

    in_xgb_section = False
    new_lines = []

    for line in lines:
        if "elif self.model_type == 'xgboost':" in line:
            in_xgb_section = True
        elif in_xgb_section and (line.strip().startswith('elif ') or line.strip().startswith('else:')):
            in_xgb_section = False

        if in_xgb_section:
            for param, val in best_params.items():
                key = f"p.get('{param}',"
                if key in line:
                    line = _re.sub(
                        rf"p\.get\('{param}',\s*[\d.]+\)",
                        f"p.get('{param}', {val})",
                        line
                    )

        new_lines.append(line)

    with open(filepath, 'w') as f:
        f.writelines(new_lines)


def tune_xgboost():
    """
    Optimise les hyperparametres XGBoost via RandomizedSearchCV + TimeSeriesSplit.

    Corrections v2 par rapport a v1:
      1. Scoring = Spearman IC (au lieu de neg_MSE) — aligne sur la metrique d'evaluation
      2. Couplage lr / n_estimators: n_estimators retire de la grille, calcule automatiquement
         apres selection du best_lr : n_estimators = max(200, int(10 / best_lr))
      3. Tuning multi-stocks (AAPL, MSFT, NVDA, JPM, META) au lieu d'un seul pivot

    Protocole:
      - CV : TimeSeriesSplit(n_splits=5) sur chaque stock
      - Methode : RandomizedSearchCV(n_iter=50, scoring=spearman_ic)
      - Score CV final = moyenne des Spearman IC sur les 5 stocks
      - Compare baseline vs tuned en walk-forward sur la periode 2024-01-01+
      - Met a jour les defaults dans _create_model() si amelioration IC moyenne > 2%
      - Sauvegarde dans results/tuning_results_v2.json
    """
    import json
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import make_scorer

    # --- Scorer custom : Spearman IC (robuste aux predictions constantes) ---
    def spearman_ic(y_true, y_pred):
        if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
            return 0.0
        corr = spearmanr(y_true, y_pred)[0]
        return corr if np.isfinite(corr) else 0.0

    ic_scorer = make_scorer(spearman_ic)

    TUNING_STOCKS = ["AAPL", "MSFT", "NVDA", "JPM", "META"]
    TEST_START = pd.Timestamp("2024-01-01")

    print("=" * 80)
    print("    TUNING XGBOOST v2 — Spearman IC + lr/n_est couple + multi-stocks")
    print("=" * 80)

    # --- Chargement des donnees pour tous les stocks ---
    model = HARRVModel(model_type='xgboost', horizon=5, train_window=252)
    vix = model.get_vix()

    print("Chargement des donnees...", end=" ")
    all_data = {}
    for s in TUNING_STOCKS:
        d = model.get_stock_data(s)
        if d is not None:
            all_data[s] = d
    print(f"{len(all_data)} stocks")

    print("Calcul des features cross-sectionnelles...", end=" ")
    cs_features = build_cross_sectional_features(all_data)
    print(f"OK ({len(cs_features)} stocks)")

    stock_data = {}  # {symbol: (X, y, dates, vix_vals)}
    for symbol in TUNING_STOCKS:
        if symbol not in all_data:
            print(f"  {symbol}: donnees non disponibles, skip.")
            continue
        df = all_data[symbol]
        features = model.create_features(df, vix, cs_features.get(symbol))
        target = model.create_target(df)
        valid_idx = ~(features.isna().any(axis=1) | np.isnan(target))
        X = features[valid_idx].values
        y = target[valid_idx]
        dates = df.index[valid_idx]
        vix_vals = features['VIX'][valid_idx].values if 'VIX' in features.columns else None
        if len(y) >= model.train_window + 50:
            stock_data[symbol] = (X, y, dates, vix_vals)
            print(f"  {symbol}: {len(y)} echantillons valides")
        else:
            print(f"  {symbol}: donnees insuffisantes ({len(y)}), skip.")

    if not stock_data:
        print("Erreur: aucun stock disponible pour le tuning.")
        return None

    print(f"\nStocks retenus: {list(stock_data.keys())} ({len(stock_data)})")
    print(f"Features: {len(model.feature_names)}")

    # --- Defaults actuels (lus depuis _create_model) ---
    current_params = {
        'max_depth': 3,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0,
        'reg_lambda': 1.5,
    }

    # --- Grille de recherche (sans n_estimators, couple avec lr) ---
    param_distributions = {
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'min_child_weight': [3, 5, 7, 10],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2],
    }

    # CV purge : gap de horizon-1 entre train et test a chaque fold
    def _purged_ts_split(n_samples, n_splits=5, purge_gap=4):
        fold_size = n_samples // (n_splits + 1)
        for i in range(n_splits):
            train_end = fold_size * (i + 1)
            test_start = train_end + purge_gap
            test_end = train_end + fold_size
            if test_start >= test_end or test_end > n_samples:
                continue
            yield np.arange(0, train_end), np.arange(test_start, min(test_end, n_samples))

    print(f"\nRandomizedSearchCV: n_iter=50, cv=PurgedTimeSeriesSplit(5, gap=4), scoring=Spearman IC")
    print("Recherche sur chaque stock...")

    # --- Phase 1 : RandomizedSearchCV par stock, puis vote par moyenne ---
    all_cv_results = []  # liste de (best_params, best_score) par stock

    for symbol, (X, y, dates, _vix) in stock_data.items():
        print(f"  {symbol}...", end=" ", flush=True)
        xgb = XGBRegressor(verbosity=0, random_state=42, n_estimators=200)
        cv_splits = list(_purged_ts_split(len(X), n_splits=5, purge_gap=4))

        search = RandomizedSearchCV(
            xgb,
            param_distributions,
            n_iter=50,
            scoring=ic_scorer,
            cv=cv_splits,
            random_state=42,
            n_jobs=1,
            verbose=0,
        )
        search.fit(X, y)
        all_cv_results.append((search.best_params_, search.best_score_))
        print(f"IC_cv={search.best_score_:.4f}")

    # --- Aggregation : parametre le plus frequent (mode) pour chaque hp ---
    from collections import Counter

    best_params = {}
    for param in param_distributions.keys():
        values = [res[0][param] for res in all_cv_results]
        # Mode (valeur la plus frequente parmi les 5 stocks)
        counter = Counter(values)
        best_params[param] = counter.most_common(1)[0][0]

    # --- Couplage lr / n_estimators ---
    best_lr = best_params['learning_rate']
    best_n_est = max(200, int(10 / best_lr))
    best_params['n_estimators'] = best_n_est

    mean_cv_score = np.mean([r[1] for r in all_cv_results])

    print(f"\nMeilleurs hyperparametres (vote majoritaire sur {len(stock_data)} stocks):")
    print("-" * 60)
    for k in sorted(best_params):
        v = best_params[k]
        cur = current_params.get(k, '—')
        changed = " <-- CHANGE" if v != cur else ""
        print(f"  {k:<20s}: {v}{changed}  (avant: {cur})")
    print(f"\n  Couplage lr/n_estimators: lr={best_lr} -> n_estimators={best_n_est}")
    print(f"  CV Score moyen (Spearman IC): {mean_cv_score:.4f}")

    # --- Phase 2 : Walk-forward comparison sur tous les stocks ---
    print(f"\n{'=' * 80}")
    print(f"  COMPARAISON WALK-FORWARD sur {len(stock_data)} stocks (test: {TEST_START.date()}+)")
    print(f"{'=' * 80}")

    def _walkforward(X, y, dates, params, train_window=252, purge_gap=4,
                     vix_vals=None):
        """Walk-forward avec purging + fenetre adaptative VIX."""
        predictions = np.full(len(y), np.nan)
        min_start = max(w for _, w in ADAPTIVE_WINDOW_ANCHORS)
        for i in range(min_start, len(y)):
            if vix_vals is not None and not np.isnan(vix_vals[i]):
                window = adaptive_train_window(vix_vals[i])
            else:
                window = train_window
            window = min(window, i - purge_gap)
            t_start = max(0, i - window)

            X_tr = X[t_start:i - purge_gap]
            y_tr = y[t_start:i - purge_gap]
            X_te = X[i:i+1]
            try:
                m = XGBRegressor(verbosity=0, random_state=42, **params)
                m.fit(X_tr, y_tr)
                predictions[i] = m.predict(X_te)[0]
            except Exception:
                continue
        return predictions

    def _compute_metrics(preds, y, dates, test_start):
        """Calcule IC et HitRate sur la periode de test."""
        valid_pred = ~np.isnan(preds)
        test_mask = dates >= test_start
        combined = valid_pred & test_mask
        if combined.sum() < 30:
            return None
        p = preds[combined]
        a = y[combined]
        ic, _ = spearmanr(p, a)
        pred_med = np.median(p)
        act_med = np.median(a)
        pred_high = p > pred_med
        act_high = a > act_med
        hr = (pred_high == act_high).mean()
        return {'ic': float(ic), 'hit_rate': float(hr), 'n': int(combined.sum())}

    results_per_stock = {}

    print(f"\n  {'Stock':<8} {'Baseline IC':>12} {'Baseline HR':>12} {'Tuned IC':>10} {'Tuned HR':>10} {'Delta IC':>10}")
    print(f"  {'-' * 66}")

    for symbol, (X, y, dates, vix_v) in stock_data.items():
        preds_base = _walkforward(X, y, dates, current_params, vix_vals=vix_v)
        preds_tuned = _walkforward(X, y, dates, best_params, vix_vals=vix_v)

        base_m = _compute_metrics(preds_base, y, dates, TEST_START)
        tuned_m = _compute_metrics(preds_tuned, y, dates, TEST_START)

        results_per_stock[symbol] = {'baseline': base_m, 'tuned': tuned_m}

        if base_m and tuned_m:
            d_ic = tuned_m['ic'] - base_m['ic']
            print(f"  {symbol:<8} {base_m['ic']:>12.4f} {base_m['hit_rate']:>11.1%} "
                  f"{tuned_m['ic']:>10.4f} {tuned_m['hit_rate']:>9.1%} {d_ic:>+10.4f}")
        elif base_m:
            print(f"  {symbol:<8} {base_m['ic']:>12.4f} {base_m['hit_rate']:>11.1%} {'n/a':>10} {'n/a':>10} {'n/a':>10}")

    # --- Moyennes cross-sectionnelles ---
    base_ics = [v['baseline']['ic'] for v in results_per_stock.values() if v['baseline']]
    tuned_ics = [v['tuned']['ic'] for v in results_per_stock.values() if v['tuned']]
    base_hrs = [v['baseline']['hit_rate'] for v in results_per_stock.values() if v['baseline']]
    tuned_hrs = [v['tuned']['hit_rate'] for v in results_per_stock.values() if v['tuned']]

    mean_base_ic = np.mean(base_ics) if base_ics else 0
    mean_tuned_ic = np.mean(tuned_ics) if tuned_ics else 0
    mean_base_hr = np.mean(base_hrs) if base_hrs else 0
    mean_tuned_hr = np.mean(tuned_hrs) if tuned_hrs else 0
    d_ic_mean = mean_tuned_ic - mean_base_ic
    d_hr_mean = mean_tuned_hr - mean_base_hr

    print(f"  {'-' * 66}")
    print(f"  {'MEAN':<8} {mean_base_ic:>12.4f} {mean_base_hr:>11.1%} "
          f"{mean_tuned_ic:>10.4f} {mean_tuned_hr:>9.1%} {d_ic_mean:>+10.4f}")

    pct_improvement = d_ic_mean / abs(mean_base_ic) * 100 if mean_base_ic != 0 else 0
    print(f"\n  Improvement moyen: {pct_improvement:+.1f}% IC relatif")

    should_update = pct_improvement > 2.0

    if should_update:
        print(f"\n  Amelioration > 2% ({pct_improvement:+.1f}%) -> Mise a jour des defaults...")
        _update_xgboost_defaults(best_params)
        print("  Defaults XGBoost mis a jour dans _create_model().")
    else:
        print(f"\n  Amelioration <= 2% ({pct_improvement:+.1f}%) -> Defaults inchanges.")

    # --- Sauvegarde JSON ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "tuning_results_v2.json")

    # Convertir numpy types pour json.dump
    bp_clean = {}
    for k, v in best_params.items():
        if isinstance(v, (np.integer,)):
            bp_clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            bp_clean[k] = float(v)
        else:
            bp_clean[k] = v

    output = {
        'tuning_stocks': list(stock_data.keys()),
        'scoring': 'spearman_ic',
        'current_params': current_params,
        'best_params': bp_clean,
        'lr_n_estimators_rule': f"n_estimators = max(200, int(10 / {best_lr})) = {best_n_est}",
        'mean_cv_score_spearman_ic': round(float(mean_cv_score), 4),
        'walkforward_results': {
            sym: {
                'baseline': v['baseline'],
                'tuned': v['tuned'],
            }
            for sym, v in results_per_stock.items()
        },
        'mean_baseline_ic': round(float(mean_base_ic), 4),
        'mean_tuned_ic': round(float(mean_tuned_ic), 4),
        'improvement_pct': round(float(pct_improvement), 2),
        'updated_file': bool(should_update),
    }
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Resultats sauvegardes: {json_path}")

    return best_params


if __name__ == "__main__":
    main()
