"""Perform CAPM regression and machine-learning diagnostics with extreme event augmentation."""
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


warnings.filterwarnings("ignore", category=FutureWarning)


RISK_FREE_RATE = 0.03  # Annual risk-free rate
TRADING_DAYS_PER_YEAR = 252
DEFAULT_LAG_COUNT = 10


@dataclass
class ModelBundle:
    ols: sm.regression.linear_model.RegressionResultsWrapper
    rf: RandomForestRegressor
    svr: SVR
    gbr: GradientBoostingRegressor


@dataclass
class Metrics:
    rmse: float
    r2: float
    adjusted_r2: float
    mae: float
    medae: float
    evs: float


def get_stock_data(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """Download close prices for a ticker."""
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    return data["Close"]


def compute_excess_returns(prices: pd.Series) -> pd.Series:
    returns = prices.pct_change().dropna()
    excess = returns - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    return excess.squeeze()


def build_design_frame(nvidia: pd.Series, market: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nvidia_excess_returns": nvidia.values,
            "market_excess_returns": market.values,
        },
        index=nvidia.index,
    )


def identify_extremes(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    X_full = sm.add_constant(df["market_excess_returns"])
    ols_full = sm.OLS(df["nvidia_excess_returns"], X_full).fit()
    df = df.copy()
    df["ols_predicted"] = ols_full.predict(X_full)
    df["ols_residuals"] = df["nvidia_excess_returns"] - df["ols_predicted"]
    threshold = 2 * np.std(df["ols_residuals"])
    extreme_mask = df["ols_residuals"].abs() > threshold
    return df[extreme_mask], df[~extreme_mask], extreme_mask


def fill_extreme_points(
    df: pd.DataFrame, extreme_mask: pd.Series, lag_count: int = DEFAULT_LAG_COUNT
) -> pd.DataFrame:
    synthetic_rows = []
    for idx in df[extreme_mask].index:
        extreme_row = df.loc[idx]
        for lag in range(1, lag_count + 1):
            decay_factor = lag ** (-3)
            synthetic_rows.append(
                {
                    "timestamp": idx + pd.Timedelta(days=lag / 10),
                    "nvidia_excess_returns": extreme_row["nvidia_excess_returns"] * decay_factor,
                    "market_excess_returns": extreme_row["market_excess_returns"],
                    "ols_predicted": np.nan,
                    "ols_residuals": np.nan,
                    "synthetic": True,
                }
            )
    synthetic_df = pd.DataFrame(synthetic_rows)
    if synthetic_df.empty:
        return synthetic_df
    synthetic_df.set_index("timestamp", inplace=True)
    return synthetic_df


def train_models(data: pd.DataFrame) -> ModelBundle:
    X = sm.add_constant(data["market_excess_returns"])
    y = data["nvidia_excess_returns"]
    ols_model = sm.OLS(y, X).fit()

    X_ml = data["market_excess_returns"].values.reshape(-1, 1)
    if len(data) < 200:
        rf_params = dict(
            n_estimators=100,
            max_depth=5,
            min_samples_split=15,
            min_samples_leaf=6,
            random_state=42,
        )
    else:
        rf_params = dict(
            n_estimators=100,
            max_depth=7,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
        )
    rf_model = RandomForestRegressor(**rf_params).fit(X_ml, y.values)

    svr_model = SVR(kernel="rbf", C=100, epsilon=0.01).fit(X_ml, y.values)
    gbr_model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42
    ).fit(X_ml, y.values)

    return ModelBundle(ols=ols_model, rf=rf_model, svr=svr_model, gbr=gbr_model)


def adjusted_r2(r2: float, n: int, p: int = 1) -> float:
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def evaluate_models(data: pd.DataFrame, models: ModelBundle) -> Dict[str, Metrics]:
    X_vals = data["market_excess_returns"].values.reshape(-1, 1)
    y_true = data["nvidia_excess_returns"].values
    X_vals_sm = sm.add_constant(X_vals)

    predictions = {
        "OLS": models.ols.predict(X_vals_sm),
        "RF": models.rf.predict(X_vals),
        "SVR": models.svr.predict(X_vals),
        "GBR": models.gbr.predict(X_vals),
    }

    n = len(y_true)
    metrics = {}

    for name, pred in predictions.items():
        r2_val = r2_score(y_true, pred)
        metrics[name] = Metrics(
            rmse=float(np.sqrt(mean_squared_error(y_true, pred))),
            r2=float(r2_val),
            adjusted_r2=float(adjusted_r2(r2_val, n)),
            mae=float(mean_absolute_error(y_true, pred)),
            medae=float(median_absolute_error(y_true, pred)),
            evs=float(explained_variance_score(y_true, pred)),
        )
    return metrics


def plot_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    metrics_to_plot = ["RMSE", "R2"]
    models_list = train_df.index.tolist()
    x = np.arange(len(models_list))
    width = 0.35

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(len(metrics_to_plot) * 6, 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        ax.bar(x - width / 2, train_df[metric].values, width, label="Train", color="skyblue")
        ax.bar(x + width / 2, test_df[metric].values, width, label="Test", color="salmon")
        ax.set_xticks(x)
        ax.set_xticklabels(models_list, rotation=45)
        ax.set_title(f"Extreme Filled {metric}")
        ax.legend()

    plt.suptitle("Extreme Filled Data: Training vs Test Metrics", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def diagnostic_plots(filtered_train: pd.DataFrame, model: sm.regression.linear_model.RegressionResultsWrapper) -> None:
    X_const = sm.add_constant(filtered_train["market_excess_returns"])
    fitted_vals = model.predict(X_const)
    residuals = filtered_train["nvidia_excess_returns"] - fitted_vals

    plt.figure(figsize=(8, 6))
    sns.residplot(
        x=fitted_vals,
        y=residuals,
        lowess=True,
        line_kws={"color": "red", "lw": 1},
        scatter_kws={"alpha": 0.7},
    )
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residual vs Fitted Plot (Filtered Data)")
    plt.show()

    sm.qqplot(residuals, line="s")
    plt.title("Normal Q-Q Plot (Filtered Data)")
    plt.show()

    std_residuals = residuals / np.std(residuals)
    plt.figure(figsize=(8, 6))
    plt.scatter(fitted_vals, np.sqrt(np.abs(std_residuals)), alpha=0.7, color="blue")
    sns.regplot(
        x=fitted_vals,
        y=np.sqrt(np.abs(std_residuals)),
        scatter=False,
        ci=False,
        lowess=True,
        line_kws={"color": "red", "lw": 1},
    )
    plt.xlabel("Fitted Values")
    plt.ylabel("Sqrt(|Standardized Residuals|)")
    plt.title("Scale-Location Plot (Filtered Data)")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    sm.graphics.influence_plot(model, ax=ax, criterion="cooks")
    ax.set_title("Residuals vs Leverage (Cook's Distance) - Filtered Data")
    plt.show()


def metrics_to_dataframe(metrics: Dict[str, Metrics]) -> pd.DataFrame:
    df = pd.DataFrame([vars(m) for m in metrics.values()], index=metrics.keys())
    df.rename(
        columns={
            "rmse": "RMSE",
            "r2": "R2",
            "adjusted_r2": "Adjusted_R2",
            "mae": "MAE",
            "medae": "MedAE",
            "evs": "EVS",
        },
        inplace=True,
    )
    return df


def main() -> None:
    nvidia_prices = get_stock_data("NVDA", "2015-01-01", "2025-01-01")
    market_prices = get_stock_data("^GSPC", "2015-01-01", "2025-01-01")

    nvidia_excess = compute_excess_returns(nvidia_prices)
    market_excess = compute_excess_returns(market_prices)

    df = build_design_frame(nvidia_excess, market_excess)

    extreme_df, filtered_df, extreme_mask = identify_extremes(df)
    synthetic_df = fill_extreme_points(df, extreme_mask)
    df["synthetic"] = False

    frames = [extreme_df.assign(synthetic=False)]
    if not synthetic_df.empty:
        frames.append(synthetic_df)
    extreme_filled = pd.concat(frames).sort_index()

    if extreme_filled.empty:
        raise ValueError("No extreme observations were detected; extreme analysis is unavailable.")

    extreme_filled_train, extreme_filled_test = train_test_split(
        extreme_filled, test_size=0.2, random_state=42
    )
    filtered_train, filtered_test = train_test_split(
        filtered_df, test_size=0.2, random_state=42
    )

    extreme_models = train_models(extreme_filled_train)
    filtered_models = train_models(filtered_train)

    extreme_train_metrics = evaluate_models(extreme_filled_train, extreme_models)
    extreme_test_metrics = evaluate_models(extreme_filled_test, extreme_models)
    filtered_train_metrics = evaluate_models(filtered_train, filtered_models)
    filtered_test_metrics = evaluate_models(filtered_test, filtered_models)

    extreme_train_df = metrics_to_dataframe(extreme_train_metrics)
    extreme_test_df = metrics_to_dataframe(extreme_test_metrics)
    filtered_train_df = metrics_to_dataframe(filtered_train_metrics)
    filtered_test_df = metrics_to_dataframe(filtered_test_metrics)

    print("\nExtended Metrics - Extreme Filled Train Data:")
    print(extreme_train_df)
    print("\nExtended Metrics - Extreme Filled Test Data:")
    print(extreme_test_df)
    print("\nExtended Metrics - Filtered Train Data:")
    print(filtered_train_df)
    print("\nExtended Metrics - Filtered Test Data:")
    print(filtered_test_df)

    plot_metrics(extreme_train_df, extreme_test_df)
    diagnostic_plots(filtered_train, filtered_models.ols)


if __name__ == "__main__":
    main()
