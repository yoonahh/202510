from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ASSET_COL = ("Price", "Ticker")
ANNUALIZATION_FACTOR = 252


@dataclass
class EfficientFrontierResult:
    target_returns: np.ndarray
    risks: np.ndarray
    weights: np.ndarray


def load_close_prices(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, header=[0, 1])
    clean = raw[raw[ASSET_COL] != "Date"].copy()
    clean[ASSET_COL] = pd.to_datetime(clean[ASSET_COL])
    clean = clean.set_index(ASSET_COL)
    clean.index.name = "Date"
    close = clean["Close"].astype(float)
    return close.sort_index()


def compute_statistics(close: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    daily_returns = close.pct_change(fill_method=None).dropna()
    expected_returns = daily_returns.mean() * ANNUALIZATION_FACTOR
    covariance = daily_returns.cov() * ANNUALIZATION_FACTOR
    return daily_returns, expected_returns, covariance


def efficient_frontier(expected_returns: pd.Series, covariance: pd.DataFrame, n_points: int = 60) -> EfficientFrontierResult:
    mu = expected_returns.to_numpy()
    cov = covariance.to_numpy()
    ones = np.ones(len(mu))
    inv_cov = np.linalg.inv(cov)

    A = ones @ inv_cov @ ones
    B = ones @ inv_cov @ mu
    C = mu @ inv_cov @ mu
    D = A * C - B ** 2

    min_return = mu.min()
    max_return = mu.max()
    target_returns = np.linspace(min_return, max_return, n_points)
    risks = []
    weights = []
    for target in target_returns:
        lambda1 = (C - B * target) / D
        lambda2 = (A * target - B) / D
        weight = inv_cov @ (lambda1 * ones + lambda2 * mu)
        risk = np.sqrt(weight @ cov @ weight)
        weights.append(weight)
        risks.append(risk)

    return EfficientFrontierResult(
        target_returns=target_returns,
        risks=np.asarray(risks),
        weights=np.asarray(weights),
    )


def make_frontier_figure(frontier: EfficientFrontierResult, expected_returns: pd.Series, covariance: pd.DataFrame) -> go.Figure:
    asset_volatility = np.sqrt(np.diag(covariance))
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frontier.risks,
            y=frontier.target_returns,
            mode="lines",
            line=dict(color="#0066cc", width=3),
            name="효율적 투자선",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=asset_volatility,
            y=expected_returns.to_numpy(),
            mode="markers+text",
            marker=dict(size=12, color="#ff7f0e"),
            text=expected_returns.index.tolist(),
            textposition="top center",
            name="개별 자산",
        )
    )

    fig.update_layout(
        title="삼성전자 · 애플 · 엔비디아 포트폴리오 효율적 투자선",
        xaxis_title="연간 변동성 (σ)",
        yaxis_title="연간 기대수익률 (μ)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def make_statistics_table(expected_returns: pd.Series, covariance: pd.DataFrame) -> str:
    volatility = np.sqrt(np.diag(covariance))
    summary = pd.DataFrame(
        {
            "연간 기대수익률": expected_returns,
            "연간 변동성": pd.Series(volatility, index=expected_returns.index),
        }
    )
    summary = summary.sort_index()
    formatted = summary.copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].map(lambda x: f"{x:.2%}")
    return formatted.to_html(classes="metrics-table", border=0, justify="center")


def build_page(csv_path: Path, output_path: Path) -> None:
    close = load_close_prices(csv_path)
    daily_returns, expected_returns, covariance = compute_statistics(close)
    frontier = efficient_frontier(expected_returns, covariance)
    fig = make_frontier_figure(frontier, expected_returns, covariance)

    dataset_period = f"{close.index.min():%Y-%m-%d} ~ {close.index.max():%Y-%m-%d}"
    num_records = len(close)
    summary_table_html = make_statistics_table(expected_returns, covariance)

    fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    html = f"""
<!DOCTYPE html>
<html lang=\"ko\">
<head>
    <meta charset=\"utf-8\">
    <title>효율적 투자선 대시보드</title>
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <style>
        :root {{
            color-scheme: light dark;
            font-family: 'Pretendard', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background-color: #f5f7fa;
            color: #1f2933;
        }}
        body {{
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }}
        header {{
            background: linear-gradient(135deg, #0f62fe, #3ddbd9);
            color: white;
            padding: 48px 24px;
            text-align: center;
        }}
        header h1 {{
            margin: 0;
            font-size: 2.5rem;
        }}
        header p {{
            margin: 12px 0 0;
            font-size: 1.1rem;
        }}
        main {{
            max-width: 960px;
            margin: -32px auto 48px;
            padding: 0 24px 48px;
        }}
        section {{
            background: white;
            border-radius: 18px;
            padding: 32px;
            box-shadow: 0 12px 40px rgba(15, 98, 254, 0.12);
            margin-top: 32px;
        }}
        h2 {{
            margin-top: 0;
            font-size: 1.75rem;
            color: #0b3d91;
        }}
        ul {{
            padding-left: 20px;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
            font-size: 0.95rem;
        }}
        .metrics-table th,
        .metrics-table td {{
            border: 1px solid #d9e2ec;
            padding: 12px 16px;
            text-align: right;
        }}
        .metrics-table th {{
            background-color: #f0f4f8;
            text-align: center;
        }}
        .chart-container {{
            margin-top: 16px;
        }}
        footer {{
            text-align: center;
            padding: 32px;
            color: #52606d;
        }}
        @media (max-width: 768px) {{
            header h1 {{
                font-size: 2rem;
            }}
            section {{
                padding: 24px;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>세 종목 포트폴리오 효율적 투자선</h1>
        <p>삼성전자, 애플, 엔비디아의 과거 일별 종가를 기반으로 효율적 투자선을 시각화한 대시보드입니다.</p>
    </header>
    <main>
        <section>
            <h2>데이터 개요</h2>
            <ul>
                <li>분석 기간: {dataset_period}</li>
                <li>거래일 수: {num_records:,d}일</li>
                <li>사용 지표: 일별 종가(USD/원 단위 혼재)</li>
            </ul>
            <p>효율적 투자선은 자산의 기대수익률과 변동성(위험) 간의 균형을 설명합니다. 해당 그래프는 과거 수익률을 바탕으로 장기 투자 관점에서 최적화된 위험-수익 조합을 제시합니다.</p>
        </section>
        <section>
            <h2>효율적 투자선</h2>
            <div class=\"chart-container\">{fig_html}</div>
        </section>
        <section>
            <h2>자산별 기대수익률 &amp; 변동성</h2>
            {summary_table_html}
            <p style=\"margin-top:16px;\">표의 수치는 일별 수익률을 기준으로 연 환산한 값입니다. 변동성은 표준편차로 표현되며, 효율적 투자선 상에서 동일한 기대수익을 제공하는 포트폴리오의 최소 위험 수준을 비교하는 기준이 됩니다.</p>
        </section>
    </main>
    <footer>
        © {pd.Timestamp.now():%Y} Efficient Frontier Dashboard
    </footer>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    build_page(Path("temp.csv"), Path("index.html"))
