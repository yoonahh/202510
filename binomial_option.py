"""이항(option) 가격 모형 구현 예제.

이 스크립트는 Cox-Ross-Rubinstein(CRR) 이항 트리 방법을 사용해
유럽형과 미국형 콜/풋 옵션의 이론가를 계산한다.

예시 사용법
------------
>>> pricer = BinomialOptionPricer()
>>> pricer.price(underlying=100, strike=105, rate=0.03, volatility=0.25, maturity=1.0, steps=200)
BinomialPrice(price=8.268..., option_type='call', american=False)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class BinomialPrice:
    """옵션 가격과 계산에 사용된 설정을 담는 결과 객체."""

    price: float
    option_type: Literal["call", "put"]
    american: bool


class BinomialOptionPricer:
    """CRR 이항 트리를 활용한 옵션 프라이서.

    Parameters
    ----------
    dividend_yield : float, optional
        연속 복리 배당수익률. 기본값은 0.0.
    """

    def __init__(self, *, dividend_yield: float = 0.0) -> None:
        self.dividend_yield = dividend_yield

    @staticmethod
    def _validate_inputs(option_type: Literal["call", "put"], steps: int, maturity: float, volatility: float) -> None:
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if maturity <= 0:
            raise ValueError("maturity must be positive")
        if volatility < 0:
            raise ValueError("volatility must be non-negative")

    def price(
        self,
        *,
        underlying: float,
        strike: float,
        rate: float,
        volatility: float,
        maturity: float,
        steps: int,
        option_type: Literal["call", "put"] = "call",
        american: bool = False,
    ) -> BinomialPrice:
        """이항 모형으로 옵션 이론가를 계산한다.

        Parameters
        ----------
        underlying : float
            기초자산 현재 가격 (S_0).
        strike : float
            행사가 (K).
        rate : float
            무위험이자율 (연속 복리 기준).
        volatility : float
            변동성 (연율, 예: 0.2는 20%).
        maturity : float
            만기까지의 시간 (년 단위).
        steps : int
            이항 트리 단계 수.
        option_type : {"call", "put"}, optional
            옵션 종류. 기본값은 "call".
        american : bool, optional
            미국형 옵션 여부. True이면 조기행사 가능성을 반영한다.

        Returns
        -------
        BinomialPrice
            계산된 옵션 가격과 설정을 담은 결과.
        """

        self._validate_inputs(option_type, steps, maturity, volatility)

        dt = maturity / steps
        up = np.exp(volatility * np.sqrt(dt))
        down = 1.0 / up
        discount = np.exp(-rate * dt)
        growth = np.exp((rate - self.dividend_yield) * dt)
        prob_up = (growth - down) / (up - down)

        if not (0.0 <= prob_up <= 1.0):
            raise ValueError("Risk-neutral probability out of bounds; adjust inputs or increase steps.")

        # 만기 시 기초자산 가격 벡터 (상승 k회, 하락 steps-k회)
        up_counts = np.arange(steps, -1, -1)
        down_counts = np.arange(0, steps + 1)
        terminal_prices = underlying * (up ** up_counts) * (down ** down_counts)

        if option_type == "call":
            option_values = np.maximum(terminal_prices - strike, 0.0)
        else:
            option_values = np.maximum(strike - terminal_prices, 0.0)

        # 후방 유도: 위쪽 노드 값은 option_values[:-1], 아래쪽은 option_values[1:]
        for step in range(steps - 1, -1, -1):
            option_values = discount * (prob_up * option_values[:-1] + (1.0 - prob_up) * option_values[1:])
            if american:
                # 현재 단계의 기초자산 가격 계산 후 조기행사 가치와 비교
                up_counts = np.arange(step, -1, -1)
                down_counts = np.arange(0, step + 1)
                current_prices = underlying * (up ** up_counts) * (down ** down_counts)
                if option_type == "call":
                    intrinsic = np.maximum(current_prices - strike, 0.0)
                else:
                    intrinsic = np.maximum(strike - current_prices, 0.0)
                option_values = np.maximum(option_values, intrinsic)

        return BinomialPrice(price=float(option_values[0]), option_type=option_type, american=american)


if __name__ == "__main__":
    pricer = BinomialOptionPricer()
    result = pricer.price(
        underlying=100.0,
        strike=105.0,
        rate=0.03,
        volatility=0.25,
        maturity=1.0,
        steps=200,
        option_type="call",
        american=False,
    )
    print(result)
