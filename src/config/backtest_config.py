from dataclasses import dataclass
from typing import Any, Dict, Type

import backtrader as bt

from .strategy_class import StrategyClass


@dataclass
class BacktestConfig:
    strategy: Type[bt.Strategy]
    params: Dict[str, Any]
    ticker: str
    strategy_class: StrategyClass
