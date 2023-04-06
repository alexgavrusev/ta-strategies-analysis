import backtrader as bt

from .base_strategy import BaseStrategy


class TwoMACrossoverStrategy(BaseStrategy):
    params = (
        ("f_p", 5),
        ("s_p", 20),
        ("ma_ind", bt.indicators.SMA),
    )

    def __init__(self):
        super().__init__()

        self.fast_ma = self.params.ma_ind(self.data, period=self.params.f_p)
        self.slow_ma = self.params.ma_ind(self.data, period=self.params.s_p)

    def next(self):
        if not self.position:
            if self.fast_ma[0] > self.slow_ma[0]:
                self.pending_order = self.buy(price=self.close)
        else:
            if (
                # regular exit cond
                self.fast_ma[0] < self.slow_ma[0]
                or
                # stop loss
                self.close < self.position.price * (1 - self.stop_loss_pct)
            ):
                self.pending_order = self.sell(price=self.close)
