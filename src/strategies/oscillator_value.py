import backtrader as bt

from .base_strategy import BaseStrategy


class OscillatorValueStrategy(BaseStrategy):
    params = (
        ("ind", bt.indicators.RSI),
        ("p", 14),
        ("ob", 70),
        ("os", 30),
    )

    def __init__(self):
        super().__init__()

        self.indicator = self.params.ind(self.data, period=self.params.p)

    def next(self):
        if self.pending_order:
            return

        if not self.position:
            if self.indicator[0] < self.params.os:
                self.pending_order = self.buy(price=self.close)
        else:
            if (
                # regular exit cond
                self.indicator[0] > self.params.ob
                or
                # stop loss
                self.close < self.position.price * (1 - self.stop_loss_pct)
            ):
                self.pending_order = self.sell(price=self.close)
