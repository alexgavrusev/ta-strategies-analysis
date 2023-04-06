import backtrader as bt

from .base_strategy import BaseStrategy


class MAPriceCrossoverStrategy(BaseStrategy):
    params = (
        ("p", 5),
        ("ma_ind", bt.indicators.SMA),
    )

    def __init__(self):
        super().__init__()

        self.ma = self.params.ma_ind(self.data, period=self.params.p)

    def next(self):
        if not self.position:
            if self.ma[0] > self.close:
                self.pending_order = self.buy(price=self.close)
        else:
            if (
                # regular exit cond
                self.ma[0] < self.close
                or
                # stop loss
                self.close < self.position.price * (1 - self.stop_loss_pct)
            ):
                self.pending_order = self.sell(price=self.close)
