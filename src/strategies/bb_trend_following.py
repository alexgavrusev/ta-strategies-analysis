import backtrader as bt

from .base_strategy import BaseStrategy
from .indicators.mfi import MFI


class BBTrendFollowingStrategy(BaseStrategy):
    params = (
        ("p", 20),
        ("df", 2.0),
        ("ind", MFI),
    )

    def __init__(self):
        super().__init__()

        self.bb = bt.indicators.BollingerBandsPct(
            self.data,
            period=self.params.p,
            devfactor=self.params.df,
        )

        self.oscillator = self.params.ind(self.data, period=self.params.p // 2)

    def next(self):
        if not self.position:
            if self.bb.lines.pctb[0] > 0.8 and self.oscillator[0] > 80:
                self.pending_order = self.buy(price=self.close)
        else:
            if (
                # regular exit cond
                self.close < self.bb.lines.bot[0]
                or
                # stop loss
                self.close < self.position.price * (1 - self.stop_loss_pct)
            ):
                self.pending_order = self.sell(price=self.close)
