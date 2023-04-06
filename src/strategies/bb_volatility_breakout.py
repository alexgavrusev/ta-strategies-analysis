import backtrader as bt

from .base_strategy import BaseStrategy


class BBVolatilityBreakoutStrategy(BaseStrategy):
    params = (
        ("p", 20),
        ("df", 2.0),
        ("lb_p", 180),
    )

    def __init__(self):
        super().__init__()

        self.bb = bt.indicators.BollingerBands(
            self.data,
            period=self.params.p,
            devfactor=self.params.df,
        )

        self.band_width = (self.bb.lines.top - self.bb.lines.bot) / self.bb.lines.mid

        self.min_band_width = bt.indicators.Lowest(
            self.band_width,
            period=self.params.lb_p,
        )

    def next(self):
        if not self.position:
            if (
                # closes above the top line
                self.close > self.bb.lines.top[0]
                and
                # band width is at the historical minimum
                self.band_width[0] <= self.min_band_width[0]
            ):
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
