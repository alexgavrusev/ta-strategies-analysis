from typing import Dict, Optional

import backtrader as bt

from utils.output import get_strategy_name


class BaseStrategy(bt.Strategy):
    """
    The base class for all strategies

    Sets up common functionality such as
    tracking of the pending order and
    the logging function
    """

    status_dict: Dict[int, str] = {
        bt.Order.Created: "Created",
        bt.Order.Submitted: "Submitted",
        bt.Order.Accepted: "Accepted",
        bt.Order.Partial: "Partial",
        bt.Order.Completed: "Completed",
        bt.Order.Canceled: "Canceled",
        bt.Order.Rejected: "Rejected",
        bt.Order.Margin: "Margin",
    }

    exec_types_dict: Dict[int, str] = {
        bt.Order.Market: "Market",
        bt.Order.Close: "Close",
        bt.Order.Limit: "Limit",
        bt.Order.Stop: "Stop",
        bt.Order.StopLimit: "StopLimit",
        bt.Order.StopTrail: "StopTrail",
        bt.Order.StopTrailLimit: "StopTrailLimit",
        bt.Order.Historical: "Historical",
    }

    def __init__(self):
        self.name = get_strategy_name(self)
        self.returns = bt.analyzers.TimeReturn()
        self.pending_order: Optional[bt.Order] = None
        self.stop_loss_pct = 0.05

    @property
    def close(self):
        return self.data.close[0]

    def notify_order(self, order: bt.Order) -> None:
        status = (
            self.status_dict[order.status]
            if order.status in self.status_dict
            else order.status
        )
        exec_type = (
            self.exec_types_dict[order.exectype]
            if order.exectype in self.exec_types_dict
            else order.exectype
        )
        order_type = "Buy" if order.isbuy() else "Sell"

        self.log(
            f"{status} Order: Type {order_type}, "
            + f"Exec type {exec_type} Size {order.size}, Price {order.price:.5f}"
        )

        if order.status in [
            bt.Order.Submitted,
            bt.Order.Accepted,
        ]:
            self.pending_order = order

        if order.status in [
            bt.Order.Completed,
            bt.Order.Canceled,
            bt.Order.Margin,
        ]:
            self.pending_order = None

    def notify_trade(self, trade: bt.Trade) -> None:
        if trade.isclosed:
            self.log(
                f"Trade completed: PnL Gross {trade.pnl:.2f}, "
                + f"PnL Net {trade.pnlcomm:.2f}"
            )

    def log(self, message: str) -> None:
        print(f"{self.name} {self.datetime.date()} {self.datetime.time()}: {message}")
