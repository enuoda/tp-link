# #! /bin/bash
# # Algorithms to use with our trading partner
# # All are implemented with the backtrader API

# # Sam Dawley
# # 08/2025

# import os
# from datetime import datetime
# from typing import Tuple

# import backtrader as bt
# import backtrader.indicators as btind

# # ==================================================
# # ALGORITHMS WHICH WORK WITH OUR TRADING PARTNER
# # ==================================================


# # class SMACrossover(bt.Strategy):
# #     """
# #     This is a long-only strategy which operates on a moving average cross
# #     Implementation taken (for testing purposes) from:
# #         https://github.com/mementum/backtrader/blob/master/backtrader/strategies/sma_crossover.py

# #     Buy Logic:
# #     ----------
# #         - No position is open on the data
# #         - The ``fast`` moving averagecrosses over the ``slow`` strategy to the upside.

# #     Sell Logic:
# #     -----------
# #         - A position exists on the data
# #         - The ``fast`` moving average crosses over the ``slow`` strategy to the downside

# #     Order Execution Type:
# #     ---------------------
# #         - Market
# #     """

# #     alias = ("SMA_Crossover",)
# #     params = (("fast", 10), ("slow", 30), ("_moveav", btind.MovAv.SMA))

# #     def __init__(self):
# #         self.sma_fast = self.p_moveav(period=self.p_fast)
# #         self.sma_slow = self.p_moveav(period=self.p.slow)

# #         self.buy_signal = btind.CrossOver(self.sma_fast, self.sma_slow)

# #     def next(self):
# #         if self.position.size:
# #             if self.buy_signal < 0:
# #                 self.sell()

# #         elif self.buy_signal > 0:
# #             self.buy()

# #         return


# # class TheStrat(bt.Strategy):
# #     """
# #     Class for implementing trading algorithms that our partner can handle

# #     References:
# #     -----------
# #         Strategies: https://www.backtrader.com/docu/strategy/
# #     """

# #     def __init__(self, *args, **kwargs) -> None:
# #         """
# #         Parameters:
# #         -----------
# #             cf. https://www.backtrader.com/docu/strategy/#reference-strategy
# #         """
# #         self.strategy = btind.SimpleMovingAverage()

# #     def __repr__(self) -> str:
# #         return f""

# #     # ==============================
# #     # DEPARTMENT OF THE INTERIOR
# #     # ==============================

# #     # ==============================
# #     # DEPARTMENT OF THE EXTERIOR
# #     # ==============================

# #     def next(self):
# #         """
# #         This method will be called for all remaining data
# #         points when the minimum period for all datas/indicators
# #         have been meet.

# #         Strategies, like a trader in the real world, will
# #         get notified when events take place. Actually once
# #         per next cycle in the backtesting process.

# #         Strategies also like traders have the chance to
# #         operate in the market during the next method to
# #         try to achieve profit with
# #         """
# #         if self.strategy > self.data.close:
# #             # do something
# #             pass

# #         elif self.strategy < self.data.close:
# #             # do something
# #             pass

# #     def nexstart(self):
# #         """
# #         This method will be called once, exactly when the minimum
# #         period for all datas/indicators have been meet.
# #         The default behavior is to call next
# #         """
# #         return super().nexstart()

# #     def prenext(self):
# #         """
# #         This method will be called before the minimum period of
# #         all datas/indicators have been meet for the strategy to
# #         start executing
# #         """
# #         return super().prenext()

# #     def start(self):
# #         """
# #         Called right before the backtesting is about to be started.
# #         """
# #         return super().start()

# #     def stop(self):
# #         """
# #         Called right before the backtesting is about to be stopped
# #         """
# #         return super().stop()

# #     def notify_order(self, order) -> int:
# #         """
# #         Receives an order whenever there has been a change in one
# #         """
# #         return super.notify_order(order)

# #     def notify_trade(self, trade):
# #         """
# #         Receives a trade whenever there has been a change in one
# #         """
# #         return super().notify_trade(trade)

# #     def notify_cashvalue(self, cash, value):
# #         """
# #         Receives the current fund value, value status of the strategy’s broker
# #         """
# #         return super().notify_cashvalue(cash, value)

# #     def notify_fund(self, cash, value, fundvalue, shares):
# #         """
# #         Receives the current cash, value, fundvalue and fund shares
# #         """
# #         return super().notify_fund(cash, value, fundvalue, shares)

# #     def notify_store(self, msg, *args, **kwargs):
# #         """
# #         Receives a notification from a store provider
# #         """
# #         return super().notify_store(msg, *args, **kwargs)

# #     def buy(
# #         self,
# #         data: int = None,
# #         size: int = None,
# #         price: int = None,
# #         plimit: int = None,
# #         exectype: int = None,
# #         valid: datetime = None,
# #         tradeid: int = 0,
# #         oco: int = None,
# #         trailamount: int = None,
# #         trailpercent: int = None,
# #         parent: int = None,
# #         transmit: bool = True,
# #         **kwargs,
# #     ):
# #         return super().buy(
# #             data,
# #             size,
# #             price,
# #             plimit,
# #             exectype,
# #             valid,
# #             tradeid,
# #             oco,
# #             trailamount,
# #             trailpercent,
# #             parent,
# #             transmit,
# #             **kwargs,
# #         )

# #     def sell(
# #         self,
# #         data: int = None,
# #         size: int = None,
# #         price: int = None,
# #         plimit: int = None,
# #         exectype: int = None,
# #         valid: datetime = None,
# #         tradeid: int = 0,
# #         oco: int = None,
# #         trailamount: int = None,
# #         trailpercent: int = None,
# #         parent: int = None,
# #         transmit: bool = True,
# #         **kwargs,
# #     ):
# #         return super().sell(
# #             data,
# #             size,
# #             price,
# #             plimit,
# #             exectype,
# #             valid,
# #             tradeid,
# #             oco,
# #             trailamount,
# #             trailpercent,
# #             parent,
# #             transmit,
# #             **kwargs,
# #         )

# #     def close(self, data: int = None, size: int = None, **kwargs):
# #         return super().close(data, size, **kwargs)

# #     def cancel(self, order):
# #         return super().cancel(order)

# #     def buy_bracket(
# #         self,
# #         data: int = None,
# #         size: int = None,
# #         price: int = None,
# #         plimit: int = None,
# #         exectype: int = bt.Order.Limit,
# #         valid: int = None,
# #         tradeid: int = 0,
# #         trailamount: int = None,
# #         trailpercent: int = None,
# #         oargs: int = ...,
# #         stopprice: int = None,
# #         stopexec: int = bt.Order.Stop,
# #         stopargs: int = ...,
# #         limitprice: int = None,
# #         limitexec: int = bt.Order.Limit,
# #         limitargs: int = ...,
# #         **kwargs,
# #     ):
# #         return super().buy_bracket(
# #             data,
# #             size,
# #             price,
# #             plimit,
# #             exectype,
# #             valid,
# #             tradeid,
# #             trailamount,
# #             trailpercent,
# #             oargs,
# #             stopprice,
# #             stopexec,
# #             stopargs,
# #             limitprice,
# #             limitexec,
# #             limitargs,
# #             **kwargs,
# #         )

# #     def sell_bracket(
# #         self,
# #         data=None,
# #         size=None,
# #         price=None,
# #         plimit=None,
# #         exectype=bt.Order.Limit,
# #         valid=None,
# #         tradeid=0,
# #         trailamount=None,
# #         trailpercent=None,
# #         oargs=...,
# #         stopprice=None,
# #         stopexec=bt.Order.Stop,
# #         stopargs=...,
# #         limitprice=None,
# #         limitexec=bt.Order.Limit,
# #         limitargs=...,
# #         **kwargs,
# #     ):
# #         return super().sell_bracket(
# #             data,
# #             size,
# #             price,
# #             plimit,
# #             exectype,
# #             valid,
# #             tradeid,
# #             trailamount,
# #             trailpercent,
# #             oargs,
# #             stopprice,
# #             stopexec,
# #             stopargs,
# #             limitprice,
# #             limitexec,
# #             limitargs,
# #             **kwargs,
# #         )

# #     def order_target_size(self, data=None, target=0, **kwargs):
# #         return super().order_target_size(data, target, **kwargs)

# #     def order_target_value(self, data=None, target=0, price=None, **kwargs):
# #         return super().order_target_value(data, target, price, **kwargs)

# #     def order_target_percent(self, data=None, target=0, **kwargs):
# #         return super().order_target_percent(data, target, **kwargs)

# #     def getsizer(self):
# #         return super().getsizer()

# #     def setsizer(self, sizer):
# #         return super().setsizer(sizer)

# #     def getsizing(self, data=None, isbuy=True):
# #         return super().getsizing(data, isbuy)

# #     def getposition(self, data=None, broker=None):
# #         return super().getposition(data, broker)

# #     def getpositionbyname(self, name=None, broker=None):
# #         return super().getpositionbyname(name, broker)

# #     def getdatanames(self):
# #         return super().getdatanames()

# #     def getdatabyname(self, name):
# #         return super().getdatabyname(name)

# #     def add_timer(
# #         self,
# #         when,
# #         offset=...,
# #         repeat=...,
# #         weekdays=...,
# #         weekcarry=False,
# #         monthdays=...,
# #         monthcarry=True,
# #         allow=None,
# #         tzdata=None,
# #         cheat=False,
# #         *args,
# #         **kwargs,
# #     ):
# #         return super().add_timer(
# #             when,
# #             offset,
# #             repeat,
# #             weekdays,
# #             weekcarry,
# #             monthdays,
# #             monthcarry,
# #             allow,
# #             tzdata,
# #             cheat,
# #             *args,
# #             **kwargs,
# #         )

# #     def notify_timer(self, timer, when, *args, **kwargs):
# #         return super().notify_timer(timer, when, *args, **kwargs)


# if __name__ == "__main__":
#     ...
