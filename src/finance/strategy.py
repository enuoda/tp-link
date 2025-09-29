#! /bin/bash

"""
Algorithms to use with our trading partner
All are implemented with the backtrader API

References
    bt.Strategy
        https://www.backtrader.com/docu/strategy/#reference-strategy

Sam Dawley
08/2025
"""

from datetime import datetime
from typing import override

import backtrader as bt
import backtrader.indicators as btind


# ==================================================
# ALGORITHMS WHICH WORK WITH OUR TRADING PARTNER
# ==================================================


class SMACrossover(bt.Strategy):
    """
    Long-only strategy which operates on a moving average cross
    Implementation taken (for testing purposes) from:
        https://github.com/mementum/backtrader/blob/master/backtrader/strategies/sma_crossover.py

    Buy Logic:
    ----------
        - No position is open on the data
        - The ``fast`` moving averagecrosses over the ``slow`` strategy to the upside.

    Sell Logic:
    -----------
        - A position exists on the data
        - The ``fast`` moving average crosses over the ``slow`` strategy to the downside

    Order Execution Type:
    ---------------------
        - Market
    """

    alias = ("SMA_Crossover",)
    params = (("fast", 10), ("slow", 30), ("_moveav", btind.MovAv.SMA))

    def __init__(self):
        """ Conception """
        self.sma_fast = self.p_moveav(period=self.p_fast)
        self.sma_slow = self.p_moveav(period=self.p.slow)

        self.buy_signal = btind.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        """ Adulthood """
        if self.position.size:
            if self.buy_signal < 0:
                self.sell()

        elif self.buy_signal > 0:
            self.buy()

        return


class MFI(bt.Indicator):
    """
    Money flow index
    A technical oscillator that uses price and volume
    data to measure the money flowing in and out
    of a security

    References:
        https://www.backtrader.com/recipes/indicators/mfi/mfi/
        https://www.investopedia.com/terms/m/mfi.asp
        https://www.backtrader.com/docu/concepts/?h=lines#lines

    """

    lines = ('mfi',)
    params = {"period": 14}

    def __init__(self) -> None:
        tprice = (self.data.close + self.data.low + self.data.high) / 3.0
        mfraw = tprice * self.data.volume

        flowpos = bt.ind.SumN(mfraw * (tprice > tprice(-1)), period=self.p.period)
        flowneg = bt.ind.SumN(mfraw * (tprice < tprice(-1)), period=self.p.period)

        mfiratio = bt.ind.DivByZero(flowpos, flowneg, zero=100.0)
        self.lines.mfi = 100.0 - 100.0 / (1.0 + mfiratio)


class StochRSI(bt.Indicator):
    """
    Applies stochastic oscillator formula to relative strength index (RSI)
    to identify overbought or oversold conditions in the market

    References:
        https://www.backtrader.com/recipes/indicators/stochrsi/stochrsi/
        https://www.investopedia.com/terms/s/stochrsi.asp
        https://www.backtrader.com/docu/concepts/?h=lines#lines
    """

    lines = ('stochrsi',)
    params = {"period": 14, "pperiod": None}

    def __init__(self) -> None:
        rsi = bt.ind.RSI(self.data, period=self.p.period)

        pperiod = self.p.pperiod or self.p.period
        maxrsi = bt.ind.Highest(rsi, period=pperiod)
        minrsi = bt.ind.Lowest(rsi, period=pperiod)

        self.lines.stochrsi = (rsi - minrsi) / (maxrsi - minrsi)


class CustomStrategy(bt.Strategy):
    """
    References:
        Strategies: https://www.backtrader.com/docu/strategy/
    """

    __slots__ = ("strategy", "ind1")

    def __init__(self) -> None:
        """
        Parameters:
            cf. https://www.backtrader.com/docu/strategy/#reference-strategy
        """
        self.strategy = btind.SimpleMovingAverage()

        # instantiate indicators
        self.ind1 = MFI()

    @override
    def next(self):
        """
        Called for all remaining data points when the
        minimum period for all datas/indicators have been meet.

        This is where indicator values are used/checked
        """
        if self.strategy > self.data.close:
            # do something
            pass

        elif self.strategy < self.data.close:
            # do something
            pass

    @override
    def nexstart(self):
        """
        This method will be called once, to mark 
        the switch from prenext to next
        The default behavior is to call next
        """
        return super().nexstart()
    @override
    def prenext(self):
        """
        This method will be called before the minimum period of
        all datas/indicators have been meet for the strategy to
        start executing
        """
        return super().prenext()
    @override
    def start(self):
        """
        Called right before the backtesting is about to be started.
        """
        return super().start()
    @override
    def stop(self):
        """
        Called right before the backtesting is about to be stopped
        """
        return super().stop()
    @override
    def notify_order(self, order) -> int:
        """
        Receives an order whenever there has been a change in one
        """
        return super.notify_order(order)
    @override
    def notify_trade(self, trade):
        """
        Receives a trade whenever there has been a change in one
        """
        return super().notify_trade(trade)
    @override
    def notify_cashvalue(self, cash, value):
        """
        Receives the current fund value, value status of the strategy’s broker
        """
        return super().notify_cashvalue(cash, value)
    @override
    def notify_fund(self, cash, value, fundvalue, shares):
        """
        Receives the current cash, value, fundvalue and fund shares
        """
        return super().notify_fund(cash, value, fundvalue, shares)
    @override
    def notify_store(self, msg, *args, **kwargs):
        """
        Receives a notification from a store provider
        """
        return super().notify_store(msg, *args, **kwargs)
    @override
    def buy(
        self,
        data: int = None,
        size: int = None,
        price: int = None,
        plimit: int = None,
        exectype: int = None,
        valid: datetime = None,
        tradeid: int = 0,
        oco: int = None,
        trailamount: int = None,
        trailpercent: int = None,
        parent: int = None,
        transmit: bool = True,
        **kwargs,
    ):
        return super().buy(
            data,
            size,
            price,
            plimit,
            exectype,
            valid,
            tradeid,
            oco,
            trailamount,
            trailpercent,
            parent,
            transmit,
            **kwargs,
        )
    @override
    def sell(
        self,
        data: int = None,
        size: int = None,
        price: int = None,
        plimit: int = None,
        exectype: int = None,
        valid: datetime = None,
        tradeid: int = 0,
        oco: int = None,
        trailamount: int = None,
        trailpercent: int = None,
        parent: int = None,
        transmit: bool = True,
        **kwargs,
    ):
        return super().sell(
            data,
            size,
            price,
            plimit,
            exectype,
            valid,
            tradeid,
            oco,
            trailamount,
            trailpercent,
            parent,
            transmit,
            **kwargs,
        )
    @override
    def close(self, data: int = None, size: int = None, **kwargs):
        return super().close(data, size, **kwargs)
    @override
    def cancel(self, order):
        return super().cancel(order)
    @override
    def buy_bracket(
        self,
        data: int = None,
        size: int = None,
        price: int = None,
        plimit: int = None,
        exectype: int = bt.Order.Limit,
        valid: int = None,
        tradeid: int = 0,
        trailamount: int = None,
        trailpercent: int = None,
        oargs: int = ...,
        stopprice: int = None,
        stopexec: int = bt.Order.Stop,
        stopargs: int = ...,
        limitprice: int = None,
        limitexec: int = bt.Order.Limit,
        limitargs: int = ...,
        **kwargs,
    ):
        return super().buy_bracket(
            data,
            size,
            price,
            plimit,
            exectype,
            valid,
            tradeid,
            trailamount,
            trailpercent,
            oargs,
            stopprice,
            stopexec,
            stopargs,
            limitprice,
            limitexec,
            limitargs,
            **kwargs,
        )
    @override
    def sell_bracket(
        self,
        data: int=None,
        size=None,
        price=None,
        plimit=None,
        exectype=bt.Order.Limit,
        valid=None,
        tradeid=0,
        trailamount=None,
        trailpercent=None,
        oargs=...,
        stopprice=None,
        stopexec=bt.Order.Stop,
        stopargs=...,
        limitprice=None,
        limitexec=bt.Order.Limit,
        limitargs=...,
        **kwargs,
    ):
        return super().sell_bracket(
            data,
            size,
            price,
            plimit,
            exectype,
            valid,
            tradeid,
            trailamount,
            trailpercent,
            oargs,
            stopprice,
            stopexec,
            stopargs,
            limitprice,
            limitexec,
            limitargs,
            **kwargs,
        )
    @override
    def order_target_size(self, data=None, target=0, **kwargs):
        return super().order_target_size(data, target, **kwargs)
    @override
    def order_target_value(self, data=None, target=0, price=None, **kwargs):
        return super().order_target_value(data, target, price, **kwargs)
    @override
    def order_target_percent(self, data=None, target=0, **kwargs):
        return super().order_target_percent(data, target, **kwargs)
    @override
    def getsizer(self):
        return super().getsizer()
    @override
    def setsizer(self, sizer):
        return super().setsizer(sizer)
    @override
    def getsizing(self, data=None, isbuy=True):
        return super().getsizing(data, isbuy)
    @override
    def getposition(self, data=None, broker=None):
        return super().getposition(data, broker)
    @override
    def getpositionbyname(self, name=None, broker=None):
        return super().getpositionbyname(name, broker)
    @override
    def getdatanames(self):
        return super().getdatanames()
    @override
    def getdatabyname(self, name):
        return super().getdatabyname(name)
    @override
    def add_timer(
        self,
        when,
        offset=...,
        repeat=...,
        weekdays=...,
        weekcarry=False,
        monthdays=...,
        monthcarry=True,
        allow=None,
        tzdata=None,
        cheat=False,
        *args,
        **kwargs,
    ):
        return super().add_timer(
            when,
            offset,
            repeat,
            weekdays,
            weekcarry,
            monthdays,
            monthcarry,
            allow,
            tzdata,
            cheat,
            *args,
            **kwargs,
        )
    @override
    def notify_timer(self, timer, when, *args, **kwargs):
        return super().notify_timer(timer, when, *args, **kwargs)


if __name__ == "__main__":
    ...
