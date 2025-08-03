#! /bin/bash

# Algorithms to use with our trading partner

# Sam Dawley
# 08/2025

import os
from typing import Tuple

import backtrader as bt
import backtrader.indicators as btind

# ==================================================
# ALGORITHMS WHICH WORK WITH OUR TRADING PARTNER
# ==================================================

class SimpleMovingAverage(bt.Strategy):
    """
    Class for implementing trading algorithms that our partner can handle
    
    References:
    -----------
        Strategies: https://www.backtrader.com/docu/strategy/
    """

    def __init__(self, period: int) -> None:
        self.strategy = btind.SimpleMovingAverage(period=period)
    

    def __repr__(self) -> str:
        return f""

    
    # ===== private functions =====


    # ===== public functions =====


    def start(self):
        return super().start()
    

    def prenext(self):
        return super().prenext()
    

    def nexstart(self):
        return super().nexstart()


    def next(self):
        if self.strategy > self.data.close:
            # do something
            pass

        elif self.strategy < self.data.close:
            # do something
            pass


    def stop(self):
        return super().stop()


if __name__ == "__main__":
    ...