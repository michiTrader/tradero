import pytest
from .data import BTCUSDT
from tradero.models import Strategy
from tradero.backtesting import Backtest, run_backtests
# @pytest.mark.parametrize([])

class TestStrategy_1(Strategy):
    """ Posiciones maximas: 1 """
    
    PAIR = "BTCUSDT"
    STGY_TIMEFRAME = "1min"
    MINUTE_BUY = 5
    MINUTE_SELL = 55

    async def init(self):
        self.log(f"Par: {self.PAIR}, timeframe: {self.STGY_TIMEFRAME}")

    async def on_live(self):
        sesh: 'BybitSesh' = self.get_sesh()
        time = await sesh.get_time(tz="UTC-05:00")
        log___ = self.log 
        
        all_data_ohlc = await sesh.get_data(symbol=self.PAIR, timeframe=self.STGY_TIMEFRAME, limit=200)
        # ohlc_closed_bars = all_data_ohlc[:-1] # solo velas cerradas
        unclosed_bar = all_data_ohlc[-1]

        if time.minute == self.MINUTE_BUY:
            log___(f"Buy, price:{unclosed_bar.Close[-1]}")
            await sesh.buy(symbol=self.PAIR, size=0.1) 

        if time.minute == self.MINUTE_SELL:
            log___(f"Sell, price:{unclosed_bar.Close[-1]}")
            await sesh.sell(symbol=self.PAIR, size=0.1) 


# test__backtest_optimize()
# 

