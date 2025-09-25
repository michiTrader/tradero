from tradero.test import BTCUSDT
from tradero.backtesting import Backtest, Strategy
from tradero.risk import adjust_entry, tp_by_ratio, sl_by_ratio, size_by_risk, adjust_sl, adjust_tp
from tradero.ta import PIVOT, RGXPIVOT, NBS, ZIGZAG, RSI
import asyncio
import re 

class Neutralizers_5P(Strategy):
    async def init(self):
        self.risk = 0.005
        self.entry_adjust_factor = -0.4 # basado en sl
        self.tp_adjustment = 1.1
        self.sl_adjustment = 1
        self.sl_ratio = 1/3
        # from tradero.core import _CryptoBacktestSesh
        # self.rq_pattern = r"(<L-(?:emp|boo|shk)[^>]*>)  (<H-(?:emp|neu)[^>]*>)  (<L-boo[^>]*>)  (<H-(?:neu|shk)[^>]*>)  (?:<L-emp[^>]*><H-(?:emp|boo)[^>]*>){0,10}  (<L-neu[^>]*>)  $  ".replace(" ", "") 
        self.A, self.B, self.C, self.D, self.E = None, None, None, None, None
        self.rq_pattern = re.compile(r"(<L-(?:emp|boo|shk)[^>]*>)  (<H-(?:emp|neu)[^>]*>) (<L-boo[^>]*>) (<H-(?:neu|shk)[^>]*>) (?:<L-emp[^>]*><H-(?:emp|boo)[^>]*>){0,10} (<L-neu[^>]*>) $".replace(" ", "") )
        # self.rq_pattern = re.compile(r" (<L-boo[^>]*>) (<H-(?:neu|shk)[^>]*>) (?:<L-emp[^>]*><H-(?:emp|boo)[^>]*>){0,10} (<L-neu[^>]*>) $".replace(" ", "") )

    async def on_data(self):
        sesh = self.get_sesh()
        time = await sesh.get_time()

        if time.minute % 5 != 4:
            return
        
        # if time.minute % 5 != 0:
        #     return
        

        all_data_5m = (await sesh.get_data(symbol='BTCUSDT', timeframe='5min', limit=300))
        data_5m = all_data_5m[:-1]
        unclosed_bar = all_data_5m[-1]

        position = await sesh.get_position_status(symbol='BTCUSDT')
        is_active_position = position['size'] != 0 # ; print(f"  position: {position["size"]:.2f}, active?: {is_active_position}")
        orders = await sesh.get_all_orders(symbol="BTCUSDT") 
        pending_orders = [order["orderStatus"] == "Pending" for order in orders]#; print(f"  pending_orders: {any(pending_orders)}")
        # Calcular Indicadores
        # print(RGX_PIV.search(self.rq_pattern))
        any_pending_order = any(pending_orders)
        if is_active_position:
            return
            #
        elif any_pending_order: # si hay ordenes pendientes
            if self.D:
                if data_5m.High[-1] > self.D:
                    self.A, self.B, self.C, self.D, self.E = None, None, None, None, None
                    await sesh.cancel_all_orders(symbol="BTCUSDT")
                    #
        else: # Buscar entrada  
            RGX_PIV = await self.I(RGXPIVOT, data=data_5m, nbs=True)
            found_pattern = RGX_PIV.search(self.rq_pattern)
            if found_pattern:
                RGX_PIV.set_tag(self.rq_pattern, ('a','b','c','d','e')) 

                total_equity = float((await sesh.get_balance())["totalEquity"]) # TODO: arreglar la funcion original para que devuelva un float (si es así en BybitSesh)
                
                pivot_values = RGX_PIV.values
                # pivot_tag_values = list(zip(nbs.ptypes, basic_pivot_values)) 
                tags = RGX_PIV.tags
                self.A = pivot_values[tags == 'a'][0]
                self.B = pivot_values[tags == 'b'][0]
                self.C = pivot_values[tags == 'c'][0]
                self.D = pivot_values[tags == 'd'][0]
                self.E = pivot_values[tags == 'e'][0]

                is_broken_pattern = any(all_data_5m.Low[-10:] < self.E)
                if is_broken_pattern:
                    return 

                # Gestion Base
                de_diff, ae_diff = abs(self.D - self.E), abs(self.A - self.E)
                is_close_to_e = (ae_diff < de_diff / 2) and (self.A < self.E)

                entry = self.A if is_close_to_e else self.E
                tp = self.D
                sl = sl_by_ratio(ratio=self.sl_ratio, entry=entry, tp=tp)
                # Adjustment
                tp = adjust_tp(entry, tp, adjustment=self.tp_adjustment)
                sl = adjust_sl(entry, sl, adjustment=self.sl_adjustment)
                price = adjust_entry(entry_original=entry, sl=sl, adjust_factor=self.entry_adjust_factor)
                # Calcular el size depues de los ajustes
                size = size_by_risk(risk=self.risk, cash=total_equity, entry=price, sl=sl)
            
                # #### ORDEN ###
                await sesh.buy(symbol="BTCUSDT", price=price, tp_price=tp, sl_price=sl, size=size)

    async def on_exit(self):
        sesh = self.sesh
        data_15m = await sesh.get_data(symbol="BTCUSDT", timeframe="15min")
        data_5m = await sesh.get_data(symbol="BTCUSDT", timeframe="5min")
        await self.I(ZIGZAG, data=data_5m)
        await self.I(NBS, data=data_5m)
        # await self.I(RSI, data=data_15m.Close)
   
""" Neutralizers 5m """
# data = BTCUSDT[2450:2700].resample("5m")#[1600:2000]
# klines = BTCUSDT[:8000].resample("5m")#[1600:2000]
packet = {"BTCUSDT": BTCUSDT[:5000].resample("1min")}#[1600:2000]
bt = Backtest(strategy=Neutralizers_5P, packet=packet, margin=1/20, cash=100_000, warmup=50)
stats = bt.run(mae_metric_type="ROE", p_bar=True)
bt.plot( 
    plot_equity=False,
    plot_return=True,
    plot_trades=True,
    relative_equity=True,
    plot_volume=True,
    # width=1000,
    # height=500,
    timeframe="5min",
)  # 19:00:00 c: 104489 v 215