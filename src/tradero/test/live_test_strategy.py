from tradero.live.exchanges.bybit import BybitSesh
from tradero.test import BTCUSDT
from tradero.backtesting import Backtest, Strategy
from tradero.risk import entry_adjustment, tp_by_ratio, sl_by_ratio, size_by_risk, sl_adjustment, tp_adjustment
from tradero.ta import PIVOT, RGXPIVOT, NBS, ZIGZAG, RSI
import re 
from tradero.test.demo_keys import api, secret
import asyncio

class Neutralizers_5P(Strategy):
    pair = "BTCUSDT"
    async def init(self):
        self.risk = 0.005
        self.entry_adjust_factor = -0.4 # basado en sl
        self.tp_adjustment = 1.1
        self.sl_adjustment = 1
        self.sl_ratio = 1/3
        # from tradero.core import _CryptoBacktestSesh
        # self.rq_pattern = r"(<L-(?:emp|boo|shk)[^>]*>)  (<H-(?:emp|neu)[^>]*>)  (<L-boo[^>]*>)  (<H-(?:neu|shk)[^>]*>)  (?:<L-emp[^>]*><H-(?:emp|boo)[^>]*>){0,10}  (<L-neu[^>]*>)  $  ".replace(" ", "") 
        self.A, self.B, self.C, self.D, self.E = None, None, None, None, None
        print("seteando patron")
        self.rq_pattern = re.compile(r"(<L-(?:emp|boo|shk)[^>]*>)  (<H-(?:emp|neu)[^>]*>) (<L-boo[^>]*>) (<H-(?:neu|shk)[^>]*>) (?:<L-emp[^>]*><H-(?:emp|boo)[^>]*>){0,10} (<L-neu[^>]*>) $".replace(" ", "") )
        # self.rq_pattern = re.compile(r" (<L-boo[^>]*>) (<H-(?:neu|shk)[^>]*>) (?:<L-emp[^>]*><H-(?:emp|boo)[^>]*>){0,10} (<L-neu[^>]*>) $".replace(" ", "") )

    async def on_data(self):
        print("obteniendo tiempo y sesion")
        sesh = self.get_sesh()
        time = await sesh.get_time()

        # if time.minute % 5 != 4:
        #     return
        
        # if time.minute % 5 != 0:
        #     return
        
        print("pidiendo data")
        all_data_5m = (await sesh.get_data(symbol=self.pair, timeframe='5min', limit=300))
        data_5m = all_data_5m[:-1] # solo velas cerradas
        unclosed_bar = all_data_5m[-1]
        print(f"  time: {time}, price: {unclosed_bar.Close}")

        position = await sesh.get_position_status(symbol=self.pair) ; print(f"position: {position}")
        is_active_position = position['size'] != 0 # ; print(f"  position: {position["size"]:.2f}, active?: {is_active_position}")
        orders = await sesh.get_all_orders(symbol=self.pair) 
        pending_orders = [order["orderStatus"] == "Pending" for order in orders]#; print(f"  pending_orders: {any(pending_orders)}")
        # Calcular Indicadores
        # print(RGX_PIV.search(self.rq_pattern))
        any_pending_order = any(pending_orders)
        if is_active_position:
            print("ya hay posicion pendiente")
            return
            #
        elif any_pending_order: # si hay ordenes pendientes
            print("hay ordenes, calculando si se puede cancelar")
            if self.D:
                if data_5m.High[-1] > self.D:
                    self.A, self.B, self.C, self.D, self.E = None, None, None, None, None
                    await sesh.cancel_all_orders(symbol=self.pair)
                    #
        else: # Buscar entrada  
            print("calculando RGXPIV")
            RGX_PIV = await self.I(RGXPIVOT, data=data_5m, nbs=True)
            print("buscando patron")
            found_pattern = RGX_PIV.search(self.rq_pattern)
            if found_pattern:
                print("se encontró patron")
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
                    print("patron roto")
                    return 

                # Gestion Base
                de_diff, ae_diff = abs(self.D - self.E), abs(self.A - self.E)
                is_close_to_e = (ae_diff < de_diff / 2) and (self.A < self.E)

                entry = self.A if is_close_to_e else self.E
                tp = self.D
                sl = sl_by_ratio(ratio=self.sl_ratio, entry=entry, tp=tp)
                # Adjustment
                tp = tp_adjustment(entry, tp, adjustment=self.tp_adjustment)
                sl = sl_adjustment(entry, sl, adjustment=self.sl_adjustment)
                price = entry_adjustment(entry_original=entry, sl=sl, adjust_factor=self.entry_adjust_factor)
                # Calcular el size depues de los ajustes
                size = size_by_risk(risk=self.risk, cash=total_equity, entry=price, sl=sl)
            
                # #### ORDEN ###
                await sesh.buy(symbol=self.pair, price=price, tp_price=tp, sl_price=sl, size=size)
                print(f"=== BUY ===  pair:{self.pair} size:{size} price:{price} tp:{tp}, sl:{sl} ")

    async def on_exit(self):
        print("ejecutando on_exit")
        sesh = self.sesh
        data_15m = await sesh.get_data(symbol=self.pair, timeframe="15min")
        data_5m = await sesh.get_data(symbol=self.pair, timeframe="5min")
        await self.I(ZIGZAG, data=data_5m)
        await self.I(NBS, data=data_5m)
        # await self.I(RSI, data=data_15m.Close)
   
class SimpleStrategy(Strategy):
    async def init(self):
        print("init de simplestrategy")
    async def on_data(self):
        sesh = self.sesh
        time = await sesh.get_time()
        # print(f"  ss1 time: {time}")
        data = await self.sym_get_data()  
        print("se obtuvo los datos ss1")

    async def on_exit(self):
        print("exitttt")

    async def sym_get_data(self):
        print("\033[95;1m  ss1: pticion !!  \033[0m")
        import time
        await asyncio.sleep(4)

class SimpleStrategy2(Strategy):
    async def init(self):
        print("init de simplestrategy")
    async def on_data(self):
        sesh = self.sesh
        time = await sesh.get_time()
        # print(f"  ss2 time: {time}")
        data = await self.sym_get_data()  
        print("se obtuvo los datos ss2")

    async def on_exit(self):
        print("exitttt")

    async def sym_get_data(self):
        print("\033[96;1m  ss2: pticion !!  \033[0m")
        import time
        await asyncio.sleep(4)

sesh = BybitSesh(api_key=api, api_secret=secret, category="linear", demo=True)

# sesh.run_live(Neutralizers_5P, init_sleep=0.0, on_data_sleep=5)
results = sesh.run_live(SimpleStrategy, SimpleStrategy2, init_sleep=0.0, on_data_sleep=0.5)

print(f"\033[96;1m[!]\033[1m Results: {results}\033[0m")
