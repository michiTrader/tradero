from ast import Dict
from tradero.live.exchanges.bybit import BybitSesh
from tradero.live import run_strategies
from tradero.test import BTCUSDT
from tradero.lib import timeframe2minutes
from tradero.backtesting import Backtest
from tradero.models import Strategy, DataOHLC
from tradero.risk import tp_by_ratio, sl_by_ratio, size_by_risk, adjust_entry, adjust_sl, adjust_tp, adjust_size, adjust_leverage
from tradero.ta import PIVOT, RGXPIVOT, NBS, ZIGZAG, RSI
import re 
import asyncio
import numpy as np

class Neutralizers_5P(Strategy):
    """ Operaciones maximas: 1 """
    
    PAIR = "HOLOUSDT"
    STGY_TIMEFRAME = "1min"
    PCT_RISK = 0.0025 # pct

    STGY_INTERVAL = timeframe2minutes(STGY_TIMEFRAME)
    SL_RATIO = 1/3
    ENTRY_ADJUST_FACTOR = -0.4 # basado en sl
    TP_ADJUSTMENT_VAL = 1.1 # real
    SL_ADJUSTMENT_VAL = 1 

    # elimnar:
    # TP_ADJUSTMENT_VAL = 1
    # ENTRY_ADJUST_FACTOR = 0 
    async def init(self):
        sesh: BybitSesh = self.sesh
        await sesh.set_leverage(symbol=self.PAIR, leverage=1)
        self.instrument_info = await sesh.get_instruments_info(symbol=self.PAIR)
    
        self.log(f"Par: {self.PAIR}, timeframe: {self.STGY_TIMEFRAME}, leverage: 1X")

        self.risk_limit = await sesh.get_risk_limit(self.PAIR)
        # self.log(f"lista risk_limit obtenido len_risk_limit:{len(self.risk_limit)}")
  
        # Parametros de logica en la Estrategia
        self.rq_pattern = re.compile(r"(<L-(?:emp|boo|shk)[^>]*>)  (<H-(?:emp|neu)[^>]*>) (<L-boo[^>]*>) (<H-(?:neu|shk)[^>]*>) (?:<L-emp[^>]*><H-(?:emp|boo)[^>]*>){0,10} (<L-neu[^>]*>) $".replace(" ", "") )
        self.A, self.B, self.C, self.D, self.E = [None]*5
        self.max_price_after_pattern = 0
        self.min_price_after_pattern = np.float64('+inf')

        # Modes
        self.in_order_mode = False  
        self.in_position_mode = False
        self.in_search_mode = False

        self.order_id = None
        self.position_tp_order_id = None 
        self.position_sl_order_id = None

        # iniciar la busqueda
        self.activate_search_mode() 
    async def on_live(self):
        sesh: 'BybitSesh' = self.get_sesh()
        time = await sesh.get_time(tz="UTC-05:00")
        log___ = self.log
        
        # if time.minute % self.stgy_interval != 0:
            # return
        # is_active_order = self.order_id is not None

        all_data_ohlc = (await sesh.get_data(symbol=self.PAIR, timeframe=self.STGY_TIMEFRAME, limit=300))
        ohlc_closed_bars = all_data_ohlc[:-1] # solo velas cerradas
        unclosed_bar = all_data_ohlc[-1]

        # orders = await sesh.get_all_orders(symbol=self.pair) 
        # new_orders = [order["orderStatus"] == "New" for order in orders]#; print(f"  pending_orders: {any(pending_orders)}")

        if self.in_order_mode:
            order = await sesh.get_order(symbol=self.PAIR, order_id=self.order_id)
            order_id = order['orderId']
            
            # Comprobar la la orden ya fue cancelada o ejecutada
            is_filled_order = order["orderStatus"] == "Filled"
            is_cancelled_order = order["orderStatus"] == "Cancelled"

            d_is_broken = self.d_is_broken(all_data_ohlc, self.D, self.D_idx) 
            
            if d_is_broken and not is_filled_order:
                log___(f"Patron roto despues de la Orden Limit: se supero el punto D({self.D}): Cancelando la orden... order_id:{order_id[-8:]} ")
                await sesh.cancel_order(symbol=self.PAIR, order_id=order_id)
                log___(f"--Orden Cancelada-- order_id:{order_id[-8:]}")
                self.deactivate_order_mode()
                self.activate_search_mode()
                #
            elif is_filled_order: # desactivar orden
                log___(f"--- Orden Ejecutada --- id:{order_id[-8:]}")
                self.position = await sesh.get_position(symbol=self.PAIR) 

                self.deactivate_order_mode()
                self.activate_position_mode()
                #  
            elif is_cancelled_order:
                log___(f"--- Orden ya Cancelada ---")
                self.deactivate_order_mode() 
                self.activate_search_mode()
                #

        if self.in_position_mode:
            # obtener el id de las ordenes TP y SL
            if not self.position_tp_order_id or not self.position_sl_order_id:
                tp_order, sl_order = await self.on_position_try_get_new_tpsl_orders()
                self.position_tp_order_id, self.position_sl_order_id = tp_order["orderId"], sl_order["orderId"]

                log___(f"TP_id: {self.position_tp_order_id}, SL_id: {self.position_sl_order_id}")

            # Primero verificar si realmente hay una posición abierta
            position = await sesh.get_position(symbol=self.PAIR)
            
            is_active_position = position['side'] != ''            
            if not is_active_position:
                if time.minute % 10 == 0:
                    log___(f"Buscando Patron...")

                tp_order, sl_order = await self.on_position_try_get_old_tpsl_orders()

                # Comprobar cual fue la orden que se cerró
                tp_result = tp_order["orderStatus"] == "Filled"
                sl_result = sl_order["orderStatus"] == "Filled"

                if tp_result or sl_result:
                    if tp_result and sl_result:
                        log___(f"WTF???", type="error")

                    if tp_result:
                        log___(f"--------------- TP alanczado tpOrderId:{tp_order["orderId"]} --------------- ", type="info")
                        closed_pnl = (await self.try_get_closed_pnl(tp_order["orderId"]))
                        log___(f"P&L: \033[92m███ ${float(closed_pnl):.2f} ███\033[0m ", type="info")
                    elif sl_result:
                        log___(f"--------------- SL alanczado slOrderId:{sl_order["orderId"]} --------------- ", type="info")
                        closed_pnl = (await self.try_get_closed_pnl(sl_order["orderId"]))
                        log___(f"P&L: \033[91m███ ${float(closed_pnl):.2f} ███\033[0m ======== ", type="info")
                else:
                    log___(f"Ordenes TP/SL no encontradas. orderId TP: {self.position_tp_order_id}, orderId SL: {self.position_sl_order_id}", type="error")

                # Desactivar el modo Posicion
                self.deactivate_position_mode()
                self.activate_search_mode()
                
            else:
                await self.sleep(5)
                if time.minute % 10 == 0:
                    log___(f"===== POSICION ACTIVA ====== size:{position["size"]}, side:{position["side"]}")
                    
        if self.in_search_mode: # Buscar entrada  
            rgx_piv: RGXPIVOT = await self.I(RGXPIVOT, data=ohlc_closed_bars, nbs=True)

            match_pivots = rgx_piv.search(self.rq_pattern)
            if match_pivots:
                # log___(f"Comprobando patron...")

                # Setear los puntos 'A, B, C, D, E' en cada pivot 
                rgx_piv.set_tag(self.rq_pattern, ('a','b','c','d','e')) 
                
                pivot_values = rgx_piv.values
                pivot_idx = rgx_piv.index
                tags = rgx_piv.tags
                # Valores de cada pivot tageado
                self.A = pivot_values[tags == 'a'][0]
                self.B = pivot_values[tags == 'b'][0]
                self.C = pivot_values[tags == 'c'][0]
                self.D = pivot_values[tags == 'd'][0]
                self.E = pivot_values[tags == 'e'][0]
                # Indices de los pivot tageados D y E (el extremo superior e inferior del patron pivote)
                self.E_idx: np.datetime64 = pivot_idx[tags == 'e'][0] 
                self.D_idx: np.datetime64 = pivot_idx[tags == 'd'][0]

                d_is_broken, e_is_broken =  self.d_is_broken(all_data_ohlc, self.D, self.D_idx),  self.e_is_broken(all_data_ohlc, self.E, self.E_idx)
                is_pattern_broken = d_is_broken or e_is_broken

                if is_pattern_broken: # No continuar con la orden
                    await self.sleep(20) 
                    return

                log___(f"--- PATRON COMPLETO --- ; (A):{self.A}, (B):{self.B}, (C):{self.C}, (D):{self.D}, (E):{self.E}")

                # Acceder al margen disponible
                balance_info = (await sesh.get_balance())["coin"][0]
                available_margin = float(balance_info["walletBalance"]) - float(balance_info["totalPositionIM"])

                d2e_diff, a2e_diff = abs(self.D - self.E), abs(self.A - self.E)
                a_is_close_to_e = (a2e_diff < d2e_diff / 2) and (self.A < self.E)
                raw_entry_price = self.A if a_is_close_to_e else self.E

                # Gestion Base
                entry_price, size, tp, sl, leverage, notional  = self.get_order_parameters(available_margin=available_margin, raw_entry_price=raw_entry_price)

                # Setear el apalancamiento maximo
                # adjusted_leverage = self.get_adjusted_leverage(notional_value)
                await sesh.set_leverage(symbol=self.PAIR, leverage=leverage) 
                log___(f"--- Leverage Seteado: {leverage}X --- ")

                # CREAR ORDEN #
                log___(f"Intentando crear Limit Order(pair:{self.PAIR} size:{size} price:{entry_price} tp:{tp}, sl:{sl})...")
                self.order_id = (await sesh.buy(symbol=self.PAIR, price=entry_price, tp_price=tp, sl_price=sl, size=size))["orderId"]

                # Modes
                self.deactivate_search_mode()
                self.activate_order_mode()
                return

        await self.sleep(20)
    async def on_stop(self):
        sesh = self.sesh
        # data_15m = await sesh.get_data(symbol=self.pair, timeframe="15min")
        data_ohlc = await sesh.get_data(symbol=self.PAIR, timeframe=self.STGY_TIMEFRAME)
        await self.I(ZIGZAG, data=data_ohlc)
        await self.I(NBS, data=data_ohlc) 
        # await self.I(RSI, data=data_15m.Close) 

    def d_is_broken(self, ohlc: DataOHLC, d_val: float, d_idx: np.datetime64) -> bool:
        """ Verifica si el punto D ha sido roto por encima del pivot high 'D'. """
        is_broken = any(ohlc.High[ohlc.index > d_idx] >= d_val)
        # if is_broken:
        #     self.log(f"Patron roto en D: el high({max(ohlc.High[ohlc.index > d_idx])}) rompió el punto D({d_val}) ")
        return is_broken
    def e_is_broken(self, ohlc: DataOHLC, e_val: float, e_idx: np.datetime64) -> bool:
        """ Verifica si el punto E ha sido roto por debajo del pivot low 'E'. """
        is_broken = any(ohlc.Low[ohlc.index > e_idx] <= e_val)
        # if is_broken:
        #     self.log(f"Patron roto en punto E: el low({min(ohlc.Low[ohlc.index > e_idx])}) rompió el punto E({e_val}) ")
        return is_broken

    def get_adjusted_size(self, size) -> float:
        instrument_info = self.instrument_info
        step = float(instrument_info["lotSizeFilter"]["qtyStep"])
        max_size = float(instrument_info["lotSizeFilter"]["maxOrderQty"])
        min_size = float(instrument_info["lotSizeFilter"]["minOrderQty"])
        return adjust_size(size, step=step, min_size=min_size, max_size=max_size)
    def get_adjusted_leverage(self, usdt_notional) -> int:
        leverage = adjust_leverage(usdt_notional, self.risk_limit)
        self.log(f"max_leverage: {leverage}")
        return leverage
    def get_order_parameters(self, available_margin, raw_entry_price) -> tuple[float, float, float, float, float, float]:
        """
            Devuelve los parametros de una orden (entry_price, size, tp, sl, leverage, notional ) basados en el precio de entrada.
        """
        # Parametros base
        raw_tp = self.D
        raw_sl = sl_by_ratio(ratio=self.SL_RATIO, entry=raw_entry_price, tp=raw_tp)
        # Ajuste de parametros
        adj_tp = adjust_tp(raw_entry_price, raw_tp, adjustment=self.TP_ADJUSTMENT_VAL)
        adj_sl = adjust_sl(raw_entry_price, raw_sl, adjustment=self.SL_ADJUSTMENT_VAL)
        adj_price = adjust_entry(entry_original=raw_entry_price, sl=adj_sl, adjust_factor=self.ENTRY_ADJUST_FACTOR)
        # Size base
        raw_size = size_by_risk(risk=self.PCT_RISK, cash=available_margin, entry=adj_price, sl=adj_sl)
        # Ajuste de Size
        adj_size = self.get_adjusted_size(raw_size)
        # valor nocional en usdt y apalancamiento ajustado
        notional_value = adj_size * adj_price # usdt
        adjusted_leverage = min(100, self.get_adjusted_leverage(notional_value))

        self.log(f"Margen Disponible: {available_margin:.2f} USDT, Margen requerido: {notional_value / adjusted_leverage:.2f} USDT, Valor Nocional: {notional_value:.2f} USDT")
        
        return adj_price, adj_size, adj_tp, adj_sl, adjusted_leverage, notional_value

    async def on_position_try_get_new_tpsl_orders(self) -> tuple:
        sesh = self.sesh
        try:
            for attempt in range(12):
                new_orders = await sesh.get_new_orders(symbol=self.PAIR)

                tp_mactch = ([order for order in new_orders if order["stopOrderType"] == "TakeProfit"])
                tp_order = tp_mactch[-1] 

                sl_match = ([order for order in new_orders if order["stopOrderType"] == "StopLoss"])
                sl_order = sl_match[-1] 
                
                if tp_order and sl_order:
                    return (tp_order, sl_order)

                self.log(f"[!] No se obtuvo la nueva order TP/SL. Reintentando({attempt})... ")

                await self.sleep(2)
        except Exception as e:
            if (await sesh.get_position(self.PAIR))["size"] == 0:
                self.log(f"[!] Posición cerrada. No se puede obtener el closedPnl.({attempt})... ")
                return "0"
            self.log(f"Posicion abierta: error No se obtuvo la nueva order TP/SL. Reintentando({attempt})... {e}", type="error")
        return (None, None)
    async def on_position_try_get_old_tpsl_orders(self) -> tuple:
        sesh = self.sesh
        try:
            for attempt in range(12):
                new_orders = await sesh.get_all_orders(symbol=self.PAIR)

                tp_mactch = ([order for order in new_orders if order["stopOrderType"] == "TakeProfit"])
                tp_order = tp_mactch[-1] 

                sl_match = ([order for order in new_orders if order["stopOrderType"] == "StopLoss"])
                sl_order = sl_match[-1] 
                
                if tp_order and sl_order:
                    return (tp_order, sl_order)

                self.log(f"[!] No se obtuvo la order TP/SL ya ejecutada. Reintentando({attempt})... ")

                await self.sleep(2)
                
        except Exception as e:
            self.log(f"No se obtuvo la order TP/SL ya ejecutada. Error: {e}", type="error")
        return (None, None)
    async def try_get_closed_pnl(self, order_id, attempts: int = 12) -> dict:
        sesh = self.sesh
        for attempt in range(attempts):
            try:
                closed_pnl_list = await sesh.get_closed_pnl(symbol=self.PAIR)
                match_orders = [order for order in closed_pnl_list if order["orderId"] == order_id]
                if match_orders:
                    return match_orders[-1]["closedPnl"]
                self.log(f"[!] No se obtuvo el closedPnl. Reintentando({attempt})... ")
            except Exception as e:
                if (await sesh.get_position(self.PAIR))["size"] == 0 :
                    self.log(f"[!] Posición cerrada. No se puede obtener el closedPnl. Reintentando({attempt})... ")
                    return "0"
                self.log(f"Error al obtener closed_pnl_list: {e}", type="error")
            await self.sleep(2)
        return None
    async def try_get_tp_or_sl_order(self, order_id, attempts: int = 10) -> list:
        sesh = self.sesh
        for attempt in range(attempts):
            try:
                all_orders_list = await sesh.get_all_orders(symbol=self.PAIR)
                match_orders = [order for order in all_orders_list if order["orderId"] == order_id]
                if match_orders:
                    return match_orders[-1]
            except Exception as e:
                self.log(f"Error al obtener all_orders_list. Reintentando({attempt})... {e}", type="error")
                await self.sleep(2)
        return []

    def activate_order_mode(self) -> None:
        self.log(f"== Creada Orden BUY ==", type='info')
        self.in_order_mode = True
    def activate_position_mode(self) -> None:
        self.log(f"================ NUEVA POSICION ACTIVA ================", type='info')
        self.in_position_mode = True
    def activate_search_mode(self) -> None:
        self.log(f"== Buscando Patron ==", type='strategy')
        # self.in_position_mode = False
        # self.in_order_mode = False
        self.in_search_mode = True

    def deactivate_order_mode(self) -> None:
        self.A, self.B, self.C, self.D, self.E, self.E_idx, self.D_idx = [None]*7
        self.in_order_mode = False
        self.order_id = None
    def deactivate_position_mode(self) -> None:
        self.in_position_mode = False
        self.position_tp_order_id = None 
        self.position_sl_order_id = None 
    def deactivate_search_mode(self) -> None:
        self.in_search_mode = False



