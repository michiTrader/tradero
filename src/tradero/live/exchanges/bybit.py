import pandas as pd
import warnings
import asyncio
import time
from pybit.unified_trading import HTTP
from tradero.models import DataOHLC, CryptoSesh
from tradero.lib import timeframe2minutes


class BybitSesh(CryptoSesh):
    """Sesión de trading para Bybit con funciones para operar"""
    def __init__(self, 
                api_key: str, 
                api_secret: str, 
                category: str, 
                http: 'HTTP' = HTTP, 
                account_type: str ="UNIFIED", # UNIFIED | CONTRACT | SPOT
                tz: str = None, # Zona horaria para convertir timestamps (default: UTC), ej: "UTC-05:00"
                demo: bool = False, 
                base_coin: str = "USDT",
                ):
        """Inicializa la sesión de trading"""
        self._session = http(api_key=api_key, api_secret=api_secret, demo=demo)
        self._category = category
        self._account_type = account_type
        self._base_coin = base_coin
        self._active_strategies = []
        self.indicators = {} # diccionario con varios diccionarios
        self.indicator_blueprints = {}

        self._tz = tz

        print(f"{self.__class__.__name__} Status: categoria: {self._category} | Tipo de cuenta: {self._account_type} | Zona Horaria: {self._tz}")
    
    @property
    async def total_equity(self):
        return float(await self.get_balance["totalEquity"])
    
    @property
    def time_zone(self):
        return self._tz

    @property
    def equity(self):
        return self._session.get_balance(
            cacountType=self._account_type, coin=self._base_coin)["result"]["list"][0]["totalEquity"]

    @property 
    def now(self) -> pd.Timestamp:
        return pd.to_datetime(time.time(), unit='ms').strftime('%Y-%m-%d %H:%M:%S')
    

    """ DATA """
    async def get_kline(self,
        symbol: str,
        timeframe: str = "1D",
        start: str = None,
        end: str = None,
        limit: int = None,
        category:"str" = "linear",
        ) -> list:
        """
            Obtiene datos de klines de manera asíncrona con paginación automática.
            
            Args:
                symbol: Par de trading (ej: 'BTCUSDT')
                timeframe: timeframe de tiempo (ej: '1m', '1h', '1D')
                start: Fecha inicial en formato 'YYYY-MM-DD'
                end: Fecha final en formato 'YYYY-MM-DD'
                limit: Número máximo de klines a retornar
        """
        
        # Constantes
        MAX_BARS_PER_REQUEST = 1000
        MAX_REQUESTS_PER_SECOND = 40
        TIME_UNITS = {
            "m": 1,
            "h": 60,
            "D": 1440,
            "W": 10080,
            "M": 43200
        }

        def timeframe_minutes_to_str_format_timeframe(timeframe_mins: int) -> str:
            """
                Convierte un timeframe en minutos a su representación en string.
                Retorna 'D' para timeframe diarios, 'W' para semanales, 
                'M' para mensuales, o el número de minutos como string para otros casos.
            """
            str_freq = "D" if 1_440 <= timeframe_mins < 10_080 else (
                "W" if 10_080 <= timeframe_mins < 43_200 else (
                    "M" if timeframe_mins >= 43_200 else (
                        str(timeframe_mins)
                    )
                )
            )
            return str_freq
                
        async def fetch_klines(start_ts: int, end_ts: int) -> list:
            return await asyncio.to_thread(
                lambda: self._session.get_kline(
                    symbol=symbol,
                    interval=timeframe_minutes_to_str_format_timeframe(timeframe_minutes),
                    start=start_ts,
                    end=end_ts,
                    limit=MAX_BARS_PER_REQUEST,
                    category=category,
                )["result"]["list"]
            )

        # Validación de entrada
        if not isinstance(timeframe, str):
            raise ValueError("El timeframe debe ser un string")

        # Procesamiento de fechas
        timeframe_minutes = timeframe2minutes(timeframe)
        end_time = pd.to_datetime(end) if end else pd.to_datetime(int(time.time() * 1000), unit="ms")
        start_time = pd.to_datetime(start) if start else end_time - pd.Timedelta(minutes=timeframe_minutes * (limit or 200))

        # Ajuste del tiempo de inicio si hay límite
        if limit:
            duration = end_time - start_time
            max_duration = pd.Timedelta(minutes=timeframe_minutes * limit)
            start_time = start_time if duration <= max_duration else end_time - max_duration

        # Cálculo de segmentos
        total_minutes = int((end_time - start_time).total_seconds() / 60)
        num_requests = max(1, int(-(-total_minutes // (MAX_BARS_PER_REQUEST * timeframe_minutes))))
        minutes_per_segment = total_minutes // num_requests

        # Preparación de tareas
        tasks = []
        current_time = start_time
        
        for r in range(num_requests):
            segment_end = current_time + pd.Timedelta(minutes=minutes_per_segment)
            
            tasks.append(fetch_klines(
                int(current_time.timestamp() * 1000),
                int(segment_end.timestamp() * 1000),
            ))
            
            if r >= MAX_REQUESTS_PER_SECOND:
                time.sleep(1)
                
            current_time = segment_end

        # Obtención y procesamiento de resultados
        try:
            responses = await asyncio.gather(*tasks) #asyncio.run() 
            return [kline for response in responses for kline in response]
        except Exception as e:
            raise Exception(f"Error al obtener klines: {str(e)}")

    async def get_data(self, 
        symbol,
        timeframe, 
        start=None, 
        end=None, 
        limit=None, 
        category="linear",
        tz:str=None
        ) -> DataOHLC:
        """
            (bybit) Obtiene y procesa datos históricos de velas (klines) para un símbolo específico.

            Args:
                session: Sesión activa de la API
                category: Categoría del mercado (ej: 'linear', 'spot')
                symbol: Símbolo del par de trading
                timeframe: timeframe de tiempo para las velas. Puede ser string (ej: '1m', '1h') o minutos
                start: str/timestamp inicial para los datos (opcional)
                end: str/timestamp final para los datos (opcional)
                limit: Número máximo de velas a retornar (opcional)
                tz: Zona horaria para convertir timestamps (default: UTC)

            Returns:
                DataFrame de pandas con los datos de velas procesados
        """
        # Extraer datos timezone general de la session si esta configurado
        tz = self._tz if self._tz else (tz if tz else "UTC")

        kline = (await self.get_kline(symbol, timeframe, start, end, limit, category))
        data = self._process_kline_data_to_frame(kline, tz)

        if data.empty:
            raise ValueError("No se encontraron datos para el símbolo y el timeframe especificados.")
            
        return data

    async def get_last_price(self, symbol) -> float:
        """
            Obtiene el último precio de un par de trading.

            Args:
                session: Sesión activa de la API
                category: Categoría del mercado (ej: 'linear','spot')
                symbol: Símbolo del par de trading

            Returns:
                float: Último precio del par de trading
        """

        ticker = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: self._session.get_tickers(
                category=self._category, 
                symbol=symbol
            )["result"]["list"][0]
        )

        return float(ticker["lastPrice"])

    async def get_time(self, tz: str = None) -> pd.Timestamp:
        # Extraer datos timezone general de la session si esta configurado
        tz = self._tz if self._tz else (tz if tz else "UTC")

        seconds_time = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: self._session.get_server_time()["result"]["timeSecond"]
        )
    
        timestamp = pd.Timestamp(int(seconds_time), unit="s", tz=tz).tz_localize(None)
        return timestamp

 
    """ CONFIGURATION """
    async def set_time_zone(self, tz: str):
        if tz:
            self._tz = tz
        else:
            warnings.warn("Se ha asignado un valor 'None' o Nulo a la zona horaria.", DeprecationWarning)

    async def set_leverage(self, symbol, leverage: int = 1):
        ""
        """
            Establece el nivel de margen (leverage) para un par de trading.
            Args:
                symbol: Símbolo del par de trading
                buy_leverage: Nivel de margen para operaciones de compra (opcional)
                sell_leverage: Nivel de margen para operaciones de venta (opcional)
        """

        assert leverage > 0, "El valor de leverage debe ser mayor que 0."

        if int(await self.get_leverage(symbol)) == int(leverage):
            # print(f"Nivel de apalancamiento para {symbol} ya establecido en {leverage}.")
            return
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self._session.set_leverage(
                    category=self._category,
                    symbol=symbol,
                    buyLeverage=str(leverage),
                    sellLeverage=str(leverage)
                )
            )
            # print (f"Nivel de apalancamiento para {symbol} establecido en {leverage}.")
        except Exception as e:
            raise ValueError(f"No se pudo establecer el nivel de apalancamiento|margen. Error: {str(e)}")

    async def set_margin_mode(self, margin_mode: str = "insolated"):
        """ 
            Establece el modo de margen para la cuenta.
            Args:
                margin_mode: Modo de margen a establecer ('insolated'/'cross'/'portfolio')
            Returns:
                dict: Respuesta de la API con el resultado de la operación
        """
        dict_margin_mode = {"insolated": "ISOLATED_MARGIN", "cross": "REGULAR_MARGIN", "porfolio": "PORTFOLIO_MARGIN"}
        assert margin_mode in dict_margin_mode

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._session.set_margin_mode(setMarginMode=dict_margin_mode[margin_mode]) 
        )
        if response["retMsg"] == 'Request accepted':
            print(f"Modo de margen establecido en {margin_mode}.")
        else:
            raise ValueError("No se pudo establecer el modo de margen.")
        

    """ INFO """
    # NEW 
    async def get_instruments_info(self, symbol) -> dict:
        """
            Obtiene información detallada sobre un instrumento (par de trading).

            Args:
                session: Sesión activa de la API
                category: Categoría del mercado (ej: 'linear','spot')
                symbol: Símbolo del par de trading

            Returns:
                dict: Información detallada del instrumento
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: self._session.get_instruments_info(
                category=self._category, 
                symbol=symbol
            )["result"]["list"][0]
        )

    async def get_account_info(self):
        """
            Obtiene información de la cuenta de trading.
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: self._session.get_account_info()["result"]
        )

    async def get_balance(self, coin: str = "USDT", account_type: str = None):
        """
            Obtiene el balance de una moneda específica en la cuenta.
            
            Args:
                coin: Símbolo de la moneda (default: USDT)
                account_type: Tipo de cuenta a consultar (default: self.account_type)
                
            Returns:
                dict: Balance y detalles de la moneda consultada
        """
        account_type = account_type if account_type else self._account_type
        return await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: self._session.get_wallet_balance(accountType=account_type, coin=coin)["result"]["list"][0]
        )
    # New
    async def get_position(self, symbol) -> dict:
        """
            Obtiene el estado de la posición actual para un par de trading.

            Args:
                symbol: Símbolo del par de trading

            Returns:
                dict: Estado de la posición actual ({"size": size, "side": side})
        """
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._session.get_positions(category=self._category, symbol=symbol)
        )

        position = response["result"]["list"][0]

        return position
    # TODO: eliminar
    async def get_position_status(self, symbol) -> dict:
        """
            Obtiene el estado de la posición actual para un par de trading.

            Args:
                symbol: Símbolo del par de trading

            Returns:
                dict: Estado de la posición actual ({"size": size, "side": side})
        """
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._session.get_positions(category=self._category, symbol=symbol)
        )

        position = response["result"]["list"][0]

        size = float(position["size"])
        side = position["side"]

        return {"size": size, "side": side}
    # NEW
    async def get_risk_limit(self, symbol) -> dict:
        """
            Obtiene el límite de riesgo para un par de trading.
        """
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._session.get_risk_limit(
                category=self._category,
                symbol=symbol
            )["result"]["list"]
        )
    # TODO: NEW agregar a las instrucciones de creacion de sesion
    async def get_new_orders(self, symbol) -> list:
        # Extraer el historial de ordenes en lista
        result_list = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._session.get_open_orders(
                category=self._category,
                symbol=symbol,
                openOnly=0
            )["result"]["list"]
        )
        return result_list[::-1]
    # NEW
    async def get_order_history(self, symbol) -> list:
        # Extraer el historial de ordenes en lista
        result_list = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._session.get_open_orders(
                category=self._category,
                symbol=symbol,
                openOnly=1
            )["result"]["list"]
        )
        return result_list[::-1]
    # MODIFIED
    async def get_all_orders(self, symbol) -> list:
        """
            Obtiene el historial de ordenes (en forma de diccionario) referente a un par de trading.
        """

        # Extraer el historial de ordenes en lista de forma concurrente
        open_orders_not_filled_task = asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._session.get_open_orders(
                category=self._category,
                symbol=symbol,
                openOnly=0
            )["result"]["list"]
        )

        open_orders_filled_task = asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._session.get_open_orders(category=self._category,
                symbol=symbol,
                openOnly=1
            )["result"]["list"]
        )
        
        # Ejecutar ambas requests concurrentemente
        open_orders_not_filled, open_orders_filled = await asyncio.gather(
            open_orders_not_filled_task,
            open_orders_filled_task
        )
                                                
        order_list = (open_orders_not_filled + open_orders_filled)[::-1]

        return order_list

    async def get_order(self, symbol, order_id, max_attempts=15, wait_time=0.1) -> dict:
        all_orders = await self.get_all_orders(symbol)
        if all_orders:
            matching_orders = [d for d in all_orders if d["orderId"] == order_id]
            if matching_orders:
                return matching_orders[0] 
        return None

    async def get_leverage(self, symbol: str) -> int:
        """
            Obtiene el nivel de apalancamiento (leverage) de un par de trading.
            Args:
                symbol: Símbolo del par de trading
            Returns:
                float: Nivel de apalancamiento del par de trading, o 0.0 si no se encuentra posición.
        """
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._session.get_positions(category=self._category,  symbol=symbol)
            )

            if response["result"]["list"]:
                position = response["result"]["list"][0]
                return int(position["leverage"])
            else:
                print(f"No se encontró posición para el símbolo {symbol}. Retornando apalancamiento 0.")
                return 0
        except Exception as e:
            print(f"Error al obtener el apalancamiento para {symbol}: {e}")
            return 0
    # NEW
    async def get_closed_pnl(self, symbol: str) -> list:
        """
            Obtiene el historial de pnl cerrado para un par de trading.
            Args:
                symbol: Símbolo del par de trading
            Returns:
                list: Historial de pnl cerrado para el par de trading.
        """
        list_result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._session.get_closed_pnl(
                category=self._category,
                symbol=symbol
            )["result"]["list"]
        )
        return list_result[::-1]


    """ Trading """
    async def _place_order(self,
                    symbol,
                    side, 
                    qty,  
                    price=None, 
                    sl_price=None, 
                    tp_price=None, 
                    pct_sl=None, 
                    pct_tp=None, 
                    time_in_force="GTC",
                    ) -> dict:
        """
            Coloca una orden de trading con las siguientes características:
            
            - Permite establecer órdenes de mercado, límite o stop
            - Los stop loss (SL) y take profit (TP) se especifican en precio absoluto
            - No acepta SL/TP en porcentajes directamente
            - Configurable para cualquier par de trading y categoría
            - El time in force por defecto es GTC (Good Till Cancel)
            
            Args:
                session: Sesión activa de trading
                category (str): Categoría del par (linear/inverse)
                symbol (str): Par de trading (ej: 'BTCUSDT')
                side (str): Dirección de la orden ('Buy'/'Sell') 
                order_type (str): Tipo de orden ('market'/'limit'/'stop')
                size (float): Tamaño de la posición
                price (float, optional): Precio límite para órdenes límite
                stop_price (float, optional): Precio de activación para órdenes stop
                sl (float, optional): Precio del stop loss
                tp (float, optional): Precio del take profit
                time_in_force (str): Validez de la orden (default: 'GTC')

            Returns:
                dict: Respuesta de la orden colocada
        """ 

        is_limit_order = bool(price)

        is_percentage_tp_sl = bool(pct_sl) or bool(pct_tp)
        is_price_tp_sl = bool(tp_price) or bool(sl_price)
        
        # Validacion y Configuracion del tipo de orden
        if is_limit_order:
            assert isinstance(price, (int, float)), "El precio debe ser un número flotante o enterto"
            assert price > 0, "El precio debe ser mayor a 0"
            order_type = "limit"
        else:
            order_type = "market"

        # Validacion de TP y SL price (opcional)
        if is_price_tp_sl:
            assert (tp_price > 0) or (sl_price > 0), "El precio de TP y SL debe ser un numero positivo"
        
        # Convertir a string como requrimiento de la API
        str_sl_price = str(sl_price) if sl_price else None
        str_tp_price = str(tp_price) if tp_price else None
        str_qty = str(qty)
        # Lanzar orden
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._session.place_order(
                category=self._category,
                symbol=symbol,
                side=side,
                orderType=order_type,
                qty=str_qty, # Convertir a string como requrimiento de la API
                price=price,
                stopLoss=str_sl_price, # Convertir a string como requrimiento de la API
                takeProfit=str_tp_price, # Convertir a string como requrimiento de la API
                timeInForce=time_in_force, #PostOnly, GTC
                triggerBy="LastPrice",
            )
        )

        # Modificar el TP/SL de la orden en base al avgPrice de la orden ya enviada
        if is_percentage_tp_sl:
            assert (pct_tp > 0) or (pct_sl > 0), "El porcentaje de TP y SL debe ser un numero positivo"

            # Obtener el orderId de la orden
            order_id = response["result"]["orderId"]

            # Obtener la orden y su datos 
            response_order = await self.get_order(symbol=symbol, order_id=order_id)
            # Obtener el avg_price y el size de la orden
            order_avg_price = float(response_order["avgPrice"])
            order_qty = float(response_order["qty"])
            
            # Calcular el precio de TP/SL porcentual en base al avg price
            if pct_tp: 
                tp_price = (order_avg_price * ((1 + pct_tp)) if side == "Buy" else (order_avg_price * (1 - pct_tp)))
            if pct_sl:
                sl_price = (order_avg_price * ((1 - pct_sl)) if side == "Buy" else (order_avg_price * (1 + pct_sl)))

            # Convertir a string como requrimiento de la API
            str_sl_price = str(sl_price) if (sl_price or pct_sl) else None
            str_tp_price = str(tp_price) if (tp_price or pct_tp) else None
            str_order_qty = str(order_qty) 
            # Modificar la orden con el precio de TP/SL
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self._session.set_trading_stop(
                    category=self._category,
                    symbol=symbol,
                    stopLoss=str_sl_price, 
                    takeProfit=str_tp_price, # Convertido a string como requrimiento de la API
                    slSize=str_order_qty, # Convertido a string como requrimiento de la API
                    tpSize=str_order_qty, # Convertido a string como requrimiento de la API
                )
            )

        return response
        
    async def sell(self,
            symbol: str,
            size: float,
            price: float = None,
            sl_price: float = None,
            tp_price: float = None,
            pct_sl: float = None,
            pct_tp: float = None,
            time_in_force: str ="GTC",
            ) -> dict:    
        """
            Coloca una orden de compra con las siguientes características:
            
            Args:
                session: Sesión activa de trading
                category (str): Categoría del par (linear/inverse)
                symbol (str): Par de trading (ej: 'BTCUSDT')
                qty (float): Cantidad a comprar
                price (float, optional): Precio límite para órdenes límite
                sl_price (float, optional): Precio absoluto del stop loss
                tp_price (float, optional): Precio absoluto del take profit  
                pct_sl (float, optional): Porcentaje de stop loss relativo al precio de entrada
                pct_tp (float, optional): Porcentaje de take profit relativo al precio de entrada

            Returns:
                dict: Información de la orden ejecutada
        """
        resp = await self._place_order(
                            symbol=symbol,
                            side="Sell",
                            qty=size,
                            price=price,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            pct_sl=pct_sl,
                            pct_tp=pct_tp,
                            time_in_force=time_in_force,
                        )

        # order = await self.get_order(symbol, response["result"]["orderId"])
        # return order

        return resp["result"]

    async def buy(self,
            symbol: str,
            size: float,
            price: float = None,
            sl_price: float = None,
            tp_price: float = None,
            pct_sl: float = None,
            pct_tp: float = None,
            time_in_force: str ="GTC",
            ) -> dict:    
        """
            Coloca una orden de compra con las siguientes características:
            
            Args:
                session: Sesión activa de trading
                category (str): Categoría del par (linear/inverse)
                symbol (str): Par de trading (ej: 'BTCUSDT')
                qty (float): Cantidad a comprar
                price (float, optional): Precio límite para órdenes límite
                sl_price (float, optional): Precio absoluto del stop loss
                tp_price (float, optional): Precio absoluto del take profit  
                pct_sl (float, optional): Porcentaje de stop loss relativo al precio de entrada
                pct_tp (float, optional): Porcentaje de take profit relativo al precio de entrada

            Returns:
                dict: Información de la orden ejecutada
        """
        resp = await self._place_order(
            symbol=symbol,
            side="Buy",
            qty=size,
            price=price,
            sl_price=sl_price,
            tp_price=tp_price,
            pct_sl=pct_sl,
            pct_tp=pct_tp,
            time_in_force=time_in_force,
        )

        # order = await self.get_order(symbol, response["result"]["orderId"])
        # return order
        # TODO: !!!!! en bybit tambien debería retornar solo la order id
        return resp["result"]

    async def close_position(self, symbol) -> (dict | None):
        position_info = await self.get_position_status(symbol=symbol)
        current_size = position_info["size"]
        current_side = position_info["side"]

        if current_size > 0:
            if current_side == "Buy": # Si tienes una posición larga, vende para cerrar
                # print(f"Cerrando posición larga de {current_size} {symbol}...")
                resp = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self._session.place_order(
                        category=self._category,
                        symbol=symbol,
                        side="Sell",
                        orderType="Market",
                        qty=str(current_size), # Cantidad para cerrar la posición
                    )
                )
                return resp["result"]
                # print(f"Orden de cierre (venta) ejecutada: {order}")
            elif current_side == "Sell": # Si tienes una posición corta, compra para cerrar
                # print(f"Cerrando posición corta de {current_size} {symbol}...")
                resp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._session.place_order(
                        category=self._category,
                        symbol=symbol,
                        side="Buy",
                        orderType="Market",
                        qty=str(current_size), # Cantidad para cerrar la posición
                    )
                )
                return resp["result"]
                # print(f"Orden de cierre (compra) ejecutada: {order}")
        else:
            # print(f"No hay posición abierta para {symbol}.")
            pass

    async def set_trading_stop(self, 
                        symbol: str,
                        tp_price: float = None,
                        sl_price: float = None,
                        tp_limit_price: float = None,
                        sl_limit_price: float = None,
                        tpsl_mode: str = None, # partial, None
                        tp_order_type: str = None, # Limit, Market(default)
                        sl_order_type: str = None,
                        tp_size: float = None,
                        sl_size: float = None,
                        tp_trigger_by: str = "MarkPrice", # MarkPrice, IndexPrice, LastPrice
                        sl_trigger_by: str = "IndexPrice",
                        position_idx: int = 0,
                        ) -> None:
        """
            Configura los niveles de take profit y stop loss para una posición abierta.

            Args:
                session (object): Sesión de trading activa
                category (str): Categoría del instrumento (ej: 'linear', 'spot')
                symbol (str): Símbolo del par de trading
                qty (float): Cantidad a operar
                takeProfit (float, opcional): Nivel de precio para take profit
                stopLoss (float, opcional): Nivel de precio para stop loss
                tpLimitPrice (float, opcional): Precio límite para la orden de take profit
                slLimitPrice (float, opcional): Precio límite para la orden de stop loss
                tpslMode (str, opcional): Modo de ejecución TP/SL ('Full' o 'Partial')
                tpOrderType (str, opcional): Tipo de orden para take profit
                slOrderType (str, opcional): Tipo de orden para stop loss
                tpTriggerBy (str, opcional): Precio de referencia para activar take profit
                slTriggerB (str, opcional): Precio de referencia para activar stop loss
                positionIdx (int, opcional): Identificador de posición (0: unidireccional, 1: hedge-mode Buy, 2: hedge-mode Sell)

            Raises:
                ValueError: Si no existe una posición abierta para configurar
        """
        # Verificar si hay una posición abierta para configurar
        position_statur = (await self.get_position_status(symbol))["size"]
        if position_statur == 0:
            raise ValueError("No hay posición abierta para configurar TP/SL.") 
    
        # Configurar salida Parcial si se especifica tp_size o sl_size
        if (tp_size or sl_size):
            tpsl_mode = "Partial"
            if tp_size:
                tp_order_type = "limit"
                tp_limit_price = tp_price
            if sl_size:
                sl_order_type = "limit"
                sl_limit_price = sl_price

        # Setear TP/SL
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._session.set_trading_stop(
                category=self._category,
                symbol=symbol,
                takeProfit=str(tp_price) if tp_price else None,
                stopLoss=str(sl_price) if sl_price else None,
                tpTriggerBy=tp_trigger_by,
                slTriggerB=sl_trigger_by,
                tpslMode=tpsl_mode,
                tpOrderType=tp_order_type,
                slOrderType=sl_order_type,
                tpSize=str(tp_size) if tp_size else None,
                slSize=str(sl_size) if sl_size else None,
                tpLimitPrice=str(tp_limit_price) if tp_limit_price else None,
                slLimitPrice=str(sl_limit_price) if sl_limit_price else None,
                positionIdx=position_idx,
            )
        )

        return resp

    async def cancel_all_orders(self, symbol, order_filter: str = None) -> dict:
        """
            Cancela todas las órdenes abiertas para un símbolo dado.

            Args:
                category (str): Categoría del símbolo (por ejemplo, "linear", "Spot").
                symbol (str): Símbolo para el que se desean cancelar las órdenes.
                order_filter (str): category=spot, puedes pasar 'Order', tpslOrder, 'StopOrder', 'OcoOrder', 'BidirectionalTpslOrder'
                                        Si no se especifica, 'Order' por defecto
                                    category=linear o inverse, puedes pasar 'Order', 'StopOrder', 'OpenOrder'
                                        Si no se especifica, se cancelarán todo tipo de órdenes, como órdenes activas,
                                        órdenes condicionales, órdenes TP/SL y órdenes de trailing stop
                                    category=option, puedes pasar Order.
        """
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._session.cancel_all_orders(
                category=self._category, 
                symbol=symbol, 
                orderFilter=order_filter
            )
        )
        return resp
    # NEW
    async def cancel_order(self, symbol, order_id):
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._session.cancel_order(
                category=self._category, 
                symbol=symbol, 
                orderId=order_id
            )
        )
        return resp["result"]




