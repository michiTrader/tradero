import pandas as pd
import warnings
import asyncio
import time
from datetime import datetime, timezone, timedelta 
import MetaTrader5 as mt5
from typing import Union
from tradero.models import DataOHLC

class MT5Sesh:
    """Sesión de trading para MetaTrader 5 con funciones para operar (Forex y Futuros)"""
    
    # Mapeo de timeframes string a constantes MT5
    # Soporta ambos formatos: "5m"/"5min", "1h"/"1hour", "1D"/"1day"
    TIMEFRAME_MAP = {
        # Formato corto (compatibilidad)
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "2h": mt5.TIMEFRAME_H2,
        "4h": mt5.TIMEFRAME_H4,
        "1D": mt5.TIMEFRAME_D1,
        "1W": mt5.TIMEFRAME_W1,
        "1M": mt5.TIMEFRAME_MN1,
        # Formato Bybit (principal)
        "1min": mt5.TIMEFRAME_M1,
        "3min": mt5.TIMEFRAME_M1,
        "5min": mt5.TIMEFRAME_M5,
        "15min": mt5.TIMEFRAME_M15,
        "30min": mt5.TIMEFRAME_M30,
        "1hour": mt5.TIMEFRAME_H1,
        "2hour": mt5.TIMEFRAME_H2,
        "4hour": mt5.TIMEFRAME_H4,
        "1day": mt5.TIMEFRAME_D1,
        "1week": mt5.TIMEFRAME_W1,
        "1month": mt5.TIMEFRAME_MN1,
    }
    
    def __init__(self, 
                account: int = None,
                password: str = None, 
                server: str = None,
                path: str = None,
                market_type: str = "forex",
                tz: str = None,
                magic_number: int = 234000,
                ):
        """
        Inicializa la sesión de trading para MT5
        
        Args:
            account: Número de cuenta de MT5
            password: Contraseña de la cuenta
            server: Servidor del broker
            path: Ruta al terminal MT5 (opcional, se busca automáticamente)
            market_type: Tipo de mercado ("forex" o "futures")
            tz: Zona horaria para convertir timestamps (default: UTC)
            magic_number: Número mágico para identificar las órdenes del bot
        """
        self._account = account
        self._password = password
        self._server = server
        self._path = path
        self._market_type = market_type.lower()
        self._tz = tz if tz else "UTC"
        self._magic_number = magic_number
        self._active_strategies = []
        self.indicators = {}
        self.indicator_blueprints = {}
        
        if self._market_type not in ["forex", "futures"]:
            raise ValueError("market_type debe ser 'forex' o 'futures'")
        
        # Inicializar MT5
        if not mt5.initialize():
            if path:
                if not mt5.initialize(path=path):
                    raise ConnectionError(
                        f"Error al inicializar MT5: {mt5.last_error()}\n"
                        f"Verifica que MetaTrader 5 esté instalado y la ruta sea correcta.\n"
                        f"Path intentado: {path}"
                    )
            else:
                raise ConnectionError(
                    f"Error al inicializar MT5: {mt5.last_error()}\n"
                    "Verifica que MetaTrader 5 esté instalado.\n"
                    "Si está en una ubicación no estándar, proporciona el parámetro 'path'."
                )
        
        # Login
        if account and password and server:
            account_int = int(account) if isinstance(account, str) else account
            
            if not mt5.login(account_int, password=password, server=server):
                error = mt5.last_error()
                mt5.shutdown()
                raise ConnectionError(
                    f"Error al hacer login en MT5: {error}\n"
                    f"Cuenta: {account_int}\n"
                    f"Servidor: {server}\n"
                    "Verifica tus credenciales y que el servidor sea correcto."
                )
        
        account_info = mt5.account_info()
        if account_info is None:
            mt5.shutdown()
            raise ConnectionError("No se pudo obtener información de la cuenta")
        
        print(f"{self.__class__.__name__} Status: Cuenta: {account_info.login} | "
              f"Broker: {account_info.company} | Mercado: {self._market_type.upper()} | "
              f"Zona Horaria: {self._tz}")
    
    def __del__(self):
        """Cerrar conexión MT5 al destruir el objeto"""
        mt5.shutdown()
    
    @property
    def time_zone(self):
        return self._tz
    
    @property
    def market_type(self):
        return self._market_type
    
    @property
    def equity(self):
        """Obtiene el equity actual de la cuenta"""
        account_info = mt5.account_info()
        return account_info.equity if account_info else 0.0
    
    @property
    def now(self) -> pd.Timestamp:
        """Obtiene el timestamp actual"""
        return pd.Timestamp.now(tz=self._tz).tz_localize(None)
    
    """ UTILITIES """
    
    def _ensure_connection(self) -> bool:
        """Verifica y restablece la conexión con MT5 si es necesario"""
        terminal_info = mt5.terminal_info()
        
        # Si no hay información del terminal, intentar reconectar
        if terminal_info is None:
            if not mt5.initialize():
                if self._path and not mt5.initialize(path=self._path):
                    return False
                elif not self._path:
                    return False
            
            # Re-login si tenemos credenciales
            if self._account and self._password and self._server:
                account_int = int(self._account) if isinstance(self._account, str) else self._account
                if not mt5.login(account_int, password=self._password, server=self._server):
                    return False
            
            terminal_info = mt5.terminal_info()
        
        # Verificar si está conectado
        return terminal_info is not None and terminal_info.connected
    
    def _get_filling_mode(self, symbol: str) -> int:
        """Determina el modo de llenado apropiado para el símbolo"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return mt5.ORDER_FILLING_FOK
        
        filling = symbol_info.filling_mode
        
        if self._market_type == "futures":
            if filling & 1:
                return mt5.ORDER_FILLING_FOK
            elif filling & 2:
                return mt5.ORDER_FILLING_IOC
        
        if filling & 2:
            return mt5.ORDER_FILLING_IOC
        elif filling & 1:
            return mt5.ORDER_FILLING_FOK
        elif filling & 4:
            return mt5.ORDER_FILLING_RETURN
        
        return mt5.ORDER_FILLING_FOK
    
    def diagnose_connection(self) -> dict:
        """Diagnostica el estado de la conexión MT5 y proporciona información útil"""
        diagnosis = {
            "mt5_initialized": False,
            "terminal_connected": False,
            "account_info": None,
            "terminal_info": None,
            "last_error": None,
            "recommendations": []
        }
        
        # Verificar si MT5 está inicializado
        terminal_info = mt5.terminal_info()
        if terminal_info is not None:
            diagnosis["mt5_initialized"] = True
            diagnosis["terminal_info"] = {
                "name": terminal_info.name,
                "version": terminal_info.version,
                "build": terminal_info.build,
                "connected": terminal_info.connected,
                "trade_allowed": terminal_info.trade_allowed,
                "path": terminal_info.path
            }
            diagnosis["terminal_connected"] = terminal_info.connected
        else:
            diagnosis["recommendations"].append("MT5 no está inicializado. Reinicia la sesión.")
        
        # Verificar información de cuenta
        account_info = mt5.account_info()
        if account_info is not None:
            diagnosis["account_info"] = {
                "login": account_info.login,
                "server": account_info.server,
                "company": account_info.company,
                "trade_allowed": account_info.trade_allowed,
                "trade_expert": account_info.trade_expert
            }
        else:
            diagnosis["recommendations"].append("No se puede obtener información de cuenta.")
        
        # Obtener último error
        last_error = mt5.last_error()
        if last_error[0] != 0:  # 0 significa sin error
            diagnosis["last_error"] = {
                "code": last_error[0],
                "description": last_error[1]
            }
            diagnosis["recommendations"].append(f"Error MT5: {last_error[1]}")
        
        # Recomendaciones adicionales
        if not diagnosis["terminal_connected"]:
            diagnosis["recommendations"].extend([
                "Verifica que MetaTrader 5 esté abierto y conectado al servidor",
                "Revisa tu conexión a internet",
                "Verifica las credenciales de login"
            ])
        
        return diagnosis

    def _normalize_volume(self, symbol: str, volume: float) -> float:
        """Normaliza el volumen según los requisitos del símbolo"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return volume
        
        volume_step = symbol_info.volume_step
        volume = round(volume / volume_step) * volume_step
        volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))
        
        return volume
    
    def _normalize_price(self, symbol: str, price: float) -> float:
        """Normaliza el precio según los requisitos del símbolo"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return price
        
        digits = symbol_info.digits
        return round(price, digits)
    
    def _format_order(self, order, is_history: bool = False) -> dict:
        """Formatea una orden de MT5 al formato estándar"""
        # Determinar side
        is_buy = order.type in [
            mt5.ORDER_TYPE_BUY, 
            mt5.ORDER_TYPE_BUY_LIMIT, 
            mt5.ORDER_TYPE_BUY_STOP,
            mt5.ORDER_TYPE_BUY_STOP_LIMIT
        ]
        side = "Buy" if is_buy else "Sell"
        
        # Determinar type basado en el tipo de orden MT5
        order_type = ""
        
        if order.type in [mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL]:
            order_type = "Market"
        elif order.type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT]:
            order_type = "Limit"
        elif order.type in [mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_SELL_STOP]:
            order_type = "Stop"
        elif order.type in [mt5.ORDER_TYPE_BUY_STOP_LIMIT, mt5.ORDER_TYPE_SELL_STOP_LIMIT]:
            order_type = "StopLimit"
        
        # Determinar status
        if is_history:
            if order.state == mt5.ORDER_STATE_FILLED:
                status = "Filled"
            elif order.state == mt5.ORDER_STATE_CANCELED:
                status = "Cancelled"
            else:
                status = "Rejected"
        else:
            # Para órdenes pendientes
            status = "Placed"
        
        return {
            "orderId": str(order.ticket),
            "symbol": order.symbol,
            "side": side,
            "type": order_type,
            "volume": float(order.volume_current),
            "price": float(order.price_open),
            "sl": float(order.sl) if order.sl else 0.0,
            "tp": float(order.tp) if order.tp else 0.0,
            "time": int(order.time_setup),
            "status": status,
            "magic": int(order.magic),
            "comment": order.comment if order.comment else "",
        }
    
    def _process_kline_data_to_data_ohlc(self, klines, symbol, timeframe, tz:str = None) -> DataOHLC:
        """
            Procesa datos de velas (klines) y los convierte a un DataFrame de pandas.
            
            Args:
                response: Respuesta de la API con datos de velas
                timezone: Zona horaria para convertir timestamps (default: UTC)
                
            Returns:
                DataFrame con datos de velas procesados
        """
        # Extraer datos timezone general de la session si esta configurado
        tz = self._tz if self._tz else (tz if tz else "UTC")

        if len(klines[0]) < 7:
            klines = [k + [0] for k in klines]

        # Crear dataframe con los datos de lista
        df = pd.DataFrame(klines, 
            columns=["datetime", "Open", "High", "Low", "Close", "Volume", "Turnover"])

        # Convertir columnas a float
        df = df.astype(float) 

        # Setear indice a datetime
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True) 
        # Convertir zona horaria
        if tz != "UTC":
            df.index = df.index.tz_convert(tz)
        df.index = df.index.tz_localize(None)

        # ordenar por indices
        df = df.sort_index(ascending=True).drop_duplicates()

        content = df.dropna().reset_index().to_dict(orient="list")

        return DataOHLC(content=content, timeframe=timeframe, symbol=symbol)


    """ DATA """
    
    async def get_kline(self,
                       symbol: str,
                       timeframe: str = "1day",
                       start: str = None,
                       end: str = None,
                       limit: int = 100,
                       ) -> list:
        """Obtiene datos de velas (klines) de MT5"""
        
        def fetch_rates():
            tf = MT5Sesh.TIMEFRAME_MAP.get(timeframe)
            if tf is None:
                available_tfs = ', '.join(MT5Sesh.TIMEFRAME_MAP.keys())
                raise ValueError(
                    f"Timeframe '{timeframe}' no soportado.\n"
                    f"Timeframes disponibles: {available_tfs}"
                )
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                raise ValueError(f"Símbolo {symbol} no encontrado")
            
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    raise ValueError(f"No se pudo seleccionar {symbol}")
            
            if end:
                date_to = pd.to_datetime(end).to_pydatetime()
            else:
                date_to = datetime.now()
            
            if start:
                date_from = pd.to_datetime(start).to_pydatetime()
                rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
            elif limit:
                rates = mt5.copy_rates_from(symbol, tf, date_to, limit)
            else:
                rates = mt5.copy_rates_from(symbol, tf, date_to, 200)
            
            if rates is None or len(rates) == 0:
                error = mt5.last_error()
                raise ValueError(f"No se pudieron obtener datos para {symbol}. Error MT5: {error}")
            
            return rates
        
        rates = await asyncio.to_thread(fetch_rates)
        
        klines = []
        for rate in rates:
            klines.append([
                int(rate['time'] * 1000),
                str(rate['open']),
                str(rate['high']),
                str(rate['low']),
                str(rate['close']),
                str(rate['tick_volume']),
            ])
        
        return klines

    async def get_data(self, 
        symbol,
        timeframe, 
        start=None, 
        end=None, 
        limit=100, 
        tz:str=None
        ) -> DataOHLC:
        """
            (mt5) Obtiene y procesa datos históricos de velas (klines) para un símbolo específico.

            Args:
                session: Sesión activa de la API
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

        kline = (await self.get_kline(symbol, timeframe, start, end, limit))
        data_ohlc = self._process_kline_data_to_data_ohlc(kline, symbol, timeframe, tz)

        if data_ohlc.empty:
            raise ValueError("No se encontraron datos para el símbolo y el timeframe especificados.")
            
        return data_ohlc
    
    async def get_last_price(self, symbol: str) -> float:
        """Obtiene el último precio de un instrumento"""
        def _get_tick():
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise ValueError(f"No se pudo obtener el precio de {symbol}")
            
            # Si tick.last es válido (no es 0.0), lo usamos
            if tick.last > 0:
                return tick.last
            
            # Si tick.last es 0.0, usamos el precio medio entre bid y ask
            if tick.bid > 0 and tick.ask > 0:
                return (tick.bid + tick.ask) / 2
            
            # Como último recurso, usar bid o ask si alguno está disponible
            if tick.bid > 0:
                return tick.bid
            if tick.ask > 0:
                return tick.ask
                
            raise ValueError(f"No se pudo obtener un precio válido para {symbol}")
        
        return await asyncio.to_thread(_get_tick)
    
    async def get_time(self, tz: str = None) -> pd.Timestamp:
        """Obtiene el tiempo actual del servidor MT5"""
        tz = self._tz if self._tz else (tz if tz else "UTC")
        
        def _get_server_time():
            symbols = mt5.symbols_get()
            if symbols and len(symbols) > 0:
                tick = mt5.symbol_info_tick(symbols[0].name)
                if tick:
                    return tick.time
            return int(time.time())
        
        timestamp = await asyncio.to_thread(_get_server_time)
        return pd.Timestamp(timestamp, unit="s", tz=tz).tz_localize(None)
    
    """ CONFIGURATION """
    
    async def set_time_zone(self, tz: str):
        """Establece la zona horaria para la sesión"""
        if tz:
            self._tz = tz
        else:
            warnings.warn("Se ha asignado un valor 'None' o Nulo a la zona horaria.", DeprecationWarning)
    
    async def set_leverage(self, symbol: str, leverage: int = 1):
        """Nota: MT5 no permite cambiar el apalancamiento programáticamente"""
        assert leverage > 0, "El valor de leverage debe ser mayor que 0."
        
        def _check_symbol():
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                raise ValueError(f"Símbolo {symbol} no encontrado")
            
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    raise ValueError(f"No se pudo seleccionar el símbolo {symbol}")
        
        await asyncio.to_thread(_check_symbol)
        
        if self._market_type == "futures":
            warnings.warn(
                "En futuros MT5, el apalancamiento está determinado por el tamaño del contrato "
                "y los requisitos de margen del broker.", 
                UserWarning
            )
        else:
            warnings.warn(
                "MT5 no permite cambiar el apalancamiento programáticamente. "
                "Configúralo en la plataforma del broker.", 
                UserWarning
            )
    
    """ INFO """

    async def is_market_open(self, symbol: str) -> bool:
        def _check():
            info = mt5.symbol_info(symbol)
            tick = mt5.symbol_info_tick(symbol)
            
            # --- Si el símbolo no existe ---
            if info is None:
                return False
            # --- Si el símbolo no permite trading ---
            if info.trade_mode not in (
                mt5.SYMBOL_TRADE_MODE_FULL,
                mt5.SYMBOL_TRADE_MODE_LONGONLY,
                mt5.SYMBOL_TRADE_MODE_SHORTONLY,
            ):
                return False

            # --- Si no hay tick reciente ---
            if tick is None:
                return False
            last_tick_time = datetime.fromtimestamp(tick.time)
            if datetime.now() - last_tick_time > timedelta(minutes=5):
                return False

            # --- Si pasa todas las comprobaciones ---
            return True

        return await asyncio.to_thread(_check)

    async def get_instruments_info(self, symbol: str) -> dict:
        """Obtiene información detallada sobre un instrumento"""
        
        def _get_info():
            # Verificar y restablecer conexión si es necesario
            if not self._ensure_connection():
                raise ConnectionError("No se pudo establecer conexión con MT5")
            
            # Intentar obtener información del símbolo
            info = mt5.symbol_info(symbol)
            if info is None:
                # Intentar seleccionar el símbolo en Market Watch
                if not mt5.symbol_select(symbol, True):
                    raise ValueError(f"No se pudo seleccionar el símbolo {symbol} en Market Watch")
                
                # Intentar obtener información nuevamente
                info = mt5.symbol_info(symbol)
                if info is None:
                    raise ValueError(f"No se encontró información para {symbol}")
            
            # Verificar que el símbolo esté visible en Market Watch
            if not info.visible:
                if not mt5.symbol_select(symbol, True):
                    raise ValueError(f"No se pudo hacer visible el símbolo {symbol}")
                
                # Obtener información actualizada
                info = mt5.symbol_info(symbol)
                if info is None:
                    raise ValueError(f"No se encontró información para {symbol} después de seleccionarlo")
            
            return {
                "symbol": info.name,
                "description": info.description,
                "digits": info.digits,
                "tradeContractVolume": float(info.trade_contract_size),
                "tradeTickVolume": float(info.trade_tick_size),
                "tradeTickValue": float(info.trade_tick_value),
                "volumeMin": float(info.volume_min),
                "volumeMax": float(info.volume_max),
                "volumeStep": float(info.volume_step),
                "currencyBase": info.currency_base,
            }
        
        return await asyncio.to_thread(_get_info)
    
    async def get_account_info(self) -> dict:
        """Obtiene información de la cuenta de trading"""
        def _get_info():
            info = mt5.account_info()
            if info is None:
                raise ValueError("No se pudo obtener información de la cuenta")
            
            return {
                "login": info.login,
                "tradeMode": info.trade_mode,
                "leverage": info.leverage,
                "balance": float(info.balance),
                "equity": float(info.equity),
                "margin": float(info.margin),
                "marginFree": float(info.margin_free),
                "marginLevel": float(info.margin_level) if info.margin_level else 0.0,
                "pnl": float(info.profit),
                "currency": info.currency,
                "company": info.company,
                "server": info.server,
                "marketType": self._market_type,
            }
        
        return await asyncio.to_thread(_get_info)
    
    async def get_position(self, symbol: str) -> dict:
        """Obtiene la posición actual para un símbolo"""
        def _get_pos():
            positions = mt5.positions_get(symbol=symbol)
            
            if positions is None or len(positions) == 0:
                return {
                    "volume": 0.0,
                    "side": "None",
                    "price": 0.0,
                    "pnl": 0.0,
                }
            
            total_volume_buy = sum(p.volume for p in positions if p.type == mt5.ORDER_TYPE_BUY)
            total_volume_sell = sum(p.volume for p in positions if p.type == mt5.ORDER_TYPE_SELL)
            
            net_volume = total_volume_buy - total_volume_sell
            side = "Buy" if net_volume > 0 else ("Sell" if net_volume < 0 else "")
            
            if net_volume != 0:
                relevant_positions = [p for p in positions if (
                    (p.type == mt5.ORDER_TYPE_BUY and net_volume > 0) or
                    (p.type == mt5.ORDER_TYPE_SELL and net_volume < 0)
                )]
                avg_price = sum(p.price_open * p.volume for p in relevant_positions) / abs(net_volume)
            else:
                avg_price = 0.0
            
            total_profit = sum(p.profit for p in positions)
            
            return {
                "volume": float(abs(net_volume)),
                "side": side,
                "price": float(avg_price),
                "pnl": float(total_profit),
            }
        
        return await asyncio.to_thread(_get_pos)
    
    async def get_all_orders(self, symbol: str) -> list:
        """Obtiene todas las órdenes (abiertas e históricas) para un símbolo"""
        def _get_orders():
            orders_list = []
            
            # Órdenes pendientes
            orders = mt5.orders_get(symbol=symbol)
            if orders:
                for order in orders:
                    orders_list.append(self._format_order(order, is_history=False))
            
            # Historial de órdenes
            from_date = datetime.now() - pd.Timedelta(days=30)
            to_date = datetime.now()
            history = mt5.history_orders_get(from_date, to_date, group=symbol)
            # history = mt5.history_orders_get(group=symbol)
            
            if history:
                for order in history:
                    orders_list.append(self._format_order(order, is_history=True))
            
            # Ordenar por tiempo (más antigua primero, más reciente al final)
            orders_list.sort(key=lambda x: x['time'])
            
            return orders_list
        
        return await asyncio.to_thread(_get_orders)
    
    async def get_new_orders(self, symbol: str) -> list:
        """
        Obtiene órdenes con status Placed o Untriggered.
        NOTA: En MT5, los TP/SL que se configuran directamente en la posición
        NO aparecen como órdenes pendientes hasta que se activan.
        Solo aparecen las órdenes pendientes creadas explícitamente.
        """
        all_orders = await self.get_all_orders(symbol)
        new_orders = [o for o in all_orders if o["status"] in ["Placed", "Untriggered"]]
        return new_orders
    
    async def get_order(self, symbol: str, order_id: str) -> dict:
        """Obtiene información de una orden específica"""
        all_orders = await self.get_all_orders(symbol)
        matching_orders = [o for o in all_orders if o["orderId"] == order_id]
        return matching_orders[-1] if matching_orders else None
    
    async def get_leverage(self, symbol: str) -> int:
        """Obtiene el apalancamiento de la cuenta"""
        def _get():
            info = mt5.account_info()
            if info is None:
                return 1
            return int(info.leverage)
        
        return await asyncio.to_thread(_get)
    
    async def set_margin_mode(self, margin_mode: str = "isolated"):
        """Nota: MT5 no soporta cambio de modo de margen programáticamente"""
        warnings.warn(
            "MT5 no permite cambiar el modo de margen programáticamente. "
            "El modo de margen está determinado por la configuración del servidor/broker.",
            UserWarning
        )
        return {"status": "not_supported", "message": "MT5 no soporta cambio de margen programático"}
    
    async def get_closed_pnl(self, symbol: str, from_date=None, to_date=None) -> list:
        """Obtiene el historial de PnL cerrado para un símbolo, ordenado por tiempo (más reciente al final)"""
        def _get_deals(from_date, to_date):
            if from_date is None:
                from_date = datetime.now() - pd.Timedelta(days=90)
            if to_date is None:
                to_date = datetime.now()    
            deals = mt5.history_deals_get(from_date, to_date, group=symbol)

            if deals is None:
                return []
            
            deals_list = []
            for deal in deals:
                if deal.entry == mt5.DEAL_ENTRY_OUT:
                    side = "Buy" if deal.type == mt5.DEAL_TYPE_BUY else "Sell"
                    
                    deals_list.append({
                        "symbol": deal.symbol,
                        "pnl": float(deal.profit),
                        "volume": float(deal.volume),
                        "price": float(deal.price),
                        "time": int(deal.time),
                        "commission": float(deal.commission),
                        "type": side,
                        "magic": int(deal.magic),
                    })
            
            # Ordenar por tiempo (más antiguo primero, más reciente al final)
            deals_list.sort(key=lambda x: x['time'])
            
            return deals_list
        
        return await asyncio.to_thread(_get_deals, from_date, to_date)

    """ TRADING """
    
    async def _place_order(self,
                          symbol: str,
                          side: str,
                          volume: float,
                          price: float = None,
                          sl_price: float = None,
                          tp_price: float = None,
                          pct_sl: float = None,
                          pct_tp: float = None,
                          order_type: str = None,
                          comment: str = "",
                          ) -> dict:
        """Coloca una orden en MT5"""
        def _send_order():
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                raise ValueError(f"Símbolo {symbol} no encontrado")
            
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    raise ValueError(f"No se pudo seleccionar {symbol}")
            
            volume_normalized = self._normalize_volume(symbol, volume)
            
            if order_type:
                if order_type.lower() == "limit":
                    action = mt5.ORDER_TYPE_BUY_LIMIT if side == "Buy" else mt5.ORDER_TYPE_SELL_LIMIT
                elif order_type.lower() == "stop":
                    action = mt5.ORDER_TYPE_BUY_STOP if side == "Buy" else mt5.ORDER_TYPE_SELL_STOP
                else:
                    action = mt5.ORDER_TYPE_BUY if side == "Buy" else mt5.ORDER_TYPE_SELL
            else:
                if price:
                    action = mt5.ORDER_TYPE_BUY_LIMIT if side == "Buy" else mt5.ORDER_TYPE_SELL_LIMIT
                else:
                    action = mt5.ORDER_TYPE_BUY if side == "Buy" else mt5.ORDER_TYPE_SELL
            
            if not price or action in [mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL]:
                tick = mt5.symbol_info_tick(symbol)
                execution_price = tick.ask if side == "Buy" else tick.bid
            else:
                execution_price = price
            
            execution_price = self._normalize_price(symbol, execution_price)
            
            sl_normalized = None
            tp_normalized = None
            
            if sl_price is not None:
                sl_normalized = self._normalize_price(symbol, float(sl_price))
            if tp_price is not None:
                tp_normalized = self._normalize_price(symbol, float(tp_price))
            
            filling_mode = self._get_filling_mode(symbol)
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL if action in [mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL] else mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume_normalized,
                "type": action,
                "price": execution_price,
                "deviation": 20 if self._market_type == "futures" else 10,
                "magic": self._magic_number,
                "comment": comment or f"{self._market_type}_order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
            
            if sl_normalized:
                request["sl"] = sl_normalized
            if tp_normalized:
                request["tp"] = tp_normalized
            
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                raise ValueError(f"Error al enviar orden: {error}")
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise ValueError(f"Error en orden: {result.comment} (retcode: {result.retcode})")
            
            order_info = {
                "orderId": str(result.order),
            }
            
            # Aplicar SL/TP porcentuales si se especificaron
            if pct_sl or pct_tp:
                actual_price = result.price
                
                tp_calc = None
                sl_calc = None
                
                if pct_tp:
                    tp_calc = actual_price * (1 + pct_tp) if side == "Buy" else actual_price * (1 - pct_tp)
                    tp_calc = self._normalize_price(symbol, tp_calc)
                
                if pct_sl:
                    sl_calc = actual_price * (1 - pct_sl) if side == "Buy" else actual_price * (1 + pct_sl)
                    sl_calc = self._normalize_price(symbol, sl_calc)
                
                if result.deal > 0:
                    modify_request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": symbol,
                        "position": result.deal,
                        "sl": sl_calc if sl_calc else 0,
                        "tp": tp_calc if tp_calc else 0,
                    }
                    mt5.order_send(modify_request)
            
            return order_info
        
        return await asyncio.to_thread(_send_order)
    
    async def buy(self,
                 symbol: str,
                 volume: float,
                 price: float = None,
                 sl_price: float = None,
                 tp_price: float = None,
                 pct_sl: float = None,
                 pct_tp: float = None,
                 comment: str = "",
                 ) -> dict:
        """Coloca una orden de compra"""
        return await self._place_order(
            symbol=symbol,
            side="Buy",
            volume=volume,
            price=price,
            sl_price=sl_price,
            tp_price=tp_price,
            pct_sl=pct_sl,
            pct_tp=pct_tp,
            comment=comment or f"buy_{self._market_type}",
        )
    
    async def sell(self,
                  symbol: str,
                  volume: float,
                  price: float = None,
                  sl_price: float = None,
                  tp_price: float = None,
                  pct_sl: float = None,
                  pct_tp: float = None,
                  comment: str = "",
                  ) -> dict:
        """Coloca una orden de venta"""
        return await self._place_order(
            symbol=symbol,
            side="Sell",
            volume=volume,
            price=price,
            sl_price=sl_price,
            tp_price=tp_price,
            pct_sl=pct_sl,
            pct_tp=pct_tp,
            comment=comment or f"sell_{self._market_type}",
        )
    
    async def close_position(self, symbol: str) -> dict:
        """Cierra todas las posiciones abiertas para un símbolo"""
        def _close():
            positions = mt5.positions_get(symbol=symbol)
            
            if not positions:
                return None
            
            results = []
            for position in positions:
                close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                
                tick = mt5.symbol_info_tick(symbol)
                price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
                price = self._normalize_price(symbol, price)
                
                filling_mode = self._get_filling_mode(symbol)
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": position.volume,
                    "type": close_type,
                    "position": position.ticket,
                    "price": price,
                    "deviation": 20 if self._market_type == "futures" else 10,
                    "magic": self._magic_number,
                    "comment": f"close_{self._market_type}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling_mode,
                }
                
                result = mt5.order_send(request)
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    results.append({
                        "orderId": str(result.order),
                    })
            
            return results[0] if len(results) == 1 else results
        
        return await asyncio.to_thread(_close)
    
    async def set_trading_stop(self,
                            symbol: str,
                            tp_price: float = None,
                            sl_price: float = None,
                            tp_volume: float = None,
                            sl_volume: float = None,
                            ) -> dict:
        """
        Crea órdenes Limit/Stop reales que funcionen como TP/SL.
        Automáticamente determina si es 'FullClose' o 'PartialClose' según el volumen.
        
        Args:
            symbol: Símbolo del instrumento
            tp_price: Precio de take profit (opcional)
            sl_price: Precio de stop loss (opcional)
            tp_volume: Volumen para TP (opcional, default: volumen total de la posición)
            sl_volume: Volumen para SL (opcional, default: volumen total de la posición)
        """
        def _create_orders():
            positions = mt5.positions_get(symbol=symbol)
            
            if not positions or len(positions) == 0:
                raise ValueError(f"No hay posiciones abiertas para {symbol}")
            
            results = []
            
            for position in positions:
                position_volume = position.volume
                
                # Para TP
                if tp_price is not None:
                    tp_normalized = self._normalize_price(symbol, float(tp_price))
                    
                    # Determinar volumen para TP
                    if tp_volume:
                        volume_normalized = self._normalize_volume(symbol, float(tp_volume))
                    else:
                        volume_normalized = position_volume
                    
                    # Determinar comentario basado en volumen
                    if volume_normalized >= position_volume:
                        comment = "FullTakeProfit"
                    else:
                        comment = "PartialTakeProfit"
                    
                    # Determinar tipo de orden opuesta para TP (Limit)
                    if position.type == mt5.ORDER_TYPE_BUY:
                        order_type = mt5.ORDER_TYPE_SELL_LIMIT
                        side = "Sell"
                    else:
                        order_type = mt5.ORDER_TYPE_BUY_LIMIT
                        side = "Buy"
                    
                    request = {
                        "action": mt5.TRADE_ACTION_PENDING,
                        "symbol": symbol,
                        "volume": volume_normalized,
                        "type": order_type,
                        "price": tp_normalized,
                        "magic": self._magic_number,
                        "comment": comment,
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": self._get_filling_mode(symbol),
                        "position": position.ticket,
                    }
                    
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        results.append({
                            "position": position.ticket,
                            "side": side,
                            "type": "Limit",
                            "orderId": str(result.order),
                            "comment": comment,
                            "success": True,
                            "retCode": result.retcode,
                        })
                    else:
                        error_msg = result.comment if result else "Unknown error"
                        results.append({
                            "position": position.ticket,
                            "side": side,
                            "type": "Limit",
                            "success": False,
                            "error": error_msg,
                            "retCode": result.retcode if result else -1,
                        })
                
                # Para SL
                if sl_price is not None:
                    sl_normalized = self._normalize_price(symbol, float(sl_price))
                    
                    # Determinar volumen para SL
                    if sl_volume:
                        volume_normalized = self._normalize_volume(symbol, float(sl_volume))
                    else:
                        volume_normalized = position_volume
                    
                    # Determinar comentario basado en volumen
                    if volume_normalized >= position_volume:
                        comment = "FullStopLoss"
                    else:
                        comment = "PartialStopLoss"
                    
                    # Determinar tipo de orden opuesta para SL (Stop)
                    if position.type == mt5.ORDER_TYPE_BUY:
                        order_type = mt5.ORDER_TYPE_SELL_STOP
                        side = "Sell"
                    else:
                        order_type = mt5.ORDER_TYPE_BUY_STOP
                        side = "Buy"
                    
                    request = {
                        "action": mt5.TRADE_ACTION_PENDING,
                        "symbol": symbol,
                        "volume": volume_normalized,
                        "type": order_type,
                        "price": sl_normalized,
                        "magic": self._magic_number,
                        "comment": comment,
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": self._get_filling_mode(symbol),
                        "position": position.ticket,
                    }
                    
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        results.append({
                            "position": position.ticket,
                            "side": side,
                            "type": "Stop",
                            "orderId": str(result.order),
                            "comment": comment,
                            "success": True,
                            "retCode": result.retcode,
                        })
                    else:
                        error_msg = result.comment if result else "Unknown error"
                        results.append({
                            "position": position.ticket,
                            "side": side,
                            "type": "Stop",
                            "success": False,
                            "error": error_msg,
                            "retCode": result.retcode if result else -1,
                        })
            
            if len(results) == 1:
                return results[0]
            return results
        
        return await asyncio.to_thread(_create_orders)

    async def cancel_all_orders(self, symbol: str) -> dict:
        """Cancela todas las órdenes pendientes para un símbolo"""
        def _cancel():
            orders = mt5.orders_get(symbol=symbol)
            
            if not orders:
                return {"totalCancelled": 0}
            
            cancelled_count = 0
            for order in orders:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                }
                
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    cancelled_count += 1
            
            return {"totalCancelled": cancelled_count}
        
        return await asyncio.to_thread(_cancel)
    
    async def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancela una orden específica"""
        def _cancel():
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": int(order_id),
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                return {
                    "retCode": error[0] if isinstance(error, tuple) else -1,
                    "comment": str(error),
                }
            
            return {
                "retCode": result.retcode,
                "comment": result.comment if result.comment else "",
            }
        
        return await asyncio.to_thread(_cancel)
    
    """ UTILIDADES ADICIONALES MT5 """
    
    async def get_symbols(self, group: str = "*") -> list:
        """Obtiene lista de símbolos disponibles"""
        def _get():
            symbols = mt5.symbols_get(group=group)
            return [s.name for s in symbols] if symbols else []
        
        return await asyncio.to_thread(_get)
    
    async def subscribe_symbol(self, symbol: str) -> bool:
        """Suscribe un símbolo para recibir cotizaciones"""
        def _subscribe():
            return mt5.symbol_select(symbol, True)
        
        return await asyncio.to_thread(_subscribe)
    
    async def get_market_book(self, symbol: str) -> dict:
        """Obtiene el libro de órdenes (depth of market)"""
        def _get_book():
            book = mt5.market_book_get(symbol)
            
            if book is None:
                return {"bids": [], "asks": []}
            
            bids = [{"price": item.price, "volume": item.volume} 
                   for item in book if item.type == mt5.BOOK_TYPE_BUY]
            asks = [{"price": item.price, "volume": item.volume} 
                   for item in book if item.type == mt5.BOOK_TYPE_SELL]
            
            return {"bids": bids, "asks": asks}
        
        return await asyncio.to_thread(_get_book)
    
    async def calculate_margin(self, symbol: str, volume: float, action: str = "Buy") -> float:
        """Calcula el margen requerido para una operación"""
        def _calc():
            order_type = mt5.ORDER_TYPE_BUY if action == "Buy" else mt5.ORDER_TYPE_SELL
            tick = mt5.symbol_info_tick(symbol)
            price = tick.ask if action == "Buy" else tick.bid
            
            volume_normalized = self._normalize_volume(symbol, volume)
            
            margin = mt5.order_calc_margin(order_type, symbol, volume_normalized, price)
            return margin if margin is not None else 0.0
        
        return await asyncio.to_thread(_calc)
    
    async def calculate_profit(self, symbol: str, volume: float, 
                              price_open: float, price_close: float,
                              action: str = "Buy") -> float:
        """Calcula la ganancia/pérdida potencial de una operación"""
        def _calc():
            order_type = mt5.ORDER_TYPE_BUY if action == "Buy" else mt5.ORDER_TYPE_SELL
            
            volume_normalized = self._normalize_volume(symbol, volume)
            price_open_norm = self._normalize_price(symbol, price_open)
            price_close_norm = self._normalize_price(symbol, price_close)
            
            profit = mt5.order_calc_profit(order_type, symbol, volume_normalized, 
                                          price_open_norm, price_close_norm)
            return profit if profit is not None else 0.0
        
        return await asyncio.to_thread(_calc)
    
    async def get_terminal_info(self) -> dict:
        """Obtiene información del terminal MT5"""
        def _get():
            info = mt5.terminal_info()
            if info is None:
                return {}
            
            return {
                "build": info.build,
                "connected": info.connected,
                "tradeAllowed": info.trade_allowed,
                "emailEnabled": info.email_enabled,
                "company": info.company,
                "name": info.name,
                "language": info.language,
                "dataPath": info.data_path,
            }
        
        return await asyncio.to_thread(_get)
    
    async def check_connection(self) -> bool:
        """Verifica si la conexión con MT5 está activa"""
        def _check():
            info = mt5.terminal_info()
            return info.connected if info else False
        
        return await asyncio.to_thread(_check)
    
    async def get_contract_specs(self, symbol: str) -> dict:
        """Obtiene especificaciones del contrato"""
        def _get_specs():
            info = mt5.symbol_info(symbol)
            if info is None:
                raise ValueError(f"No se encontró información para {symbol}")
            
            specs = {
                "symbol": info.name,
                "contractSize": float(info.trade_contract_size),
                "tickSize": float(info.trade_tick_size),
                "tickValue": float(info.trade_tick_value),
                "point": float(info.point),
                "digits": info.digits,
                "volumeMin": float(info.volume_min),
                "volumeMax": float(info.volume_max),
                "volumeStep": float(info.volume_step),
                "currencyBase": info.currency_base,
                "currencyProfit": info.currency_profit,
                "currencyMargin": info.currency_margin,
            }
            
            if self._market_type == "futures" and hasattr(info, 'expiration_time'):
                specs["expirationTime"] = info.expiration_time
                specs["expirationMode"] = info.expiration_mode
            
            return specs
        
        return await asyncio.to_thread(_get_specs)


# Ejemplos de uso
"""
# ============== INICIALIZACIÓN ==============
sesh = MT5Sesh(
    account=1545688,
    password='ReM_Kx5t', 
    server='AMPGlobalUSA-Demo',
    market_type='futures'
)

# ============== DATA ==============
# Obtener tiempo del servidor
time = await sesh.get_time()
# Timestamp('2025-10-29 05:25:58')

# Obtener datos históricos
data = await sesh.get_data("MESZ25", "5min", limit=100)

# ============== INFO ==============
# Información del instrumento
info = await sesh.get_instruments_info("MES")
# {
#   'symbol': 'MESZ25',
#   'description': 'Micro E-mini S&P 500: December 2025',
#   'digits': 2,
#   'tradeContractSize': 1.0,
#   'tradeTickSize': 0.25,
#   'tradeTickValue': 1.25,
#   'volumeMin': 1.0,
#   'volumeMax': 100000.0,
#   'volumeStep': 1.0,
#   'currencyBase': 'USD'
# }

# Balance
balance = await sesh.get_balance()
# {
#   'balance': 99707.8,
#   'equity': 99692.8,
#   'margin': 160.0,
#   'marginFree': 99532.8,
#   'pnl': -15.0,
#   'marginLevel': 62308.0
# }

# Posición actual
position = await sesh.get_position("MESZ25")
# {
#   'volume': 4.0,
#   'side': 'Buy',
#   'price': 6906.25,
#   'pnl': -35.0
# }

# Obtener orden específica
order = await sesh.get_order("MESZ25", '277819434')
# {
#   'orderId': '277819434',
#   'symbol': 'MESZ25',
#   'side': 'Sell',
#   'type': 'Market',      # Market, Limit, Stop, StopLimit
#   'volume': 2.0,
#   'price': 6900.0,
#   'sl': 0.0,
#   'tp': 0.0,
#   'time': 1761636838,
#   'status': 'Filled',    # Filled, Placed, Untriggered, Cancelled, Rejected
#   'magic': 234000,
#   'comment': 'tp hit'    # El comentario puede indicar si fue TP/SL
# }

# Las órdenes Stop/Limit con comentarios 'partial tp' o 'partial sl' 
# son órdenes normales que funcionan como salidas parciales

# Órdenes nuevas/pendientes
new_orders = await sesh.get_new_orders("MESZ25")
# Lista de órdenes con status 'Placed' o 'Untriggered'

# Historial de PnL
pnl = await sesh.get_closed_pnl("MESZ25")
# [
#   {
#     'symbol': 'MESZ25',
#     'profit': -5.0,
#     'volume': 2.0,
#     'price': 6910.0,
#     'time': 1761610909,
#     'commission': -1.2,
#     'type': 'Sell',
#     'magic': 0
#   }
# ]

# ============== TRADING ==============
# Comprar
order = await sesh.buy(symbol="MESZ25", volume=2)
# {'orderId': '277819434'}

# Comprar con SL y TP porcentuales
order = await sesh.buy(symbol="MESZ25", volume=2, pct_sl=0.02, pct_tp=0.03)
# {'orderId': '277819435'}

# Establecer TP/SL completo
result = await sesh.set_trading_stop(symbol="MESZ25", tp_price=6909.0)
# {
#   'position': 277873554,
#   'success': True,
#   'retCode': 10009
# }

# Establecer TP/SL parcial
# Crea órdenes Limit/Stop normales que actúan como salidas parciales
result = await sesh.set_trading_stop(
    symbol="MESZ25", 
    tp_price=6909.0, 
    tp_volume=1.0  # Solo cerrar 1 contrato en TP (crea orden Limit)
)
# {
#   'position': 277873554,
#   'success': True,
#   'retCode': 10009
# }
# Esto crea una orden Limit con comment='partial tp'

# Cerrar posición
result = await sesh.close_position("MESZ25")
# {'orderId': '5de68f6f-55f8-4346-b6d1-9d59db7d7a16'}

# Cancelar orden
result = await sesh.cancel_order("MESZ25", '277819434')
# {
#   'retCode': 10033,
#   'comment': ''
# }

# Cancelar todas las órdenes
result = await sesh.cancel_all_orders("MESZ25")
# {'totalCancelled': 0}
"""