from .nb import count_available_resample_bars_nb, count_closed_resample_bars_nb
from ..lib import timeframe2minutes, npdt64_to_datetime
from ..models import DataOHLC
from ..stats import compute_stats
from ..core import Strategy
import pandas as pd
import numpy as np
import asyncio
import time
import uuid
from collections import deque
from typing import Dict, Optional, Union, List
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from ..plotting import plot as _plot
from bokeh.layouts import gridplot
from bokeh.io import show, output_notebook

class Order:
    __slots__ = ['id', 'symbol', 'size', 'price', 'order_type', 'side', 'take_profit', 'stop_loss', 'status', 'created_at', 'executed_at']
    
    def __init__(self, symbol: str, size: float, price: Optional[float] = None, 
             order_type: str = 'market', side: str = 'buy',
             take_profit: Optional[float] = None, stop_loss: Optional[float] = None,
             created_at: Optional[datetime] = None):
        self.id = str(uuid.uuid4())[:8]
        self.symbol = symbol.upper()
        self.size = abs(size)
        self.price = price
        self.order_type = order_type.lower()
        self.side = side.lower()
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.status = 'pending'
        self.created_at = created_at
        self.executed_at = None
        
        
    def __repr__(self):
        attrs = [f"Id={self.id}", f"Symbol={self.symbol}", f"Side={self.side.title()}", 
                f"Type={self.order_type.title()}", f"Size={self.size}", f"Status={self.status.title()}"]
        
        if self.price is not None: attrs.append(f"Price={self.price}")
        if self.take_profit is not None: attrs.append(f"TP={self.take_profit}")
        if self.stop_loss is not None: attrs.append(f"SL={self.stop_loss}")
        
        return f"Order({', '.join(attrs)})"


    def __str__(self):
        return self.__repr__()

class Trade:
    __slots__ = ['id', 'order_id', 'symbol', 'size', 'entry_price', 'exit_price', 'side', 'commission', 
        'entry_time', 'exit_time', 'take_profit', 'stop_loss', 'max_price', 'min_price', 'mae', 'mfe']
    
    def __init__(self, order_id: str, symbol: str, size: float, entry_price: float, 
                 side: str, commission: float, exit_price: Optional[float] = None,
                 entry_time: Optional[datetime] = None, exit_time: Optional[datetime] = None):
        self.id = str(uuid.uuid4())[:8]
        self.order_id = order_id
        self.symbol = symbol.upper()
        self.size = size
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.side = side.lower()
        self.commission = commission
        self.entry_time = entry_time 
        self.exit_time = exit_time
        self.take_profit = None
        self.stop_loss = None

    @property
    def pnl(self) -> Optional[float]:
        """Calcula el PnL del trade si está cerrado"""
        if self.exit_price is None:
            return None
        
        side_normalized = self.side.lower()
        if side_normalized in ['buy', 'long']:
            return (self.exit_price - self.entry_price) * self.size - self.commission
        else:  # sell/short
            return (self.entry_price - self.exit_price) * self.size - self.commission
    
    @property
    def return_pct(self) -> Optional[float]:
        """Calcula el retorno porcentual del trade"""
        if self.pnl is None:
            return None
        return (self.pnl / (self.entry_price * self.size)) * 100
    
    @property
    def type(self) -> str:
        """Tipo de trade basado en si tiene exit_price"""
        return "Market" if self.exit_price is not None else "Open"
        
    def __repr__(self):
        attrs = [f"Id={self.id}", f"Symbol={self.symbol}", f"Side={self.side.title()}", 
                f"Type={self.type}", f"Size={self.size}", f"EntryPrice={self.entry_price}"]
        
        # optional_attrs = [
        #     (self.exit_price, f"ExitPrice={self.exit_price}"),
        #     (self.pnl, f"Pnl={self.pnl:.2f}"),
        #     (self.return_pct, f"ReturnPct={self.return_pct:.2f}%"),
        #     (self.entry_time, f"EntryTime={self.entry_time.strftime('%Y-%m-%d %H:%M:%S')}"),
        #     (self.exit_time, f"ExitTime={self.exit_time.strftime('%Y-%m-%d %H:%M:%S')}"),
        #     (self.take_profit, f"TP={self.take_profit}"),
        #     (self.stop_loss, f"SL={self.stop_loss}"),
        #     (self.mae, f"mae={self.mae}"),
        #     (self.mfe, f"mfe={self.mfe}"),
        # ]
        
        # attrs.extend([attr for value, attr in optional_attrs if value is not None])
        return f"Trade({', '.join(attrs)})"

    def __str__(self):
        return self.__repr__()

class Position:
    __slots__ = ['id', 'symbol', 'size', 'entry_price', 'side', 'unrealized_pnl', 
                 'take_profit_levels', 'stop_loss_levels', 'status', 'opened_at', 'closed_at',
                 'max_price', 'min_price', 'mfe', 'mae']  
    
    def __init__(self, symbol: str, size: float, entry_price: float, side: str, opened_at=None):
        self.id = str(uuid.uuid4())[:8]
        self.symbol = symbol.upper()
        self.size = abs(size)
        self.entry_price = entry_price
        self.side = side.lower()
        self.unrealized_pnl = 0.0
        self.take_profit_levels = []  # Lista de (precio, tamaño)
        self.stop_loss_levels = []    # Lista de (precio, tamaño)
        self.status = 'open'
        self.opened_at = opened_at
        self.closed_at = None
        # -- EDIT -- > 
        # Agregar estas líneas para rastrear el MAE y el MFE por barra (broker.next())
        self.max_price = entry_price
        self.min_price = entry_price
        self.mfe = 0.0 # Porcentajes no potenciados a 100
        self.mae = 0.0
        # < -- EDIT -- 

    #  -- EDIT -- >
    # Agregar estas líneas para rastrear precios máximos y mínimos 
    def update_max_min_prices(self, high: float, low: float): 
        """Actualizar precios máximos y mínimos durante la vida útil de la operación"""
        # Actualizar los precios máximos y mínimos
        self.max_price = max(self.max_price, high)
        self.min_price = min(self.min_price, low)
        
    # Actualizar el MAE y el MFE según el tipo de métrica ('ROI'|'ROE')
    # def calculate_mae_mfe(self, cash: float, metric_type: str = "ROI"):
        # """Calcular el MAE y el MFE para una operación dada"""
        # is_long = self.side == "Long"
        # up_distance, down_distance = (self._max_price - self.entry_price), (self.entry_price - self._min_price)
        # entry_price = self.entry_price
        # size = self.size
        # if metric_type == 'ROI':
        #     self.mfe = (up_distance / entry_price) if is_long else (down_distance / entry_price)
        #     self.mae = (down_distance / entry_price) if is_long else (up_distance / entry_price)
        # elif metric_type == 'ROE':
        #     self.mfe = (up_distance * size / cash) if is_long else (down_distance * size / cash)
        #     self.mae = (down_distance * size / cash) if is_long else (up_distance * size / cash)
    # < -- EDIT -- 
        
    def update_unrealized_pnl(self, current_price: float):
        """Actualiza el PnL no realizado basado en el precio actual"""
        assert isinstance(current_price, float) or isinstance(current_price, int), "El precio actual debe ser mayor que cero"
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:  # short
            self.unrealized_pnl = (self.entry_price - current_price) * self.size

    # def update_unrealized_pnl(self, current_price: float):
    #     """Actualiza el PnL no realizado basado en el precio actual"""
    #     # Asegurar que current_price sea un escalar
    #     import numpy as np
    #     if isinstance(current_price, np.ndarray):
    #         current_price = float(current_price.item() if current_price.size == 1 else current_price[-1])
    #     elif hasattr(current_price, '__len__') and len(current_price) > 0:
    #         current_price = float(current_price[-1] if hasattr(current_price, '__getitem__') else current_price)
        
    #     assert isinstance(current_price, (float, int)) and current_price > 0, "El precio actual debe ser un número positivo"
        
    #     if self.side == 'long':
    #         self.unrealized_pnl = (current_price - self.entry_price) * self.size
    #     else:  # short
    #         self.unrealized_pnl = (self.entry_price - current_price) * self.size  

    def close(self, closed_at):
        """Cierra la posición"""
        self.status = 'closed'
        self.closed_at = closed_at
        
    def __repr__(self):
        attrs = []
        attrs.append(f"Id={self.id}")
        attrs.append(f"Symbol={self.symbol}")
        attrs.append(f"Side={self.side.title()}")
        attrs.append(f"Size={self.size}")
        attrs.append(f"EntryPrice={self.entry_price}")
        attrs.append(f"UnrealizedPnl={self.unrealized_pnl:.2f}")
        attrs.append(f"Status={self.status.title()}")
        return f"Position({', '.join(attrs)})"
    
    def __str__(self):
        return self.__repr__()

class _Broker:
    """Broker principal para el sistema de backtesting"""
    
    def __init__(self, cash: float, maker_fee: float = 0.000, 
                 taker_fee: float = 0.000, margin: float = 0.1, 
                 margin_mode: str = 'isolated', mae_mfe_metric_type='ROI'
    ):
        self.initial_cash = cash 
        self.cash = cash
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.margin = margin
        self.margin_mode = margin_mode.lower()
        
        # Contenedores de datos
        self.orders = self.orders = deque()
        self.trades = []
        self.closed_trades = []
        self.all_orders_history = []  # Historial completo de órdenes
        self.all_trades_history = []  # Historial completo de trades

        # Posiciones por símbolo
        self.positions = {}

        # Precios actuales por símbolo
        self.current_close = {}
        self.current_ohlc = {}

        self.mae_mfe_metric_type = mae_mfe_metric_type
        self.current_timestamp = None  # Agregar esta línea

    @property
    def total_equity(self) -> float:
        # Calcular siempre sin cache
        equity = self.cash
        for position in self.positions.values():
            if position.status == 'open':
                equity += position.unrealized_pnl
        return equity

    def _get_available_margin(self) -> float:
        """Calcula el margen disponible según el modo"""
        if self.margin_mode == 'isolated':
            return self.cash
        else:  # cross
            return self.total_equity
    
    def _calculate_required_margin(self, size: float, price: float) -> float:
        """Calcula el margen requerido para una posición"""
        notional_value = size * price
        return notional_value * self.margin
    
    def _can_open_position(self, size: float, price: float) -> bool:
        """Verifica si hay suficiente margen para abrir una posición"""
        required_margin = self._calculate_required_margin(size, price)
        available_margin = self._get_available_margin()
        return available_margin >= required_margin
    
    def place_order(self, symbol: str, type: str, size: float, limit: Optional[float] = None,
                   stop: Optional[float] = None, take_profit: Optional[float] = None,
                   stop_loss: Optional[float] = None) -> str:
        """
        Coloca una orden de trading
        
        Args:
            symbol: símbolo del activo (ej: "BTCUSDT")
            type: 'buy' o 'sell'
            size: tamaño de la orden
            limit: precio límite (para órdenes limit)
            stop: precio de stop (para órdenes stop)
            take_profit: precio de take profit
            stop_loss: precio de stop loss
            
        Returns:
            ID de la orden creada
        """
        symbol = symbol.upper()
        type = type.lower()
        timestamp = self.current_timestamp

        # Verificar que el símbolo tenga precio disponible
        if symbol not in self.current_close:
            raise ValueError(f"No hay precio disponible para {symbol}. Ejecuta update_market primero.")
        
        # Determinar tipo de orden
        if limit is not None:
            order_type = 'limit'
            price = limit
        elif stop is not None:
            order_type = 'stop'
            price = stop
        else:
            order_type = 'market'
            price = self.current_close[symbol]
        # else:
        #     order_type = 'market'
        #     # Usar precio de apertura para mayor realismo
        #     if symbol in self.current_ohlc:
        #         price = self.current_ohlc[symbol]['Open']
        #     else:
        #         price = self.current_prices[symbol]  # Fallback si no hay OHLC
            
        # Crear la orden
        order = Order(
            symbol=symbol,
            size=size,
            price=price,
            order_type=order_type,
            side=type,
            take_profit=take_profit,
            stop_loss=stop_loss,
            created_at=timestamp
        )
        # print("place order", order)

        # Agregar al historial completo
        self.all_orders_history.append(order)
        # Ejecutar inmediatamente si es market order
        if order_type == 'market':
            self._execute_order(order, self.current_close[symbol])
        else:
            self.orders.append(order)
            
        return order.id

    # TODO: crear un atributo donde se guarden las ordenes canceladas
    def _execute_order(self, order: Order, execution_price: Union[float, int]) -> bool:
        """Ejecuta una orden al precio especificado"""
        assert isinstance(execution_price, (float, int)), "execution_price debe ser float o int"
        assert isinstance(order, Order), "order debe ser una instancia de Order"
        symbol = order.symbol

        timestamp = self.current_timestamp
        
        # Verificar si hay posición contraria que cerrar
        if symbol in self.positions and self.positions[symbol].status == 'open':
            position = self.positions[symbol]
            if (order.side == 'buy' and position.side == 'short') or \
               (order.side == 'sell' and position.side == 'long'):
                self._close_position_with_price(symbol, execution_price)
        
        # Verificar margen disponible
        if not self._can_open_position(order.size, execution_price):
            print(f"\033[91mOrden {order.id} cancelada: margen insuficiente\033[0m")
            order.status = 'cancelled'
            return False

        # Calcular comisión
        is_maker = order.order_type == 'limit'
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee = order.size * execution_price * fee_rate
        
        # Crear el trade
        trade = Trade(
            order_id=order.id,
            symbol=symbol,
            size=order.size,
            entry_price=execution_price,
            side=order.side,
            commission=fee,
            entry_time=timestamp
        )

        # Actualizar cash
        self.cash -= fee
        
        # Crear o actualizar posición
        if symbol not in self.positions or self.positions[symbol].status == 'closed':
            # Nueva posición
            side = 'long' if order.side == 'buy' else 'short'
            self.positions[symbol] = Position(
                symbol=symbol,
                size=order.size,
                entry_price=execution_price,
                side=side,
                opened_at=timestamp
            )
            
            # Configurar órdenes contingentes si las hay
            if order.take_profit:
                self.positions[symbol].take_profit_levels.append((order.take_profit, order.size))
            if order.stop_loss:
                self.positions[symbol].stop_loss_levels.append((order.stop_loss, order.size))
                
        else:
            # Actualizar posición existente (promedio de precios)
            position = self.positions[symbol]
            if position.side == ('long' if order.side == 'buy' else 'short'):
                total_cost = (position.size * position.entry_price) + \
                           (order.size * execution_price)
                total_size = position.size + order.size
                position.entry_price = total_cost / total_size
                position.size = total_size
        
        # Actualizar estado de la orden
        order.status = 'filled'
        order.executed_at = timestamp
        
        # Agregar trade a las listas
        self.trades.append(trade)
        self.all_trades_history.append(trade)
        
        # Remover orden de pendientes si estaba ahí
        if order in self.orders:
            self.orders.remove(order)
            
        return True
     
    # def update_market(self, symbol: str, ohlc: dict):
        # """Versión ULTRA-OPTIMIZADA"""
        # # Evitar conversiones innecesarias
        # self.current_close[symbol] = ohlc['Close']
        # self.current_ohlc[symbol] = ohlc

        # # Early return si no hay posición
        # if symbol not in self.positions:
        #     return
            
        # position = self.positions[symbol]
        # if position.status != 'open':
        #     return

        # # Solo operaciones esenciales
        # position.update_unrealized_pnl(ohlc['Close'])
        # # Comentar MAE/MFE si no es crítico para tu estrategia
        # # position.update_max_min_prices(ohlc["High"], ohlc["Low"])
        # # position.calculate_mae_mfe(cash=self.cash, metric_type=self.mae_mfe_metric_type)

    def update_market(self, symbol: str, ohlc_content: dict):
        """Versión ULTRA-OPTIMIZADA"""
        # Evitar conversiones innecesarias
        
        self.current_close[symbol] = ohlc_content['Close'][-1]
        self.current_ohlc[symbol] = ohlc_content
        
        # Procesar órdenes pendientes para este símbolo
        orders_to_execute = []
        for order in self.orders:
            if order.status == "cancelled": # ignorar ordenes canceladas
                continue
            if order.symbol == symbol:
                execution_price = self._should_execute_order(order, ohlc_content)
                if execution_price is not None:
                    orders_to_execute.append((order, execution_price))
        
        # Ejecutar órdenes encontradas
        for order, execution_price in orders_to_execute:
            self._execute_order(order, execution_price)

        # Early return si no hay posición
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]

        if position.status != 'open':
            return

        # -- EDIT -->
        position.update_max_min_prices(ohlc_content["High"][-1], ohlc_content["Low"][-1])
        # <-- EDIT --

        # Solo operaciones esenciales
        position.update_unrealized_pnl(ohlc_content['Close'][-1])
        
        # Verificar stops para la posición
        self._check_position_stops(symbol, ohlc_content)

    def _should_execute_order(self, order: Order, ohlc: dict) -> Optional[float]:
        """Método auxiliar para determinar si una orden debe ejecutarse"""
        if order.order_type == 'limit':
            if order.side == 'buy' and ohlc['Low'][-1] <= order.price:
                return order.price
            elif order.side == 'sell' and ohlc['High'][-1] >= order.price:
                return order.price
        elif order.order_type == 'stop':
            if order.side == 'buy' and ohlc['High'][-1] >= order.price:
                return order.price
            elif order.side == 'sell' and ohlc['Low'][-1] <= order.price:
                return order.price
        return None
    
    def _check_position_stops(self, symbol: str, ohlc: pd.Series):
        """Verifica y ejecuta stop loss y take profit de la posición"""
        if symbol not in self.positions or self.positions[symbol].status != 'open':
            return
            
        position = self.positions[symbol]
        
        # Verificar take profit levels
        for i, (tp_price, tp_size) in enumerate(position.take_profit_levels[:]):
            should_trigger = False
            if position.side == 'long' and ohlc['High'] >= tp_price:
                should_trigger = True
            elif position.side == 'short' and ohlc['Low'] <= tp_price:
                should_trigger = True
                
            if should_trigger:
                self._execute_partial_close(symbol, tp_price, tp_size)
                position.take_profit_levels.pop(i)
                if position.size <= 0:
                    break
        
        # Verificar stop loss levels
        remaining_levels = []
        
        for sl_price, sl_size in position.stop_loss_levels:
            should_trigger = False
            
            if position.side == 'long' and ohlc['Low'] <= sl_price:
                should_trigger = True
            elif position.side == 'short' and ohlc['High'] >= sl_price:
                should_trigger = True
                
            if should_trigger:
                self._execute_partial_close(symbol, sl_price, sl_size)
                if position.size <= 0:
                    break
            else:
                remaining_levels.append((sl_price, sl_size))
        
        # Update the stop loss levels list
        position.stop_loss_levels = remaining_levels
    
    def _execute_partial_close(self, symbol: str, exit_price: float, close_size: float):
        """Ejecuta el cierre parcial de una posición"""
        if symbol not in self.positions or self.positions[symbol].status != 'open':
            return
        
        timestamp = self.current_timestamp

        position = self.positions[symbol]
        close_size = min(close_size, position.size)
        
        # Calcular comisión
        fee = close_size * exit_price * self.taker_fee
        
        # Calcular PnL
        if position.side == 'long':
            pnl = (exit_price - position.entry_price) * close_size - fee
        else:  # short
            pnl = (position.entry_price - exit_price) * close_size - fee
        
        # Actualizar cash
        self.cash += pnl + fee  # El fee ya está descontado del pnl
        
        # Crear trade cerrado
        closed_trade = Trade(
            order_id=f"close_{position.id}",
            symbol=symbol,
            size=close_size,
            entry_price=position.entry_price,
            side=position.side,
            commission=fee,
            exit_price=exit_price,
            entry_time=position.opened_at,
            exit_time=timestamp,
        )

        # -- EDIT -- >
        # Transferir MAE y MFE desde la posición al trade cerrado
        # closed_trade.mae = position.mae
        # closed_trade.mfe = position.mfe
        closed_trade.max_price = position.max_price
        closed_trade.min_price = position.min_price
        # < -- EDIT --
        
        self.closed_trades.append(closed_trade)
        self.all_trades_history.append(closed_trade)
        
        # Actualizar posición
        position.size -= close_size
        
        if position.size <= 0:
            position.close(closed_at=timestamp)
            # Remover de trades activos
            self.trades = [t for t in self.trades if t.symbol != symbol]
        
    def set_trading_stop(self, symbol: str, tp_price: Optional[float] = None, 
                        sl_price: Optional[float] = None,
                        tp_size: Optional[float] = None,
                        sl_size: Optional[float] = None):
        """
        Establece stop loss y take profit para una posición
        
        Args:
            symbol: símbolo del activo
            tp_price: precio de take profit
            sl_price: precio de stop loss
            tp_size: tamaño para take profit (None = toda la posición)
            sl_size: tamaño para stop loss (None = toda la posición)
        """
        symbol = symbol.upper()
        
        if symbol not in self.positions or self.positions[symbol].status != 'open':
            raise ValueError(f"No hay posición abierta para {symbol}")
            
        position = self.positions[symbol]
        
        if tp_price is not None:
            size = tp_size if tp_size is not None else position.size
            size = min(size, position.size)
            position.take_profit_levels.append((tp_price, size))
            
        if sl_price is not None:
            size = sl_size if sl_size is not None else position.size
            size = min(size, position.size)
            position.stop_loss_levels.append((sl_price, size))
    
    def _close_position_with_price(self, symbol: str, exit_price: float):
        """Cierra completamente una posición al precio especificado"""
        if symbol not in self.positions or self.positions[symbol].status != 'open':
            return
            
        position = self.positions[symbol]
        self._execute_partial_close(symbol, exit_price, position.size)
    
    def close_position(self, symbol: str):
        """Cierra una posición al precio de mercado actual"""
        symbol = symbol.upper()
        
        if symbol not in self.positions or self.positions[symbol].status != 'open':
            raise ValueError(f"No hay posición abierta para {symbol}")
            
        if symbol not in self.current_close:
            raise ValueError(f"No hay precio de mercado para {symbol}")
            
        self._close_position_with_price(symbol, self.current_close[symbol])
    
    def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancela todas las órdenes pendientes"""
        if symbol:
            symbol = symbol.upper()
            orders_to_cancel = [o for o in self.orders if o.symbol == symbol]
        else:
            orders_to_cancel = self.orders[:]
    
        for order in orders_to_cancel:
            order.status = 'cancelled'
            self.orders.remove(order)
    
    def get_pending_orders(self, symbol: str = None) -> List[Order]:
        """Versión optimizada con filtrado opcional por símbolo"""
        if symbol:
            return [order for order in self.orders if order.symbol == symbol.upper()]
        return self.orders.copy()
    
    def get_position_status(self, symbol: str = None) -> Union[Position, Dict[str, Position], None]:
        """Obtiene el estado de posiciones"""
        if symbol:
            return self.positions.get(symbol.upper())
        return {k: v for k, v in self.positions.items() if v.status == 'open'}
    
    def get_all_orders_history(self) -> List[Order]:
        """Obtiene el historial completo de todas las órdenes"""
        return self.all_orders_history
    
    def get_all_trades_history(self) -> List[Trade]:
        """Obtiene el historial completo de todos los trades"""
        return self.all_trades_history
    
    def get_summary(self) -> Dict:
        """Resumen compacto del estado del broker"""
        open_positions = {k: v for k, v in self.positions.items() if v.status == 'open'}
        
        return {
            'cash': self.cash,
            'equity': self.total_equity,
            'initial_cash': self.initial_cash,
            'total_return': ((self.total_equity - self.initial_cash) / self.initial_cash) * 100,
            'open_positions': len(open_positions),
            'pending_orders': len(self.orders),
            'total_trades': len(self.all_trades_history),
            'closed_trades': len(self.closed_trades)
        }
    
    def __repr__(self):
        open_positions = len([p for p in self.positions.values() if p.status == 'open'])
        return f"Broker(Cash={self.cash:.2f}, Equity={self.total_equity:.2f}, OpenPositions={open_positions})"
    
    def __str__(self):
        return self.__repr__()

class _DataProviderBT:
    __slots__ = ["__len", "__packet", "__cache", "__symbols"]
    # posible_resample_freqs=["1min", "3min", "5min", "15min", "30min", "1h", "2h", "4h", "6h", "8h", "12h", "1D", "1W", "1ME", "1YE"]

    def __init__(self, packet: Dict[str, DataOHLC]):
        self.__cache = {} # :DataOHLC:BTCUSDT:5min:0:100
        self.__len = len(packet[list(packet.keys())[0]])
        self._validate_packet(packet) 
        self.__packet = packet# packet: {"BTCUSDT": DataOHLC}}
        self.__symbols = list(packet.keys())

    @property
    def packet(self) -> Dict[str, DataOHLC]:
        return self.__packet

    @property
    def symbols(self) -> List[str]:
        return self.__symbols

    def request(self, idx: int, symbol, timeframe: str = None, limit: int = None) -> Dict[str, DataOHLC]:
        if limit is None:
            limit = 100

        # ORIGIN DATA
        origin_data = self.__packet[symbol]
        orig_content_copy = origin_data.content.copy()
        orig_tf = origin_data.timeframe
        orig_minutes_tf = origin_data.minutes

        if timeframe is None:
            timeframe = orig_tf

        # Cache save
        __cache_key__resample = f"request:DataOHLC:{symbol}:{timeframe}:{orig_minutes_tf}"
        __cache_key__orig_idx_vals = f"request:array:{symbol}:{orig_tf}:{orig_minutes_tf}"
        if __cache_key__resample not in self.__cache:
            self.__cache[__cache_key__resample] = origin_data.resample(timeframe)  # ✅ Se ejecuta una sola vez, como antes
        if __cache_key__orig_idx_vals not in self.__cache:
            self.__cache[__cache_key__orig_idx_vals] = origin_data.index.astype("int64") 

        # RESAMPLE DATA
        resamp_data = self.__cache[__cache_key__resample]
        
        resample_minutes_tf = resamp_data.minutes 
        
        orig_index_ms = self.__cache[__cache_key__orig_idx_vals]
        
        group_resample_bars = count_available_resample_bars_nb(
            orig_index_ms, idx, orig_minutes_tf, resample_minutes_tf
        )

        effective_limit = min(limit, group_resample_bars)
        if effective_limit == 0:
            raise ValueError("No hay datos suficientes")
        start_idx = max(0, group_resample_bars - effective_limit)
        filtered_data = resamp_data[start_idx:group_resample_bars]

        #######################################
        # MODIFICAR LAS BARRAS NO CERRADAS (sin cambios)
        last_ms = orig_index_ms[idx-1]

        MS_POR_MINUTO = 60_000
        last_minute = (last_ms // MS_POR_MINUTO) % 60
        rows_to_take = ((last_minute % resample_minutes_tf) // orig_minutes_tf) + 1 # anterior: (last_minute % resample_interval) + 1

        if rows_to_take > idx:
            rows_to_take = idx
        slice_rows_to_take = slice(idx-rows_to_take, idx)

        # CREAR UNA COPIA PROFUNDA DEL CONTENIDO ANTES DE MODIFICAR
        filtered_data_content = filtered_data.content.copy()  # ← CAMBIO AQUÍ
        
        # Crear copias de los arrays numpy también
        for key in filtered_data_content:
            filtered_data_content[key] = filtered_data_content[key].copy()
        
        filtered_data_content["Open"][-1] = orig_content_copy["Open"][idx-rows_to_take]
        filtered_data_content["High"][-1] = np.max(orig_content_copy["High"][slice_rows_to_take])
        filtered_data_content["Low"][-1] = np.min(orig_content_copy["Low"][slice_rows_to_take])
        filtered_data_content["Close"][-1] = orig_content_copy["Close"][idx-1]
        filtered_data_content["Volume"][-1] = np.sum(orig_content_copy["Volume"][slice_rows_to_take])
        filtered_data_content["Turnover"][-1] = np.sum(orig_content_copy["Turnover"][slice_rows_to_take])
        
        filtered_data.update(content=filtered_data_content)

        return filtered_data

    def _validate_packet(self, data: dict) -> None:
        for symbol in data.keys():
            assert len(data[symbol]) == self.__len, f"Symbol {symbol} has different length"

class _CryptoBacktestSesh:
    """Simulador de sesión para backtesting que mantiene compatibilidad con Strategy"""
    # TODO: hacer que el warmup sea automatico con una funcion que analiza los metodos de obtencion de datos de la estrategia data 
    def __init__(self, packet: Dict[str, DataOHLC], cash: float = 100_000, maker_fee: float = 0.000, 
                taker_fee: float = 0.000, margin: float = 0.1, 
                margin_mode: str = 'isolated', mae_mfe_metric_type='ROI',
                base_coin: str = "USDT", tz: str = "UTC", warmup_period: int = 0):
        
        self.info = False
        
        # Soporte para múltiples símbolos
        if isinstance(packet, dict):
            self._main_symbol = list(packet.keys())[0]
            # data es un diccionario {"BTCUSDT": _Data, "ADAUSDT": _Data}
            self.symbols = list(packet.keys())
            # Verificar que todos los datasets tengan el mismo tamaño
            lengths = [len(d) for d in packet.values()]
            if len(set(lengths)) > 1:
                raise ValueError(f"Todos los datasets deben tener el mismo tamaño. Encontrados: {lengths}") if self.info else None
            self.data_length = lengths[0]
            # Usar el primer símbolo como principal
            self.main_symbol = self.symbols[0]
            print(f"BacktestSesh multi-símbolo configurada: {self.symbols} | {self.data_length} barras | Cash inicial: {cash}") if self.info else None
        else:
            # data es un solo _Data (compatibilidad hacia atrás)
            self._main_symbol = "BTCUSD"
            self.multi_data = {"BTCUSD": packet}
            self.symbols = ["BTCUSD"]
            self.data_length = len(packet)
            self.main_symbol = "BTCUSD"
            print(f"BacktestSesh configurada: Datos desde {packet.index[0]} hasta {packet.index[-1]} | Cash inicial: {cash}") if self.info else None
        
        # Configuración del warmup
        self.warmup_period = max(0, warmup_period)
        self.warmup_completed = False
        
        # Inicializar current_index con el período de calentamiento
        self.current_index = self.warmup_period
        
        # Mensaje informativo
        if self.warmup_period > 0:
            print(f"Período de warmup configurado: {self.warmup_period} barras") if self.info else None

        # Nuevo: Set para rastrear warnings ya mostrados
        self._shown_warnings = set() 

        self.current_bars = {}  # Barras actuales por símbolo
        self._tz = tz 
        self._base_coin = base_coin 
        self._category = "linear" 
        self._account_type = "UNIFIED" 

        self.indicator_blueprints = {} 

        self.data_provider = _DataProviderBT(packet=packet)
        # Inicializar broker interno
        self.broker = _Broker(
            cash=cash, 
            maker_fee=maker_fee, 
            taker_fee=taker_fee, 
            margin=margin,
            margin_mode=margin_mode, 
            mae_mfe_metric_type=mae_mfe_metric_type
        )

        # Agregar equity tracking en la sesión
        self._equity_curve = np.full(self.data_length, np.nan)

    # bueno
    def next(self):
        """Versión MÁXIMO RENDIMIENTO con cache NumPy"""
        if self.current_index >= self.data_length:
            return False
        
        #  Verificar warmup solo una vez
        if not self.warmup_completed and self.current_index >= self.warmup_period:
            self.warmup_completed = True
        
        # Obtener timestamp del índice actual de los datos
        current_timestamp = self.data_provider.packet[self.main_symbol].index[self.current_index] # TODO : dp
        # Actualizar timestamp en el broker
        self.broker.current_timestamp = current_timestamp

        #  ACCESO RÁPIDO
        idx = self.current_index
        for symbol in self.data_provider.symbols:            
            # Actualizar broker con datos de mercado
            filtered_content_dict = self.data_provider.request(idx=idx, symbol=symbol, limit=1).content

            self.broker.update_market(symbol, filtered_content_dict)

        #  Actualizar equity curve (solo una línea)
        self._equity_curve[self.current_index] = self.broker.total_equity
        
        self.current_index += 1
        return True

    def reset(self):
        """Reinicia el backtest"""
        self.current_index = self.warmup_period  # Reiniciar con warmup
        self.warmup_completed = False
        self.current_bars = {}

        # Limpiar warnings mostrados al reiniciar
        self._shown_warnings.clear()
        
        # Reiniciar broker
        initial_cash = self.broker.initial_cash
        self.broker = _Broker(
            cash=initial_cash,
            maker_fee=self.broker.maker_fee,
            taker_fee=self.broker.taker_fee,
            margin=self.broker.margin,
            margin_mode=self.broker.margin_mode,
            mae_mfe_metric_type=self.broker.mae_mfe_metric_type
        )

    def _validate_symbol(self, symbol: str) -> str:
        """Valida que el símbolo esté disponible en los datos"""
        symbol = symbol.upper()
        if symbol not in self.symbols:
            raise ValueError(f"Símbolo {symbol} no disponible. Símbolos disponibles: {self.symbols}")
        return symbol
    
    def _convert_timezone(self, df: pd.DataFrame, target_tz: str) -> pd.DataFrame:
        """Convierte el timezone del DataFrame"""
        if target_tz and target_tz != "UTC":
            df_copy = df.copy()
            if df_copy.index.tz is None:
                # Asumir UTC si no hay timezone
                df_copy.index = df_copy.index.tz_localize('UTC')
            df_copy.index = df_copy.index.tz_convert(target_tz)
            return df_copy
        return df
    
    def _validate_timeframe_compatibility(self, from_timeframe: str, to_timeframe: str) -> bool:
        """
        Valida que el timeframe de destino sea compatible (mayor) que el de origen.
        
        Args:
            from_timeframe: timeframe actual
            to_timeframe: timeframe solicitado
        
        Returns:
            True si es compatible, False si no
        """

        # Crear objeto Data temporal para usar el método timeframe2minutes
        temp_data = DataOHLC(pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume']))
        
        try:
            from_minutes = timeframe2minutes(from_timeframe)
            to_minutes = timeframe2minutes(to_timeframe)
            return to_minutes >= from_minutes
        except ValueError:
            return False

    def _apply_date_filters(self, df: pd.DataFrame, start: str = None, end: str = None) -> pd.DataFrame:
        """Aplica filtros de fecha start y end"""
        if start:
            start_date = pd.to_datetime(start)
            df = df[df.index >= start_date]
        if end:
            end_date = pd.to_datetime(end)
            df = df[df.index <= end_date]
        return df
    
    def _apply_limit(self, df: pd.DataFrame, limit: int = None) -> pd.DataFrame:
        """Aplica límite de cantidad de datos"""
        if limit and limit > 0:
            return df.tail(limit)  # Tomar los últimos 'limit' registros
        return df

    # Propiedades compatibles con BybitSesh
    @property
    def time_zone(self):
        return self._tz
    
    @property
    def equity(self):
        return self.broker.total_equity
    
    @property
    def equity_curve(self):
        """Retorna la curva de equity como pandas Series"""
        # Obtener el índice de tiempo del primer símbolo
        main_data = self.data_provider.packet[self.main_symbol]
        return pd.Series(self._equity_curve, index=main_data.df.index, name='Equity')

    @property
    async def total_equity(self):
        return self.broker.total_equity
    
    """Metodos comunes y caracteristicos de una Sesh"""
    # Métodos DATA - Simulados para backtesting
    # TODO: agregar el tema del start, desde el objeto data provider y de forma optima
    async def get_kline(self, symbol: str, timeframe: str = "1D", start: str = None, 
                    end: str = None, limit: int = None, category: str = "linear") -> list:
        """Simula get_kline con soporte multi-símbolo y filtros mejorados"""
        return self.data_provider.request(idx=self.current_index, symbol=symbol, timeframe=timeframe, limit=limit).klines 

    async def get_data(self, symbol: str, timeframe: str, start: str = None, end: str = None, 
                    limit: int = None, tz: str = None, category: str = "linear") -> DataOHLC:
        """Devuelve los datos históricos con soporte multi-símbolo y filtros completos"""
        return self.data_provider.request(idx=self.current_index, symbol=symbol, timeframe=timeframe, limit=limit)

    async def get_last_price(self, symbol: str) -> float: # TODO: quitar limite 2 y poner limite 1
        """Obtiene el último precio del símbolo especificado"""
        return self.data_provider.request(idx=self.current_index, symbol=symbol, limit=1).Close[-1]
 
    async def get_time(self, tz: str = None) -> pd.Timestamp:
        """Obtiene el timestamp del bar actual con conversión de timezone solo si se especifica"""
        numpy_datetime = self.data_provider.request(idx=self.current_index, symbol=self.main_symbol, limit=1).index[-1]
        return numpy_datetime.astype("M8[ms]").astype("O") # objeto datetime de python

    # Métodos CONFIGURATION (uso recomendable en el metodo 'init()')
    # async def set_time_zone(self, tz: str):
        # """Establece la zona horaria de la sesión"""
        # if tz:
        #     self._tz = tz
        #     print(f"Zona horaria establecida en: {tz}") if self.info else None
        # else:
        #     import warnings
        #     warnings.warn("Se ha asignado un valor 'None' o Nulo a la zona horaria.", DeprecationWarning)

    async def set_time_zone(self, tz: str): # TODO
        """Establece la zona horaria de la sesión y modifica los datos originales"""
        pass

    async def set_leverage(self, symbol, leverage: int = 1):
        """Establece el leverage simulado con validaciones"""
        symbol = self._validate_symbol(symbol)
        
        assert leverage > 0, "El valor de leverage debe ser mayor que 0."
        assert isinstance(leverage, int), "El valor de leverage debe ser un entero."
        
        # Simular el comportamiento de BybitSesh
        if not hasattr(self, '_leverages'):
            self._leverages = {}
        
        current_leverage = self._leverages.get(symbol, 1)
        if current_leverage == leverage:
            print(f"Nivel de apalancamiento para {symbol} ya establecido en {leverage}.") if self.info else None
            return
        
        self._leverages[symbol] = leverage
        print(f"Nivel de apalancamiento para {symbol} establecido en {leverage}.") if self.info else None
    
    async def set_margin_mode(self, margin_mode: str = "isolated"):
        """Establece el modo de margen con validaciones"""
        valid_modes = ["isolated", "cross", "portfolio"]
        
        if margin_mode.lower() not in valid_modes:
            raise ValueError(f"Modo de margen inválido. Opciones válidas: {valid_modes}")
        
        self._margin_mode = margin_mode.lower()
        print(f"Modo de margen simulado establecido en {margin_mode}.") if self.info else None

    # Métodos INFO
    async def get_account_info(self):
        """Información de cuenta más detallada"""
        return {
            "result": {
                "totalEquity": str(self.broker.total_equity),
                "totalWalletBalance": str(self.broker.cash),
                "totalMarginBalance": str(self.broker.total_equity),
                "totalAvailableBalance": str(self.broker.cash),
                "accountType": self._account_type,
                "marginMode": getattr(self, '_margin_mode', 'isolated')
            }
        }

    async def get_balance(self, coin: str = "USDT", account_type: str = None):
        """Balance más detallado"""
        account_type = account_type or self._account_type
        return {
            "totalEquity": str(self.broker.total_equity),
            "totalWalletBalance": str(self.broker.cash),
            "totalMarginBalance": str(self.broker.total_equity),
            "totalAvailableBalance": str(self.broker.cash),
            "coin": coin,
            "accountType": account_type,
            "list": [{
                "coin": coin,
                "equity": str(self.broker.total_equity),
                "walletBalance": str(self.broker.cash),
                "availableBalance": str(self.broker.cash)
            }]
        }

    async def get_position_status(self, symbol) -> dict:
        position = self.broker.get_position_status(symbol)
        if position and position.status == 'open':
            return {
                "size": position.size,
                "side": "Buy" if position.side == "long" else "Sell"
            }
        return {"size": 0.0, "side": "None"}
    
    async def get_all_orders(self, symbol) -> list:
        orders = self.broker.get_all_orders_history()
        return [{
            "orderId": order.id,
            "symbol": order.symbol,
            "side": order.side.title(),
            "orderType": order.order_type.title(),
            "qty": str(order.size),
            "price": str(order.price) if order.price else "",
            "orderStatus": order.status.title(),
            "createdTime": str(int(npdt64_to_datetime(order.created_at).timestamp() * 1000))
        } for order in orders if order.symbol == symbol.upper()]
    
    async def get_leverage(self, symbol: str) -> int:
        """Obtiene el leverage configurado para un símbolo"""
        symbol = self._validate_symbol(symbol)
        
        if not hasattr(self, '_leverages'):
            self._leverages = {}
        
        return self._leverages.get(symbol, 1)

    # Métodos TRADING - Delegados al broker
    async def buy(self, symbol: str, size: float, price: float = None, 
                 sl_price: float = None, tp_price: float = None,
                 pct_sl: float = None, pct_tp: float = None, 
                 time_in_force: str = "GTC") -> dict:
        """
            Coloca una orden de compra con soporte para SL y TP por precio o porcentaje.
            
            Parámetros:
            - symbol: Símbolo del activo (ej. "BTCUSDT")
            - size: Cantidad a comprar
            - price: Precio límite (opcional)
            - sl_price: Precio de stop loss (opcional)
            - tp_price: Precio de take profit (opcional)
            - pct_sl: Porcentaje de stop loss (opcional)
            - pct_tp: Porcentaje de take profit (opcional)
            - time_in_force: Duración de la orden (GTC por defecto)
        """
        
        # Calcular precios de SL y TP si se proporcionan porcentajes
        current_price = await self.get_last_price(symbol)
        
        if pct_sl and not sl_price:
            sl_price = current_price * (1 - pct_sl)
        if pct_tp and not tp_price:
            tp_price = current_price * (1 + pct_tp)
        
        # Colocar orden
        order_id = self.broker.place_order(
            symbol=symbol,
            type='buy',
            size=size,
            limit=price,
            take_profit=tp_price,
            stop_loss=sl_price
        )
        
        return {
            "result": {
                "orderId": order_id
            },
            "retMsg": "OK"
        }
    
    async def sell(self, symbol: str, size: float, price: float = None,
                  sl_price: float = None, tp_price: float = None,
                  pct_sl: float = None, pct_tp: float = None,
                  time_in_force: str = "GTC") -> dict:
        
        # Calcular precios de SL y TP si se proporcionan porcentajes
        current_price = await self.get_last_price(symbol)
        
        if pct_sl and not sl_price:
            sl_price = current_price * (1 + pct_sl)
        if pct_tp and not tp_price:
            tp_price = current_price * (1 - pct_tp)
        
        # Colocar orden
        order_id = self.broker.place_order(
            symbol=symbol,
            type='sell',
            size=size,
            limit=price,
            take_profit=tp_price,
            stop_loss=sl_price
        )
        
        return {
            "result": {
                "orderId": order_id
            },
            "retMsg": "OK"
        }
    
    async def close_position(self, symbol):
        try:
            self.broker.close_position(symbol)
            print(f"Posición cerrada para {symbol}") if self.info else None
        except ValueError as e:
            print(f"Error al cerrar posición: {e}") if self.info else None
    
    async def set_trading_stop(self, symbol: str, tp_price: float = None,
                              sl_price: float = None, **kwargs):
        try:
            self.broker.set_trading_stop(
                symbol=symbol,
                tp_price=tp_price,
                sl_price=sl_price
            )
        except ValueError as e:
            print(f"Error al establecer trading stop: {e}") if self.info else None
    
    async def cancel_all_orders(self, symbol, order_filter: str = None) -> dict:
        self.broker.cancel_all_orders(symbol)
        return {"retMsg": "OK"}
    
    # Métodos específicos para backtesting
    def get_broker_summary(self):
        return self.broker.get_summary()
    
    def get_trades_history(self):
        return self.broker.get_all_trades_history()
    
    def get_closed_trades(self):
        return self.broker.closed_trades

class Backtest:
    """Clase principal para ejecutar backtests con estrategias"""
    def __init__(self, strategy: Strategy, packet: Dict[str, DataOHLC], cash: float = 100_000, 
                 maker_fee: float = 0.000, taker_fee: float = 0.000, 
                 margin: float = 0.1, margin_mode: str = 'isolated', 
                 mae_mfe_metric_type='ROI', tz: str = "UTC",
                 warmup: int = 0,
                 **strategy_params
    ):
        self._total_bars = len(packet)
        self._strategy = strategy
        self._strategy_params = strategy_params
        self._warmup = max(0, warmup)  # Asegurar que no sea negativo
        
        # Procesar el parámetro data según su tipo
        # processed_data = self._process_data_parameter(packet)
        self._packet = packet
        self._symbols = list(packet.keys())
        # Determinar la longitud de los datos
        self._packet_data_length = len(packet[self._symbols[0]])

        # Validar que el warmup no sea mayor que los datos disponibles
        if self._warmup >= self._packet_data_length:
            raise ValueError(f"El período de warmup ({self._warmup}) no puede ser mayor o igual al número de barras disponibles ({self._packet_data_length})")
        
        # Crear sesión de backtest con warmup
        self._sesh = _CryptoBacktestSesh(
            packet=self._packet,
            cash=cash,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            margin=margin,
            margin_mode=margin_mode,
            mae_mfe_metric_type=mae_mfe_metric_type,
            tz=tz,
            warmup_period=self._warmup  # Pasar warmup a la sesión
        )

        # Initialize backtest attributes
        self._cash = cash
        self._packet = packet
        self._maker_fee = maker_fee
        self._taker_fee = taker_fee
        self._margin = margin
        self._margin_mode = margin_mode
        self._mae_mfe_metric_type = mae_mfe_metric_type
        self._tz = tz
        self._warmup = warmup
        self._strategy_params = strategy_params

        # Inicializar variable para almacenar resultados
        self._results = None

        # Instanciar estrategia y concederle permisos de session
        self._strategy = strategy(**strategy_params).set_sesh(self._sesh)

        # Traer los blueprints de indicadores
        self._indicator_blueprints = self._sesh.indicator_blueprints

        self._current_timestamp = None

        self._info = False

    def _process_data_parameter(self, data): # TODO
        """Procesa el parámetro data según su tipo y lo convierte al formato esperado"""
    
        pass
        # # Caso 1: Ya es un diccionario
        # if isinstance(data, dict):
        #     processed_dict = {}
            
        #     for symbol, data_obj in data.items():
        #         if not isinstance(symbol, str):
        #             raise ValueError(f"Las claves del diccionario deben ser strings (símbolos). Encontrado: {type(symbol)}")
                
        #         # Verificar si es un DataFrame de pandas
        #         if isinstance(data_obj, pd.DataFrame):
        #             # Convertir DataFrame a objeto Data
        #             processed_dict[symbol] = DataOHLC(data_obj, symbol_name=symbol)
                
        #         # Verificar si es un objeto Data
        #         elif hasattr(data_obj, 'df') and hasattr(data_obj, 'symbol_name'):
        #             processed_dict[symbol] = data_obj
                
        #         else:
        #             raise ValueError(
        #                 f"Los valores del diccionario deben ser objetos Data o DataFrames de pandas. "
        #                 f"Encontrado para '{symbol}': {type(data_obj)}"
        #             )
            
        #     return processed_dict
        
        # # Caso 2: Es una lista de objetos Data
        # elif isinstance(data, list):
        #     processed_dict = {}
        #     for i, data_obj in enumerate(data):
        #         if not hasattr(data_obj, 'df') or not hasattr(data_obj, 'symbol_name'):
        #             raise ValueError(f"Todos los elementos de la lista deben ser objetos Data. Elemento {i}: {type(data_obj)}")
                
        #         # Extraer el symbol_name
        #         symbol = data_obj.symbol_name
        #         if not symbol:
        #             raise ValueError(f"El objeto Data en la posición {i} no tiene symbol_name definido")
                
        #         # Verificar que no haya símbolos duplicados
        #         if symbol in processed_dict:
        #             raise ValueError(f"Símbolo duplicado encontrado: {symbol}")
                
        #         processed_dict[symbol] = data_obj
            
        #     if not processed_dict:
        #         raise ValueError("La lista de datos no puede estar vacía")
            
        #     return processed_dict
        
        # # Caso 3: Es un solo objeto Data
        # elif hasattr(data, 'df') and hasattr(data, 'symbol_name'):
        #     symbol = data.symbol_name
        #     if not symbol:
        #         raise ValueError("El objeto Data debe tener symbol_name definido")
            
        #     return {symbol: data}
        
        # # Caso 4: Es un DataFrame de pandas individual
        # elif isinstance(data, pd.DataFrame):
        #     raise ValueError(
        #         "Para usar un DataFrame individual, debe proporcionarse dentro de un diccionario "
        #         "con el símbolo como clave: {'BTCUSDT': dataframe}"
        #     )
        
        # # Caso 5: Formato no soportado
        # else:
        #     raise ValueError(
        #         f"El parámetro 'data' debe ser:\n"
        #         f"- Un diccionario {{símbolo: objeto_Data}} o {{símbolo: DataFrame}}\n"
        #         f"- Un objeto Data con symbol_name definido\n"
        #         f"- Una lista de objetos Data con symbol_name definido\n"
        #         f"Recibido: {type(data)}"
        #     )

    def reset(self):
        """Reinicia el backtest"""
        self._sesh.reset()
        # Reinicializar current_index con warmup
        self._sesh.current_index = self._warmup
        self._sesh.warmup_completed = False
        self._strategy = self._strategy(**self._strategy_params).set_sesh(self._sesh)

    async def _run(self, p_bar=True, mae_metric_type="ROI"):
        """Ejecuta el backtest completo con barra de progreso avanzada"""
        effective_bars = self._packet_data_length - self._warmup
        print(f" Iniciando backtest con {self._packet_data_length} barras (warmup: {self._warmup}, efectivas: {effective_bars})...") if self._info else None

        # Calcular tiempo inicial 
        self.start_time = time.time()
        
        # Inicializar estrategia
        await self._strategy.init()

        # pre-Limpiar terminal de barras duplicadas
        print("\r", end="")  # Retorno de carro para limpiar línea actual   
        print("\033[K", end="")  # Limpiar desde cursor hasta final de línea

        # Crear barra de progreso con posición específica
        progress_bar = tqdm(
            total=effective_bars, 
            desc=f" • 〽Backtesting {" "+self._strategy.name+" ":ꞏ^20}", # ꞏ 
            leave=False,
            # ncols=120,
            # dynamic_ncols=True,
            disable=not p_bar,
            colour="#57C8D2", #'#B86217', #03A7D0, #AA77DA
            position=0,
            bar_format='{desc} {percentage:3.0f}% ﴾{bar}﴿ [{elapsed}<{remaining}, {rate_fmt}]{postfix}', # ⌠⌡ |││ ﴾﴿
            miniters=100,
            mininterval=0.25,
            maxinterval=5.0,
        )
        
        # Iterar por cada barra
        bars_processed = 0
        strategy_bars = 0 
        
        try:
            while self._sesh.next():
                bars_processed += 1
                
                # Solo ejecutar estrategia después del período de warmup
                if self._sesh.warmup_completed:
                    await self._strategy.on_data()

                    strategy_bars += 1
                    progress_bar.update()
                    
            await self._strategy.on_stop()
            
        finally:
            # progress_bar.colour = "#8DBA54"
            progress_bar.leave = True
            progress_bar.refresh()
            progress_bar.close()


        # Calcular tiempo final
        self.total_time = time.time() - self.start_time

        # Limpiar terminal de barras duplicadas
        print("\r", end="")  # Retorno de carro para limpiar línea actual   
        print("\033[K", end="")  # Limpiar desde cursor hasta final de línea

        # Calcular estadísticas completas
        summary = self._sesh.get_broker_summary()
        trades = self._sesh.get_closed_trades()
        
        # Obtener datos OHLC y equity curve
        # Asumiendo que los datos están en el primer símbolo si es un diccionario
        if isinstance(self._packet, dict):
            first_symbol = list(self._packet.keys())[0]
            ohlc_data = self._packet[first_symbol].df
        else:
            ohlc_data = self._packet.df if hasattr(self._packet, 'df') else self._packet
        
        # Calcular estadísticas completas
        equity_curve = self._sesh.equity_curve.bfill()
        stats = compute_stats(
            trades=trades,
            ohlc_data=ohlc_data,
            equity_curve=equity_curve,
            strategy_instance=self._strategy,
            mae_metric_type=mae_metric_type
        )

        # Guardar los indicadores de la sesion
        self._indicator_blueprints = self._sesh.indicator_blueprints
        
        # Guardar resultados en self._results
        self._results = stats
        
        return stats
        # return summary, trades

    def run(self, p_bar=True, mae_metric_type="ROI"): # _run_bt_as_sync
        """ Ejejcutar backtest de forma syncrona """
        return asyncio.run(self._run(p_bar=p_bar, mae_metric_type=mae_metric_type))

    def plot(self, 
        results=None,
        filename=None,
        plot_equity=True,
        plot_return=False,
        # plot_pl=True,
        plot_volume=True,
        # plot_drawdown=False,
        plot_trades=True,
        # smooth_equity=False,
        relative_equity=True,
        # superimpose=True,
        # superimpose_freq_rule=None,
        resample=True,
        resample_freq_rule=None,
        # show_legend=True,
        timeframe=None
    ):
        """
            Genera un gráfico interactivo del backtest usando Bokeh
            
            Args:
                results: Resultados del backtest (usa self._results si es None)
                filename: Nombre del archivo para guardar (opcional)
                plot_width: Ancho del gráfico
                plot_equity: Si mostrar curva de equity
                plot_return: Si mostrar returns
                plot_pl: Si mostrar P&L
                plot_volume: Si mostrar volumen
                plot_drawdown: Si mostrar drawdown
                plot_trades: Si mostrar trades en el gráfico
                smooth_equity: Si suavizar la curva de equity
                relative_equity: Si mostrar equity relativo
                superimpose: Si superponer indicadores
                superimpose_freq_rule: Regla de frecuencia para superposición
                resample: Si remuestrear datos
                resample_freq_rule: Regla de frecuencia para remuestreo
                show_legend: Si mostrar leyenda
                **kwargs: Argumentos adicionales para indicadores
            
            Returns:
                Layout de Bokeh con el gráfico completo
        """

        
        # Usar resultados almacenados si no se proporcionan
        if results is None:
            if self._results is None:
                raise ValueError("No hay resultados disponibles. Ejecuta bt.run() primero.")
            results = self._results
        
        # Obtener datos OHLC principales
        if isinstance(self._packet, dict):
            # Si es un diccionario, usar el primer símbolo
            first_symbol = list(self._packet.keys())[0]
            main_ohlc_data = self._packet[first_symbol]
        else:
            # Si es un objeto Data único
            main_ohlc_data = self._packet# if hasattr(self._data, 'df') else self._data
        
        # Configurar output para notebook si es necesario
        try:
            # Verificar si estamos en un notebook
            from IPython import get_ipython
            if get_ipython() is not None:
                output_notebook()
        except ImportError:
            pass
        
        # Mostrar el gráfico
        if filename:
            from bokeh.io import output_file
            output_file(filename)
        
        # Convertir a Data object
        # main_ohlc_data = Data(main_ohlc_data)
        return _plot(
            stats=results,
            ohlc_base_data=main_ohlc_data,
            indicators_blueprints=self._indicator_blueprints,
            timeframe=timeframe,
            plot_equity = plot_equity,
            plot_return = plot_return,
            # TODO plot_drawdown = plot_drawdown,
            plot_trades = plot_trades,
            plot_volume = plot_volume,
            relative_equity = relative_equity
        )

    @property
    def sesh(self):
        return self._sesh


def run_backtests(backtests:[Backtest, ...], p_bar=True):
    """Permite ejecutar uno o mas backtestings en Paralelo"""
    # Asignar color de barra a cada backtest
    colors = ["#57C8D2FF","#7477CEFF","#AA77DAFF","#6593D8FF","#B7C577FF"]
    for i, bt in enumerate(backtests):
        color = colors[i % 5]
        bt._bar_color = color[0:7] # limpiar el codigo hex de caracteres adicionales

    # Ejecutar backtests en paralelo
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(Backtest.run, bt, p_bar) for bt in backtests] 
        results = [f.result() for f in futures]

    # Limpiar terminal después de todos los procesos
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    print("\r", end="")
    print("\033[K", end="")

    return results
