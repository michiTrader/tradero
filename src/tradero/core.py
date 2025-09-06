
import pandas as pd
from typing import Callable, Union, Sequence
from numbers import Number
from .ta import PivotIndicator
from .models import DataOHLC, CryptoSesh
from .util import  find_minutes_timeframe, minutes2timeframe

class Strategy:
    """ Clase base para estrategias de trading en Live"""
    def __init__(self): # , sesh: BybitSesh
        self.sesh = None 
        self._backtest = None
        
    def set_sesh(self, sesh):
        self.sesh = sesh
        return self

    def get_sesh(self) -> CryptoSesh: 
        """ 
        DATA:
            get_kline, 
            get_data, 
            get_last_price, 
            get_time
        INFO: 
            get_account_info, 
            get_balance, 
            get_position_status, 
            get_all_orders, 
            get_leverage
        CONFIGURATION:
            set_time_zone, 
            set_leverage, 
            set_margin_mode
        Trading:
            sell, 
            buy, 
            close_position, 
            set_trading_stop, 
            cancel_all_orders
        """
        return self.sesh
         
    def optimize(self, *args, **kwargs):
        """ """
        # # Backtetear y optimizar parametros (la estrategia debe ser compatible con backtesting.py)
        # bt = backtesting.optimize(self, data)  (visualizacion por ejemplo)

        # # setear los parametros optimizados
        # self.params = bt.best_paramas
        pass

    async def init(self):
        """ """

    async def on_data(self):
        """ """

    async def on_exit(self):
        """ """

    async def I(self, 
        indicator_obj: Callable, 
        data: Union[DataOHLC, PivotIndicator], 
        only_visual=False,
        **kwargs, 
    ):
        """ Guarda en la Sesh los datos del indicador en formato: dict(func, timeframe, kwargs)"""

        # TODO: assert: el timeframe del indicador no puede ser menor< al timeframe de la data original 
        if not isinstance(data, DataOHLC):
            data = data.data # extraer el objeto Data del indicador

        # Guardar el in indicador en la session una sola vez
        self._save_bp_indicator_in_sesh_only_once(obj=indicator_obj, data=data, **kwargs)

        if only_visual:
            return 
        
        # Retornar el indicador instanciado
        return indicator_obj(data, **kwargs)

    def _save_bp_indicator_in_sesh_only_once(self, obj: Callable, data: 'DataOHLC', **kwargs):
        f"""Guarda los indicadores con los claves ['func', 'index', 'timeframe', 'kwargs']"""

        # identificar el timeframe de los datos del indicador
        # print(f"data.index: {data.index} core 1373 _save_bp_indicator_in_sesh_only_once")
        indicator_data_timeframe = find_minutes_timeframe(data.index)

        # Crear el tag idetificador 
        str_timeframe = f"{minutes2timeframe(indicator_data_timeframe)}"
        str_name = obj.__name__
        str_kwargs = " ".join((map(lambda x: f"{x[1]}", (kwargs.items()))))

        tag_indicator = f"{str_name} {str_kwargs} ·{str_timeframe}" 

        if tag_indicator not in self.sesh.indicator_blueprints.keys():
            sesh = self.sesh
            sesh.indicator_blueprints[tag_indicator] = dict(
                obj=obj, 
                timeframe=str_timeframe, 
                kwargs=kwargs
                )

    @property
    def name(self):
        return self.__class__.__name__

def crossover(series1: Sequence, series2: Sequence) -> bool:
    """
    Return `True` if `series1` just crossed over (above)
    `series2`.

        >>> crossover(self.data.Close, self.sma)
        True
    """
    series1 = (
        series1.values if isinstance(series1, pd.Series) else
        (series1, series1) if isinstance(series1, Number) else
        series1)
    series2 = (
        series2.values if isinstance(series2, pd.Series) else
        (series2, series2) if isinstance(series2, Number) else
        series2)
    try:
        return series1[-2] < series2[-2] and series1[-1] > series2[-1]  # type: ignore
    except IndexError:
        return False

def cross(series1: Sequence, series2: Sequence) -> bool:
    """
    Return `True` if `series1` and `series2` just crossed
    (above or below) each other.

        >>> cross(self.data.Close, self.sma)
        True

    """
    return crossover(series1, series2) or crossover(series2, series1)


