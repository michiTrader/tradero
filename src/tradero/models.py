import numpy as np
import pandas as pd
from typing import Dict, Callable, Union
from abc import ABC, abstractclassmethod
import pandas as pd
import asyncio
from tradero.lib import timeframe2minutes, minutes2timeframe, find_minutes_timeframe
from tradero._util import _as_str, BacktestLogger
import logging
import sounddevice as sd
from pathlib import Path
from pintar import HEX
from pintar.logging import Theme, PintarStreamHandler, PintarFileHandler

# TODO: Pasar a algun submodulo y tambien una funcion 'beep' para que el metodo 'beep' de strategy use esa funcion
sd.default.latency = ('low', 'low')
sd.default.blocksize = 256

"""
    Una Sesh debe tener las siguientes funciones: 
        DATA: get_kline, get_data, get_last_price, get_time
        CONFIGURATION: set_margin_mode, set_leverage, set_time_zone
        INFO: get_account_info, get_balance, get_position_status, get_all_orders, get_leverage
        TRADING: sell, sell, close_position, set_trading_stop, cancel_all_orders
    Una Sesh debe tener las siguientes propiedades: 
        time_zone, equity, total_equity, 
"""

class _Array(np.ndarray):
    """Wrapper para arrays numpy que mantiene el índice del DataFrame original."""
    def __new__(cls, input_array, index=None):
        obj = np.asarray(input_array).view(cls)
        obj.index = index
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.index = getattr(obj, 'index', None) # saw  

    @property
    def s(self):
        """Convierte el array actual a una Serie de pandas con el índice almacenado."""
        return pd.Series(self, index=self.index)

# class DataOHLC:
    # """
    # Acceso optimizado a datos OHLCV usando NumPy para operaciones rápidas.
    # Proporciona acceso a columnas como arrays numpy con caché para mejor performance.
    # """ 
    # __slots__ = ["symbol", "__content", "__len", "__cache", "__arrays", "__timeframe", "__minutes_tf", "__kwargs"]

    # def __init__(self, content: Dict[str, list], timeframe: str = None, symbol=None, **kwargs):
    #     """ content: dict con las culmas o h l c v datetime"""
    #     self.__cache = {}
    #     self.__content = None # en update()
    #     self.__len = None # en update()
    #     self.__arrays: Dict[str, _Array] = {}
    #     self.update(content)
    #     self.__timeframe = timeframe
    #     self.__minutes_tf = None
    #     self.symbol = symbol 

    #     for key, value in kwargs.items():
    #         setattr(self, key, value)

    # def update(self, content: Dict[str, list]):
    #     """Actualiza los arrays internos desde el DataFrame."""
    #     self.__cache = {}
    #     self.__content = content.copy()
    #     self.__arrays["__index"] = np.array(content['datetime'], dtype="datetime64[ms]")
    #     self.__arrays["Open"] = np.array(content["Open"])
    #     self.__arrays["High"] = np.array(content["High"])
    #     self.__arrays["Low"] = np.array(content["Low"])
    #     self.__arrays["Close"] = np.array(content["Close"])
    #     self.__arrays["Volume"] = np.array(content["Volume"])
    #     self.__arrays["Turnover"] = np.array(content["Turnover"])

    #     self.__len = len(self.__arrays["__index"])

    # def resample(self, timeframe: str) -> 'DataOHLC':
    #     """
    #     Resamplea los datos a un intervalo diferente.
    #     """
    #     freq = timeframe
        
    #     df = pd.DataFrame(self.__content.copy()).set_index('datetime')
    #     df.sort_index(inplace=True)

    #     # Resamplear OHLC exactamente como resample_ohlc_indicator
    #     resampler = df.resample(freq)
    #     ohlc_resampled = resampler.agg({
    #         'Open': 'first',
    #         'High': 'max', 
    #         'Low': 'min',
    #         'Close': 'last',
    #         'Volume': 'sum',
    #         'Turnover': 'sum',
    #     })

    #     # Convertir a content dict
    #     dict_with_data_resampled = ohlc_resampled.dropna().reset_index().to_dict(orient="list")
    #     return DataOHLC(
    #         content = dict_with_data_resampled, 
    #         timeframe = freq, 
    #         symbol = self.symbol,
    #     )

    # def copy(self) -> 'DataOHLC':
    #     """Crea una copia profunda del objeto Data."""
    #     return DataOHLC(self.__content.copy(), self.__timeframe, self.symbol)

    # def __get_array(self, key: str) -> _Array:
    #     """Obtiene un array desde el caché o lo carga si no está."""
    #     return self.__arrays[key]

    # def __len__(self) -> int:
    #     """Longitud actual de los datos."""
    #     return self.__len

    # def __repr__(self) -> str:
    #     # i = min(self.__len, len(self.__df)) - 1
    #     i = self.__len - 1
    #     start_index = self.index[0]
    #     end_index = self.index[i]
    #     # items = ', '.join(f'{k}={v:.2f}' if isinstance(v, (float, np.floating)) else f'{k}={v}'
    #     #                   for k, v in self.__df.iloc[i].items())
    #     return f"<'{self.symbol}': timeframe={self.__timeframe}, len={self.__len}, start='{start_index}', end='{end_index}')>"

    # def __getitem__(self, key):
    #     """Dictionary-style access to arrays with support for slicing."""
    #     if isinstance(key, slice):
    #         # Evitar crear un nuevo objeto si el slice es completo
    #         if key.start is None and key.stop is None and key.step is None:
    #             return self
    #         # Create a new _Data object with sliced data
    #         # sliced_dict = {col: arr[key] for col, arr in self.__content.items()}
    #         sliced_dict = {}
    #         sliced_dict['datetime'] = self.index[key]
    #         sliced_dict['Open'] = self.Open[key]
    #         sliced_dict['High'] = self.High[key]
    #         sliced_dict['Low'] = self.Low[key]
    #         sliced_dict['Close'] = self.Close[key]
    #         sliced_dict['Volume'] = self.Volume[key]
    #         sliced_dict['Turnover'] = self.Turnover[key]

    #         data_obj = DataOHLC(sliced_dict, self.__timeframe, self.symbol)
    #         return data_obj
    #     elif isinstance(key, (int, np.integer)):
    #         # Return a dictionary with values at the specified index
    #         end_point = key+1 or None
    #         sliced_dict = {}
    #         sliced_dict['datetime'] = self.index[key:end_point]
    #         sliced_dict['Open'] = self.Open[key:end_point]
    #         sliced_dict['High'] = self.High[key:end_point]
    #         sliced_dict['Low'] = self.Low[key:end_point]
    #         sliced_dict['Close'] = self.Close[key:end_point]
    #         sliced_dict['Volume'] = self.Volume[key:end_point]
    #         sliced_dict['Turnover'] = self.Turnover[key:end_point]  

    #         data_obj = DataOHLC(sliced_dict, self.__timeframe, self.symbol)
    #         return data_obj
    #     else:
    #         # Handle string keys (column names)
    #         return self.__get_array(key)

    # # def __getattr__(self, key) -> _Array:
    # #     """Acceso tipo atributo a las columnas OHLCV."""
    # #     try:
    # #         return self.__get_array(key)
    # #     except KeyError:
    # #         raise AttributeError(f"Columna '{key}' no encontrada") from None

    # def __getattr__(self, key):
    #     # Evitar recursión infinita
    #     if key.startswith('_'):  # o cualquier condición que evite el bucle
    #         raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        
    #     try:
    #         return self.__get_array(key)
    #     except KeyError:
    #         raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    

    # # def __getstate__(self):
    # #     # Retorna el estado que puede ser serializado
    # #     return self.__dict__.copy()

    # # def __setstate__(self, state):
    # #     # Restaura el estado después de deserializar
    # #     self.__dict__.update(state)

    # @property
    # def timeframe(self) -> str:
    #     if self.__timeframe is None:
    #         self.__timeframe = minutes2timeframe(find_minutes_timeframe(self.index))
    #     return self.__timeframe

    # @property
    # def minutes(self) -> int:
    #     if self.__minutes_tf is None:
    #         self.__minutes_tf = timeframe2minutes(self.__timeframe)
    #     return self.__minutes_tf

    # @property
    # def content(self) -> dict:
    #     """Devuelve el DataFrame."""
    #     return self.__content

    # @property
    # def df(self) -> pd.DataFrame:
    #     """Devuelve el DataFrame."""
    #     cache_key = 'df'
    #     if cache_key not in self.__cache: # actualiza el caché
    #         self.__cache[cache_key] = pd.DataFrame(self.__content).set_index("datetime")
    #     return  self.__cache[cache_key]
    #     # return pd.DataFrame(self.__content).set_index("datetime")

    # @property 
    # def klines(self) -> np.ndarray: 
    #     cache_key = f'klines'
    #     if cache_key not in self.__cache:
    #         timestamps_ms = (self.df.index.values.astype(np.int64) // 10**6).reshape(-1, 1)
    #         data_values = self.df.to_numpy()
    #         self.__cache[cache_key] = np.hstack([timestamps_ms, data_values])
    #     return self.__cache[cache_key]  

    # @property
    # def Open(self) -> _Array:
    #     """Precios de apertura como array numpy."""
    #     return self.__get_array('Open')

    # @property
    # def High(self) -> _Array:
    #     """Precios máximos como array numpy."""
    #     return self.__get_array('High')

    # @property
    # def Low(self) -> _Array:
    #     """Precios mínimos como array numpy."""
    #     return self.__get_array('Low')

    # @property
    # def Close(self) -> _Array:
    #     """Precios de cierre como array numpy."""
    #     return self.__get_array('Close')

    # @property
    # def Volume(self) -> _Array:
    #     """Volúmenes como array numpy."""
    #     return self.__get_array('Volume')

    # @property
    # def Turnover(self) -> _Array:
    #     """Volúmenes como array numpy."""
    #     return self.__get_array('Turnover')

    # @property
    # def index(self) -> pd.Index:
    #     """Índice temporal del DataFrame."""
    #     return self.__get_array('__index')

    # @property
    # def empty(self):
    #     """Devuelve True si no hay datos disponibles."""
    #     return self.__len == 0

class DataOHLC:
    """
    Acceso optimizado a datos OHLCV usando NumPy para operaciones rápidas.
    Proporciona acceso a columnas como arrays numpy con caché para mejor performance.
    """ 
    __slots__ = ["symbol", "__content", "__len", "__cache", "__arrays", "__timeframe", "__minutes_tf", "__kwargs"]

    def __init__(self, content: Dict[str, list], timeframe: str = None, symbol=None, **kwargs):
        """ content: dict con las culmas o h l c v datetime"""
        self.__cache = {}
        self.__content = None # en update()
        self.__len = None # en update()
        self.__arrays: Dict[str, np.ndarray] = {}
        self.update(content)
        self.__timeframe = timeframe
        self.__minutes_tf = None
        self.symbol = symbol 

        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, content: Dict[str, list]):
        """Actualiza los arrays internos desde el DataFrame."""
        self.__cache = {}
        self.__content = content.copy()
        self.__arrays["__index"] = np.array(content['datetime'], dtype="datetime64[ms]")
        self.__arrays["Open"] = np.array(content["Open"])
        self.__arrays["High"] = np.array(content["High"])
        self.__arrays["Low"] = np.array(content["Low"])
        self.__arrays["Close"] = np.array(content["Close"])
        self.__arrays["Volume"] = np.array(content["Volume"])
        self.__arrays["Turnover"] = np.array(content["Turnover"])

        self.__len = len(self.__arrays["__index"])

        self.__arrays["__nbars"] = np.arange(1, self.__len + 1) # np.arange(1, self.__len + 1)  self.__arrays["__index"].astype(np.int64)

    def resample(self, timeframe: str) -> 'DataOHLC':
        """
        Resamplea los datos a un intervalo diferente.
        """
        freq = timeframe
        
        df = pd.DataFrame(self.__content.copy()).set_index('datetime')
        df.sort_index(inplace=True)

        # Resamplear OHLC exactamente como resample_ohlc_indicator
        resampler = df.resample(freq)
        ohlc_resampled = resampler.agg({
            'Open': 'first',
            'High': 'max', 
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Turnover': 'sum',
        })

        # Convertir a content dict
        dict_with_data_resampled = ohlc_resampled.dropna().reset_index().to_dict(orient="list")
        return DataOHLC(
            content = dict_with_data_resampled, 
            timeframe = freq, 
            symbol = self.symbol,
        )

    def copy(self) -> 'DataOHLC':
        """Crea una copia profunda del objeto Data."""
        return DataOHLC(self.__content.copy(), self.__timeframe, self.symbol)

    def __get_array(self, key: str) -> np.ndarray:
        """Obtiene un array desde el caché o lo carga si no está."""
        return self.__arrays[key]

    def __len__(self) -> int:
        """Longitud actual de los datos."""
        return self.__len

    def __repr__(self) -> str:
        # i = min(self.__len, len(self.__df)) - 1
        i = self.__len - 1
        start_index = self.index[0]
        end_index = self.index[i]
        # items = ', '.join(f'{k}={v:.2f}' if isinstance(v, (float, np.floating)) else f'{k}={v}'
        #                   for k, v in self.__df.iloc[i].items())
        return f"<'{self.symbol}': timeframe={self.__timeframe}, len={self.__len}, start='{start_index}', end='{end_index}')>"

    def __getitem__(self, key):
        """Dictionary-style access to arrays with support for slicing."""
        if isinstance(key, slice):
            # Evitar crear un nuevo objeto si el slice es completo
            if key.start is None and key.stop is None and key.step is None:
                return self
            # Create a new _Data object with sliced data
            # sliced_dict = {col: arr[key] for col, arr in self.__content.items()}
            sliced_dict = {}
            sliced_dict['datetime'] = self.index[key]
            sliced_dict['Open'] = self.Open[key]
            sliced_dict['High'] = self.High[key]
            sliced_dict['Low'] = self.Low[key]
            sliced_dict['Close'] = self.Close[key]
            sliced_dict['Volume'] = self.Volume[key]
            sliced_dict['Turnover'] = self.Turnover[key]

            data_obj = DataOHLC(sliced_dict, self.__timeframe, self.symbol)
            return data_obj
        elif isinstance(key, (int, np.integer)):
            # Return a dictionary with values at the specified index
            end_point = key+1 or None
            sliced_dict = {}
            sliced_dict['datetime'] = self.index[key:end_point]
            sliced_dict['Open'] = self.Open[key:end_point]
            sliced_dict['High'] = self.High[key:end_point]
            sliced_dict['Low'] = self.Low[key:end_point]
            sliced_dict['Close'] = self.Close[key:end_point]
            sliced_dict['Volume'] = self.Volume[key:end_point]
            sliced_dict['Turnover'] = self.Turnover[key:end_point]  

            data_obj = DataOHLC(sliced_dict, self.__timeframe, self.symbol)
            return data_obj
        else:
            # Handle string keys (column names)
            return self.__get_array(key)

    # def __getattr__(self, key) -> np.ndarray:
        # """Acceso tipo atributo a las columnas OHLCV."""
        # try:
        #     return self.__get_array(key)
        # except KeyError:
        #     raise AttributeError(f"Columna '{key}' no encontrada") from None

    def __getattr__(self, key):
        # Evitar recursión infinita
        if key.startswith('_'):  # o cualquier condición que evite el bucle
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        
        try:
            return self.__get_array(key)
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    

    # def __getstate__(self):
        # Retorna el estado que puede ser serializado
        # return self.__dict__.copy()

    # def __setstate__(self, state):
        # # Restaura el estado después de deserializar
        # self.__dict__.update(state)

    @property
    def timeframe(self) -> str:
        if self.__timeframe is None:
            self.__timeframe = minutes2timeframe(find_minutes_timeframe(self.index))
        return self.__timeframe

    @property
    def minutes(self) -> int:
        if self.__minutes_tf is None:
            self.__minutes_tf = timeframe2minutes(self.__timeframe)
        return self.__minutes_tf

    @property
    def content(self) -> dict:
        """Devuelve el DataFrame."""
        return self.__content

    @property
    def df(self) -> pd.DataFrame:
        """Devuelve el DataFrame."""
        cache_key = 'df'
        if cache_key not in self.__cache: # actualiza el caché
            self.__cache[cache_key] = pd.DataFrame(self.__content).set_index("datetime")
        return  self.__cache[cache_key]
        # return pd.DataFrame(self.__content).set_index("datetime")

    @property 
    def klines(self) -> np.ndarray: 
        cache_key = f'klines'
        if cache_key not in self.__cache:
            timestamps_ms = (self.df.index.values.astype(np.int64) // 10**6).reshape(-1, 1)
            data_values = self.df.to_numpy()
            self.__cache[cache_key] = np.hstack([timestamps_ms, data_values])
        return self.__cache[cache_key]  

    @property
    def Open(self) -> np.ndarray:
        """Precios de apertura como array numpy."""
        return self.__get_array('Open')

    @property
    def High(self) -> np.ndarray:
        """Precios máximos como array numpy."""
        return self.__get_array('High')

    @property
    def Low(self) -> np.ndarray:
        """Precios mínimos como array numpy."""
        return self.__get_array('Low')

    @property
    def Close(self) -> np.ndarray:
        """Precios de cierre como array numpy."""
        return self.__get_array('Close')

    @property
    def Volume(self) -> np.ndarray:
        """Volúmenes como array numpy."""
        return self.__get_array('Volume')

    @property
    def Turnover(self) -> np.ndarray:
        """Volúmenes como array numpy."""
        return self.__get_array('Turnover')

    @property
    def index(self) -> pd.Index:
        """Índice temporal del DataFrame."""
        return self.__get_array('__index')

    @property
    def empty(self):
        """Devuelve True si no hay datos disponibles."""
        return self.__len == 0

    @property
    def nbars(self) -> np.ndarray:
        return self.__get_array('__nbars')

class Strategy:
    """ Clase base para estrategias de trading en Live""" 
    def __init__(self, sesh, params: dict = None, on_backtest=False, **kwargs): # , sesh: BybitSesh
        self.sesh = sesh 
        self.on_backtest = on_backtest

        self.__status = "waiting" # waiting , stopped, live
        self.__params = self.set_params(params) 
        self.__logger = None
        
        self._logs_path = None
        
        self.id_color = '#B5D6FF'

        [setattr(self, k, v) for k, v in kwargs.items()]
        # TODO: ordenar

    def __str__(self):
        # if self._params:
        params = ','.join(f'{i[0]}={i[1]}' for i in zip(self.__params.keys(),
                                                        map(_as_str, self.__params.values())))
        if params:
            params = '(' + params + ')'
        return f'{self.__class__.__name__}{params}'

    def __repr__(self):
        return '<Strategy ' + str(self) + '>'
        
    def __getstate__(self):
        """Prepara el objeto para ser pickled, excluyendo el logger"""
        state = self.__dict__.copy()
        # Excluir el logger ya que no es picklable
        if '_Strategy__logger' in state:
            del state['_Strategy__logger']
        return state

    def __setstate__(self, state):
        """Restaura el objeto después de ser unpickled"""
        self.__dict__.update(state)
        # Recrear el logger si es necesario
        self.__logger = None # Dejarlo vacio para recrearlo luego si se usa

    def _create_and_configure_logger(self):
    
        # Crear el direcorio de los logs
        logs_path_dir = (Path.cwd() / 'logs') if not self._logs_path else Path(self._logs_path)
        if not logs_path_dir.exists():
            logs_path_dir.mkdir(exist_ok=True)

        # definir el tipo de logger segun sea en modo backtest o live
        if self.on_backtest:
            logger = BacktestLogger(name=self.__class__.__name__, sesh=self.sesh)
        else:
            logger = logging.getLogger(self.__class__.__name__)

        #region LEVELS: Configuracion y Actualizacion de nuevos niveles
        _PERF_LEVEL = 5
        _SIGNAL_LEVEL = 15
        _TRADING_LEVEL = 25

        logging.addLevelName(_PERF_LEVEL, f'{'PERF'}')
        logging.addLevelName(logging.DEBUG, f'{'DEBUG'}')
        logging.addLevelName(_SIGNAL_LEVEL, f'{'SIGNAL'}')
        logging.addLevelName(logging.INFO, f'{'INFO'}')
        logging.addLevelName(_TRADING_LEVEL, f'{'TRADING'}')
        logging.addLevelName(logging.WARNING, f'{'WARNING'}')
        logging.addLevelName(logging.ERROR, f'{'ERROR'}')
        logging.addLevelName(logging.CRITICAL, f'{'CRITICAL'}')

        def perf(self, msg, *args, **kwargs):
            if self.isEnabledFor(_PERF_LEVEL):
                self._log(_PERF_LEVEL, msg, args, **kwargs)

        def signal(self, msg, *args, **kwargs):
            if self.isEnabledFor(_SIGNAL_LEVEL):
                self._log(_SIGNAL_LEVEL, msg, args, **kwargs)

        def trading(self, msg, *args, **kwargs):
            if self.isEnabledFor(_TRADING_LEVEL):
                self._log(_TRADING_LEVEL, msg, args, **kwargs)

        logging.Logger.perf = perf
        logging.Logger.signal = signal
        logging.Logger.trading = trading
        
        logger.setLevel(_PERF_LEVEL)
        #endregion

        # Crear los manejadores (destinos)
        stgy_name = self.__class__.__name__
        log_save_path = logs_path_dir / (
            f'backtest_log_{stgy_name}.log' if self.on_backtest 
                else f'live_log_{stgy_name}.log'
        )
  
        LOG_NAME_COLOR = HEX(self.id_color) # "#BFD8FF"
        SHADE, SAT = 0.35, 0

        theme = Theme(
            color_config=self._get_logger_color_config(LOG_NAME_COLOR, SHADE, SAT),
            fmt='{asctime}  {levelname}  {name} - {message}',
            style='{',
            dye=True
        )
        # Crear el formateador
        stream_handler = PintarStreamHandler(theme=theme) 
        file_handler = PintarFileHandler(log_save_path, theme=theme)
        
        # Establecer niveles de filtro para cada manejador
        stream_handler.setLevel(_PERF_LEVEL) 
        file_handler.setLevel(_PERF_LEVEL) 

        # Añadir los manejadores al logger
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

        return logger

    def _get_logger_color_config(self, log_name_color: HEX, shade, sat):
        color_config = {
            'PERF': {
                'name': {
                    'fore_color': log_name_color.shade(shade).saturate(sat), 
                    'style':1}
            }, 
            'DEBUG': {
                'name': {
                    'fore_color': log_name_color.shade(shade).saturate(sat), 
                    'style':1
                }
            }, 
            'SIGNAL': {
                'name': {
                    'fore_color': log_name_color, 
                    'style':1
                }
            },
            'INFO': {
                'name': {
                    'fore_color': log_name_color.shade(shade).saturate(sat), 
                    'style':1
                    }
                },
            'TRADING': {
                'asctime': {
                    'fore_color': '#D1D1D1', 
                    'style':1
                },
                'name': {
                    'fore_color': log_name_color, 
                    'style':1
                }, 
                'levelname': {
                    'fore_color':'#EAEFFF',
                    'style':1
                },
                'message': {
                    'fore_color': '#D1D1D1', 
                    'style':1
                },
            },
            'WARNING': {
                'name': {
                    'fore_color': log_name_color, 
                    'style':1
                }
            },
            'ERROR': {
                'name': {
                    'fore_color': log_name_color, 
                    'style':1
                }
            },
        }
        return color_config
    
    def set_params(self, params):
        if params is None: return {}

        for k, v in params.items():
            if not hasattr(self, k):
                raise AttributeError(
                    f"La estrategia '{self.__class__.__name__}' no tiene el parámetro '{k}'."
                    "La clase de la estrategia debe definir los parámetros como variables de clase antes de que "
                    "puedan ser optimizados o ejecutados.")
            setattr(self, k, v) 
        return params 

    def set_sesh(self, sesh):
        self.sesh = sesh
        return self

    def get_sesh(self) -> 'CryptoSesh': 
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

    def get_status(self):
        return self.__status

    def start(self):
        self.logger.warning(f"on_live \033[92;1m°\033[0m")
        self.__status = "live" 

    def stop(self):
        self.logger.warning(f"on_stop \033[91;1m°\033[0m")
        self.__status = "stopped" 

    def wait(self):
        self.logger.warning(f"waiting \033[93m°\033[0m")
        self.__status = "waiting"

    def beep(self, f=440, d=200, volume=0.02):
        """ 
        Reproduce un sonido de frecuencia f por d milisegundos.
        No se reproduce en Backtesting.
        Args:
            f (int, optional): Frecuencia en Hz. Defaults to 440.
            d (int, optional): Duracion en ms. Defaults to 200.
            volume (float, optional): Volumen del sonido. Defaults to 0.02.
        """
        if isinstance(f, int): f = [f]
        if isinstance(d, int): d = [d] 
        for f, d in zip(f, d):
            fs = 44100
            t = np.linspace(0, d / 1000, int(fs * d / 1000), endpoint=False)
            wave = np.sin(2 * np.pi * f * t) * volume
            sd.play(wave, fs, blocking=True)
            # sd.wait()

    # def update(self):
        # TODO: guardar reporte de la estrategia 
        # pass

    async def sleep(self, seconds: int):
        """ Pausa la ejecucion de la estrategia por un tiempo en segundos """
        await asyncio.sleep(seconds)

    async def init(self):
        """ """

    async def on_live(self):
        """ """

    async def on_stop(self):
        """ """

    async def I(self, 
        indicator_obj: Callable, 
        data: Union[DataOHLC, 'PivotIndicator'], 
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
        # indicator_data_timeframe = find_minutes_timeframe(data.index)
        indicator_data_minutes_timeframe = data.minutes

        # Crear el tag idetificador 
        str_timeframe = f"{minutes2timeframe(indicator_data_minutes_timeframe)}"
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
    def params(self):
        return self.__params

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def is_closed(self):
        return self.__status == "closed"

    @property
    def logger(self):
        if not self.__logger:
            self.__logger = self._create_and_configure_logger()
        return self.__logger

    @logger.setter
    def logger(self, logger):
        self.__logger = logger

class CryptoSesh(ABC):  
    """Sesión de trading para Bybit con funciones para operar"""

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
        
        # Crear dataframe con los datos de lista
        df = pd.DataFrame(klines,
            columns = ["datetime", "Open", "High", "Low", "Close", "Volume", "Turnover"] )

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
    
    @property
    async def total_equity(self):
        pass
    
    @property
    def time_zone(self):
        pass

    @property
    def equity(self):
        pass

    @property 
    def now(self) -> pd.Timestamp:
        raise NotImplementedError
    

    """ DATA """
    @abstractclassmethod
    async def get_kline(self, symbol: str, timeframe: str = "1D", start: str = None, 
        end: str = None, limit: int = None, category:"str" = "linear",
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
        pass
    
    @abstractclassmethod
    async def get_data(self, symbol, timeframe, start=None, end=None, 
        limit=None, category="linear", tz:str=None
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
        pass

    # @abstractclassmethod
    # async def _get_ticker(self, symbol) -> dict:
        # """
        #     Obtiene los detalles del ticker de un par de trading.

        #     Args:
        #         session: Sesión activa de la API
        #         category: Categoría del mercado (ej: 'linear','spot')
        #         symbol: Símbolo del par de trading

        #     Returns:
        #         dict: Detalles del ticker del par de trading
        # """
        # pass

    @abstractclassmethod
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
        pass

    @abstractclassmethod
    async def get_time(self, tz: str = None) -> pd.Timestamp:
        pass


    """ CONFIGURATION """
    @abstractclassmethod
    async def set_time_zone(self, tz: str):
        pass
    
    @abstractclassmethod
    async def set_leverage(self, symbol, leverage: int = 1):
        ""
        """
            Establece el nivel de margen (leverage) para un par de trading.
            Args:
                symbol: Símbolo del par de trading
                buy_leverage: Nivel de margen para operaciones de compra (opcional)
                sell_leverage: Nivel de margen para operaciones de venta (opcional)
        """
        pass
    
    @abstractclassmethod
    async def set_margin_mode(self, margin_mode: str = "insolated"):
        """ 
            Establece el modo de margen para la cuenta.
            Args:
                margin_mode: Modo de margen a establecer ('insolated'/'cross'/'portfolio')
            Returns:
                dict: Respuesta de la API con el resultado de la operación
        """
        pass

    """ INFO """
    @abstractclassmethod
    async def get_account_info(self):
        """
            Obtiene información de la cuenta de trading.
        """
        pass
    
    @abstractclassmethod
    async def get_all_orders(self, symbol) -> list:
        """
            Obtiene el historial de ordenes (en forma de diccionario) referente a un par de trading.
        """
        pass
    
    # @abstractclassmethod
    # async def _get_order(self, symbol, order_id, max_attempts=15, wait_time=0.5) -> dict:
        # """
        #     Obtiene los detalles de una orden (como diccionario) para un ID de orden dado usando dos métodos de búsqueda:
        #     1. Primero intenta encontrar la orden en las órdenes abiertas actualmente (más rápido)
        #     2. Si no se encuentra, busca en todo el historial de órdenes con múltiples intentos
            
        #     Args:
        #         session: Sesión activa de la API
        #         category: Categoría del mercado (ej: 'linear', 'spot')
        #         symbol: Símbolo del par de trading
        #         order_id: ID de la orden a buscar
        #         max_attempts: Número máximo de reintentos al buscar en el historial (default: 15)
        #         wait_time: Tiempo de espera entre reintentos en segundos (default: 0.5)
                
        #     Returns:
        #         dict: Detalles de la orden si se encuentra
                
        #     Raises:
        #         ValueError: Si la orden no se encuentra después de max_attempts
        # """
        # pass
    
    @abstractclassmethod
    async def get_leverage(self, symbol: str) -> int:
        """
            Obtiene el nivel de apalancamiento (leverage) para un par de trading.
            Args:
                symbol: Símbolo del par de trading
            Returns:
                float: Nivel de apalancamiento del par de trading, o 0.0 si no se encuentra posición.
        """
        pass

    """ Trading """
    
    # @abstractclassmethod
    # async def _place_order(self,
        #             symbol,
        #             side, 
        #             qty,  
        #             price=None, 
        #             sl_price=None, 
        #             tp_price=None, 
        #             pct_sl=None, 
        #             pct_tp=None, 
        #             time_in_force="GTC",
        #             ) -> dict:
        # """
        #     Coloca una orden de trading con las siguientes características:
            
        #     - Permite establecer órdenes de mercado, límite o stop
        #     - Los stop loss (SL) y take profit (TP) se especifican en precio absoluto
        #     - No acepta SL/TP en porcentajes directamente
        #     - Configurable para cualquier par de trading y categoría
        #     - El time in force por defecto es GTC (Good Till Cancel)
            
        #     Args:
        #         session: Sesión activa de trading
        #         category (str): Categoría del par (linear/inverse)
        #         symbol (str): Par de trading (ej: 'BTCUSDT')
        #         side (str): Dirección de la orden ('Buy'/'Sell') 
        #         order_type (str): Tipo de orden ('market'/'limit'/'stop')
        #         size (float): Tamaño de la posición
        #         price (float, optional): Precio límite para órdenes límite
        #         stop_price (float, optional): Precio de activación para órdenes stop
        #         sl (float, optional): Precio del stop loss
        #         tp (float, optional): Precio del take profit
        #         time_in_force (str): Validez de la orden (default: 'GTC')

        #     Returns:
        #         dict: Respuesta de la orden colocada
        # """ 
        # pass
    
    @abstractclassmethod
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
        pass
    
    @abstractclassmethod
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
        pass
    
    @abstractclassmethod
    async def close_position(self, symbol):
        pass
    
    @abstractclassmethod
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
        pass
    
    @abstractclassmethod
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
        pass

    @abstractclassmethod
    async def cancel_order(self, symbol, order_id) -> dict:
        """
            Cancela una orden abierta por su ID.

            Args:
                category (str): Categoría del símbolo (por ejemplo, "linear", "Spot").
                symbol (str): Símbolo para el que se desea cancelar la orden.
                order_id (str): ID de la orden que se desea cancelar.

            Returns:
                dict: Respuesta de la cancelación de la orden.
        """
        pass


