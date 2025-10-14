from tradero.lib import get_str_datetime_now
from pintar import dye, Stencil, Brush
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Tuple, Iterable, Union
import numpy as np
import pandas as pd
from numbers import Number
import requests
import logging
from pathlib import Path
import json
from pintar import dye, Brush, Stencil
from pintar.colors import RGB, HSL, HEX
from multiprocessing import shared_memory
from contextlib import contextmanager
SharedMemory = shared_memory.SharedMemory

def try_(lazy_func, default=None, exception=Exception):
    try:
        return lazy_func()
    except exception:
        return default


class ColorWayGenerator:
    """
    Generador de colores para gráficos y visualizaciones.

    Esta clase permite generar colores de forma secuencial a partir de paletas predefinidas.
    Cada vez que se llama a la instancia, devuelve el siguiente color de la paleta de forma cíclica.

    Paletas disponibles:
        - colorway_1: ['#24c1dd', '#ff59a7', '#e5ac4a', '#17ca7c', '#c270eb', '#d3bf10', '#554cde']
        - colorway_2: ['#554cde', '#d3bf10', '#c270eb', '#17ca7c', '#e5ac4a', '#ff59a7', '#24c1dd']

    Ejemplo de uso:
        >>> # Crear instancia con paleta por defecto (colorway_1)
        >>> color_generator = ColorWay()
        >>> first_color = color_generator()  # Retorna '#24c1dd'
        >>> second_color = color_generator() # Retorna '#ff59a7'
        
        >>> # Crear instancia con paleta específica
        >>> color_generator = ColorWay(way="colorway_2")
        >>> first_color = color_generator()  # Retorna '#554cde'
    """
    def __init__(self, way="colorway_1"):
        self.indice = 0
        self.colors_dict = {
            "colorway_1": ['#ff59a7', '#24c1dd', '#e5ac4a', '#17ca7c', '#c270eb', '#d3bf10', '#554cde'],
            "colorway_2": ['#554cde', '#d3bf10', '#c270eb', '#17ca7c', '#e5ac4a', '#ff59a7', '#24c1dd'],
            }
        self.colores = self.colors_dict[way]
    
    def __call__(self):
        color = self.colores[self.indice]
        self.indice = (self.indice + 1) % len(self.colores)
        return color


with open(Path(__file__).resolve().parent / 'config/log_theme_strategy_2.json') as f:
    CUSTOM_COLOR_THEME = json.load(f)

def dict_deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            dict_deep_update(d[k], v)
        else:
            d[k] = v
    return d

class BacktestLogger(logging.Logger):
    """ """
    def __init__(self, name, sesh):
        super().__init__(name)
        self._bt_sesh = sesh  # la estrategia que tiene el timestamp de la vela actual

    def makeRecord(self, *args, **kwargs):
        record = super().makeRecord(*args, **kwargs)
        if hasattr(self._bt_sesh, "now"):
            # Sobreescribir con el tiempo de la vela actual
            record.created = self._bt_sesh.now.timestamp()
        return record


def _as_str(value) -> str:
    if isinstance(value, (Number, str)):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return 'df'
    name = str(getattr(value, 'name', '') or '')
    if name in ('Open', 'High', 'Low', 'Close', 'Volume'):
        return name[:1]
    if callable(value):
        name = getattr(value, '__name__', value.__class__.__name__).replace('<lambda>', 'λ')
    if len(name) > 10:
        name = name[:9] + '…'
    return name


# XXX
def info_log(*args, **kwargs): # info_print
    time = get_str_datetime_now()
    first_c = "\033[93m[ info]\033[0m"
    print()
    print(f"  {time} {first_c}" + "\033[93;3m", *args, "\033[0m", **kwargs)
    print()
# XXX
def eprint(*args, **kwargs): # exception/error print
    first_c = "\033[91;1m[error]\033[0m"
    print(first_c + "\033[91;3m", *args, "\033[0m", **kwargs)

def execution_time_measure(func, iterations=100):
    start = time.time()
    for i in range(iterations):
        func()
    end = time.time()
    return (end - start) / iterations


def has_internet_connection(url="http://www.google.com/", timeout=5) -> bool:
    try:
        requests.get(url, timeout=timeout)
        return True
    except requests.RequestException:
        return False

def distribution_between_processes(
    obj: List[Any], 
    workers: int
) -> List[List[Any]]:
    """Distribuye estrategias en grupos consecutivos para procesamiento paralelo."""
    chunk_size = (len(obj) + workers - 1) // workers  # división redondeada hacia arriba
    return [
        obj[i * chunk_size:(i + 1) * chunk_size]
        for i in range(workers)
    ]

class SharedMemoryManager:
    """
        Gestor de múltiples memorias compartidas con limpieza automática.
        
        Uso:
            with SharedMemoryManager() as smm:
                metadata = smm.df_to_shared_memory(df)
                # ... usar metadata en otros procesos ...
            # Automáticamente cierra y elimina toda la memoria
    """
    
    # Constante para identificar el índice del DataFrame
    INDEX_COLUMN_NAME = '__dataframe_index'
    
    def __init__(self):
        """Inicializa el gestor sin memorias compartidas."""
        self._shared_memories: List[SharedMemory] = []
    
    def __enter__(self):
        """Entra al context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sale del context manager y limpia todas las memorias."""
        self._cleanup_all_memories()
        return False  # No suprimir excepciones
    
    def _cleanup_all_memories(self):
        for shm in self._shared_memories:
            try:
                shm.close()
                # Solo hacer unlink si esta memoria fue creada 
                if hasattr(shm, '_created_here') and shm._created_here:
                    shm.unlink()
            except Exception as e:
                warnings.warn(
                    f'No se pudo limpiar shared memory {shm.name!r}: {e}',
                    category=ResourceWarning,
                    stacklevel=2
                )
    
    def create_shm(self, size: int) -> SharedMemory:
        shm = SharedMemory(create=True, size=size)
        shm._created_here = True 
        self._shared_memories.append(shm)
        return shm
    
    def connect_to_shm(self, name: str) -> SharedMemory:
        shm = SharedMemory(name=name, create=False)
        shm._created_here = False  # No la creamos nosotros, no hacer unlink
        self._shared_memories.append(shm)
        return shm


    def arr2shm(self, values) -> Tuple[str, tuple, np.dtype]:
        assert values.ndim == 1, (values.ndim, values.shape, values)
        shm = self.create_shm(size=values.nbytes)
        # NumPy no puede manejar tz-aware directamente en shared memory
        # https://github.com/numpy/numpy/issues/18279
        buf = np.ndarray(values.shape, dtype=values.dtype.base, buffer=shm.buf)
        has_tz = getattr(values.dtype, 'tz', None)
        buf[:] = values.tz_localize(None) if has_tz else values  # Copy into shared memory
        return shm.name, values.shape, values.dtype

    def df2shm(self, df: pd.DataFrame) -> Tuple:
        columns_metadata = []
        
        # Quardar el indice (puede tener timezone)
        index_metadata = (
            self.INDEX_COLUMN_NAME,
            *self.arr2shm(df.index)
        )
        columns_metadata.append(index_metadata)
        
        # Quardar cada columna (pueden tener timezone)
        for column_name, values in df.items():
            column_metadata = (
                column_name,
                *self.arr2shm(values)
            )
            columns_metadata.append(column_metadata)
        
        return tuple(columns_metadata)

    def packet2shm(self, packet: Dict[str, 'DataOHLC']) -> Tuple[str, Tuple[List[SharedMemory]]]:
        shm_packet_data = []
        for symbol, dOHLC in packet.items():
            shm_packet_data.append((symbol, self.df2shm(dOHLC.df)))
        return shm_packet_data

    @staticmethod
    def shm2s(shm: SharedMemory, shape: tuple, dtype) -> pd.Series:
        # Usar dtype.base para leer datos (sin timezone)
        base_dtype = dtype.base if hasattr(dtype, 'base') else dtype
        
        # Crear vista del buffer
        array = np.ndarray(shape=shape, dtype=base_dtype, buffer=shm.buf)
        
        # Hacer read-only para evitar modificaciones accidentales
        array.setflags(write=False)
        
        # Crear Serie con el dtype original (que incluye timezone si existía)
        return pd.Series(array, dtype=dtype)

    @staticmethod
    def shm2df(shm_columns_data: Tuple) -> Tuple[pd.DataFrame, List[SharedMemory]]:
        # Conectar a todas las shared memories
        shared_memories = [
            SharedMemory(name=shm_name, create=False)
            for _, shm_name, _, _ in shm_columns_data
        ]
        
        # Reconstruir cada columna
        columns_data = {}
        for shm, (col_name, _, shape, dtype) in zip(shared_memories, shm_columns_data):
            columns_data[col_name] = SharedMemoryManager.shm2s(
                shm, shape, dtype
            )
        
        # Crear DataFrame
        df = pd.DataFrame(columns_data)
        
        # Restaurar el índice
        df.set_index(SharedMemoryManager.INDEX_COLUMN_NAME, drop=True, inplace=True)
        df.index.name = None
        
        return df, shared_memories

    @staticmethod
    def shm2packet(shm_packet_data: Tuple) -> Tuple[Dict[str, 'DataOHLC'], List[SharedMemory]]:
        from tradero.models import DataOHLC
        packet = {}
        all_shared_memories = []
        for symbol, shm_df in shm_packet_data:
            df, shared_memories = SharedMemoryManager.shm2df(shm_df)
            df.index.name = 'datetime'
            df.reset_index(inplace=True)
            content = df.to_dict(orient='list')
            content['datetime'] = np.array(content['datetime'], dtype='datetime64[ms]')
            packet[symbol] = DataOHLC(content)
            all_shared_memories.extend(shared_memories)
        return packet, all_shared_memories


@contextmanager
def patch(obj, attr, new_value):
    had_attr = hasattr(obj, attr)
    orig_value = getattr(obj, attr, None)
    setattr(obj, attr, new_value)
    try:
        yield
    finally:
        if had_attr:
            setattr(obj, attr, orig_value)
        else:
            delattr(obj, attr)



