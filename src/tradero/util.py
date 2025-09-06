from typing import Union
import pandas as pd
import numpy as np
import time

def read_csv_ohlc(rute) -> 'DataOHLC':
    "formato nobre del symbolo en la ruta: 'ADAUSDT_1min_2025_JUNE.csv' "
    from .models import DataOHLC
    str_objects = rute.split('\\')[-1].split('_')
    symbol_name = str_objects[0]
    timeframe = str_objects[1]
    content = pd.read_csv(rute, parse_dates=True).to_dict(orient='list')
    content["datetime"] = np.array(content["datetime"], dtype='datetime64[ms]')
    return DataOHLC(content=content, timeframe=timeframe, symbol=symbol_name)

def try_(func, fallback):
    """Intenta ejecutar func(). Si falla, devuelve fallback."""
    try:
        return func()  
    except Exception:
        return fallback  

def timeframe2minutes(timeframe: str) -> int:
    """
        Convierte un timeframe string a minutos.
        
        Args:
            timeframe: timeframe como string (ej: '1s', '5min', '1h', '1D', '1W', '1ME', '1YE')
        
        Returns:
            Número de minutos
    """
    # Manejar casos especiales
    if timeframe == "unknown":
        return 1  # Asumir 1 minuto por defecto
    
    # Casos para segundos
    if timeframe.endswith("s"):
        number = int(timeframe[:-1])
        return number / 60
    
    # Casos para minutos
    if timeframe.endswith("min"):
        number = int(timeframe[:-3])
        return number
    
    # Casos para horas
    if timeframe.endswith("h"):
        number = int(timeframe[:-1])
        return number * 60
    
    # Casos para días
    if timeframe.endswith("D"):
        number = int(timeframe[:-1])
        return number * 1440
    
    # Casos para semanas
    if timeframe.endswith("W"):
        number = int(timeframe[:-1])
        return number * 10080
    
    # Casos para meses
    if timeframe.endswith("ME"):
        number = int(timeframe[:-2])
        return number * 43200  # Aproximadamente 30 días
    
    # Casos para años
    if timeframe.endswith("YE"):
        number = int(timeframe[:-2])
        return number * 525600  # Aproximadamente 365 días
    
    # Si no coincide con ninguno de los formatos anteriores
    raise ValueError(f"Formato de timeframe inválido: {timeframe}")

def minutes2timeframe(minutes: int|float) -> str:
    """
        Convierte un entero a timeframe string de pandas automáticamente.

        Reglas:
        - < 60  -> minutos
        - múltiplo de 60 -> horas
        - múltiplo de 1440 -> días
        - múltiplo de 10080 -> semanas
        - múltiplo de 43200 -> meses (30 días aprox)
        - múltiplo de 525600 -> años (365 días)
    """
    if minutes < 1:
        return f"{int(60*minutes)}S"
    elif minutes < 60:
        return f"{minutes}min"
    elif minutes % 60 == 0 and minutes < 1440:
        return f"{minutes // 60}h"
    elif minutes % 1440 == 0 and minutes < 10080:
        return f"{minutes // 1440}D"
    elif minutes % 10080 == 0 and minutes < 43200:
        return f"{minutes // 10080}W"
    elif minutes % 43200 == 0 and minutes < 525600:
        return f"{minutes // 43200}ME"
    elif minutes % 525600 == 0:
        return f"{minutes // 525600}YE"
    else:
        raise ValueError(f"No se puede convertir automáticamente el valor {minutes} a timeframe válido")

# def find_minutes_timeframe(index: np.ndarray[np.datetime64]) -> int:
#     """
#         Calcula la diferencia de tiempo entre los dos primeros índices de un DataFrame.

#         Parámetros:
#         - data (pd.DataFrame): DataFrame con un índice de tipo datetime.

#         Retorna:
#         - int: Diferencia de tiempo en minutos entre los dos primeros registros.
#         - None: Si el DataFrame tiene menos de dos registros.

#         Ejemplo de uso:
#         >>> df = pd.DataFrame(index=pd.to_datetime(["2025-02-07 12:00:00", "2025-02-07 12:05:00"]))
#         >>> find_timeframe(df)
#     """
#     if len(index) < 2:
#         return None  # Retorna None si hay menos de dos registros   
    
#     first_time = index[1].astype("M8[ms]").astype("O")
#     second_time = index[0].astype("M8[ms]").astype("O")
    
#     return (first_time - second_time).seconds // 60  # Convertir segundos a minutos  # Retorna un entero con los minutos completos

def find_minutes_timeframe(index: Union[pd.DatetimeIndex, np.ndarray]) -> int:
    """
        Calcula la diferencia de tiempo entre los dos primeros índices de un DataFrame.

        Parámetros:
        - data (pd.DataFrame): DataFrame con un índice de tipo datetime.

        Retorna:
        - int: Diferencia de tiempo en minutos entre los dos primeros registros.
        - None: Si el DataFrame tiene menos de dos registros.

        Ejemplo de uso:
        >>> df = pd.DataFrame(index=pd.to_datetime(["2025-02-07 12:00:00", "2025-02-07 12:05:00"]))
        >>> find_timeframe(df)
    """
    if len(index) < 2:
        return None  # Retorna None si hay menos de dos registros   
    
    if isinstance(index, np.ndarray):
        index = pd.DatetimeIndex(index)

    first_time = index[1]
    second_time = index[0]
    
    return (first_time - second_time).total_seconds() // 60  # Convertir segundos a minutos  # Retorna un entero con los minutos completos

def npdt64_to_datetime(np_dt64):
    """Convierte numpy.datetime64 a datetime de Python"""
    return np_dt64.astype("M8[ms]").astype("O")

def execution_time_measure(func, iterations=100):
    start = time.time()
    for i in range(iterations):
        func()
    end = time.time()
    return (end - start) / iterations

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
        
