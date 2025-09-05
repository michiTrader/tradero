from ..models import DataOHLC
import pandas as pd
import numpy as np
from numba import jit
from typing import Sequence, Union, Iterable
from inspect import isclass
from numbers import Number
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import Span, HoverTool, Legend, LegendItem, ColumnDataSource
from bokeh.layouts import column, gridplot
from abc import abstractclassmethod, ABC
import re

# Indicators

"""Analisis"""
def n_pct_change(data: np.ndarray, periods=1) -> np.ndarray:
    """
    
    Calcula el porcentaje de cambio de un array utilizando Numba para mayor velocidad.
    """
    result = np.full_like(data, 0.0, dtype=np.float64)
    for i in range(periods, len(data)):
        result[i] = (data[i] - data[i - periods]) / data[i - periods]
    return result
  
def pct_change_from_time(data, index=None, time_ago=None) -> pd.Series:
    """
        Calcula el cambio porcentual en el precio durante un período de tiempo específico para cada punto de datos,
        utilizando operaciones vectorizadas para un mejor rendimiento.

        Parámetros:
        -----------
        data : Union[pd.Series, np.ndarray, pd.DataFrame]
            Datos de precio a analizar. Puede ser una Serie de pandas, array de numpy o DataFrame.
        index : Union[pd.DatetimeIndex, np.ndarray, list], opcional
            Índice temporal correspondiente a los datos. Requerido si data es un array de numpy.
            Si data es una Serie o DataFrame con índice datetime, este parámetro se ignora.
        time_ago : str, opcional
            Intervalo de tiempo para calcular la variación. Ejemplos: '24h', '48h', '5h', '30m'.
            Formato: número seguido de 'h' (horas), 'm' (minutos) o 'd' (días).
            Si es None, se calculará la variación en las últimas 24 horas.

        Retorna:
        --------
        pd.Series
            Una Serie de pandas con el cambio porcentual en el precio para cada punto de datos,
            basado en el valor del 'time_ago' especificado. Los valores que no pueden ser calculados
            (por falta de datos previos) serán NaN.

        Ejemplo:
        --------
        >>> # Datos de ejemplo
        >>> dates = pd.to_datetime(['2023-01-01 00:00', '2023-01-01 01:00', '2023-01-01 02:00',
        ...                       '2023-01-01 03:00', '2023-01-01 04:00', '2023-01-01 05:00'])
        >>> prices = pd.Series([100, 102, 101, 105, 103, 106], index=dates)

        >>> # Calcular el cambio porcentual cada 2 horas
        >>> pct_change_2h = pct_change_from_time(prices, time_ago='2h')
        >>> print(pct_change_2h)
        2023-01-01 00:00:00          NaN
        2023-01-01 01:00:00          NaN
        2023-01-01 02:00:00     1.000000
        2023-01-01 03:00:00     2.941176
        2023-01-01 04:00:00     1.980198
        2023-01-01 05:00:00     0.961538
        dtype: float64
    """
    # Convertir los datos a una Serie si es un array o DataFrame
    if isinstance(data, np.ndarray):
        if index is None:
            raise ValueError("Se requiere un índice cuando los datos son un array de numpy")
        data = pd.Series(data, index=index)
    elif isinstance(data, pd.DataFrame):
        # Intentar seleccionar la columna 'Close', si no existe, usar la primera columna o squeeze
        if 'Close' in data.columns:
            data = data['Close']
        else:
            data = data.squeeze()
            if isinstance(data, pd.DataFrame): # Si squeeze aún devuelve un DataFrame (más de 1 columna)
                raise ValueError("El DataFrame tiene múltiples columnas y no se encontró 'Close'. Por favor, especifique la columna a usar o asegúrese de que el DataFrame tenga solo una columna.")

    # Verificar que el índice sea de tipo DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("El índice debe ser de tipo DatetimeIndex")

    # Determinar el Timedelta a usar
    if time_ago is not None:
        match = re.match(r'(\d+)([hmd])', time_ago)
        if not match:
            raise ValueError("Formato de tiempo inválido. Debe ser un número seguido de 'h', 'm', o 'd'")

        amount = int(match.group(1))
        unit = match.group(2)

        if unit == 'h':
            delta = pd.Timedelta(hours=amount)
        elif unit == 'm':
            delta = pd.Timedelta(minutes=amount)
        elif unit == 'd':
            delta = pd.Timedelta(days=amount)
    else:
        delta = pd.Timedelta(hours=24)

    # --- Lógica vectorizada para un rendimiento mucho mejor ---
    # Calcular el tiempo de referencia para cada punto de datos
    time_ago_series = data.index - delta

    # Usar `reindex` con `method='bfill'` para encontrar el valor en `time_ago_series`
    # Esto busca el siguiente valor válido si no hay uno exactamente en el tiempo de referencia.
    # Alternativamente, si quisiéramos el valor ANTERIOR o EXACTO, usaríamos `ffill` o sin método
    # Sin embargo, para 'el precio hace X tiempo', necesitamos el precio que estaba en ese momento.
    # La forma más robusta y eficiente es usar `pd.merge_asof` o `searchsorted` con el índice.

    # Paso 1: Crear un DataFrame temporal para facilitar el merge_asof
    # Renombramos el índice de `data` para que sea una columna con la que podamos hacer merge.
    df_temp = data.to_frame(name='current_price')
    df_temp['lookup_time'] = data.index - delta

    # Paso 2: Usar merge_asof para encontrar el precio correspondiente `delta` tiempo atrás.
    # `direction='forward'` busca el primer valor en `right` que es mayor o igual que `left_on`.
    # `direction='backward'` busca el último valor en `right` que es menor o igual que `left_on`.
    # `direction='nearest'` busca el más cercano.
    # Para "hace X tiempo", queremos el valor en o justo antes de `lookup_time`, por lo que 'backward' es apropiado.
    # Asegúrate de que ambos DataFrames estén ordenados por el tiempo.
    df_temp_sorted = df_temp.sort_index()

    # El DataFrame 'right' es la propia 'data' pero como un DataFrame con una columna de valor y su índice.
    # Esto permite buscar los precios 'antiguos'.
    df_prices_for_merge = data.to_frame(name='past_price')

    merged_df = pd.merge_asof(
        df_temp_sorted,             # left DataFrame: contiene el 'lookup_time' y el 'current_price'
        df_prices_for_merge,        # right DataFrame: contiene todos los precios posibles para el 'past_price'
        left_on='lookup_time',      # Columna en 'left' para la búsqueda
        right_index=True,           # Usar el índice de 'right' (df_prices_for_merge) para la búsqueda
        direction='backward'        # Encuentra el último valor en 'right_index' menor o igual a 'left_on'
    )

    # Reindexar el DataFrame resultante para que coincida con el orden original de los datos de entrada
    merged_df = merged_df.set_index(df_temp_sorted.index)
    merged_df = merged_df.reindex(data.index) # Asegura el orden original

    # Calcular el cambio porcentual de forma vectorizada
    # Asegúrate de manejar la división por cero: si 'past_price' es 0, el resultado es NaN.
    pct_changes = ((merged_df['current_price'] / merged_df['past_price']) - 1) * 100
    pct_changes = pct_changes.replace([np.inf, -np.inf], np.nan) # Reemplazar inf con NaN si hay división por cero

    # Los valores donde no se encontró un 'past_price' (debido a que 'lookup_time' es anterior al inicio de los datos)
    # ya serán NaN debido al funcionamiento de merge_asof y reindex.
    return pct_changes

def binance_get_klines(symbol, interval, start=None, end=None, api_key=None, api_secret=None, save_rute=None, timezone='UTC-00:00'):
    """
        Descarga datos históricos de Binance y los guarda en un archivo CSV.
        ejemplo:
            binance_get_klines("BTCUSDT", interval="1m", start="1 Jan, 2023", end="1 Jan, 2024",
                            api_key="tu_api_key", api_secret="tu_api_secret", save_rute=r"C:/ruta/a/guardar")
    """
    symbol, interval = symbol.upper(), interval.lower()
    try:
        binance_client = Client(api_key, api_secret)
    except:
        print("Error. No se puedo iniciar en el cliente de binacne")

    intervals = {"1m":Client.KLINE_INTERVAL_1MINUTE,
                 "3m":Client.KLINE_INTERVAL_3MINUTE,
                 "5m":Client.KLINE_INTERVAL_5MINUTE,
                 "15m":Client.KLINE_INTERVAL_15MINUTE,
                 "30m":Client.KLINE_INTERVAL_30MINUTE,
                 "1h":Client.KLINE_INTERVAL_1HOUR,
                 "4h":Client.KLINE_INTERVAL_4HOUR,
                 "1d":Client.KLINE_INTERVAL_1DAY,
                 }

    try:
        data = binance_client.get_historical_klines(symbol, intervals[interval], start, end)
    except:
        raise ValueError("Error. No se pudo obtener los datos de binance")

    df = pd.DataFrame(data)[[0,1,2,3,4,5]]
    df.columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # Setear indice datetime
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index("datetime", inplace=True)
    if df.index.tz is None:  # Si el índice es naive
        df.index = df.index.tz_localize("UTC").tz_convert(timezone).tz_localize(None)
    else:  # Si ya tiene zona horaria
        df.index = df.index.tz_convert(timezone).tz_localize(None)
    


    # Convertir todas las columnas a tipo numérico
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if save_rute:
        start_date = convert_date(start) #01jan23
        end_date = convert_date(end) #01jan24
        nombre = f'{symbol}_{interval}_{start_date}-{end_date}_binance_DATA'
        ruta = save_rute + r"\%s.csv" % nombre #r'C:\Users\dimne\OneDrive\Documentos\__Projects__\%s.csv'

        # Guardar el DataFrame como CSV
        df.to_csv(ruta, index=False)  
    


    return df



"""Herramientas"""
def n_shift(arr, shift=1, fill_value=np.nan):
    """ 
    Desplaza los elementos de un array hacia adelante o hacia atrás.
    Habilitada para @njit
    """
    # Crear un array de resultados con el mismo tamaño que arr
    result = np.zeros_like(arr, dtype=np.float64)
    
    # Llenar el array con el valor de relleno
    for i in range(len(result)):
        result[i] = fill_value
    
    # Realizar el desplazamiento
    if shift > 0:
        for i in range(shift, len(arr)):
            result[i] = arr[i - shift]
    elif shift < 0:
        for i in range(0, len(arr) + shift):
            result[i] = arr[i - shift]
    else:
        for i in range(len(arr)):
            result[i] = arr[i]
            
    return result

def n_ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(len(mask)), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]

def color_way(quantity: int, color_way=None) -> list:
    """
        Genera una lista de colores repetidos hasta alcanzar la cantidad deseada.

        Parámetros:
        - quantity (int): Número total de colores requeridos.
        - color_way (str | list, opcional): Puede ser el nombre de un esquema de color 
        predefinido en `colors_dict` o una lista personalizada de colores. 
        Si no se proporciona, se usa "colorway 1" por defecto.

        Retorna:
        - list: Lista de colores con la longitud especificada.

        Ejemplo:
        >>> color_way(7, "colorway 2")
        ['#554cde', '#d3bf10', '#c270eb', '#17ca7c', '#e5ac4a', '#ff59a7', '#24c1dd']

        >>> color_way(5, ['#ff0000', '#00ff00', '#0000ff'])
        ['#ff0000', '#00ff00', '#0000ff', '#ff0000', '#00ff00']
    """
    
    colors_dict = {
        "colorway 1": ['#24c1dd', '#ff59a7', '#e5ac4a', '#17ca7c', '#c270eb', '#d3bf10', '#554cde'],
        "colorway 2": ['#554cde', '#d3bf10', '#c270eb', '#17ca7c', '#e5ac4a', '#ff59a7', '#24c1dd'],
        }

    # Si no se especifica un esquema de color, usar "colorway 1" por defecto
    if color_way is None:
        color_way = "colorway 1"

    # Si color_way es una lista, usarla directamente; si es un string, buscar en el diccionario
    colors = color_way if isinstance(color_way, list) else colors_dict.get(color_way, colors_dict["colorway 1"])

    # Asegurar que la lista tiene al menos la cantidad requerida
    return (colors * (quantity // len(colors) + 1))[:quantity]

class evaluator:
    """Clase para evaluar rapidamente indicadores."""

    @staticmethod
    def cross_over_method(arr1, arr2):
        """ 
            Evalúa la estrategia de trading por cruce. Genera compra (1) cuando arr1 cruza por encima de arr2,
            venta (-1) cuando cruza por debajo, y 0 en otros casos. Retorna un array con los rendimientos calculados.

            Args:
                arr1 (array-like): Primer valor a comparar
                arr2 (array-like): Segundo valor contra el que comparar

            Returns:
                numpy.ndarray: Rendimientos de las señales de cruce

            Nota: Los arrays deben tener la misma longitud.
        """
        assert len(arr1) == len(arr2), "Los tamaños de los arrays no coinciden."
        
        arr1 = np.nan_to_num(arr1)
        arr2 = np.nan_to_num(arr2)

        position = np.where(arr1 > arr2, 1, -1)

        arr_pct_change = n_pct_change(arr1)
        returns = arr_pct_change * n_shift(position, 1)
        returns[0] = 0

        return returns

    @staticmethod
    def on_trigger_method(arr1, arr2, arr3):
        """
        Evalúa la estrategia de trading basada en niveles de disparo.
        Retorna un array de rendimientos calculados a partir de cambios de precio y posiciones.

        Args:
            arr1: Array de datos de precio
            arr2: Array del nivel de disparo superior
            arr3: Array del nivel de disparo inferior

        Returns:
            numpy.ndarray: Array de rendimientos de la estrategia
        """
        assert "arr2" in kwargs and "arr3" in kwargs, "Se deben proporcionar los parametros arr2 y arr3"
        assert len(arr1) == len(arr2) == len(arr3), "Los tamaños de los arrays no coinciden."  
        arr1 = np.nan_to_num(arr1)            
        position = np.where(arr1 >= arr2, 1, np.where(arr1 <= arr3, -1, 0))

        arr_pct_change = n_pct_change(arr1)
        returns = arr_pct_change * n_shift(position, 1)
        returns[0] = 0

        return returns  

    @staticmethod
    def on_trigger_exit_method(arr1, arr2, arr3):
        assert "arr2" in kwargs and "arr3" in kwargs, "Se deben proporcionar los parametros arr2 y arr3"
        assert len(arr1) == len(arr2) == len(arr3), "Los tamaños de los arrays no coinciden."  
        position = np.full(len(arr1), np.nan) #np.zeros(len(arr))

        short_wait = False
        long_wait = False

        in_top = arr1 >= arr2
        in_bot = arr1 <= arr3
        in_mid = ~(in_top | in_bot)  

        for i in range(len(arr1)):
            if in_top[i]:  # Si estamos arriba
                short_wait = True
                long_wait = False
            elif in_bot[i]:  # Si estamos abajo
                long_wait = True
                short_wait = False
            elif in_mid[i]:  # Si estamos en medio
                if short_wait:
                    position[i] = -1
                elif long_wait:
                    position[i] = 1
                
                short_wait = False
                long_wait = False

        # hacer ffill a las posiciones
        position = n_ffill(position)

        arr_pct_change = n_pct_change(arr1) 
        returns = arr_pct_change * n_shift(position, 1)
        returns[0] = 0

        print(arr_pct_change)
        return returns

def optimize(indicator, data, method, **kwargs) -> pd.Series:
    """
        Optimiza los parámetros de un indicador técnico evaluando diferentes valores.

        Parámetros:
        -----------
        indicator : callable
            Función del indicador técnico a optimizar
        data : array-like
            Datos de precio sobre los que optimizar
        method : str
            Método de evaluación a utilizar ('cross_over', 'on_trigger', 'on_trigger_exit')
        **kwargs : dict
            Parámetros del indicador y sus rangos de valores a probar

        Retorna:
        --------
        pd.Series
            Serie con los mejores y peores valores encontrados para cada parámetro

        Ejemplo:
        --------
        >>> optimize(SMA, price_data, "cross_over", length=range(10,100,10))
    """
    func_methods = {
        "cross_over":evaluator.cross_over_method,
        "on_trigger":evaluator.on_trigger_method,
        "on_trigger_exit":evaluator.on_trigger_exit_method,
    }
    evaluation_method = func_methods[method]

    result_of_keys = {}
    for key, values in kwargs.items():
        resturns_for_values = {}
        for val in values:
            # Calcular el indicador
            indicator_values = indicator(data=data, **{f"{key}":val}).values
            if indicator_values.ndim > 1:
                # Evaluar el indicador con suma acumulada y guardar
                evaluation_returns = evaluation_method(arr1=data, arr2=indicator_values[0], arr3=indicator_values[-1]).cumsum()[-1]
                resturns_for_values[val] = evaluation_returns
            else:
                # Evaluar el indicador con suma acumulada y guardar
                evaluation_returns = evaluation_method(arr1=data, arr2=indicator_values).cumsum()[-1]
                resturns_for_values[val] = evaluation_returns     

        result_of_keys[key] = resturns_for_values

    final_results = {}
    for key, values_and_returns in result_of_keys.items():
        best_value = int(max(values_and_returns, key=values_and_returns.get))
        best_returns = values_and_returns.get(best_value)

        worst_value = int(min(values_and_returns, key=values_and_returns.get))
        worst_returns = values_and_returns.get(worst_value)
        
        final_results[key] = pd.Series(dict(
            best_value=best_value,
            best_returns=best_returns,
            worst_value=worst_value,
            worst_returns=worst_returns,
        ))

    # Ordenar los resultados por valor de evaluación
    return pd.Series(final_results)



""" OPTIMIZE """
def optimize_take_profit(mfe, returns, range_step=0.01):
    """
    Optimiza el nivel de **Take Profit (TP)** para maximizar el balance acumulado en un conjunto de operaciones,
    evaluando diferentes niveles de TP. Si no se encuentra un nivel mejor que el actual, retorna los valores originales.
    
    Parámetros:
    -----------
    mfe : pandas.Series
        Serie de datos que contiene el *Maximum Favorable Excursion* (MFE), la máxima ganancia no realizada durante cada operación.
    
    returns : pandas.Series
        Serie de datos que contiene los retornos de cada operación.
    
    range_step : float, opcional
        Paso para generar los niveles de TP en términos de porcentaje del máximo valor de MFE. 
        Por defecto, es 0.01 (1%).

    Retorna:
    --------
    dict
        - `tp` : float
            El valor óptimo de TP encontrado que maximiza el balance acumulado.
        - `returns` : float
            El balance acumulado máximo correspondiente al TP óptimo.
        - `returns_series` : pandas.Series
            La serie de balance acumulado que corresponde al TP óptimo.
        
        Si no se encuentra un nivel de TP que sea mejor que el original, retorna:
        - `tp` : float
            El valor original de TP basado en el máximo de MFE.
        - `returns` : float
            El balance acumulado original.
        - `returns_series` : pandas.Series
            La serie de balance acumulado original.
    """
    
    mfe_series = mfe.copy()
    return_series = returns.copy()

    # Convertir la serie MFE a valores absolutos
    mfe_series = abs(mfe_series)

    # Calcular el balance acumulado original
    original_balance_Series = return_series.cumsum()
    original_balance = original_balance_Series.iloc[-1]
    original_x = max(mfe_series)

    # Definir el rango de valores para el TP y establecer el paso
    max_data_value = original_x
    step = range_step
    range_of_x = np.arange(0, max_data_value + (step * max_data_value), step * max_data_value)

    max_balance = -float('inf')  # Inicializar el balance máximo
    optimal_tp = None            # Valor óptimo de TP
    optimal_balance_series = None  # Serie de balance para el TP óptimo

    # Optimizar el nivel de TP
    for X in range_of_x:
        # Calcular los valores de TP (ganancias si el MFE es menor que el TP, sino TP fijo)
        Tx = np.where(mfe_series <= X, return_series, X)

        # Calcular el balance y balance acumulado
        balance_series = pd.Series(Tx)
        final_balance = balance_series.cumsum().iloc[-1]

        # Actualizar si encontramos un balance mejor
        if final_balance > max_balance:
            max_balance = final_balance
            optimal_tp = X
            optimal_balance_series = balance_series

    # Si encontramos un nivel de TP que sea mejor que el original, lo devolvemos
    if max_balance > original_balance:
        result = {"tp": optimal_tp, "returns": max_balance, "returns_series": optimal_balance_series}
    else:
        result = {"tp": original_x, "returns": original_balance, "returns_series": returns}
    
    return result  

def optimize_stop_loss(mae, returns, range_step=0.01):
    """
    Optimiza el nivel de **Stop Loss (SL)** para maximizar el balance acumulado en un conjunto de operaciones,
    evaluando diferentes niveles de SL. Si no se encuentra un nivel mejor que el actual, retorna los valores originales.
    
    Parámetros:
    -----------
        mae : pandas.Series
            Serie de datos que contiene el *Maximum Adverse Excursion* (MAE), la máxima pérdida no realizada durante cada operación
        returns : pandas.Series
            Serie de datos que contiene los retornos de cada operación.
    
    range_step : float, opcional
        Paso para generar los niveles de SL en términos de porcentaje del máximo valor de MAE. 
        Por defecto, es 0.01 (1%).

    Retorna:
    --------
    dict
        - `sl` : float
            El valor óptimo de SL encontrado que maximiza el balance acumulado.
        - `returns` : float
            El balance acumulado máximo correspondiente al SL óptimo.
        - `returns_series` : pandas.Series
            La serie de balance acumulado que corresponde al SL óptimo.
        
        Si no se encuentra un nivel de SL que sea mejor que el original, retorna:
        - `sl` : float
            El valor original de SL basado en el máximo de MAE.
        - `returns` : float
            El balance acumulado original.
        - `returns_series` : pandas.Series
            La serie de balance acumulado original.
    """
    
    mae_series = mae.copy()
    return_series = returns.copy()

    # Convertir la serie MAE a valores absolutos
    mae_series = abs(mae_series)

    # Calcular el balance acumulado original
    original_balance_Series = return_series.cumsum()
    original_balance = original_balance_Series.iloc[-1]
    original_x = max(mae_series)

    # Definir el rango de valores para el SL y establecer el paso
    max_data_value = original_x
    step = range_step
    range_of_x = np.arange(0, max_data_value + (step * max_data_value), step * max_data_value)

    max_balance = -float('inf')  # Inicializar el balance máximo
    optimal_sl = None            # Valor óptimo de SL
    optimal_balance_series = None  # Serie de balance para el SL óptimo

    # Optimizar el nivel de SL
    for X in range_of_x:
        # Calcular los valores de SL (retornos si el MAE es menor que el SL, sino -SL)
        Tx = np.where(mae_series <= X, return_series, -X)

        # Calcular el balance y balance acumulado
        balance_series = pd.Series(Tx)
        final_balance = balance_series.cumsum().iloc[-1]

        # Actualizar si encontramos un balance mejor
        if final_balance > max_balance:
            max_balance = final_balance
            optimal_sl = X
            optimal_balance_series = balance_series

    # Si encontramos un nivel de SL que sea mejor que el original, lo devolvemos
    if max_balance > original_balance:
        result = {"sl": optimal_sl, "returns": max_balance, "returns_series": optimal_balance_series}
    else:
        result = {"sl": original_x, "returns": original_balance, "returns_series": returns}
    
    return result


class plot:
    def _find_timeframe(data: pd.DataFrame) -> int:
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
        if len(data) < 2:
            return None  # Retorna None si hay menos de dos registros   
        
        first_time = data.index[1]
        second_time = data.index[0]
        
        result = (first_time - second_time).total_seconds() / 60  # Convertir segundos a minutos
        return int(result)  # Retorna un entero con los minutos completos

    @classmethod
    def plot_ohlc(cls, data, height=400, width=800, output="notebook", **kwargs):

        color_way = ColorWay()

        df = data.copy()
        
        df.index.name = "Datetime"

        # Asegurar el formato correcto de fecha si de un dia o mas
        if cls._find_timeframe(df) >= 1440:
            df.index = df.index.normalize().tz_localize("UTC")

        df.columns = [c.capitalize() for c in df.columns]

        df["Date"] = df.index

        df = df[["Open", "High", "Low", "Close", "Date"]]
        
        # Separar datos alcistas y bajistas
        inc = df['Close'] > df['Open']
        dec = df['Open'] > df['Close']
        
        # Convertir a ColumnDataSource para Bokeh
        source_inc = ColumnDataSource(df.loc[inc]) # velas alcista
        source_dec = ColumnDataSource(df.loc[dec]) # velas bajista
        source = ColumnDataSource(df)
        source.add((df.Close >= df.Open).values.astype(np.uint8).astype(str), 'inc')
        
        # Ajustar el ancho de las velas para datos de 5 minutos (5 minutos * 60 segundos * 1000 ms)
        mintimeframe = cls._find_timeframe(df)
        separation = 0.10
        w = (mintimeframe - (separation * mintimeframe)) * 60 * 1000 

        fig = figure(
            x_axis_type="datetime",
            title="Gráfico de Velas - Bokeh",
            width=width,
            height=height,
            background_fill_color="#e0e6eb",  # Fondo interior
            border_fill_color="#e0e6eb",     # Fondo exterior
        )
        if True:
            fig.xaxis.axis_label = 'Fecha'
            fig.yaxis.axis_label = 'Precio'
            fig.title.text_color = "#333333"
            fig.xaxis.axis_label_text_color = "#333333"
            fig.yaxis.axis_label_text_color = "#333333"
            fig.xaxis.major_label_text_color = "#333333"
            fig.yaxis.major_label_text_color = "#333333"

            # Evitar mostrar en connotacion cientifica los numeros del eje y
            fig.yaxis.formatter.use_scientific = False

            # Cambiar el color de las líneas de los ejes y ticks a azul
            fig.xaxis.axis_line_color = "#e0e6eb"
            fig.yaxis.axis_line_color = "#e0e6eb"
            fig.xaxis.major_tick_line_color = "#a5a5a5"
            fig.yaxis.major_tick_line_color = "#c7c7c7"
            fig.xaxis.minor_tick_line_color = "#c7c7c7"
            fig.yaxis.minor_tick_line_color = "#c7c7c7"

            # Cambiar el color del grid
            fig.xgrid.grid_line_color = "#c7c7c7"
            fig.ygrid.grid_line_color = "#c7c7c7"
            fig.xgrid.grid_line_alpha = 0.5  # Opcional: ajustar la transparencia
            fig.ygrid.grid_line_alpha = 0.5  # Opcional: ajustar la transparencia
            fig.xgrid.minor_grid_line_color = "#e0e6eb"  # Color del grid menor
            fig.ygrid.minor_grid_line_color = "#e0e6eb"  # Color del grid menor
            fig.xgrid.minor_grid_line_alpha = 0.3  # Transparencia del grid menor
            fig.ygrid.minor_grid_line_alpha = 0.3  # Transparencia del grid menor

        COLORS = ["#d63030", "#259956"] # <-- EDIT COLOR --
        BAR_WIDTH = .8

        inc_cmap = factor_cmap('inc', COLORS, ['0', '1'])

        # Velas bajistas (rojas)
        fig.segment('Date', 'High', 'Date', 'Low', 
                source=source, color=inc_cmap, line_width=2, legend_label='OHLC'
        )
        fig.vbar(
            'Date', w, 'Open', 'Close', source=source, 
            fill_color=inc_cmap, line_color=inc_cmap, line_width=0, legend_label='OHLC'
        )
        
        legend_items = []   
        if kwargs:
            for i, (key, params) in enumerate(kwargs.items()):
                # Asegurar alineación con el índice de fechas
                values = params["values"]
                render_type = params["type"]
                color = params["color"]
                legend_label = key

                # Manejar datos 1D (Series)
                if values.ndim == 1:
                    # Usar el índice de fechas directamente
                    series_values = pd.Series(values, index=df.index)  # Reindexar usando el índice del gráfico
                    is_intermittent = pd.isna(values[int(len(values)*0.4):int(len(values)*0.9)]).sum() >= 1

                    if render_type == "scatter":
                        render = fig.scatter(
                            x=df.index, 
                            y=series_values.values, 
                            color=color, 
                            size=8,
                            legend_label=legend_label)
                    elif render_type == "line":
                        render = fig.line(
                            x=series_values.dropna().index, 
                            y=series_values.dropna(), 
                            color=color, 
                            line_width=3, 
                            legend_label=legend_label)
                    # legend_items.append(LegendItem(label=key, renderers=[render]))
                
                # Manejar datos 2D (DataFrames/matrices)
                elif values.ndim == 2:
                    for j in range(values.shape[0]):
                        series_values = pd.Series(values[j], index=df.index)
                        # print("guat")
                        COLOR_GENERATOR = color_way() 
                        if render_type == "scatter":
                            render = fig.scatter(
                                x=series_values.dropna().index,
                                y=series_values.dropna(),
                                color=COLOR_GENERATOR,
                                size=8, 
                                legend_label=legend_label)
                        elif render_type == "line":
                            render = fig.line(
                                x=series_values.dropna().index, 
                                y=series_values.dropna(), 
                                color=COLOR_GENERATOR, 
                                line_width=3, 
                                legend_label=legend_label)
                        # legend_items.append(LegendItem(label=f"{key}_{j}", renderers=[render]))


        # Añadir interacción con HoverTool
        hover = HoverTool(
            tooltips=[
                ("Fecha", "@Date{%F %H:%M}"),
                ("Apertura", "@Open{0,0}"),
                ("Cierre", "@Close{0,0}"),
                ("Máximo", "@High{0,0}"),
                ("Mínimo", "@Low{0,0}"),
            ],
            formatters={"@Date": "datetime"},
        )
        fig.add_tools(hover)

        # Crear leyenda
        legend = Legend(items=legend_items)
        fig.add_layout(legend, "right")
        fig.legend.background_fill_color = "#ced5db"  # Cambia el color del fondo
        fig.legend.background_fill_alpha = 0  # Ajusta la transparencia del fondo
        fig.legend.click_policy = "hide"

        if output.lower() == "notebook":
            output_notebook()

        show(fig)

class calculate:

    def volatility(data, period=14):
        """
        Calcula la volatilidad de los precios de cierre de un activo.

        La volatilidad se mide como la desviación estándar de los retornos porcentuales 
        sobre un periodo de tiempo especificado. Por defecto, se utiliza un periodo de 14 
        días, que es comúnmente utilizado en análisis técnico.

        Args:
            data (pd.DataFrame): Un DataFrame de pandas que debe contener una columna llamada 
                                "close" con los precios de cierre del activo.
            period (int, optional): El número de periodos sobre los cuales calcular la 
                                    volatilidad. Por defecto es 14.

        Returns:
            pd.Series: Una serie de pandas que contiene la volatilidad calculada para 
                    cada punto de tiempo, donde los primeros `period` valores serán 
                    `NaN` debido a la falta de datos suficientes para calcular la 
                    desviación estándar.

        Example:
            >>> import pandas as pd
            >>> data = pd.DataFrame({'close': [100, 102, 101, 105, 103, 108, 107]})
            >>> volatility = calculate_volatility(data)
            >>> print(volatility)

        Note:
            Asegúrate de que el DataFrame `data` tenga el índice correspondiente 
            (por ejemplo, fechas) y que la columna "close" contenga valores numéricos.

        """
        returns = data["close"].pct_change()  # Calcula los retornos porcentuales
        volatility = returns.rolling(window=period).std()  # Calcula la desviación estándar sobre el periodo especificado
        return volatility  # Retorna la serie de vo
    def duration(fecha_inicio, fecha_fin, str_format=False):
        """ 
        Calcula la duración entre dos objetos datetime y devuelve una cadena formateada con días, horas, minutos y segundos.

        Parámetros:
        - fecha_inicio (datetime): La fecha y hora de inicio.
        - fecha_fin (datetime): La fecha y hora de finalización.

        Retorna:
        - str: Una cadena con el formato "<días>D <horas>h <minutos>m <segundos>s". 
                Si la duración no incluye días, omite los días en la cadena.

        Ejemplo:
        calculate_duration(datetime(2023, 10, 1, 12, 0, 0), datetime(2023, 10, 2, 15, 30, 45))w
        # "1D 3h 30m 45s"
        """ 

        # Calcula la diferencia entre las dos fechas
        duracion = fecha_fin - fecha_inicio

        # Extrae días, segundos, horas y minutos
        dias = duracion.days
        segundos_totales = duracion.seconds
        horas = segundos_totales // 3600
        minutos = (segundos_totales % 3600) // 60
        segundos = segundos_totales % 60

        if str_format:
            return f"{f"{dias}D " if dias != 0 else ""}"  + f"{horas}h {minutos}m {segundos}s"
        else:
            return

    def annualized_return(valor_inicial, valor_final, duracion_dias):
        """
        Calcula el retorno anualizado en porcentaje.

        Parámetros:
        - valor_inicial (float): Valor inicial de la inversión.
        - valor_final (float): Valor final de la inversión.
        - duracion_dias (int): Duración de la inversión en días.

        Retorna:
        - float: Retorno anualizado como porcentaje.
        """
        duracion_dias = duracion_dias if duracion_dias != 0 else 1
        años = (duracion_dias / 365) 
        retorno_total = (valor_final / valor_inicial) - 1  # Corrección aquí
        retorno_anualizado = (1 + retorno_total) ** (1 / años) - 1
        return retorno_a

    def consistency(trades):
        """
        Calcula la consistencia de las operaciones basándose en los beneficios (Pnl) de cada operación.

        Parámetros:
        - trades_o (DataFrame): DataFrame que contiene información sobre las operaciones, incluyendo la columna "Pnl".

        Retorna:
        - float: El valor máximo de consistencia redondeado a cinco decimales.
        """
        # Hacer una copia del DataFrame original para evitar modificarlo
        trades_c = trades.copy()
        
        # Calcular la consistencia solo para las operaciones con Pnl positivo
        trades_c.loc[trades_c["Pnl"] > 0, "consistency"] = trades_c["Pnl"] / trades_c["Pnl"].sum()
        
        # Encontrar el valor máximo de la columna "consistency"
        max_consistency = np.max(trades_c["consistency"])
        
        # Redondear el valor máximo a cinco decimales
        max_consistency = round(float(max_consistency), 5)
        
        return max_co

    def streak(Pnl): # retorna: tuple: La mayor racha ganadora y la mayor racha perdedora.
        """
        Calcula la mayor racha ganadora y la mayor racha perdedora
        basada en una serie de resultados de PnL.

        Parámetros:
        Pnl (pd.Series): Serie de resultados del PnL, donde valores positivos indican ganancias 
                        y valores negativos indican pérdidas.

        Retorna:
        tuple: La mayor racha ganadora y la mayor racha perdedora.
        """
        # Crear la serie de resultados (True si ganancia, False si pérdida)
        resultados = Pnl > 0

        # Calcular la racha ganadora más larga
        racha_ganadora = (resultados.cumsum() - resultados.cumsum().where(~resultados).ffill().fillna(0)).max()

        # Calcular la racha perdedora más larga
        racha_perdedora = ((~resultados).cumsum() - (~resultados).cumsum().where(resultados).ffill().fillna(0)).max()
        
        return racha_ganadora, racha_

    def f_kelly(operations):
        """
        Calcula la fracción óptima de la fórmula de **Kelly** que maximiza el crecimiento del capital 
        en función de los resultados de una serie de operaciones (ganancias y pérdidas).

        La fórmula de **Kelly** se utiliza para determinar la fracción del capital a arriesgar en una operación 
        para maximizar el crecimiento esperado a largo plazo.

        Parámetros:
        -----------
        operations : list
            Lista de resultados de las operaciones. Los valores positivos representan ganancias y los valores negativos representan pérdidas.

        Retorna:
        --------
        float
            La fracción óptima de Kelly (valor entre 0 y 1), que indica la fracción del capital que se debe arriesgar 
            en cada operación para maximizar el crecimiento esperado.

        Descripción:
        ------------
        La función calcula la **fracción de Kelly** basándose en la siguiente fórmula:
        
        \\[
        f_{kelly} = \\frac{(P \\times B) - Q}{B}
        \\]
        Donde:
        - \\(P\\) es la probabilidad de una ganancia (la proporción de operaciones ganadoras).
        - \\(Q\\) es la probabilidad de una pérdida (la proporción de operaciones perdedoras).
        - \\(B\\) es el **payoff**, la relación entre la ganancia promedio y la pérdida promedio.

        La fracción calculada se ajusta para que esté entre 0 y 1. Si no hay ganancias o no hay pérdidas, 
        la fracción se ajustará correctamente para evitar errores de cálculo.
        """
        
        # Filtrar ganancias y pérdidas
        gains = [x for x in operations if x > 0]
        losses = [x for x in operations if x < 0]
        
        # Calcular probabilidades
        prob_gain = len(gains) / len(operations)
        prob_loss = len(losses) / len(operations)
        
        # Calcular ganancia promedio y pérdida promedio
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        
        # Calcular Payoff (relación ganancia/pérdida)
        payoff = avg_gain / avg_loss if avg_loss != 0 else 0
        
        # Fórmula de Kelly clásica
        kelly_fraction = (prob_gain * payoff - prob_loss) / payoff
        
        # Asegurarse de que f_optima esté entre 0 y 1
        optimal_fraction = max(0, min(kelly_fraction, 1))  # Fracción entre 0 y 1
        
        return optimal

    def expected_value(winrate : float, avg_win : float, avg_loss : float):
        """
        Calcula la esperanza matemática de un sistema de trading.

        Parámetros:
        -----------
        winrate : float
            Probabilidad de ganar (entre 0 y 1).
        ganancia_promedio : float
            Promedio de ganancia cuando se gana.
        perdida_promedio : float
            Promedio de pérdida cuando se pierde (valor negativo).

        Retorno:
        --------
        float
            La esperanza matemática del sistema.
        """
        # Probabilidad de perder
        lossrate = 1 - winrate
        
        # Esperanza matemática
        expected = (winrate * avg_win) - (lossrate * avg_loss)
        
        return

    def ror(winrate, avg_win_pct, avg_loss_pct, max_for_ruin=0.3):
        """
        Calcula el riesgo de ruina (Risk of Ruin, ROR) basado en el rendimiento de una estrategia de trading.

        Parámetros:
        -----------
        winrate : float
            Probabilidad de ganar una operación (entre 0 y 1).
        avg_win_pct : float
            Ganancia promedio por operación ganadora (porcentaje positivo, como 0.02 para 2%).
        avg_loss_pct : float
            Pérdida promedio por operación perdedora (porcentaje positivo, como 0.01 para 1%).
        max_for_ruin : float, opcional
            Porcentaje máximo del capital en riesgo antes de ruina (por defecto 0.3).

        Retorno:
        --------
        float
            Riesgo de ruina como un valor entre 0 y 1.

        Ejemplo:
        --------
        >>> calculate_ror(0.56, 0.02, 0.01)
        0.04158913976181979
        """
    
        win_pct = winrate
        loss_pct = 1 - winrate
        z = (win_pct * avg_win_pct) - (loss_pct * avg_loss_pct)
        a = (win_pct * (avg_win_pct)**2 + loss_pct * (avg_loss_pct)**2)**0.5
        p = 0.5 * (1 + z / a) # Probabilidad crítica p
        ROR = ((1 - p) / p)**(max_for_ruin / a)

        r

    def sortino(returns : pd.Series, 
                        scalar = 252):

        """
        Calcula el ratio de Sortino, una métrica que mide el rendimiento ajustado al riesgo penalizando 
        únicamente la volatilidad negativa (pérdidas).

        Parámetros:
        -----------
        returns : pd.Series
            Serie de rendimientos (diarios, semanales, mensuales, etc.) expresados en decimales. 
            Por ejemplo, 0.02 para un 2% de rendimiento.
        scalar : opcional
            Factor de escala para anualizar el ratio. 
            - Usar 252 para rendimientos diarios (días de mercado en un año).
            - Usar 52 para rendimientos semanales.
            - Usar 12 para rendimientos mensuales.
            Por defecto, 252.

        Retorno:
        --------
        float
            Ratio de Sortino. Un valor que mide el rendimiento ajustado al riesgo penalizando las pérdidas.

        Interpretación:
        ---------------
        - **Sortino < 0**: El rendimiento promedio es negativo, lo que indica una estrategia no rentable.
        - **Sortino entre 0 y 1**: La estrategia tiene un rendimiento ajustado al riesgo bajo, pero puede ser aceptable dependiendo del contexto.
        - **Sortino > 1**: Buen rendimiento ajustado al riesgo. Valores más altos indican estrategias más eficientes en generar ganancias con baja volatilidad negativa.

        Notas:
        ------
        - Penaliza únicamente la volatilidad de las pérdidas, ignorando la volatilidad de las ganancias.
        - Un valor alto indica que el rendimiento compensa suficientemente los riesgos asumidos.
        - Si `returns` contiene demasiadas pérdidas (valores negativos), el denominador puede ser grande, resultando en un ratio bajo.

        Ejemplo de uso:
        ---------------
        >>> import pandas as pd
        >>> import numpy as np
        >>> daily_returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Simula rendimientos diarios
        >>> calculate_sortino(daily_returns)
        0.75  # Ejemplo de un Sortino razonable.
        """

        mean = np.mean(returns)
        volatility = np.std(returns[returns < 0]) 
        if volatility == 0:  # Evitar división por cero si no hay pérdidas
            return float('nan')  # Mejor devolver NaN para consistencia
        sortino = np.sqrt(scalar) * mean / volatility 

        retur

    def sharpe(returns: pd.Series, risk_free_rate: float = 0.0, scalar: int = 252):
        """
        Calcula el Sharpe ratio, una métrica que mide el rendimiento ajustado al riesgo total.

        Parámetros:
        -----------
        returns : pd.Series
            Serie de rendimientos (diarios, semanales, mensuales, etc.) expresados en decimales.
            Ejemplo: 0.02 para un 2% de rendimiento.
        risk_free_rate : float, opcional
            Tasa libre de riesgo anualizada. Por defecto es 0.0 (se asume tasa cero).
        scalar : int, opcional
            Factor de escala para anualizar el ratio. 
            - Usar 252 para rendimientos diarios (días de mercado en un año).
            - Usar 52 para rendimientos semanales.
            - Usar 12 para rendimientos mensuales.
            Por defecto, 252.

        Retorno:
        --------
        float
            Sharpe ratio. Un valor que mide el rendimiento ajustado al riesgo total.

        Interpretación:
        ---------------
        - **Sharpe < 0**: El rendimiento de la estrategia es menor que la tasa libre de riesgo.
        - **Sharpe entre 0 y 1**: El rendimiento es positivo, pero no compensa suficientemente la volatilidad total.
        - **Sharpe > 1**: Buena relación rendimiento/riesgo. Valores más altos indican estrategias más eficientes.
        - **Sharpe > 2**: Excelente relación rendimiento/riesgo, típico en estrategias muy robustas.

        Notas:
        ------
        - A diferencia del Sortino ratio, el Sharpe ratio utiliza la volatilidad total (tanto pérdidas como ganancias).
        - El riesgo se mide mediante la desviación estándar de los rendimientos.

        Ejemplo de uso:
        ---------------
        >>> import pandas as pd
        >>> import numpy as np
        >>> daily_returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Simula rendimientos diarios
        >>> calculate_sharpe(daily_returns, risk_free_rate=0.01)
        1.25  # Ejemplo de un Sharpe razonable.
        """
        mean_excess_return = np.mean(returns) - (risk_free_rate / scalar)  # Rendimiento promedio ajustado
        volatility = np.std(returns)  # Volatilidad total (positiva y negativa)
        if volatility == 0:  # Evitar división por cero
            return float('nan')  # Consistencia con otros casos extremos
        sharpe_ratio = np.sqrt(scalar) * mean_excess_return / volatility
        return sha

    def beta(returns: pd.Series, benchmark_returns: pd.Series):
        """
        Calcula el Beta de una estrategia o activo respecto a un índice de referencia (benchmark).

        Parámetros:
        -----------
        returns : pd.Series
            Serie de rendimientos del activo o estrategia.
        benchmark_returns : pd.Series
            Serie de rendimientos del índice de referencia (benchmark).

        Retorno:
        --------
        float
            Beta del activo respecto al benchmark.

        Interpretación:
        ---------------
        - **Beta = 1**: El activo se mueve en línea con el mercado.
        - **Beta > 1**: El activo es más volátil que el mercado (más riesgoso).
        - **Beta < 1**: El activo es menos volátil que el mercado (menos riesgoso).
        - **Beta < 0**: El activo tiende a moverse en dirección opuesta al mercado.

        Ejemplo de uso:
        ---------------
        >>> asset_returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.02])
        >>> market_returns = pd.Series([0.01, 0.015, -0.005, 0.025, 0.02])
        >>> calculate_beta(asset_returns, market_returns)
        1.2  # El activo es un 20% más volátil que el mercado.
        """

        if len(returns) != len(benchmark_returns):
            raise ValueError("Las series de retornos y del benchmark deben tener la misma longitud.")

        variance = np.var(benchmark_returns)
        if variance == 0:  # Evitar división por cero
            return float('nan')  # Beta no es definible si el benchmark no tiene movimiento
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        beta = covariance / variance
        re

    def alpha(returns: pd.Series, 
                        benchmark_returns: pd.Series, 
                        risk_free_rate: float = 0.0):
        """
        Calcula el Alpha de una estrategia o activo respecto a un índice de referencia (benchmark).

        Parámetros:
        -----------
        returns : pd.Series
            Serie de rendimientos del activo o estrategia.
        benchmark_returns : pd.Series
            Serie de rendimientos del índice de referencia (benchmark).
        risk_free_rate : float, opcional
            Tasa libre de riesgo anualizada. Por defecto es 0.0 (se asume tasa cero).

        Retorno:
        --------
        float
            Alpha del activo, que representa el rendimiento superior al esperado ajustado al riesgo.

        Fórmula:
        --------
        Alpha = Rendimiento del activo - [Rendimiento libre de riesgo + Beta * (Rendimiento del benchmark - Rendimiento libre de riesgo)]

        Interpretación:
        ---------------
        - **Alpha > 0**: La estrategia está superando al mercado ajustado por riesgo.
        - **Alpha < 0**: La estrategia está por debajo del rendimiento esperado dado su riesgo (Beta).
        - **Alpha = 0**: La estrategia está en línea con el rendimiento esperado.

        Ejemplo de uso:
        ---------------
        >>> asset_returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.02])
        >>> market_returns = pd.Series([0.01, 0.015, -0.005, 0.025, 0.02])
        >>> calculate_alpha(asset_returns, market_returns, risk_free_rate=0.01)
        0.003  # Alpha positivo indica que la estrategia superó al mercado.
        """
        
        if len(returns) != len(benchmark_returns):
            raise ValueError(f"Las series tienen longitudes diferentes: "
                            f"returns({len(returns)}) y benchmark_returns({len(benchmark_returns)})")

        beta = calculate_beta(returns, benchmark_returns)
        if np.isinf(beta) or np.isnan(beta):  # Validar Beta
            return float('nan')  # Alpha no puede calcularse sin Beta
        mean_asset_return = np.mean(returns)
        mean_benchmark_return = np.mean(benchmark_returns)
        alpha = mean_asset_return - (risk_free_rate + beta * (mean_benchmark_return - risk_free_rate))
        ret

    def avg_price(price, position: 'Position', trade: 'Trade'):
        """
        Calcula el precio promedio ponderado de una posición.

        Args:
            price (float): Precio actual para el calculo
            position (Position): Objeto Position con precio y tamaño actual
            trade (Trade): Objeto Trade con precio y tamaño a agregar

        Returns:
            float: Nuevo precio promedio de la posición
        """
        return ((position.avg_price * abs(position.size)) + (price * abs(trade.size))) / (abs(position.size) + abs(trade.size))


"""Indicadores"""   
class _Indicator:
    """Clase base para indicadores."""

        # def create_base_figure(title="Indicator", 
        #                         x_axis_type="datetime", 
        #                         height=350, 
        #                         width=800, 
        #                         tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"],
        #                         background_fill_color="#e0e6eb",  # Fondo interior
        #                         border_fill_color="#e0e6eb"  # Fondo exterior
        #                         ):
        #     print(self.__class__.__name__)
        #     fig = figure(title=title, 
        #                 x_axis_type=x_axis_type, 
        #                 height=height, 
        #                 width=width, 
        #                 tools=tools,
        #                 background_fill_color=background_fill_color,  # Fondo interior
        #                 border_fill_color=border_fill_color,     # Fondo exterior
        #             )

        #     return fig

    @property
    def dim(self):
        """Devuelve la dimensión del indicador."""
        return self.values.ndim

class Overlay(_Indicator): 
    """Clase base para medias móviles."""

    def __init__(self, 
                 data : Union[pd.Series, np.ndarray, 'DataOHLC'], 
                 length : int):
        # Convertir los datos a np.array si es necesario
        # Validar el formato de los datos y definir data e index
        if isinstance(data, pd.DataFrame):
            self.data = data.Close.Values
            self.index = data.index.values
        elif isinstance(data, pd.Series):
            self.data = data.values
            self.index = data.index.values
        elif isinstance(data, np.ndarray):
            self.data = data
            self.index = np.arange(len(data))
        else:
            if hasattr(data, 'Close') and hasattr(data, 'index'):
                try:
                    self.data = data.Close
                    self.index = data.index
                except AttributeError as e:
                        raise ValueError(f"'Close' e 'index' no tienen un formato valido, (formato valido: np.array): {e}")
            else: 
                raise ValueError("La data de entrada debe ser un pd.Series, np.array u Objeto con atributos 'Close' e 'index' en forma de np.array")

        self.length = int(length)
        self.name = f"{self.__class__.__name__} {self.length}" # Base name
        self.tag = [self.name] # tag por defecto
        

    @property
    def values(self):
        return self._get_result()

    def calculate(self):
        """Método para calcular las medias móviles. Lo implementarán las clases derivadas."""
        raise NotImplementedError("Método para calcular las medias móviles. Lo implementarán las clases derivadas.")

    def plot(self, height=350, width=800, output="notebook", sizing_mode="stretch_width"):
        """
            Grafica el precio de un activo junto con sus medias móviles en Bokeh.
            
            Parámetros:
            - price: pd.Series con los precios del activo.
            - moving_averages: pd.DataFrame con las medias móviles.
            - height: Altura del gráfico.
            - width: Ancho del gráfico.
        """

        if output.lower() == "notebook":
            output_notebook()

        data = self.data
        lines_data = self.values
        index = self.index
        name = self.name
        tag = self.tag

        # Verificar si lines_data es un array unidimensional y convertirlo a bidimensional
        if lines_data.ndim == 1:
            lines_data = [lines_data]

        # Colores para las medias móviles
        colors = color_way(len(tag))

        legend_items = []  # Lista para guardar las leyendas manuales

        # Crear figura
        fig = figure(title=name, 
                x_axis_type="datetime", 
                height=height, 
                width=width, 
                tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"],
                background_fill_color="#e0e6eb",  # Fondo interior
                border_fill_color="#e0e6eb",     # Fondo exterior
                sizing_mode=sizing_mode,
            )
       
        # Agregar línea de precio
        price_line = fig.line(index, data, color="black", line_width=2)
        legend_items.append(LegendItem(label="Price" , renderers=[price_line]))

        # Agregar lineas
        for i, tag, line in zip(range(len(tag)), tag, lines_data):
            price_line = fig.line(index, line, color=colors[i], line_width=2)
            legend_items.append(LegendItem(label=tag , renderers=[price_line]))
        
        # Agregar herramienta Hover
        hover = HoverTool(
            tooltips=[("Date", "@x{%F %H:%M}"),
                    ("Price", "@y{0.2f}")],
            formatters={"@x": "datetime"},
            mode="vline"
            )
        fig.add_tools(hover)

        # Modificar Grafico
        fig.xaxis.axis_label = 'Fecha'
        fig.yaxis.axis_label = 'Precio'
        fig.title.text_color = "#333333"
        fig.xaxis.axis_label_text_color = "#333333"
        fig.yaxis.axis_label_text_color = "#333333"
        fig.xaxis.major_label_text_color = "#333333"
        fig.yaxis.major_label_text_color = "#333333"
        # Evitar mostrar en connotacion cientifica los numeros del eje y
        fig.yaxis.formatter.use_scientific = False
        # Cambiar el color de las líneas de los ejes y ticks a azul
        fig.xaxis.axis_line_color = "#e0e6eb"
        fig.yaxis.axis_line_color = "#e0e6eb"
        fig.xaxis.major_tick_line_color = "#a5a5a5"
        fig.yaxis.major_tick_line_color = "#c7c7c7"
        fig.xaxis.minor_tick_line_color = "#c7c7c7"
        fig.yaxis.minor_tick_line_color = "#c7c7c7"
        # Cambiar el color del grid
        fig.xgrid.grid_line_color = "#c7c7c7"
        fig.ygrid.grid_line_color = "#c7c7c7"
        fig.xgrid.grid_line_alpha = 0.5  # Opcional: ajustar la transparencia
        fig.ygrid.grid_line_alpha = 0.5  # Opcional: ajustar la transparencia
        fig.xgrid.minor_grid_line_color = "#e0e6eb"  # Color del grid menor
        fig.ygrid.minor_grid_line_color = "#e0e6eb"  # Color del grid menor
        fig.xgrid.minor_grid_line_alpha = 0.3  # Transparencia del grid menor
        fig.ygrid.minor_grid_line_alpha = 0.3  # Transparencia del grid menor

        # Crear leyenda
        legend = Legend(items=legend_items)
        # Colocar la leyenda dentro del gráfico en la esquina superior izquierda
        fig.add_layout(legend, "center")
        # Modificar la leyenda
        fig.legend.background_fill_color = "#E6EDF4FF"  # Cambia el color del fondo
        fig.legend.background_fill_alpha = 0.35  # Ajusta la transparencia del fondo
        fig.legend.click_policy = "hide"
        # Posicionar la leyenda en la esquina superior izquierda
        fig.legend.location = "top_left"
        # Opcional: ajustar el padding interno de la leyenda
        fig.legend.padding = 5
        fig.legend.border_radius = 7
        # Opcional: agregar un borde a la leyenda para distinguirla mejor
        fig.legend.border_line_color = "#E6EDF4FF"
        fig.legend.border_line_alpha = 0.35
        
        # Mostrar figura
        show(fig)


class SMA(Overlay):
    # def evaluate(self, method="cross_over"):
    #     return evaluate.cross_over_method(arr=self.data, reference_value=self.values)
    
    @staticmethod
    def optimize(data, method, **kwargs):
        func_methods = {
            "cross_over":evaluator.cross_over_method,
            "on_trigger":evaluator.on_trigger_method,
            "on_trigger_exit":evaluator.on_trigger_exit_method,
        }
        func_method = func_methods[method]

        results = {}
        for key, values in kwargs.items():
            for val in values:
                # Calcular el indicador
                ref_line = self(data=data, **dic(key=val)).values

                # Evaluar el indicador
                results[key] = func_method(arr=data, reference_value=ref_line)
        return results 


    def _get_result(self):
        return self._sma(self.data, self.length)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _sma(data, length):
        """
        Calcula la media móvil simple (SMA) de un período dado utilizando NumPy.
        """
        # Usamos np.convolve para calcular la SMA con un kernel de unos (1's)
        sma_values = np.convolve(data, np.ones(length), mode='valid') / length
        # Devolvemos una serie con los valores NaN al inicio y los valores de la SMA
        result = np.full(len(data), np.nan)  # Inicializamos con NaN
        # Colocamos los valores de la SMA a partir de la posición donde podemos calcularla
        result[length - 1:] = sma_values
        return result

class EMA(Overlay):

    def _get_result(self):
        return self._ema(self.data, self.length)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _ema(data, length):
        """
            Calcula la media móvil exponencial (EMA) de un período dado utilizando NumPy.
            Optimizado para Numba.
        """
        if len(data) < length or length <= 0:
            # Devuelve un array de NaNs si no hay suficientes datos o el período no es válido
            return np.full(len(data), np.nan)

        alpha = 2.0 / (length + 1.0)
        ema_array = np.full(len(data), np.nan)

        # Calcula la SMA para el primer valor de la EMA
        # Numba maneja bien np.mean sobre arrays de NumPy en modo nopython
        if len(data) >= length:
            initial_sma = 0.0
            for i in range(length):
                initial_sma += data[i]
            ema_array[length - 1] = initial_sma / length

            # Calcula los valores subsiguientes de la EMA
            for i in range(length, len(data)):
                ema_array[i] = alpha * data[i] + (1.0 - alpha) * ema_array[i - 1]
        
        return ema_array

class WMA(Overlay):

    def _get_result(self):
        return self._wma(self.data, self.length)
        
    @staticmethod
    @jit(nopython=True, cache=True)
    def _wma(data, length):
        """
        Calcula la media móvil ponderada (WMA) de un período dado utilizando NumPy.
        """
        weights = np.arange(1, length + 1)  # Pesos de la WMA (más grande al final)
        
        # Usamos la convolución para aplicar los pesos sobre los precios
        wma_values = np.convolve(data, weights[::-1], mode='valid') / weights.sum()
        
        # Creamos un array lleno de NaN y lo completamos con los valores calculados de la WMA
        result = np.full(len(data), np.nan)
        result[length - 1:] = wma_values
        
        return result

class HMA(Overlay):

    def _get_result(self):
        return self._hma(self.data, self.length)

    @staticmethod
    def _hma(data, length):
        """
            Calcula la Hull Moving Average (HMA) utilizando NumPy.
        """
        def _wma(data, length):
            weights = np.arange(1, length + 1)  
            wma_values = np.convolve(data, weights[::-1], mode='valid') / weights.sum()
            result = np.full(len(data), np.nan)
            result[length - 1:] = wma_values
            return result

        # Paso 1: WMA del período completo
        wma_full = _wma(data, length)
        
        # Paso 2: WMA de la mitad del período
        half_period = length // 2
        wma_half = _wma(data, half_period)
        
        # Paso 3: Restamos la WMA de mitad de periodo de la WMA del periodo completo
        diff_wma = 2 * wma_half - wma_full
        
        # Paso 4: Calculamos la WMA del resultado con el período sqrt(n)
        sqrt_length = int(np.sqrt(length))
        hma = _wma(diff_wma, sqrt_length)
        
        return hma

class BBANDS(Overlay):
    """
        Implementación de las Bandas de Bollinger.
        
        Las Bandas de Bollinger consisten en:
        - Una banda superior (SMA + n desviaciones estándar)
        - Una media móvil simple (SMA) central
        - Una banda inferior (SMA - n desviaciones estándar)
    """

    def __init__(self, data, length=20, num_std: float = 2.0):
        """
            Inicializa las Bandas de Bollinger.
            
            Parámetros:
            -----------
            price: Serie de precios
            period: Período para la SMA y el cálculo de la desviación estándar
            num_std: Número de desviaciones estándar para las bandas superior e inferior
        """
        super().__init__(data, length)
        self.num_std = num_std
        self.name = f"Bollinger Bands ({self.length}, {self.num_std})"
        self.tag = ["Upper_band", "Middle_band", "Lower_band"]
        # Asegurar que el precio sea un pd.Series 
        # Definir slef.price e self.index
        if isinstance(data, pd.DataFrame):
            try:
                self.data = data.Close
            except:
                self.data = data.squeeze()
            self.index = self.data.index
        elif isinstance(data, pd.Series):
            self.data = data
            self.index = data.index
        elif isinstance(data, np.ndarray):
            self.data = pd.Series(data)
            self.index = range(len(data))
        else:
            raise ValueError("El precio debe ser un pd.Series, np.array o pd.DataFrame")
        
    def _get_result(self):
        """Calcula las Bandas de Bollinger."""
        # Calcular la SMA
        self.middle_band = SMA(self.data, self.length)._get_result()
        
        # Calcular la desviación estándar
        rolling_std = pd.Series(self.data).rolling(window=self.length).std()
        
        # Calcular las bandas
        self.upper_band = self.middle_band + (rolling_std * self.num_std)
        self.lower_band = self.middle_band - (rolling_std * self.num_std)
        
        # Crear un DataFrame con los resultados
        bands = np.array([self.upper_band, self.middle_band, self.lower_band])

        return bands
    
    class Evaluation:
        """
        Clase para evaluar estrategias de trading basadas en Bandas de Bollinger.
        """
        
        def __init__(self, bollinger_bands: "BBANDS", method: str = "band_touch", inverted: bool = False):
            """
                Inicializa la evaluación de las Bandas de Bollinger.
                
                Parámetros:
                -----------
                bollinger_bands: Objeto BollingerBands
                method: Método de evaluación ('band_touch', 'band_breakout', etc.)
                inverted: Si es True, invierte las señales de compra y venta
            """
            # Guardar datos
            self.bollinger_data = bollinger_bands.bands
            self.price_series = bollinger_bands.data
            
            # Método para evaluar
            self.method = method
            
            # Si se invierten las posiciones
            self.inverted_positions = inverted
            
            # Crear DataFrame con los datos
            self.df = pd.concat([self.bollinger_data, self.price_series], axis=1)
            
            # Eliminar valores nulos
            self.df.dropna(inplace=True)
            
            # Calcular el porcentaje de cambio de los precios
            self.price_pct = self.df["Close"].pct_change(1)
            self.price_pct[self.price_pct.index[0]] = 0
            
            # Guardar rendimientos y posiciones
            self.returns = pd.DataFrame()
            self.positions = pd.DataFrame()
            
            # Evaluar usando el método seleccionado
            if self.method == "band_touch":
                self.__evaluate_band_touch_method()
            elif self.method == "band_breakout":
                self.__evaluate_band_breakout_method()
            else:
                raise ValueError(f"El método '{self.method}' no es reconocido.")
        
        def __evaluate_band_touch_method(self):
            """
            Evalúa la estrategia de 'band_touch', donde:
            - Compra cuando el precio toca la banda inferior
            - Vende cuando el precio toca la banda superior
            """
            # Condiciones de compra y venta
            buy_condition = self.df["Close"] <= self.df["Lower"]
            sell_condition = self.df["Close"] >= self.df["Upper"]
            
            # Inicializar posiciones en 0 (neutral)
            positions = pd.Series(0, index=self.df.index)
            
            # Establecer posiciones
            positions[buy_condition] = 1  # Compra
            positions[sell_condition] = -1  # Venta
            
            # Si las posiciones están invertidas, cambiar
            if self.inverted_positions:
                positions = -positions
            
            # Calcular rendimientos con un desfase de 1 día
            returns = self.price_pct * positions.shift(1)
            returns.fillna(0, inplace=True)
            
            # Almacenar resultados
            self.returns["band_touch returns"] = returns
            self.positions["band_touch positions"] = positions
        
        def __evaluate_band_breakout_method(self):
            """
            Evalúa la estrategia de 'band_breakout', donde:
            - Compra cuando el precio rompe hacia arriba la banda superior
            - Vende cuando el precio rompe hacia abajo la banda inferior
            """
            # Condiciones de compra y venta (con cruce de bandas)
            price_shift = self.df["Close"].shift(1)
            upper_shift = self.df["Upper"].shift(1)
            lower_shift = self.df["Lower"].shift(1)
            
            buy_condition = (price_shift <= upper_shift) & (self.df["Close"] > self.df["Upper"])
            sell_condition = (price_shift >= lower_shift) & (self.df["Close"] < self.df["Lower"])
            
            # Inicializar posiciones en 0 (neutral)
            positions = pd.Series(0, index=self.df.index)
            
            # Establecer posiciones
            positions[buy_condition] = 1  # Compra
            positions[sell_condition] = -1  # Venta
            
            # Mantener la última posición conocida (forward fill)
            positions = positions.replace(0, np.nan).fillna(method='ffill').fillna(0)
            
            # Si las posiciones están invertidas, cambiar
            if self.inverted_positions:
                positions = -positions
            
            # Calcular rendimientos con un desfase de 1 día
            returns = self.price_pct * positions
            returns.fillna(0, inplace=True)
            
            # Almacenar resultados
            self.returns["band_breakout returns"] = returns
            self.positions["band_breakout positions"] = positions
        
        def plot(self, view_returns=True, height=350, width=800):
            """
            Crea gráficos de los rendimientos acumulados con Bokeh.
            
            Parámetros:
            -----------
            view_returns: Si es True, muestra el gráfico de rendimientos acumulados
            height: Altura de cada gráfico
            width: Ancho del gráfico
            """
            # Similar a tu implementación existente para MovingAverage.Evaluation.plot()
            plots = []
            
            if view_returns:
                fig_returns = figure(
                    title="Rendimientos Acumulados", 
                    width=width, 
                    height=height, 
                    x_axis_type="datetime", 
                    tools="pan,wheel_zoom,box_zoom,reset,save",
                    background_fill_color="#e0e6eb",
                    border_fill_color="#e0e6eb",
                )
                
                colors = color_way(len(self.returns))
                legend_items = []
                
                # Graficar rendimientos de los precios
                line_price = fig_returns.line(
                    self.price_pct.index, self.price_pct.cumsum() * 100, 
                    line_width=1.6, color="#333333"
                )
                legend_items.append(LegendItem(label="Rendimientos del Precio", renderers=[line_price]))
                
                # Graficar rendimientos de cada estrategia
                for i, ret_tag in enumerate(self.returns.columns):
                    line = fig_returns.line(
                        self.returns.index, self.returns[ret_tag].cumsum() * 100, 
                        color=colors[i], line_width=1.6
                    )
                    legend_items.append(LegendItem(label=ret_tag, renderers=[line]))
                
                # Modificar Gráfico (similar a tu estilo existente)
                fig_returns.xaxis.axis_label = 'Fecha'
                fig_returns.yaxis.axis_label = 'Rendimiento (%)'
                fig_returns.title.text_color = "#333333"
                fig_returns.xaxis.axis_label_text_color = "#333333"
                fig_returns.yaxis.axis_label_text_color = "#333333"
                fig_returns.xaxis.major_label_text_color = "#333333"
                fig_returns.yaxis.major_label_text_color = "#333333"
                fig_returns.yaxis.formatter.use_scientific = False
                fig_returns.xaxis.axis_line_color = "#e0e6eb"
                fig_returns.yaxis.axis_line_color = "#e0e6eb"
                fig_returns.xaxis.major_tick_line_color = "#a5a5a5"
                fig_returns.yaxis.major_tick_line_color = "#c7c7c7"
                fig_returns.xaxis.minor_tick_line_color = "#c7c7c7"
                fig_returns.yaxis.minor_tick_line_color = "#c7c7c7"
                fig_returns.xgrid.grid_line_color = "#c7c7c7"
                fig_returns.ygrid.grid_line_color = "#c7c7c7"
                fig_returns.xgrid.grid_line_alpha = 0.5
                fig_returns.ygrid.grid_line_alpha = 0.5
                fig_returns.xgrid.minor_grid_line_color = "#e0e6eb"
                fig_returns.ygrid.minor_grid_line_color = "#e0e6eb"
                fig_returns.xgrid.minor_grid_line_alpha = 0.3
                fig_returns.ygrid.minor_grid_line_alpha = 0.3
                
                # Crear leyenda
                legend = Legend(items=legend_items)
                fig_returns.add_layout(legend, "right")
                
                # Modificar la leyenda
                fig_returns.legend.background_fill_color = "#ced5db"
                fig_returns.legend.background_fill_alpha = 0
                fig_returns.legend.click_policy = "hide"
                
                # Agregar línea en y=0
                zero_line = Span(
                    location=0, 
                    dimension='width', 
                    line_color="#9297a3", 
                    line_width=0.7, 
                    line_dash="dashed"
                )
                fig_returns.add_layout(zero_line)
                
                # Agregar herramienta Hover
                hover = HoverTool(
                    tooltips=[
                        ("Fecha", "@x{%F %H:%M}"),
                        ("Rendimiento", "@y{0.3f}%")
                    ],
                    formatters={"@x": "datetime"},
                    mode="vline"
                )
                fig_returns.add_tools(hover)
                
                plots.append(fig_returns)
            
            # Mostrar gráficos
            show(column(*plots))

class BBANDS_2_optimizando(Overlay):
    """
    Clase para calcular las Bandas de Bollinger.
    
    Las Bandas de Bollinger son un indicador técnico que consiste en:
    - Una banda media (SMA del precio)
    - Una banda superior (SMA + n*desviación estándar)
    - Una banda inferior (SMA - n*desviación estándar)
    """
    
    def __init__(self, 
                 price: Union[pd.Series, np.array, pd.DataFrame], 
                 period: int = 20,
                 num_std: float = 2.0):
        """
        Inicializa el cálculo de las Bandas de Bollinger.
        
        Parámetros:
        -----------
        price: Serie de precios
        period: Período para el cálculo de la SMA y la desviación estándar
        num_std: Número de desviaciones estándar para las bandas superior e inferior
        """
        super().__init__(price, period)
        self.num_std = num_std
        self.name = f"BBANDS({self.length}, {self.num_std})"
        self.value_tag = ["pper_band", "Middle_band", "Lower_band"]
    
    @staticmethod
    # @jit(nopython=True)
    def _calculate_bbands(price_array, period, num_std):
        """
            Calcula las Bandas de Bollinger utilizando NumPy y acelerado con Numba.
            
            Parámetros:
            -----------
            price_array: Array de precios
            period: Período para el cálculo
            num_std: Número de desviaciones estándar
            
            Retorna:
            --------
            Tupla con (banda_superior, banda_media, banda_inferior)
        """ 
        n = len(price_array)
        rolling_mean = np.full(n, np.nan)
        rolling_std = np.full(n, np.nan)
        
        # Calcular la media móvil y la desviación estándar
        for i in range(period - 1, n):
            window = price_array[i - period + 1:i + 1]
            rolling_mean[i] = np.mean(window)
            # ddof=1 para usar la desviación estándar muestral (equivalente a pandas)
            rolling_std[i] = np.std(window, ddof=1)
        
        # Calcular las bandas superior e inferior
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return upper_band, rolling_mean, lower_band
    
    def _get_result(self):
        """
            Obtiene el resultado del cálculo de las Bandas de Bollinger.
            
            Retorna:
            --------
            Array con [banda_superior, banda_media, banda_inferior]
        """
        price_array = np.array(self.values)
        upper_band, middle_band, lower_band = self._calculate_bbands(
            price_array, self.length, self.num_std
            )
        
        # Convertir a Series de pandas para mantener el índice
        if isinstance(self.values, pd.Series):
            upper_band = pd.Series(upper_band, index=self.values.index)
            middle_band = pd.Series(middle_band, index=self.values.index)
            lower_band = pd.Series(lower_band, index=self.values.index)
            return pd.DataFrame({
                'upper_band': upper_band,
                'middle_band': middle_band,
                'lower_band': lower_band
            })
        
        # Si no es una Serie, devolver un array de NumPy
        return np.vstack((upper_band, middle_band, lower_band))
    
    def plot(self, height=350, width=800, output="notebook"):
        """
        Grafica las Bandas de Bollinger junto con el precio.
        
        Parámetros:
        -----------
        height: Altura del gráfico
        width: Ancho del gráfico
        output: Tipo de salida ('notebook' para Jupyter)
        """
        price = self.values
        bbands_data = pd.DataFrame(self.values)
        
        # Crear figura
        fig = figure(
            title="Price & Bollinger Bands", 
            x_axis_type="datetime" if isinstance(price.index, pd.DatetimeIndex) else "auto", 
            height=height, 
            width=width, 
            tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"],
            background_fill_color="#e0e6eb",
            border_fill_color="#e0e6eb",
        )
        
        # Colores para las bandas
        colors = color_way(3)
        legend_items = []
        
        # Agregar línea de precio
        price_line = fig.line(price.index, price, color="black", line_width=2)
        legend_items.append(LegendItem(label="Price", renderers=[price_line]))
        
        # Agregar bandas
        upper_line = fig.line(bbands_data.index, bbands_data['upper_band'], 
                             color=colors[0], line_width=1.5, line_dash="dashed")
        middle_line = fig.line(bbands_data.index, bbands_data['middle_band'], 
                              color=colors[1], line_width=1.5)
        lower_line = fig.line(bbands_data.index, bbands_data['lower_band'], 
                             color=colors[2], line_width=1.5, line_dash="dashed")
        
        # Agregar área sombreada entre las bandas
        band_area = fig.patch(
            xs=np.concatenate([bbands_data.index, bbands_data.index[::-1]]),
            ys=np.concatenate([bbands_data['upper_band'], bbands_data['lower_band'][::-1]]),
            color=colors[1], alpha=0.1
        )
        
        # Agregar leyendas
        legend_items.extend([
            LegendItem(label=f"Upper Band ({self.num_std}σ)", renderers=[upper_line]),
            LegendItem(label="Middle Band (SMA)", renderers=[middle_line]),
            LegendItem(label=f"Lower Band ({self.num_std}σ)", renderers=[lower_line])
        ])
        
        # Agregar herramienta Hover
        hover = HoverTool(
            tooltips=[
                ("Date", "@x{%F %H:%M}" if isinstance(price.index, pd.DatetimeIndex) else "@x"),
                ("Price", "@y{0.2f}")
            ],
            formatters={"@x": "datetime"} if isinstance(price.index, pd.DatetimeIndex) else {},
            mode="vline"
        )
        fig.add_tools(hover)
        
        # Modificar gráfico
        fig.xaxis.axis_label = 'Fecha'
        fig.yaxis.axis_label = 'Precio'
        fig.title.text_color = "#333333"
        fig.xaxis.axis_label_text_color = "#333333"
        fig.yaxis.axis_label_text_color = "#333333"
        fig.xaxis.major_label_text_color = "#333333"
        fig.yaxis.major_label_text_color = "#333333"
        fig.yaxis.formatter.use_scientific = False
        fig.xaxis.axis_line_color = "#e0e6eb"
        fig.yaxis.axis_line_color = "#e0e6eb"
        fig.xaxis.major_tick_line_color = "#a5a5a5"
        fig.yaxis.major_tick_line_color = "#c7c7c7"
        fig.xaxis.minor_tick_line_color = "#c7c7c7"
        fig.yaxis.minor_tick_line_color = "#c7c7c7"
        fig.xgrid.grid_line_color = "#c7c7c7"
        fig.ygrid.grid_line_color = "#c7c7c7"
        fig.xgrid.grid_line_alpha = 0.5
        fig.ygrid.grid_line_alpha = 0.5
        fig.xgrid.minor_grid_line_color = "#e0e6eb"
        fig.ygrid.minor_grid_line_color = "#e0e6eb"
        fig.xgrid.minor_grid_line_alpha = 0.3
        fig.ygrid.minor_grid_line_alpha = 0.3
        
        # Crear leyenda
        legend = Legend(items=legend_items)
        fig.add_layout(legend, "right")
        fig.legend.background_fill_color = "#ced5db"
        fig.legend.background_fill_alpha = 0
        fig.legend.click_policy = "hide"
        
        if output.lower() == "notebook":
            output_notebook()
        
        # Mostrar figura
        show(fig)

class OPV_MA:
    """
    Clase para calcular el HMA con una longitud de período variable.
    """

    def __init__(self, 
                    data: Union[pd.Series, np.array],
                    ma: Overlay, 
                    length: range = range(10, 50, 1),
                    method: str = "cross_over", 
                    direction: str = "long",
                    opt_window: int = None,
                    opt_interval: int = 10,
                    ):
        ""
        ""
        self.data = data
        self.MA = ma
        self.method = method
        self.length = length
        self.direction = direction
        self.opt_window = opt_window
        self.interval = opt_interval

        self.name = f"OP-{ma.__name__}({self.length})"

    @property
    def values(self):
        return self.get_result()

    # def get_result(self):
    #     # Ventana de datos para optimizar la media movil
    #     window = self.length[-1] if not self.window else self.window

    #     # incializar array vacio
    #     ma_empty = np.full(len(self.data), np.nan)

    #     # indica el ultimo indice en el que ya se calculo la media movil
    #     idx_confirmed = 0

    #     for i in range(len(self.data) + 1):
    #         # ignorar si no hay suficientes datos o si no estamos en el intervalo 
    #         if i < window or i % self.interval != 0:
    #             continue

    #         df_filtered = self.data[:i] # datos filtrados

    #         # optimizar la media movil y extraer el mejor valor
    #         length_ptimization = optimize(indicator=self.MA, data=df_filtered[-window:], method=self.method, length= self.length).length
    #         best_length = length_ptimization.best_value if self.direction == "long" else length_ptimization.worst_value

    #         # actualizar la media movil vacia con el valor de la media movil optimizada
    #         ma_empty[idx_confirmed:i] = self.MA(df_filtered, best_length).values[idx_confirmed:i]
    #         idx_confirmed = i 
            
    #     return ma_empty

    def get_result(self):
        # Ventana de datos para optimizar la media movil
        window = self.length[-1] if not self.opt_window else self.opt_window

        # incializar array vacio
        ma_empty = np.full(len(self.data), np.nan)

        # Valor actual de longitud
        current_length = None

        for i in range(window, len(self.data) + 1):
            in_interval = i % self.interval == 0
            
            # Verificar si estamos en un intervalo de optimizacion
            if in_interval: # Optimizar la longitud del indicador

                # window = current_length - 1 if current_length else self.length[-1]
                # Datos históricos hasta el punto actual
                df_filtered = self.data[:i]

                # Optimizar la media movil con datos históricos
                length_optimization = optimize(indicator=self.MA, data=df_filtered[-window:], 
                                            method=self.method, length=self.length).length
                
                # Seleccionar el mejor valor según la dirección
                best_length = length_optimization.best_value if self.direction == "long" else length_optimization.worst_value

                # Actualizar el valor actual de longitud
                current_length = best_length
                
            if current_length:
                    # Calcular solo el punto actual con los parámetros actuales
                    ma_empty[i-1] = self.MA(self.data[:i], current_length).values[-1]
        
        return ma_empty
        
class RSI(_Indicator):
    """Clase para calcular el Índice de Fuerza Relativa (RSI)."""

    class Evaluation:
        """
        Esta clase evalúa el rendimiento de una estrategia de trading utilizando RSI.
        Compra cuando el RSI está por debajo del nivel de sobrevendido y
        vende cuando está por encima del nivel de sobrecomprado.
        """

        def __init__(self, rsi: "RSI", oversold=30, overbought=70, inverted=False):
            """
            Inicializa la clase con los datos del RSI y los niveles de sobrecompra/sobreventa.

            Parámetros:
            rsi (RSI): Objeto RSI que contiene los datos calculados.
            oversold (int): Nivel de sobreventa (por defecto 30).
            overbought (int): Nivel de sobrecompra (por defecto 70).
            inverted (bool): Si es True, invierte las señales de compra y venta.
            """
            # Guardamos los datos del RSI
            self.rsi_data = pd.DataFrame(pd.Series(rsi.values, name=f"RSI_{rsi.__length}"))
            self.price_series = rsi.price

            # Niveles de sobrecompra y sobreventa
            self.oversold = oversold
            self.overbought = overbought

            # Si se invierten las posiciones, cambiaremos las señales
            self.inverted_positions = inverted

            # Crear un DataFrame con los datos del RSI y el precio
            self.df = pd.concat([self.rsi_data, self.price_series], axis=1)

            # Eliminar valores nulos
            self.df.dropna(inplace=True)

            # Calcular el porcentaje de cambio de los precios
            self.price_pct = self.df["Close"].pct_change(1)
            # Iniciar el valor en 0 en price_pct
            self.price_pct[self.price_pct.index[0]] = 0
            self.returns = pd.DataFrame()  # Guardar los rendimientos
            self.positions = pd.DataFrame()  # Guardar las posiciones de compra/venta

            # Evaluar la estrategia
            self.__evaluate_rsi_strategy()

        def __evaluate_rsi_strategy(self):
            """
            Evalúa la estrategia de trading basada en RSI:
            - Compra cuando el RSI está por debajo del nivel de sobreventa
            - Vende cuando el RSI está por encima del nivel de sobrecompra
            """
            rsi_tags = self.rsi_data.columns  # Los nombres de las columnas RSI

            # Iteramos sobre cada columna RSI
            for rsi_tag in rsi_tags:
                rsi_series = self.df[rsi_tag].copy()  # Copiamos la serie del RSI
                price_series = self.df["Close"].copy()  # Copiamos la serie de precios

                # Inicializamos posiciones en 0 (sin posición)
                positions = pd.Series(0, index=price_series.index)
                
                # Condición de compra: RSI por debajo del nivel de sobreventa
                buy_condition = rsi_series < self.oversold
                # Condición de venta: RSI por encima del nivel de sobrecompra
                sell_condition = rsi_series > self.overbought
                
                # Asignamos 1 para compra, -1 para venta
                positions[buy_condition] = 1
                positions[sell_condition] = -1
                
                # Si las posiciones están invertidas, cambiamos la condición
                if self.inverted_positions:
                    positions = -positions

                # Calculamos los rendimientos de la estrategia con un desfase de 1 día
                returns = self.price_pct * positions.shift(1)
                returns.fillna(0, inplace=True)  # Rellenar los valores nulos con 0

                # Almacenar los resultados
                self.returns[f"{rsi_tag} returns"] = returns
                self.positions[f"{rsi_tag} positions"] = positions

        def plot(self, view_returns=True, height=350, width=800):
            """
            Crea gráficos de los rendimientos acumulados y las posiciones de compra/venta con Bokeh.

            Parámetros:
            view_returns (bool): Si es True, muestra el gráfico de rendimientos acumulados.
            height (int): Altura de cada gráfico.
            width (int): Ancho del gráfico.
            """
            plots = []  # Lista de gráficos para organizar en un layout

            # 📈 Gráfico de rendimientos acumulados
            if view_returns:
                fig_returns = figure(
                    title="Cumulative Returns", 
                    width=width, 
                    height=height, 
                    x_axis_type="datetime", 
                    tools="pan,wheel_zoom,box_zoom,reset,save",
                    background_fill_color="#e0e6eb",  # Fondo interior
                    border_fill_color="#e0e6eb",     # Fondo exterior         
                    )
               
                colors = color_way(len(self.returns))
                legend_items = []  # Lista para guardar las leyendas manuales

                # Graficar rendimientos de los precios
                line_price = fig_returns.line(
                    self.price_pct.index, self.price_pct.cumsum() * 100, 
                    line_width=1.6, color="#333333"
                    )
                legend_items.append(LegendItem(label="Price Returns", renderers=[line_price]))
       
                # Graficar rendimientos de cada RSI
                for i, rsi_ret_tag in enumerate(self.returns.columns):
                    line = fig_returns.line(
                        self.returns.index, self.returns[rsi_ret_tag].cumsum() * 100, 
                        color=colors[i], line_width=1.6
                        )
                    legend_items.append(LegendItem(label=rsi_ret_tag, renderers=[line]))

                # Modificar Grafico
                fig_returns.xaxis.axis_label = 'Fecha'
                fig_returns.yaxis.axis_label = 'Precio'
                fig_returns.title.text_color = "#333333"
                fig_returns.xaxis.axis_label_text_color = "#333333"
                fig_returns.yaxis.axis_label_text_color = "#333333"
                fig_returns.xaxis.major_label_text_color = "#333333"
                fig_returns.yaxis.major_label_text_color = "#333333"
                # Evitar mostrar en connotacion cientifica los numeros del eje y
                fig_returns.yaxis.formatter.use_scientific = False
                # Cambiar el color de las líneas de los ejes y ticks a azul
                fig_returns.xaxis.axis_line_color = "#e0e6eb"
                fig_returns.yaxis.axis_line_color = "#e0e6eb"
                fig_returns.xaxis.major_tick_line_color = "#a5a5a5"
                fig_returns.yaxis.major_tick_line_color = "#c7c7c7"
                fig_returns.xaxis.minor_tick_line_color = "#c7c7c7"
                fig_returns.yaxis.minor_tick_line_color = "#c7c7c7"
                # Cambiar el color del grid
                fig_returns.xgrid.grid_line_color = "#c7c7c7"
                fig_returns.ygrid.grid_line_color = "#c7c7c7"
                fig_returns.xgrid.grid_line_alpha = 0.5  # Opcional: ajustar la transparencia
                fig_returns.ygrid.grid_line_alpha = 0.5  # Opcional: ajustar la transparencia
                fig_returns.xgrid.minor_grid_line_color = "#e0e6eb"  # Color del grid menor
                fig_returns.ygrid.minor_grid_line_color = "#e0e6eb"  # Color del grid menor
                fig_returns.xgrid.minor_grid_line_alpha = 0.3  # Transparencia del grid menor
                fig_returns.ygrid.minor_grid_line_alpha = 0.3  # Transparencia del grid menor
                
                # Crear la leyenda manualmente con todas las líneas
                legend = Legend(items=legend_items)
                fig_returns.add_layout(legend, "right")  # Ubicar la leyenda a la derecha del gráfico
                # Modificar la leyenda
                fig_returns.legend.background_fill_color = "#ced5db"  # Cambia el color del fondo
                fig_returns.legend.background_fill_alpha = 0  # Ajusta la transparencia del fondo
                fig_returns.legend.click_policy = "hide"

                # Agregar línea en y=0
                zero_line = Span(
                    location=0, 
                    dimension='width', 
                    line_color="#9297a3", 
                    line_width=0.7, 
                    line_dash="dashed"
                    )
                fig_returns.add_layout(zero_line)

                # Agregar herramienta Hover
                hover = HoverTool(
                    tooltips=[("Date", "@x{%F %H:%M}"),
                            ("Returns", "@y{0.3f}%")],
                    formatters={"@x": "datetime"},
                    mode="vline"
                    )
                fig_returns.add_tools(hover)

                plots.append(fig_returns)

            # Mostrar gráficos
            show(column(*plots))

    def __init__(self, data: Union[pd.Series, np.array, pd.DataFrame], length: int = 14):
        """
            Inicializa el indicador RSI.
            
            Parámetros:
            price: Serie de precios.
            period: Período para el cálculo del RSI (por defecto 14).
        """
        # Asegurar que el precio sea un pd.Series 
        if isinstance(data, pd.DataFrame):
            try:
                self.price = data.Close
            except:
                self.price = data.squeeze()
            self.index = self.price.index
        elif isinstance(data, pd.Series):
            self.price = data
            self.index = data.index
        elif isinstance(data, np.ndarray):
            self.price = pd.Series(data)
            self.index = range(len(data))
        else:
            raise ValueError("El precio debe ser un pd.Series, np.array o pd.DataFrame")
            
        self.length = length
        self.name = f"RSI {self.length}"

    @property
    def values(self):
        return self._get_result()

    def _get_result(self):
        return self._rsi()

    def _rsi(self):
        """
        Calcula el Índice de Fuerza Relativa (RSI).
        """
        # Calcular los cambios en el precio
        delta = self.price.diff()
        
        # Separar los cambios positivos y negativos
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calcular la media móvil exponencial de ganancias y pérdidas
        avg_gain = gain.rolling(window=self.length).mean()
        avg_loss = loss.rolling(window=self.length).mean()
        
        # Calcular el índice de fuerza relativa (RSI)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.values

    def evaluate(self, oversold=30, overbought=70, inverted=False):
        """
        Evalúa la estrategia de trading basada en RSI.
        
        Parámetros:
        oversold: Nivel de sobreventa (por defecto 30).
        overbought: Nivel de sobrecompra (por defecto 70).
        inverted: Si es True, invierte las señales de compra y venta.
        """
        return self.Evaluation(self, oversold=oversold, overbought=overbought, inverted=inverted)

    def plot(self, height=350, width=800, output="notebook"):
        """
            Grafica el RSI junto con los niveles de sobrecompra y sobreventa.
            
            Parámetros:
            height: Altura del gráfico.
            width: Ancho del gráfico.
            output: Tipo de salida (por defecto "notebook").
        """
        # Crear DataFrame con los datos del RSI
        rsi_data = pd.DataFrame(self.values, index=self.index, columns=[f"RSI_{self.length}"])
        
        # Colores para el RSI
        colors = color_way(1)
        
        # Crear figura
        fig = figure(
            title=f"RSI ({self.length})", 
            x_axis_type="datetime", 
            height=height, 
            width=width, 
            tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"],
            background_fill_color="#e0e6eb",  # Fondo interior
            border_fill_color="#e0e6eb",     # Fondo exterior
        )
        
        # Agregar línea del RSI
        rsi_line = fig.line(
            rsi_data.index, 
            rsi_data[f"RSI_{self.length}"], 
            color=colors[0], 
            line_width=2
        )
        
        # Agregar líneas de sobrecompra (70) y sobreventa (30)
        overbought_line = Span(
            location=70, 
            dimension='width', 
            line_color="red", 
            line_width=1, 
            line_dash="dashed"
        )
        oversold_line = Span(
            location=30, 
            dimension='width', 
            line_color="green", 
            line_width=1, 
            line_dash="dashed"
        )
        midline = Span(
            location=50, 
            dimension='width', 
            line_color="gray", 
            line_width=0.5
        )
        
        fig.add_layout(overbought_line)
        fig.add_layout(oversold_line)
        fig.add_layout(midline)
        
        # Agregar herramienta Hover
        hover = HoverTool(
            tooltips=[("Date", "@x{%F %H:%M}"),
                    ("RSI", "@y{0.2f}")],
            formatters={"@x": "datetime"},
            mode="vline"
        )
        fig.add_tools(hover)
        
        # Modificar Grafico
        fig.xaxis.axis_label = 'Fecha'
        fig.yaxis.axis_label = 'RSI'
        fig.title.text_color = "#333333"
        fig.xaxis.axis_label_text_color = "#333333"
        fig.yaxis.axis_label_text_color = "#333333"
        fig.xaxis.major_label_text_color = "#333333"
        fig.yaxis.major_label_text_color = "#333333"
        # Evitar mostrar en connotacion cientifica los numeros del eje y
        fig.yaxis.formatter.use_scientific = False
        # Cambiar el color de las líneas de los ejes y ticks
        fig.xaxis.axis_line_color = "#e0e6eb"
        fig.yaxis.axis_line_color = "#e0e6eb"
        fig.xaxis.major_tick_line_color = "#a5a5a5"
        fig.yaxis.major_tick_line_color = "#c7c7c7"
        fig.xaxis.minor_tick_line_color = "#c7c7c7"
        fig.yaxis.minor_tick_line_color = "#c7c7c7"
        # Cambiar el color del grid
        fig.xgrid.grid_line_color = "#c7c7c7"
        fig.ygrid.grid_line_color = "#c7c7c7"
        fig.xgrid.grid_line_alpha = 0.5
        fig.ygrid.grid_line_alpha = 0.5
        fig.xgrid.minor_grid_line_color = "#e0e6eb"
        fig.ygrid.minor_grid_line_color = "#e0e6eb"
        fig.xgrid.minor_grid_line_alpha = 0.3
        fig.ygrid.minor_grid_line_alpha = 0.3
        
        # Establecer el rango del eje y entre 0 y 100
        fig.y_range.start = 0
        fig.y_range.end = 100
        
        # Crear leyenda
        legend = Legend(items=[
            LegendItem(label=f"RSI ({self.length})", renderers=[rsi_line]),
        ])
        fig.add_layout(legend, "right")
        # Modificar la leyenda
        fig.legend.background_fill_color = "#ced5db"
        fig.legend.background_fill_alpha = 0
        
        if output.lower() == "notebook":
            output_notebook()
            
        # Mostrar figura
        show(fig)

    @classmethod
    def optimize(
        cls, 
        price: pd.Series, 
        periods_range=range(5, 30 + 1, 1),
        oversold_range=range(20, 40 + 1, 5),
        overbought_range=range(60, 80 + 1, 5),
        inverted=False, 
        info=False
    ):
        """
            Optimiza los parámetros del RSI para encontrar la mejor combinación.
            
            Parámetros:
            price: Serie de precios.
            periods_range: Rango de períodos a probar.
            oversold_range: Rango de niveles de sobreventa a probar.
            overbought_range: Rango de niveles de sobrecompra a probar.
            inverted: Si es True, invierte las señales de compra y venta.
            info: Si es True, muestra información adicional.
        """
        max_returns = -float("inf")
        best_params = None

        for period in periods_range:
            for oversold in oversold_range:
                for overbought in overbought_range:
                    if oversold >= overbought:
                        continue  # Saltamos combinaciones inválidas
                        
                    # Crear el objeto RSI
                    rsi_obj = cls(price, period)
                    
                    # Evaluar la estrategia
                    evaluation = rsi_obj.evaluate(
                        oversold=oversold, 
                        overbought=overbought, 
                        inverted=inverted
                    )
                    
                    # Obtener los rendimientos totales
                    total_returns = round(evaluation.returns.cumsum().iloc[-1], 9).values[0]
                    
                    # Actualizar los mejores parámetros si encontramos mejores rendimientos
                    if total_returns > max_returns:
                        max_returns = total_returns
                        best_params = {
                            "period": period,
                            "oversold": oversold,
                            "overbought": overbought
                        }
        
        if info:
            print(f"Mejores parámetros para RSI:")
            print(f"Período: {best_params['period']}")
            print(f"Nivel de sobreventa: {best_params['oversold']}")
            print(f"Nivel de sobrecompra: {best_params['overbought']}")
            print(f"Rendimientos: {max_returns}")
        
        return best_params
# PIVOTS
class PIVOT:
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._index = df.index
        # Inicializar array de tipos de pivot
        self.pivot_type = np.full(len(df), '', dtype='U1')
        self.values = self._get_result()

    def __len__(self):
        return len(self.values)

    def _get_result(self) -> np.array:
        """
        Calcula los puntos pivote confirmados usando el nuevo algoritmo de confirmación
        También actualiza self.pivot_type durante el proceso
        """
        df = self._df
        
        # Validación de entrada
        required_cols = ['High', 'Low']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame debe contener las columnas: {required_cols}")
        
        if len(df) < 5:
            raise ValueError("DataFrame debe tener al menos 5 filas para calcular pivotes")
        
        high_vals = df['High'].values
        low_vals = df['Low'].values
        base_pivot_type = self._base_pivot_type()
        
        # Array para almacenar pivotes confirmados
        confirmed_pivots = np.full(len(df), np.nan)
        
        # Variables de estado
        last_confirmed_type = None  # Tipo del último pivot confirmado
        confirmed_indices = []      # Lista de índices confirmados
        
        # Iterar desde el índice 2 para tener suficiente contexto
        for i in range(2, len(df) - 1):  # -1 para poder mirar hacia adelante
            current_pivot_type = base_pivot_type[i]
            
            # Si no hay pivot en esta posición, continuar
            if current_pivot_type == '':
                continue
                
            # Determinar qué tipos de pivot podemos considerar
            possible_types = []
            if current_pivot_type == 'B':
                possible_types = ['H', 'L']
            else:
                possible_types = [current_pivot_type]
            
            # Para cada tipo posible, verificar si podemos confirmarlo
            for pivot_type in possible_types:
                # Verificar si estamos esperando el tipo contrario
                if last_confirmed_type is not None:
                    if ((last_confirmed_type == 'H' and pivot_type != 'L') or 
                        (last_confirmed_type == 'L' and pivot_type != 'H')):
                        continue  # No es el tipo contrario que esperamos
                
                # Intentar confirmar el pivot
                if self._try_confirm_pivot(i, pivot_type, high_vals, low_vals, base_pivot_type):
                    # Pivot confirmado - guardar valor y tipo
                    if pivot_type == 'H':
                        confirmed_pivots[i] = high_vals[i]
                        self.pivot_type[i] = 'H'  # Guardar tipo confirmado
                    else:  # pivot_type == 'L'
                        confirmed_pivots[i] = low_vals[i]
                        self.pivot_type[i] = 'L'  # Guardar tipo confirmado
                    
                    confirmed_indices.append(i)
                    last_confirmed_type = pivot_type
                    break  # Solo confirmamos uno por posición
        
        return confirmed_pivots
    
    def _try_confirm_pivot(self, pivot_idx, pivot_type, high_vals, low_vals, arr_pivot_type):
        """
        Intenta confirmar un pivot mirando hacia adelante
        """
        pivot_value = high_vals[pivot_idx] if pivot_type == 'H' else low_vals[pivot_idx]
        
        # Iterar hacia adelante para confirmar
        j = pivot_idx + 1
        while j < len(high_vals):
            current_high = high_vals[j]
            current_low = low_vals[j]
            current_pivot_type = arr_pivot_type[j]
            
            # Verificar si el pivot se confirma (se supera en dirección contraria)
            if pivot_type == 'H':
                if current_low < low_vals[pivot_idx]:
                    return True  # Pivot high confirmado
            else:  # pivot_type == 'L'
                if current_high > high_vals[pivot_idx]:
                    return True  # Pivot low confirmado
            
            # Verificar si encontramos otro pivot antes de confirmarse
            if current_pivot_type != '':
                # Hay otro pivot candidato
                other_possible_types = []
                if current_pivot_type == 'B':
                    other_possible_types = ['H', 'L']
                else:
                    other_possible_types = [current_pivot_type]
                
                # Si el nuevo pivot es del mismo tipo, el actual no se puede confirmar
                if pivot_type in other_possible_types:
                    # Verificar si el nuevo pivot es "mejor" (más extremo)
                    if pivot_type == 'H':
                        if current_high > pivot_value:
                            return False  # Hay un high más alto, no se confirma
                    else:  # pivot_type == 'L'
                        if current_low < pivot_value:
                            return False  # Hay un low más bajo, no se confirma
            
            j += 1
        
        # Si llegamos al final sin confirmación, no se confirma
        return False

    def _base_pivot_type(self) -> np.ndarray:
        """
        Identifica los tipos de pivots: 'H' (High), 'L' (Low), 'B' (Both), o '' (ninguno)
        Estos son los pivots candidatos, no los confirmados
        """
        high_vals, low_vals = self._df.High.values, self._df.Low.values
        
        # Pre-calcular los shifts para mejor rendimiento
        high_shifts = {
            -2: np.concatenate([np.full(2, np.nan), high_vals[:-2]]),
            -1: np.concatenate([np.full(1, np.nan), high_vals[:-1]]),
            1: np.concatenate([high_vals[1:], np.full(1, np.nan)]),
            2: np.concatenate([high_vals[2:], np.full(2, np.nan)])
        }
        
        low_shifts = {
            -2: np.concatenate([np.full(2, np.nan), low_vals[:-2]]),
            -1: np.concatenate([np.full(1, np.nan), low_vals[:-1]]),
            1: np.concatenate([low_vals[1:], np.full(1, np.nan)]),
            2: np.concatenate([low_vals[2:], np.full(2, np.nan)])
        }
        
        # Condiciones vectorizadas para pivotes high
        condition_high = (
            (high_vals >= high_shifts[-1]) &
            (high_vals >= high_shifts[-2]) &
            (high_vals > high_shifts[1]) &
            (high_vals > high_shifts[2]) &
            ~np.isnan(high_shifts[-1]) &
            ~np.isnan(high_shifts[-2]) &
            ~np.isnan(high_shifts[1]) &
            ~np.isnan(high_shifts[2])
        )
        
        # Condiciones vectorizadas para pivotes low
        condition_low = (
            (low_vals <= low_shifts[-1]) &
            (low_vals <= low_shifts[-2]) &
            (low_vals < low_shifts[1]) &
            (low_vals < low_shifts[2]) &
            ~np.isnan(low_shifts[-1]) &
            ~np.isnan(low_shifts[-2]) &
            ~np.isnan(low_shifts[1]) &
            ~np.isnan(low_shifts[2])
        )
        
        # Crear array de tipos de pivotes
        pivot_type = np.full(len(self._df), '', dtype='U1')
        
        # Asignar tipos según las condiciones
        both_condition = condition_high & condition_low
        pivot_type[both_condition] = 'B'
        pivot_type[condition_high & ~both_condition] = 'H'
        pivot_type[condition_low & ~both_condition] = 'L'

        return pivot_type

    @property
    def index(self):
        return self._index

    @property
    def df(self):
        return self._df

class HH_LL:
    def __init__(self, data: Sequence):
        # Si se proporcionó un df calcular los pivots
        if isinstance(data, PIVOT):
            self.pivots = data
        elif isinstance(data, pd.DataFrame): 
            assert all(('High' in data.columns, 'Low' in data.columns)), "El DataFrame debe contener columnas 'High' y 'Low'"
            self.pivots = PIVOT(data)
        else:
            raise ValueError("No se reconoce el tipo de dato especificado")
        
        # Inicializar arrays
        self.pivot_type = np.full(len(data), '', dtype='U4') # Almacenará los tipos "HH", "LH", "HL", "LL"
        self.values = self._get_result()

    def _get_result(self):
        """
            Identifica Higher Highs, Lower Highs, Higher Lows y Lower Lows basado en puntos pivot.
            
            Retorna:
            --------
            tuple : (HH_array, LH_array, HL_array, LL_array)
            Cada array tiene la misma longitud que el DataFrame original,
            con valores en las posiciones correspondientes y NaN en el resto.
            - HH_array: Higher Highs (máximos más altos)
            - LH_array: Lower Highs (máximos más bajos) 
            - HL_array: Higher Lows (mínimos más altos)
            - LL_array: Lower Lows (mínimos más bajos)
        """
        
        # Obtener datos de pivots
        pivot = self.pivots
        pivot_values = pivot._values  # [nan,1,nan,2,nan,1.5,3, nan, nan,nan,2,nan]
        pivot_type = pivot._base_pivot_type()  # [nan,"L",nan,"H",nan,"L","H",nan,nan,nan,"L",nan]
        
        # Inicializar arrays con NaN del mismo tamaño que el DataFrame
        data_length = len(pivot)
        HH_array = np.full(data_length, np.nan)
        LH_array = np.full(data_length, np.nan)
        HL_array = np.full(data_length, np.nan)
        LL_array = np.full(data_length, np.nan)
        
        # NO reiniciar pivot_type, mantener el array original
        # self.pivot_type ya está inicializado como array de numpy
        
        # Filtrar solo los pivots válidos (no NaN)
        valid_mask = ~pd.isna(pivot_values)
        valid_indices = np.where(valid_mask)[0]
        valid_values = np.array(pivot_values)[valid_mask]
        valid_types = np.array(pivot_type)[valid_mask]
        
        # Separar highs y lows con sus posiciones originales
        highs_mask = valid_types == 'H'
        lows_mask = valid_types == 'L'
        
        high_values = valid_values[highs_mask]
        high_positions = valid_indices[highs_mask]
        
        low_values = valid_values[lows_mask]
        low_positions = valid_indices[lows_mask]
        
        # Procesar Higher Highs y Lower Highs
        if len(high_values) > 1:
            for i in range(1, len(high_values)):
                current_high = high_values[i]
                previous_high = high_values[i-1]
                position = high_positions[i]
                
                if current_high > previous_high:
                    # Higher High
                    HH_array[position] = current_high
                    self.pivot_type[position] = "HH"
                else:
                    # Lower High
                    LH_array[position] = current_high
                    self.pivot_type[position] = "LH"
        
        # Procesar Higher Lows y Lower Lows
        if len(low_values) > 1:
            for i in range(1, len(low_values)):
                current_low = low_values[i]
                previous_low = low_values[i-1]
                position = low_positions[i]
                
                if current_low > previous_low:
                    # Higher Low
                    HL_array[position] = current_low
                    self.pivot_type[position] = "HL"
                else:
                    # Lower Low
                    LL_array[position] = current_low
                    self.pivot_type[position] = "LL"
        
        return np.array([HH_array, LH_array, HL_array, LL_array])    

class MAXH_MINL:
    def __init__(self, data: Union[PIVOT, pd.DataFrame]):
        # Si se proporcionó un df calcular los pivots
        if isinstance(data, PIVOT):
            self.pivots = data
        elif isinstance(data, pd.DataFrame): 
            assert all(('High' in data.columns, 'Low' in data.columns)), "El DataFrame debe contener columnas 'High' y 'Low'"
            self.pivots = PIVOT(data)
        else:
            raise ValueError("No se reconoce el tipo de dato especificado")
        
        # Inicializar arrays
        self.pivot_type = np.full(len(data), '', dtype='U4')
        self.values = self._get_result()
    
    def _get_result(self):
        """
        Retorna tupla (max_highs_values, min_lows_values)
        Con valores reales de pivots y NaN, actualizándose históricamente
        También actualiza self.pivot_type durante el proceso
        """
        pivot_values = self.pivots._values
        pivot_type = self.pivots._ptypes
        
        # Arrays de salida con NaN
        max_highs_values = np.full(len(pivot_values), np.nan)
        min_lows_values = np.full(len(pivot_values), np.nan)
        
        # Variables para tracking
        current_max_high = -np.inf
        current_min_low = np.inf
        
        # Procesar secuencialmente para mantener histórico
        for i in range(len(pivot_values)):
            if not np.isnan(pivot_values[i]):
                if pivot_type[i] == 'H':
                    # Nuevo pivot High
                    if pivot_values[i] > current_max_high:
                        current_max_high = pivot_values[i]
                        max_highs_values[i] = current_max_high
                        self.pivot_type[i] = 'MAXH'
                
                elif pivot_type[i] == 'L':
                    # Nuevo pivot Low
                    if pivot_values[i] < current_min_low:
                        current_min_low = pivot_values[i]
                        min_lows_values[i] = current_min_low
                        self.pivot_type[i] = 'MINL'
        
        return np.array([max_highs_values, min_lows_values]) 

class NBS:
    style = {
        "renders": {
            "scatter": {
                "size": [7, 5, 6],
                "mark": ["x", ".", "s"],
                "color": ["#FF2929FF", "#31BACFFF", "#FFD700FF"],  # Rojo, Azul, Dorado
            },
            "line": {
                "line_width": [4, 2, 3],
                "line_color": ["#FF29298A", "#31BACF8A", "#FFD7008A"],  # Transparencias
            },
        }
    }

    def __init__(self, data):
        """
            Inicializa el indicador NBS basado en una instancia de PIVOTS
            
            Args:
                pivots_instance: Instancia de la clase PIVOTS
        """
        # Si se proporcionó un df calcular los pivots
        if isinstance(data, PIVOT):
            self.pivots = data
            self.df = self.pivots.df
            self._index = self.df.index
        elif isinstance(data, pd.DataFrame): 
            assert all(('High' in data.columns, 'Low' in data.columns)), "El DataFrame debe contener columnas 'High' y 'Low'"
            self.pivots = PIVOT(data)
            self.df = data
            self._index = data.index
        elif isinstance(data, DataOHLC):
            data = data.df
            assert all(('High' in data.columns, 'Low' in data.columns)), "El DataFrame debe contener columnas 'High' y 'Low'"
            self.pivots = PIVOT(data)
            self.df = data
            self._index = data.index
        else:
            raise ValueError("No se reconoce el tipo de dato especificado")

        # self.pivots = data
        # self.df = data.df
        # self._index = data.index
        
        # Arrays para almacenar los resultados
        self._neu_values = np.full(len(self.df), np.nan)  # Neutralizer
        self._boo_values = np.full(len(self.df), np.nan)  # Booster
        self._shk_values = np.full(len(self.df), np.nan)  # Shaker
        
        # Atributo pivot_type completo (NEU_H, NEU_L, BOO_H, BOO_L, SHK_H, SHK_L, H, L)
        self.pivot_type = np.full(len(self.df), '', dtype='U5')
        
        # Variables de estado para pendientes (diccionarios con índice y valor de neutralización)
        self.pending_boo_h = {}  # {index: neutralization_value}
        self.pending_boo_l = {}  # {index: neutralization_value}
        self.pending_shk_h = {}  # {index: neutralization_value}
        self.pending_shk_l = {}  # {index: neutralization_value}
        
        self.current_neu_h = None  # Índice del NEU_H actual
        self.current_neu_l = None  # Índice del NEU_L actual
        self.current_shk_h = None  # Índice del SHK_H actual
        self.current_shk_l = None  # Índice del SHK_L actual
        
        # Arrays para coordenadas de líneas [neu_lines, boo_lines, shk_lines]
        self._neu_lines = []  # Lista de tuplas (start_idx, end_idx)
        self._boo_lines = []  # Lista de tuplas (start_idx, end_idx)
        self._shk_lines = []  # Lista de tuplas (start_idx, end_idx)
        
        # Procesar los pivots
        self._calculate_nbs()
        
        # Procesar neutralizaciones en tiempo real
        self._process_realtime_neutralizations()

    def _find_break_index(self, start_idx, target_value, is_high_break):
        """
        Encuentra el índice exacto donde se rompió un nivel
        
        Args:
            start_idx: Índice desde donde empezar a buscar
            target_value: Valor que debe ser superado/roto
            is_high_break: True si es ruptura de High, False si es ruptura de Low
            
        Returns:
            Índice donde ocurrió la ruptura, o None si no se encontró
        """
        for i in range(start_idx, len(self.df)):
            if is_high_break:
                if self.df.iloc[i]['High'] >= target_value:
                    return i
            else:
                if self.df.iloc[i]['Low'] <= target_value:
                    return i
        return None

    def _calculate_nbs(self):
        """
        Calcula los puntos BOO, NEU y SHK basándose en los pivots confirmados
        """
        pivot_values = self.pivots._values
        pivot_type = self.pivots._ptypes
        
        # Primero, asignar tipos de pivots ignorados
        for i in range(len(pivot_values)):
            if not np.isnan(pivot_values[i]):
                continue  # Este pivot será procesado
            elif pivot_type[i] in ['H', 'L']:
                self.pivot_type[i] = pivot_type[i]  # Guardar como 'H' o 'L' ignorado
        
        # Obtener índices de pivots válidos (no NaN)
        pivot_indices = np.where(~np.isnan(pivot_values))[0]
        
        if len(pivot_indices) < 2:
            return  # No hay suficientes pivots para procesar
        
        # Inicialización: primeros dos pivots
        self._initialize_first_pivots(pivot_indices)
        
        # Procesar el resto de pivots
        for i in range(2, len(pivot_indices)):
            current_idx = pivot_indices[i]
            current_type = pivot_type[current_idx]
            current_value = pivot_values[current_idx]
            
            if current_type == 'H':
                self._process_pivot_h(current_idx, current_value, pivot_indices, i)
            elif current_type == 'L':
                self._process_pivot_l(current_idx, current_value, pivot_indices, i)

    def _initialize_first_pivots(self, pivot_indices):
        """
        Inicializa los primeros dos pivots como BOO y NEU
        """
        first_idx = pivot_indices[0]
        second_idx = pivot_indices[1]
        
        first_type = self.pivots._ptypes[first_idx]
        second_type = self.pivots._ptypes[second_idx]
        
        first_value = self.pivots._values[first_idx]
        second_value = self.pivots._values[second_idx]
        
        # Primer pivot como BOO
        self._boo_values[first_idx] = first_value
        self.pivot_type[first_idx] = f'BOO_{first_type}'
        
        # Segundo pivot como NEU
        self._neu_values[second_idx] = second_value
        self.pivot_type[second_idx] = f'NEU_{second_type}'
        
        # Establecer el neutralizer actual
        if second_type == 'H':
            self.current_neu_h = second_idx
        else:  # second_type == 'L'
            self.current_neu_l = second_idx
        
        # Agregar a pendientes con valor de neutralización
        if first_type == 'H':
            self.pending_boo_h[first_idx] = first_value
        else:  # first_type == 'L'
            self.pending_boo_l[first_idx] = first_value

    def _convert_neu_to_shk(self, neu_idx, neu_type):
        """
        Convierte un NEU en SHK cuando se convierte también en BOO
        """
        # Obtener el valor del neutralizer
        neu_value = self._neu_values[neu_idx]
        
        # Limpiar de arrays de neutralizer
        self._neu_values[neu_idx] = np.nan
        
        # Agregar a arrays de shaker
        self._shk_values[neu_idx] = neu_value
        self.pivot_type[neu_idx] = f'SHK_{neu_type}'
        
        # Actualizar variables de estado
        if neu_type == 'H':
            self.current_neu_h = None
            self.current_shk_h = neu_idx
            self.pending_shk_h[neu_idx] = neu_value
        else:  # neu_type == 'L'
            self.current_neu_l = None
            self.current_shk_l = neu_idx
            self.pending_shk_l[neu_idx] = neu_value

    def _process_pivot_h(self, current_idx, current_value, pivot_indices, position):
        """
        Procesa un pivot H para determinar si es NEU_H, BOO o se ignora
        """
        # Verificar si hay BOO_H pendientes y si este H los supera
        neutralized_boo_indices = []
        for boo_idx, boo_value in self.pending_boo_h.items():
            if current_value >= boo_value:
                neutralized_boo_indices.append(boo_idx)
        
        # Verificar si hay SHK_H pendientes y si este H los supera
        neutralized_shk_indices = []
        for shk_idx, shk_value in self.pending_shk_h.items():
            if current_value >= shk_value:
                neutralized_shk_indices.append(shk_idx)
        
        # Verificar si supera al NEU_H actual
        supera_neu_actual = False
        if self.current_neu_h is not None:
            current_neu_value = self._neu_values[self.current_neu_h]
            if current_value >= current_neu_value:
                supera_neu_actual = True
        
        # Verificar si supera al SHK_H actual
        supera_shk_actual = False
        if self.current_shk_h is not None:
            current_shk_value = self._shk_values[self.current_shk_h]
            if current_value >= current_shk_value:
                supera_shk_actual = True
        
        # Se convierte en neutralizer si supera cualquier pendiente o actual
        if neutralized_boo_indices or neutralized_shk_indices or supera_neu_actual or supera_shk_actual:
            # Crear líneas para los neutralizados - Usar current_idx como punto de ruptura
            for boo_idx in neutralized_boo_indices:
                self._boo_lines.append((boo_idx, current_idx))
                del self.pending_boo_h[boo_idx]
            
            for shk_idx in neutralized_shk_indices:
                self._shk_lines.append((shk_idx, current_idx))
                del self.pending_shk_h[shk_idx]
            
            if supera_neu_actual:
                self._neu_lines.append((self.current_neu_h, current_idx))
            
            if supera_shk_actual:
                self._shk_lines.append((self.current_shk_h, current_idx))
            
            # Este H se convierte en NEU_H
            self._neu_values[current_idx] = current_value
            self.pivot_type[current_idx] = 'NEU_H'
            
            # Actualizar neutralizers actuales
            self.current_neu_h = current_idx
            if supera_shk_actual:
                self.current_shk_h = None
            
            # El pivot L anterior se convierte en BOO_L
            if position > 0:
                prev_idx = pivot_indices[position - 1]
                prev_type = self.pivots._ptypes[prev_idx]
                prev_value = self.pivots._values[prev_idx]
                
                if prev_type == 'L':
                    # Verificar si el pivot anterior ya es un neutralizer
                    if self.pivot_type[prev_idx].startswith('NEU'):
                        # Convertir NEU_L a SHK_L
                        self._convert_neu_to_shk(prev_idx, 'L')
                    else:
                        # Convertir a BOO_L normal
                        self._boo_values[prev_idx] = prev_value
                        self.pivot_type[prev_idx] = 'BOO_L'
                        self.pending_boo_l[prev_idx] = prev_value
        else:
            # Si no supera nada, se marca como pivot ignorado
            self.pivot_type[current_idx] = 'H'

    def _process_pivot_l(self, current_idx, current_value, pivot_indices, position):
        """
        Procesa un pivot L para determinar si es NEU_L, BOO o se ignora
        """
        # Verificar si hay BOO_L pendientes y si este L es inferior
        neutralized_boo_indices = []
        for boo_idx, boo_value in self.pending_boo_l.items():
            if current_value <= boo_value:
                neutralized_boo_indices.append(boo_idx)
        
        # Verificar si hay SHK_L pendientes y si este L es inferior
        neutralized_shk_indices = []
        for shk_idx, shk_value in self.pending_shk_l.items():
            if current_value <= shk_value:
                neutralized_shk_indices.append(shk_idx)
        
        # Verificar si es inferior al NEU_L actual
        supera_neu_actual = False
        if self.current_neu_l is not None:
            current_neu_value = self._neu_values[self.current_neu_l]
            if current_value <= current_neu_value:
                supera_neu_actual = True
        
        # Verificar si es inferior al SHK_L actual
        supera_shk_actual = False
        if self.current_shk_l is not None:
            current_shk_value = self._shk_values[self.current_shk_l]
            if current_value <= current_shk_value:
                supera_shk_actual = True
        
        # Se convierte en neutralizer si supera cualquier pendiente o actual
        if neutralized_boo_indices or neutralized_shk_indices or supera_neu_actual or supera_shk_actual:
            # Crear líneas para los neutralizados - Usar current_idx como punto de ruptura
            for boo_idx in neutralized_boo_indices:
                self._boo_lines.append((boo_idx, current_idx))
                del self.pending_boo_l[boo_idx]
            
            for shk_idx in neutralized_shk_indices:
                self._shk_lines.append((shk_idx, current_idx))
                del self.pending_shk_l[shk_idx]
            
            if supera_neu_actual:
                self._neu_lines.append((self.current_neu_l, current_idx))
            
            if supera_shk_actual:
                self._shk_lines.append((self.current_shk_l, current_idx))
            
            # Este L se convierte en NEU_L
            self._neu_values[current_idx] = current_value
            self.pivot_type[current_idx] = 'NEU_L'
            
            # Actualizar neutralizers actuales
            self.current_neu_l = current_idx
            if supera_shk_actual:
                self.current_shk_l = None
            
            # El pivot H anterior se convierte en BOO_H
            if position > 0:
                prev_idx = pivot_indices[position - 1]
                prev_type = self.pivots._ptypes[prev_idx]
                prev_value = self.pivots._values[prev_idx]
                
                if prev_type == 'H':
                    # Verificar si el pivot anterior ya es un neutralizer
                    if self.pivot_type[prev_idx].startswith('NEU'):
                        # Convertir NEU_H a SHK_H
                        self._convert_neu_to_shk(prev_idx, 'H')
                    else:
                        # Convertir a BOO_H normal
                        self._boo_values[prev_idx] = prev_value
                        self.pivot_type[prev_idx] = 'BOO_H'
                        self.pending_boo_h[prev_idx] = prev_value
        else:
            # Si no es inferior a nada, se marca como pivot ignorado
            self.pivot_type[current_idx] = 'L'

    def _process_realtime_neutralizations(self):
        """
        Procesa neutralizaciones en tiempo real basándose en OHLC actual
        """
        # Procesar todas las barras desde el último pivot confirmado
        pivot_indices = np.where(~np.isnan(self.pivots._values))[0]
        
        if len(pivot_indices) == 0:
            return
            
        # Empezar desde el último pivot confirmado
        start_idx = pivot_indices[-1] + 1 if len(pivot_indices) > 0 else 0
        
        for i in range(start_idx, len(self.df)):
            # Acceder a las columnas 'High' y 'Low' capitalizadas
            high = self.df.iloc[i]['High']
            low = self.df.iloc[i]['Low']
            
            # Verificar neutralizaciones de BOO_H pendientes
            neutralized_boo_h = []
            for boo_idx, boo_value in self.pending_boo_h.items():
                if high >= boo_value:
                    # Encontrar el índice exacto donde se rompió
                    break_idx = self._find_break_index(boo_idx + 1, boo_value, True)
                    if break_idx is not None:
                        self._boo_lines.append((boo_idx, break_idx))
                        neutralized_boo_h.append(boo_idx)
            
            for boo_idx in neutralized_boo_h:
                del self.pending_boo_h[boo_idx]
            
            # Verificar neutralizaciones de BOO_L pendientes
            neutralized_boo_l = []
            for boo_idx, boo_value in self.pending_boo_l.items():
                if low <= boo_value:
                    # Encontrar el índice exacto donde se rompió
                    break_idx = self._find_break_index(boo_idx + 1, boo_value, False)
                    if break_idx is not None:
                        self._boo_lines.append((boo_idx, break_idx))
                        neutralized_boo_l.append(boo_idx)
            
            for boo_idx in neutralized_boo_l:
                del self.pending_boo_l[boo_idx]
            
            # Verificar neutralizaciones de SHK_H pendientes
            neutralized_shk_h = []
            for shk_idx, shk_value in self.pending_shk_h.items():
                if high >= shk_value:
                    # Encontrar el índice exacto donde se rompió
                    break_idx = self._find_break_index(shk_idx + 1, shk_value, True)
                    if break_idx is not None:
                        self._shk_lines.append((shk_idx, break_idx))
                        neutralized_shk_h.append(shk_idx)
            
            for shk_idx in neutralized_shk_h:
                del self.pending_shk_h[shk_idx]
            
            # Verificar neutralizaciones de SHK_L pendientes
            neutralized_shk_l = []
            for shk_idx, shk_value in self.pending_shk_l.items():
                if low <= shk_value:
                    # Encontrar el índice exacto donde se rompió
                    break_idx = self._find_break_index(shk_idx + 1, shk_value, False)
                    if break_idx is not None:
                        self._shk_lines.append((shk_idx, break_idx))
                        neutralized_shk_l.append(shk_idx)
            
            for shk_idx in neutralized_shk_l:
                del self.pending_shk_l[shk_idx]

    @property
    def index(self):
        """Retorna el índice del DataFrame"""
        return self._index
    
    @property
    def values(self):
        """Retorna un array con tres arrays: [neutralizers, boosters, shakers]"""
        return np.array([self._neu_values, self._boo_values, self._shk_values])
    
    @property
    def neutralizers(self):
        """Retorna los valores de neutralizer"""
        return self._neu_values
    
    @property
    def boosters(self):
        """Retorna los valores de booster"""
        return self._boo_values
    
    @property
    def shakers(self):
        """Retorna los valores de shaker"""
        return self._shk_values
    
    @property
    def horizontal_lines_coordinates(self):
        """Retorna las coordenadas de líneas como array de 3 arrays"""
        return np.array([self._neu_lines, self._boo_lines, self._shk_lines], dtype=object)
    
    def get_pending_boosters(self):
        """
        Retorna información sobre los boosters pendientes (no neutralizados)
        """
        return {
            'pending_boo_h_indices': list(self.pending_boo_h.keys()),
            'pending_boo_l_indices': list(self.pending_boo_l.keys()),
            'pending_boo_h_values': list(self.pending_boo_h.values()),
            'pending_boo_l_values': list(self.pending_boo_l.values())
        }
    
    def get_pending_shakers(self):
        """
        Retorna información sobre los shakers pendientes (no neutralizados)
        """
        return {
            'pending_shk_h_indices': list(self.pending_shk_h.keys()),
            'pending_shk_l_indices': list(self.pending_shk_l.keys()),
            'pending_shk_h_values': list(self.pending_shk_h.values()),
            'pending_shk_l_values': list(self.pending_shk_l.values())
        }
    
    def get_current_neutralizers(self):
        """
        Retorna información sobre los neutralizers actuales
        """
        result = {}
        
        if self.current_neu_h is not None:
            result['current_neu_h'] = {
                'index': self.current_neu_h,
                'value': self._neu_values[self.current_neu_h]
            }
        else:
            result['current_neu_h'] = None
            
        if self.current_neu_l is not None:
            result['current_neu_l'] = {
                'index': self.current_neu_l,
                'value': self._neu_values[self.current_neu_l]
            }
        else:
            result['current_neu_l'] = None
            
        return result
    
    def get_current_shakers(self):
        """
        Retorna información sobre los shakers actuales
        """
        result = {}
        
        if self.current_shk_h is not None:
            result['current_shk_h'] = {
                'index': self.current_shk_h,
                'value': self._shk_values[self.current_shk_h]
            }
        else:
            result['current_shk_h'] = None
            
        if self.current_shk_l is not None:
            result['current_shk_l'] = {
                'index': self.current_shk_l,
                'value': self._shk_values[self.current_shk_l]
            }
        else:
            result['current_shk_l'] = None
            
        return result
    
    def __len__(self):
        return len(self._boo_values)


# NEWWWWW
#######################
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

#######################
# @jit(nopython=True, cache=True)
def sma(data, length):
    """
    Calcula la media móvil simple (SMA) de un período dado utilizando NumPy.
    """
    # Usamos np.convolve para calcular la SMA con un kernel de unos (1's)
    sma_values = np.convolve(data, np.ones(length), mode='valid') / length
    # Devolvemos una serie con los valores NaN al inicio y los valores de la SMA
    result = np.full(len(data), np.nan)  # Inicializamos con NaN
    # Colocamos los valores de la SMA a partir de la posición donde podemos calcularla
    result[length - 1:] = sma_values

    return result

# @jit(nopython=True, cache=True)
def ema(data, length):
    """
        Calcula la media móvil exponencial (EMA) de un período dado utilizando NumPy.
        Optimizado para Numba.
    """
    if isinstance(data, DataOHLC):
        data = data.Close
        
    if len(data) < length or length <= 0:
        # Devuelve un array de NaNs si no hay suficientes datos o el período no es válido
        return np.full(len(data), np.nan)

    alpha = 2.0 / (length + 1.0)
    ema_array = np.full(len(data), np.nan)

    # Calcula la SMA para el primer valor de la EMA
    # Numba maneja bien np.mean sobre arrays de NumPy en modo nopython
    if len(data) >= length:
        initial_sma = 0.0
        for i in range(length):
            initial_sma += data[i]
        ema_array[length - 1] = initial_sma / length

        # Calcula los valores subsiguientes de la EMA
        for i in range(length, len(data)):
            ema_array[i] = alpha * data[i] + (1.0 - alpha) * ema_array[i - 1]
    
    return ema_array

# @jit(nopython=True, cache=True)
def wma(data, length):
    """
    Calcula la media móvil ponderada (WMA) de un período dado utilizando NumPy.
    """
    weights = np.arange(1, length + 1)  # Pesos de la WMA (más grande al final)
    
    # Usamos la convolución para aplicar los pesos sobre los precios
    wma_values = np.convolve(data, weights[::-1], mode='valid') / weights.sum()
    
    # Creamos un array lleno de NaN y lo completamos con los valores calculados de la WMA
    result = np.full(len(data), np.nan)
    result[length - 1:] = wma_values
    
    return result

# TODO: @jit(nopython=True, cache=True)
def hma(data, length):
    """
        Calcula la Hull Moving Average (HMA) utilizando NumPy.
    """
    data = np.array(data)

    def _wma(data, length):
        weights = np.arange(1, length + 1)  
        wma_values = np.convolve(data, weights[::-1], mode='valid') / weights.sum()
        result = np.full(len(data), np.nan)
        result[length - 1:] = wma_values
        return result

    # Paso 1: WMA del período completo
    wma_full = _wma(data, length)
    
    # Paso 2: WMA de la mitad del período
    half_period = length // 2
    wma_half = _wma(data, half_period)
    
    # Paso 3: Restamos la WMA de mitad de periodo de la WMA del periodo completo
    diff_wma = 2 * wma_half - wma_full
    
    # Paso 4: Calculamos la WMA del resultado con el período sqrt(n)
    sqrt_length = int(np.sqrt(length))
    hma = _wma(diff_wma, sqrt_length)

    return hma

# TODO: @jit(nopython=True, cache=True)
def bbands(prices, length: int = 20, std_dev: float = 2.0):
    """
    Versión ultra-optimizada usando vectorización completa de NumPy.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    
    if n < length:
        nan_array = np.full(n, np.nan)
        return nan_array, nan_array, nan_array
    
    # Usar rolling views para cálculos vectorizados
    shape = (n - length + 1, length)
    strides = (prices.strides[0], prices.strides[0])
    rolling_windows = np.lib.stride_tricks.as_strided(
        prices, shape=shape, strides=strides
    )
    
    # Calcular media y desviación estándar de forma vectorizada
    means = np.mean(rolling_windows, axis=1)
    stds = np.std(rolling_windows, axis=1, ddof=0)
    
    # Pre-allocar resultados
    upper_band = np.full(n, np.nan)
    middle_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    # Asignar valores calculados
    upper_band[length-1:] = means + (std_dev * stds)
    middle_band[length-1:] = means
    lower_band[length-1:] = means - (std_dev * stds)
    
    return upper_band, middle_band, lower_band


#######################
def pivot(data):
    """
    Calculate confirmed pivot points based on price action.
    
    Args:
        data: Data object with 'High' and 'Low' attributes, or DataFrame with 'High' and 'Low' columns
    
    Returns:
        tuple: (pivot_types, confirmed_pivots) where:
            - pivot_types: np.array of pivot types ('H' for High, 'L' for Low, '' for no pivot)
            - confirmed_pivots: np.array of confirmed pivot values (NaN for no pivot)
    """
    
    # Handle different input types
    if hasattr(data, 'High') and hasattr(data, 'Low'):

        if len(data) < 5:
            raise ValueError("los datos almenos tener al menos 5 filas")

        high_vals = data.High
        low_vals = data.Low
    else:
        raise ValueError("El Input debe ser un pd.DataFrame o objeto Data con atributos: 'High', 'Low'")
    
    # Convert to numpy arrays for consistency
    high_vals = np.asarray(high_vals)
    low_vals = np.asarray(low_vals)
    
    # Get base pivot types
    def get_base_pivot_type(high_vals, low_vals):
        """Identify candidate pivots: 'H' (High), 'L' (Low), 'B' (Both), or '' (none)"""
        
        # Pre-calculate shifts for performance
        high_shifts = {
            -2: np.concatenate([np.full(2, np.nan), high_vals[:-2]]),
            -1: np.concatenate([np.full(1, np.nan), high_vals[:-1]]),
            1: np.concatenate([high_vals[1:], np.full(1, np.nan)]),
            2: np.concatenate([high_vals[2:], np.full(2, np.nan)])
        }
        
        low_shifts = {
            -2: np.concatenate([np.full(2, np.nan), low_vals[:-2]]),
            -1: np.concatenate([np.full(1, np.nan), low_vals[:-1]]),
            1: np.concatenate([low_vals[1:], np.full(1, np.nan)]),
            2: np.concatenate([low_vals[2:], np.full(2, np.nan)])
        }
        
        # Vectorized conditions for high pivots
        condition_high = (
            (high_vals >= high_shifts[-1]) &
            (high_vals >= high_shifts[-2]) &
            (high_vals > high_shifts[1]) &
            (high_vals > high_shifts[2]) &
            ~np.isnan(high_shifts[-1]) &
            ~np.isnan(high_shifts[-2]) &
            ~np.isnan(high_shifts[1]) &
            ~np.isnan(high_shifts[2])
        )
        
        # Vectorized conditions for low pivots
        condition_low = (
            (low_vals <= low_shifts[-1]) &
            (low_vals <= low_shifts[-2]) &
            (low_vals < low_shifts[1]) &
            (low_vals < low_shifts[2]) &
            ~np.isnan(low_shifts[-1]) &
            ~np.isnan(low_shifts[-2]) &
            ~np.isnan(low_shifts[1]) &
            ~np.isnan(low_shifts[2])
        )
        
        # Create pivot type array
        pivot_type = np.full(len(high_vals), '', dtype='U1')
        
        # Assign types based on conditions
        both_condition = condition_high & condition_low
        pivot_type[both_condition] = 'B'
        pivot_type[condition_high & ~both_condition] = 'H'
        pivot_type[condition_low & ~both_condition] = 'L'
        
        return pivot_type
    
    def try_confirm_pivot(pivot_idx, pivot_type, high_vals, low_vals, arr_pivot_type):
        """Try to confirm a pivot by looking forward"""
        pivot_value = high_vals[pivot_idx] if pivot_type == 'H' else low_vals[pivot_idx]
        
        # Iterate forward to confirm
        j = pivot_idx + 1
        while j < len(high_vals):
            current_high = high_vals[j]
            current_low = low_vals[j]
            current_pivot_type = arr_pivot_type[j]
            
            # Check if pivot is confirmed (surpassed in opposite direction)
            if pivot_type == 'H':
                if current_low < low_vals[pivot_idx]:
                    return True  # High pivot confirmed
            else:  # pivot_type == 'L'
                if current_high > high_vals[pivot_idx]:
                    return True  # Low pivot confirmed
            
            # Check if we find another pivot before confirmation
            if current_pivot_type != '':
                # There's another candidate pivot
                other_possible_types = []
                if current_pivot_type == 'B':
                    other_possible_types = ['H', 'L']
                else:
                    other_possible_types = [current_pivot_type]
                
                # If new pivot is same type, current cannot be confirmed
                if pivot_type in other_possible_types:
                    # Check if new pivot is "better" (more extreme)
                    if pivot_type == 'H':
                        if current_high > pivot_value:
                            return False  # There's a higher high, not confirmed
                    else:  # pivot_type == 'L'
                        if current_low < pivot_value:
                            return False  # There's a lower low, not confirmed
            
            j += 1
        
        # If we reach the end without confirmation, it's not confirmed
        return False
    
    # Get base pivot types
    base_pivot_type = get_base_pivot_type(high_vals, low_vals)
    
    # Arrays to store confirmed pivots and their types
    confirmed_pivots = np.full(len(high_vals), np.nan)
    pivot_types = np.full(len(high_vals), '', dtype='U1')
    
    # State variables
    last_confirmed_type = None
    confirmed_indices = []
    
    # Iterate from index 2 to have enough context
    for i in range(2, len(high_vals) - 1):
        current_pivot_type = base_pivot_type[i]
        
        # If no pivot at this position, continue
        if current_pivot_type == '':
            continue
        
        # Determine which pivot types to consider
        possible_types = []
        if current_pivot_type == 'B':
            possible_types = ['H', 'L']
        else:
            possible_types = [current_pivot_type]
        
        # For each possible type, check if we can confirm it
        for pivot_type in possible_types:
            # Check if we're waiting for the opposite type
            if last_confirmed_type is not None:
                if ((last_confirmed_type == 'H' and pivot_type != 'L') or 
                    (last_confirmed_type == 'L' and pivot_type != 'H')):
                    continue  # Not the opposite type we expect
            
            # Try to confirm the pivot
            if try_confirm_pivot(i, pivot_type, high_vals, low_vals, base_pivot_type):
                # Pivot confirmed - store value and type
                if pivot_type == 'H':
                    confirmed_pivots[i] = high_vals[i]
                    pivot_types[i] = 'H'
                else:  # pivot_type == 'L'
                    confirmed_pivots[i] = low_vals[i]
                    pivot_types[i] = 'L'
                
                confirmed_indices.append(i)
                last_confirmed_type = pivot_type
                break  # Only confirm one per position
    
    return pivot_types, confirmed_pivots

def zigzag(data=None, pivot_tuple=None, index=None):
    """
        Combina pivot highs y lows en un solo array manteniendo los índices correctos.
        
        Args:
            data: Objeto Data o DataFrame con columnas 'High' y 'Low' (opcional)
            pivot_tuple: Tupla (pivot_types, pivot_values) de la función pivot (opcional)
            index: Índice opcional (puede ser None)
        
        Returns:
            numpy.ndarray: Array combinado con pivot highs y lows en sus posiciones correctas
        
        Uso:
            # Opción 1: Calcular pivots automáticamente
            combined = zigzag(data=my_data)
            
            # Opción 2: Usar pivots ya calculados (optimización)
            pivot_types, pivot_values = pivot(my_data)
            combined = zigzag(pivot_tuple=(pivot_types, pivot_values))
    """

    # Si se proporciona data, calcular pivots automáticamente
    if data is not None:
        pivot_types, pivot_values = pivot(data)
    elif pivot_tuple is not None:
        # Usar la tupla proporcionada directamente
        pivot_types, pivot_values = pivot_tuple
    else:
        raise ValueError("Debe proporcionar 'data' o 'pivot_tuple'")
    
    # El pivot_values ya contiene todos los pivots combinados
    # Solo necesitamos retornarlo directamente
    return pivot_values.copy()

def nbs(data: DataOHLC, 
        neu_format: str = "NEU",
        boo_format: str = "BOO", 
        shk_format: str = "SHK",
        empty_format: str = "___",
        combine_format: str = "{pivot}-{nbs}"):
    """
    Calcula puntos pivot confirmados basados en la acción del precio y retorna tipos formateados.
    
    Args:
        data: Objeto Data con atributos 'High' y 'Low', o DataFrame con columnas 'High' y 'Low'
        neu_format: Formato personalizado para Neutralizers (por defecto: "NEU")
        boo_format: Formato personalizado para Boosters (por defecto: "BOO")
        shk_format: Formato personalizado para Shakers (por defecto: "SHK")
        empty_format: Formato para valores vacíos cuando no hay NBS (por defecto: "___")
        combine_format: Formato para combinar pivot + NBS (por defecto: "{pivot}-{nbs}")
    
    Returns:
        tuple: (pivot_types, pivot_values) donde:
            - pivot_types: Array con tipos formateados (ej: "H-BOO", "L-NEU", etc.)
            - pivot_values: Array con valores de todos los pivots (igual que la función pivot())
    """
    
    # Procesar el tipo de datos de entrada
    if hasattr(data, 'High') and hasattr(data, 'Low'):
        ohlc = data
    else:
        raise ValueError("No se reconoce el tipo de dato especificado")
    
    # Arrays internos para cálculos NBS
    neu_values = np.full(len(ohlc), np.nan)
    boo_values = np.full(len(ohlc), np.nan)
    shk_values = np.full(len(ohlc), np.nan)
    
    # Variables de estado para pendientes
    pending_boo_h = {}
    pending_boo_l = {}
    pending_shk_h = {}
    pending_shk_l = {}
    
    current_neu_h = None
    current_neu_l = None
    current_shk_h = None
    current_shk_l = None
    
    # Obtener pivots válidos usando la función pivot existente
    pivot_types_raw, pivot_values = pivot(ohlc)
    
    # Obtener índices de todos los pivots válidos
    pivot_indices = np.where(~np.isnan(pivot_values))[0]
    
    if len(pivot_indices) < 2:
        # Crear tipos formateados para pivots sin NBS
        max_length = max(20, len(combine_format) + len(boo_format) + 5)
        formatted_pivot_types = np.full(len(ohlc), '', dtype=f'U{max_length}')
        
        # Formatear pivots básicos
        high_indices = np.where(pivot_types_raw == 'H')[0]
        low_indices = np.where(pivot_types_raw == 'L')[0]
        
        formatted_pivot_types[high_indices] = combine_format.format(pivot='H', nbs=empty_format)
        formatted_pivot_types[low_indices] = combine_format.format(pivot='L', nbs=empty_format)
        
        return formatted_pivot_types, pivot_values
    
    # Inicializar primeros dos pivots
    first_idx = pivot_indices[0]
    second_idx = pivot_indices[1]
    first_type = pivot_types_raw[first_idx]
    second_type = pivot_types_raw[second_idx]
    first_value = pivot_values[first_idx]
    second_value = pivot_values[second_idx]
    
    # Primer pivot como BOO
    boo_values[first_idx] = first_value
    if first_type == 'H':
        pending_boo_h[first_idx] = first_value
    else:
        pending_boo_l[first_idx] = first_value
    
    # Segundo pivot como NEU
    neu_values[second_idx] = second_value
    if second_type == 'H':
        current_neu_h = second_idx
    else:
        current_neu_l = second_idx
    
    # Función auxiliar para convertir NEU a SHK
    def convert_neu_to_shk(neu_idx, neu_type):
        nonlocal current_neu_h, current_neu_l, current_shk_h, current_shk_l
        
        neu_value = neu_values[neu_idx]
        neu_values[neu_idx] = np.nan
        shk_values[neu_idx] = neu_value
        
        if neu_type == 'H':
            current_neu_h = None
            current_shk_h = neu_idx
            pending_shk_h[neu_idx] = neu_value
        else:
            current_neu_l = None
            current_shk_l = neu_idx
            pending_shk_l[neu_idx] = neu_value
    
    # Procesar resto de pivots
    for i in range(2, len(pivot_indices)):
        current_idx = pivot_indices[i]
        current_type = pivot_types_raw[current_idx]
        current_value = pivot_values[current_idx]
        
        if current_type == 'H':
            # Verificar neutralizaciones
            neutralized_boo_h = [idx for idx, val in pending_boo_h.items() if current_value >= val]
            neutralized_shk_h = [idx for idx, val in pending_shk_h.items() if current_value >= val]
            
            supera_neu_actual = (current_neu_h is not None and 
                               current_value >= neu_values[current_neu_h])
            supera_shk_actual = (current_shk_h is not None and 
                               current_value >= shk_values[current_shk_h])
            
            # Si neutraliza algo, se convierte en NEU_H
            if neutralized_boo_h or neutralized_shk_h or supera_neu_actual or supera_shk_actual:
                # Limpiar neutralizados
                for idx in neutralized_boo_h:
                    del pending_boo_h[idx]
                for idx in neutralized_shk_h:
                    del pending_shk_h[idx]
                
                # Establecer como NEU_H
                neu_values[current_idx] = current_value
                current_neu_h = current_idx
                
                if supera_shk_actual:
                    current_shk_h = None
                
                # Pivot anterior se convierte en BOO
                if i > 0:
                    prev_idx = pivot_indices[i - 1]
                    prev_type = pivot_types_raw[prev_idx]
                    prev_value = pivot_values[prev_idx]
                    
                    if prev_type == 'L':
                        if not np.isnan(neu_values[prev_idx]):  # Era NEU_L
                            convert_neu_to_shk(prev_idx, 'L')
                        else:
                            boo_values[prev_idx] = prev_value
                            pending_boo_l[prev_idx] = prev_value
                            
        elif current_type == 'L':
            # Verificar neutralizaciones
            neutralized_boo_l = [idx for idx, val in pending_boo_l.items() if current_value <= val]
            neutralized_shk_l = [idx for idx, val in pending_shk_l.items() if current_value <= val]
            
            supera_neu_actual = (current_neu_l is not None and 
                               current_value <= neu_values[current_neu_l])
            supera_shk_actual = (current_shk_l is not None and 
                               current_value <= shk_values[current_shk_l])
            
            # Si neutraliza algo, se convierte en NEU_L
            if neutralized_boo_l or neutralized_shk_l or supera_neu_actual or supera_shk_actual:
                # Limpiar neutralizados
                for idx in neutralized_boo_l:
                    del pending_boo_l[idx]
                for idx in neutralized_shk_l:
                    del pending_shk_l[idx]
                
                # Establecer como NEU_L
                neu_values[current_idx] = current_value
                current_neu_l = current_idx
                
                if supera_shk_actual:
                    current_shk_l = None
                
                # Pivot anterior se convierte en BOO
                if i > 0:
                    prev_idx = pivot_indices[i - 1]
                    prev_type = pivot_types_raw[prev_idx]
                    prev_value = pivot_values[prev_idx]
                    
                    if prev_type == 'H':
                        if not np.isnan(neu_values[prev_idx]):  # Era NEU_H
                            convert_neu_to_shk(prev_idx, 'H')
                        else:
                            boo_values[prev_idx] = prev_value
                            pending_boo_h[prev_idx] = prev_value
    
    # Procesar neutralizaciones en tiempo real
    start_idx = pivot_indices[-1] + 1 if len(pivot_indices) > 0 else 0
    
    for i in range(start_idx, len(ohlc)):
        if hasattr(ohlc, 'High') and hasattr(ohlc, 'Low'):  # Objeto Data
            high = ohlc.High[i]
            low = ohlc.Low[i]
        else:  # DataFrame
            high = ohlc.iloc[i]['High']
            low = ohlc.iloc[i]['Low']
        
        # Neutralizar BOO_H pendientes
        neutralized_boo_h = [idx for idx, val in pending_boo_h.items() if high >= val]
        for idx in neutralized_boo_h:
            del pending_boo_h[idx]
        
        # Neutralizar BOO_L pendientes  
        neutralized_boo_l = [idx for idx, val in pending_boo_l.items() if low <= val]
        for idx in neutralized_boo_l:
            del pending_boo_l[idx]
        
        # Neutralizar SHK_H pendientes
        neutralized_shk_h = [idx for idx, val in pending_shk_h.items() if high >= val]
        for idx in neutralized_shk_h:
            del pending_shk_h[idx]
        
        # Neutralizar SHK_L pendientes
        neutralized_shk_l = [idx for idx, val in pending_shk_l.items() if low <= val]
        for idx in neutralized_shk_l:
            del pending_shk_l[idx]
    
    # Crear tipos formateados - ESTE ES EL ÚNICO ARRAY DE SALIDA PARA TIPOS
    max_length = max(20, len(combine_format) + len(boo_format) + 5)
    formatted_pivot_types = np.full(len(ohlc), '', dtype=f'U{max_length}')
    
    # Obtener índices para cada categoría NBS
    neu_indices = np.where(~np.isnan(neu_values))[0]
    boo_indices = np.where(~np.isnan(boo_values))[0] 
    shk_indices = np.where(~np.isnan(shk_values))[0]
    
    # Marcar combinaciones HL+NBS
    for idx in neu_indices:
        pivot_char = pivot_types_raw[idx]  # 'H' o 'L'
        formatted_pivot_types[idx] = combine_format.format(pivot=pivot_char, nbs=neu_format)
        
    for idx in boo_indices:
        pivot_char = pivot_types_raw[idx]  # 'H' o 'L'
        formatted_pivot_types[idx] = combine_format.format(pivot=pivot_char, nbs=boo_format)
        
    for idx in shk_indices:
        pivot_char = pivot_types_raw[idx]  # 'H' o 'L'
        formatted_pivot_types[idx] = combine_format.format(pivot=pivot_char, nbs=shk_format)
    
    # Procesar pivots sin clasificación NBS
    all_pivot_indices = np.where(~np.isnan(pivot_values))[0]
    for idx in all_pivot_indices:
        if formatted_pivot_types[idx] == '':  # No fue asignado por NBS
            pivot_char = pivot_types_raw[idx]  # 'H' o 'L'
            formatted_pivot_types[idx] = combine_format.format(pivot=pivot_char, nbs=empty_format)
    
    return formatted_pivot_types, pivot_values

# def hh_ll(data,
    #       hh_format: str = "HH",
    #       lh_format: str = "LH", 
    #       hl_format: str = "HL",
    #       ll_format: str = "LL",
    #       empty_format: str = "___",
    #       combine_format: str = "{pivot}-{hh_ll}"):
    # """
    #     Identifica Higher Highs, Lower Highs, Higher Lows y Lower Lows basado en puntos pivot.
        
    #     Args:
    #         data: Objeto Data con atributos 'High' y 'Low'
    #         hh_format: Formato para Higher Highs (por defecto: "HH")
    #         lh_format: Formato para Lower Highs (por defecto: "LH")
    #         hl_format: Formato para Higher Lows (por defecto: "HL")
    #         ll_format: Formato para Lower Lows (por defecto: "LL")
    #         empty_format: Formato para pivots sin clasificación HH/LL (por defecto: "___")
    #         combine_format: Formato para combinar pivot + HH/LL (por defecto: "{pivot}-{hh_ll}")
        
    #     Returns:
    #         tuple: (pivot_types, pivot_values) donde:
    #             - pivot_types: Array con tipos formateados (ej: "H-HH", "L-HL", etc.)
    #             - pivot_values: Array con valores de todos los pivots
    # """
    # # Obtener datos de pivots
    # pivot_types_raw, pivot_values = pivot(data)
    
    # # Crear array de tipos formateados
    # max_length = max(20, len(combine_format) + len(hh_format) + 5)
    # formatted_pivot_types = np.full(len(data), '', dtype=f'U{max_length}')
    
    # # Separar pivots por tipo
    # high_indices = np.where(pivot_types_raw == 'H')[0]
    # low_indices = np.where(pivot_types_raw == 'L')[0]
    
    # # Procesar Higher Highs y Lower Highs
    # if len(high_indices) > 1:
    #     high_values = pivot_values[high_indices]
        
    #     # Primer high siempre se marca como sin clasificación
    #     first_high_pos = high_indices[0]
    #     formatted_pivot_types[first_high_pos] = combine_format.format(pivot='H', hh_ll=empty_format)
        
    #     for i in range(1, len(high_values)):
    #         current_high = high_values[i]
    #         previous_high = high_values[i-1]
    #         position = high_indices[i]
            
    #         if current_high > previous_high:
    #             # Higher High
    #             formatted_pivot_types[position] = combine_format.format(pivot='H', hh_ll=hh_format)
    #         else:
    #             # Lower High
    #             formatted_pivot_types[position] = combine_format.format(pivot='H', hh_ll=lh_format)
    # elif len(high_indices) == 1:
    #     # Solo un high, marcarlo como sin clasificación
    #     first_high_pos = high_indices[0]
    #     formatted_pivot_types[first_high_pos] = combine_format.format(pivot='H', hh_ll=empty_format)
    
    # # Procesar Higher Lows y Lower Lows
    # if len(low_indices) > 1:
    #     low_values = pivot_values[low_indices]
        
    #     # Primer low siempre se marca como sin clasificación
    #     first_low_pos = low_indices[0]
    #     formatted_pivot_types[first_low_pos] = combine_format.format(pivot='L', hh_ll=empty_format)
        
    #     for i in range(1, len(low_values)):
    #         current_low = low_values[i]
    #         previous_low = low_values[i-1]
    #         position = low_indices[i]
            
    #         if current_low > previous_low:
    #             # Higher Low
    #             formatted_pivot_types[position] = combine_format.format(pivot='L', hh_ll=hl_format)
    #         else:
    #             # Lower Low
    #             formatted_pivot_types[position] = combine_format.format(pivot='L', hh_ll=ll_format)
    # elif len(low_indices) == 1:
    #     # Solo un low, marcarlo como sin clasificación
    #     first_low_pos = low_indices[0]
    #     formatted_pivot_types[first_low_pos] = combine_format.format(pivot='L', hh_ll=empty_format)

    # return formatted_pivot_types, pivot_values

# def maxh_minl(data):
    # """
    #     Retorna tupla (max_highs_values, min_lows_values)
    #     Con valores reales de pivots y NaN, actualizándose históricamente
    #     También actualiza self.pivot_type durante el proceso
    # """
    # final_pivot_type = np.full(len(data), '', dtype='U4')

    # # Obtener datos de pivots - FORMATO NUEVO
    # pivot_types, pivot_values = pivot(data)
    
    # # Arrays de salida con NaN
    # max_highs_values = np.full(len(data), np.nan)
    # min_lows_values = np.full(len(data), np.nan)
    
    # # Variables para tracking
    # current_max_high = -np.inf
    # current_min_low = np.inf
    
    # # Procesar secuencialmente para mantener histórico
    # for i in range(len(data)):
    #     # Verificar si hay un pivot en esta posición
    #     if not np.isnan(pivot_values[i]):
    #         pivot_type = pivot_types[i]
    #         pivot_value = pivot_values[i]
            
    #         if pivot_type == 'H':
    #             # Nuevo pivot High
    #             if pivot_value > current_max_high:
    #                 current_max_high = pivot_value
    #                 max_highs_values[i] = current_max_high
    #                 final_pivot_type[i] = 'MAXH'
            
    #         elif pivot_type == 'L':
    #             # Nuevo pivot Low
    #             if pivot_value < current_min_low:
    #                 current_min_low = pivot_value
    #                 min_lows_values[i] = current_min_low
    #                 final_pivot_type[i] = 'MINL'
    
    # return max_highs_values, min_lows_values


#######################
def rsi(data, length: int = 14):
    """
    Calcula el Índice de Fuerza Relativa (RSI) usando arrays de NumPy.
    
    Args:
        prices: Array de precios de cierre (numpy array o lista)
        length: Período para el cálculo del RSI (por defecto 14)
    
    Returns:
        numpy.ndarray: Array con los valores RSI
    """
    # Convertir a array de numpy si es necesario
    data = np.asarray(data, dtype=float)
    
    if len(data) < length + 1:
        # Retornar array de NaN si no hay suficientes datos
        return np.full(len(data), np.nan)
    
    # Calcular los cambios en el precio
    delta = np.diff(data)
    
    # Separar los cambios positivos y negativos
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Calcular la media móvil simple inicial (SMA)
    avg_gain = np.convolve(gain, np.ones(length)/length, mode='valid')
    avg_loss = np.convolve(loss, np.ones(length)/length, mode='valid')
    
    # Ajustar los arrays para tener la misma longitud que prices
    # Los primeros 'length' elementos serán NaN
    rsi_values = np.full(len(data), np.nan)
    
    # Calcular RSI usando la fórmula estándar
    rs = avg_gain / (avg_loss + 1e-10)  # Evitar división por cero
    rsi_calc = 100 - (100 / (1 + rs))
    
    # Asignar valores calculados (después del período inicial)
    rsi_values[length:] = rsi_calc

    return rsi_values

def rsi_ema(data, length: int = 14):
    """
    Calcula el RSI usando EMA (más eficiente y similar al RSI estándar)
    """
    data = np.asarray(data, dtype=float)
    
    if len(data) < length + 1:
        return np.full(len(data), np.nan)
    
    delta = np.diff(data)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Calcular EMA de ganancias y pérdidas
    alpha = 1.0 / length
    
    # Inicializar con SMA
    avg_gain = np.mean(gain[:length])
    avg_loss = np.mean(loss[:length])
    
    rsi_values = np.full(len(data), np.nan)
    rsi_values[length] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))
    
    # Calcular EMA para el resto
    for i in range(length + 1, len(data)):
        avg_gain = alpha * gain[i-1] + (1 - alpha) * avg_gain
        avg_loss = alpha * loss[i-1] + (1 - alpha) * avg_loss
        
        if avg_loss == 0:
            rsi_values[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100 - (100 / (1 + rs))
    
    return rsi_values




# ATRIBUTOS -----------------------------------------
pivot.filled = False
zigzag.filled = False
nbs.filled = False
# hh_ll.filled = False
# maxh_minl.filled = False
sma.filled = True
ema.filled = True
wma.filled = True
hma.filled = True
rsi.filled = True
rsi_ema.filled = True

pivot.colors = ["#343434"]
zigzag.colors = ["#343434"]
nbs.colors = ["#EC316C", "#CCCF22", "#EC8E31"]
# hh_ll.colors = ["#939523", "#939523", "#A23B5C", "#9E2D1F"]
# maxh_minl.colors = ["#939523", "#A23B5C"]
sma.colors = ["#48D7CB"]
ema.colors = ["#ACD748"]
wma.colors = ["#D7A048"]
hma.colors = ["#D74848"]
rsi.colors = ["#3266E1"]
rsi_ema.colors = ["#3266E1"]

# Definir atributos de renderizado
_line_indicators = [rsi, bbands, sma, ema, wma, hma, zigzag]
_scatter_indicators = [pivot, nbs] # , hh_ll, maxh_minl
_tex_indicators = []

_overlay_indicators = [pivot, bbands, sma, ema, wma, hma, nbs, zigzag] # hh_ll, maxh_minl, 
_pivot_indicators = [pivot, nbs]

# _not_alignable_indicators = [zigzag] 
_all_indicators = list(set(
    _line_indicators + _scatter_indicators + _overlay_indicators 
    + _tex_indicators
))

[setattr(ind, "renderer_type", "line") for ind in _line_indicators]
[setattr(ind, "renderer_type", "scatter") for ind in _scatter_indicators]
[setattr(ind, "is_overlay", True if ind in _overlay_indicators else False) for ind in _all_indicators]
[setattr(ind, "is_pivot", True if ind in _pivot_indicators else False) for ind in _all_indicators]

# TODO 
class PivotIndicator(ABC): 
    @property
    def index(self):
        """Retorna el índice del objeto Data"""
        return self._index
    
    @property
    def values(self):
        """Retorna una tupla con los tres arrays (neutralizers, boosters, shakers)"""
        return self._values
    
    @property
    def ptypes(self):
        """Retorna una tupla con los tres arrays (neutralizers, boosters, shakers)"""
        return self._ptypes
    
    @property
    def data(self):
        """Retorna el objeto Data original"""
        return self._data
    
class PIVOT(PivotIndicator):
    renderer_style=dict(
        renderer_type="scatter",
        colors=["#343434"],
        size=5,
        filled=False,
        is_overlay=True,
    )

    def __init__(self, data: 'DataOHLC'):
        if not hasattr(data, 'High') or not hasattr(data, 'Low'):
            raise ValueError("El objeto Data debe contener propiedades 'High' y 'Low'")
        
        self._data = data
        self._index = data.index
        # Inicializar array de tipos de pivot
        self._ptypes = np.full(len(data), '', dtype='U1')
        self._values = self._get_result()

    def __len__(self):
        return len(self._values)

    def _get_result(self) -> np.array:
        """
        Calcula los puntos pivote confirmados usando el nuevo algoritmo de confirmación
        También actualiza self.pivot_type durante el proceso
        """
        data = self._data
        
        # Validación de entrada usando las propiedades del objeto Data
        if len(data) < 5:
            raise ValueError("Data debe tener al menos 5 elementos para calcular pivotes")
        
        # Obtener arrays numpy directamente del objeto Data
        high_vals = data.High  # Los objetos _Array ya son arrays numpy
        low_vals = data.Low
        base_pivot_type = self._base_pivot_type()
        
        # Array para almacenar pivotes confirmados
        confirmed_pivots = np.full(len(data), np.nan)
        
        # Variables de estado
        last_confirmed_type = None  # Tipo del último pivot confirmado
        confirmed_indices = []      # Lista de índices confirmados
        
        # Iterar desde el índice 2 para tener suficiente contexto
        for i in range(2, len(data) - 1):  # -1 para poder mirar hacia adelante
            current_pivot_type = base_pivot_type[i]
            
            # Si no hay pivot en esta posición, continuar
            if current_pivot_type == '':
                continue
                
            # Determinar qué tipos de pivot podemos considerar
            possible_types = []
            if current_pivot_type == 'B':
                possible_types = ['H', 'L']
            else:
                possible_types = [current_pivot_type]
            
            # Para cada tipo posible, verificar si podemos confirmarlo
            for pivot_type in possible_types:
                # Verificar si estamos esperando el tipo contrario
                if last_confirmed_type is not None:
                    if ((last_confirmed_type == 'H' and pivot_type != 'L') or 
                        (last_confirmed_type == 'L' and pivot_type != 'H')):
                        continue  # No es el tipo contrario que esperamos
                
                # Intentar confirmar el pivot
                if self._try_confirm_pivot(i, pivot_type, high_vals, low_vals, base_pivot_type):
                    # Pivot confirmado - guardar valor y tipo
                    if pivot_type == 'H':
                        confirmed_pivots[i] = high_vals[i]
                        self._ptypes[i] = 'H'  # Guardar tipo confirmado
                    else:  # pivot_type == 'L'
                        confirmed_pivots[i] = low_vals[i]
                        self._ptypes[i] = 'L'  # Guardar tipo confirmado
                    
                    confirmed_indices.append(i)
                    last_confirmed_type = pivot_type
                    break  # Solo confirmamos uno por posición
        
        return confirmed_pivots
    
    def _try_confirm_pivot(self, pivot_idx, pivot_type, high_vals, low_vals, arr_pivot_type):
        """
        Intenta confirmar un pivot mirando hacia adelante
        """
        pivot_value = high_vals[pivot_idx] if pivot_type == 'H' else low_vals[pivot_idx]
        
        # Iterar hacia adelante para confirmar
        j = pivot_idx + 1
        while j < len(high_vals):
            current_high = high_vals[j]
            current_low = low_vals[j]
            current_pivot_type = arr_pivot_type[j]
            
            # Verificar si el pivot se confirma (se supera en dirección contraria)
            if pivot_type == 'H':
                if current_low < low_vals[pivot_idx]:
                    return True  # Pivot high confirmado
            else:  # pivot_type == 'L'
                if current_high > high_vals[pivot_idx]:
                    return True  # Pivot low confirmado
            
            # Verificar si encontramos otro pivot antes de confirmarse
            if current_pivot_type != '':
                # Hay otro pivot candidato
                other_possible_types = []
                if current_pivot_type == 'B':
                    other_possible_types = ['H', 'L']
                else:
                    other_possible_types = [current_pivot_type]
                
                # Si el nuevo pivot es del mismo tipo, el actual no se puede confirmar
                if pivot_type in other_possible_types:
                    # Verificar si el nuevo pivot es "mejor" (más extremo)
                    if pivot_type == 'H':
                        if current_high > pivot_value:
                            return False  # Hay un high más alto, no se confirma
                    else:  # pivot_type == 'L'
                        if current_low < pivot_value:
                            return False  # Hay un low más bajo, no se confirma
            
            j += 1
        
        # Si llegamos al final sin confirmación, no se confirma
        return False

    def _base_pivot_type(self) -> np.ndarray:
        """
        Identifica los tipos de pivots: 'H' (High), 'L' (Low), 'B' (Both), o '' (ninguno)
        Estos son los pivots candidatos, no los confirmados
        """
        # Usar directamente los arrays del objeto Data
        high_vals = self._data.High  # Los objetos _Array ya funcionan como numpy arrays
        low_vals = self._data.Low
        
        # Pre-calcular los shifts para mejor rendimiento
        high_shifts = {
            -2: np.concatenate([np.full(2, np.nan), high_vals[:-2]]),
            -1: np.concatenate([np.full(1, np.nan), high_vals[:-1]]),
            1: np.concatenate([high_vals[1:], np.full(1, np.nan)]),
            2: np.concatenate([high_vals[2:], np.full(2, np.nan)])
        }
        
        low_shifts = {
            -2: np.concatenate([np.full(2, np.nan), low_vals[:-2]]),
            -1: np.concatenate([np.full(1, np.nan), low_vals[:-1]]),
            1: np.concatenate([low_vals[1:], np.full(1, np.nan)]),
            2: np.concatenate([low_vals[2:], np.full(2, np.nan)])
        }
        
        # Condiciones vectorizadas para pivotes high
        condition_high = (
            (high_vals >= high_shifts[-1]) &
            (high_vals >= high_shifts[-2]) &
            (high_vals > high_shifts[1]) &
            (high_vals > high_shifts[2]) &
            ~np.isnan(high_shifts[-1]) &
            ~np.isnan(high_shifts[-2]) &
            ~np.isnan(high_shifts[1]) &
            ~np.isnan(high_shifts[2])
        )
        
        # Condiciones vectorizadas para pivotes low
        condition_low = (
            (low_vals <= low_shifts[-1]) &
            (low_vals <= low_shifts[-2]) &
            (low_vals < low_shifts[1]) &
            (low_vals < low_shifts[2]) &
            ~np.isnan(low_shifts[-1]) &
            ~np.isnan(low_shifts[-2]) &
            ~np.isnan(low_shifts[1]) &
            ~np.isnan(low_shifts[2])
        )
        
        # Crear array de tipos de pivotes
        pivot_type = np.full(len(self._data), '', dtype='U1')
        
        # Asignar tipos según las condiciones
        both_condition = condition_high & condition_low
        pivot_type[both_condition] = 'B'
        pivot_type[condition_high & ~both_condition] = 'H'
        pivot_type[condition_low & ~both_condition] = 'L'

        return pivot_type

class ZIGZAG(PIVOT):
    renderer_style=dict(
        renderer_type="line",
        colors=["#343434"],
        line_width=1.8,
        filled=False,
        is_overlay=True,
    )

class RGXPIVOT(PivotIndicator):
    renderer_style=dict(
        renderer_type="none",
        colors=["#343434"],
        line_width=1.8,
        filled=False,
        is_overlay=True,
    )

    def __init__(self, data: Union[PivotIndicator, DataOHLC], nbs=True):
        """formato del pivot que retorna: <L-neu-lh/i25/''/3433> """ 
        self.pivot = PIVOT(data) if isinstance(data, DataOHLC) else data
        self._nbs = nbs 
        
        types_ = NBS(data).ptypes if nbs else data.ptypes
        mask = types_ != ''
        self._values = (self.pivot.values if len(self.pivot.values) == len(self.pivot.data) else self.pivot.base_pivots.values)[mask]
        self._ptypes = types_[mask]
        ###
        self._index = self.pivot.data.index[mask]
        self._nbar = np.array(range(len(self.pivot.data)))[mask]
        ###
        self._code_sequence = self._create_rgx_pivot_groups()
        self._string = "".join(self._code_sequence)
        self._tags = np.full(len(self._index), '', dtype='U5')

        # TODO: cuando hh_ll esté funcionando correctamente, adaptar esta clase para que funcione bien en conjunto con nbs y los pivotes normales

    def _create_rgx_pivot_groups(self) -> np.ndarray:
        # Convertir a formato de patron de busqueda "<L-neu-lh/i25/''/3433>"
        pivot_values = self._values
        pivot_types = self._ptypes
        pivot_index = self._nbar
        
        assert len(pivot_types) == len(pivot_values) == len(pivot_index)

        # Convertir a formato de patron de busqueda "<L-neu-lh/i25/''/3433>"
        pivot_groups = map(lambda x: f"<{x[1]}/i{x[0]}/''/{x[2]}>", zip(pivot_index, pivot_types, pivot_values))

        return list(pivot_groups)


    def set_tag(self, pattern: re.Pattern, tags: Iterable):
        """
        Aplica etiquetas a los grupos capturados por un patrón regex de forma optimizada.
        
        Este método busca coincidencias del patrón en self._string, extrae los grupos
        capturados y los reemplaza en self._codes. Optimizado para listas grandes
        usando sets para búsqueda O(1) en lugar de búsqueda lineal O(n).
        
        Args:
            pattern (re.Pattern): Patrón regex compilado que debe contener grupos de 
                                captura entre paréntesis. Cada grupo debe coincidir 
                                con elementos en self._codes.
            tags (Iterable): Secuencia de strings que serán usados como etiquetas.
                           Debe tener exactamente la misma cantidad de elementos
                           que grupos de captura en el patrón.
        
        Raises:
            ValueError: Si el patrón no encuentra coincidencias en self._string.
            ValueError: Si la cantidad de grupos capturados no coincide con la
                       cantidad de tags proporcionados.
        
        Modifies:
            self._codes: Modifica in-place los elementos que coinciden con los
                        grupos capturados, reemplazando "''" por "'tag'".
        
        Example:
            >>> processor = GCodeProcessor(["G1", "X''", "Y''", "F1000"])
            >>> pattern = re.compile(r"(X'').*?(Y'')")
            >>> tags = ["coord_x", "coord_y"]
            >>> processor.set_tag_optimized(pattern, tags)
            >>> print(processor._codes)
            ["G1", "X'coord_x'", "Y'coord_y'", "F1000"]

        Note:
            - El método asume que los elementos a reemplazar contienen "''"
            - Solo se procesan elementos que están en los grupos capturados
            - Modifica la instancia actual (no es funcional/inmutable)
        """

        match = pattern.search(self._string)
        if not match:
            raise ValueError("No se logró la búsqueda")
        
        found_groups = match.groups()
        tags_list = list(tags)
        
        if len(found_groups) != len(tags_list):
            raise ValueError(f"Esperaba {len(tags_list)} grupos, pero se encontraron {len(found_groups)}")
        
        # Usar set para búsqueda rápida
        groups_set = set(found_groups)
        replacements = dict(zip(found_groups, tags_list))
        
        # Solo procesar elementos que están en el set
        for i in range(len(self._code_sequence)):
            if self._code_sequence[i] in groups_set:
                new_tag = replacements[self._code_sequence[i]]
                self._code_sequence[i] = self._code_sequence[i].replace("''", f"'{new_tag}'")
                self._tags[i] = new_tag


    def match(self, pattern: re.Pattern, **kwargs):
        return re.match(pattern=pattern, string=self.string, **kwargs)

    def search(self, pattern: re.Pattern, **kwargs):
        return re.search(pattern=pattern, string=self.string, **kwargs)

    def findall(self, pattern: re.Pattern, **kwargs):
        return re.findall(pattern=pattern, string=self.string, **kwargs)

    def __str__(self):
        return self.string.replace(">", "\033[96m> \033[0m").replace("<", "\033[96m<\033[0m")

    @property
    def code_sequence(self):
        return self._code_sequence

    @property
    def string(self) -> str:
        return "".join(self._code_sequence)

    @property
    def nbar(self):
        return self._nbar

    @property
    def tags(self):
        return self._tags

class NBS(PivotIndicator):
    """
    Clase NBS simplificada que calcula Neutralizers, Boosters y Shakers
    basándose en pivots confirmados de un objeto Data.
    """
    
    # Configuración de estilo por defecto
    renderer_style=dict(
        renderer_type="scatter",
        colors=["#315DECFF", "#CA1BCDFF", "#98CE18FF"],
        size=5,
        filled=False,
        is_overlay=True,
    )
    
    # Formateo por defecto para cada tipo de pivot
    DEFAULT_FORMAT = {
        'H-neu': 'H-neu',
        'L-neu': 'L-neu', 
        'H-boo': 'H-boo',
        'L-boo': 'L-boo',
        'H-shk': 'H-shk',
        'L-shk': 'L-shk',
        'H-emp': 'H-emp',
        'L-emp': 'L-emp'
    }

    def __new__(cls, data, *args, **kwargs):
        if isinstance(data, NBS):
            return data
        return super().__new__(cls)

    def __init__(self, data: Union[DataOHLC, PivotIndicator], style=None, format_config=None):
        """
        Inicializa el indicador NBS basado en un objeto Data
        
        Args:
            data: Objeto Data con datos OHLCV o objeto PIVOT (para evitar volver a calcular los pivot)
            style: Diccionario con configuración de estilo personalizada
            format_config: Diccionario con formato personalizado para cada tipo de pivot
        """
        # if not hasattr(data, 'High') or not hasattr(data, 'Low'):
        #     raise ValueError("El objeto Data debe contener propiedades 'High' y 'Low'")
            # Si ya es un NBS, no lo vuelvo a inicializar
        if isinstance(data, NBS):
            return  
  
        assert isinstance(data, DataOHLC) or isinstance(data, PivotIndicator), "Data debe ser objeto 'Data' o 'PIVOT'"
        
        # Calcular pivots usando la clase PIVOT
        if isinstance(data, PIVOT):
            self.base_pivots = data
        else:
            self.base_pivots = PIVOT(data)

        self._data = data
        self._index = data.index
        
        # Configurar formato de visualización
        self._format_config = format_config if format_config is not None else self.DEFAULT_FORMAT.copy()
        
        # Arrays para almacenar los resultados (N, B, S)
        self._neutralizers = np.full(len(data), np.nan)
        self._boosters = np.full(len(data), np.nan)
        self._shakers = np.full(len(data), np.nan)
        
        # Array de tipos de pivot detallados
        self._ptypes = np.full(len(data), '', dtype='U5')
        
        # Primero asignar todos los pivots vacíos (confirmados pero no parte de NBS)
        self._assign_empty_pivots()
        
        # Variables de estado para tracking
        self._current_neu_h = None  # Índice del neutralizer H actual
        self._current_neu_l = None  # Índice del neutralizer L actual
        self._pending_boosters_h = {}  # {index: value} para BOO_H pendientes
        self._pending_boosters_l = {}  # {index: value} para BOO_L pendientes
        self._pending_shakers_h = {}   # {index: value} para SHK_H pendientes
        self._pending_shakers_l = {}   # {index: value} para SHK_L pendientes
        
        # Procesar los pivots para generar NBS
        self._calculate_nbs()
    
    def _assign_empty_pivots(self):
        """
        Asigna tipos 'H-emp' y 'L-emp' a todos los pivots confirmados inicialmente
        """
        pivot_values = self.base_pivots._values
        pivot_types = self.base_pivots._ptypes
        
        for i in range(len(pivot_values)):
            if not np.isnan(pivot_values[i]):
                # Es un pivot confirmado, asignar como empty inicialmente
                if pivot_types[i] == 'H':
                    self._ptypes[i] = 'H-emp'
                elif pivot_types[i] == 'L':
                    self._ptypes[i] = 'L-emp'
    
    def _calculate_nbs(self):
        """
        Calcula los puntos Neutralizer, Booster y Shaker basándose en pivots confirmados
        """
        pivot_values = self.base_pivots._values
        pivot_types = self.base_pivots._ptypes
        
        # Obtener índices de pivots válidos (no NaN)
        valid_pivots = []
        for i, value in enumerate(pivot_values):
            if not np.isnan(value):
                valid_pivots.append((i, pivot_types[i], value))
        
        if len(valid_pivots) < 2:
            return
        
        # Inicializar primeros dos pivots
        self._initialize_first_pivots(valid_pivots)
        
        # Procesar el resto de pivots
        for i in range(2, len(valid_pivots)):
            pivot_idx, pivot_type, pivot_value = valid_pivots[i]
            
            if pivot_type == 'H':
                self._process_high_pivot(pivot_idx, pivot_value, valid_pivots, i)
            elif pivot_type == 'L':
                self._process_low_pivot(pivot_idx, pivot_value, valid_pivots, i)
    
    def _initialize_first_pivots(self, valid_pivots):
        """
        Inicializa los primeros dos pivots como Booster y Neutralizer
        """
        first_idx, first_type, first_value = valid_pivots[0]
        second_idx, second_type, second_value = valid_pivots[1]
        
        # Primer pivot = Booster
        self._boosters[first_idx] = first_value
        self._ptypes[first_idx] = f'{first_type}-boo'
        
        # Segundo pivot = Neutralizer
        self._neutralizers[second_idx] = second_value
        self._ptypes[second_idx] = f'{second_type}-neu'
        
        # Actualizar estado
        if second_type == 'H':
            self._current_neu_h = second_idx
        else:
            self._current_neu_l = second_idx
        
        # Agregar primer pivot a pendientes
        if first_type == 'H':
            self._pending_boosters_h[first_idx] = first_value
        else:
            self._pending_boosters_l[first_idx] = first_value
    
    def _process_high_pivot(self, pivot_idx, pivot_value, valid_pivots, position):
        """
        Procesa un pivot High para determinar si neutraliza algo o se ignora
        """
        neutralized_something = False
        
        # Verificar neutralización de boosters H pendientes
        to_remove_boo_h = []
        for boo_idx, boo_value in self._pending_boosters_h.items():
            if pivot_value >= boo_value:
                to_remove_boo_h.append(boo_idx)
                neutralized_something = True
        
        # Verificar neutralización de shakers H pendientes
        to_remove_shk_h = []
        for shk_idx, shk_value in self._pending_shakers_h.items():
            if pivot_value >= shk_value:
                to_remove_shk_h.append(shk_idx)
                neutralized_something = True
        
        # Verificar neutralización del neutralizer H actual
        neutralizes_current_neu_h = False
        if self._current_neu_h is not None:
            neu_value = self._neutralizers[self._current_neu_h]
            if pivot_value >= neu_value:
                neutralizes_current_neu_h = True
                neutralized_something = True
        
        # Si neutraliza algo, este pivot se convierte en neutralizer
        if neutralized_something:
            # Limpiar neutralizados
            for idx in to_remove_boo_h:
                del self._pending_boosters_h[idx]
            for idx in to_remove_shk_h:
                del self._pending_shakers_h[idx]
            
            # Convertir a neutralizer
            self._neutralizers[pivot_idx] = pivot_value
            self._ptypes[pivot_idx] = 'H-neu'
            self._current_neu_h = pivot_idx
            
            # El pivot anterior (si existe y es L) se convierte en booster
            if position > 0:
                prev_idx, prev_type, prev_value = valid_pivots[position - 1]
                if prev_type == 'L':
                    self._convert_to_booster_or_shaker(prev_idx, prev_type, prev_value)
        else:
            # No neutraliza nada, mantener como empty
            # Ya está asignado como 'H-emp' en _assign_empty_pivots
            pass
    
    def _process_low_pivot(self, pivot_idx, pivot_value, valid_pivots, position):
        """
        Procesa un pivot Low para determinar si neutraliza algo o se ignora
        """
        neutralized_something = False
        
        # Verificar neutralización de boosters L pendientes
        to_remove_boo_l = []
        for boo_idx, boo_value in self._pending_boosters_l.items():
            if pivot_value <= boo_value:
                to_remove_boo_l.append(boo_idx)
                neutralized_something = True
        
        # Verificar neutralización de shakers L pendientes
        to_remove_shk_l = []
        for shk_idx, shk_value in self._pending_shakers_l.items():
            if pivot_value <= shk_value:
                to_remove_shk_l.append(shk_idx)
                neutralized_something = True
        
        # Verificar neutralización del neutralizer L actual
        neutralizes_current_neu_l = False
        if self._current_neu_l is not None:
            neu_value = self._neutralizers[self._current_neu_l]
            if pivot_value <= neu_value:
                neutralizes_current_neu_l = True
                neutralized_something = True
        
        # Si neutraliza algo, este pivot se convierte en neutralizer
        if neutralized_something:
            # Limpiar neutralizados
            for idx in to_remove_boo_l:
                del self._pending_boosters_l[idx]
            for idx in to_remove_shk_l:
                del self._pending_shakers_l[idx]
            
            # Convertir a neutralizer
            self._neutralizers[pivot_idx] = pivot_value
            self._ptypes[pivot_idx] = 'L-neu'
            self._current_neu_l = pivot_idx
            
            # El pivot anterior (si existe y es H) se convierte en booster
            if position > 0:
                prev_idx, prev_type, prev_value = valid_pivots[position - 1]
                if prev_type == 'H':
                    self._convert_to_booster_or_shaker(prev_idx, prev_type, prev_value)
        else:
            # No neutraliza nada, mantener como empty
            # Ya está asignado como 'L-emp' en _assign_empty_pivots
            pass
    
    def _convert_to_booster_or_shaker(self, pivot_idx, pivot_type, pivot_value):
        """
        Convierte un pivot en Booster o Shaker según si ya era Neutralizer
        """
        current_pivot_type = self._ptypes[pivot_idx]
        
        if current_pivot_type.endswith('-neu'):
            # Era neutralizer, convertir a shaker
            self._neutralizers[pivot_idx] = np.nan  # Limpiar de neutralizers
            self._shakers[pivot_idx] = pivot_value
            self._ptypes[pivot_idx] = f'{pivot_type}-shk'
            
            # Agregar a pendientes de shakers
            if pivot_type == 'H':
                self._pending_shakers_h[pivot_idx] = pivot_value
                if self._current_neu_h == pivot_idx:
                    self._current_neu_h = None
            else:
                self._pending_shakers_l[pivot_idx] = pivot_value
                if self._current_neu_l == pivot_idx:
                    self._current_neu_l = None
        else:
            # No era neutralizer, convertir a booster
            self._boosters[pivot_idx] = pivot_value
            self._ptypes[pivot_idx] = f'{pivot_type}-boo'
            
            # Agregar a pendientes de boosters
            if pivot_type == 'H':
                self._pending_boosters_h[pivot_idx] = pivot_value
            else:
                self._pending_boosters_l[pivot_idx] = pivot_value
    
    @property
    def values(self):
        """Retorna una tupla con los tres arrays (neutralizers, boosters, shakers)"""
        return (self._neutralizers, self._boosters, self._shakers)
    
    def __len__(self):
        """Retorna la longitud de los datos"""
        return len(self._data)
    
    def get_formatted_pivot_type(self, index):
        """
        Retorna el tipo de pivot formateado según la configuración
        
        Args:
            index: Índice del pivot
            
        Returns:
            String con el tipo de pivot formateado
        """
        raw_type = self._ptypes[index]
        if raw_type in self._format_config:
            return self._format_config[raw_type]
        return raw_type
    
    def get_all_formatted_pivot_types(self):
        """
        Retorna un array con todos los tipos de pivot formateados
        
        Returns:
            numpy array con los tipos formateados
        """
        formatted = np.full(len(self._ptypes), '', dtype='U20')
        for i, raw_type in enumerate(self._ptypes):
            if raw_type != '':
                formatted[i] = self.get_formatted_pivot_type(i)
        return formatted
    
    def update_format_config(self, new_format):
        """
        Actualiza la configuración de formato
        
        Args:
            new_format: Diccionario con nuevos formatos
        """
        self._format_config.update(new_format)
    
    def set_format_config(self, format_config):
        """
        Establece una nueva configuración de formato completa
        
        Args:
            format_config: Diccionario completo con formatos
        """
        self._format_config = format_config.copy()
    
    def get_format_config(self):
        """
        Retorna la configuración de formato actual
        
        Returns:
            Diccionario con la configuración actual
        """
        return self._format_config.copy()


class RSI:
    renderer_style=dict(
        renderer_type="line",
        colors=["#BB8A46FF"],
        line_width=2,
        filled=True,
        is_overlay=False,
    )

    def __init__(self, data, length: int = 14):
        """
        Calcula el Índice de Fuerza Relativa (RSI) usando arrays de NumPy.
        
        Args:
            data: Objeto Data completo o Array de precios de cierre
            length: Período para el cálculo del RSI (por defecto 14)
        """
        # Detectar si es un objeto Data completo o un _Array
        if hasattr(data, 'Close'):
            # Es un objeto Data completo, extraer Close
            self.__data_source = data  # Guardar referencia al objeto Data completo
            self.__data = data.Close   # Usar los precios de cierre
            self.__index = data.index  # Usar el índice del objeto Data
        else:
            # Es un _Array individual
            self.__data_source = None
            self.__data = data
            self.__index = data.index if hasattr(data, 'index') else None
            
        self.__length = length
        self.__values = self._calculate_rsi()
    
    def _calculate_rsi(self):
        """Calcula los valores RSI."""
        if len(self.__data) < self.__length + 1:
            return np.full(len(self.__data), np.nan)
        
        # Calcular los cambios en el precio
        delta = np.diff(self.__data)
        
        # Separar los cambios positivos y negativos
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Calcular la media móvil simple inicial (SMA)
        avg_gain = np.convolve(gain, np.ones(self.__length)/self.__length, mode='valid')
        avg_loss = np.convolve(loss, np.ones(self.__length)/self.__length, mode='valid')
        
        # Ajustar los arrays para tener la misma longitud que data
        rsi_values = np.full(len(self.__data), np.nan)
        
        # Calcular RSI usando la fórmula estándar
        rs = avg_gain / (avg_loss + 1e-10)  # Evitar división por cero
        rsi_calc = 100 - (100 / (1 + rs))
        
        # Asignar valores calculados (después del período inicial)
        rsi_values[self.__length:] = rsi_calc
        
        return rsi_values
    
    @property
    def index(self):
        """Retorna el índice del objeto Data"""
        return self.__index
    
    @property
    def data(self):
        """Retorna el objeto Data original (si se pasó un objeto Data completo)"""
        return self.__data_source if self.__data_source is not None else self.__data
    
    @property
    def length(self):
        """ Retorna el length del indicador """  
        return self.__length

    @property
    def values(self):
        """Accede a los valores RSI calculados."""
        return self.__values

    def __len__(self):
        """Retorna la longitud de los datos"""
        return len(self.__data)

"""
si un indicador es de tipo pivot 

"""





