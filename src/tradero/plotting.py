import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import (
    Span, HoverTool, Legend, LegendItem, 
    ColumnDataSource, Range1d, DataRange1d, CustomJS
    )
from bokeh.layouts import column, gridplot
from bokeh.transform import factor_cmap
from bokeh.models.formatters import NumeralTickFormatter
from typing import Union, Literal
from .util import try_, minutes2timeframe, timeframe2minutes
from .models import DataOHLC
from .util import ColorWayGenerator

# sin uso
def align_indicator_to_original(
    values: np.ndarray,
    index: pd.DatetimeIndex,
    objetive_index: pd.DatetimeIndex
    ) -> np.ndarray:
    """
        Alinea un indicador de 5m al índice original de 1m con ffill.
        
        Args:
            indicator_5m: Array del indicador calculado en 5m
            index_5m: Índice temporal del indicador de 5m  
            original_index_1m: Índice original de 1m al que alinear
            
        Returns:
            Array del mismo tamaño que original_index_1m con ffill
    """
    # Crear DataFrame con el indicador
    indicator_series = pd.Series(data=values, index=index)
    
    # Reindexar al índice original de 1m y hacer ffill
    aligned = indicator_series.reindex(objetive_index).ffill()
    
    return aligned.values

def find_minutes_timeframe(index: np.ndarray[np.datetime64]) -> int:
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
    
    first_time = index[1].astype("M8[ms]").astype("O")
    second_time = index[0].astype("M8[ms]").astype("O")
    
    return (first_time - second_time).seconds // 60  # Convertir segundos a minutos  # Retorna un entero con los minutos completos

_AUTOSCALE_JS_CALLBACK = """
    const data = source.data;
    const x = data['Date'];
    const start = x_range.start;
    const end = x_range.end;

    let minY = Infinity;
    let maxY = -Infinity;

    if (mode === "ohlc") {
        const high = data['High'];
        const low = data['Low'];
        for (let i = 0; i < x.length; i++) {
            if (x[i] >= start && x[i] <= end) {
                if (low[i] < minY) minY = low[i];
                if (high[i] > maxY) maxY = high[i];
            }
        }
    } else if (mode === "volume") {
        const vol = data['Volume'];
        minY = 0;
        for (let i = 0; i < x.length; i++) {
            if (x[i] >= start && x[i] <= end) {
                if (vol[i] > maxY) maxY = vol[i];
            }
        }
    }

    if (minY < Infinity && maxY > -Infinity) {
        const range = maxY - minY;
        const padding = range * padding_factor; // parámetro externo
        y_range.start = minY - padding;
        y_range.end = maxY + padding;
    }
    """

_MAX_CANDLES = 10_000
_INDICATOR_HEIGHT = 90  # valor de altura del indicador por defecto

def plot(
    stats: pd.Series,
    ohlc_base_data: DataOHLC,
    indicators_blueprints: dict,
    plot_equity: bool = True,
    plot_return: bool = True,
    plot_drawdown: bool = False,
    plot_trades: bool = True,
    plot_volume: bool = True,
    relative_equity: bool = True,
    timeframe: str = None,
) -> None:

    # TODO: assert: el intervalo de algun indicador no puede ser menor < al intervalo de la ohlc_base_data 
    
    default_data_interval = find_minutes_timeframe(ohlc_base_data.index)
    default_data_timeframe = minutes2timeframe(default_data_interval)

    # procesar el parametro timeframe
    if timeframe is None:
        if indicators_blueprints:
            indicators_intervals = [timeframe2minutes(i[1]["timeframe"]) for i in indicators_blueprints.items()]
            indicators_min_interval = min(indicators_intervals)
            timeframe = minutes2timeframe(indicators_min_interval)
        else:
            timeframe = default_data_timeframe

    required_timeframe = timeframe  # ex: str "5m"
    required_interval = timeframe2minutes(timeframe) # ex: int 5

    # variables
    ohlc_base_data = ohlc_base_data
    resampled_ohlc_base_data = ohlc_base_data.resample(required_timeframe)

    # Map Colors
    BAR_COLORS = ["#d63030", "#259956"]

    def resample_indicator(
        indicator_values: np.ndarray,
        time_index: Union[np.ndarray, pd.DatetimeIndex, list],
        target_timeframe: str,
        method: Literal['last', 'mean', 'max', 'min', 'sum', 'first'] = 'last'
    ) -> tuple[np.ndarray, pd.DatetimeIndex]:
        """
            Resamplea un indicador a un timeframe específico.
            
            Parameters:
            -----------
            indicator_values : np.ndarray
                Array con los valores del indicador
            time_index : Union[np.ndarray, pd.DatetimeIndex, list]
                Índice temporal correspondiente a los valores
            target_timeframe : str
                Timeframe objetivo (ej: '5m', '15m', '1h', '4h', '1d')
                Formatos soportados: 'm' (minutos), 'h' (horas), 'd' (días)
            method : str, default 'last'
                Método de agregación: 'last', 'mean', 'max', 'min', 'sum', 'first'
            
            Returns:
            --------
            tuple[np.ndarray, pd.DatetimeIndex]
                Tupla con (valores_resampleados, nuevo_indice_temporal)
            
            Examples:
            ---------
            >>> import numpy as np
            >>> import pandas as pd
            >>> 
            >>> # Datos de ejemplo (1 minuto)
            >>> values = np.random.randn(100)
            >>> times = pd.date_range('2024-01-01 09:00:00', periods=100, freq='1min')
            >>> 
            >>> # Resamplear a 5 minutos
            >>> new_values, new_times = resample_indicator(values, times, '5m', method='mean')
            >>> print(f"Original: {len(values)} puntos -> Resampleado: {len(new_values)} puntos")
        """
        
        # Validaciones
        if len(indicator_values) != len(time_index):
            raise ValueError("El tamaño de indicator_values debe coincidir con time_index")
        
        if len(indicator_values) == 0:
            return np.array([]), pd.DatetimeIndex([])
        
        # Convertir a DataFrame para usar pandas resample
        if isinstance(time_index, np.ndarray):
            if time_index.dtype.kind == 'M':  # numpy datetime64
                time_index = pd.to_datetime(time_index)
            else:
                time_index = pd.to_datetime(time_index)
        elif not isinstance(time_index, pd.DatetimeIndex):
            time_index = pd.DatetimeIndex(time_index)
        
        # Crear DataFrame temporal
        df = pd.DataFrame({
            'value': indicator_values,
            'time': time_index
        })
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        

            
        pandas_freq = target_timeframe
        
        # Aplicar resampling según el método especificado
        method_map = {
            'last': lambda x: x.iloc[-1] if len(x) > 0 else np.nan,
            'first': lambda x: x.iloc[0] if len(x) > 0 else np.nan,
            'mean': lambda x: x.mean(),
            'max': lambda x: x.max(),
            'min': lambda x: x.min(),
            'sum': lambda x: x.sum()
        }
        
        if method not in method_map:
            raise ValueError(f"Método no soportado: {method}. Opciones: {list(method_map.keys())}")
        
        # Realizar el resampling
        resampled = df.resample(pandas_freq).apply(method_map[method])
        
        return resampled['value'].values, resampled.index

    def _process_bool_plot_parameters():
        """Procesa los parámetros booleanos de plotting"""
        # global plot_equity, plot_trades, plot_drawdown
        # plot_equity = bool(plot_equity)
        # plot_trades = bool(plot_trades)
        # plot_drawdown = bool(plot_drawdown)
        pass

    def _process_stats_parameter():
        """Procesa y valida el parámetro stats"""
        if stats is None:
            return
        # Validaciones específicas para stats

    def _process_ohlc_data_parameter():
        """Procesa y valida los datos OHLC"""
        assert isinstance(ohlc_base_data, DataOHLC), "ohlc_data debe ser un objeto Data. con atributos 'Open',High', 'Low', 'Close'"
        if ohlc_base_data is None or ohlc_base_data.empty:
            raise ValueError("ohlc_data no puede estar vacío")


    def _create_fig_ohlc(data: DataOHLC, height=400, width=1200, output="notebook") -> tuple:

        data = data.copy()

        # Asegurar el formato correcto de fecha si de un dia o mas
        if default_data_interval >= 1440:
            data.index = data.index.normalize().tz_localize("UTC")

        # Separar datos alcistas y bajistas
        inc = data.Close > data.Open 
        dec = data.Open > data.Close 
        
        df = data.df
        df["Date"] = df.index.values.astype('int64') // 1_000_000

        # Convertir a ColumnDataSource para Bokeh
        source_ohlc = ColumnDataSource(df)
        source_ohlc.add((data.Close >= data.Open).astype(np.uint8).astype(str), 'inc')
        
        # Ajustar el ancho de las velas para datos de 5 minutos (5 minutos * 60 segundos * 1000 ms)
        min_interval = required_interval
        separation = 0.10
        w = (min_interval - (separation * min_interval)) * 60 * 1000 

        fig = figure(
            x_axis_type="datetime",
            # title="Gráfico de Velas - Bokeh",
            tools="xpan,xwheel_zoom,xwheel_pan,box_zoom,undo,redo,reset,save",
            width=width,
            height=height,
            background_fill_color="#e0e6eb",  # Fondo interior
            border_fill_color="#e0e6eb",     # Fondo exterior
            active_drag='xpan',
            active_scroll='xwheel_zoom',
        )
        # estilos
        if True:
            # Configuracion de ejes 
            # fig.xaxis.axis_label = 'Fecha'
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

        inc_cmap = factor_cmap('inc', BAR_COLORS, ['0', '1'])

        # Velas 
        fig.segment('Date', 'High', 'Date', 'Low', 
                source=source_ohlc, color=inc_cmap, line_width=2, legend_label=f"OHLC ·{required_timeframe}"
        )
        fig.vbar(
            'Date', w, 'Open', 'Close', source=source_ohlc, 
            fill_color=inc_cmap, line_color=inc_cmap, line_width=0, legend_label=f"OHLC ·{required_timeframe}"
        )
        
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
        
        # JS para recalcular eje 'Y' automáticamente
        ohlc_callback = CustomJS(args=dict(
            source=source_ohlc, 
            y_range=fig.y_range, 
            x_range=fig.x_range, 
            mode="ohlc",
            padding_factor=0.07,
            ), 
            code=_AUTOSCALE_JS_CALLBACK
        )
        fig.x_range.js_on_change('start', ohlc_callback)
        fig.x_range.js_on_change('end', ohlc_callback)

        return fig, source_ohlc

    fig_ohlc, source_ohlc = _create_fig_ohlc(resampled_ohlc_base_data)

    def _create_indicator_fig(**kwargs):
        """Crea una figura para indicadores externos"""
        kwargs_default = {'height': _INDICATOR_HEIGHT, 'x_range': fig_ohlc.x_range}
        kwargs_default.update(kwargs)
        
        fig = _new_bokeh_figure(**kwargs_default)
        fig.yaxis.minor_tick_line_color = None
        fig.yaxis.ticker.desired_num_ticks = 3
        return fig

    def _new_bokeh_figure(**kwargs):
        """Crea una nueva figura de Bokeh con configuración base"""
        defaults = {
            'x_axis_type': 'datetime',
            'tools': 'xpan,xwheel_zoom,xwheel_pan,box_zoom,undo,redo,reset,save',
            'width': 1100,
            'height': 400,
            'background_fill_color': '#e0e6eb',
            'border_fill_color': '#e0e6eb',
            'active_drag': 'xpan',
            'active_scroll': 'xwheel_zoom',
            'y_range': DataRange1d(range_padding=0.4),
        }
        defaults.update(kwargs)
        fig = figure(**defaults)

        # fig.y_range=Range1d(start=valor_min, end=valor_max),
        # fig.y_range.bounds = (valor_min, valor_max)

        # configurar estilos
        if True:
            # CONFIGURACION DE EJES
            fig.xaxis.visible = False
            fig.yaxis.formatter.use_scientific = False
            # Ocultar las etiquetas del eje X para ahorrar espacio
            fig.xaxis.major_label_text_font_size = "0pt"
            fig.xaxis.minor_tick_line_color = None
            fig.xaxis.major_tick_line_color = None

            fig.border_fill_color = "#E3EAF1FF"
            fig.background_fill_color = "#E3EAF1FF"  # <-- EDIT COLOR -- 
            fig.outline_line_color = '#E3EAF1FF' # <-- EDIT COLOR --

            # # Configuracion de ejes 
            # fig.xaxis.axis_label = 'Fecha'
            # fig.yaxis.axis_label = 'Precio'
            # fig.title.text_color = "#FF0000FF"
            # fig.xaxis.axis_label_text_color = "#333333"
            # fig.yaxis.axis_label_text_color = "#333333"
            # fig.xaxis.major_label_text_color = "#333333"
            # fig.yaxis.major_label_text_color = "#333333"

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
        
        return fig



    def _validate_indicator(key, values):
        """Valida que el indicador tenga el formato correcto"""
        # if not isinstance(values, dict):
        #     raise ValueError(f"Indicador '{key}' debe ser un diccionario")
        
        # if 'values' not in values:
        #     raise ValueError(f"Indicador '{key}' debe tener clave 'values'")
        
        # if 'type' not in values:
        #     values['type'] = 'line'  # Tipo por defecto
        
        # if values['type'] not in ['line', 'scatter', 'candlestick']:
        #     raise ValueError(f"Tipo de indicador '{values['type']}' no válido para '{key}'")
        return 

    def _is_external_indicator(values: np.ndarray) -> bool:
        """
        Identifica si el indicador es overlay o external
        Retorna True si es un indicador externo
        """
        
        vals = values
        
        # Obtener rango de precios OHLC para comparar
        ohlc_min = ohlc_base_data.df[['Open', 'High', 'Low', 'Close']].min().min()
        ohlc_max = ohlc_base_data.df[['Open', 'High', 'Low', 'Close']].max().max()
        
        # Filtrar valores no nulos
        valid_vals = vals[~pd.isna(vals)] if hasattr(pd, 'isna') else [v for v in vals if v is not None]
        
        if len(valid_vals) == 0:
            return True  # Si no hay valores válidos, tratarlo como externo
        
        indicator_min = min(valid_vals)
        indicator_max = max(valid_vals)
        
        # Si los valores del indicador están muy fuera del rango de precios, es externo
        price_range = ohlc_max - ohlc_min
        tolerance = price_range * 0.1  # 10% de tolerancia
        
        if (indicator_max < ohlc_min - tolerance or 
            indicator_min > ohlc_max + tolerance or
            indicator_max - indicator_min < price_range * 0.01):  # Rango muy pequeño comparado con precios
            return True
        
        return False

    def calculate_indidicator(bp):
        func = bp["func"]
        timeframe = bp["timeframe"]
        kwargs = bp["kwargs"]
        indicator_values = func(data=ohlc_base_data, **kwargs)
        return (indicator_values)

    def _indicator_renderer_type(bp, indicator_values=None) -> str:
        """Retorna el tipo de renderer: 'scatter', 'line' o 'candlestick'"""
        import numpy as np

        # Si values es un dict y tiene 'type', devolverlo
        if isinstance(bp, dict) and 'type' in bp:
            return bp['type']
        
        # Si no se proporciona indicator_values, y values es un dict con 'values', usarlo
        if indicator_values is None:
            if isinstance(bp, dict) and 'values' in bp:
                indicator_values = bp['values']
            else:
                # Asumir que values es directamente el array de datos
                indicator_values = bp
        
        # Convertir a array numpy
        values_array = np.array(indicator_values)

        # Si es 2D, tratar cada fila (banda) por separado
        if values_array.ndim == 2:
            renderer_types = []
            for i in range(values_array.shape[0]):
                band_values = values_array[i]
                renderer_type = _indicator_renderer_type(band_values)
                renderer_types.append(renderer_type)
            
            # Si todas las bandas son scatter, usar scatter
            if all(rt == 'scatter' for rt in renderer_types):
                return 'scatter'
            # Si hay mezcla, usar scatter para indicadores muy dispersos
            elif any(rt == 'scatter' for rt in renderer_types):
                # Verificar si es muy disperso
                total_non_nan = np.sum(~np.isnan(values_array))
                total_elements = values_array.size
                if total_non_nan < total_elements * 0.4:  # Aumentar el umbral
                    return 'scatter'
            return 'line'  # por defecto
        
        # Para arrays 1D - Ajustar el umbral para detectar indicadores dispersos
        non_nan_count = np.sum(~np.isnan(values_array))
        total_count = len(values_array)

        # Reducir el umbral para detectar mejor indicadores con valores puntuales
        if non_nan_count < total_count * 0.5 and non_nan_count > 0:
            return 'scatter'

        return 'line'



    # def _add_drawdown_fig():
        # """Añade figura de drawdown"""
        # if not plot_drawdown or stats is None:
        #     return None
        
        # if hasattr(stats, '_equity_curve') and 'DrawdownPct' in stats._equity_curve.columns:
        #     drawdown_fig = _create_indicator_fig()
        #     drawdown_fig.title.text = "Drawdown %"
            
        #     drawdown_data = stats._equity_curve['DrawdownPct']
        #     drawdown_fig.line(x=drawdown_data.index, y=drawdown_data.values, 
        #                     color='red', line_width=2, legend_label='Drawdown %')
            
        #     # Rellenar área bajo la curva
        #     drawdown_fig.varea(x=drawdown_data.index, y1=0, y2=drawdown_data.values, 
        #                         color='red', alpha=0.3)
            
        #     return drawdown_fig
        
        # return None

    def _add_equity_fig(is_return=False):
        """Añade figura de equity"""
        
        # if not hasattr(stats, '_equity_curve') and not 'Equity' in stats._equity_curve.columns:
        #     return None

        equity = stats._equity_curve['Equity'].copy().bfill()
        
        if relative_equity: # return %
            equity /= equity.iloc[0]
        if is_return: # return $
            equity -= equity.iloc[0]

        # 
        if is_return:
            legend_label =f'Return {equity.iloc[-1]:,.2%}' if relative_equity else f'Return ${equity.iloc[-1]:,.2f}'
        else: 
            legend_label =f'Equity {equity.iloc[-1]:,.2%}' if relative_equity else f'Equity ${equity.iloc[-1]:,.2f}'


        tick_format = "0,0.[00]%" if relative_equity else "$ 0.0 a"
        color='#7A37EDFF' if is_return else '#3F56F0FF'

        # resamplear al timeframe requerido        
        resampled_values, index = resample_indicator(equity, ohlc_base_data.index, required_timeframe)

        # Crear la figura y renderer
        equity_fig = _create_indicator_fig(name="equity", height=90)
        # Agregar liena zero
        equity_fig.add_layout(Span(
            location=0, dimension='width', line_color='#929292FF', line_width=0.5, line_dash='dashed'))
        # Agregar linea de equity/retorno
        equity_fig.line(x=index, y=resampled_values, 
            color=color, line_width=2, 
            legend_label=legend_label)

        equity_fig.yaxis.formatter = NumeralTickFormatter(format=tick_format)
        equity_fig.yaxis.axis_label = 'Return' if is_return else 'Equity'
        
        return equity_fig

    def _add_volume_fig(source_ohlc):
        """Añade figura de volumen"""
        
        fig = _create_indicator_fig(name="volume", height=60, x_range=fig_ohlc.x_range)
        # Añadir barras de volumen
        # volume_data = resampled_ohlc_base_data.Volume
        # index = resampled_ohlc_base_data.index

        # source_volume = ColumnDataSource(dict(
        #     x=index,
        #     top=volume_data,
        #     inc=(resampled_ohlc_base_data.Close > resampled_ohlc_base_data.Open).astype(np.uint8).astype(str),
        # ))

        inc_cmap = factor_cmap('inc', BAR_COLORS, ['0', '1'])

        min_interval = required_interval
        separation = 0.10
        w = (min_interval - (separation * min_interval)) * 60 * 1000 
        
        fig.vbar(
            x='Date', 
            width=w, 
            top=0, 
            bottom="Volume",
            color=inc_cmap, 
            alpha=0.6,
            source=source_ohlc,
        )

        fig.yaxis.axis_label = f'Volume'

        # JS para recalcular eje 'Y' automáticamente
        volume_callback = CustomJS(
            args=dict(
                source=source_ohlc, 
                y_range=fig.y_range, 
                x_range=fig.x_range, 
                mode="volume",
                padding_factor=0.0,
            ), 
            code=_AUTOSCALE_JS_CALLBACK
        )
        fig.x_range.js_on_change('start', volume_callback)
        fig.x_range.js_on_change('end', volume_callback)
        
        return fig

    def _add_trades_renderer():
        """Añade los trades como puntos en el gráfico OHLC"""
        if not plot_trades or stats is None:
            return

        if hasattr(stats, '_trades') and not stats._trades.empty:
            trades_df = stats._trades

            trades_source = ColumnDataSource(dict(
                index=trades_df["ExitBar"],
                datetime=trades_df["ExitTime"],
                size=trades_df["Size"],
                returns_positive=(trades_df['ReturnPct'] > 0).astype(int).astype(str),
                # lines_x0=trades_df["EntryTime"],
                # lines_x1=trades_df["ExitTime"],
            ))

            trades_cmap = factor_cmap('returns_positive', ["red", "green"], ['0', '1'])

            lines_xs_list = (trades_df[["EntryTime", "ExitTime"]].values.astype('int64') // 1_000_000).tolist()
            lines_ys_list = trades_df[["EntryPrice", "ExitPrice"]].values.tolist()

            trades_source.add(lines_xs_list, "lines_xs")
            trades_source.add(lines_ys_list, "lines_ys")
            # Añadir lineas de entrada y salida (compras)
            if not trades_df.empty:
                fig_ohlc.multi_line(xs="lines_xs", ys="lines_ys",
                    source=trades_source, line_width=8, 
                    line_color=trades_cmap, 
                    line_dash='dotted',
                    legend_label=f'Trades {len(trades_df)}' 
                )

    def _add_indicators_figs_or_renderers(blueprints: dict):

        """Procesa todos los indicadores y los añade como figuras o renderers"""
        def _align_indicator_to_original_data(
            values: np.ndarray,
            index: pd.DatetimeIndex,
            objetive_index: pd.DatetimeIndex,
            fill_na: bool = True,
            dropna=False
        ) -> np.ndarray:
            """
                Alinea un indicador de 5m al índice original de 1m con ffill.
                
                Args:
                    indicator_5m: Array del indicador calculado en 5m
                    index_5m: Índice temporal del indicador de 5m  
                    original_index_1m: Índice original de 1m al que alinear
                    
                Returns:
                    Array del mismo tamaño que original_index_1m con ffill
            """
            # Crear DataFrame con el indicador
            indicator_series = pd.Series(data=values, index=index)
            
            # Reindexar al índice original de 1m y hacer ffill
            aligned = indicator_series.reindex(objetive_index)
            
            # rellenar espacios vacíos 
            if fill_na:
                aligned.ffill(inplace=True)
            
            # liminar espacios vacíos
            elif dropna:
                aligned.dropna(inplace=True)
                
            return aligned.values

        color_generator = ColorWayGenerator()

        processed_indicators = []
        for tag, bp in blueprints.items():
            
            obj = bp["obj"]
            style = obj.renderer_style # dict
            is_overlay_indicator = style["is_overlay"]
            renderer_type = style["renderer_type"]

            timeframe = bp["timeframe"]
            kwargs = bp["kwargs"]

            indicator_resampled_data = ohlc_base_data.resample(timeframe)

            try:
                indicator_return = obj(indicator_resampled_data, **kwargs).values
            except:
                indicator_return = obj(indicator_resampled_data.Close, **kwargs).values
            
            # agregar el indicador calculado
            processed_indicators.append(indicator_return)

            # convertir a tupla si no lo es
            indicator_return = indicator_return if isinstance(indicator_return, tuple) else (indicator_return, )

            # if is_pivot_indicator: # modificar indicator_returns
            #     pivot_type = indicator_returns[0]
            #     pivot_values = indicator_returns[1]

            #     # Sufijos que queremos separar
            #     modified_pivot_type = map(lambda x: x[1:], pivot_type)
            #     suffixes = [v for v in dict.fromkeys(modified_pivot_type) if v != ""]

            #     # Generar arrays filtrados
            #     result = tuple(
            #         np.where(np.char.endswith(pivot_type, suf), pivot_values, np.nan) 
            #         for suf in suffixes
            #     )

            #     indicator_returns = result
            #     print(indicator_returns)

            general_index = resampled_ohlc_base_data.index
            for i, arr in enumerate(indicator_return):
                indicator_values = arr
                indicator_index = indicator_resampled_data.index
                color = try_(lambda: style["colors"][i], color_generator())
                filled = try_(lambda: style["filled"], None)
                legend_tag_label=tag.replace(f"{obj.__name__}", f'{obj.__name__.upper()}')

                # if func.not_alignable:
                #     mask = ~np.isnan(indicator_values)
                #     x = general_index[mask]
                #     y = indicator_values[mask]
                #     fig_ohlc.line(x=x, y=y, color=color, line_width=1.8, legend_label=tag)

                if renderer_type == "line":
                    line_width = style["line_width"]
                    x = general_index
                    y = _align_indicator_to_original_data(
                        values=indicator_values, index=indicator_index, 
                        objetive_index=general_index,
                        fill_na=True if filled is None else filled
                    )

                    mask = ~np.isnan(y)
                    x = x[mask]
                    y = y[mask]
                    if is_overlay_indicator:
                        fig_ohlc.line(x=x, y=y, color=color, line_width=line_width, legend_label=legend_tag_label)
                    else:
                        p = _create_indicator_fig()
                        p.line(x, y, color=color, line_width=line_width, legend_label=legend_tag_label)
                        p.yaxis.axis_label = obj.__name__.upper()
                        figs_below_ohlc.append(p)

                elif renderer_type == "scatter":
                    size = style["size"]
                    x = general_index
                    y = _align_indicator_to_original_data(
                        values=indicator_values, index=indicator_index, 
                        objetive_index=general_index,
                        fill_na=False if filled is None else filled
                    )
                    if is_overlay_indicator:
                        fig_ohlc.scatter(x, y, size=size, color=color, legend_label=legend_tag_label)
                    else:
                        p = _create_indicator_fig()
                        p.scatter(x, y, size=size, color=color, legend_label=legend_tag_label)
                        p.yaxis.axis_label = obj.__name__.upper()
                        figs_below_ohlc.append(p)



    def _configure_legends(figs: list):
        for f in figs:
            if f.legend:
                f.legend.visible = True# show_legend
                f.legend.location = 'top_left'
                f.legend.border_line_width = 0.1
                f.legend.border_line_color = '#C6C6C6FF'
                f.legend.padding = 3
                f.legend.spacing = 20
                f.legend.margin = 0
                f.legend.border_radius = 3
                f.legend.label_text_font_size = '8pt'
                f.legend.click_policy = "hide"
                f.legend.background_fill_color = "#e0e6eb"  # Cambia el color del fondo
                f.legend.background_fill_alpha = 0.8  # Ajusta la transparencia del fondo
                f.legend.orientation = "horizontal"
            # if f.legend:
            #     f.legend.visible = show_legend
            #     f.legend.location = 'top_left'
            #     f.legend.border_line_width = 1
            #     f.legend.border_line_color = '#333333'
            #     f.legend.padding = 5
            #     f.legend.spacing = 0
            #     f.legend.margin = 0
            #     f.legend.label_text_font_size = '8pt'
            #     f.legend.click_policy = "hide"
            #     f.legend.background_fill_alpha = .9
            
    def _configure_margin(figs: list, exceptions:list[str] = ["volume"]):
        for f in figs:
            if f.name not in exceptions:
                f.min_border_left = 70    # Para labels de precios
                f.min_border_right = 30   # Espacio derecho
                f.min_border_top = 10     # Espacio superior
                f.min_border_bottom = 10  # Espacio inferior
            
    # f.outline_line_color = '#666666'

    # Procesar parámetros
    _process_bool_plot_parameters()
    _process_stats_parameter()
    _process_ohlc_data_parameter()

    # Inicializar listas de figuras
    figs_above_ohlc = []  # equity, drawdown, returns
    figs_below_ohlc = []  # volume, indicadores tipo RSI, MACD

    # Inicializar generador de colores
    color_generator = ColorWayGenerator()

    # Añadir figuras superiores
    if plot_equity:
        equity_fig = _add_equity_fig()
        figs_above_ohlc.append(equity_fig)
    if plot_return:
        equity_fig = _add_equity_fig(is_return=True)
        figs_above_ohlc.append(equity_fig)
    
    # TODO: drawdown_fig
    # drawdown_fig = _add_drawdown_fig()
    # if drawdown_fig:
    #     figs_above_ohlc.append(drawdown_fig)

    # Añadir figuras inferiores
    if plot_volume:
        volume_fig = _add_volume_fig(source_ohlc)
        figs_below_ohlc.append(volume_fig)

    # Añadir trades
    if plot_trades:
        _add_trades_renderer()

    # Procesar indicadores
    _add_indicators_figs_or_renderers(blueprints=indicators_blueprints)

    # Combinar todas las figuras
    all_figs = figs_above_ohlc + [fig_ohlc] + figs_below_ohlc

    # Configuraciones adicionales 
    _configure_margin(all_figs)
    _configure_legends(all_figs)

    # preparar el grid de figuras (en una columna)
    grid_layout = [[fig] for fig in all_figs]
    # Crear layout final
    if len(all_figs) == 1:
        final_layout = fig_ohlc
    else:
        from bokeh.layouts import gridplot

        final_layout = gridplot(
            grid_layout, 
            toolbar_options=dict(logo=None),
            toolbar_location='right',
            merge_tools=True,
            sizing_mode='stretch_width',
        )

    show(final_layout)

def plot_position(price, tp, sl):
    from bokeh.plotting import figure, show
    from bokeh.models import Span
    from bokeh.io import output_notebook
    direction = "buy" if price < tp else "sell"

    # Configurar el rango del gráfico
    y_min = min(tp, sl) - price * 0.02
    y_max = max(tp, sl) + price * 0.02
    
    # Crear figura
    p = figure(title="Análisis de Posición Trading",
                x_axis_label="Tiempo",
                y_axis_label="Precio ($)",
                width=400, 
                height=400,
                x_range=(0, 50),
                y_range=(y_min, y_max))
    
    if direction == "buy":
        # Área de ganancia (verde)
        p.quad(top=price, bottom=sl, left=20, right=40, color="red", alpha=0.6)
        p.quad(top=tp, bottom=price, left=20, right=40, color="green", alpha=0.6)
    else:
        # Área de ganancia (verde)
        p.quad(top=price, bottom=tp, left=20, right=40, color="green", alpha=0.6)
        p.quad(top=sl, bottom=price, left=20, right=40, color="red", alpha=0.6)
    
    # Línea de entrada (blanco)
    entry_line = Span(location=price, dimension='width', line_color='white', line_width=3, line_alpha=0.8)
    p.add_layout(entry_line)
    
    # Líneas de TP y SL
    tp_line = Span(location=tp, dimension='width', line_color='green', line_width=2, line_dash='dashed')
    sl_line = Span(location=sl, dimension='width', line_color='red', line_width=2, line_dash='dashed')
    p.add_layout(tp_line)
    p.add_layout(sl_line)
    
    # Añadir etiquetas de texto
    p.text(x=[0, 0, 0], y=[tp, price, sl], 
            text=[f"TP: {tp:.1f}", f"Entry: {price:.1f}", f"SL: {sl:.1f}"],
            text_color=["green", "white", "red"],
            text_font_size="10pt")
    
    # Configurar el estilo
    p.title.text_font_size = "14pt"
    p.background_fill_color = "#20222CFF"
    p.border_fill_color = "#20222CFF"
    # p.grid.grid_line_color = "#ffffff"
    p.grid.grid_line_alpha = 0.15

    p.yaxis.formatter.use_scientific = False
    p.yaxis.major_label_text_color = "white"
    p.yaxis.axis_label_text_color = "white"

    p.xaxis.major_label_text_color = "white"
    p.xaxis.axis_label_text_color = "white"

    p.toolbar.logo = None
    
    # Mostrar el gráfico
    show(p)

