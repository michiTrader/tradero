import pandas as pd
import numpy as np
from typing import cast

class Stats(pd.Series):
    """Clase que extiende pd.Series para contener estadísticas de backtest"""
    
    def __init__(self, data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):
        super().__init__(data, index, dtype, name, copy, fastpath)
    
    @property
    def _constructor(self):
        return Stats
    
    def __repr__(self):
        # Personalizar la representación para mostrar mejor las estadísticas
        return super().__repr__()

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

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def geometric_mean(returns):
    if len(returns) == 0:
        return 0
    
    # Convertir a array numpy y forzar tipo float
    try:
        returns_array = np.array(returns, dtype=float)
    except (ValueError, TypeError):
        # Si falla, convertir elemento por elemento
        clean_returns = []
        for r in returns:
            try:
                clean_returns.append(float(r))
            except:
                clean_returns.append(np.nan)
        returns_array = np.array(clean_returns)
    
    # Filtrar valores válidos
    valid_returns = returns_array[np.isfinite(returns_array)]
    
    if len(valid_returns) == 0:
        return 0
    
    # Calcular media geométrica
    if np.any(valid_returns <= -1):
        return np.nan
    
    product = np.prod(1 + valid_returns)
    if product <= 0:
        return np.nan
    
    return np.power(product, 1/len(valid_returns)) - 1

def compute_drawdown_duration_peaks(dd_series):
    """Calcula la duración y picos de drawdown - versión robusta"""
    dd = dd_series.values
    index = dd_series.index
    
    # Verificar que dd_series no esté vacío
    if len(dd) == 0:
        return pd.Series(0, index=index, dtype='timedelta64[ns]'), pd.Series([], dtype=float)
    
    # Encontrar períodos de drawdown (donde dd > 0)
    # Usar una tolerancia pequeña para evitar problemas de punto flotante
    tolerance = 1e-10
    in_dd = dd > tolerance
    
    # Calcular duraciones
    dd_dur = pd.Series(pd.Timedelta(0), index=index, dtype='timedelta64[ns]')
    dd_peaks = []
    
    if not in_dd.any():
        return dd_dur, pd.Series(dd_peaks, dtype=float)
    
    try:
        # Encontrar inicio y fin de cada período de drawdown
        dd_starts = np.where(np.diff(np.concatenate(([False], in_dd))))[0]
        dd_ends = np.where(np.diff(np.concatenate((in_dd, [False]))))[0]
        
        # Verificar que tengamos pares válidos de inicio y fin
        if len(dd_starts) == 0 or len(dd_ends) == 0:
            return dd_dur, pd.Series(dd_peaks, dtype=float)
        
        # Asegurar que tenemos el mismo número de inicios y finales
        min_pairs = min(len(dd_starts), len(dd_ends))
        dd_starts = dd_starts[:min_pairs]
        dd_ends = dd_ends[:min_pairs]
        
        for start, end in zip(dd_starts, dd_ends):
            # Verificar que los índices sean válidos
            if start >= len(index) or end >= len(index) or start > end:
                continue
                
            # Calcular duración
            try:
                duration = index[end] - index[start]
                
                # Asegurar que end no exceda la longitud del array
                actual_end = min(end + 1, len(dd_dur))
                dd_dur.iloc[start:actual_end] = duration
                
                # Calcular el pico de drawdown para este período
                dd_segment = dd[start:end+1]
                if len(dd_segment) > 0:
                    peak_dd = np.nanmax(dd_segment)  # Usar nanmax para manejar NaN
                    if not np.isnan(peak_dd):
                        dd_peaks.append(peak_dd)
                        
            except (IndexError, ValueError) as e:
                print(f"Warning: Error processing drawdown period {start}-{end}: {e}")
                continue
                
    except Exception as e:
        print(f"Error en compute_drawdown_duration_peaks: {e}")
        # Retornar valores por defecto en caso de error
        return dd_dur, pd.Series([], dtype=float)
    
    return dd_dur, pd.Series(dd_peaks, dtype=float)

def _data_period(index):
    """Determina el período de los datos basado en el índice"""
    if len(index) < 2:
        return pd.Timedelta(days=1)
    
    # Calcular la diferencia más común entre timestamps
    diffs = pd.Series(index).diff().dropna()
    if len(diffs) == 0:
        return pd.Timedelta(days=1)
    
    # Usar la mediana para evitar outliers
    return diffs.median()

def _indicator_warmup_nbars(strategy_instance):
    """Calcula el número de barras de warmup necesarias para los indicadores"""
    if not hasattr(strategy_instance, '_indicators') or not strategy_instance._indicators:
        return 0
    
    max_warmup = 0
    for indicator in strategy_instance._indicators:
        if hasattr(indicator, 'period'):
            max_warmup = max(max_warmup, indicator.period)
        elif hasattr(indicator, 'length'):
            max_warmup = max(max_warmup, indicator.length)
    
    return max_warmup

def _round_timedelta(value, period=None):
    """Redondea un timedelta basado en el período de datos"""
    if not isinstance(value, pd.Timedelta):
        return value
    
    # Si no hay período válido, devolver el valor original
    if period is None or not isinstance(period, pd.Timedelta):
        return value
    
    try:
        # Convertir el período a una frecuencia válida usando las nuevas convenciones
        if period.days >= 1:
            freq = f'{period.days}D'
        elif period.seconds >= 3600:
            freq = f'{period.seconds // 3600}H'
        elif period.seconds >= 60:
            freq = f'{period.seconds // 60}min'  # Cambiar 'T' por 'min'
        else:
            freq = f'{period.seconds}S'
        
        return value.ceil(freq)
    except (ValueError, AttributeError):
        # Si hay algún error, devolver el valor original
        return value

# ============================================================================
# FUNCIONES DE PREPARACIÓN DE DATOS
# ============================================================================

def _prepare_trades_dataframe(trades, strategy_instance):
    """Prepara el DataFrame de trades desde la lista de objetos Trade"""
    if isinstance(trades, pd.DataFrame):
        return trades, None
    
    # Crear DataFrame desde objetos Trade
    trades_df = pd.DataFrame({
        'Direction': [getattr(t, 'direction', 'Long') for t in trades],
        'Size': [t.size for t in trades],
        'EntryPrice': [t.entry_price for t in trades],
        'ExitPrice': [t.exit_price for t in trades],
        'PnL': [getattr(t, 'pl', getattr(t, 'pnl', 0)) for t in trades],  # Mejorar extracción de PnL
        'ReturnPct': [getattr(t, 'pl_pct', getattr(t, 'return_pct', 0)) for t in trades],  # Mejorar extracción de retorno
        'SL': [getattr(t, 'sl', None) for t in trades],
        'TP': [getattr(t, 'tp', None) for t in trades],
        'Commission': [getattr(t, '_commissions', getattr(t, 'commission', 0)) for t in trades],
        'EntryTime': [t.entry_time for t in trades],
        'ExitTime': [t.exit_time for t in trades],
        'MaxPrice': [getattr(t, 'max_price', 0) for t in trades],
        'MinPrice': [getattr(t, 'min_price', 0) for t in trades],
        # 'MAE': [getattr(t, 'mae', 0) for t in trades],
        # 'MFE': [getattr(t, 'mfe', 0) for t in trades],
        'EntryBar': [getattr(t, 'entry_bar', 0) for t in trades],
        'ExitBar': [getattr(t, 'exit_bar', 0) for t in trades],
    })
    
    # Si ReturnPct está en 0 pero tenemos precios, calcularlo manualmente
    if trades_df['ReturnPct'].sum() == 0 and len(trades_df) > 0:
        for idx, row in trades_df.iterrows():
            if row['EntryPrice'] != 0 and row['ExitPrice'] != 0:
                if row['Direction'].lower() == 'long':
                    return_pct = (row['ExitPrice'] - row['EntryPrice']) / row['EntryPrice']
                else:  # Short
                    return_pct = (row['EntryPrice'] - row['ExitPrice']) / row['EntryPrice']
                trades_df.at[idx, 'ReturnPct'] = return_pct
                
                # Calcular PnL si no está disponible
                if row['PnL'] == 0:
                    trades_df.at[idx, 'PnL'] = return_pct * row['Size'] * row['EntryPrice']
    
    trades_df['Duration'] = trades_df['ExitTime'] - trades_df['EntryTime']
    trades_df['Tag'] = [getattr(t, 'tag', '') for t in trades]
    
    # Agregar valores de indicadores
    if len(trades_df) and strategy_instance and hasattr(strategy_instance, '_indicators'):
        for ind in strategy_instance._indicators:
            ind = np.atleast_2d(ind)
            for i, values in enumerate(ind):
                suffix = f'_{i}' if len(ind) > 1 else ''
                if hasattr(ind, 'name'):
                    trades_df[f'Entry_{ind.name}{suffix}'] = values[trades_df['EntryBar'].values]
                    trades_df[f'Exit_{ind.name}{suffix}'] = values[trades_df['ExitBar'].values]
    
    commissions = sum(getattr(t, '_commissions', getattr(t, 'commission', 0)) for t in trades)
    return trades_df, commissions

def _prepare_equity_dataframe(equity, index):
    """Prepara el DataFrame de equity con drawdown"""
    dd = 1 - equity / np.maximum.accumulate(equity)
    dd_dur, dd_peaks = compute_drawdown_duration_peaks(pd.Series(dd, index=index))
    
    equity_df = pd.DataFrame({
        'Equity': equity,
        'DrawdownPct': dd,
        'DrawdownDuration': dd_dur
    }, index=index)
    
    return equity_df, dd, dd_dur, dd_peaks
    
# ============================================================================
# FUNCIONES DE CÁLCULO DE ESTADÍSTICAS
# ============================================================================

def _calculate_basic_stats(equity, index, trades_df, commissions, strategy_instance):
    """Calcula estadísticas básicas del backtest"""
    s = pd.Series(dtype=object)
    s.loc['Start'] = index[0]
    s.loc['End'] = index[-1]
    s.loc['Duration'] = s.End - s.Start
    
    # Calcular tiempo de exposición
    have_position = np.repeat(0, len(index))
    for t in trades_df.itertuples(index=False):
        have_position[t.EntryBar:t.ExitBar + 1] = 1
    s.loc['Exposure Time [%]'] = have_position.mean() * 100
    
    # Estadísticas de equity
    s.loc['Equity Final [$]'] = equity.iloc[-1]
    s.loc['Equity Peak [$]'] = equity.max()
    if commissions:
        s.loc['Commissions [$]'] = commissions
    
    # Retorno total
    s.loc['Return [%]'] = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0] * 100
    
    return s

def _calculate_benchmark_stats(s, ohlc_data, strategy_instance):
    """Calcula estadísticas de benchmark (Buy & Hold)"""
    first_trading_bar = _indicator_warmup_nbars(strategy_instance)
    c = ohlc_data.Close.values if hasattr(ohlc_data, 'Close') else ohlc_data['Close'].values
    s.loc['Buy & Hold Return [%]'] = (c[-1] - c[first_trading_bar]) / c[first_trading_bar] * 100
    return s

def _calculate_annualized_metrics(s, equity_df, index):
    """Calcula métricas anualizadas (retorno, volatilidad, CAGR)"""
    is_datetime_index = isinstance(index, pd.DatetimeIndex)
    if not is_datetime_index:
        return s, np.array([np.nan]), np.nan, 0, np.nan
    
    freq_days = cast(pd.Timedelta, _data_period(index)).days
    have_weekends = index.dayofweek.to_series().between(5, 6).mean() > 2 / 7 * .6
    annual_trading_days = (
        52 if freq_days == 7 else
        12 if freq_days == 31 else
        1 if freq_days == 365 else
        (365 if have_weekends else 252))
    
    freq = {7: 'W', 31: 'ME', 365: 'YE'}.get(freq_days, 'D')
    day_returns = equity_df['Equity'].resample(freq).last().dropna().pct_change()
    gmean_day_return = geometric_mean(day_returns)
    
    # Retorno anualizado
    annualized_return = (1 + gmean_day_return)**annual_trading_days - 1
    s.loc['Return (Ann.) [%]'] = annualized_return * 100
    
    # Volatilidad anualizada
    volatility = np.sqrt((day_returns.var(ddof=int(bool(day_returns.shape))) + 
                        (1 + gmean_day_return)**2)**annual_trading_days - 
                        (1 + gmean_day_return)**(2 * annual_trading_days)) * 100
    s.loc['Volatility (Ann.) [%]'] = volatility
    
    # CAGR
    time_in_years = (s.loc['Duration'].days + s.loc['Duration'].seconds / 86400) / annual_trading_days
    s.loc['CAGR [%]'] = ((s.loc['Equity Final [$]'] / equity_df['Equity'].iloc[0])**(1 / time_in_years) - 1) * 100 if time_in_years else np.nan
    
    return s, day_returns, annual_trading_days, annualized_return, gmean_day_return

def _calculate_risk_metrics(s, day_returns, annual_trading_days, annualized_return, dd, risk_free_rate=0.0):
    """Calcula métricas de riesgo (Sharpe, Sortino, Calmar)"""
    # Sharpe Ratio
    volatility = s.loc['Volatility (Ann.) [%]']
    s.loc['Sharpe Ratio'] = (s.loc['Return (Ann.) [%]'] - risk_free_rate * 100) / (volatility or np.nan)
    
    # Sortino Ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        negative_returns = day_returns[day_returns < 0]  # Only actual negative returns
        
        if len(negative_returns) == 0:
            # No negative returns - perfect performance, Sortino should be infinite
            s.loc['Sortino Ratio'] = np.inf
        else:
            downside_deviation = np.sqrt(np.mean(negative_returns**2)) * np.sqrt(annual_trading_days)
            
            if downside_deviation == 0 or np.isnan(downside_deviation) or np.isinf(downside_deviation):
                s.loc['Sortino Ratio'] = np.nan
            else:
                s.loc['Sortino Ratio'] = (annualized_return - risk_free_rate) / downside_deviation
    
    # Calmar Ratio
    max_dd = -np.nan_to_num(dd.max())
    s.loc['Calmar Ratio'] = annualized_return / (-max_dd or np.nan)
    
    return s, max_dd

def _calculate_alpha_beta(s, equity, ohlc_data, risk_free_rate=0.0):
    """Calcula Alpha y Beta usando CAPM"""
    # Asegurar que equity y ohlc_data tengan la misma longitud
    # equity_curve puede tener un elemento extra al inicio
    c = ohlc_data.Close.values if hasattr(ohlc_data, 'Close') else ohlc_data['Close'].values
    
    # Ajustar equity para que coincida con la longitud de los datos OHLC
    if len(equity) == len(c) + 1:
        # equity tiene un elemento extra al inicio, usar desde el segundo elemento
        equity_aligned = equity[1:]
    else:
        # equity y c tienen la misma longitud
        equity_aligned = equity
    
    # Calcular log returns con arrays de la misma longitud
    equity_log_returns = np.log(equity_aligned[1:] / equity_aligned[:-1])
    market_log_returns = np.log(c[1:] / c[:-1])
    
    beta = np.nan
    if len(equity_log_returns) > 1 and len(market_log_returns) > 1 and len(equity_log_returns) == len(market_log_returns):
        cov_matrix = np.cov(equity_log_returns, market_log_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    
    # Jensen CAPM Alpha
    s.loc['Alpha [%]'] = (s.loc['Return [%]'] - risk_free_rate * 100 - 
                        beta * (s.loc['Buy & Hold Return [%]'] - risk_free_rate * 100))
    s.loc['Beta'] = beta
    
    return s

def _calculate_drawdown_stats(s, max_dd, dd_peaks, dd_dur):
    """Calcula estadísticas de drawdown"""
    s.loc['Max. Drawdown [%]'] = max_dd * 100
    s.loc['Avg. Drawdown [%]'] = -dd_peaks.mean() * 100
    
    # Manejar duración de drawdown de manera segura
    if len(dd_dur) > 0 and hasattr(dd_dur, 'index'):
        period = _data_period(dd_dur.index)
        s.loc['Max. Drawdown Duration'] = _round_timedelta(dd_dur.max(), period)
        s.loc['Avg. Drawdown Duration'] = _round_timedelta(dd_dur.mean(), period)
    else:
        s.loc['Max. Drawdown Duration'] = pd.Timedelta(0)
        s.loc['Avg. Drawdown Duration'] = pd.Timedelta(0)
    
    return s

def _calculate_trade_stats(s, trades_df, pl, returns, durations):
    """Calcula estadísticas de trades"""
    n_trades = len(trades_df)
    s.loc['# Trades'] = n_trades
    
    # Win Rate
    win_rate = np.nan if not n_trades else (pl > 0).mean()
    s.loc['Win Rate [%]'] = win_rate * 100
    
    # Trade Performance
    s.loc['Best Trade [%]'] = returns.max() * 100
    s.loc['Worst Trade [%]'] = returns.min() * 100
    s.loc['Avg. Trade [%]'] = geometric_mean(returns) * 100
    
    # Duración de trades - manejar casos donde durations puede estar vacío
    if len(durations) > 0 and not durations.isna().all():
        # Obtener el período de los datos del índice de durations si está disponible
        if hasattr(durations, 'index') and len(durations.index) > 1:
            period = _data_period(durations.index)
        else:
            period = pd.Timedelta(minutes=5)  # Período por defecto para datos de 5 minutos
        
        s.loc['Max. Trade Duration'] = _round_timedelta(durations.max(), period)
        s.loc['Avg. Trade Duration'] = _round_timedelta(durations.mean(), period)
    else:
        s.loc['Max. Trade Duration'] = pd.Timedelta(0)
        s.loc['Avg. Trade Duration'] = pd.Timedelta(0)
    
    # Profit Factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    
    if gross_loss > 0:
        s.loc['Profit Factor'] = gross_profit / gross_loss
    else:
        s.loc['Profit Factor'] = np.inf if gross_profit > 0 else np.nan
    
    # Expectancy
    s.loc['Expectancy [%]'] = returns.mean() * 100
    
    return s, n_trades, win_rate

# def _calculate_advanced_metrics(s, n_trades, win_rate, pl):
    # """Calcula métricas avanzadas (SQN, Kelly Criterion)"""
    # # System Quality Number (SQN)
    # s.loc['SQN'] = np.sqrt(n_trades) * pl.mean() / (pl.std() or np.nan)
    
    # # Kelly Criterion
    # if n_trades > 0 and len(pl[pl > 0]) > 0 and len(pl[pl < 0]) > 0:
    #     avg_win = pl[pl > 0].mean()
    #     avg_loss = -pl[pl < 0].mean()
    #     if avg_loss > 0:
    #         s.loc['Kelly Criterion'] = win_rate - (1 - win_rate) / (avg_win / avg_loss)
    #     else:
    #         s.loc['Kelly Criterion'] = np.nan
    # else:
    #     s.loc['Kelly Criterion'] = np.nan
    
    # return s

def _calculate_advanced_metrics(s, n_trades, win_rate, pl):
    # Convertir pl a numérico
    pl_clean = pd.to_numeric(pd.Series(pl), errors='coerce').dropna()
    
    if len(pl_clean) == 0:
        s.loc['SQN'] = np.nan
    else:
        pl_std = pl_clean.std()
        s.loc['SQN'] = np.sqrt(len(pl_clean)) * pl_clean.mean() / (pl_std if pl_std != 0 else np.nan)
    
    # Kelly Criterion
    if len(pl_clean) > 0:
        wins = pl_clean[pl_clean > 0]
        losses = pl_clean[pl_clean < 0]
        
        if len(wins) > 0 and len(losses) > 0:
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())
            win_prob = len(wins) / len(pl_clean)
            s.loc['Kelly %'] = (win_prob - ((1 - win_prob) / (avg_win / avg_loss))) * 100
        else:
            s.loc['Kelly %'] = np.nan
    else:
        s.loc['Kelly %'] = np.nan
    
    return s

def _calculate_mae_mfe(s, equity: float, trades_df, metric_type = "ROI"):
    #  calculat el MAE y el MFE según el tipo de métrica ('ROI'|'ROE')
    """Calcular el MAE y el MFE para una operación dada"""
    trades_df.Mae = None
    trades_df.Mfe = None

    maes = []
    mfes = []
    for t in trades_df.itertuples(index=False):
        cash = equity.loc[t.ExitTime]
        is_long = t.Direction == "Long"
        entry_price = t.EntryPrice
        up_distance, down_distance = (t.MaxPrice - entry_price), (entry_price - t.MinPrice)
        size = t.Size
        if metric_type == 'ROI':
            maes.append((down_distance / entry_price) if is_long else (up_distance / entry_price))
            mfes.append((up_distance / entry_price) if is_long else (down_distance / entry_price))
        elif metric_type == 'ROE':
            maes.append((down_distance * size / cash) if is_long else (up_distance * size / cash))
            mfes.append((up_distance * size / cash) if is_long else (down_distance * size / cash))

    trades_df["Mae"] = maes
    trades_df["Mfe"] = mfes

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def compute_stats(trades, ohlc_data, equity_curve, strategy_instance=None, risk_free_rate=0.0, mae_metric_type="ROI"):
    """Calcula estadísticas completas del backtest de manera organizada"""
    index = ohlc_data.index
    
    # 1. Preparar datos
    trades_df, commissions = _prepare_trades_dataframe(trades, strategy_instance)
    equity_df, dd, dd_dur, dd_peaks = _prepare_equity_dataframe(equity_curve, index)
    # Extraer datos de trades
    pl = trades_df['PnL']
    returns = trades_df['ReturnPct']
    durations = trades_df['Duration']
    
    # 2. Calcular estadísticas básicas
    s = _calculate_basic_stats(equity_curve, index, trades_df, commissions, strategy_instance)
    
    # 3. Calcular estadísticas de benchmark
    s = _calculate_benchmark_stats(s, ohlc_data, strategy_instance)
    
    # 4. Calcular métricas anualizadas
    s, day_returns, annual_trading_days, annualized_return, gmean_day_return = _calculate_annualized_metrics(s, equity_df, index)
    
    # 5. Calcular métricas de riesgo
    s, max_dd = _calculate_risk_metrics(s, day_returns, annual_trading_days, annualized_return, dd, risk_free_rate)
    
    # 6. Calcular Alpha y Beta
    s = _calculate_alpha_beta(s, equity_curve, ohlc_data, risk_free_rate)
    
    # 7. Calcular estadísticas de drawdown
    s = _calculate_drawdown_stats(s, max_dd, dd_peaks, dd_dur)
    
    # 8. Calcular estadísticas de trades
    s, n_trades, win_rate = _calculate_trade_stats(s, trades_df, pl, returns, durations)
    
    # 9. Calcular métricas avanzadas
    s = _calculate_advanced_metrics(s, n_trades, win_rate, pl)

    # 10. calcular mae y mfe
    _calculate_mae_mfe(s, equity_curve, trades_df, metric_type=mae_metric_type)
    
    # 10. Agregar datos internos
    s.loc['_strategy'] = strategy_instance
    s.loc['_equity_curve'] = equity_df
    s.loc['_trades'] = trades_df
    
    return Stats(s)



    