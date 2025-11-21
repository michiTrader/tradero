

def sl_by_ratio(entry, tp, ratio=1/1):
    """
    Calcula el Stop Loss basado en la distancia al Take Profit y un ratio determinado.
    
    Parámetros:
    - entry (float): Precio de entrada
    - tp (float): Precio de Take Profit
    - ratio (float): Ratio SL/TP (default: 1/1)
    
    Returns:
    - float: Precio del Stop Loss calculado
    """
    is_buy = tp > entry
    
    if is_buy:
        tp_dist = tp - entry
        return entry - tp_dist * ratio
    else:
        tp_dist = entry - tp
        return entry + tp_dist * ratio
def tp_by_ratio(entry, sl, ratio=1/1):
    """
    Calcula el Take Profit basado en la distancia al Stop Loss y un ratio determinado.
    
    Parámetros:
    - entry (float): Precio de entrada
    - sl (float): Precio de Stop Loss  
    - ratio (float): Ratio TP/SL (default: 1/1)
    
    Returns:
    - float: Precio del Take Profit calculado
    """
    is_buy = entry > sl
    
    if is_buy:
        sl_dist = entry - sl
        return entry + sl_dist * ratio
    else:
        sl_dist = sl - entry  
        return entry - sl_dist * ratio
def size_by_risk(cash, risk, entry, sl):
    """
        Calcula el tamaño de la posición basado en el riesgo porcentual.
        
        Parámetros:
        - cash (float): Capital disponible
        - risk (float): Porcentaje de riesgo (ej: 0.01 para 1%)
        - entry (float): Precio de entrada
        - sl (float): Precio del Stop Loss
        
        Returns:
        - float: Tamaño de la posición
    """
    risk_amount = cash * risk
    sl_dist = abs(entry - sl)
    size = risk_amount / sl_dist
    return size


def adjust_sl(entry, sl, adjustment=1):
    """
    Ajusta el Stop Loss basado en la distancia desde el entry.
    
    Args:
        entry: Precio de entrada
        sl: Stop Loss original
        adjustment: Factor de ajuste (1 = misma distancia)
        
    Returns:
        Nuevo Stop Loss ajustado
        
    Examples:
        >>> sl_adjustment(100, 90, 0.5)  # COMPRA
        95.0
        >>> sl_adjustment(100, 90, 1.5)  # VENTA
        85.0
    """
    is_buy = entry > sl
    new_sl_dist = abs(entry - sl) * adjustment
    return (entry - new_sl_dist) if is_buy else (entry + new_sl_dist)
def adjust_tp(entry, tp, adjustment=1):
    """
    Ajusta el Take Profit basado en la distancia desde el entry.
    
    Args:
        entry: Precio de entrada
        tp: Take Profit original
        adjustment: Factor de ajuste (1 = misma distancia)
        
    Returns:
        Nuevo Take Profit ajustado
        
    Examples:
        >>> tp_adjustment(100, 120, 0.5)  # COMPRA
        95.0
        >>> tp_adjustment(100, 120, 1.5)  # VENTA
        130.0
    """
    is_buy = tp > entry
    new_tp_dist = abs(tp - entry) * adjustment
    return (entry + new_tp_dist) if is_buy else (entry - new_tp_dist)
def adjust_entry(entry_original, sl, adjust_factor=0):
    """
    Calcula el precio de entrada ajustado basado en la distancia al Stop Loss.
    
    Parámetros:
    - entry_original (float): Precio de entrada original
    - sl (float): Precio de Stop Loss
    - adjust_factor (float): Factor de ajuste basado en la distancia al SL
                           - 0.0 = Sin ajuste (entrada original)
                           - Valores positivos = Entrada más conservadora (más lejos del SL)
                           - Valores negativos = Entrada más agresiva (más cerca del SL)
    
    Returns:
    - float: Precio de entrada ajustado
    """
    is_buy = entry_original > sl
    sl_dist = abs(entry_original - sl)
    
    if is_buy:
        # Para LONG: ajuste positivo sube la entrada (más conservador, más lejos del SL)
        return entry_original + (sl_dist * adjust_factor)
    else:
        # Para SHORT: ajuste positivo baja la entrada (más conservador, más lejos del SL)  
        return entry_original - (sl_dist * adjust_factor)
def adjust_size(size, step, min_size=0, max_size=float('inf')):
    """
        Ajusta un tamaño dado para que sea un múltiplo de un 'step' específico,
        asegurándose de que el resultado se mantenga dentro de un rango definido
        por 'min_size' y 'max_size'.

        Args:
            size (int or float): El tamaño inicial a ajustar.
            step (int or float): El incremento o paso al que se debe ajustar el tamaño.
            min_size (int or float): El tamaño mínimo permitido.
            max_size (int or float): El tamaño máximo permitido.

        Returns:
            int or float: El tamaño ajustado que cumple con los criterios de paso y límites.
    """
    # Usar round() para evitar errores de precisión de punto flotante
    size = round((size // step) * step, len(str(step).split('.')[-1]))
    # Respetar min y max
    size = max(min_size, min(size, max_size))
    return float(size)
def adjust_leverage(notional, risk_limit):
    """ Ajusta el apalancamiento maximo basado en el size y el diccionario de limites de riesgo (risk_limit). """
    # Filtrar apalancamientos disponibles basados en el size
    # filter_available_leverages = list(map(lambda x: x["maxLeverage"] if (float(x["riskLimitValue"]) > size) else None, risk_limit))

    leverages = dict(
        map(
            lambda x: (float(x["riskLimitValue"]), float(x["maxLeverage"])), risk_limit))

    available_leverages = [val for key, val in leverages.items() if key > notional]
    # Dropear valores None y extraer el maximo apalancamiento
    if available_leverages:
        max_leverage = max(available_leverages)
    else:
        max_leverage = max(leverages.values())
    return int(max_leverage // 1)


