
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
    direction = "buy" if entry < tp else "sell"
    
    if direction == "buy":
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
    direction = "buy" if entry > sl else "sell"
    
    if direction == "buy":
        sl_dist = entry - sl
        return entry + sl_dist * ratio
    else:
        sl_dist = sl - entry  
        return entry - sl_dist * ratio

def entry_adjustment(entry_original, sl, adjust_factor=0):
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
    direction = "buy" if entry_original > sl else "sell"
    sl_dist = abs(entry_original - sl)
    
    if direction == "buy":
        # Para LONG: ajuste positivo sube la entrada (más conservador, más lejos del SL)
        return entry_original + (sl_dist * adjust_factor)
    else:
        # Para SHORT: ajuste positivo baja la entrada (más conservador, más lejos del SL)  
        return entry_original - (sl_dist * adjust_factor)


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

def sl_adjustment(entry, sl, adjustment=1):
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

def tp_adjustment(entry, tp, adjustment=1):
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
