
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
