import numpy as np
from numba import njit

@njit
def count_closed_resample_bars_nb(
    timestamps_values: np.ndarray, idx: int, origin_minutes_tf: int, resample_minutes_tf: int
    ):
    """ 
        Para extraer la cantidad de velas resampleadas cerradas disponibles (el índice de acceso del resampleado).
        
        Esta función calcula cuántas velas resampleadas están completamente cerradas hasta el índice dado.
        Solo cuenta las velas que han sido completamente formadas, excluyendo la vela actual incompleta.
        
        Args:
            timestamps_values (np.ndarray): Array de timestamps en nanosegundos
            idx (int): Índice actual en el array de timestamps
            origin_interval (int): Intervalo de origen en minutos
            resample_interval (int): Intervalo de resampleo en minutos
        
        Returns:
            int: Número de velas resampleadas cerradas disponibles
        
        Note:
            - Verifica si la última caja está completa antes de contarla
            - Usa milisegundos como unidad base de tiempo
            - Retorna 0 si no hay suficientes datos
    """
    if len(timestamps_values) == 0:
        return 0

    # Usar los valores en milisegundos directamente
    primer_ms = timestamps_values[0]
    ultimo_ms = timestamps_values[idx]
    
    # Convertir a minutos (1 minuto = 60 * 1000 milisegundos)
    MS_POR_MINUTO = 60_000
    
    primer_minuto = primer_ms // MS_POR_MINUTO
    ultimo_minuto = ultimo_ms // MS_POR_MINUTO
    
    # Calcular cajas usando el intervalo de resampleo
    primera_caja = primer_minuto // resample_minutes_tf
    ultima_caja = ultimo_minuto // resample_minutes_tf
    
    # Verificar si la última caja está completa
    ultimo_minuto_de_ultima_caja = (ultima_caja + 1) * resample_minutes_tf - origin_minutes_tf
    
    if ultimo_minuto >= ultimo_minuto_de_ultima_caja:
        return ultima_caja - primera_caja + 1
    else:
        return max(0, ultima_caja - primera_caja)

# funciona con intervalos superiores?
@njit
def count_available_resample_bars_nb(
    timestamps_values: np.ndarray, idx: int, origin_minutes_tf: int, resample_minutes_tf: int
    ):
    """ 
    Para extraer la cantidad de velas resampleadas disponibles (cerradas + incompletas).
    """
    if len(timestamps_values) == 0:
        return 0
        
    # Usar los valores en milisegundos directamente
    primer_ms = timestamps_values[0]
    ultimo_ms = timestamps_values[idx]
    
    # Convertir a minutos
    MS_POR_MINUTO = 60_000
    
    primer_minuto = primer_ms // MS_POR_MINUTO
    ultimo_minuto = ultimo_ms // MS_POR_MINUTO
    
    # CORRECCIÓN: Ajustar los minutos al intervalo de origen
    primer_minuto_ajustado = (primer_minuto // origin_minutes_tf) * origin_minutes_tf
    ultimo_minuto_ajustado = (ultimo_minuto // origin_minutes_tf) * origin_minutes_tf
    
    # Calcular cajas usando el intervalo de resampleo desde el primer minuto ajustado
    primera_caja = primer_minuto_ajustado // resample_minutes_tf
    ultima_caja = ultimo_minuto_ajustado // resample_minutes_tf
    
    # Retornar TODAS las cajas (completas e incompletas)
    return ultima_caja - primera_caja + 1
