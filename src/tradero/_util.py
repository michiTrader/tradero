from tradero.lib import get_str_datetime_now
from pintar import dye
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
from numbers import Number

#region Strategy LOG MANAGER 
class LogType(Enum):
    """Tipos de log disponibles"""
    STRATEGY = "strategy"
    ERROR = "error" 
    INFO = "info"
    NORMAL = "normal"

class StrategyStatus(Enum):
    """Estados posibles de una estrategia"""
    WAITING = "waiting"
    LIVE = "live"
    STOPPED = "stopped"
    OTHER = "other"

@dataclass
class ColorScheme:
    """Esquema de colores para el logging"""
    time: str
    circle: str
    circle_bg: Optional[str]
    message: str

class LogColorManager:
    """Gestor de colores para diferentes tipos de log y estados"""
    
    # Colores por tipo de log
    LOG_COLORS = {
        LogType.STRATEGY: "#7F838E",
        LogType.INFO: "#DEF462FF", 
        LogType.ERROR: "#F82141",
        LogType.NORMAL: "#A6AAB5FF"
    }
    
    # Colores por estado de estrategia
    STATUS_COLORS = {
        StrategyStatus.WAITING: "#CF8F18FF",
        StrategyStatus.LIVE: "#1EFF3CFF",
        StrategyStatus.STOPPED: "#E31F3DFF",
        StrategyStatus.OTHER: "#7972FFFF"
    }
    
    @classmethod
    def get_colors(cls, log_type: LogType, status: str, strategy_color: str) -> ColorScheme:
        """Obtiene el esquema de colores basado en el tipo de log y estado"""
        
        # Colores base
        time_color = cls.LOG_COLORS.get(log_type, "#7F838E")
        message_color = strategy_color
        
        # Color del círculo basado en el estado
        try:
            status_enum = StrategyStatus(status)
            circle_color = cls.STATUS_COLORS.get(status_enum, "#495DE4")
        except ValueError:
            circle_color = "#495DE4"
        
        circle_bg = None
        
        # Caso especial para errores
        if log_type == LogType.ERROR:
            time_color = message_color = circle_color = cls.LOG_COLORS[LogType.ERROR]
            circle_bg = circle_color

        if log_type == LogType.INFO:
            time_color = message_color = circle_color = cls.LOG_COLORS[LogType.INFO]
            circle_bg = circle_color
        
        return ColorScheme(
            time=time_color,
            circle=circle_color,
            circle_bg=circle_bg,
            message=message_color
        )

class StrategyLogManager:
    """Gestor principal de logging para estrategias"""
    
    def __init__(self, strategy_name: str, strategy_color: str, status: str):
        self.strategy_name = strategy_name
        self.strategy_color = strategy_color or "#72D3F1"
        self.status = status
    
    def log(self, *args, **kwargs):
        """Método principal de logging"""
        # Configuración por defecto
        defaults = {
            "end": "\n",
            "sep": " ",
            "flush": False,
            "type": "strategy",
            "time": get_str_datetime_now(),
        }
        
        config = {**defaults, **kwargs}
        log_type_str = config["type"]

        # Caso especial para log normal
        if log_type_str == 'normal':
            print_kwargs = self._get_print_kwargs(**config)
            return print(*args, **print_kwargs)
        
        # Procesar log con formato
        try:
            log_type = LogType(log_type_str)
        except ValueError:
            log_type = LogType.STRATEGY
        
        # Obtener colores y formatear
        colors = LogColorManager.get_colors(log_type, self.status, self.strategy_color)
        timestamp = config["time"]
        name_tag = f" {self.strategy_name} "
        print_kwargs = self._get_print_kwargs(**config)
        
        self._print_formatted_log(colors, timestamp, name_tag, *args, **print_kwargs)
    
    def _get_print_kwargs(self, **kwargs) -> Dict:
        """Filtra argumentos válidos para print()"""
        valid_keys = {"end", "sep", "flush"}
        return {k: v for k, v in kwargs.items() if k in valid_keys}
    
    def _print_formatted_log(self, colors: ColorScheme, timestamp: str, name_tag: str, *args, **print_kwargs):
        """Imprime el log con formato completo"""
        # Círculo de estado
        print(dye("° ", tex=colors.circle, bg=colors.circle_bg), end='')
        
        # Timestamp
        dye.start(tex=colors.time, bg=None, sty=None)
        print(timestamp, end=' ')
        
        # Etiqueta del nombre
        dye.start(tex="#000000", bg=self.strategy_color, sty="bold")
        print(name_tag, end=f"{dye.end(return_repr=True)} ")
        
        # Mensaje
        dye.start(tex=colors.message, bg=None, sty="bold")
        print(*args, **print_kwargs)
        
        dye.end()

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


def info_log(*args, **kwargs): # info_print
    time = get_str_datetime_now()
    first_c = "\033[93m[ info]\033[0m"
    print()
    print(f"  {time} {first_c}" + "\033[93;3m", *args, "\033[0m", **kwargs)
    print()

def eprint(*args, **kwargs): # exception/error print
    first_c = "\033[91;1m[error]\033[0m"
    print(first_c + "\033[91;3m", *args, "\033[0m", **kwargs)

def execution_time_measure(func, iterations=100):
    start = time.time()
    for i in range(iterations):
        func()
    end = time.time()
    return (end - start) / iterations


