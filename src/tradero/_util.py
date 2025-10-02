from tradero.lib import get_str_datetime_now
from pintar import dye, Stencil, Brush
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
from numbers import Number
import requests
import logging
from pathlib import Path
import json
from pintar import dye, Brush, Stencil

def try_(lazy_func, default=None, exception=Exception):
    try:
        return lazy_func()
    except exception:
        return default


# TODO: eliminar
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


# TODO: ordenar
# Abrimos el archivo JSON que contiene el tema de colores para los logs
with open(Path.cwd() / 'config/custom_log_color_theme.json') as f:
    CUSTOM_COLOR_THEME = json.load(f)

def deep_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_dict_update(d[k], v)
        else:
            d[k] = v
    return d

class LevelColorFormatTheme:
    def __init__(self, config: dict = None, color_theme: dict = None, no_color=False):
        self._color_theme = CUSTOM_COLOR_THEME if not color_theme else color_theme 
        self._config = config
        self._level_format = None # a la espera 
        
        # Configurar el color theme 
        if config is not None:
            self._color_theme = deep_dict_update(self._color_theme, config)
        
        # Relacionar los niveles de log con los temas de color definidos en el JSON
        levels_color_themes = {
            logging.DEBUG : self._color_theme['debug'],
            logging.INFO : self._color_theme['info'],
            logging.WARNING : self._color_theme['warning'],
            logging.ERROR : self._color_theme['error'],
            logging.CRITICAL : self._color_theme['critical'],
        }

        # Definimos una configuración de "caracteres" para el formato de los logs.
        # Cada tupla tiene:
        # (clave interna, clave usada en el tema, string de formato o carácter)
        CHARACTERS_CONFIG = (
            ('name', '%(name)s:'),       # nombre del logger (:%(name)s:)
            ('asctime', '%(asctime)s'), # fecha y hora del log
            ('levelname', '%(levelname)s'), # nivel del log (DEBUG, INFO, etc.)
            ('message', '%(message)s'), # mensaje del log
            ('circle', '%(circle)s'),          # bloque visual para resaltar
            ('block', '%(block)s'),          # bloque visual para resaltar
            ('bar', '%(bar)s'),              # barra separadora
        )
        
        # Diccionario final que contendrá los formatos generados para cada nivel de log
        self._level_format = {}

        # Recorremos cada tema según el nivel de log
        for lvl, c_theme in levels_color_themes.items():

            f_results = {}
            # Para cada elemento (nombre, fecha, nivel, mensaje, etc.) se aplica un "Stencil"
            # que colorea el texto con los colores definidos en el JSON
            for key, cha in CHARACTERS_CONFIG:
                if no_color:
                    f_results[key] = cha  
                else:
                    f_results[key] = Stencil(string=cha).spray(
                        tex_color=c_theme[key]['text_color'],  # color del texto
                        bg_color=c_theme[key]['bg_color'],     # color de fondo
                        style=c_theme[key]['style'],     # estilo
                    )

            # Construimos el formato final del log, por ejemplo:
            # █logger_name 2025-10-02 |INFO| Mensaje de log
            self._level_format[lvl] = (
                f"{f_results["block"]}{f_results["name"]} {f_results["asctime"]} " 
                f"{f_results["bar"]}{f_results["levelname"]}{f_results["bar"]} {f_results["message"]}"
            )

    def get_level_format(self) -> dict:
        return self._level_format

class CustomFormatter(logging.Formatter):
    """
    Un formateador personalizado que usa diferentes formatos
    basados en el nivel del registro.
    """

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True, defaults=None, format_theme: LevelColorFormatTheme = None):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate, defaults=defaults)
        self._format_theme: LevelColorFormatTheme = format_theme

        if format_theme:
            self.level_format = self._format_theme.get_level_format()

    def format(self, record):
        if not hasattr(record, 'block'):
            record.block = ':'

        if not hasattr(record, 'bar'):
            record.bar = '|'

        if not hasattr(record, 'circle'):
            record.circle = '°'

        # Obtiene la cadena de formato correcta, usando un formato por defecto
        # si el nivel del registro no está en el diccionario.
        fmt_string = self.level_format.get(record.levelno, self.level_format[logging.INFO])
        # Crea una nueva instancia de Formatter con la cadena de formato seleccionada
        formatter = logging.Formatter( 
            fmt=fmt_string,
            datefmt='%Y-%m-%d %H:%M:%S',
            )
        # Formatea el registro y devuelve el resultado
        return formatter.format(record)

class BacktestLogger(logging.Logger):
    def __init__(self, name, strategy):
        super().__init__(name)
        self.strategy: 'Strategy' = strategy  # la estrategia que tiene el timestamp de la vela actual

    def makeRecord(self, *args, **kwargs):
        record = super().makeRecord(*args, **kwargs)
        if hasattr(self.strategy.sesh, "now"):
            # Sobreescribir con el tiempo de la vela actual
            record.created = self.strategy.sesh.now.timestamp()
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




