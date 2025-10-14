# import logging
# from pathlib import Path
# from pintar import dye, Brush, Stencil
# import json

    
# # Abrimos el archivo JSON que contiene el tema de colores para los logs
# with open(Path.cwd() / 'config/log_color_theme.json') as f:
#     CUSTOM_COLOR_THEME = json.load(f)

# class LogLevelFormatTheme:
#     def __init__(self, config: dict = None, color_theme: dict = None) -> None:
#         self._color_theme = CUSTOM_COLOR_THEME if not color_theme else color_theme 
#         self._config = config
#         self._level_format = None # a la espera 
        
#         # Configurar el color theme 
#         if config is not None:
#             self._color_theme.update(config)
        
#         # Relacionar los niveles de log con los temas de color definidos en el JSON
#         levels_color_themes = {
#             logging.DEBUG : self._color_theme['debug'],
#             logging.INFO : self._color_theme['info'],
#             logging.WARNING : self._color_theme['warning'],
#             logging.ERROR : self._color_theme['error'],
#             logging.CRITICAL : self._color_theme['critical'],
#         }

#         # Definimos una configuración de "caracteres" para el formato de los logs.
#         # Cada tupla tiene:
#         # (clave interna, clave usada en el tema, string de formato o carácter)
#         CHARACTERS_CONFIG = (
#             ('name', ':%(name)s:'),       # nombre del logger (:%(name)s:)
#             ('asctime', '%(asctime)s'), # fecha y hora del log
#             ('levelname', '%(levelname)s'), # nivel del log (DEBUG, INFO, etc.)
#             ('message', '%(message)s'), # mensaje del log
#             ('block', '█'),          # bloque visual para resaltar
#             ('bar', '|'),              # barra separadora
#         )
        
#         # Diccionario final que contendrá los formatos generados para cada nivel de log
#         self._level_format = {}

#         # Recorremos cada tema según el nivel de log
#         for lvl, c_theme in levels_color_themes.items():

#             f_results = {}
#             # Para cada elemento (nombre, fecha, nivel, mensaje, etc.) se aplica un "Stencil"
#             # que colorea el texto con los colores definidos en el JSON
#             for key, cha in CHARACTERS_CONFIG:
#                 f_results[key] = Stencil(string=cha).spray(
#                     tex_color=c_theme[key]['text_color'],  # color del texto
#                     bg_color=c_theme[key]['bg_color']     # color de fondo
#                 )

#             # Construimos el formato final del log, por ejemplo:
#             # █logger_name 2025-10-02 |INFO| Mensaje de log
#             self._level_format[lvl] = (
#                 f'{f_results["block"]}{f_results["name"]} {f_results["asctime"]} '
#                 f'{f_results["bar"]}{f_results["levelname"]}{f_results["bar"]} {f_results["message"]}'
#             )

#     @property
#     def level_format(self):
#         return self._level_format
