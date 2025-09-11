from tradero.lib import get_str_datetime


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
        
def info_log(*args, **kwargs): # info_print
    time = get_str_datetime()
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

