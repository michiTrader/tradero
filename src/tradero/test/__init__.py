from pathlib import Path
from ..util import read_csv_ohlc

_current_dir = Path(__file__).parent

BTCUSDT = read_csv_ohlc(str(_current_dir / "BTCUSDT_1m_2025_JUNE.csv"))
ADAUSDT = read_csv_ohlc(str(_current_dir / "ADAUSDT_1m_2025_JUNE.csv"))

# Lista de todos los datasets disponibles
__all__ = ['BTCUSDT', 'ADAUSDT']