from pathlib import Path
from ..lib import read_csv_ohlc

_current_dir = Path(__file__).parent
_data_dir = _current_dir / "data"

BTCUSDT = read_csv_ohlc(str(_data_dir / "BTCUSDT_1m_2025_JUNE.csv"))
ADAUSDT = read_csv_ohlc(str(_data_dir / "ADAUSDT_1m_2025_JUNE.csv"))
MESZ25 = read_csv_ohlc(str(_data_dir / "MESZ25_1m_2025_OCT.csv"))

# Lista de todos los datasets disponibles
__all__ = ['BTCUSDT', 'ADAUSDT', 'MESZ25']