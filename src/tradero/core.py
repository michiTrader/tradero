import pandas as pd 
from typing import Callable, Union, Sequence 
from numbers import Number 
from .ta import PivotIndicator 
from .models import DataOHLC, CryptoSesh 
from .lib import  find_minutes_timeframe, minutes2timeframe 

from .models import Strategy 

def crossover(series1: Sequence, series2: Sequence) -> bool:
    """
    Return `True` if `series1` just crossed over (above)
    `series2`.

        >>> crossover(self.data.Close, self.sma)
        True
    """
    series1 = (
        series1.values if isinstance(series1, pd.Series) else
        (series1, series1) if isinstance(series1, Number) else
        series1)
    series2 = (
        series2.values if isinstance(series2, pd.Series) else
        (series2, series2) if isinstance(series2, Number) else
        series2)
    try:
        return series1[-2] < series2[-2] and series1[-1] > series2[-1]  # type: ignore
    except IndexError:
        return False

def cross(series1: Sequence, series2: Sequence) -> bool:
    """
    Return `True` if `series1` and `series2` just crossed
    (above or below) each other.

        >>> cross(self.data.Close, self.sma)
        True

    """
    return crossover(series1, series2) or crossover(series2, series1)


