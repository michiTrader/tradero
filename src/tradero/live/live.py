import asyncio 
from shutil import ExecError 
import traceback 
from pandas.tseries.frequencies import key 
from pybit.unified_trading import HTTP 
import pandas as pd 
import psutil 
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor  
from typing import List, Never, Coroutine, Any 
from functools import partial 
from tradero.models import CryptoSesh, Strategy 
from tradero._util import ColorWayGenerator, eprint, info_log, try_, has_internet_connection
import time 
from datetime import datetime
from requests.exceptions import ReadTimeout
from pintar import Stencil, dye, Brush, RGB, HEX

RED_CIRCLE = '\033[91;1m°\033[0m'
GREEN_CIRCLE = '\033[92;1m°\033[0m'

def _run_strategies_in_process(strategies_instances: List[Strategy]) -> Never: 
    async def keyboard_interrupt_protocol(strategies_instances: List[Strategy] | Strategy):
        # Procesar parametro
        if isinstance(strategies_instances, Strategy):
            strategies_instances = [strategies_instances]

        for stgy in strategies_instances:
            if stgy.status == "stopped":
                continue

            try:
                stgy.stop()
                stgy.logger.warning(f"{RED_CIRCLE} on_stop. Deteniendo estrategia.")  # 🔴
                await stgy.on_stop()

                await asyncio.sleep(0.1)
            except Exception as e:
                stgy.logger.error(f"Error en on_stop de {stgy.__class__.__name__}")

    async def error_protocol(strategy_instance: Strategy, exception: Exception = None):
        strategy_instance.logger.error(f"Excepción en {strategy_instance.__class__.__name__}: {exception}" ) 
        try:
            strategy_instance.logger.warning(f"{RED_CIRCLE} on_stop. Deteniendo estrategia.")  # 🔴
            strategy_instance.stop()
            await strategy_instance.on_stop()
            # await asyncio.sleep(0.1)
        except Exception as e:
            strategy_instance.logger.error(f"Error en al ejecutar on_stop de {strategy_instance.__class__.__name__}")

        traceback.print_exc() # TODO: darle un mejor manejo
                        
    async def run_single_strategy(strategy_instance: Strategy) -> Never: #
        
        try:
            try:
                strategy_instance.logger.warning(f"Inicializando...")  
                await strategy_instance.init()
                time.sleep(0.1)
                await asyncio.sleep(0.5) 
                
            except Exception as e:
                await error_protocol(strategy_instance, e)

            strategy_instance.start()
            strategy_instance.logger.warning(f"{GREEN_CIRCLE} on_live...") # 🟢
            await asyncio.sleep(0.1) 
            while True:
                if strategy_instance.status == "live":
                    try:
                        await strategy_instance.on_live()
                    except ReadTimeout as e:
                        if has_internet_connection():
                            await asyncio.sleep(7)
                        strategy_instance.logger.error(f"Sin conexion a internet se deben cerrar las operaciones/ordenes manualmente") 
                    except Exception as e:
                        await error_protocol(strategy_instance, e)
        finally:
            pass
        # except Exception as e:
        #     # TODO: usar taceback.format_exc() para guardar la informacion de las lineas de codigo de la excepcion
        #     # TODO: enviar error a celular y guardar format_exc 
        #     # on_exit de emergencia si falla una estrategia
        #     await error_protocol(strategy_instance, e)

    async def main_execution(strategies_instances: List) -> Coroutine[Any, Any, Never]:
        running_tasks = []
        for s in strategies_instances:
            running_tasks.append(asyncio.create_task(run_single_strategy(s)))

        await asyncio.gather(*running_tasks, return_exceptions=True)
    
    try:
        asyncio.run(main_execution(strategies_instances))
    except KeyboardInterrupt:
        asyncio.run(keyboard_interrupt_protocol(strategies_instances)) 

def _run_strategies_multicore(strategies_instances: List[Strategy], max_workers: int = None):

    if max_workers is None:
        max_workers = psutil.cpu_count(logical=False) # only cores

    assert bool(strategies_instances)
    assert isinstance(max_workers, int) and max_workers > 0

    # setear colores de log
    colors = ["#A3DFE6FF", "#EDC77DFF", "#C997DDFF", "#DE9BA8FF", "#9EA2EFFF", "#A0B487FF", "#71CFD7FF", 
        "#BCF599FF", "#E3A989FF", "#5EDDAEFF", "#6B80BFFF", "#AA429CFF", "#794AAFFF", "#D67C2EFF"]
    [setattr(s, "log_color", colors[i % len(colors)]) for i, s in enumerate(strategies_instances)]
        
    # Distribuir las estrategias en grupos WWW
    total_strategies = len(strategies_instances)
    max_works_to_use = min(total_strategies, max_workers) #; print(f"works_to_use: {max_works_to_use}")
    # Distribuir las tareas por grupos
    strategies_groups = [strategies_instances[n_wrk::max_works_to_use] for n_wrk in range(max_works_to_use)] # ; print(f"strategies_groups: {[list(map(lambda s: s.__class__.__name__, group)) for group in strategies_groups]}")

    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=max_works_to_use) as executor:
        running_tasks = []
        for group in strategies_groups:
            running_tasks.append(
                loop.run_in_executor(executor, _run_strategies_in_process, group)
            )

    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(asyncio.gather(*running_tasks, return_exceptions=True))
    except KeyboardInterrupt:
        print("KeyboardInterrupt in run_strategies_multicore")
    finally:
        loop.close()

def run_strategies(sesh: CryptoSesh, strategies: List[Strategy], max_workers: int = None):
    # try:
    # intanciar estrategias
    strategies_instances = [s(sesh) for s in strategies]
    
    try:
        _run_strategies_multicore(strategies_instances=strategies_instances, max_workers=max_workers)
    except KeyboardInterrupt:
        now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print() ; print(now_time, '- '*5, Stencil('Interrupción por teclado').spray('#9983FFFF'), '- '*15)
    except Exception as e:
        print(f"Error en run_strategies {e}")
    # except KeyboardInterrupt:
    #     print("run_strategies wiuuu")


