import asyncio
from shutil import ExecError
from pandas.io.sql import execute
from pybit.unified_trading import HTTP
import pandas as pd
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Never, Coroutine, Any
from functools import partial
from tradero._util import eprint, iprint
from tradero.models import CryptoSesh, Strategy
from tradero._util import ColorWayGenerator

def _run_strategies_in_process(strategies_instances: List[Strategy]) -> Never: 
    async def interrupt_protocol(strategies_instances: List[Strategy]):
        for s in strategies_instances:
            if s.is_closed:
                continue

            try:
                s.stop()
                await s.on_stop()
                s.log(f"Cerrando estrategia on_stop", s.status) # 🔴

                await asyncio.sleep(0.1)
            except Exception as e:
                s.log(f"Error en on_stop de {s.__class__.__name__}: {e}", type="error")

    async def run_single_strategy(strategy_instance: Strategy) -> Never: #

        strategy_name = strategy_instance.__class__.__name__

        try:
            strategy_instance.log(f"init 🔵")  
            await strategy_instance.init()
            await asyncio.sleep(0.1)

            strategy_instance.start()
            strategy_instance.log(f"on_live 🟢") 
            await asyncio.sleep(0.1) 

            while True:
                await strategy_instance.on_live()
                await asyncio.sleep(0.5) # TODO: volver el sleep un parametro de usuario

        except KeyboardInterrupt:
            iprint("interrupcion por teclado detectada en run_single_strategy") 
        except Exception as e:
            strategy_instance.log(f"Excepción en {strategy_name}: {e}", type="error")
            # TODO: usar taceback.format_exc() para guardar la informacion de las lineas de codigo de la excepcion
            # TODO: enviar error a celular y guardar format_exc 
            # on_exit de emergencia si falla una estrategia
            await interrupt_protocol([strategy_instance])

    async def main_execution(strategies_instances: List) -> Coroutine[Any, Any, Never]:
        running_tasks = []
        for s in strategies_instances:
            running_tasks.append(asyncio.create_task(run_single_strategy(s)))
            
        try:
            await asyncio.gather(*running_tasks, return_exceptions=False)
        except Exception as e:
            eprint(f"Error en run_group_of_strategies: {e}")
            
    try:
        return asyncio.run(main_execution(strategies_instances))
    except KeyboardInterrupt:
        asyncio.run(interrupt_protocol(strategies_instances)) # no funciona

def _run_strategies_multicore(strategies_instances: List[Strategy], max_workers: int = None):

    if max_workers is None:
        max_workers = psutil.cpu_count(logical=False) # only cores

    assert bool(strategies_instances)
    assert isinstance(max_workers, int) and max_workers > 0

    # setear colores de log
    colors = ["#93D8E0FF", "#EACC92FF", "#C997DDFF", "#D88C9BFF", "#9EA2EFFF", "#B5F7B8FF", "#FDCBB0FF", "#88C4AFFF"]
    [setattr(s, "log_color", colors[i % 8]) for i, s in enumerate(strategies_instances)]
        
    # Distribuir las estrategias en grupos 
    total_strategies = len(strategies_instances)
    works_to_use = min(total_strategies, max_workers) ; print(f"works_to_use: {works_to_use}")
    strategies_groups = [strategies_instances[n_wrk::works_to_use] for n_wrk in range(works_to_use)] ; print(f"strategies_groups: {[list(map(lambda s: s.__class__.__name__, group)) for group in strategies_groups]}")

    loop = asyncio.new_event_loop()
    with ProcessPoolExecutor(max_workers=works_to_use) as executor:
        running_tasks = []
        for group in strategies_groups:
            running_tasks.append(
                loop.run_in_executor(executor, _run_strategies_in_process, group)
            )
    
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(asyncio.gather(*running_tasks, return_exceptions=True))
    except KeyboardInterrupt:
        iprint("KeyboardInterrupt in run_strategies_multicore")
    except Exception as e:
        iprint("aaaaaaaaaaaa")
    finally:
        loop.close()

def run_strategies(sesh: CryptoSesh, strategies: List[Strategy]):
    # intanciar estrategias
    strategies_instances = [s(sesh) for s in strategies]
    
    try:
        _run_strategies_multicore(strategies_instances=strategies_instances) 
    except KeyboardInterrupt:
        iprint("\033[91m ======== Interrupción por teclado  ======== \033[0m")
    except Exception as e:
        print(f"Error en run_strategies {e}")
    


