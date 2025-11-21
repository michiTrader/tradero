import asyncio
import traceback
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from typing import List, NoReturn, Optional

import psutil
from requests.exceptions import ReadTimeout
from pintar import dye

from tradero.models import CryptoSesh, Strategy
from tradero._util import has_internet_connection,  distribution_between_processes


LOG_COLORS = [
    "#B5D6FF", "#D3FFEEFF", "#FEFFD5FF", "#F2C9B7FF", "#9EA2EFFF", 
    "#A0B487FF", "#A3DFE6FF", "#BCF599FF", "#E3A989FF", "#5EDDAEFF", 
    "#6B80BFFF", "#AA429CFF", "#F3D3FFFF", "#D67C2EFF"
]

class StrategyLifecycleManager:
    """Maneja el ciclo de vida de las estrategias (init, start, stop)."""
    
    @staticmethod
    async def handle_keyboard_interrupt(strategies: List[Strategy]) -> None:
        """Maneja la interrupción por teclado cerrando todas las estrategias."""
        for strategy in strategies:
            if strategy.get_status() == "stopped":
                continue
            
            try:
                strategy.stop()
                await strategy.on_stop()
                await asyncio.sleep(0.1)
            except Exception as e:
                strategy.logger.error(
                    f"Error en on_stop de {strategy.__class__.__name__}: {e}"
                )
    
    @staticmethod
    async def handle_readtimeout_error(strategy: Strategy, exception: Exception) -> bool:
        """Maneja errores durante la ejecucion de una estrategia."""

        ATTEMPTS = 30

        # Detener temporalmente la estrategia
        strategy.wait()
        strategy.logger.error(f'Sin conexión a internet. reintentando conexion...')

        attempt = 0
        while attempt < ATTEMPTS:
            if not has_internet_connection():
                attempt += 1 
                await asyncio.sleep(4) 
            else:
                strategy.logger.warning('Conexion restablecida')
                strategy.start()
                return True       
        else:             
            # Si no se pudo restablecer la conexion
            strategy.logger.critical(
                "No se pudo restablecer la conexion. Se deben revisar las "
                "posiciones/ordenes manualmente"
            )

            traceback.print_exc()

            return False

    @staticmethod
    async def handle_unknown_error(strategy: Strategy, exception: Exception) -> None:
        """Maneja errores durante la ejecucion de una estrategia."""
        strategy.logger.error(
            f"Excepción en {strategy.__class__.__name__}: {exception}"
        )
        
        try:
            strategy.stop()
            await strategy.on_stop()
        except Exception as e:
            strategy.logger.error(
                f"Error al ejecutar on_stop de {strategy.__class__.__name__}: {e}"
            )
        
        traceback.print_exc()


class StrategyRunner:
    """Ejecuta estrategias de trading de forma asíncrona."""
    
    def __init__(self, lifecycle_manager: Optional[StrategyLifecycleManager] = None):
        self.lifecycle_manager = lifecycle_manager or StrategyLifecycleManager()
    
    async def initialize_strategy(self, strategy: Strategy) -> bool:
        """Inicializa una estrategia. Retorna True si fue exitoso."""
        try:
            strategy.logger.warning("Inicializando...")
            await strategy.init()
            time.sleep(0.1)
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            await self.lifecycle_manager.handle_unknown_error(strategy, e)
            return False
    
    async def run_strategy_loop(self, strategy: Strategy) -> None:
        """Ejecuta el loop principal de una estrategia."""

        while strategy.get_status() == "live":

            try:
                await strategy.on_live()
            except ReadTimeout as e:
                is_restored = await \
                    self.lifecycle_manager.handle_readtimeout_error(strategy, e)
                if is_restored:
                    continue
                break

            except Exception as e:
                await self.lifecycle_manager.handle_unknown_error(strategy, e)
                break
    
    async def run_single_strategy(self, strategy: Strategy) -> None:
        """Ejecuta una estrategia completa (init + loop)."""
        # Inicializacion
        if not await self.initialize_strategy(strategy):
            return
        
        # Inicio del loop
        strategy.start()
        await asyncio.sleep(0.1)
        
        # Ejecución del loop principal
        await self.run_strategy_loop(strategy)
    
    async def run_multiple_strategies(self, strategies: List[Strategy]) -> None:
        """Ejecuta multiples estrategias en paralelo."""
        tasks = [
            asyncio.create_task(self.run_single_strategy(strategy))
            for strategy in strategies
        ]
        await asyncio.gather(*tasks, return_exceptions=True)


def _run_strategies_in_process(strategies: List[Strategy]) -> NoReturn:
    """Ejecuta estrategias en un proceso aislado."""
    runner = StrategyRunner()
    lifecycle_manager = StrategyLifecycleManager()
    
    try:
        asyncio.run(runner.run_multiple_strategies(strategies))
    except KeyboardInterrupt:
        asyncio.run(lifecycle_manager.handle_keyboard_interrupt(strategies))
    except Exception as e: # XXX
        print(f"\nerror en _run_strategies_in_process\n")


def _assign_log_colors(strategies: List[Strategy]) -> None:
    """Asigna colores a los logs de cada estrategia."""
    for i, strategy in enumerate(strategies):
        strategy.id_color = LOG_COLORS[i % len(LOG_COLORS)]


def  distribution_between_processes(
    strategies: List[Strategy], 
    num_workers: int
) -> List[List[Strategy]]:
    """Distribuye estrategias en grupos para procesamiento paralelo."""
    return [
        strategies[worker_idx::num_workers] 
        for worker_idx in range(num_workers)
    ]


def _run_strategies_multicore(
    strategies: List[Strategy], 
    max_workers: Optional[int] = None
) -> None:
    """Ejecuta estrategias en multiples cores usando ProcessPoolExecutor."""
    if not strategies:
        raise ValueError("La lista de estrategias no puede estar vacía")
    
    # Determinar numero de workers
    if max_workers is None:
        max_workers = psutil.cpu_count(logical=False)
    
    if not isinstance(max_workers, int) or max_workers <= 0:
        raise ValueError("max_workers debe ser un entero positivo")
    
    # Configuracion
    _assign_log_colors(strategies)
    num_workers = min(len(strategies), max_workers)
    strategy_groups =  distribution_between_processes(strategies, num_workers)
    
    # Ejecucion
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop) 
    
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, _run_strategies_in_process, group)
                for group in strategy_groups
            ]
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    except KeyboardInterrupt as e:
        raise e
    finally:
        loop.close()


def run_strategies(
    sesh: CryptoSesh, 
    strategies: List[Strategy], 
    max_workers: Optional[int] = None
) -> None:
    """
    Punto de entrada principal para ejecutar estrategias de trading.
    
    Args:
        sesh: Sesión de criptomonedas
        strategies: Lista de clases de estrategias a ejecutar
        max_workers: Número máximo de procesos paralelos (None = todos los cores)
    """
    # Instanciar estrategias
    strategy_instances = [StrategyClass(sesh) for StrategyClass in strategies]
    
    try:
        _run_strategies_multicore(strategy_instances, max_workers)
    except KeyboardInterrupt:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(
            timestamp, 
            dye('>>> KeyboardInterrupt <<<','#C1AEFF', None, [1,3]), 
        )
    except Exception as e:
        print(f"Error en run_strategies: {e}")
        traceback.print_exc()

