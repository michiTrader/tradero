"""
Ejemplo de uso del manejo de discontinuidades temporales en gráficos de trading.

Este ejemplo demuestra cómo usar las nuevas funcionalidades para manejar huecos
en los datos de mercado causados por pausas, fines de semana, o cambios bruscos de tiempo.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tradero.models import DataOHLC
from tradero.plotting import plot, analyze_time_gaps, print_gap_analysis

def create_sample_data_with_gaps():
    """Crea datos de ejemplo con discontinuidades temporales simuladas"""
    
    # Crear datos base de 5 minutos
    start_date = datetime(2024, 1, 15, 9, 0)  # Lunes 9:00 AM
    dates = []
    
    # Sesión de mañana (9:00 - 12:00)
    current_date = start_date
    while current_date.hour < 12:
        dates.append(current_date)
        current_date += timedelta(minutes=5)
    
    # HUECO: Pausa de almuerzo (12:00 - 14:00) - 2 horas
    current_date = current_date.replace(hour=14, minute=0)
    
    # Sesión de tarde (14:00 - 17:00)
    while current_date.hour < 17:
        dates.append(current_date)
        current_date += timedelta(minutes=5)
    
    # HUECO: Fin de día hasta siguiente día (17:00 - 9:00 siguiente día) - 16 horas
    current_date = current_date.replace(day=16, hour=9, minute=0)
    
    # Segundo día - sesión completa con algunos huecos menores
    while current_date.hour < 17:
        dates.append(current_date)
        current_date += timedelta(minutes=5)
        
        # Simular algunos huecos menores aleatorios
        if np.random.random() < 0.05:  # 5% probabilidad de hueco menor
            current_date += timedelta(minutes=15)  # Hueco de 15 minutos
    
    # HUECO EXTREMO: Fin de semana (viernes 17:00 - lunes 9:00) - ~64 horas
    current_date = current_date.replace(day=19, hour=9, minute=0)  # Lunes siguiente
    
    # Tercer día con datos normales
    while current_date.hour < 17 and len(dates) < 200:
        dates.append(current_date)
        current_date += timedelta(minutes=5)
    
    # Generar datos OHLC sintéticos
    np.random.seed(42)  # Para reproducibilidad
    n_points = len(dates)
    
    # Precio base con tendencia y volatilidad
    base_price = 100
    price_changes = np.random.normal(0, 0.5, n_points).cumsum()
    close_prices = base_price + price_changes
    
    # Generar OHLC basado en close
    opens = close_prices + np.random.normal(0, 0.2, n_points)
    highs = np.maximum(opens, close_prices) + np.abs(np.random.normal(0, 0.3, n_points))
    lows = np.minimum(opens, close_prices) - np.abs(np.random.normal(0, 0.3, n_points))
    volumes = np.random.randint(1000, 10000, n_points)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': close_prices,
        'Volume': volumes
    }, index=pd.DatetimeIndex(dates))
    
    return DataOHLC(df)

def demonstrate_gap_handling():
    """Demuestra las diferentes opciones de manejo de huecos"""
    
    print("🚀 DEMOSTRACIÓN DE MANEJO DE DISCONTINUIDADES TEMPORALES")
    print("=" * 70)
    
    # Crear datos con huecos
    print("📊 Creando datos de ejemplo con discontinuidades temporales...")
    data = create_sample_data_with_gaps()
    
    # Análisis automático de huecos
    print("\n🔍 ANÁLISIS AUTOMÁTICO DE DISCONTINUIDADES:")
    print_gap_analysis(data, threshold_multiplier=1.5)
    
    # Obtener análisis programático
    analysis = analyze_time_gaps(data, threshold_multiplier=1.5)
    
    print(f"\n📈 RESUMEN EJECUTIVO:")
    print(f"  • Total de registros: {len(data)}")
    print(f"  • Huecos detectados: {analysis['total_gaps']}")
    print(f"  • Cobertura de datos: {analysis['data_coverage_percentage']:.1f}%")
    
    # Crear stats simulados para el ejemplo
    stats = pd.Series({
        'Total Return [%]': 15.5,
        'Max Drawdown [%]': -8.2,
        'Sharpe Ratio': 1.8,
        'Win Rate [%]': 65.0
    })
    
    print(f"\n🎨 GENERANDO GRÁFICOS CON DIFERENTES CONFIGURACIONES...")
    
    # Ejemplo 1: Mostrar huecos tal como están
    print("  1️⃣  Configuración 'show' - Mostrar huecos tal como están")
    try:
        plot(stats, data, {}, 
             gap_handling="show",
             plot_equity=False, plot_return=False, plot_drawdown=False,
             plot_trades=False, plot_volume=False)
        print("     ✅ Gráfico generado exitosamente")
    except Exception as e:
        print(f"     ❌ Error: {e}")
    
    # Ejemplo 2: Marcar huecos con líneas divisorias
    print("  2️⃣  Configuración 'mark' - Marcar huecos con líneas divisorias")
    try:
        plot(stats, data, {}, 
             gap_handling="mark",
             show_gap_lines=True,
             gap_threshold_multiplier=1.5,
             plot_equity=False, plot_return=False, plot_drawdown=False,
             plot_trades=False, plot_volume=False)
        print("     ✅ Gráfico con marcadores generado exitosamente")
    except Exception as e:
        print(f"     ❌ Error: {e}")
    
    # Ejemplo 3: Ocultar huecos para visualización continua
    print("  3️⃣  Configuración 'hide' - Ocultar huecos (visualización continua)")
    try:
        plot(stats, data, {}, 
             gap_handling="hide",
             plot_equity=False, plot_return=False, plot_drawdown=False,
             plot_trades=False, plot_volume=False)
        print("     ✅ Gráfico continuo generado exitosamente")
    except Exception as e:
        print(f"     ❌ Error: {e}")
    
    print(f"\n✨ DEMOSTRACIÓN COMPLETADA")
    print("=" * 70)
    
    return data, analysis

if __name__ == "__main__":
    # Ejecutar demostración
    sample_data, gap_analysis = demonstrate_gap_handling()
    
    print(f"\n💡 CONSEJOS DE USO:")
    print("  • Usa analyze_time_gaps() antes de graficar para entender tus datos")
    print("  • Para datos con muchos huecos, 'hide' ofrece mejor rendimiento")
    print("  • Para análisis detallado, 'mark' muestra claramente las discontinuidades")
    print("  • Ajusta gap_threshold_multiplier según la sensibilidad deseada")