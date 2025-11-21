"""
Test simple para verificar que el manejo de discontinuidades temporales funciona
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_import():
    """Test básico de importación"""
    try:
        print("🔍 Probando importación de plotting...")
        from tradero.plotting import analyze_time_gaps, print_gap_analysis
        from tradero.models import DataOHLC
        print("✅ Importación exitosa!")
        return True
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_test_data():
    """Crea datos de prueba simples con algunos huecos"""
    # Crear datos de 5 minutos con algunos huecos
    dates = []
    start_date = datetime(2024, 1, 15, 9, 0)
    
    # Datos normales por 1 hora
    current_date = start_date
    for i in range(12):  # 12 períodos de 5 minutos = 1 hora
        dates.append(current_date)
        current_date += timedelta(minutes=5)
    
    # Hueco de 30 minutos
    current_date += timedelta(minutes=30)
    
    # Más datos normales
    for i in range(12):
        dates.append(current_date)
        current_date += timedelta(minutes=5)
    
    # Generar datos OHLC sintéticos
    n_points = len(dates)
    np.random.seed(42)
    
    base_price = 100
    close_prices = base_price + np.random.normal(0, 1, n_points).cumsum()
    opens = close_prices + np.random.normal(0, 0.5, n_points)
    highs = np.maximum(opens, close_prices) + np.abs(np.random.normal(0, 0.5, n_points))
    lows = np.minimum(opens, close_prices) - np.abs(np.random.normal(0, 0.5, n_points))
    volumes = np.random.randint(1000, 5000, n_points)
    
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': close_prices,
        'Volume': volumes
    }, index=pd.DatetimeIndex(dates))
    
    return df

def test_gap_analysis():
    """Test del análisis de huecos"""
    try:
        from tradero.plotting import analyze_time_gaps, print_gap_analysis
        from tradero.models import DataOHLC
        
        print("\n📊 Creando datos de prueba...")
        df = create_simple_test_data()
        data = DataOHLC(df)
        
        print(f"✅ Datos creados: {len(data)} registros")
        
        print("\n🔍 Analizando huecos...")
        analysis = analyze_time_gaps(data, threshold_multiplier=1.5)
        
        print(f"✅ Análisis completado:")
        print(f"  • Huecos detectados: {analysis['total_gaps']}")
        print(f"  • Cobertura de datos: {analysis['data_coverage_percentage']:.1f}%")
        
        if analysis['total_gaps'] > 0:
            print(f"  • Tiempo faltante: {analysis['total_missing_time_minutes']:.1f} minutos")
            print(f"  • Recomendaciones: {len(analysis['recommendations'])}")
        
        print("\n📋 Reporte detallado:")
        print_gap_analysis(data, threshold_multiplier=1.5)
        
        return True
        
    except Exception as e:
        print(f"❌ Error en análisis de huecos: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 INICIANDO TESTS DE MANEJO DE DISCONTINUIDADES TEMPORALES")
    print("=" * 70)
    
    # Test 1: Importación
    if not test_import():
        print("❌ Test de importación falló")
        exit(1)
    
    # Test 2: Análisis de huecos
    if not test_gap_analysis():
        print("❌ Test de análisis de huecos falló")
        exit(1)
    
    print("\n🎉 TODOS LOS TESTS PASARON EXITOSAMENTE!")
    print("=" * 70)
    print("\n💡 La funcionalidad de manejo de discontinuidades temporales está lista para usar.")
    print("\n📚 Ejemplos de uso:")
    print("  from tradero.plotting import plot, analyze_time_gaps")
    print("  analysis = analyze_time_gaps(data)")
    print("  plot(stats, data, {}, gap_handling='mark', show_gap_lines=True)")