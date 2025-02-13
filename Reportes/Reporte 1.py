import sys
import os

# Agregar la carpeta raíz al path de búsqueda de módulos
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# Importar todas las funciones de Modulo
from Modulo import error_sum, FitLineal

# Datos de prueba
import numpy as np
valores = np.array([5.0, 3.0, 2.5])
errores = np.array([0.2, 0.1, 0.05])

# Uso de una función de propagación de errores
suma, error_suma = error_sum(valores, errores)
print(f"Suma: {suma:.2f} ± {error_suma:.2f}")

# Uso de una función de ajuste de curvas
modelo = FitLineal(valores, errores)