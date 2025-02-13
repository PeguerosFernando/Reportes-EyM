import numpy as np

def error_sum(values: np.array, errors: np.array):
    """
    Propagación de errores en la suma o resta de múltiples valores.

    Parámetros:
    - values: np.array con los valores de las magnitudes
    - errors: np.array con los errores absolutos de cada magnitud

    Retorna:
    - resultado (float): Suma de los valores
    - error_total (float): Error absoluto propagado
    """
    resultado = np.sum(values)
    error_total = np.sqrt(np.sum(errors**2))
    return resultado, error_total

def error_product(values: np.array, errors: np.array):
    """
    Propagación de errores en el producto de múltiples valores.

    Parámetros:
    - values: np.array con los valores de las magnitudes
    - errors: np.array con los errores absolutos de cada magnitud

    Retorna:
    - resultado (float): Producto de los valores
    - error_total (float): Error absoluto propagado
    """
    resultado = np.prod(values)
    relative_errors = (errors / values) ** 2
    error_total = abs(resultado) * np.sqrt(np.sum(relative_errors))
    return resultado, error_total

def error_power(value: float, error: float, exponent: float):
    """
    Propagación de errores en una potencia: f(x) = x^n.

    Parámetros:
    - value: Valor de la magnitud
    - error: Error absoluto de la magnitud
    - exponent: Exponente de la función

    Retorna:
    - resultado (float): Resultado de la potencia
    - error_total (float): Error absoluto propagado
    """
    resultado = value ** exponent
    error_total = abs(exponent) * (error / value) * resultado
    return resultado, error_total

def error_log(value: float, error: float):
    """
    Propagación de errores en el logaritmo natural: f(x) = ln(x).

    Parámetros:
    - value: Valor de la magnitud
    - error: Error absoluto de la magnitud

    Retorna:
    - resultado (float): ln(value)
    - error_total (float): Error absoluto propagado
    """
    resultado = np.log(value)
    error_total = abs(error / value)
    return resultado, error_total

def error_exponential(value: float, error: float):
    """
    Propagación de errores en una función exponencial: f(x) = e^x.

    Parámetros:
    - value: Valor de la magnitud
    - error: Error absoluto de la magnitud

    Retorna:
    - resultado (float): e^value
    - error_total (float): Error absoluto propagado
    """
    resultado = np.exp(value)
    error_total = resultado * error
    return resultado, error_total

def error_mean(values: np.array, errors: np.array):
    """
    Cálculo del promedio de valores con propagación de errores.

    Parámetros:
    - values: np.array con los valores de las magnitudes
    - errors: np.array con los errores absolutos de cada magnitud

    Retorna:
    - promedio (float): Media de los valores
    - error_total (float): Error absoluto propagado del promedio
    """
    promedio = np.mean(values)
    error_total = np.sqrt(np.sum(errors**2)) / len(values)
    return promedio, error_total
