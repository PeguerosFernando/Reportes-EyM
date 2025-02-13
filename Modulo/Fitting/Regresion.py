import numpy as np
import matplotlib.pyplot as plt

def FitLineal(x, y, s=None, graf=False):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    if graf:
        plt.scatter(x, y, color='red', label='Datos')
        plt.plot(x, m*x + c, 'b-', label='Ajuste lineal')
        if s is not None:
            plt.fill_between(x, (m*x + c) - s, (m*x + c) + s, color='blue', alpha=0.2, label='Error')
        plt.title('Ajuste Lineal de Mínimos Cuadrados')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return m, c

def fbase(n, k, x):
    if n > 2:
        return x**k
    elif n == 2:
        return np.ones_like(x) if k == 0 else np.sin(x)
    else:
        raise ValueError("n debe ser mayor o igual a 2")

def GFit(x, y, s, n, graf=False):
    N = x.size
    A = np.zeros((N, n))
    
    for k in range(n):
        A[:, k] = fbase(3, k, x) / s
    bs = y / s
    
    matI = A.T @ A
    InvmatI = np.linalg.inv(matI)
    matD = A.T @ bs
    cs = InvmatI @ matD
    
    sigS = np.sqrt(np.diagonal(InvmatI))
    chisq = np.sum((bs - A @ cs)**2)
    
    if graf:
        x_fit = np.linspace(np.min(x), np.max(x), 500)
        y_fit = sum(cs[k] * fbase(3, k, x_fit) for k in range(n))
        
        # Calcular el error de ajuste
        y_fit_err = np.sqrt(sum((fbase(3, k, x_fit) * sigS[k])**2 for k in range(n)))
        plt.fill_between(x_fit, y_fit - y_fit_err, y_fit + y_fit_err, color='blue', alpha=0.2, label='Error')
        
        plt.errorbar(x, y, yerr=s, fmt='o', label='Datos', color='red')
        plt.plot(x_fit, y_fit, 'b-', label='Ajuste')
        
        plt.title('Ajuste Polinomial General')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return cs, chisq, sigS, InvmatI

def fbayes(data, batches, primus, priS, graf=False):
    n = primus.size
    i = 0
    all_postmus = []

    for N in batches:
        A = np.zeros((N, n))
        for k in range(n):
            A[:, k] = fbase(n, k, data[0, i: i+N]) / data[2, i:i+N]
        bs = data[1, i:i+N] / data[2, i:i+N]
        
        prinSinv = np.linalg.inv(priS)
        postS = np.linalg.inv(A.T @ A + prinSinv)  # covarianza
        postmus = postS @ (A.T @ bs + prinSinv @ primus)
        
        primus, priS = postmus, postS  # valor posterior de la media y prior
        
        all_postmus.append((postmus, postS))
        i += N

    if graf:
        plot_results(data, all_postmus)

    return postmus, postS

def generatedata(N, a=0., b=9, sts=(2, 5, 0.5, 1)):
    sa, sb, sc, sd = sts
    np.random.seed(7921)
    data = np.zeros((3, N))
    data[0, :] = np.linspace(a, b, N)
    data[1, :] = sa + sb * np.sin(data[0, :])
    data[2, :] = sc + sd * np.random.random(N)
    data[1, :] += np.random.normal(0, data[2, :])
    return data

def PostDistrib(c, primus, priS):
    term = c - primus
    prinSinv = np.linalg.inv(priS)
    argum = -term.T @ prinSinv @ term / 2
    SigM = np.exp(argum)
    return SigM

def plot_results(data, all_postmus):
    plt.errorbar(data[0], data[1], yerr=data[2], fmt='o', label='Datos', color='red', alpha=0.5)
    
    for postmus, postS in all_postmus:
        x_fit = np.linspace(np.min(data[0]), np.max(data[0]), 500)
        y_fit = sum(postmus[k] * fbase(len(postmus), k, x_fit) for k in range(len(postmus)))
        y_fit_var = np.array([fbase(len(postmus), k, x_fit) for k in range(len(postmus))]).T @ postS @ np.array([fbase(len(postmus), k, x_fit) for k in range(len(postmus))])
        y_fit_std = np.sqrt(np.diagonal(y_fit_var))
        plt.plot(x_fit, y_fit, label='Ajuste Bayesiano')
        plt.fill_between(x_fit, y_fit - y_fit_std, y_fit + y_fit_std, color='blue', alpha=0.2, label='Error')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.title('Ajuste Bayesiano')
    plt.show()

def model(cs, xi):
    p = cs[0] + cs[1] * xi ** cs[2]
    return p

def relErrorTot(xolds, xnews):
    errs = np.abs((xnews - xolds) / xnews)
    return np.sum(errs)

def gdata():
    data = np.zeros((3, 13))
    data[0, :] = np.array([373.1, 492.5, 733, 755, 799, 820,
                           877, 1106, 1125, 1403, 1492, 1522, 1561])
    data[1, :] = np.array([156., 638, 3320, 3810, 4440, 5150,
                           6910, 16400, 17700, 44700, 57400, 60600, 67800])
    data[2, :] = np.ones(data.shape[1])*2  # dispersion 1 para todos

    return data

def getKrs(data, cs):
    """
    rho = p(xj)/sigma_j, b = yj/sigma_j
    """
    K = np.zeros((data.shape[1], cs.size))

    K[:, 0] = 1 / data[2, :]
    K[:, 1] = data[0, :] ** cs[2] / data[2, :]
    K[:, 2] = cs[1] * data[0, :] ** cs[2] * np.log(data[0, :]) / data[2, :]

    rs = (data[1, :] - model(cs, data[0, :])) / data[2, :]  # b-rho
    return K, rs

def GaussNewton(data, ckm1, kmax=50, tol=1.e-8):
    for _ in range(kmax):
        K, rs = getKrs(data, ckm1)
        matK = K.T @ K
        invmatK = np.linalg.inv(matK)
        ck = ckm1 + invmatK @ K.T @ rs
        err = relErrorTot(ckm1, ck)
        if err < tol:
            break
        ckm1 = np.copy(ck)

    return ckm1

def getKrs2(data, cs):
    """
    rho = p(xj)/sigma_j, b = yj/sigma_j
    """
    K = np.zeros((data.shape[1], cs.size))

    K[:, 0] = 1 / data[2, :]
    K[:, 1] = np.exp(-data[0, :] * cs[2]) / data[2, :]
    K[:, 2] = -cs[1] * data[0, :] * np.exp(-data[0, :] * cs[2]) / data[2, :]

    rs = (data[1, :] - model2(cs, data[0, :])) / data[2, :]  # b-rho

    return K, rs

def model2(cs, xi):
    p = cs[0] + cs[1] * np.exp(-cs[2] * xi)
    return p

def GaussNeewton2(data, ckm1, kmax=50, tol=1.e-8):
    for _ in range(kmax):
        K, rs = getKrs2(data, ckm1)
        matK = K.T @ K
        invmatK = np.linalg.inv(matK)
        ck = ckm1 + invmatK @ K.T @ rs
        err = relErrorTot(ckm1, ck)
        if err < tol:
            break
        ckm1 = np.copy(ck)

    return ckm1

def plot_data_and_fit(data, c):
    plt.figure(figsize=(8, 6))

    # Graficar los datos con barras de error en ambas direcciones
    plt.errorbar(data[0, :], data[1, :], xerr=data[2, :], fmt='o', markersize=3, label='Datos', capsize=5)

    # Calcular el ajuste de Gauss-Newton y graficarlo
    x_vals = np.linspace(min(data[0, :]), max(data[0, :]), 100)
    plt.plot(x_vals, model(c, x_vals), label='Ajuste de Gauss-Newton', color='red')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Ajuste de Gauss-Newton con barras de error')
    plt.grid(True)

    plt.tight_layout()
    plt.show()