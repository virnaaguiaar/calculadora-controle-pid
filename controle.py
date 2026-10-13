"""
Núcleo de cálculo: projeto de controladores pelo lugar das raízes
(critérios de ângulo e de módulo) e discretização.

Convenção: polinômios são np.poly1d; o controlador é C(s) = C_num / C_den.
"""
import math
import re

import numpy as np
from scipy import signal


# ======================================================================
# LEITURA DE POLINÔMIOS
# ======================================================================
def ler_polinomio(texto):
    """
    Converte texto em polinômio de s.
    Aceita: '5*(s+3)', '5(s+3)', 's(s+4)', 's^3 + 4s^2 + 4s',
            '10000*(s^2 - 1.1772)', '0.2', ou coeficientes '1 4 4 0'.
    """
    t = texto.strip().replace(',', '.')   # vírgula decimal (1,1772) vira ponto
    if not t:
        raise ValueError("Campo vazio.")

    # Lista de coeficientes: "1 4 4 0" ou "1; 4; 4; 0"
    if re.fullmatch(r'[-+\d.\s;eE]+', t) and len(re.split(r'[\s;]+', t)) > 1:
        coefs = [float(c) for c in re.split(r'[\s;]+', t) if c]
        return np.poly1d(coefs)

    expr = t.replace('^', '**')
    # multiplicação implícita: 5(s+3), 5s, s(s+4), )(, )s
    expr = re.sub(r'(?<=[\d.)s])\s*\(', '*(', expr)
    expr = re.sub(r'(?<=[\d.)])\s*s', '*s', expr)
    if re.search(r'[^0-9s+\-*/().\s eE]', expr):
        raise ValueError(f"Caractere inválido em '{texto}'. Use apenas números, s, + - * ^ ( ).")

    s = np.poly1d([1.0, 0.0])
    resultado = eval(expr, {"__builtins__": {}}, {"s": s})
    if not isinstance(resultado, np.poly1d):
        resultado = np.poly1d([float(resultado)])
    return np.poly1d(resultado.coeffs.astype(float))


def poli_para_texto(p, var='s', casas=4):
    """Escreve um polinômio de forma legível: 2.5s^2 + 3s - 1."""
    coefs = np.atleast_1d(p.coeffs)
    n = len(coefs) - 1
    termos = []
    for i, c in enumerate(coefs):
        c = float(np.real(c))
        if abs(c) < 1e-12:
            continue
        k = n - i
        c_txt = f'{abs(round(c, casas)):g}'
        if k == 0:
            corpo = c_txt
        else:
            corpo = ('' if c_txt == '1' else c_txt) + (var if k == 1 else f'{var}^{k}')
        sinal = '-' if c < 0 else '+'
        termos.append((sinal, corpo))
    if not termos:
        return '0'
    texto = ('-' if termos[0][0] == '-' else '') + termos[0][1]
    for sinal, corpo in termos[1:]:
        texto += f' {sinal} {corpo}'
    return texto


# ======================================================================
# POLO DESEJADO
# ======================================================================
def zeta_de_mp(mp_porcento):
    """ξ a partir do sobressinal máximo Mp (%)."""
    ln_mp = math.log(mp_porcento / 100)
    return math.sqrt(ln_mp ** 2 / (math.pi ** 2 + ln_mp ** 2))


def zeta_wn_minimo(ts, criterio):
    """ξ·ωn mínimo para o tempo de acomodação: 5% -> 3/ts ; 2% -> 4/ts."""
    if criterio == 5:
        return 3 / ts
    elif criterio == 2:
        return 4 / ts
    raise ValueError("Critério deve ser 2 ou 5.")


def polo_desejado(zeta, wn):
    return complex(-zeta * wn, wn * math.sqrt(1 - zeta ** 2))


# ======================================================================
# CRITÉRIOS DE ÂNGULO E MÓDULO
# ======================================================================
def angulo_graus(s_d, ponto):
    """Ângulo (0° a 360°) do vetor que vai de 'ponto' até s_d."""
    ang = math.degrees(np.angle(s_d - ponto))
    return ang + 360 if ang < 0 else ang


def ponto_no_eixo_real(s_d, theta_graus):
    """Ponto do eixo real que 'enxerga' s_d com o ângulo theta."""
    tg = math.tan(math.radians(theta_graus))
    if abs(tg) < 1e-12:
        raise ValueError("Ângulo de 0°/180°: não há ponto real que forneça esse ângulo.")
    return s_d.real - s_d.imag / tg


def tabela_angulos(s_d, polos, zeros):
    linhas = []
    for p in polos:
        linhas.append({"Tipo": "Polo", "Local": np.round(p, 4),
                       "Ângulo (°)": round(angulo_graus(s_d, p), 3),
                       "Módulo": round(abs(s_d - p), 4)})
    for z in zeros:
        linhas.append({"Tipo": "Zero", "Local": np.round(z, 4),
                       "Ângulo (°)": round(angulo_graus(s_d, z), 3),
                       "Módulo": round(abs(s_d - z), 4)})
    for l in linhas:
        l["Local"] = str(l["Local"]).replace('j', 'i')
    return linhas


def projetar(tipo, N_GH, D_GH, s_d):
    """
    tipo: 'PD', 'PI', 'PID' (zeros reais e iguais) ou 'POLO' (a/(s+b)).
    Retorna dict com controlador, ganhos e passos intermediários.
    """
    s = np.poly1d([1.0, 0.0])
    integrador = tipo in ('PI', 'PID')

    polos = list(np.roots(D_GH.coeffs))
    zeros = list(np.roots(N_GH.coeffs)) if N_GH.order > 0 else []
    if integrador:
        polos.append(0.0)

    soma_p = sum(angulo_graus(s_d, p) for p in polos)
    soma_z = sum(angulo_graus(s_d, z) for z in zeros)
    phi = soma_p - soma_z

    # |GH(s_d)| (incluindo o 1/s do integrador quando houver)
    L0 = N_GH(s_d) / D_GH(s_d)
    if integrador:
        L0 = L0 / s_d
    modL0 = abs(L0)

    r = {"tipo": tipo, "s_d": s_d, "polos": polos, "zeros": zeros,
         "soma_polos": soma_p, "soma_zeros": soma_z, "phi": phi, "modulo_GH": modL0}

    if tipo in ('PD', 'PI'):
        theta = (phi - 180) % 360
        r["theta"] = theta
        if theta >= 180:
            raise ValueError(f"O zero precisaria fornecer {theta:.2f}°: impossível com um único zero real.")
        z_c = ponto_no_eixo_real(s_d, theta)
        K = 1 / (abs(s_d - z_c) * modL0)
        r["z_c"] = z_c
        if tipo == 'PD':
            Kd, Kp = K, -z_c * K
            r.update(Kd=Kd, Kp=Kp, C_num=np.poly1d([Kd, Kp]), C_den=np.poly1d([1.0]))
        else:
            Kp, Ki = K, -z_c * K
            r.update(Kp=Kp, Ki=Ki, C_num=np.poly1d([Kp, Ki]), C_den=s)

    elif tipo == 'PID':
        theta_total = (phi - 180) % 360
        theta = theta_total / 2
        r["theta_total"] = theta_total
        r["theta"] = theta
        z_c = ponto_no_eixo_real(s_d, theta)
        Kd = 1 / (abs(s_d - z_c) ** 2 * modL0)
        Kp = -2 * z_c * Kd
        Ki = z_c ** 2 * Kd
        r.update(z_c=z_c, Kd=Kd, Kp=Kp, Ki=Ki,
                 C_num=np.poly1d([Kd, Kp, Ki]), C_den=s)

    elif tipo == 'POLO':
        theta = (180 - phi) % 360
        r["theta"] = theta
        if theta >= 180:
            raise ValueError(f"O polo precisaria fornecer {theta:.2f}°: impossível com um único polo real.")
        p_c = ponto_no_eixo_real(s_d, theta)
        a = abs(s_d - p_c) / modL0
        b = -p_c
        r.update(p_c=p_c, a=a, b=b, C_num=np.poly1d([a]), C_den=np.poly1d([1.0, b]))
    else:
        raise ValueError("Tipo de controlador desconhecido.")

    return r


# ======================================================================
# MALHA FECHADA E RESPOSTA AO DEGRAU
# ======================================================================
def malha_fechada(C_num, C_den, N_G, D_G, N_H, D_H):
    """T(s) = C·G / (1 + C·G·H)."""
    num = C_num * N_G * D_H
    den = C_den * D_G * D_H + C_num * N_G * N_H
    return num, den


def resposta_degrau(num, den, t_final=None):
    polos = np.roots(den.coeffs)
    if np.any(np.real(polos) > 1e-9):
        return None
    if t_final is None:
        sig = [abs(np.real(p)) for p in polos if abs(np.real(p)) > 1e-6]
        t_final = min(max(8 / min(sig), 5), 500) if sig else 20
    t = np.linspace(0, t_final, 5000)
    _, y = signal.step((num.coeffs, den.coeffs), T=t)
    return t, y


def metricas(t, y):
    yf = y[-1]
    if abs(yf) < 1e-12:
        return {"yf": yf, "Mp": float('nan'), "ts2": float('nan'), "ts5": float('nan')}
    mp = max(0.0, (np.max(y) - yf) / abs(yf) * 100)
    res = {"yf": yf, "Mp": mp}
    for pct in (2, 5):
        fora = np.where(np.abs(y - yf) > pct / 100 * abs(yf))[0]
        res[f"ts{pct}"] = 0.0 if len(fora) == 0 else t[min(fora[-1] + 1, len(t) - 1)]
    return res


# ======================================================================
# DISCRETIZAÇÃO
# ======================================================================
def discretizar(num, den, T, metodo):
    """
    Substitui s por c·(z-1)/w(z) e multiplica tudo por w(z)^N.
      'tustin'        s = (2/T)(z-1)/(z+1)
      'euler_avanco'  s = (z-1)/T
      'euler_atraso'  s = (z-1)/(T z)
      'zoh'           segurador de ordem zero (só função própria)
    """
    num, den = np.poly1d(num), np.poly1d(den)

    if metodo == 'zoh':
        if num.order > den.order:
            raise ValueError("ZOH exige função própria (grau do numerador ≤ grau do denominador).")
        nz, dz, _ = signal.cont2discrete((num.coeffs, den.coeffs), T, method='zoh')
        nz = np.atleast_1d(np.squeeze(nz))
        num_z, den_z = np.poly1d(nz), np.poly1d(dz)
    else:
        if metodo == 'tustin':
            c, w = 2 / T, np.poly1d([1.0, 1.0])
        elif metodo == 'euler_avanco':
            c, w = 1 / T, np.poly1d([1.0])
        elif metodo == 'euler_atraso':
            c, w = 1 / T, np.poly1d([1.0, 0.0])
        else:
            raise ValueError("Método inválido.")
        N = max(num.order, den.order)
        z_menos_1 = np.poly1d([1.0, -1.0])

        def substitui(p):
            resultado = np.poly1d([0.0])
            coefs = p.coeffs[::-1]          # coefs[k] multiplica s^k
            for k in range(len(coefs)):
                termo = np.poly1d([float(coefs[k]) * c ** k])
                resultado = resultado + termo * (z_menos_1 ** k) * (w ** (N - k))
            return resultado

        num_z, den_z = substitui(num), substitui(den)

    lider = den_z.coeffs[0]
    nc = num_z.coeffs / lider
    dc = den_z.coeffs / lider
    nc[np.abs(nc) < 1e-10] = 0
    dc[np.abs(dc) < 1e-10] = 0
    return np.poly1d(nc), np.poly1d(dc)


def equacao_diferencas(num_z, den_z, entrada='e', saida='u'):
    """u[k] = Σ b_i·e[k-i] - Σ a_i·u[k-i]  (a partir de C(z) em potências de z^-1)."""
    n = den_z.order
    b = np.concatenate([np.zeros(n + 1 - len(num_z.coeffs)), num_z.coeffs])
    a = den_z.coeffs
    partes = []
    for i, bi in enumerate(b):
        if abs(bi) > 1e-12:
            partes.append(f"{bi:+.4f}·{entrada}[k-{i}]" if i else f"{bi:+.4f}·{entrada}[k]")
    for i in range(1, len(a)):
        if abs(a[i]) > 1e-12:
            partes.append(f"{-a[i]:+.4f}·{saida}[k-{i}]")
<<<<<<< HEAD
    return f"{saida}[k] = " + ' '.join(partes)
=======
    return f"{saida}[k] = " + ' '.join(partes)
>>>>>>> a74fefccb7184df471da8dca2ae3e4eb6112ab52
