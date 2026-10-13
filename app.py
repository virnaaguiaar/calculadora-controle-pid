"""
Calculadora de Controle: projeto de PD, PI, PID e a/(s+b) pelo lugar das raízes,
com resposta ao degrau e discretização (Tustin, Euler, ZOH).
Cada etapa mostra a fórmula usada e a substituição numérica.

Rodar localmente:  streamlit run app.py
"""
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from controle import (angulo_graus, discretizar, equacao_diferencas, ler_polinomio,
                      malha_fechada, metricas, poli_para_texto, polo_desejado, projetar,
                      resposta_degrau, tabela_angulos, zeta_de_mp, zeta_wn_minimo)

st.set_page_config(page_title="Calculadora de Controle", page_icon="📐", layout="wide")


# ----------------------------------------------------------------------
# utilidades de exibição
# ----------------------------------------------------------------------
def latex_poli(p, var='s'):
    txt = poli_para_texto(p, var)
    return re.sub(r'\^(\d+)', r'^{\1}', txt)


def latex_fracao(num, den, var='s'):
    return r'\dfrac{' + latex_poli(num, var) + '}{' + latex_poli(den, var) + '}'


def fmt_c(z, casas=4):
    z = complex(z)
    if abs(z.imag) < 1e-9:
        return f'{z.real:.{casas}f}'
    sinal = '+' if z.imag >= 0 else '−'
    return f'{z.real:.{casas}f} {sinal} j{abs(z.imag):.{casas}f}'


def lx_c(z, casas=4):
    """Número (complexo ou real) em LaTeX."""
    z = complex(z)
    if abs(z.imag) < 1e-9:
        return f'{z.real:.{casas}f}'
    sinal = '+' if z.imag >= 0 else '-'
    return f'{z.real:.{casas}f} {sinal} j{abs(z.imag):.{casas}f}'


def lx_menos(x, casas=4):
    """Escreve '- x' tratando o sinal: s_d - (-2) vira s_d + 2."""
    x = complex(x)
    if abs(x.imag) > 1e-9:
        return rf'- ({lx_c(x, casas)})'
    return f'- {x.real:.{casas}f}' if x.real >= 0 else f'+ {abs(x.real):.{casas}f}'


st.sidebar.title("📐 Calculadora de Controle")
MOSTRAR = st.sidebar.toggle("Mostrar fórmulas e contas passo a passo", value=True)


def formula(*linhas):
    """Mostra uma ou mais linhas de LaTeX, se o modo detalhado estiver ligado."""
    if MOSTRAR:
        for l in linhas:
            st.latex(l)


def nota(texto):
    if MOSTRAR:
        st.markdown(texto)


METODOS = {
    "Tustin (bilinear)": "tustin",
    "Euler de avanço (forward): s = (z−1)/T": "euler_avanco",
    "Euler de atraso (backward): s = (z−1)/(Tz)": "euler_atraso",
    "ZOH (segurador de ordem zero)": "zoh",
}

SUBSTITUICAO = {
    "tustin": (r"s = \frac{2}{T}\cdot\frac{z-1}{z+1}",
               "Tustin aproxima a integral pela regra do trapézio. Mapeia o semiplano "
               "esquerdo inteiro dentro do círculo unitário, então preserva a estabilidade."),
    "euler_avanco": (r"s = \frac{z-1}{T}",
                     "Euler de avanço aproxima a derivada por [x(k+1) − x(k)]/T. Mapeia cada polo "
                     "como z = 1 + sT, e por isso pode jogar polos estáveis para fora do círculo unitário."),
    "euler_atraso": (r"s = \frac{z-1}{T\,z}",
                     "Euler de atraso aproxima a derivada por [x(k) − x(k−1)]/T. Mapeia cada polo "
                     "como z = 1/(1 − sT), sempre dentro do círculo se o polo for estável."),
    "zoh": (r"F(z) = (1 - z^{-1})\,\mathcal{Z}\left\{\frac{F(s)}{s}\right\}",
            "O segurador de ordem zero mantém a entrada constante entre amostras. É a "
            "discretização exata da planta vista por um conversor D/A."),
}


def bloco_discretizacao(num, den, chave, rotulo='C', mostrar_eq=True):
    c1, c2 = st.columns(2)
    T = c1.number_input("Período de amostragem T (s)", min_value=1e-6, value=1.0,
                        step=0.1, format="%.4f", key=f"T_{chave}")
    metodo_nome = c2.selectbox("Método", list(METODOS), key=f"m_{chave}")
    metodo = METODOS[metodo_nome]

    subst, explicacao = SUBSTITUICAO[metodo]
    formula(subst)
    nota(explicacao)
    if metodo != 'zoh':
        N = max(num.order, den.order)
        nota(f"Substitui-se s em cada termo e multiplica-se numerador e denominador pela "
             f"maior potência necessária (aqui, grau N = {N}) para eliminar as frações. "
             f"Depois, divide-se tudo pelo coeficiente de maior grau do denominador.")
        if metodo == 'tustin':
            formula(rf"a_k s^k \;\rightarrow\; a_k\left(\frac{{2}}{{T}}\right)^k (z-1)^k (z+1)^{{N-k}}"
                    rf"\qquad \frac{{2}}{{T}} = {2 / T:.4f}")
        elif metodo == 'euler_avanco':
            formula(rf"a_k s^k \;\rightarrow\; a_k\left(\frac{{1}}{{T}}\right)^k (z-1)^k"
                    rf"\qquad \frac{{1}}{{T}} = {1 / T:.4f}")
        else:
            formula(rf"a_k s^k \;\rightarrow\; a_k\left(\frac{{1}}{{T}}\right)^k (z-1)^k z^{{N-k}}"
                    rf"\qquad \frac{{1}}{{T}} = {1 / T:.4f}")

    try:
        nz, dz = discretizar(num, den, T, metodo)
    except ValueError as e:
        st.error(str(e))
        return
    st.latex(rf'{rotulo}(z) = ' + latex_fracao(nz, dz, 'z'))

    polos_z = np.roots(dz.coeffs) if dz.order > 0 else []
    zeros_z = np.roots(nz.coeffs) if nz.order > 0 else []
    c1, c2 = st.columns(2)
    c1.markdown("**Polos em z:** " + (", ".join(fmt_c(p) for p in polos_z) or "—"))
    c2.markdown("**Zeros em z:** " + (", ".join(fmt_c(z) for z in zeros_z) or "—"))

    if MOSTRAR and len(polos_z):
        polos_s = np.roots(np.poly1d(den).coeffs)
        if metodo in ('tustin', 'euler_avanco', 'euler_atraso', 'zoh') and len(polos_s):
            mapa = {"tustin": (r"z = \frac{1 + sT/2}{1 - sT/2}", lambda p: (1 + p * T / 2) / (1 - p * T / 2)),
                    "euler_avanco": (r"z = 1 + sT", lambda p: 1 + p * T),
                    "euler_atraso": (r"z = \frac{1}{1 - sT}", lambda p: 1 / (1 - p * T)),
                    "zoh": (r"z = e^{sT}", lambda p: np.exp(p * T))}
            expr, f = mapa[metodo]
            nota("Conferência: cada polo contínuo vai para o plano z pela regra do método:")
            formula(expr)
            linhas = [{"Polo em s": fmt_c(p), "Polo em z": fmt_c(f(p)), "|z|": round(abs(f(p)), 4)}
                      for p in polos_s]
            st.dataframe(pd.DataFrame(linhas), hide_index=True)

    fora = [p for p in polos_z if abs(p) > 1 + 1e-9]
    if fora:
        st.warning("Há polos fora do círculo unitário (|z| > 1). "
                   "Com esse T e esse método, a versão discreta é instável.")
    if mostrar_eq:
        st.markdown("**Equação a diferenças:**")
        nota("Divide-se numerador e denominador pela maior potência de z, obtendo potências de "
             "z⁻¹. Como z⁻ᵏ significa atraso de k amostras, U(z)·(a₀ + a₁z⁻¹ + …) = "
             "E(z)·(b₀ + b₁z⁻¹ + …) vira a equação abaixo, que o microcontrolador executa a cada T.")
        st.code(equacao_diferencas(nz, dz), language=None)


# ----------------------------------------------------------------------
# barra lateral
# ----------------------------------------------------------------------
modo = st.sidebar.radio("O que você quer fazer?",
                        ["Projetar controlador", "Só discretizar uma função"])

with st.sidebar.expander("Como digitar os polinômios"):
    st.markdown(
        "- Forma fatorada: `5(s+3)`, `s(s+4)`, `(s+1)^2`\n"
        "- Forma expandida: `s^3 + 4s^2 + 4s`\n"
        "- Coeficientes separados por espaço: `1 4 4 0`\n"
        "- Constante: `1` ou `0,4`\n"
        "- Vírgula decimal é aceita: `1,1772`"
    )

# ======================================================================
# MODO 2: só discretizar
# ======================================================================
if modo == "Só discretizar uma função":
    st.title("Discretização de F(s)")
    c1, c2 = st.columns(2)
    try:
        num = ler_polinomio(c1.text_input("Numerador de F(s)", "1"))
        den = ler_polinomio(c2.text_input("Denominador de F(s)", "s(s+1)"))
    except Exception as e:
        st.error(f"Erro ao ler o polinômio: {e}")
        st.stop()
    st.latex(r'F(s) = ' + latex_fracao(num, den))
    bloco_discretizacao(num, den, "solo", rotulo='F')
    st.stop()

# ======================================================================
# MODO 1: projeto
# ======================================================================
st.title("Projeto pelo lugar das raízes")

# ---------- 1. planta ----------
st.header("1. Planta e realimentação")
c1, c2, c3, c4 = st.columns(4)
try:
    N_G = ler_polinomio(c1.text_input("Numerador de G(s)", "4(s+4)"))
    D_G = ler_polinomio(c2.text_input("Denominador de G(s)", "s^3 + 4s^2 + 4s"))
    N_H = ler_polinomio(c3.text_input("Numerador de H(s)", "1"))
    D_H = ler_polinomio(c4.text_input("Denominador de H(s)", "1"))
except Exception as e:
    st.error(f"Erro ao ler o polinômio: {e}")
    st.stop()

c1, c2 = st.columns(2)
c1.latex(r'G(s) = ' + latex_fracao(N_G, D_G))
c2.latex(r'H(s) = ' + latex_fracao(N_H, D_H))
N_GH, D_GH = N_G * N_H, D_G * D_H
nota("Multiplicando G e H obtém-se a função de malha aberta, que é a usada nos critérios:")
formula(r'G(s)H(s) = ' + latex_fracao(N_GH, D_GH))

# ---------- 2. controlador ----------
st.header("2. Controlador")
tipos = {
    "PD  →  Kp + Kd·s": "PD",
    "PI  →  Kp + Ki/s": "PI",
    "PID com zeros reais e iguais  →  Kd(s − z)²/s": "PID",
    "Polo puro  →  a/(s + b)": "POLO",
}
tipo = tipos[st.radio("Tipo", list(tipos), horizontal=True)]
if tipo == 'PD':
    formula(r"C(s) = K_p + K_d\,s = K_d\,(s - z_c), \qquad z_c = -\frac{K_p}{K_d}")
    nota("O PD acrescenta **um zero** em z_c. Esse zero dá o ângulo que falta para s_d entrar no lugar das raízes.")
elif tipo == 'PI':
    formula(r"C(s) = K_p + \frac{K_i}{s} = \frac{K_p\,(s - z_c)}{s}, \qquad z_c = -\frac{K_i}{K_p}")
    nota("O PI acrescenta **um polo na origem** (o integrador, que zera o erro ao degrau) e **um zero** em z_c.")
elif tipo == 'PID':
    formula(r"C(s) = K_p + \frac{K_i}{s} + K_d\,s = \frac{K_d\,(s - z_c)^2}{s}",
            r"K_d(s - z_c)^2/s = K_d\,s \;-\; 2z_cK_d \;+\; \frac{z_c^2K_d}{s}"
            r"\;\;\Rightarrow\;\; K_p = -2z_cK_d,\quad K_i = z_c^2K_d")
    nota("O PID com zeros iguais acrescenta **um polo na origem** e **dois zeros no mesmo ponto** z_c.")
else:
    formula(r"G_c(s) = \frac{a}{s + b}, \qquad \text{polo em } p_c = -b")
    nota("Este controlador só acrescenta **um polo** em −b (nenhum zero) e um ganho a.")

# ---------- 3. especificação ----------
st.header("3. Especificação → polo desejado s_d")
espec = st.radio("Como a questão dá o polo desejado?",
                 ["Mp (%) e tempo de acomodação", "ξ e ωn", "Polo s_d direto"],
                 horizontal=True)

mp_alvo = ts_alvo = criterio = None
if espec == "Mp (%) e tempo de acomodação":
    c1, c2, c3 = st.columns(3)
    mp_alvo = c1.number_input("Mp máximo (%)", min_value=0.01, max_value=99.0, value=10.0, step=1.0)
    ts_alvo = c2.number_input("Tempo de acomodação máximo (s)", min_value=0.001, value=4.0, step=0.5)
    criterio = c3.selectbox("Critério do ts", [5, 2],
                            format_func=lambda x: f"{x}%  (ts = {3 if x == 5 else 4}/ξωn)")

    zeta_exato = zeta_de_mp(mp_alvo)
    arred = st.checkbox("Arredondar ξ para cima com 1 casa decimal (como no caderno)", value=True)
    zeta = math.ceil(zeta_exato * 10 - 1e-9) / 10 if arred else zeta_exato
    zwn = zeta_wn_minimo(ts_alvo, criterio)
    wn_min = zwn / zeta
    ln_mp = math.log(mp_alvo / 100)

    st.subheader("Amortecimento a partir do Mp")
    formula(r"M_p = e^{-\xi\pi/\sqrt{1-\xi^2}} \;\;\Rightarrow\;\; "
            r"\xi = \sqrt{\frac{(\ln M_p)^2}{\pi^2 + (\ln M_p)^2}}",
            rf"\ln({mp_alvo / 100:g}) = {ln_mp:.4f}"
            rf"\qquad \xi = \sqrt{{\frac{{({ln_mp:.4f})^2}}{{\pi^2 + ({ln_mp:.4f})^2}}}}"
            rf" = \sqrt{{\frac{{{ln_mp ** 2:.4f}}}{{{math.pi ** 2 + ln_mp ** 2:.4f}}}}} = {zeta_exato:.4f}")
    if arred:
        nota(f"Arredondando para cima: **ξ = {zeta:.1f}**. Para cima porque ξ maior dá Mp menor, "
             f"então a especificação continua atendida.")

    st.subheader("Frequência natural a partir do ts")
    k_ts = 3 if criterio == 5 else 4
    formula(rf"t_s({criterio}\%) \approx \frac{{{k_ts}}}{{\xi\,\omega_n}} \;\;\Rightarrow\;\; "
            rf"\xi\,\omega_n \geq \frac{{{k_ts}}}{{t_s}} = \frac{{{k_ts}}}{{{ts_alvo:g}}} = {zwn:.4f}",
            rf"\omega_{{n,\min}} = \frac{{{zwn:.4f}}}{{\xi}} = \frac{{{zwn:.4f}}}{{{zeta:.4f}}} = {wn_min:.4f}\ \text{{rad/s}}")

    wn = st.number_input("ωn de projeto (rad/s). Dica: use um valor acima do mínimo para ter folga.",
                         min_value=1e-6, value=float(np.round(wn_min, 4)), step=0.1, format="%.4f")
    if wn < wn_min - 1e-9:
        st.warning("ωn abaixo do mínimo: o tempo de acomodação estimado não atende.")

elif espec == "ξ e ωn":
    c1, c2 = st.columns(2)
    zeta = c1.number_input("ξ", min_value=0.001, max_value=0.999, value=0.7, step=0.05, format="%.4f")
    wn = c2.number_input("ωn (rad/s)", min_value=1e-6, value=0.5, step=0.1, format="%.4f")

else:
    c1, c2 = st.columns(2)
    re_sd = c1.number_input("Parte real de s_d", value=-4.0, step=0.5, format="%.4f")
    im_sd = c2.number_input("Parte imaginária de s_d (positiva)", min_value=1e-6, value=4.0,
                            step=0.5, format="%.4f")

if espec == "Polo s_d direto":
    s_d = complex(re_sd, im_sd)
    wn_eq = abs(s_d)
    zeta_eq = -s_d.real / wn_eq
    nota("O polo foi dado diretamente. Dele se tiram ξ e ωn equivalentes:")
    formula(rf"\omega_n = |s_d| = \sqrt{{({s_d.real:.4f})^2 + ({s_d.imag:.4f})^2}} = {wn_eq:.4f}"
            rf"\qquad \xi = \frac{{-\mathrm{{Re}}(s_d)}}{{\omega_n}} = {zeta_eq:.4f}")
else:
    s_d = polo_desejado(zeta, wn)
    st.subheader("Polo desejado")
    formula(r"s_d = -\xi\omega_n \pm j\,\omega_n\sqrt{1-\xi^2}",
            rf"s_d = -({zeta:.4f})({wn:.4f}) \pm j\,({wn:.4f})\sqrt{{1-{zeta:.4f}^2}}"
            rf" = {s_d.real:.4f} \pm j{s_d.imag:.4f}")

st.info(f"Polo desejado:  s_d = {fmt_c(s_d)}   (e o conjugado)")

# ---------- cálculo ----------
try:
    r = projetar(tipo, N_GH, D_GH, s_d)
except Exception as e:
    st.error(f"Não foi possível projetar: {e}")
    st.stop()

# ---------- 4. ângulo ----------
st.header("4. Critério de ângulo")
nota("Para s_d pertencer ao lugar das raízes, o ângulo de C(s_d)G(s_d)H(s_d) deve ser ±180°:")
formula(r"\sum \angle(s_d - z_i) \;-\; \sum \angle(s_d - p_i) = \pm 180^\circ")
nota("Cada ângulo é medido do polo/zero até s_d, a partir do eixo real positivo:")
formula(r"\theta_i = \operatorname{arctg}\!\left(\frac{\mathrm{Im}(s_d) - \mathrm{Im}(p_i)}"
        r"{\mathrm{Re}(s_d) - \mathrm{Re}(p_i)}\right)"
        r"\quad(\text{somar }180^\circ\text{ se o denominador for negativo})")

st.dataframe(pd.DataFrame(tabela_angulos(s_d, r["polos"], r["zeros"])), hide_index=True)
if tipo in ('PI', 'PID'):
    st.caption("O polo em 0 é o integrador do controlador e já está incluído.")

if MOSTRAR:
    with st.expander("Ver a conta de cada ângulo"):
        for rotulo, lista in (("p", r["polos"]), ("z", r["zeros"])):
            for i, pt in enumerate(lista, 1):
                pt = complex(pt)
                dre, dim = s_d.real - pt.real, s_d.imag - pt.imag
                ang = angulo_graus(s_d, pt)
                base = math.degrees(math.atan(dim / dre)) if abs(dre) > 1e-12 else 90.0
                corr = r" + 180^\circ" if dre < -1e-12 else ""
                st.latex(rf"\theta_{{{rotulo}{i}}}\ ({lx_c(pt, 3)}):\quad "
                         rf"\operatorname{{arctg}}\frac{{{dim:.4f}}}{{{dre:.4f}}} = {base:.3f}^\circ{corr}"
                         rf" \;\Rightarrow\; {ang:.3f}^\circ")

c1, c2, c3 = st.columns(3)
c1.metric("Σ ângulos dos polos", f'{r["soma_polos"]:.3f}°')
c2.metric("Σ ângulos dos zeros", f'{r["soma_zeros"]:.3f}°')
c3.metric("φ = Σpolos − Σzeros", f'{r["phi"]:.3f}°')
formula(rf"\varphi = \sum\theta_p - \sum\theta_z = {r['soma_polos']:.3f}^\circ - {r['soma_zeros']:.3f}^\circ"
        rf" = {r['phi']:.3f}^\circ")

if tipo in ('PD', 'PI'):
    nota("O zero do controlador entra com sinal positivo na soma, então:")
    formula(r"\theta_{z_c} - \varphi = -180^\circ \;\;\Rightarrow\;\; \theta_{z_c} = \varphi - 180^\circ",
            rf"\theta_{{z_c}} = {r['phi']:.3f}^\circ - 180^\circ = {r['theta']:.3f}^\circ")
    alvo_ang, ponto, nome_pt = r['theta'], r['z_c'], 'z_c'
elif tipo == 'PID':
    nota("Os dois zeros iguais contribuem com o mesmo ângulo, então cada um fornece metade:")
    formula(r"2\,\theta_{z_c} = \varphi - 180^\circ \;\;\Rightarrow\;\; \theta_{z_c} = \frac{\varphi - 180^\circ}{2}",
            rf"\theta_{{z_c}} = \frac{{{r['theta_total']:.3f}^\circ}}{{2}} = {r['theta']:.3f}^\circ")
    alvo_ang, ponto, nome_pt = r['theta'], r['z_c'], 'z_c'
else:
    nota("O polo do controlador entra com sinal negativo na soma, então:")
    formula(r"-\theta_{p_c} - \varphi = -180^\circ \;\;\Rightarrow\;\; \theta_{p_c} = 180^\circ - \varphi",
            rf"\theta_{{p_c}} = 180^\circ - {r['phi']:.3f}^\circ = {r['theta']:.3f}^\circ")
    alvo_ang, ponto, nome_pt = r['theta'], r['p_c'], 'p_c'

nota(f"Com o ângulo, localiza-se o ponto no eixo real pela tangente "
     f"(cateto oposto = Im(s_d), cateto adjacente = Re(s_d) − {nome_pt}):")
formula(rf"\operatorname{{tg}}\theta = \frac{{\mathrm{{Im}}(s_d)}}{{\mathrm{{Re}}(s_d) - {nome_pt}}}"
        rf"\;\;\Rightarrow\;\; {nome_pt} = \mathrm{{Re}}(s_d) - \frac{{\mathrm{{Im}}(s_d)}}{{\operatorname{{tg}}\theta}}",
        rf"{nome_pt} = {s_d.real:.4f} - \frac{{{s_d.imag:.4f}}}{{\operatorname{{tg}}({alvo_ang:.3f}^\circ)}}"
        rf" = {s_d.real:.4f} - \frac{{{s_d.imag:.4f}}}{{{math.tan(math.radians(alvo_ang)):.4f}}} = {ponto:.4f}")

# ---------- 5. módulo ----------
st.header("5. Critério de módulo e ganhos")
nota("Com o ângulo garantido, o ganho é escolhido para que o módulo seja 1:")
formula(r"|C(s_d)\,G(s_d)\,H(s_d)| = 1")

K_lider = N_GH.coeffs[0] / D_GH.coeffs[0]
prod_z = np.prod([abs(s_d - z) for z in r["zeros"]]) if r["zeros"] else 1.0
prod_p = np.prod([abs(s_d - p) for p in r["polos"]])
txt_int = r"\ (\text{incluindo o polo } 1/s \text{ do integrador})" if tipo in ('PI', 'PID') else ""
lhs = r"\left|\frac{GH(s_d)}{s_d}\right|" if tipo in ('PI', 'PID') else r"|GH(s_d)|"
formula(rf"{lhs} = K\,\frac{{\prod |s_d - z_i|}}{{\prod |s_d - p_i|}}{txt_int}",
        rf"{lhs} = {K_lider:g}\cdot\frac{{{prod_z:.4f}}}{{{prod_p:.4f}}} = {r['modulo_GH']:.6f}")
nota(f"K = {K_lider:g} é a razão entre os coeficientes de maior grau do numerador e do denominador de G·H. "
     f"Os módulos |s_d − p| e |s_d − z| estão na tabela do item 4.")

d = abs(s_d - ponto)
formula(rf"|s_d - {nome_pt}| = \sqrt{{({s_d.real:.4f} {lx_menos(ponto)})^2 + {s_d.imag:.4f}^2}} = {d:.4f}")

if tipo == 'PD':
    formula(r"K_d\,|s_d - z_c|\,|GH(s_d)| = 1 \;\;\Rightarrow\;\; K_d = \frac{1}{|s_d - z_c|\,|GH(s_d)|}",
            rf"K_d = \frac{{1}}{{{d:.4f}\cdot{r['modulo_GH']:.6f}}} = {r['Kd']:.4f}",
            rf"K_p = -z_c\,K_d = -({r['z_c']:.4f})({r['Kd']:.4f}) = {r['Kp']:.4f}")
    st.latex(rf"\boxed{{C(s) = {r['Kp']:.4f} + {r['Kd']:.4f}\,s}}")
elif tipo == 'PI':
    formula(r"K_p\,\frac{|s_d - z_c|}{|s_d|}\,|GH(s_d)| = 1 \;\;\Rightarrow\;\; "
            r"K_p = \frac{1}{|s_d - z_c|\cdot|GH(s_d)/s_d|}",
            rf"K_p = \frac{{1}}{{{d:.4f}\cdot{r['modulo_GH']:.6f}}} = {r['Kp']:.4f}",
            rf"K_i = -z_c\,K_p = -({r['z_c']:.4f})({r['Kp']:.4f}) = {r['Ki']:.4f}")
    st.latex(rf"\boxed{{C(s) = {r['Kp']:.4f} + \frac{{{r['Ki']:.4f}}}{{s}}}}")
elif tipo == 'PID':
    formula(r"K_d\,\frac{|s_d - z_c|^2}{|s_d|}\,|GH(s_d)| = 1 \;\;\Rightarrow\;\; "
            r"K_d = \frac{1}{|s_d - z_c|^2\cdot|GH(s_d)/s_d|}",
            rf"K_d = \frac{{1}}{{{d:.4f}^2\cdot{r['modulo_GH']:.6f}}} = {r['Kd']:.4f}",
            rf"K_p = -2z_cK_d = -2({r['z_c']:.4f})({r['Kd']:.4f}) = {r['Kp']:.4f}",
            rf"K_i = z_c^2K_d = ({r['z_c']:.4f})^2({r['Kd']:.4f}) = {r['Ki']:.4f}")
    st.latex(rf"\boxed{{C(s) = {r['Kp']:.4f} + \frac{{{r['Ki']:.4f}}}{{s}} + {r['Kd']:.4f}\,s}}")
else:
    formula(r"\frac{a}{|s_d - p_c|}\,|GH(s_d)| = 1 \;\;\Rightarrow\;\; a = \frac{|s_d - p_c|}{|GH(s_d)|}",
            rf"a = \frac{{{d:.4f}}}{{{r['modulo_GH']:.6f}}} = {r['a']:.4f}"
            rf"\qquad b = -p_c = {r['b']:.4f}")
    st.latex(rf"\boxed{{G_c(s) = \frac{{{r['a']:.4f}}}{{s + {r['b']:.4f}}}}}")

# ---------- 6. malha fechada ----------
st.header("6. Verificação em malha fechada")
num_mf, den_mf = malha_fechada(r["C_num"], r["C_den"], N_G, D_G, N_H, D_H)
nota("Com C = N_C/D_C, G = N_G/D_G e H = N_H/D_H:")
formula(r"T(s) = \frac{C\,G}{1 + C\,G\,H} = \frac{N_C\,N_G\,D_H}{D_C\,D_G\,D_H + N_C\,N_G\,N_H}",
        r"T(s) = " + latex_fracao(num_mf, den_mf))
nota("As raízes do denominador (equação característica) são os polos de malha fechada. "
     "O par s_d deve aparecer entre elas:")
formula(latex_poli(den_mf) + " = 0")
polos_mf = np.roots(den_mf.coeffs)
st.markdown("**Polos de malha fechada:** " + ", ".join(fmt_c(p) for p in polos_mf))

resp = resposta_degrau(num_mf, den_mf)
if resp is None:
    st.error("Malha fechada instável: não há resposta ao degrau para mostrar.")
else:
    t, y = resp
    m = metricas(t, y)
    nota("Simulando a resposta ao degrau unitário, as métricas reais são medidas na curva:")
    formula(r"y(\infty) = T(0)\qquad M_p = \frac{y_{\max} - y(\infty)}{y(\infty)}\cdot 100\%"
            r"\qquad t_s = \text{último instante fora da faixa } \pm 2\% \text{ ou } \pm 5\%")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valor final", f"{m['yf']:.4f}")
    c2.metric("Mp simulado", f"{m['Mp']:.2f}%",
              delta=None if mp_alvo is None else ("atende" if m['Mp'] <= mp_alvo else "NÃO atende"),
              delta_color="normal" if mp_alvo is None or m['Mp'] <= mp_alvo else "inverse")
    c3.metric("ts (5%)", f"{m['ts5']:.3f} s",
              delta=None if criterio != 5 else ("atende" if m['ts5'] < ts_alvo else "NÃO atende"),
              delta_color="normal" if criterio != 5 or m['ts5'] < ts_alvo else "inverse")
    c4.metric("ts (2%)", f"{m['ts2']:.3f} s",
              delta=None if criterio != 2 else ("atende" if m['ts2'] < ts_alvo else "NÃO atende"),
              delta_color="normal" if criterio != 2 or m['ts2'] < ts_alvo else "inverse")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, y, lw=2, label="y(t)")
    ax.axhline(m['yf'], color='gray', ls='--', lw=1, label="valor final")
    pct = criterio or 5
    ax.axhspan(m['yf'] * (1 - pct / 100), m['yf'] * (1 + pct / 100), color='green', alpha=0.08,
               label=f"faixa de ±{pct}%")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("saída")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    st.pyplot(fig)
    if num_mf.order >= den_mf.order:
        st.warning("T(s) tem numerador e denominador de mesmo grau: a saída salta em t = 0 "
                   "e o Mp simulado inclui esse salto.")
    st.caption("A simulação considera todos os polos e zeros de malha fechada. Se ela não atender "
               "à especificação, aumente o ωn de projeto: o método supõe que o par s_d domina.")

# ---------- 7. discretização ----------
st.header("7. Discretização (opcional)")
alvo = st.radio("O que discretizar?",
                ["Nada", "Só o controlador C(s)", "G(s)·H(s)·C(s) (malha aberta completa)"],
                horizontal=True)
if alvo == "Só o controlador C(s)":
    st.latex(r'C(s) = ' + latex_fracao(r["C_num"], r["C_den"]))
    bloco_discretizacao(r["C_num"], r["C_den"], "ctrl", rotulo='C')
elif alvo.startswith("G(s)"):
    L_num, L_den = N_GH * r["C_num"], D_GH * r["C_den"]
    st.latex(r'G(s)H(s)C(s) = ' + latex_fracao(L_num, L_den))
    bloco_discretizacao(L_num, L_den, "malha", rotulo='GHC', mostrar_eq=False)