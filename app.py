"""
Sitio Streamlit — dos pestañas:
  Historia: la narrativa (visualizaciones precalculadas)
  Herramienta: optimizador interactivo
Ejecutar: streamlit run app.py
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

from fetch_data import load_returns
from explorer_lib import (
    UNIVERSE_SOURCES, REGION_GROUP, ticker_region,
    BENCHMARK_TICKERS, get_benchmark, drop_benchmarks, perf_stats,
    optimize_portfolio, asset_metrics, equity_curve, backtest_strategy,
)

st.set_page_config(page_title="Estrategia de Clustering para Portafolios",
                   page_icon="📈", layout="wide")

# Estilo académico y pestañas fijas en el encabezado
st.markdown("""
<style>
.main .block-container { max-width: 1100px; padding-top: 1rem; }
h1, h2, h3 { color: #1a2940; font-family: Georgia, serif; }
blockquote { background: #f7f7f7; border-left: 3px solid #c0c0c0;
             padding: 10px 16px; color: #444; font-size: 0.95em; }

/* Pestañas fijas en el encabezado */
div[data-testid="stTabs"] > div[data-baseweb="tab-list"] {
    position: sticky;
    top: 0;
    background: white;
    z-index: 999;
    padding: 8px 0;
    border-bottom: 1px solid #e0e0e0;
    box-shadow: 0 2px 6px -3px rgba(0,0,0,.15);
}
div[data-testid="stTabs"] button[role="tab"] {
    font-size: 1.05rem;
    font-weight: 600;
}

.explainer-box {
    background: #eef4fb !important;
    color: #1a2940 !important;
    border-left: 4px solid #4a6fa5;
    padding: 14px 18px;
    border-radius: 6px;
    margin: 12px 0;
}
.explainer-box * { color: #1a2940 !important; }
.explainer-box strong { color: #0d1e3a !important; }

.workflow-step {
    background: #f8f8f8 !important;
    color: #1a2940 !important;
    border-left: 3px solid #2c4a7a;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 4px;
}
.workflow-step * { color: #1a2940 !important; }
.workflow-step strong { color: #0d1e3a !important; }
</style>
""", unsafe_allow_html=True)

STORY_DIR = Path("data/story")


@st.cache_data
def cached_returns():
    return load_returns()


# ─────────────────────────────────────────────────────────────────
# Pestañas
# ─────────────────────────────────────────────────────────────────
home_tab, tool_tab = st.tabs(["📖 La Historia", "🛠 La Herramienta"])


# =================================================================
# PESTAÑA HISTORIA
# =================================================================
with home_tab:
    st.title("¿Puede el Clustering Jerárquico Ganarle al Mercado?")
    st.markdown("*Un proyecto de investigación que extiende la teoría de Markowitz "
                "para escoger **qué activos comprar y en qué proporción**, "
                "desde un universo de miles de acciones globales.*")

    # Resultado principal
    try:
        headline = json.loads((STORY_DIR / "headline_metrics.json").read_text())
        s, b = headline["strategy"], headline["spy"]
        st.markdown("### Resultado principal")
        st.markdown(f"Durante **{headline['n_periods']} meses fuera de muestra** "
                    f"({headline['start']} → {headline['end']}), "
                    f"la estrategia obtuvo un **Sharpe realizado de {s['sharpe']:.2f}** "
                    f"frente al **{b['sharpe']:.2f} del SPY**, "
                    f"con un retorno total de **{s['total_return']:.0%}** "
                    f"contra **{b['total_return']:.0%}** del SPY.")
        st.image(str(STORY_DIR / "hero_equity.png"), use_container_width=True)
    except Exception as e:
        st.warning(f"No se encontraron los archivos precalculados. Ejecuta primero `python3 precompute.py`. ({e})")

    # ── 1. El problema ──
    st.markdown("## 1. El Problema")
    st.markdown("""
    La **optimización de Markowitz** clásica resuelve los *pesos* de un portafolio
    una vez que ya decidiste qué activos comprar. Responde a la pregunta:
    > *"Dados estos 5 ETFs, ¿qué combinación maximiza el Sharpe?"*

    Este proyecto invierte la pregunta. Desde un universo de **~2,000 activos globales**,
    queremos elegir **simultáneamente qué N activos incluir y cómo ponderarlos** — una optimización conjunta.

    ¿Por qué es difícil?

    - Elegir N activos de M es un problema **combinatorio** (NP-difícil): existen C(2000, 50) ≈ 10⁸⁵ portafolios posibles de 50 activos.
    - No se puede ejecutar Markowitz directamente sobre 2,000 activos — la matriz de covarianza no se puede estimar de forma confiable con tan pocos datos (lo explicamos en detalle en la sección 4).

    **Nuestro enfoque:** usar **clustering jerárquico** para reducir el universo a N representantes
    con comportamiento distinto entre sí, y luego correr Markowitz sobre esos N.
    El clustering convierte un problema combinatorio intratable en un pipeline limpio de dos etapas.
    """)

    # ── 2. El universo ──
    st.markdown("## 2. Diseñando el Universo")
    st.markdown("**¿Qué activos entran en el conjunto candidato?**")
    st.markdown("""
    Elegimos **activos individuales** y no ETFs (no queremos un *fondo de fondos*; queremos
    el comportamiento subyacente). Cubrimos mercados globales para aprovechar la diversificación
    entre economías no correlacionadas. Un portafolio solo de EE.UU. no puede escapar a un crash sistémico estadounidense.

    El universo se ensambla desde **17 fuentes**: índices bursátiles y las principales tenencias de ETFs regionales.
    """)
    rows_html = "".join(f"<tr><td><b>{n}</b></td><td>{d}</td></tr>" for n, d in UNIVERSE_SOURCES)
    st.markdown(f"<table>{rows_html}</table>", unsafe_allow_html=True)

    st.markdown("""
    **Filtros de calidad aplicados:**
    - Descartar tickers con menos del 80% de cobertura en los 5 años (maneja IPOs recientes y delistings)
    - Descartar tickers con cualquier retorno mensual > 50% en valor absoluto (atrapa errores de yfinance por stock splits mal ajustados)

    Después del filtrado quedan ~2,000 tickers utilizables.
    """)
    if (STORY_DIR / "universe.png").exists():
        st.image(str(STORY_DIR / "universe.png"), use_container_width=True)

    # ── 3. Frecuencia de datos ──
    st.markdown("## 3. Eligiendo la Frecuencia de los Datos")
    st.markdown("""
    **¿Con qué frecuencia muestreamos los retornos?**

    | Frecuencia | Observaciones (5 años) | Trade-off |
    |---|---|---|
    | Diaria   | ~1,250 | Más datos, pero cada punto trae mucho ruido intradiario |
    | Semanal  | 260    | Buen balance |
    | Mensual  | 60     | Señal más limpia pero la estimación se vuelve frágil |

    Contraintuitivamente, **la frecuencia mensual funcionó al menos tan bien como la semanal**
    en nuestros experimentos fuera de muestra — la señal más limpia compensa el menor tamaño
    de muestra, sobre todo tras aplicar la reducción de Ledoit-Wolf (siguiente sección).
    Usamos mensual en todo el proyecto.
    """)

    # ── 4. Por qué Markowitz se rompe ──
    st.markdown("## 4. Por Qué Markowitz Se Rompe con 2,000 Activos")

    st.markdown("""
    La matriz de covarianza es una **tabla gigante de relaciones**: para cada par de activos
    (Apple-Microsoft, Apple-Toyota, etc.) intenta capturar qué tan juntos se mueven.
    Con 2,000 activos hay alrededor de **2 millones de relaciones** distintas que estimar.
    Pero solo tenemos **60 meses de historia** para estimarlas. El problema no es que no se pueda
    calcular la matriz — la fórmula funciona y devuelve números. **El problema es que esos números
    están llenos de ruido.**

    ### ¿De dónde viene el ruido?

    La correlación "verdadera" entre dos activos (digamos AAPL y MSFT) es un número real que existe
    en el mundo pero que **nunca podemos observar directamente**. Para conocerlo perfectamente
    necesitaríamos infinitos datos.

    Lo único que podemos hacer es **estimarlo** con los datos disponibles. En nuestro caso, 60 meses.

    Imagina que tomaras dos ventanas de 60 meses distintas:

    - **Ventana A** (2018-2023) → calculas la correlación AAPL-MSFT y te da 0.72
    - **Ventana B** (2019-2024) → calculas la misma correlación y te da 0.81

    ¿Cuál es la "verdadera"? Ninguna. Ambas son **estimaciones ruidosas** del valor real (que tal vez
    sea 0.77, pero no lo sabemos). La diferencia entre ambas —esos 9 puntos porcentuales— es
    **ruido de muestreo**.

    ### Cuánto ruido

    Hay una regla aproximada: el ruido en una correlación estimada es de aproximadamente:

    > ruido ≈ 1 / √(número de observaciones)

    Con 60 meses:

    > ruido ≈ 1 / √60 ≈ ±0.13

    Esto significa que cada correlación que calculamos tiene un **error típico de ±0.13** alrededor
    del valor verdadero.

    ### El verdadero problema: dos activos sin relación

    Imagina dos activos cuya correlación verdadera es **0** (no tienen ninguna relación real).
    Con 60 meses, nuestro estimado va a oscilar entre **-0.13 y +0.13** solo por puro azar.

    Ahora, el optimizador de Markowitz mira una correlación de -0.13 y piensa:
    *"¡Estos dos activos van en direcciones opuestas! Es diversificación gratis, voy a cargar peso aquí."*

    Pero esa correlación de -0.13 **no significa nada** — la relación real es cero. Es ruido puro.
    El optimizador acaba de apostar el portafolio a una "oportunidad" que solo existe en los datos
    pasados, no en la realidad.

    ### Por qué empeora con 2,000 activos

    Con 2,000 activos tenemos **2 millones de pares**. Si cada uno tiene un ruido típico de ±0.13,
    muchos de esos 2 millones van a aterrizar por azar en valores que **parecen relaciones reales
    pero no lo son**. El optimizador, que busca correlaciones extremas para construir el portafolio,
    va a encontrar miles de "oportunidades" falsas. Por eso falla.

    ### Por qué con más datos sería menos malo

    Con 600 meses (50 años) de datos:

    > ruido ≈ 1 / √600 ≈ ±0.04

    El error se reduce, las correlaciones espurias casi desaparecen. Pero esos datos no existen
    para la mayoría de activos.
    """)

    st.markdown("""
    ### La solución tiene dos partes

    Para que Markowitz funcione necesitamos atacar el problema en **dos frentes complementarios**:

    1. **Reducir el universo a un número manejable de activos** mediante clustering (sección 5).
        Esto pasa de 2 millones de pares ruidosos a solo ~1,275 pares.
    2. **Limpiar lo que queda** con la reducción de Ledoit-Wolf antes de optimizar (sección 7).
        Esto suaviza los valores extremos que aún sobreviven.

    El clustering hace el trabajo pesado. Ledoit-Wolf es el pulido final.
    """)

    # ── 5. Clustering ──
    st.markdown("## 5. La Idea del Clustering")
    st.markdown("""
    **Reducir 2,000 activos a N grupos con comportamiento similar.**
    Los activos que se mueven juntos (alta correlación) son sustitutos entre sí —
    tener *cualquiera* de ellos te da prácticamente la misma exposición que tener cualquier otro del grupo.

    Aplicamos **clustering jerárquico (Ward)** sobre una matriz de distancias derivada de las correlaciones:

    > distancia = 1 − correlación

    Dos activos con correlación = 1 están a distancia 0 (mismo cluster).
    Activos independientes están a distancia 1.
    Activos con correlación negativa están a distancia 2 (clusters muy distintos).

    Los dendrogramas de abajo muestran cómo se forma el árbol de agrupamiento.
    La **línea de corte horizontal** determina N: corte alto → pocos clusters grandes;
    corte bajo → muchos clusters finos.
    """)

    st.markdown("""
    <div class='explainer-box'>
    <strong>Espera — ¿no acabamos de decir que la matriz de 2,000 activos está llena de ruido?
    ¿Por qué la usamos para clustering?</strong>
    <br><br>
    Porque <strong>el clustering tolera el ruido</strong>. Solo necesita ver la estructura gruesa:
    "estos activos tecnológicos se parecen entre sí, este grupo de utilities se parece entre sí,
    los dos grupos son distintos". Aunque cada correlación individual tenga error de ±0.13,
    la <em>estructura general</em> de qué se agrupa con qué emerge correctamente. Es como leer
    un mapa borroso: no distingues cada casa, pero sí ves dónde están los barrios.
    <br><br>
    En cambio, Markowitz <strong>no tolera el ruido</strong>: busca activamente los valores extremos
    (correlaciones cercanas a cero le parecen "diversificación gratis") y apuesta el portafolio
    a esas oportunidades que muchas veces son ficticias. Por eso podemos usar la matriz ruidosa
    para clustering pero <em>no</em> para optimización directa.
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(3)
    for col, n in zip(cols, (5, 20, 50)):
        if (STORY_DIR / f"dendrogram_n{n}.png").exists():
            col.image(str(STORY_DIR / f"dendrogram_n{n}.png"),
                      caption=f"Corte en N = {n}", use_container_width=True)

    st.markdown("### ¿Por qué correlación y no covarianza directa?")
    st.markdown("""
    Esta es una pregunta importante porque **Markowitz sí usa la matriz de covarianza** en su fórmula original.
    No la estamos descartando — la usamos más adelante, en el paso 7. Pero el **clustering del paso 5
    usa correlación**, no covarianza. Son dos pasos distintos del pipeline:

    | Paso | Qué necesita | Por qué |
    |---|---|---|
    | **Paso 5: agrupar activos similares (clustering)** | **Correlación** | Queremos una medida pura de *qué tan parecidos se comportan dos activos*, sin que la volatilidad la distorsione |
    | **Paso 7: optimizar pesos (Markowitz)** | **Covarianza** (con Ledoit-Wolf) | El problema real de portafolio sí necesita las magnitudes — saber cuánto riesgo aporta cada activo, no solo el "patrón" |

    **¿Y por qué la covarianza confundiría al clustering?** Porque mezcla dos cosas:

    - *Qué tanto se mueven juntos* (lo que queremos para el clustering)
    - *Qué tan volátiles son* (irrelevante para agrupar)

    Ejemplo concreto: dos activos de centavos que se mueven idénticamente tienen covarianza pequeña.
    Dos blue chips que se mueven idénticamente tienen covarianza grande. Si usáramos covarianza,
    los activos de centavos terminarían en clusters distintos solo porque sus números son pequeños —
    aunque su *comportamiento* sea exactamente el mismo.

    La correlación divide la covarianza por las volatilidades y elimina ese efecto.
    Queda un número entre -1 y +1 que mide solo el patrón. Perfecto para clustering.
    """)

    if (STORY_DIR / "correlation_full.png").exists():
        st.markdown("**Matriz de correlación de una muestra de 200 activos** — el mar de rojo es lo que necesitamos comprimir:")
        st.image(str(STORY_DIR / "correlation_full.png"), use_container_width=True)

    # ── 6. Representantes ──
    st.markdown("## 6. Eligiendo un Representante de Cada Cluster")
    st.markdown("""
    Una vez que tenemos N clusters de activos similares, necesitamos un representante de cada uno.

    | Regla | Lógica | Trade-off |
    |---|---|---|
    | **Mayor Sharpe individual** ✓ | Mejor retorno ajustado por riesgo del grupo | Puede sobreajustarse a ganadores del pasado |
    | Mayor retorno | El más agresivo | Ignora el riesgo |
    | Más cercano al centroide | El más "promedio" | No premia desempeño |
    | Menor volatilidad | El más defensivo | Pierde activos de crecimiento |

    Elegimos **mayor Sharpe individual**: el activo cuyo Sharpe histórico es el más alto dentro
    de su cluster. Es intuitivo pero introduce un *sesgo de supervivencia* — los ganadores
    del pasado no necesariamente seguirán siéndolo. Parte de la brecha entre dentro y fuera de muestra
    proviene de esta elección.
    """)

    # ── 7. Optimización ──
    st.markdown("## 7. Optimizando los Pesos de los N Representantes")
    st.markdown("""
    Ya tenemos los 50 representantes. Lo importante: **descartamos por completo la matriz "sucia"
    de 2,000 activos y construimos una nueva matriz de covarianza desde cero, solo con los 50 elegidos.**

    El pipeline completo se ve así:
    """)

    st.markdown("""
    <div class='workflow-step'><strong>Paso 1:</strong> Universo de 2,000 activos × 60 meses</div>
    <div class='workflow-step'><strong>Paso 2:</strong> Calcular matriz de correlaciones (ruidosa, pero <em>el clustering tolera el ruido</em>)</div>
    <div class='workflow-step'><strong>Paso 3:</strong> Clusterizar en 50 grupos</div>
    <div class='workflow-step'><strong>Paso 4:</strong> Elegir un representante por cluster → 50 activos</div>
    <div class='workflow-step'><strong>Paso 5:</strong> 🔁 <em>RECALCULAR</em> la matriz de covarianza desde cero, <strong>solo con esos 50 activos</strong></div>
    <div class='workflow-step'><strong>Paso 6:</strong> Aplicar Ledoit-Wolf a esa matriz nueva de 50×50</div>
    <div class='workflow-step'><strong>Paso 7:</strong> Markowitz max-Sharpe sobre los 50 activos</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### ¿Cuánto mejoró el problema con esto?

    | | Sin clustering | Con clustering |
    |---|---|---|
    | Pares a estimar | 2,000,000 | **1,275** |
    | Observaciones | 60 meses | 60 meses |
    | Ruido por par | ±0.13 | ±0.13 (igual) |
    | "Oportunidades falsas" en las que el optimizador puede caer | Miles | Pocas |

    Nota algo interesante: **el ruido por par no cambia** (sigue siendo ±0.13, depende solo del
    número de observaciones). Lo que cambia drásticamente es la **cantidad de pares**. Con 1,275 pares
    ruidosos en vez de 2 millones, el optimizador tiene muchísimas menos trampas en las que caer.

    ### Ledoit-Wolf en este paso

    Aplicamos Ledoit-Wolf sobre la matriz nueva de 50×50 (no sobre la original de 2,000×2,000).
    Aquí no es el "rescatador" del pipeline — el clustering ya hizo el trabajo pesado.
    Ledoit-Wolf es un **pulido final** que:

    - Empuja los valores extremos que aún sobreviven hacia el promedio
    - Estabiliza los pesos entre rebalanceos (menos turnover si se ejecutara en la vida real)
    - Es gratis y siempre ayuda un poco

    """)

    st.markdown("""
    <div class='explainer-box'>
    <strong>Ledoit-Wolf en lenguaje sencillo:</strong> en vez de creerle ciegamente a los valores
    extremos de la matriz, los "tiramos" un poco hacia el promedio. Si una correlación dice 0.95,
    sospechamos que es ruido y la bajamos un poco. Si dice -0.05, sospechamos lo mismo y la subimos
    hacia el promedio del mercado.
    <br><br>
    El método de Ledoit-Wolf calcula <em>matemáticamente</em> cuánto hay que encoger,
    sin necesidad de elegir parámetros a mano.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Finalmente, resolvemos los pesos que **maximizan el Sharpe** sujetos a:

    - `peso_mínimo` y `peso_máximo` por activo
    - Activos `fijos` que siempre se incluyen
    """)

    # ── 8. La prueba honesta ──
    st.markdown("## 8. La Prueba Honesta: Backtest Walk-Forward")

    st.markdown("""
    <div class='explainer-box'>
    <strong>Dentro de muestra vs. fuera de muestra — la distinción clave</strong>
    <br><br>
    El <strong>Sharpe dentro de muestra (in-sample)</strong> se calcula sobre los mismos datos
    que el optimizador usó para elegir los pesos. Es la versión <em>tramposa</em>:
    el optimizador mira la historia, encuentra la combinación que mejor se ajusta a esa historia,
    y luego le preguntas qué Sharpe tiene. Por supuesto se ve excelente — fue diseñado precisamente
    para esa historia exacta. Casi nunca refleja el desempeño real futuro.
    <br><br>
    El <strong>Sharpe realizado (fuera de muestra, OOS)</strong> se calcula sobre datos que el
    optimizador <em>nunca vio</em> cuando eligió los pesos. Es la versión honesta. Es el único
    número que sirve para juzgar si la estrategia realmente funciona.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='explainer-box'>
    <strong>¿Cómo conseguimos datos "que el optimizador nunca vio"?</strong>
    <br><br>
    Imagina que viajamos al pasado y nos detenemos cada trimestre.
    En cada parada le decimos al algoritmo: "<em>solo puedes ver los datos hasta este punto.
    Construye un portafolio.</em>"
    <br><br>
    Luego dejamos correr el reloj 3 meses hacia adelante <em>sin tocar nada</em>, y observamos
    cómo le fue al portafolio. Ese es un periodo "fuera de muestra": el optimizador no vio esos
    3 meses cuando eligió los pesos.
    <br><br>
    Repetimos esto trimestre tras trimestre. Cosemos todo en una sola serie de retornos.
    El Sharpe calculado sobre esa serie cosida es el <strong>Sharpe realizado</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Esquema visual del proceso:**")
    st.markdown("""
    <div class='workflow-step'><strong>Trimestre 1:</strong> Entrena con datos de Jul 2021 → Jun 2024 →
    elige pesos W₁ → aplica W₁ a Jul-Sep 2024 → registra 3 retornos mensuales reales</div>
    <div class='workflow-step'><strong>Trimestre 2:</strong> Entrena con datos de Oct 2021 → Sep 2024 →
    elige pesos W₂ → aplica W₂ a Oct-Dic 2024 → registra 3 retornos mensuales reales</div>
    <div class='workflow-step'><strong>Trimestre 3:</strong> Entrena con datos de Ene 2022 → Dic 2024 →
    elige pesos W₃ → aplica W₃ a Ene-Mar 2025 → registra 3 retornos mensuales reales</div>
    <div class='workflow-step'>... (se repite 8 veces, en total 24 meses fuera de muestra)</div>
    <div class='workflow-step'><strong>Final:</strong> calculamos Sharpe, retorno y drawdown sobre
    los 24 retornos mensuales recolectados. Eso es lo que reportamos.</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    El gráfico de abajo compara, en cada trimestre, el **Sharpe dentro de muestra** (naranja, lo que el optimizador
    prometió) contra el **Sharpe realizado fuera de muestra** (azul, lo que realmente ocurrió en los 3 meses siguientes).

    La brecha entre las dos barras es el **costo del error de estimación**.
    """)
    if (STORY_DIR / "bias_variance.png").exists():
        st.image(str(STORY_DIR / "bias_variance.png"), use_container_width=True)

    # ── 9. Resultados ──
    st.markdown("## 9. Los Resultados")
    if (STORY_DIR / "drawdown.png").exists():
        st.image(str(STORY_DIR / "drawdown.png"), use_container_width=True)
        st.caption("**Drawdown (gráfico de inmersión):** la peor pérdida desde un pico anterior en cada momento. Un punto de −20% significa que en ese momento el portafolio estaba 20% por debajo de su máximo previo.")
    if (STORY_DIR / "rolling_sharpe.png").exists():
        st.image(str(STORY_DIR / "rolling_sharpe.png"), use_container_width=True)
        st.caption("**Sharpe móvil a 12 meses:** ¿la estrategia es consistentemente buena, o depende de un solo trimestre afortunado?")
    if (STORY_DIR / "outperformance.png").exists():
        st.image(str(STORY_DIR / "outperformance.png"), use_container_width=True)
        st.caption("**Sobre-desempeño acumulado vs SPY:** áreas verdes = la estrategia va arriba, naranja = SPY va arriba.")
    if (STORY_DIR / "n_sensitivity.png").exists():
        st.image(str(STORY_DIR / "n_sensitivity.png"), use_container_width=True)
        st.caption("**Sensibilidad a N:** ¿cómo cambia el Sharpe realizado al variar el número de clusters? Útil para elegir N.")

    # ── 10. Veredicto ──
    st.markdown("## 10. El Veredicto")
    try:
        beat = s['total_return'] > b['total_return']
        st.markdown(f"""
        - **¿Le ganó al mercado por retorno total?** {'Sí' if beat else 'No'} — {s['total_return']:.0%} vs {b['total_return']:.0%} del SPY.
        - **¿Por Sharpe (retorno ajustado por riesgo)?** Estrategia {s['sharpe']:.2f} vs SPY {b['sharpe']:.2f}.
        - **Máximo drawdown:** estrategia {s['max_drawdown']:.0%} vs SPY {b['max_drawdown']:.0%}.

        **Advertencias importantes que podrían cambiar la conclusión:**
        - **Sin costos de transacción** — rebalancear 50 acciones globales cada trimestre quitaría 0.5%–1% anual
        - **Ventana corta** — solo ~2 años de prueba fuera de muestra. Un mercado bajista podría cambiar todo
        - **Sesgo de supervivencia** en el universo — usamos los componentes *actuales* de los índices (empresas que sobrevivieron)
        - **Sin fricción fiscal**
        - **El periodo OOS fue inusualmente favorable para SPY** (Sharpe 1.18 vs el histórico ~0.5). Es decir, la estrategia le ganó en un periodo donde SPY estuvo *especialmente fuerte*

        La estrategia es **plausiblemente viable** — un Sharpe realizado de {s['sharpe']:.2f} está cómodamente
        por encima de 1 (el umbral aproximado para que una estrategia activa "valga la pena"), y la
        diversificación geográfica protege contra riesgos regionales. Pero la brecha entre dentro de
        muestra (4-6) y fuera de muestra ({s['sharpe']:.2f}) es el mensaje honesto:
        **la mayor parte de lo que el optimizador "descubre" en datos históricos es ruido**,
        y solo una fracción sobrevive al contacto con el futuro.
        """)
    except Exception:
        pass

    st.success("**Pruébalo tú mismo** en la pestaña **🛠 La Herramienta** — mueve las palancas y observa cómo cambia el portafolio.")


# =================================================================
# PESTAÑA HERRAMIENTA
# =================================================================
with tool_tab:
    st.title("Construye un Portafolio")
    st.caption("Configura las palancas, presiona Optimizar. Pasa el cursor sobre cualquier métrica para ver una definición.")

    RETURNS = cached_returns()
    all_regions = sorted({ticker_region(t) for t in drop_benchmarks(RETURNS).columns})

    with st.sidebar:
        st.header("Controles")
        st.caption("Cada palanca ajusta cómo se construye el portafolio.")

        n_clusters = st.slider("# de clusters (tamaño del portafolio)", 5, 100, 20,
                               help="Número de grupos behaviorales en los que dividir el universo. Se elige un representante por cluster. Más clusters = portafolio más granular.")
        col_a, col_b = st.columns(2)
        min_w = col_a.slider("Peso mínimo", 0.0, 0.10, 0.0, 0.005, format="%.1f%%",
                             help="Piso del peso por activo. Si lo subes por encima de 0, fuerzas a que todos los activos seleccionados estén en el portafolio. En 0, el optimizador puede eliminar activos débiles.")
        max_w = col_b.slider("Peso máximo", 0.05, 1.0, 0.25, 0.05, format="%.0f%%",
                             help="Techo del peso por activo. Evita la concentración en una sola acción.")
        min_sharpe = st.slider("Sharpe individual mínimo", -1.0, 2.5, 0.0, 0.1,
                               help="Excluye activos cuyo Sharpe individual histórico esté por debajo de este umbral. Más alto = universo más exigente.")
        pinned = st.text_input("Activos fijos (incluir siempre)", "",
                               placeholder="AAPL, BRK-B, ASML.AS",
                               help="Fuerza la inclusión de estos activos, sin importar el clustering. Separa con comas.")
        excluded = st.text_input("Activos excluidos", "",
                                 placeholder="TSLA, GME",
                                 help="Excluye estos activos del universo. Separa con comas.")
        regions_sel = st.multiselect("Regiones", all_regions, default=all_regions,
                                     help="Restringe el universo a las regiones elegidas. Vacío = ninguna.")
        run_backtest = st.checkbox("Ejecutar backtest fuera de muestra", True,
                                   help="Si está activado, corre también el backtest walk-forward. Agrega ~30 segundos pero produce el Sharpe realizado honesto.")
        go = st.button("🚀 Optimizar", use_container_width=True, type="primary")

    if "result" not in st.session_state:
        st.session_state.result = None
        st.session_state.bt = None

    def parse_t(s): return [x.strip().upper() for x in s.split(",") if x.strip()]

    if go:
        with st.spinner("Optimizando portafolio... (puede tomar 20-40 segundos con backtest)"):
            try:
                regions = regions_sel if len(regions_sel) < len(all_regions) else None
                st.session_state.result = optimize_portfolio(
                    RETURNS,
                    n_clusters=n_clusters,
                    min_weight=min_w, max_weight=max_w,
                    pinned=parse_t(pinned), excluded=parse_t(excluded),
                    min_sharpe=min_sharpe if min_sharpe > -0.99 else None,
                    regions=regions,
                )
                st.session_state.bt = None
                if run_backtest:
                    st.session_state.bt = backtest_strategy(
                        RETURNS,
                        n_clusters=n_clusters,
                        min_weight=min_w, max_weight=max_w,
                        pinned=parse_t(pinned), excluded=parse_t(excluded),
                        min_sharpe=min_sharpe if min_sharpe > -0.99 else None,
                        regions=regions,
                    )
            except Exception as e:
                st.error(f"Falló la optimización: {e}")

    result = st.session_state.result
    bt = st.session_state.bt

    if result is None:
        st.info("Configura las palancas en la barra lateral y presiona **Optimizar** para construir un portafolio.")
    else:
        weights = result["weights"]
        weights = weights[weights > 0.001].sort_values(ascending=False)
        perf = result["performance"]
        metrics = asset_metrics(result["selected_returns"], weights)

        # KPIs
        spy = get_benchmark(RETURNS, "SPY")
        bt_stats = perf_stats(bt) if bt is not None and len(bt) else None
        spy_stats = perf_stats(spy.loc[bt.index].dropna()) if bt is not None and len(bt) else None

        st.markdown("### Métricas principales")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Activos", len(weights), help="Número de activos con peso superior al 0.1%.")
        c2.metric("Retorno anual (OOS)",
                  f"{bt_stats['annual_return']:.2%}" if bt_stats else "Ejecuta backtest →",
                  delta=f"SPY: {spy_stats['annual_return']:.2%}" if spy_stats else None,
                  help="Retorno anualizado calculado sobre el backtest fuera de muestra (walk-forward). El SPY se muestra como referencia.")
        c3.metric("Volatilidad (OOS)",
                  f"{bt_stats['annual_vol']:.2%}" if bt_stats else "—",
                  delta=f"SPY: {spy_stats['annual_vol']:.2%}" if spy_stats else None, delta_color="inverse",
                  help="Desviación estándar anualizada de los retornos. Menor = camino más suave.")
        c4.metric("Sharpe realizado (OOS)",
                  f"{bt_stats['sharpe']:.2f}" if bt_stats else "—",
                  delta=f"SPY: {spy_stats['sharpe']:.2f}" if spy_stats else None,
                  help="Retorno ajustado por riesgo, calculado de forma honesta sobre el backtest walk-forward. Compáralo con el Sharpe dentro de muestra (más abajo) — la diferencia suele ser grande.")
        c5.metric("Máximo drawdown (OOS)",
                  f"{bt_stats['max_drawdown']:.0%}" if bt_stats else "—",
                  delta=f"SPY: {spy_stats['max_drawdown']:.0%}" if spy_stats else None, delta_color="inverse",
                  help="Peor pérdida desde un pico anterior durante el periodo fuera de muestra.")

        with st.expander("Más métricas (dentro de muestra)", expanded=False):
            st.markdown("""
            <div class='explainer-box'>
            <strong>¿Qué es "dentro de muestra"?</strong>
            Estas métricas se calculan sobre los <em>mismos datos</em> que el optimizador usó para elegir los pesos.
            Son la versión <strong>tramposa y optimista</strong> de las métricas: el optimizador fue diseñado
            exactamente para esa historia, así que por supuesto se ve genial. Compara contra el
            <strong>Sharpe realizado (OOS)</strong> de arriba — la brecha entre ambos es el costo del
            error de estimación, y es el único número honesto sobre qué tan bien funciona la estrategia.
            </div>
            """, unsafe_allow_html=True)
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Sharpe dentro de muestra", f"{perf['sharpe']:.2f}",
                       help="Sharpe calculado sobre los mismos datos que el optimizador usó para elegir los pesos. Típicamente mucho más alto que el Sharpe realizado.")
            cc1.metric("Retorno anual dentro de muestra", f"{perf['annual_return']:.2%}",
                       help="Retorno anualizado calculado sobre los mismos datos usados para elegir pesos. Optimista.")
            cc2.metric("Volatilidad dentro de muestra", f"{perf['annual_vol']:.2%}",
                       help="Volatilidad del portafolio elegido sobre los datos de entrenamiento.")
            if bt_stats:
                cc3.metric("Retorno total OOS", f"{bt_stats['total_return']:.2%}",
                           help="Retorno acumulado durante toda la ventana fuera de muestra.")
                cc4.metric("Periodos OOS evaluados", len(bt) // 3,
                           help="Número de rebalanceos trimestrales simulados en el backtest.")

        # Composición
        st.markdown("### Composición")
        st.caption("Cómo se reparte el capital entre los activos elegidos y cómo se traduce en regiones.")
        col_p, col_r = st.columns([1.2, 1])
        with col_p:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(weights.values, labels=weights.index, autopct="%1.0f%%", startangle=90,
                   colors=sns.color_palette("tab20", len(weights)),
                   textprops={"fontsize": 9})
            ax.set_title("Pesos")
            st.pyplot(fig, use_container_width=True)
        with col_r:
            fig, ax = plt.subplots(figsize=(6, 6))
            by_region = metrics.groupby("region")["weight"].sum().sort_values(ascending=False)
            ax.barh(by_region.index, by_region.values, color="#4a6fa5")
            ax.invert_yaxis()
            ax.set_xlabel("Peso"); ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            ax.set_title("Por región")
            st.pyplot(fig, use_container_width=True)

        # Tabla por activo
        st.markdown("### Detalle por activo")
        st.caption("Una fila por tenencia. Haz click en los encabezados para ordenar.")
        view = metrics.copy()
        view["weight"] = (view["weight"] * 100).round(2)
        view["annual_return"] = (view["annual_return"] * 100).round(2)
        view["volatility"] = (view["volatility"] * 100).round(2)
        view["sharpe"] = view["sharpe"].round(2)
        view.columns = ["Peso %", "Retorno anual %", "Volatilidad %", "Sharpe", "Región"]
        st.dataframe(view, use_container_width=True)

        # Curva de capital
        st.markdown("### Curva de capital")
        st.caption("Crecimiento acumulado de $1 invertido al inicio. Un valor de 2.0 = se duplicó.")
        in_sample_returns = (result["selected_returns"][weights.index] * weights).sum(axis=1)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(equity_curve(in_sample_returns).index, equity_curve(in_sample_returns).values,
                label="Dentro de muestra (pesos fijos sobre toda la historia)", linewidth=2, color="#2c4a7a")
        if "SPY" in RETURNS.columns:
            spy_eq = (1 + RETURNS["SPY"].loc[in_sample_returns.index].dropna()).cumprod()
            ax.plot(spy_eq.index, spy_eq.values, label="SPY", linewidth=2, color="#a55c3f", alpha=0.85)
        if bt is not None and len(bt):
            ax.plot(equity_curve(bt).index, equity_curve(bt).values,
                    label="Backtest fuera de muestra (estrategia)", linewidth=2, linestyle="--", color="#3a7a3a")
        ax.legend(); ax.grid(alpha=0.3)
        ax.set_ylabel("Valor acumulado")
        st.pyplot(fig, use_container_width=True)

        # Sobre-desempeño acumulado vs SPY
        if bt is not None and len(bt) and "SPY" in RETURNS.columns:
            st.markdown("### Sobre-desempeño acumulado vs SPY")
            st.caption("Diferencia entre la curva de la estrategia y la del SPY. Verde = estrategia adelante, naranja = SPY adelante.")
            spy_oos = RETURNS["SPY"].loc[bt.index].dropna()
            diff = (1 + bt).cumprod() - (1 + spy_oos).cumprod()
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.fill_between(diff.index, diff.values, 0, where=(diff.values >= 0), color="#3a7a3a", alpha=0.5)
            ax.fill_between(diff.index, diff.values, 0, where=(diff.values < 0), color="#a55c3f", alpha=0.5)
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.grid(alpha=0.3)
            st.pyplot(fig, use_container_width=True)

        # Retornos por periodo
        if bt is not None and len(bt):
            st.markdown("### Retornos por periodo (OOS)")
            st.caption("Cómo le fue a la estrategia cada mes fuera de muestra. La consistencia importa tanto como el total.")
            fig, ax = plt.subplots(figsize=(11, 3.5))
            colors = ["#3a7a3a" if r >= 0 else "#a55c3f" for r in bt.values]
            ax.bar(bt.index, bt.values, color=colors, width=20)
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            ax.set_ylabel("Retorno mensual")
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig, use_container_width=True)

        # Scatter
        with st.expander("📊 Scatter riesgo-retorno (por activo)", expanded=False):
            st.caption("Cada punto es una tenencia. X = volatilidad, Y = retorno anualizado. Tamaño = peso. Color = Sharpe individual.")
            fig, ax = plt.subplots(figsize=(10, 6))
            sizes = (metrics["weight"] * 2000).clip(lower=30)
            sc = ax.scatter(metrics["volatility"], metrics["annual_return"], s=sizes, alpha=0.6,
                            c=metrics["sharpe"], cmap="RdYlGn")
            for t in metrics.index:
                ax.annotate(t, (metrics.loc[t,"volatility"], metrics.loc[t,"annual_return"]), fontsize=8)
            plt.colorbar(sc, label="Sharpe individual")
            ax.set_xlabel("Volatilidad"); ax.set_ylabel("Retorno anual")
            ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            ax.grid(alpha=0.3)
            st.pyplot(fig, use_container_width=True)

        # Correlación
        with st.expander("🔥 Matriz de correlación (activos elegidos)", expanded=False):
            st.caption("Correlación entre las tenencias. Rojo = se mueven juntas, azul = se mueven opuestas, blanco = independientes. Buena diversificación = mayoría de colores pálidos.")
            fig, ax = plt.subplots(figsize=(9, 7))
            corr = result["selected_returns"][weights.index].corr()
            sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax,
                        cbar_kws={"label": "Correlación"})
            st.pyplot(fig, use_container_width=True)

        # Dendrograma
        with st.expander("🌳 Dendrograma (corte actual de clusters)", expanded=False):
            st.caption("Árbol de todos los activos filtrados del universo. La línea de corte determina N.")
            fig, ax = plt.subplots(figsize=(14, 5))
            dendrogram(result["linkage"],
                       labels=list(result["filtered_returns"].columns),
                       ax=ax, leaf_font_size=5,
                       no_labels=result["filtered_returns"].shape[1] > 150,
                       color_threshold=result["linkage"][-(n_clusters - 1), 2])
            st.pyplot(fig, use_container_width=True)
