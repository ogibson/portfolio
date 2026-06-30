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
    buy_and_hold_backtest,
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
    st.title("¿Puede el Clustering Ganarle al Mercado?")
    st.markdown("*Un proyecto de investigación que extiende la teoría de Markowitz "
                "para escoger **qué activos comprar y en qué proporción**, "
                "desde un universo de miles de acciones globales.*")

    # ── El Problema ──
    st.markdown("## El Problema")
    st.markdown("""
    La optimización de Markowitz clásica resuelve los pesos de un portafolio una vez que ya decidiste
    qué activos comprar. Responde a la pregunta:
    > *"Dados estos 5 ETFs, ¿qué combinación maximiza el Sharpe?"*

    Este proyecto invierte la pregunta: desde un universo de ~2,000 activos globales, queremos elegir
    simultáneamente **qué N activos incluir y cómo ponderarlos**. En teoría, esto permitiría encontrar
    el portafolio que maximiza el retorno por unidad de riesgo. En la práctica sabemos que los resultados
    pasados no aseguran resultados futuros, pero la herramienta puede ser útil para diseñar ETFs o para
    complementar portafolios existentes con activos globales que aporten diversificación de riesgo.

    ### ¿Qué dificultades nos encontramos?

    - Elegir N activos de M es un problema **combinatorio** (NP-difícil): existen C(2000, 50) ≈ 10⁸⁵ portafolios posibles de 50 activos.
    - No se puede ejecutar Markowitz directamente sobre 2,000 activos, porque la matriz de covarianza no se puede estimar de forma confiable con tan pocos datos (lo explicamos en detalle en la sección 2).
    """)

    # ── La Solución ──
    st.markdown("## La Solución")
    st.markdown("""
    La solución propuesta consiste en usar **clustering jerárquico** para reducir el universo a N representantes
    con comportamiento distinto entre sí, y luego correr Markowitz sobre esos N. El clustering convierte un
    problema combinatorio intratable en un pipeline de dos etapas.

    **El pipeline completo se ve así:**
    """)
    st.markdown("""
    <div class='workflow-step'><strong>Paso 1:</strong> Se elige un universo de 2,000 activos × 60 meses.</div>
    <div class='workflow-step'><strong>Paso 2:</strong> Se calcula la matriz de correlaciones entre los 2,000 activos.</div>
    <div class='workflow-step'><strong>Paso 3:</strong> Se clusterizan los activos en N grupos, dependiendo de cuántos activos se quiera en el portafolio final.</div>
    <div class='workflow-step'><strong>Paso 4:</strong> Se elige un representante por cluster, llegando a N activos.</div>
    <div class='workflow-step'><strong>Paso 5:</strong> Se recalcula la matriz de covarianza con esos N activos.</div>
    <div class='workflow-step'><strong>Paso 6:</strong> Se aplica Ledoit-Wolf a esa matriz nueva para estabilizar las covarianzas.</div>
    <div class='workflow-step'><strong>Paso 7:</strong> Se concluye con la aplicación tradicional de Markowitz, para optimizar la composición del portafolio maximizando el ratio de Sharpe.</div>
    """, unsafe_allow_html=True)

    # ── 1. Diseñando el Universo ──
    st.markdown("## 1. Diseñando el Universo")
    st.markdown("**¿Qué activos entran en el conjunto candidato?**")
    st.markdown("""
    Elegimos activos individuales y no ETFs (no queremos un *fondo de fondos*; queremos el comportamiento
    subyacente). Cubrimos mercados globales para aprovechar la diversificación entre economías no correlacionadas.
    Un portafolio solo de EE.UU. no puede escapar a un crash sistémico estadounidense.

    El universo se ensambla desde **17 fuentes**: índices bursátiles y las principales tenencias de ETFs regionales.
    Los datos se muestrean con **frecuencia mensual** sobre una ventana de **5 años (2021-2026)**, lo que nos da
    60 observaciones por activo.

    **Filtros de calidad aplicados:**
    - Descartar activos con menos del 80% de cobertura en los 5 años (maneja IPOs recientes y delistings)
    - Descartar activos con cualquier retorno mensual > 50% en valor absoluto (atrapa errores de yfinance por stock splits mal ajustados)

    Después del filtrado quedan ~2,000 activos utilizables.
    """)
    if (STORY_DIR / "universe.png").exists():
        st.image(str(STORY_DIR / "universe.png"), use_container_width=True)

    # ── 2. Cálculo de la matriz de correlaciones ──
    st.markdown("## 2. Cálculo de la matriz de correlaciones")
    st.markdown("""
    Si quisiéramos aplicar el modelo de Markowitz como suele utilizarse en finanzas para optimizar directamente
    la composición del portafolio, deberíamos calcular la matriz de covarianzas sobre los 2,000 activos que se
    incluyeron en el universo. Sin embargo, en la práctica esto resulta imposible.

    La matriz de covarianza es una tabla gigante de relaciones: para cada par de activos (Apple-Microsoft,
    Apple-Toyota, etc.) intenta capturar qué tan juntos se mueven. Con 2,000 activos hay alrededor de
    **2 millones de relaciones** distintas que estimar. Pero solo tenemos **60 meses de historia** para estimarlas.
    El problema no es que no se pueda calcular la matriz, la fórmula funciona y devuelve números.
    **El problema es que esos números están llenos de ruido.**

    ### ¿De dónde viene el ruido?

    La correlación "verdadera" entre dos activos (digamos AAPL y MSFT) es un número real que existe en el mundo
    pero que **nunca podemos observar directamente**. Para conocerlo perfectamente necesitaríamos infinitos datos.

    Lo único que podemos hacer es **estimarlo** con los datos disponibles. En nuestro caso, 60 meses.

    Imaginemos que tomamos dos ventanas de 60 meses distintas:

    - **Ventana A** (2018-2023) → calculas la correlación AAPL-MSFT y te da 0.72
    - **Ventana B** (2019-2024) → calculas la misma correlación y te da 0.81

    Ninguna de estos valores es la correlación verdadera. Ambas son **estimaciones ruidosas** del valor real
    (que tal vez sea 0.77, pero no lo sabemos). La diferencia entre ambas —esos 9 puntos porcentuales— es
    **ruido de muestreo**.

    ### Cuánto ruido

    Hay una regla aproximada: el ruido en una correlación estimada es de aproximadamente:

    > ruido ≈ 1 / √(número de observaciones)

    Con 60 meses:

    > ruido ≈ 1 / √60 ≈ ±0.13

    Esto significa que cada correlación que calculamos tiene un **error típico de ±0.13** alrededor del valor verdadero.

    ### El verdadero problema: dos activos sin relación

    Pensemos en dos activos cuya correlación verdadera es **0** (no tienen ninguna relación real).
    Con 60 meses, nuestro estimado va a oscilar entre **-0.13 y +0.13** solo por puro azar.

    Ahora, el optimizador de Markowitz mira una correlación de -0.13 y piensa:
    *"¡Estos dos activos van en direcciones opuestas! Es diversificación gratis, voy a cargar peso acá."*

    Pero esa correlación de -0.13 **no significa nada**, ya que la relación real es cero.
    El optimizador acaba de apostar el portafolio a una "oportunidad" que solo existe en los datos pasados, no en la realidad.

    ### Por qué empeora con 2,000 activos

    Con 2,000 activos tenemos **2 millones de pares**. Si cada uno tiene un ruido típico de ±0.13, muchos
    de esos 2 millones van a aterrizar por azar en valores que **parecen relaciones reales pero no lo son**.
    El optimizador, que busca correlaciones extremas para construir el portafolio, va a encontrar miles
    de "oportunidades" falsas, y esto nos generaba fallos al intentar aplicar el modelo de Markowitz sin alteraciones.

    ### Por qué con más datos sería menos malo

    Con 600 meses (50 años) de datos:

    > ruido ≈ 1 / √600 ≈ ±0.04

    El error se reduce, las correlaciones espurias casi desaparecen. Pero esos datos no existen para la mayoría
    de activos. Otra posibilidad sería medir los datos semanales para aumentar el número de observaciones,
    pero observamos empíricamente que esto resulta en una performance peor que utilizando datos mensuales
    ya que la variación semana a semana es muy baja.

    ### La solución a este problema

    Para que Markowitz funcione necesitamos atacar el problema en **dos frentes complementarios**:

    1. **Reducir el universo a un número manejable de activos** mediante clustering (sección 3). Esto pasa de 2 millones de pares ruidosos a solo ~1,275 pares.
    2. **Limpiar lo que queda** con la reducción de Ledoit-Wolf antes de optimizar (sección 6). Esto suaviza los valores extremos que aún sobreviven.
    """)

    # ── 3. Clustering de los datos ──
    st.markdown("## 3. Clustering de los datos")
    st.markdown("""
    En este paso buscamos reducir 2,000 activos a N grupos con comportamiento similar, para seleccionar luego
    los N activos candidatos. Los activos que se mueven juntos (alta correlación) actúan casi como sustitutos
    entre sí. Tener cualquiera de ellos te da prácticamente la misma exposición que tener cualquier otro del grupo.

    Aplicamos **clustering jerárquico (Ward)** sobre una matriz de distancias derivada de las correlaciones:

    > distancia = 1 − correlación

    Dos activos con correlación = 1 están a distancia 0 (mismo cluster). Activos independientes están a distancia 1.
    Activos con correlación negativa están a distancia 2 (clusters muy distintos).

    Los dendrogramas de abajo muestran cómo se forma el árbol de agrupamiento.
    La **línea de corte horizontal** determina N: corte alto, pocos clusters grandes; corte bajo, muchos clusters finos.
    """)
    st.markdown("""
    <div class='explainer-box'>
    <strong>Si la matriz de 2,000 activos tiene mucho ruido, ¿por qué la usamos para clustering?</strong>
    <br><br>
    Porque <strong>el clustering tolera el ruido</strong>. Solo necesita ver la estructura gruesa:
    "estos activos tecnológicos se parecen entre sí, este grupo de utilities se parece entre sí,
    los dos grupos son distintos". Aunque cada correlación individual tenga error, la <em>estructura general</em>
    de qué se agrupa con qué emerge correctamente. Es como leer un mapa borroso: no se distingue cada casa,
    pero sí se ve dónde están los barrios.
    <br><br>
    En cambio, Markowitz <strong>no tolera el ruido</strong>: busca activamente los valores extremos
    (correlaciones cercanas a cero le parecen "diversificación gratis") y apuesta el portafolio a esas
    oportunidades que muchas veces son ficticias. Por eso podemos usar la matriz ruidosa para clustering
    pero <em>no</em> para optimización directa por Markowitz.
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(3)
    for col, n in zip(cols, (5, 20, 50)):
        if (STORY_DIR / f"dendrogram_n{n}.png").exists():
            col.image(str(STORY_DIR / f"dendrogram_n{n}.png"),
                      caption=f"Corte en N = {n}", use_container_width=True)

    st.markdown("### ¿Por qué correlación y no covarianza?")
    st.markdown("""
    Esta es una pregunta importante porque **Markowitz sí usa la matriz de covarianza** en su fórmula original.
    No la estamos descartando, sino que la usamos más adelante, en el paso 6.

    La covarianza no es ideal para realizar el clustering de los datos porque mezcla dos piezas de información sobre los activos:

    - *Qué tanto se mueven juntos* (lo que queremos para el clustering)
    - *Qué tan volátiles son* (irrelevante para agrupar)

    Ejemplo concreto: Coca-Cola (KO) y Pepsi (PEP) tienen correlación de ~0.7 y son poco volátiles.
    Tesla (TSLA) y Rivian (RIVN) también tienen correlación de ~0.7, pero son muchísimo más volátiles.
    Misma correlación, pero la covarianza TSLA-RIVN es varias veces más grande que la de KO-PEP, solo porque
    sus números son más grandes.

    Si usáramos covarianza directamente, el clustering agruparía por "qué tan volátil es el activo" en lugar
    de únicamente "con quién se mueve". Coca-Cola podría terminar en un cluster distinto a Pepsi simplemente
    porque sus retornos son chiquitos, aunque su comportamiento sea idéntico.

    La correlación divide la covarianza por las volatilidades y elimina ese efecto. Queda un número entre -1 y +1
    que mide solo el patrón. Perfecto para clustering.
    """)
    if (STORY_DIR / "correlation_full.png").exists():
        st.markdown("**Matriz de correlación de una muestra de 200 activos:**")
        st.image(str(STORY_DIR / "correlation_full.png"), use_container_width=True)

    # ── 4. Eligiendo un Representante ──
    st.markdown("## 4. Eligiendo un Representante de Cada Cluster")
    st.markdown("""
    Una vez que tenemos N clusters de activos similares, necesitamos un representante de cada uno.
    """)
    st.markdown("""
    <table style="width:100%; border-collapse:collapse;">
      <thead>
        <tr style="background:#d5e8f0;">
          <th style="padding:8px; border:1px solid #ccc; text-align:left;">Regla</th>
          <th style="padding:8px; border:1px solid #ccc; text-align:left;">Lógica</th>
          <th style="padding:8px; border:1px solid #ccc; text-align:left;">Trade-off</th>
        </tr>
      </thead>
      <tbody>
        <tr style="background:#d4edda;">
          <td style="padding:8px; border:1px solid #ccc;"><strong>✓ Mayor Sharpe individual</strong> (elegida)</td>
          <td style="padding:8px; border:1px solid #ccc;">Mejor retorno ajustado por riesgo del grupo</td>
          <td style="padding:8px; border:1px solid #ccc;">Puede sobreajustarse a ganadores del pasado</td>
        </tr>
        <tr>
          <td style="padding:8px; border:1px solid #ccc;">Mayor retorno</td>
          <td style="padding:8px; border:1px solid #ccc;">El más agresivo</td>
          <td style="padding:8px; border:1px solid #ccc;">Ignora el riesgo</td>
        </tr>
        <tr>
          <td style="padding:8px; border:1px solid #ccc;">Más cercano al centroide</td>
          <td style="padding:8px; border:1px solid #ccc;">El más "promedio"</td>
          <td style="padding:8px; border:1px solid #ccc;">No premia desempeño</td>
        </tr>
        <tr>
          <td style="padding:8px; border:1px solid #ccc;">Menor volatilidad</td>
          <td style="padding:8px; border:1px solid #ccc;">El más defensivo</td>
          <td style="padding:8px; border:1px solid #ccc;">Pierde activos de crecimiento</td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)
    st.markdown("""
    Elegimos **mayor Sharpe individual**: el activo cuyo Sharpe histórico es el más alto dentro de su cluster.
    Esta es la debilidad fundamental de este modelo, ya que los ganadores del pasado no necesariamente seguirán
    siéndolo. Hay una cuota de percibir el futuro en las inversiones que no puede ser reemplazada simplemente
    por los datos. Fundamentalmente por este motivo incorporamos la **selección manual de acciones**, lo que
    permite complementar una tésis de inversión con activos de riesgo complementario de todo el mundo.
    """)

    # ── 5, 6 y 7. Optimización ──
    st.markdown("## 5, 6 y 7. Optimizando los Pesos de los N Representantes")
    st.markdown("""
    Una vez que ya tenemos los N representantes elegidos, **descartamos por completo la matriz de 2,000 activos
    y construimos una nueva matriz de covarianza desde cero, solo con los 50 elegidos**.

    ### Ledoit-Wolf en este paso

    Aplicamos Ledoit-Wolf sobre la matriz nueva de 50×50 (no sobre la original de 2,000×2,000).
    Ledoit-Wolf es un pulido final que:

    - Empuja los valores extremos que aún sobreviven hacia el promedio
    - Estabiliza los pesos entre rebalanceos (menos turnover si se ejecutara en la vida real)

    Decidimos aplicar Ledoit-Wolf porque en nuestros experimentos notamos que de no estabilizar la matriz
    de covarianzas usualmente terminábamos con portfolios imposibles con ratio de Sharpe artificialmente inflado.
    """)
    st.markdown("""
    <div class='explainer-box'>
    <strong>Ledoit-Wolf en lenguaje sencillo:</strong> en vez de creerle ciegamente a los valores extremos
    de la matriz, los "tiramos" un poco hacia el promedio. Si una correlación dice 0.95, sospechamos que
    es ruido y la bajamos un poco. Si dice -0.05, sospechamos lo mismo y la subimos hacia el promedio
    del mercado. El método de Ledoit-Wolf calcula <em>matemáticamente</em> cuánto hay que encoger,
    sin necesidad de elegir parámetros a mano.
    </div>
    """, unsafe_allow_html=True)

    # ── El perfil del usuario ──
    st.markdown("## El perfil del usuario")
    st.markdown("""
    Es crucial que el usuario se sienta cómodo en la elección de los activos. Un portafolio de elevado retorno
    pero que no permite al inversor dormir por las noches no serviría jamás. Por este motivo incluímos varias
    palancas de funcionamiento que permiten modificar qué tipos de activos terminarán en el resultado final.
    """)
    st.markdown("""
    <table style="width:100%; border-collapse:collapse;">
      <thead>
        <tr style="background:#d5e8f0;">
          <th style="padding:8px; border:1px solid #ccc; text-align:left;">Palanca</th>
          <th style="padding:8px; border:1px solid #ccc; text-align:left;">Qué controla</th>
          <th style="padding:8px; border:1px solid #ccc; text-align:left;">Para qué sirve</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="padding:8px; border:1px solid #ccc;"><strong># de clusters (N)</strong></td>
          <td style="padding:8px; border:1px solid #ccc;">Cantidad de activos en el portafolio final</td>
          <td style="padding:8px; border:1px solid #ccc;">Más clusters = portafolio más diversificado y granular; menos clusters = portafolio concentrado</td>
        </tr>
        <tr>
          <td style="padding:8px; border:1px solid #ccc;"><strong>Peso mínimo / máximo</strong></td>
          <td style="padding:8px; border:1px solid #ccc;">Piso y techo de capital por activo</td>
          <td style="padding:8px; border:1px solid #ccc;">Evita que el optimizador concentre todo en una sola acción o descarte activos arbitrariamente</td>
        </tr>
        <tr>
          <td style="padding:8px; border:1px solid #ccc;"><strong>Sharpe individual mínimo</strong></td>
          <td style="padding:8px; border:1px solid #ccc;">Filtra activos con bajo retorno ajustado por riesgo</td>
          <td style="padding:8px; border:1px solid #ccc;">Excluye candidatos débiles antes del clustering. Más alto = universo más exigente</td>
        </tr>
        <tr>
          <td style="padding:8px; border:1px solid #ccc;"><strong>Tolerancia al riesgo</strong></td>
          <td style="padding:8px; border:1px solid #ccc;">Volatilidad máxima admisible por activo</td>
          <td style="padding:8px; border:1px solid #ccc;">Los botones Baja/Media/Alta corresponden a los percentiles 25/50/75 de volatilidad del universo</td>
        </tr>
        <tr>
          <td style="padding:8px; border:1px solid #ccc;"><strong>Activos fijos</strong></td>
          <td style="padding:8px; border:1px solid #ccc;">Acciones que se incluyen sí o sí</td>
          <td style="padding:8px; border:1px solid #ccc;">Reemplazan al representante de su cluster — útil para inyectar una tésis de inversión sin perder diversificación</td>
        </tr>
        <tr>
          <td style="padding:8px; border:1px solid #ccc;"><strong>Activos excluidos</strong></td>
          <td style="padding:8px; border:1px solid #ccc;">Acciones que nunca entran al universo</td>
          <td style="padding:8px; border:1px solid #ccc;">Filtra empresas específicas por razones éticas, de exposición o personales</td>
        </tr>
        <tr>
          <td style="padding:8px; border:1px solid #ccc;"><strong>Regiones</strong></td>
          <td style="padding:8px; border:1px solid #ccc;">Países o continentes permitidos</td>
          <td style="padding:8px; border:1px solid #ccc;">Restringe el universo geográficamente. Por ejemplo: solo Asia-Pacífico, o excluir mercados emergentes</td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

    # ── Backtest ──
    st.markdown("## Testeando el funcionamiento: Backtest Walk-Forward")

    st.markdown("""
    <div class='explainer-box'>
    <strong>Dentro de muestra vs. fuera de muestra.</strong>
    <br><br>
    El <strong>Sharpe dentro de muestra (in-sample)</strong> se calcula sobre los mismos datos que el optimizador
    usó para elegir los pesos. El optimizador mira la historia, encuentra la combinación que mejor se ajusta a esa
    historia, y luego le preguntas qué Sharpe tiene. Por supuesto se ve excelente, ya que fue diseñado maximizando
    el Sharpe con esos datos históricos. Casi nunca refleja el desempeño real futuro.
    <br><br>
    El <strong>Sharpe realizado (fuera de muestra, OOS)</strong> se calcula sobre datos que el optimizador
    <em>nunca vio</em> cuando eligió los pesos. Es aproximadamente igual a utilizar un set de entrenamiento
    y uno de validación, y es el único número que sirve para juzgar si la estrategia realmente funciona.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='explainer-box'>
    <strong>¿Cómo conseguimos datos "que el optimizador nunca vio"?</strong>
    <br><br>
    Para esto, viajamos al pasado y nos detenemos en cada trimestre. En cada parada le decimos al algoritmo:
    "<em>solo podés ver los datos hasta este punto. Construí un portafolio.</em>"
    <br><br>
    Luego dejamos correr el reloj 3 meses hacia adelante <em>sin tocar nada</em>, y observamos cómo le fue
    al portafolio. Ese es un periodo "fuera de muestra": el optimizador no vio esos 3 meses cuando eligió los pesos.
    <br><br>
    Repetimos esto trimestre tras trimestre. Cosemos todo en una sola serie de retornos. El Sharpe calculado
    sobre esa serie cosida es el <strong>Sharpe realizado</strong>. En la práctica, esto resulta equivalente
    a actualizar la composición del portafolio cada 3 meses.
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
    <div class='workflow-step'><strong>Final:</strong> calculamos Sharpe, retorno y drawdown sobre los 24
    retornos mensuales recolectados. Eso es lo que reportamos.</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    El gráfico de abajo compara, en cada trimestre, el **Sharpe dentro de muestra** (naranja, lo que el optimizador
    prometió) contra el **Sharpe realizado fuera de muestra** (azul, lo que realmente ocurrió en los 3 meses siguientes).

    La brecha entre las dos barras es el **costo del error de estimación**.
    """)
    if (STORY_DIR / "bias_variance.png").exists():
        st.image(str(STORY_DIR / "bias_variance.png"), use_container_width=True)

    # ── Resultados ──
    st.markdown("## Los Resultados")
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

    # ── El Veredicto ──
    st.markdown("## El Veredicto")
    st.markdown("""
    - **¿Le ganó al mercado por retorno total?** Sí — 57% vs 40% del SPY.
    - **¿Por Sharpe (retorno ajustado por riesgo)?** Estrategia 1.74 vs SPY 1.18.
    - **Máximo drawdown:** estrategia −8% vs SPY −8%.

    **Advertencias importantes que podrían cambiar la conclusión:**

    - **Sin costos de transacción.** Rebalancear 50 acciones globales cada trimestre quitaría 0.5%–1% anual. Esta es tal vez una de las mayores debilidades.
    - **Ventana corta**, con solo ~2 años de prueba fuera de muestra.
    - **Sesgo de supervivencia** en el universo. Usamos los componentes actuales de los índices (empresas que sobrevivieron).
    - **Sin efecto de realización de ganancias** en las ventas intermedias.
    - **El periodo OOS fue inusualmente favorable para SPY** (Sharpe 1.18 vs el histórico ~0.5). Es decir, la estrategia le ganó en un periodo donde SPY estuvo especialmente fuerte.
    - **Fuerte efecto de volatilidad de acciones individuales.**

    La estrategia es **plausiblemente viable**. Un Sharpe realizado de 1.74 está cómodamente por encima de 1,
    y la diversificación geográfica protege contra riesgos regionales. Pero la brecha entre dentro de muestra (4-6)
    y fuera de muestra (1.74) muestra que **la mayor parte de lo que el optimizador "descubre" en datos históricos
    es ruido**, y solo una fracción de los beneficios esperados se materializa al analizarlo en un escenario real.
    """)

    st.success("**Pruébalo tú mismo** en la pestaña **🛠 La Herramienta** — mueve las palancas y observa cómo cambia el portafolio.")



# =================================================================
# PESTAÑA HERRAMIENTA
# =================================================================
with tool_tab:
    st.title("Construye un Portafolio")
    st.caption("Configura las palancas, presiona Optimizar. Pasa el cursor sobre cualquier métrica para ver una definición.")

    RETURNS = cached_returns()
    investable = drop_benchmarks(RETURNS)
    all_regions = sorted({ticker_region(t) for t in investable.columns})
    all_tickers = sorted(investable.columns.tolist())

    # Percentiles de volatilidad (anualizada) sobre el universo, para los botones de tolerancia
    asset_vols = (investable.std() * np.sqrt(12)).dropna()
    VOL_P25 = float(asset_vols.quantile(0.25))
    VOL_P50 = float(asset_vols.quantile(0.50))
    VOL_P75 = float(asset_vols.quantile(0.75))
    VOL_MAX = float(asset_vols.max())

    # El slider trabaja en puntos porcentuales enteros (5, 50, 100) para que el formato sea claro;
    # convertimos a decimal cuando lo usamos.
    if "max_vol_pct" not in st.session_state:
        st.session_state.max_vol_pct = int(round(VOL_P75 * 100))

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

        st.markdown("**Tolerancia al riesgo**")
        st.caption("Excluye los activos cuya volatilidad anual histórica supere este umbral, antes de hacer clustering.")
        b1, b2, b3 = st.columns(3)
        if b1.button("Baja", use_container_width=True, help=f"Solo activos con volatilidad ≤ {VOL_P25:.0%} (cuartil inferior)"):
            st.session_state.max_vol_pct = int(round(VOL_P25 * 100))
        if b2.button("Media", use_container_width=True, help=f"Solo activos con volatilidad ≤ {VOL_P50:.0%} (mediana)"):
            st.session_state.max_vol_pct = int(round(VOL_P50 * 100))
        if b3.button("Alta", use_container_width=True, help=f"Solo activos con volatilidad ≤ {VOL_P75:.0%} (cuartil superior)"):
            st.session_state.max_vol_pct = int(round(VOL_P75 * 100))
        max_vol_pct = st.slider("Volatilidad anual máxima permitida",
                                min_value=5, max_value=int(round(VOL_MAX * 100)), step=1,
                                format="%d%%", key="max_vol_pct",
                                help="Filtro aplicado antes del clustering. Los botones de arriba mueven este slider al percentil 25/50/75 del universo.")
        max_vol = max_vol_pct / 100.0
        pinned = st.multiselect(
            "Activos fijos (incluir siempre)",
            options=all_tickers,
            default=[],
            help="Escribe para buscar entre los ~2,000 activos del universo. Cada activo elegido reemplaza al representante automático de su cluster — la diversificación entre clusters se mantiene.")
        excluded = st.multiselect(
            "Activos excluidos",
            options=all_tickers,
            default=[],
            help="Escribe para buscar y excluir activos del universo. Útil para descartar empresas específicas antes del clustering.")
        regions_sel = st.multiselect("Regiones", all_regions, default=all_regions,
                                     help="Restringe el universo a las regiones elegidas. Vacío = ninguna.")
        run_backtest = st.checkbox("Ejecutar backtest fuera de muestra", True,
                                   help="Si está activado, corre también el backtest walk-forward. Agrega ~30 segundos pero produce el Sharpe realizado honesto.")
        go = st.button("🚀 Optimizar", use_container_width=True, type="primary")

    if "result" not in st.session_state:
        st.session_state.result = None
        st.session_state.bt = None
        st.session_state.bh = None

    if go:
        with st.spinner("Optimizando portafolio... (puede tomar 20-40 segundos con backtest)"):
            try:
                regions = regions_sel if len(regions_sel) < len(all_regions) else None
                common_kwargs = dict(
                    n_clusters=n_clusters,
                    min_weight=min_w, max_weight=max_w,
                    pinned=pinned, excluded=excluded,
                    min_sharpe=min_sharpe if min_sharpe > -0.99 else None,
                    regions=regions,
                    max_volatility=max_vol if max_vol < VOL_MAX else None,
                )
                st.session_state.result = optimize_portfolio(RETURNS, **common_kwargs)
                st.session_state.bt = None
                st.session_state.bh = None
                if run_backtest:
                    st.session_state.bt = backtest_strategy(RETURNS, **common_kwargs)
                    # Buy-and-hold: entrenar con primeros 3 años, holdear los siguientes 2
                    st.session_state.bh = buy_and_hold_backtest(
                        RETURNS, train_months=36, hold_months=24, **common_kwargs
                    )
            except Exception as e:
                st.error(f"Falló la optimización: {e}")

    result = st.session_state.result
    bt = st.session_state.bt
    bh = st.session_state.get("bh")

    if result is None:
        st.info("Configura las palancas en la barra lateral y presiona **Optimizar** para construir un portafolio.")
    else:
        weights_all = result["weights"].sort_values(ascending=False)
        weights = weights_all[weights_all > 0.001]
        perf = result["performance"]
        metrics_all = asset_metrics(result["selected_returns"], weights_all)
        metrics = metrics_all.loc[weights.index]

        # KPIs
        spy = get_benchmark(RETURNS, "SPY")
        bt_stats = perf_stats(bt) if bt is not None and len(bt) else None
        spy_stats = perf_stats(spy.loc[bt.index].dropna()) if bt is not None and len(bt) else None

        st.markdown("### Métricas principales")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Activos", len(weights), help="Número de activos con peso superior al 0.1%.")

        # Diferencias numéricas vs SPY (con signo) para que Streamlit colore correctamente.
        def _pp(diff):  # formatea una diferencia de puntos porcentuales
            return f"{diff*100:+.2f}pp vs SPY"

        c2.metric("Retorno anual (OOS)",
                  f"{bt_stats['annual_return']:.2%}" if bt_stats else "Ejecuta backtest →",
                  delta=_pp(bt_stats['annual_return'] - spy_stats['annual_return']) if spy_stats else None,
                  help=f"Retorno anualizado en el walk-forward OOS. SPY: {spy_stats['annual_return']:.2%}." if spy_stats else "Retorno anualizado.")
        c3.metric("Volatilidad (OOS)",
                  f"{bt_stats['annual_vol']:.2%}" if bt_stats else "—",
                  delta=_pp(bt_stats['annual_vol'] - spy_stats['annual_vol']) if spy_stats else None,
                  delta_color="inverse",
                  help=f"Desviación estándar anualizada. Menor = camino más suave. SPY: {spy_stats['annual_vol']:.2%}." if spy_stats else "Desviación estándar anualizada.")
        c4.metric("Sharpe realizado (OOS)",
                  f"{bt_stats['sharpe']:.2f}" if bt_stats else "—",
                  delta=f"{bt_stats['sharpe'] - spy_stats['sharpe']:+.2f} vs SPY" if spy_stats else None,
                  help=f"Retorno ajustado por riesgo honesto. SPY: {spy_stats['sharpe']:.2f}." if spy_stats else "Sharpe.")
        c5.metric("Máximo drawdown (OOS)",
                  f"{bt_stats['max_drawdown']:.0%}" if bt_stats else "—",
                  delta=_pp(bt_stats['max_drawdown'] - spy_stats['max_drawdown']) if spy_stats else None,
                  help=f"Peor pérdida desde un pico anterior. SPY: {spy_stats['max_drawdown']:.0%}." if spy_stats else "Máximo drawdown.")

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

        # Tabla por activo (incluye los descartados con peso 0)
        st.markdown("### Detalle por activo")
        st.caption("Una fila por activo seleccionado por el clustering. "
                   "Incluye los que el optimizador descartó (peso 0) para que veas qué candidatos había. "
                   "Haz click en los encabezados para ordenar.")
        view = metrics_all.copy()
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

        # Comparación 3yr train + 2yr hold vs walk-forward vs SPY
        if bh is not None and len(bh) and bt is not None and len(bt) and "SPY" in RETURNS.columns:
            st.markdown("### Comparación de estrategias (2 años de holdeo)")
            st.caption(
                "Tres formas distintas de invertir, todas medidas sobre el mismo período de prueba "
                "(2 años fuera de muestra): "
                "**(1) Buy-and-hold de la estrategia** — entrenamos con 3 años, fijamos los pesos y los mantenemos 2 años sin tocar. "
                "**(2) Walk-forward de la estrategia** — rebalanceamos los pesos cada trimestre con datos nuevos. "
                "**(3) SPY buy-and-hold** — el benchmark."
            )
            spy_hold = RETURNS["SPY"].loc[bh.index].dropna()

            bh_stats = perf_stats(bh)
            bt_hold_stats = perf_stats(bt.loc[bh.index]) if set(bh.index).issubset(set(bt.index)) else perf_stats(bt)
            spy_hold_stats = perf_stats(spy_hold)

            comp = pd.DataFrame({
                "Sharpe realizado": [bh_stats["sharpe"], bt_hold_stats["sharpe"], spy_hold_stats["sharpe"]],
                "Desvío estándar (anual)": [bh_stats["annual_vol"], bt_hold_stats["annual_vol"], spy_hold_stats["annual_vol"]],
                "Retorno acumulado": [bh_stats["total_return"], bt_hold_stats["total_return"], spy_hold_stats["total_return"]],
            }, index=["Estrategia (buy-and-hold 3yr→2yr)", "Estrategia (walk-forward)", "SPY (buy-and-hold)"])

            comp_view = comp.copy()
            comp_view["Sharpe realizado"] = comp_view["Sharpe realizado"].round(2)
            comp_view["Desvío estándar (anual)"] = (comp_view["Desvío estándar (anual)"] * 100).round(2).astype(str) + "%"
            comp_view["Retorno acumulado"] = (comp_view["Retorno acumulado"] * 100).round(2).astype(str) + "%"
            st.dataframe(comp_view, use_container_width=True)

            # Curva comparativa
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.plot(equity_curve(bh).index, equity_curve(bh).values,
                    label="Estrategia buy-and-hold (3yr→2yr)", linewidth=2, color="#2c4a7a")
            ax.plot(equity_curve(bt).index, equity_curve(bt).values,
                    label="Estrategia walk-forward", linewidth=2, linestyle="--", color="#3a7a3a")
            ax.plot(equity_curve(spy_hold).index, equity_curve(spy_hold).values,
                    label="SPY buy-and-hold", linewidth=2, color="#a55c3f", alpha=0.85)
            ax.set_ylabel("Valor acumulado de $1")
            ax.legend(); ax.grid(alpha=0.3)
            st.pyplot(fig, use_container_width=True)

        # Ventaja porcentual sobre SPY
        if bt is not None and len(bt) and "SPY" in RETURNS.columns:
            st.markdown("### Ventaja porcentual sobre SPY")
            st.caption(
                "Cuánto más (o menos) vale tu portafolio respecto a haber comprado SPY con el mismo dinero. "
                "Si la línea está en **+15%**, significa que en ese momento tu portafolio vale 15% más "
                "que si hubieras invertido todo en SPY desde el inicio. Si está en −5%, vale 5% menos."
            )
            spy_oos = RETURNS["SPY"].loc[bt.index].dropna()
            strat_eq = (1 + bt).cumprod()
            spy_eq   = (1 + spy_oos).cumprod()
            advantage = (strat_eq / spy_eq - 1) * 100  # en puntos porcentuales

            fig, ax = plt.subplots(figsize=(11, 4))
            ax.fill_between(advantage.index, advantage.values, 0,
                            where=(advantage.values >= 0), color="#3a7a3a", alpha=0.5, label="Estrategia adelante")
            ax.fill_between(advantage.index, advantage.values, 0,
                            where=(advantage.values < 0), color="#a55c3f", alpha=0.5, label="SPY adelante")
            ax.plot(advantage.index, advantage.values, color="#2c4a7a", linewidth=1.5)
            ax.axhline(0, color="grey", linewidth=0.7)
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(decimals=0))
            ax.set_ylabel("Ventaja sobre SPY")
            ax.legend(loc="best")
            ax.grid(alpha=0.3)

            # Anotar el valor final
            last_val = advantage.iloc[-1]
            ax.annotate(f"{last_val:+.1f}%",
                        xy=(advantage.index[-1], last_val),
                        xytext=(10, 0), textcoords="offset points",
                        fontsize=11, fontweight="bold",
                        color="#3a7a3a" if last_val >= 0 else "#a55c3f")
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
