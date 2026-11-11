# Toy MDLM: difusión discreta para texto

Un modelo de lenguaje de **difusión enmascarada** (Masked Diffusion Language Model),
basado en las ideas de LLaDA y *discrete flow matching*. Está pensado para una
demo en vivo: entrena en CPU en aproximadamente 1 minuto y muestra en la terminal
cómo una frase va apareciendo **en cualquier orden**, no de izquierda a derecha
como en GPT.

Todo está en un solo archivo: `toy_mdlm.py`.

```bash
python toy_mdlm.py                     # entrena y genera (aprox. 1 min)
python toy_mdlm.py --seed 1 --pausa 0.3
```

Requisito: Python 3 + PyTorch (solo CPU).

---

## 1. La idea en 30 segundos

| | Modelo autoregresivo (GPT) | Modelo de difusión enmascarada (este) |
|---|---|---|
| Cómo genera | Una letra a la vez, de izquierda a derecha | Parte de todo tapado (`█████`) y va destapando letras en cualquier orden |
| Qué ve | Solo lo que está a la izquierda (atención causal) | Toda la frase, a ambos lados (atención bidireccional) |
| Qué aprende | "Dada la parte anterior, ¿qué letra sigue?" | "Dada una frase con huecos, ¿qué letra va en cada hueco?" |

**El "ruido"** en esta difusión discreta consiste en tapar letras con un token
especial `[MASK]`, no en sumar ruido gaussiano como en la difusión de imágenes.

**Convención de tiempo** (la misma de flow matching):

```
t = 0.00  |█████████████████████████|   <- todo tapado (ruido puro)
t = 0.50  |el ga█o███me█pa█····██···|
t = 1.00  |el gato come pan·········|   <- frase limpia (datos)
```

---

## 2. Cómo funciona

### Entrenamiento (proceso "forward": ensuciar)
Por cada frase del batch:
1. Se samplea un tiempo `t ~ Uniform(0, 1)`.
2. Cada letra se tapa con `[MASK]` de forma independiente con probabilidad
   `1 − t`. Este es el scheduler lineal `κ_t = t`.
3. El modelo recibe la frase tapada y predice, para **cada** posición, una
   distribución de probabilidad sobre el vocabulario.
4. La loss es la **cross-entropy calculada solo sobre las posiciones tapadas**:
   las letras visibles no cuentan, porque ya se conocen.

### Generación (proceso "reverse": limpiar), paso de Euler
1. Se arranca con una secuencia de largo fijo, toda en `[MASK]`, en `t = 0`.
2. Se hacen `n` pasos de tamaño `h = 1/n`. En cada paso:
   - el modelo predice una distribución para cada posición;
   - se samplea una letra candidata por posición;
   - cada posición **todavía tapada** se destapa con probabilidad `h / (1 − t)`.
3. Una letra revelada **nunca vuelve a taparse**.
4. En el último paso `t = 1 − h`, así que `h/(1−t) = 1` y todo queda revelado.

### Infilling (completar huecos)
Es el mismo proceso de generación, pero partiendo de una frase donde solo
algunas posiciones están tapadas. Como el algoritmo solo modifica posiciones
`[MASK]`, lo demás queda intacto. El modelo usa el contexto de **ambos lados**:
en `el perro ____ agua` completa "bebe" porque ve "agua" más adelante.

---

## 3. Estructura del código

Cada bloque tiene comentarios `#?` explicando qué hace. Las líneas indicadas
corresponden a la versión actual del archivo.

| Sección | Líneas aprox. | Qué contiene |
|---|---|---|
| **(0) Dataset** | 40–68 | `SUJETOS`, `VERBOS_OBJETOS` y `build_dataset()`: generan 102 frases "sujeto verbo objeto" |
| **(1) Tokenizador** | 70–117 | `CharTokenizer`: texto ↔ ids por carácter. Tokens especiales `[PAD]` (`·`) y `[MASK]` (`█`) |
| **(2) Modelo** | 119–161 | `MaskedDiffusionTransformer`: embeddings + 3 capas Transformer **sin máscara causal** + capa de salida |
| **(3) Enmascarado** | 163–186 | `enmascarar()`: samplea `t` y tapa cada letra con prob. `1 − t` |
| **(4) Entrenamiento** | 188–233 | `entrenar()`: loop con Adam y cross-entropy solo en posiciones tapadas |
| **(5) Sampling** | 236–331 | `mostrar()` (colores), `samplear()` (paso de Euler) e `infill()` (completar huecos) |
| **(6) Main** | 333–404 | Opciones de terminal, demo del ruido, entrenamiento, generaciones e infilling |

### Salida de la demo, en orden
1. Info del dataset y 3 frases de ejemplo.
2. **Proceso forward**: una frase tapada a distintos niveles de `t`.
3. **Entrenamiento**: el loss cada 250 pasos. Debe bajar de ~1.6 a ~0.4.
4. **Generaciones**: frases desde todo `[MASK]`, paso a paso. Lo recién revelado
   aparece en **verde**, y al final se indica si la frase existía en el dataset
   o si el modelo inventó una combinación nueva.
5. **Infilling** de 3 plantillas.

---

## 4. Qué se puede variar

### 4.1 Desde la terminal (sin tocar el código)
Definidas en `main()`, líneas 342–348.

| Opción | Default | Efecto |
|---|---|---|
| `--steps` | `2000` | Pasos de entrenamiento. Menos es más rápido pero peor (con 250 sale basura tipo "el getg quiere joo") |
| `--sample-steps` | `20` | Número `n` de pasos de sampling. Más pasos hacen el revelado más gradual |
| `--n-samples` | `3` | Cuántas frases generar desde cero |
| `--seed` | `0` | Semilla aleatoria. La misma semilla da siempre el mismo resultado |
| `--pausa` | `0.15` | Segundos de espera entre pasos al imprimir (efecto "en vivo") |
| `--no-color` | apagado | Desactiva los colores ANSI |

Ejemplo: `python toy_mdlm.py --steps 1000 --sample-steps 30 --pausa 0.3 --seed 2`

### 4.2 En el código

#### Dataset: sección (0)
- **Líneas 49–55** (`SUJETOS`, `VERBOS_OBJETOS`): agrega sujetos, verbos u objetos.
  El número de frases se ajusta solo.
- **Línea 60** (`build_dataset()`): para usar tu propia lista, haz que retorne
  `["frase uno", "frase dos", ...]`. El vocabulario y la longitud máxima se
  calculan automáticamente a partir de las frases.
- Frases más largas o más variadas necesitan más `--steps`.

#### Tokenizador: sección (1)
- **Línea 80** (`PAD_SHOW, MASK_SHOW`): los símbolos con que se dibujan `[PAD]`
  y `[MASK]` en pantalla (por ejemplo `"_"` y `"?"` si tu terminal no muestra `█`).

#### Arquitectura del modelo: sección (2)
- **Líneas 133–134**, argumentos de `MaskedDiffusionTransformer.__init__`:

  | Parámetro | Default | Qué es |
  |---|---|---|
  | `d_model` | `128` | Tamaño de los vectores internos. Debe ser divisible por `n_heads` |
  | `n_heads` | `4` | Cabezas de atención |
  | `n_layers` | `3` | Capas Transformer. Menos capas entrenan más rápido |
  | `d_ff` | `256` | Tamaño de la capa feed-forward interna |
  | `dropout` | `0.0` | Regularización (no hace falta en un dataset tan chico) |

  Un modelo más chico (`d_model=64, n_layers=2`) entrena bastante más rápido y
  sigue funcionando con este dataset.
- **Línea 158**: el truco que impide que el modelo prediga `[MASK]`. No lo quites.

#### Enmascarado / scheduler: sección (3)
- **Línea 181**: `torch.rand(B, L) < (1 - t)`. Aquí está el scheduler lineal
  `κ_t = t`. Para probar otro scheduler, cambia `(1 - t)` por, por ejemplo,
  `(1 - t**2)`. En ese caso también debe cambiar la probabilidad de revelado
  en el sampling (ver 4.2, sección 5), para que ambos procesos sean coherentes.

#### Entrenamiento: sección (4)
- **Línea 194**, argumentos de `entrenar()`:

  | Parámetro | Default | Qué es |
  |---|---|---|
  | `pasos` | `2000` | Se sobrescribe con `--steps` |
  | `batch_size` | `64` | Frases por paso |
  | `lr` | `1e-3` | Learning rate de Adam |
  | `log_cada` | `250` | Cada cuántos pasos se imprime el loss |

- **Línea 219**: la loss. Usa la media simple. El MDLM "exacto" (el ELBO)
  pondera cada ejemplo por `1/(1 − t)`. Para probarlo, usa el `t` que devuelve
  `enmascarar()` (hoy se descarta con `_`) y cambia la loss por:
  ```python
  xt, mask, t = enmascarar(x1, tok.mask_id)
  ...
  ce = F.cross_entropy(logits[mask], x1[mask], reduction="none")
  peso = (1 / (1 - t).clamp(min=1e-3)).expand_as(mask)[mask]
  loss = (ce * peso).mean()
  ```

#### Sampling: sección (5)
- **Línea 240** (`VERDE, GRIS`): códigos de color. Por ejemplo `"\033[1;33m"`
  es amarillo.
- **Línea 261**, argumentos de `samplear()`:
  - `temperatura` (default `1.0`) no está expuesta en la terminal. Para usarla,
    pásala en las llamadas de `main()` (líneas 387 y 397), por ejemplo
    `samplear(..., temperatura=0.7)`. Menos de 1 da frases más "seguras" y
    repetidas del dataset; más de 1 da frases más creativas pero con más errores.
- **Línea 302**: `p_revelar = h / (1 - t)`, la regla de Euler para el scheduler
  lineal. Si cambias el scheduler en la línea 181, esta fórmula general es
  `h · κ'(t) / (1 − κ(t))`.
- **Línea 317** (`infill()`): el carácter `hueco="_"` marca los huecos. Cada
  `_` corresponde exactamente a **una** letra.

#### Main: sección (6)
- **Línea 359**: `seq_len = max(len) + 2`, la longitud fija de las
  secuencias (frase más larga + 2 de margen).
- **Línea 370**: los niveles de `t` que se muestran en la demo del ruido.
- **Línea 395**: **las plantillas de infilling**. Esta es la parte más fácil de
  personalizar para la expo. Reglas:
  - cada `_` es un solo carácter, así que pon tantos `_` como letras esperas;
  - si pones `_` de más, el modelo rellena el sobrante con `[PAD]` (`·`);
  - solo sirven caracteres que existan en el dataset. Una mayúscula, una "z" o
    un signo que no aparezca en ninguna frase produce `KeyError`.

---

## 5. Limitaciones (es un *toy model*)
- El dataset es pequeño y artificial, así que el modelo mezcla combinaciones
  vistas en vez de "entender" español. A veces sale algo como "el gato mira
  la mar", que no está en el dataset: es una combinación nueva y gramaticalmente
  imperfecta, útil para mostrar que el modelo generaliza.
- La longitud de la secuencia es fija. El modelo decide dónde termina la frase
  prediciendo `[PAD]`.
- El modelo no recibe `t` como entrada (igual que en LLaDA): la cantidad de
  `[MASK]` en la secuencia ya le indica en qué punto del proceso está.
- Se re-entrena cada vez que se ejecuta el script; no guarda el modelo.

## Referencias
- Nie et al., *Large Language Diffusion Models* (LLaDA), 2025.
- Sahoo et al., *Simple and Effective Masked Diffusion Language Models* (MDLM), 2024.
- Gat et al., *Discrete Flow Matching*, 2024.
