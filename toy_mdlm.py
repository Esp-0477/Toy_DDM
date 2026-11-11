"""
toy_mdlm.py — Toy Masked Diffusion Language Model (MDLM / discrete diffusion)
============================================================================

Demo didáctica inspirada en LLaDA / discrete flow matching:

  * Transformer encoder BIDIRECCIONAL (sin máscara causal).
  * Tokenización a nivel de carácter + token especial [MASK].
  * Entrenamiento: scheduler lineal kappa_t = t. Con t ~ U(0,1), cada token
    se enmascara independientemente con probabilidad (1 - t). La loss es
    cross-entropy SOLO en las posiciones enmascaradas.
  * Sampling: Euler para una CTMC factorizada. Se empieza con todo [MASK] en
    t = 0 y se avanza hasta t = 1 con paso h = 1/n. En cada paso, cada
    posición enmascarada se revela con probabilidad h / (1 - t), usando un
    sample de la distribución que predice el modelo. Lo revelado no vuelve
    a enmascararse.

Convención de tiempo (la de flow matching):
    t = 0  ->  secuencia totalmente enmascarada (ruido)
    t = 1  ->  secuencia limpia (datos)

Uso:
    python toy_mdlm.py                 # entrena (~1 min en CPU) y genera
    python toy_mdlm.py --steps 1500    # entrenamiento más corto
    python toy_mdlm.py --no-color      # sin colores ANSI en la terminal
"""

#? Librerías: argparse (opciones por terminal), random/time (azar y tiempos)
#? y PyTorch (tensores, capas de red neuronal y funciones como softmax).
import argparse
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# (0) Dataset de ejemplo: frases cortas en español generadas por plantilla
# ============================================================================
# Se generan frases del tipo  "<sujeto> <verbo> <objeto>"  con concordancia
# simple. Así el modelo tiene estructura que aprender (qué verbo va con qué
# sujeto, qué objeto tiene sentido) pero el dataset sigue siendo chico.
# Si quieres usar tu propia lista, reemplaza build_dataset() por tu lista.

#? Los "ingredientes" de las frases: 6 sujetos y, para cada verbo, la lista
#? de objetos que tienen sentido con él (no se "come agua", se "bebe agua").
SUJETOS = ["el gato", "el perro", "la vaca", "el niño", "la niña", "mi abuela"]
VERBOS_OBJETOS = {
    "come": ["pan", "queso", "fruta", "sopa", "arroz"],
    "bebe": ["agua", "leche", "jugo", "café"],
    "mira": ["el mar", "la luna", "el sol", "la casa"],
    "quiere": ["jugar", "dormir", "correr", "cantar"],
}


#? Combina todos los sujetos con todos los pares (verbo, objeto) válidos.
#? Resultado es una lista de 102 strings como "el gato come pan".
def build_dataset():
    frases = []
    for s in SUJETOS:
        for v, objetos in VERBOS_OBJETOS.items():
            for o in objetos:
                frases.append(f"{s} {v} {o}")
    return frases  # 6 sujetos * 17 combinaciones = 102 frases


# ============================================================================
# (1) Tokenizador de caracteres
# ============================================================================
# Vocabulario = caracteres del dataset + dos tokens especiales:
#   [PAD]  : relleno hasta la longitud fija. Se trata como un token normal
#            (el modelo aprende a predecirlo), igual que el EOS en LLaDA.
#            Así el modelo "decide" cuán larga es la frase.
#   [MASK] : el estado absorbente del proceso de difusión.

#? Nombres de los tokens especiales y el símbolo con que se dibujan en pantalla.
PAD, MASK = "[PAD]", "[MASK]"
PAD_SHOW, MASK_SHOW = "·", "█"  # cómo se ven al imprimir


#? El tokenizador traduce texto <-> números. La red neuronal solo entiende
#? números, así que cada carácter recibe un id (por ejemplo 'a' -> 3).
class CharTokenizer:
    #? Construye el vocabulario: [PAD] + cada carácter distinto del dataset
    #? + [MASK]. 'itos' = id -> carácter, 'stoi' = carácter -> id.
    #? Todas las secuencias tendrán la misma longitud fija 'seq_len'.
    def __init__(self, textos, seq_len):
        chars = sorted(set("".join(textos)))
        self.itos = [PAD] + chars + [MASK]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.mask_id = self.stoi[MASK]
        self.vocab_size = len(self.itos)
        self.seq_len = seq_len

    #? Texto -> tensor de ids, rellenando con [PAD] hasta 'seq_len'.
    def encode(self, texto):
        ids = [self.stoi[c] for c in texto][: self.seq_len]
        ids += [self.pad_id] * (self.seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    #? Tensor de ids -> texto. Con mostrar_especiales=True dibuja [MASK] como
    #? '█' y [PAD] como '·'; con False los omite (texto "limpio" final).
    def decode(self, ids, mostrar_especiales=True):
        out = []
        for i in ids.tolist():
            if i == self.mask_id:
                out.append(MASK_SHOW if mostrar_especiales else "")
            elif i == self.pad_id:
                out.append(PAD_SHOW if mostrar_especiales else "")
            else:
                out.append(self.itos[i])
        return "".join(out)


# ============================================================================
# (2) Definición del modelo: Transformer encoder bidireccional
# ============================================================================
# Nota: igual que en LLaDA, el modelo NO recibe t como entrada. La cantidad
# de [MASK] en la secuencia ya le "dice" en qué punto del proceso está.


#? La red neuronal. Recibe una frase con huecos ([MASK]) y, para CADA
#? posición, devuelve una probabilidad de qué carácter va ahí.
class MaskedDiffusionTransformer(nn.Module):
    #? Piezas del modelo:
    #?  - tok_emb: convierte cada id en un vector de 128 números.
    #?  - pos_emb: vector aprendido por posición (para saber "dónde" está cada letra).
    #?  - encoder: 3 capas Transformer; cada letra "mira" a todas las demás.
    #?  - head: convierte cada vector final en un puntaje por cada token del vocabulario.
    def __init__(self, vocab_size, seq_len, mask_id, d_model=128, n_heads=4,
                 n_layers=3, d_ff=256, dropout=0.0):
        super().__init__()
        self.mask_id = mask_id
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        capa = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        # Sin src_mask -> cada posición atiende a TODAS las demás (bidireccional)
        self.encoder = nn.TransformerEncoder(capa, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    #? Paso hacia adelante: ids -> vectores -> Transformer (bidireccional,
    #? ve izquierda Y derecha, a diferencia de GPT) -> logits (puntajes).
    #? Al final se prohíbe predecir [MASK] poniéndole puntaje -infinito.
    def forward(self, x):
        """x: (B, L) ids  ->  logits: (B, L, V)"""
        h = self.tok_emb(x) + self.pos_emb[:, : x.size(1)]
        h = self.encoder(h)  # ¡sin máscara causal!
        logits = self.head(self.norm(h))
        # El modelo nunca debe predecir [MASK] como token "limpio".
        logits[..., self.mask_id] = float("-inf")
        return logits


# ============================================================================
# (3) Preparación de datos y enmascaramiento (proceso "forward" / ruido)
# ============================================================================


#? El "ruido" de la difusión discreta: tapar letras al azar con [MASK].
#? Cada frase recibe su propio t: si t≈1 casi no se tapa nada, si t≈0 se
#? tapa casi todo. Así el modelo practica con todos los niveles de dificultad.
def enmascarar(x1, mask_id):
    """
    Dado un batch limpio x1 (B, L):
      - samplea t ~ U(0,1) por frase,
      - enmascara cada token independientemente con prob (1 - t)
        (scheduler lineal kappa_t = t: prob. de que un token ya esté "limpio").
    Devuelve x_t (batch ruidoso), la máscara booleana y t.
    """
    B, L = x1.shape
    #? Un t aleatorio por frase, y un "dado" por letra: si sale < (1-t), se tapa.
    t = torch.rand(B, 1)                           # (B, 1)
    esta_enmascarado = torch.rand(B, L) < (1 - t)  # (B, L)
    #? Donde toca tapar se pone [MASK]; en el resto se deja la letra original.
    xt = torch.where(esta_enmascarado, torch.full_like(x1, mask_id), x1)
    return xt, esta_enmascarado, t


# ============================================================================
# (4) Loop de entrenamiento
# ============================================================================


#? Entrenamiento: repetir muchas veces "tapar letras -> que el modelo adivine
#? -> corregirlo". El modelo solo se evalúa en las letras que estaban tapadas.
def entrenar(model, datos, tok, pasos=2000, batch_size=64, lr=1e-3, log_cada=250):
    """
    datos: tensor (N, L) con todas las frases tokenizadas.
    Loss = cross-entropy sobre las posiciones enmascaradas únicamente.
    (El MDLM "exacto" pondera por 1/(1-t) para obtener el ELBO; aquí usamos
     la media simple, que es más estable y suficiente para la demo.)
    """
    #? Adam es el optimizador: el que ajusta los pesos de la red para reducir el error.
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    acumulado = 0.0
    for paso in range(1, pasos + 1):
        #? 1) Tomar 64 frases al azar del dataset y taparles letras.
        idx = torch.randint(0, len(datos), (batch_size,))
        x1 = datos[idx]                                   # datos limpios
        xt, mask, _ = enmascarar(x1, tok.mask_id)         # datos ruidosos

        #? 2) El modelo intenta adivinar qué va en cada posición.
        logits = model(xt)                                # (B, L, V)
        # Solo posiciones enmascaradas: logits[mask] -> (M, V), x1[mask] -> (M,)
        if mask.sum() == 0:
            continue
        #? 3) Error (cross-entropy) solo en las posiciones tapadas: ¿qué tan
        #?    alta probabilidad le dio a la letra correcta?
        loss = F.cross_entropy(logits[mask], x1[mask])

        #? 4) Backpropagation: calcular cómo cambiar cada peso y aplicarlo.
        opt.zero_grad()
        loss.backward()
        opt.step()

        #? Cada 'log_cada' pasos imprime el loss promedio (debe ir bajando).
        acumulado += loss.item()
        if paso % log_cada == 0:
            print(f"  paso {paso:5d} | loss {acumulado / log_cada:.4f} "
                  f"| {time.time() - t0:5.1f}s")
            acumulado = 0.0
    model.eval()


# ============================================================================
# (5) Sampling: Euler para CTMC factorizada (con logs de cada paso)
# ============================================================================

#? Códigos de color ANSI de la terminal: verde = recién revelado, gris = [MASK]/[PAD].
VERDE, GRIS, RESET = "\033[1;32m", "\033[90m", "\033[0m"


#? Solo visual: arma el string de una secuencia pintando en verde las letras
#? que se revelaron en este paso y en gris las máscaras y el relleno.
def mostrar(x, nuevos, tok, color=True):
    """Imprime la secuencia; resalta en verde lo revelado en este paso."""
    s = []
    for tid, nuevo in zip(x.tolist(), nuevos.tolist()):
        c = tok.decode(torch.tensor([tid]))
        if color and nuevo:
            c = f"{VERDE}{c}{RESET}"
        elif color and tid in (tok.mask_id, tok.pad_id):
            c = f"{GRIS}{c}{RESET}"
        s.append(c)
    return "".join(s)


#? La generación ("proceso inverso"): se parte de todo tapado y se van
#? destapando letras poco a poco, en cualquier orden, hasta completar la frase.
@torch.no_grad()
def samplear(model, tok, n_pasos=20, x_inicial=None, temperatura=1.0,
             verbose=True, color=True, pausa=0.0):
    """
    Genera UNA secuencia.

    x_inicial: tensor (L,) opcional. Las posiciones que NO son [MASK] se
               consideran fijas (infilling). Si es None, se parte de todo [MASK].

    En cada paso k (t = k*h, h = 1/n_pasos):
      1. El modelo predice p(x1 | x_t) para todas las posiciones.
      2. Para cada posición enmascarada: con prob h/(1-t) se revela,
         tomando un sample categórico de esa distribución.
    En el último paso t = 1 - h  ->  h/(1-t) = 1, así que todo queda revelado.
    """
    #? Estado inicial: todo [MASK] (o la plantilla con huecos, si es infilling).
    L = tok.seq_len
    x = (x_inicial.clone() if x_inicial is not None
         else torch.full((L,), tok.mask_id, dtype=torch.long))
    h = 1.0 / n_pasos

    if verbose:
        print(f"  t=0.00  |{mostrar(x, torch.zeros(L, dtype=torch.bool), tok, color)}|")

    for k in range(n_pasos):
        t = k * h
        #? ¿Qué posiciones siguen tapadas? Si ninguna, ya terminamos.
        enmascarados = x == tok.mask_id
        if not enmascarados.any():
            break

        #? El modelo mira la frase actual (con huecos) y da, para cada
        #? posición, probabilidades sobre las letras. Se "tira un dado"
        #? con esas probabilidades para elegir una letra candidata por posición.
        #? (temperatura < 1 = más conservador, > 1 = más creativo/aleatorio)
        logits = model(x.unsqueeze(0))[0] / temperatura       # (L, V)
        probs = F.softmax(logits, dim=-1)
        x1_pred = torch.multinomial(probs, 1).squeeze(-1)     # sample por posición

        #? Paso de Euler: cada posición tapada se destapa con prob h/(1-t).
        #? Al principio se destapan pocas; al final la prob llega a 1.
        #? Las letras ya reveladas nunca cambian.
        p_revelar = min(1.0, h / (1.0 - t))
        revelar = enmascarados & (torch.rand(L) < p_revelar)
        x = torch.where(revelar, x1_pred, x)                  # lo revelado queda fijo

        #? Imprime el estado en este paso (y espera 'pausa' segundos para el efecto en vivo).
        if verbose:
            print(f"  t={t + h:.2f}  |{mostrar(x, revelar, tok, color)}|")
            if pausa:
                time.sleep(pausa)
    return x


#? Infilling = completar huecos. Se convierte la plantilla (con '_' en los
#? huecos) en una secuencia donde solo los '_' son [MASK], y se llama al
#? mismo samplear(): como solo se destapan máscaras, lo demás queda intacto.
def infill(model, tok, plantilla, hueco="_", **kw):
    """
    Infilling: 'plantilla' es un string donde cada carácter `hueco` es una
    posición a completar. El resto de caracteres queda FIJO.
    Las posiciones después del final de la plantilla se fijan como [PAD]
    (si quieres que el modelo también elija la longitud, termina la
    plantilla con varios `hueco`).
      ej: infill(model, tok, "el perro ____ agua")
    """
    ids = [tok.mask_id if c == hueco else tok.stoi[c] for c in plantilla]
    ids += [tok.pad_id] * (tok.seq_len - len(ids))
    x0 = torch.tensor(ids[: tok.seq_len], dtype=torch.long)
    return samplear(model, tok, x_inicial=x0, **kw)


# ============================================================================
# (6) Main: entrena y genera ejemplos
# ============================================================================


#? Orquesta toda la demo: lee opciones, prepara datos, muestra el ruido,
#? entrena, genera frases desde cero y finalmente hace infilling.
def main():
    #? Opciones que se pueden cambiar desde la terminal (ej: --steps 1000).
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000, help="pasos de entrenamiento")
    ap.add_argument("--sample-steps", type=int, default=20, help="pasos de sampling n")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pausa", type=float, default=0.15,
                    help="segundos entre pasos al imprimir (efecto en vivo)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    #? Semilla fija: con la misma --seed, el resultado sale igual cada vez.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    color = not args.no_color

    #? Construir dataset y tokenizador, y pasar todas las frases a números.
    # --- Datos -------------------------------------------------------------
    frases = build_dataset()
    seq_len = max(len(f) for f in frases) + 2
    tok = CharTokenizer(frases, seq_len)
    datos = torch.stack([tok.encode(f) for f in frases])
    print(f"Dataset: {len(frases)} frases | vocab: {tok.vocab_size} tokens | "
          f"longitud fija: {seq_len}")
    print("Ejemplos:", *random.sample(frases, 3), sep="\n   ")

    #? Muestra en pantalla cómo se ve una frase con distintos niveles de ruido.
    # Ilustración del proceso forward (ruido) para una frase
    print("\nProceso forward (enmascarar con prob 1-t):")
    x1 = datos[0]
    for t in [1.0, 0.75, 0.5, 0.25, 0.0]:
        m = torch.rand(seq_len) < (1 - t)
        print(f"  t={t:.2f}  |{tok.decode(torch.where(m, tok.mask_id, x1))}|")

    #? Crear el modelo (con pesos aleatorios) y entrenarlo.
    # --- Modelo ------------------------------------------------------------
    model = MaskedDiffusionTransformer(tok.vocab_size, seq_len, tok.mask_id)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModelo: {n_params / 1e3:.0f}k parámetros\n\nEntrenando...")

    entrenar(model, datos, tok, pasos=args.steps)

    #? Generar frases desde cero (todo [MASK]) y decir si la frase ya existía
    #? en el dataset o si el modelo inventó una combinación nueva.
    # --- Generación incondicional ------------------------------------------
    for i in range(args.n_samples):
        print(f"\nGeneración {i + 1} (desde todo [MASK], {args.sample_steps} pasos):")
        x = samplear(model, tok, n_pasos=args.sample_steps, color=color,
                     pausa=args.pausa)
        texto = tok.decode(x, mostrar_especiales=False)
        esta = "está" if texto in frases else "NO está"
        print(f"  -> \"{texto}\"  ({esta} en el dataset)")

    #? Completar huecos en frases predefinidas (cambia esta lista para la expo).
    # --- Infilling ----------------------------------------------------------
    for plantilla in ["el perro ____ agua", "la niña mira ________", "el ____ quiere ______"]:
        print(f"\nInfilling de \"{plantilla}\":")
        x = infill(model, tok, plantilla, n_pasos=args.sample_steps, color=color,
                   pausa=args.pausa)
        print(f"  -> \"{tok.decode(x, mostrar_especiales=False)}\"")


#? Punto de entrada: solo corre main() si ejecutas este archivo directamente.
if __name__ == "__main__":
    main()
