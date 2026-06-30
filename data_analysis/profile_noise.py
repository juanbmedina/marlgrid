"""
profile_noise.py
----------------
Utilidades para visualizar el efecto del ruido multiplicativo
(como se aplica en P2PEnergyEnv._sample_noisy_profiles) sobre los
perfiles D (demanda) y G (generación) de cada agente.

Modelo de ruido:

    tilde_X[i, h] = max(X[i, h] * (1 + sigma * xi[i, h]), 0)
    xi ~ N(0, 1)

Diseñado para usarse desde un notebook:

    from profile_noise import load_profiles, plot_noise
    profiles = load_profiles("profiles/agents_profiles_24h.json")
    fig, axes = plot_noise(profiles, sigmas=(0.1, 0.2, 0.3))
"""

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Carga de perfiles
# ---------------------------------------------------------------------------

def load_profiles(path: str) -> dict:
    """Carga un agents_profiles_*.json y lo devuelve como dict."""
    with open(path, "r") as f:
        return json.load(f)


def synthetic_profiles(n_hours: int = 24, seed: int = 0) -> dict:
    """Genera 6 prosumidores sintéticos (placeholder para test rápido)."""
    rng = np.random.default_rng(seed)
    h = np.arange(n_hours)

    def demand_curve(scale_m, scale_e, base):
        return (
            base
            + scale_m * np.exp(-0.5 * ((h - 7) / 1.8) ** 2)
            + scale_e * np.exp(-0.5 * ((h - 20) / 2.2) ** 2)
        )

    def pv_curve(peak):
        return peak * np.clip(np.sin(np.pi * (h - 6) / 12.0), 0.0, None) ** 1.5

    profiles = {
        "agent_0": dict(consumer_profile=demand_curve(1.2, 2.5, 0.4).tolist(),
                        generator_profile=pv_curve(3.5).tolist()),
        "agent_1": dict(consumer_profile=demand_curve(2.0, 3.0, 0.5).tolist(),
                        generator_profile=pv_curve(1.0).tolist()),
        "agent_2": dict(consumer_profile=demand_curve(0.8, 1.5, 0.3).tolist(),
                        generator_profile=pv_curve(4.5).tolist()),
        "agent_3": dict(consumer_profile=demand_curve(1.5, 2.0, 0.6).tolist(),
                        generator_profile=pv_curve(2.0).tolist()),
        "agent_4": dict(consumer_profile=demand_curve(1.0, 2.2, 0.35).tolist(),
                        generator_profile=pv_curve(3.0).tolist()),
        "agent_5": dict(consumer_profile=demand_curve(2.2, 3.5, 0.55).tolist(),
                        generator_profile=pv_curve(0.8).tolist()),
    }
    for k in profiles:
        d = np.array(profiles[k]["consumer_profile"])
        g = np.array(profiles[k]["generator_profile"])
        d *= 1 + 0.05 * rng.standard_normal(n_hours)
        g *= 1 + 0.05 * rng.standard_normal(n_hours)
        profiles[k]["consumer_profile"] = np.clip(d, 0, None).tolist()
        profiles[k]["generator_profile"] = np.clip(g, 0, None).tolist()
    return profiles


# ---------------------------------------------------------------------------
# Aplicación del ruido (idéntico al entorno)
# ---------------------------------------------------------------------------

def apply_noise(
    orig: np.ndarray,
    sigma: float,
    n_realizations: int = 30,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Aplica el mismo ruido multiplicativo que P2PEnergyEnv.

    Parameters
    ----------
    orig : (H,) array
        Perfil original (D o G de un agente).
    sigma : float
        Desviación estándar relativa del ruido.
    n_realizations : int
        Número de muestras a generar.
    rng : np.random.Generator, opcional
        Generador a usar (si None, se crea uno por defecto).

    Returns
    -------
    samples : (n_realizations, H) array
        Realizaciones de tilde_X = max(X * (1 + sigma * xi), 0).
    """
    if rng is None:
        rng = np.random.default_rng()
    xi = rng.standard_normal(size=(n_realizations, orig.shape[0]))
    return np.clip(orig[None, :] * (1.0 + sigma * xi), 0.0, None)


# ---------------------------------------------------------------------------
# Plot principal
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Tuple

def plot_noise(
    profiles: dict,
    sigmas: Sequence[float] = (0.1, 0.2, 0.3),
    n_realizations: int = 30,
    seed: int = 42,
    sigma_colors: Optional[dict] = None,
    show_trajectories: int = 2,
    outfile: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    suptitle: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = (0, 6)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Dibuja los perfiles D y G originales junto a las bandas de ruido para
    cada nivel de sigma. Optimizado para diapositivas horizontales (16:9).
    """
    rng = np.random.default_rng(seed)
    agent_ids = list(profiles.keys())
    n_agents = len(agent_ids)

    if sigma_colors is None:
        default_palette = ["#1f77b4", "#ff7f0e", "#d62728",
                           "#2ca02c", "#9467bd", "#8c564b"]
        sigma_colors = {s: default_palette[i % len(default_palette)]
                        for i, s in enumerate(sigmas)}

    sigmas_draw = sorted(sigmas, reverse=True)

    # 1. Ajuste de tamaño horizontal (Formato 16:9 aprox)
    if figsize is None:
        # Ancho dinámico basado en el número de agentes, altura fija optimizada
        width = max(15.0, 4.5 * n_agents)
        figsize = (width, 15.0)
        
    # 2. Invertir la grilla: 2 filas (D y G), N columnas (Agentes)
    # squeeze=False garantiza que axes siempre sea una matriz 2D, incluso con 1 agente
    fig, axes = plt.subplots(nrows=2, ncols=n_agents,
                             figsize=figsize, sharex=True, squeeze=False)

    hours = np.arange(len(profiles[agent_ids[0]]["consumer_profile"]))

    for col, aid in enumerate(agent_ids):
        D = np.asarray(profiles[aid]["consumer_profile"], dtype=np.float64)
        G = np.asarray(profiles[aid]["generator_profile"], dtype=np.float64)

        for row, (label, X) in enumerate([("Demanda D", D),
                                          ("Generación G", G)]):
            ax = axes[row, col]

            for sigma in sigmas_draw:
                samples = apply_noise(X, sigma, n_realizations, rng)
                mean = samples.mean(axis=0)
                std = samples.std(axis=0)
                color = sigma_colors[sigma]

                ax.fill_between(
                    hours,
                    mean - 1.96 * std,
                    mean + 1.96 * std,
                    color=color, alpha=0.25, linewidth=0,
                )
                for k in range(min(show_trajectories, n_realizations)):
                    # Trayectorias un poco más visibles
                    ax.plot(hours, samples[k], color=color,
                            alpha=0.5, linewidth=1.2)

            # 3. Línea original más gruesa para que destaque en la proyección
            ax.plot(hours, X, color="black", linewidth=2.8, zorder=10)

            # Títulos legibles y jerarquizados
            if row == 0:
                ax.set_title(f"Agente: {aid}\n{label}", fontsize=20, fontweight='bold')
            else:
                ax.set_title(f"{label}", fontsize=20, fontweight='bold')

            if col == 0:
                ax.set_ylabel("Potencia", fontsize=20)
                
            ax.set_ylim(ylim)
            ax.tick_params(axis='both', which='major', labelsize=11)
            ax.grid(True, alpha=0.4)

    # Etiquetas del eje X solo en la fila inferior
    for ax in axes[-1, :]:
        ax.set_xlabel("Hora del día", fontsize=20)

    legend_handles = [
        plt.Line2D([], [], color="black", linewidth=2.8, label="Original"),
    ]
    for sigma in sigmas:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=sigma_colors[sigma], alpha=0.25,
                          label=f"σ = {sigma:.2f} (banda 95%)")
        )
        
    # 4. Leyenda en la parte inferior para liberar espacio vertical
    fig.legend(handles=legend_handles,
               loc="lower center", ncol=len(legend_handles),
               bbox_to_anchor=(0.5, 0.02), fontsize=18, frameon=False)

    if suptitle is None:
        suptitle = (
            "Efecto del ruido multiplicativo sobre los perfiles de Demanda y Generación\n"
            r"$\tilde{X}_{i,h} = \max\!\left(X_{i,h}\,(1 + \sigma\,\xi_{i,h}),\ 0\right)$"
        )
    fig.suptitle(suptitle, y=0.98, fontsize=22, fontweight='bold')

    # Ajustamos los márgenes para acomodar la leyenda inferior y el título superior
    fig.tight_layout()
    fig.subplots_adjust(top=0.86, bottom=0.16)

    if outfile is not None:
        # dpi=300 asegura que se vea nítida en pantallas grandes
        fig.savefig(outfile, dpi=300, bbox_inches="tight")

    return fig, axes