"""
ÉTAPE 4 — VISUALISATIONS (prêtes à coller dans les slides, format 16:9)

Lit resultats/metriques.json et data/processed/dataset.npz, produit :
  fig1_comparaison_modeles.png   le LSTM face aux modèles de référence (F1 macro, 2 sorties)
  fig2_courbes_apprentissage.png perte entraînement / validation (sur-apprentissage ?)
  fig3_matrices_confusion.png    où le modèle se trompe (mouvement et gestion)
  fig4_demarrage_a_froid.png     apport du profil joueur au fil des parties
  fig5_positions_par_style.png   où jouent les joueurs selon leur style (carte de chaleur)
  fig6_mouvements_par_style.png  répartition des mouvements par style
  fig8_niveau2_comparaison.png   niveau 2 : case exacte et cible d'attaque, LSTM contre règles simples
(fig7_tour_predit.png est produite par 05_predire_tour.py)
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import config as C

# Palette validée (daltonisme, contraste) — voir README
SURFACE, ENCRE, ENCRE_2, DISCRET, GRILLE, AXE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
BLEUS = LinearSegmentedColormap.from_list(
    "bleus", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
NOMS = {"AVANCER_GAUCHE": "Avancer gauche", "AVANCER_CENTRE": "Avancer centre", "AVANCER_DROITE": "Avancer droite",
        "ATTAQUER": "Attaquer", "DEFENDRE": "Défendre", "RIEN": "Rien", "ENTRAINER": "Entraîner",
        "CONSTRUIRE": "Construire"}
STYLES = {"agressif": "Agressif", "defensif": "Défensif", "flanqueur": "Flanqueur",
          "economiste": "Économiste", "opportuniste": "Opportuniste"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": ENCRE,
    "axes.edgecolor": AXE, "axes.labelcolor": ENCRE_2, "xtick.color": DISCRET, "ytick.color": ENCRE_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 15, "axes.titlelocation": "left", "axes.titlepad": 14, "legend.frameon": False,
})
TAILLE = (11, 6.2)


def sauver(fig, nom):
    fig.tight_layout()
    fig.savefig(C.DOSSIER_FIGURES / nom, dpi=180)
    plt.close(fig)
    print(f"  {nom}")


def fig_comparaison(r):
    noms = list(r["modeles"])
    mvt = [r["modeles"][n]["mouvement"]["f1_macro"] * 100 for n in noms]
    ges = [r["modeles"][n]["gestion"]["f1_macro"] * 100 for n in noms]
    fig, ax = plt.subplots(figsize=TAILLE)
    y = np.arange(len(noms))[::-1]
    h = 0.36
    ax.barh(y + h / 2, mvt, h, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, label="Mouvement (5 classes)")
    ax.barh(y - h / 2, ges, h, color=SERIES[1], edgecolor=SURFACE, linewidth=1.5, label="Gestion (3 classes)")
    for yi, a, b in zip(y, mvt, ges):
        ax.text(a + 1, yi + h / 2, f"{a:.0f} %", va="center", fontsize=10, color=ENCRE_2)
        ax.text(b + 1, yi - h / 2, f"{b:.0f} %", va="center", fontsize=10, color=ENCRE_2)
    ax.set_yticks(y)
    ax.set_yticklabels(noms)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == "LSTM v3":
            lbl.set_fontweight("bold")
            lbl.set_color(ENCRE)
    ax.set_xlim(0, 100)
    ax.set_ylim(y[-1] - 0.7, y[0] + 0.7)
    ax.set_xlabel("F1 macro en %  (jeu de test : joueurs jamais vus)")
    ax.grid(axis="x", color=GRILLE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Le LSTM prédit mieux que les modèles de référence, sur les deux sorties")
    ax.legend(loc="upper right")
    sauver(fig, "fig1_comparaison_modeles.png")


def fig_courbes(r):
    h = r["modele_final"]["historique"]
    ep = np.arange(1, len(h["perte_train"]) + 1)
    fig, ax = plt.subplots(figsize=TAILLE)
    ax.plot(ep, h["perte_train"], color=SERIES[0], linewidth=2, label="Entraînement")
    ax.plot(ep, h["perte_val"], color=SERIES[1], linewidth=2, label="Validation")
    m = int(np.argmin(h["perte_val"]))
    ax.scatter([ep[m]], [h["perte_val"][m]], s=60, color=SERIES[1], zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.annotate(f"modèle retenu (époque {ep[m]})", (ep[m], h["perte_val"][m]), xytext=(-10, 30),
                textcoords="offset points", ha="right", color=ENCRE_2, fontsize=10)
    ax.text(ep[-1] + 0.3, h["perte_train"][-1], "Entraînement", color=ENCRE_2, va="center", fontsize=10)
    ax.text(ep[-1] + 0.3, h["perte_val"][-1], "Validation", color=ENCRE_2, va="center", fontsize=10)
    ax.set_xlim(1, ep[-1] + 3)
    ax.set_xlabel("Époque")
    ax.set_ylabel("Perte (Cross-Entropy mouvement + gestion)")
    ax.grid(axis="y", color=GRILLE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Courbes d'apprentissage du LSTM v3 — pas de sur-apprentissage marqué")
    ax.legend(loc="upper right")
    sauver(fig, "fig2_courbes_apprentissage.png")


def matrice(ax, m, classes, titre):
    m = np.array(m, dtype=float)
    pct = m / m.sum(1, keepdims=True) * 100
    im = ax.imshow(pct, cmap=BLEUS, vmin=0, vmax=100)
    noms = [NOMS[c] for c in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(noms, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(noms)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if pct[i, j] >= 0.5:
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=11,
                        color="#ffffff" if pct[i, j] > 55 else ENCRE)
    ax.set_xlabel("Prédit")
    if titre == "Mouvement":
        ax.set_ylabel("Réel")
    ax.set_title(titre, fontsize=13)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def fig_confusion(r):
    mf = r["modele_final"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), gridspec_kw={"width_ratios": [5, 3.6]})
    matrice(axes[0], mf["mouvement"]["matrice_confusion"], C.MOUVEMENTS, "Mouvement")
    im = matrice(axes[1], mf["gestion"]["matrice_confusion"], C.GESTIONS, "Gestion")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("% des tours de la ligne", color=ENCRE_2)
    cb.outline.set_visible(False)
    fig.suptitle("Matrices de confusion du LSTM v3 (en % par classe réelle)", x=0.02, ha="left",
                 fontweight="bold", fontsize=15)
    fig.savefig(C.DOSSIER_FIGURES / "fig3_matrices_confusion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  fig3_matrices_confusion.png")


def fig_demarrage(r):
    fig, axes = plt.subplots(1, 2, figsize=TAILLE)
    for ax, s, titre in zip(axes, ("mouvement", "gestion"), ("Mouvement", "Gestion")):
        v3 = r["modele_final"][s]["exactitude_par_num_partie"]
        v2 = r["modele_final"][s]["exactitude_par_num_partie_v2_sans_profil"]
        x = [int(k) for k in v3]
        ax.plot(x, [v * 100 for v in v2.values()], color=DISCRET, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label="v2 — sans profil")
        ax.plot(x, [v * 100 for v in v3.values()], color=SERIES[0], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label="v3 — avec profil")
        bas = min(min(v2.values()), min(v3.values())) * 100
        ax.set_ylim(bas - 4, bas + 8)
        ax.set_xticks(x)
        ax.set_xlabel("Numéro de la partie du joueur")
        ax.set_ylabel("Exactitude (%)")
        ax.grid(axis="y", color=GRILLE, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(titre, fontsize=13)
        ax.legend(loc="lower right", fontsize=10)
    fig.suptitle("Le profil joueur aide dès la 2e partie (1re partie : aucun historique)", x=0.02, ha="left",
                 fontweight="bold", fontsize=15)
    sauver(fig, "fig4_demarrage_a_froid.png")


def fig_positions(d):
    pos = d["positions_par_style"]
    pos = pos / pos.sum(axis=(1, 2), keepdims=True) * 100
    choix = ["agressif", "defensif", "flanqueur"]
    vmax = np.percentile(pos[[C.STYLES.index(s) for s in choix]], 99.5)
    fig, axes = plt.subplots(1, 3, figsize=TAILLE)
    for ax, s in zip(axes, choix):
        ax.imshow(pos[C.STYLES.index(s)], cmap=BLEUS, origin="lower", vmin=0, vmax=vmax)
        ax.add_patch(Rectangle((-0.5, -0.5), C.LARGEUR, C.HAUTEUR, fill=False, edgecolor=AXE, linewidth=1))
        for xs in (3.5, 7.5):
            ax.axvline(xs, color=AXE, linestyle=(0, (3, 3)), linewidth=0.8)
        ax.set_title(STYLES[s], fontsize=13, loc="center")
        ax.set_xticks([1.5, 5.5, 9.5])
        ax.set_xticklabels(["gauche", "centre", "droite"], fontsize=10)
        ax.tick_params(length=0)
        ax.set_yticks([0, C.HAUTEUR - 1] if s == choix[0] else [])
        if s == choix[0]:
            ax.set_yticklabels(["Base joueur", "Base IA"])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Où se trouvent les unités de combat selon le style (foncé = plus de présence)",
                 x=0.02, ha="left", fontweight="bold", fontsize=15)
    sauver(fig, "fig5_positions_par_style.png")


def fig_mouvements_style(d):
    y, st = d["y_mvt"], d["style"]
    ok = y >= 0
    parts = np.array([np.bincount(y[ok & (st == i)], minlength=C.NB_MOUVEMENTS) for i in range(len(C.STYLES))],
                     dtype=float)
    parts = parts / parts.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=TAILLE)
    gauche = np.zeros(len(C.STYLES))
    ypos = np.arange(len(C.STYLES))[::-1]
    for k, a in enumerate(C.MOUVEMENTS):
        ax.barh(ypos, parts[:, k], left=gauche, color=SERIES[k], edgecolor=SURFACE, linewidth=2,
                label=NOMS[a], height=0.6)
        for yi, l, v in zip(ypos, gauche, parts[:, k]):
            if v >= 6:
                ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=10, color=ENCRE)
        gauche += parts[:, k]
    ax.set_yticks(ypos)
    ax.set_yticklabels([STYLES[s] for s in C.STYLES])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% des tours")
    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, -0.13), fontsize=10)
    ax.spines["left"].set_visible(False)
    ax.set_title("Chaque style a une signature de mouvements différente")
    sauver(fig, "fig6_mouvements_par_style.png")


def fig_niveau2(r):
    if "niveau2" not in r:
        return
    n2 = r["niveau2"]
    fig, axes = plt.subplots(1, 2, figsize=TAILLE, gridspec_kw={"width_ratios": [3, 2]})
    panneaux = [
        (axes[0], n2["deplacement"], "case_exacte_unites_combat", "Case d'arrivée exacte\n(unités de combat)"),
        (axes[1], n2["attaque"], "bonne_cible_quand_il_attaque", "Bonne cible\n(quand le joueur attaque)"),
    ]
    for ax, bloc, cle, titre in panneaux:
        noms = list(bloc)
        vals = [bloc[n][cle] * 100 for n in noms]
        couleurs = [SERIES[0] if n.startswith("LSTM") else "#b5b3ab" for n in noms]
        x = np.arange(len(noms))
        ax.bar(x, vals, color=couleurs, width=0.6, edgecolor=SURFACE, linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 1.5, f"{v:.0f} %", ha="center", fontsize=12, color=ENCRE, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("Attaquer la cible la plus faible à portée", "Règle : la plus\nfaible à portée")
                            .replace("Répéter le dernier déplacement", "Répéter le dernier\ndéplacement")
                            for n in noms], fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% (joueurs jamais vus)")
        ax.grid(axis="y", color=GRILLE, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(titre, fontsize=13)
    fig.suptitle("Niveau 2 : le LSTM devine la case et la cible bien mieux que des règles simples",
                 x=0.02, ha="left", fontweight="bold", fontsize=15)
    sauver(fig, "fig8_niveau2_comparaison.png")


def main():
    C.DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    r = json.loads((C.DOSSIER_RESULTATS / "metriques.json").read_text(encoding="utf-8"))
    d = np.load(C.DOSSIER_PROCESSED / "dataset.npz")
    fig_comparaison(r)
    fig_courbes(r)
    fig_confusion(r)
    fig_demarrage(r)
    fig_positions(d)
    fig_mouvements_style(d)
    fig_niveau2(r)
    print(f"-> {C.DOSSIER_FIGURES}")


if __name__ == "__main__":
    main()
