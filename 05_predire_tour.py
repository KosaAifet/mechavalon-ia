"""
ÉTAPE 5 — PRÉDIRE UN TOUR (ce que ferait l'API du jeu)

Prend un tour d'un joueur du TEST (jamais vu à l'entraînement), lit les données brutes au format
du jeu, et produit :
  - resultats/exemple_tour_predit.json : le tour suivant prédit, dans le format du jeu,
    avec la probabilité de chaque choix, à côté de ce que le joueur a réellement fait ;
  - resultats/figures/fig7_tour_predit.png : plateau avec la carte de présence prévue
    (ce que le bot tactique utiliserait), les déplacements prédits et les déplacements réels.

Usage : python 05_predire_tour.py            (exemple choisi automatiquement)
        python 05_predire_tour.py 12345      (numéro de ligne du dataset)
"""
import gzip
import importlib.util
import json
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

import config as C
import entites as E


def module(fichier, nom):
    spec = importlib.util.spec_from_file_location(nom, C.RACINE / fichier)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


etl = module("02_etl.py", "etl")
n2 = module("03b_entrainement_niveau2.py", "n2")


def charger_tours(index_joueur, num_partie):
    """Relit les tours bruts d'une partie (format du jeu) et les nettoie comme l'ETL."""
    with gzip.open(C.DOSSIER_RAW / "joueurs.jsonl.gz", "rt", encoding="utf-8") as f:
        for i, ligne in enumerate(f):
            if i == index_joueur:
                (id_brut, parties), = json.loads(ligne).items()
                break
    tours = parties[f"Partie_{num_partie}"]
    for t in tours.values():
        for camp in ("joueur", "IA"):
            for u in t[camp]["unites"].values():
                etl.nettoyer_unite(u, Counter())
    return id_brut, tours


def predire(modele, d, ligne, tour_n, tour_n1):
    """Le cœur de l'inférence : 10 derniers tours + plateau actuel -> tour suivant prédit."""
    unites, bats = E.extraire(tour_n, tour_n1, avec_cibles=False)
    debut = d["debut"][ligne]
    bloc = d["X"][max(debut, ligne - n2.FENETRE + 1): ligne + 1]
    seq = np.zeros((1, n2.FENETRE, d["X"].shape[1]), dtype=np.float32)
    seq[0, n2.FENETRE - len(bloc):] = bloc
    c_uloc, c_rang, c_var = [], [], []
    for j, u in enumerate(unites):
        for r, (_, v) in enumerate(u["candidats"]):
            c_uloc.append(j)
            c_rang.append(r)
            c_var.append(v)
    b = {"seq": torch.tensor(seq),
         "u_loc": torch.zeros(len(unites), dtype=torch.long),
         "u_var": torch.tensor([u["variables"] for u in unites], dtype=torch.float32),
         "u_masque": torch.tensor([u["masque"] for u in unites]),
         "c_uloc": torch.tensor(c_uloc, dtype=torch.long), "c_rang": torch.tensor(c_rang, dtype=torch.long),
         "c_var": torch.tensor(c_var, dtype=torch.float32).reshape(-1, E.NB_VAR_CANDIDAT),
         "b_loc": torch.zeros(len(bats), dtype=torch.long),
         "b_var": torch.tensor([b_["variables"] for b_ in bats], dtype=torch.float32).reshape(-1, E.NB_VAR_BATIMENT)}
    with torch.no_grad():
        s = modele(b)
    p = {k: torch.softmax(v, 1).numpy() for k, v in s.items()}
    return unites, bats, p


def main():
    d, e = n2.charger()
    ck = torch.load(C.DOSSIER_MODELES / "mechavalon_niveau2.pt")
    modele = n2.MechavalonNiveau2(ck["nb_entrees"])
    modele.load_state_dict(ck["etat"])
    modele.eval()

    ordre = np.random.RandomState(C.GRAINE).permutation(int(d["id_joueur"].max()) + 1)
    test = ordre[int(0.85 * len(ordre)):]
    if len(sys.argv) > 1:
        ligne = int(sys.argv[1])
    else:   # un tour de test avec au moins 4 unités de combat dont une au contact de l'ennemi
        rng = np.random.RandomState(7)
        candidates = [i for i in np.where(np.isin(d["id_joueur"], test) & (d["y_mvt"] >= 0))[0]
                      if (e["u_fin"][i] - e["u_debut"][i]) >= 5
                      and (e["u_attaque"][e["u_debut"][i]:e["u_fin"][i]] > 0).any()
                      and d["y_ges"][i] == C.GESTIONS.index("ENTRAINER")
                      and len({tuple(p) for p in e["u_pos"][e["u_debut"][i]:e["u_fin"][i]]})
                      == e["u_fin"][i] - e["u_debut"][i]           # pas deux unités sur la même case (lisibilité)
                      and (e["c_var"][e["c_debut"][e["u_debut"][i]]:e["c_fin"][e["u_fin"][i] - 1], 2] > 0).all()]
        # on garde un exemple représentatif : la plupart des unités justes, mais pas toutes
        candidates = np.array(candidates)
        seq = n2.sequences(d, candidates)
        pos = np.zeros(len(d["X"]), dtype=np.int64)
        pos[candidates] = np.arange(len(candidates))
        with torch.no_grad():
            b = n2.lot(d, e, seq, pos, candidates)
            s = modele(b)
        juste = (s["deplacement"].argmax(1).numpy() == e["u_delta"][b["u_idx"]]) & \
                (s["attaque"].argmax(1).numpy() == e["u_attaque"][b["u_idx"]])
        part = np.bincount(b["u_loc"].numpy(), weights=juste, minlength=len(candidates)) / \
            np.bincount(b["u_loc"].numpy(), minlength=len(candidates))
        representatifs = candidates[(part >= 0.7) & (part < 1)]
        ligne = int(rng.choice(representatifs if len(representatifs) else candidates))
    n, num_partie = int(d["tour"][ligne]), int(d["num_partie"][ligne])
    id_brut, tours = charger_tours(int(d["id_joueur"][ligne]), num_partie)
    tour_n, tour_n1 = tours[f"Tour_{n}"], tours[f"Tour_{n + 1}"]
    unites, bats, p = predire(modele, d, ligne, tour_n, tour_n1)

    # ------------------------------ tour prédit au format du jeu ----------------------------
    pred_unites, pred_bats, presence = {}, {}, np.zeros((C.HAUTEUR, C.LARGEUR))
    for j, u in enumerate(unites):
        x, y = u["pos"]
        ordre_cases = np.argsort(-p["deplacement"][j])
        for k in ordre_cases:
            if p["deplacement"][j, k] < 0.01:
                break
            dx, dy = E.DELTAS[k]
            presence[y + dy, x + dx] += p["deplacement"][j, k]
        best = ordre_cases[0]
        dx, dy = E.DELTAS[best]
        att = p["attaque"][j]
        ia = int(att.argmax())
        reel = tour_n1["joueur"]["unites"][u["id"]]
        pred_unites[u["id"]] = {
            "Nom": u["Nom"], "position": list(u["pos"]),
            "Movement": [x + dx, y + dy], "proba_Movement": round(float(p["deplacement"][j, best]), 2),
            "autres_cases": [{"case": [x + E.DELTAS[k][0], y + E.DELTAS[k][1]],
                              "proba": round(float(p["deplacement"][j, k]), 2)} for k in ordre_cases[1:3]],
            "Attaque": None if ia == 0 else u["candidats"][ia - 1][0], "proba_Attaque": round(float(att[ia]), 2),
            "construct": None if u["Nom"] != "Ouvrier" or p["construction"][j].argmax() == 0
            else E.CONSTRUCTIONS[int(p["construction"][j].argmax())],
            "REEL": {"Movement": reel["Movement"], "Attaque": reel["Attaque"], "construct": reel["construct"]},
        }
    for j, b in enumerate(bats):
        k = int(p["entraine"][j].argmax())
        pred_bats[b["id"]] = {"Nom": b["Nom"], "entraine": None if k == 0 else E.ENTRAINEMENTS[k],
                              "proba_entraine": round(float(p["entraine"][j, k]), 2),
                              "REEL": {"entraine": tour_n1["joueur"]["batiments"][b["id"]]["entraine"]}}
    sortie = {
        "joueur": "joueur de test (pseudonymisé)", "partie": num_partie, "tour_predit": n + 1,
        "intentions": {"mouvement": {m: round(float(v), 2) for m, v in zip(C.MOUVEMENTS, p["mouvement"][0])},
                       "gestion": {g: round(float(v), 2) for g, v in zip(C.GESTIONS, p["gestion"][0])}},
        f"Tour_{n + 1}_predit": {"unites": pred_unites, "batiments": pred_bats},
        "carte_de_presence": [[round(float(v), 2) for v in rang] for rang in presence],
    }
    (C.DOSSIER_RESULTATS / "exemple_tour_predit.json").write_text(
        json.dumps(sortie, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------ figure ----------------------------------------------------
    BLEUS = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.2))
    for ax, titre in zip(axes, ("Prédiction du modèle", "Ce que le joueur a fait")):
        ax.set_facecolor("#fcfcfb")
        if titre.startswith("Prédiction"):
            ax.imshow(presence, cmap=BLEUS, origin="lower", vmin=0, vmax=max(1, presence.max()), alpha=0.9)
        for xs in range(C.LARGEUR + 1):
            ax.axvline(xs - 0.5, color="#ebe9e3", linewidth=0.6)
        for ys in range(C.HAUTEUR + 1):
            ax.axhline(ys - 0.5, color="#ebe9e3", linewidth=0.6)
        for k, b in tour_n1["IA"]["batiments"].items():
            ax.scatter(*b["position"], marker="s", s=260, color="#eb6834", edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        for k, u in tour_n1["IA"]["unites"].items():
            ax.scatter(*u["position"], marker="D", s=120, color="#eb6834", edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        for k, b in tour_n1["joueur"]["batiments"].items():
            ax.scatter(*b["position"], marker="s", s=260, color="#184f95", edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        deja = set()
        for uid, u in pred_unites.items():
            x0, y0 = u["position"]
            x1, y1 = u["Movement"] if titre.startswith("Prédiction") else u["REEL"]["Movement"]
            cible = u["Attaque"] if titre.startswith("Prédiction") else u["REEL"]["Attaque"]
            ax.scatter(x0, y0, s=150, color="#2a78d6", edgecolor="#fcfcfb", linewidth=1.5, zorder=4)
            if (x1, y1) != (x0, y0):
                ax.annotate("", (x1, y1), (x0, y0), zorder=5,
                            arrowprops=dict(arrowstyle="-|>", color="#0b0b0b", lw=1.8, mutation_scale=14))
            if cible:
                pos_c = (tour_n1["IA"]["unites"].get(cible) or tour_n1["IA"]["batiments"].get(cible))["position"]
                ax.annotate("", pos_c, (x1, y1), zorder=5,
                            arrowprops=dict(arrowstyle="-|>", color="#d03b3b", lw=1.6, linestyle=(0, (3, 2)),
                                            mutation_scale=12))
            if titre.startswith("Prédiction") and (x1, y1) not in deja:
                deja.add((x1, y1))
                ax.text(x1 + 0.22, y1 + 0.22, f"{u['proba_Movement']:.0%}", fontsize=10, color="#0b0b0b", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none", alpha=0.85))
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, C.LARGEUR - 0.5)
        ax.set_ylim(-0.5, C.HAUTEUR - 0.5)
        ax.set_xticks([])
        ax.set_yticks([0, C.HAUTEUR - 1])
        ax.set_yticklabels(["Base joueur", "Base IA"] if ax is axes[0] else ["", ""])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(titre, fontsize=14, fontweight="bold", loc="left")
    from matplotlib.lines import Line2D
    legende = [Line2D([], [], marker="o", ls="", color="#2a78d6", markersize=10, label="Unité du joueur (début du tour)"),
               Line2D([], [], marker="D", ls="", color="#eb6834", markersize=8, label="Unité de l'IA"),
               Line2D([], [], marker="s", ls="", color="#184f95", markersize=10, label="Bâtiment du joueur"),
               Line2D([], [], marker="s", ls="", color="#eb6834", markersize=10, label="Bâtiment de l'IA"),
               Line2D([], [], color="#0b0b0b", lw=1.8, label="Déplacement"),
               Line2D([], [], color="#d03b3b", lw=1.6, ls=(0, (3, 2)), label="Attaque")]
    fig.legend(handles=legende, loc="lower center", ncol=6, frameon=False, fontsize=10)
    ent = ", ".join(f"{v['Nom']} : {v['entraine'] or 'rien'} ({v['proba_entraine']:.0%})" for v in pred_bats.values())
    ent_r = ", ".join(f"{v['Nom']} : {v['REEL']['entraine'] or 'rien'}" for v in pred_bats.values())
    fig.suptitle(f"Tour {n + 1} prédit pour un joueur jamais vu — fond bleu : carte de présence prévue\n"
                 f"entraine prédit : {ent}   |   réel : {ent_r}", x=0.02, ha="left", fontsize=13)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(C.DOSSIER_FIGURES / "fig7_tour_predit.png", dpi=170)
    print(f"Ligne {ligne} : joueur {id_brut}, partie {num_partie}, tour {n + 1} prédit")
    print(json.dumps({k: v for k, v in sortie.items() if k != "carte_de_presence"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
