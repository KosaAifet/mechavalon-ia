"""
NIVEAU 2 — préparation des données « unité par unité ».

Au début du tour N+1, le plateau est connu (positions des unités des deux camps, PV, bâtiments).
Pour chaque unité et chaque bâtiment du joueur, on décrit sa situation et on note ce qu'il a
réellement fait pendant le tour N+1 (la cible à prédire) :
  - Movement  : déplacement (dx, dy) parmi les 25 possibles de -2 à +2 (masqué selon la capacité de l'unité)
  - Attaque   : aucune, ou l'une des cibles ennemies à portée (au plus MAX_CANDIDATS)
  - construct : rien / Scierie / Mine / Raffinerie (ouvriers uniquement)
  - entraine  : rien / Piquier / Archer / Cavalier / Ouvrier (QG et caserne)

Ce module sert à l'ETL (02_etl.py) et à la prédiction d'un tour (05_predire_tour.py).
"""
import numpy as np

import config as C

DELTAS = [(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)]       # 25 déplacements possibles
INDEX_DELTA = {d: i for i, d in enumerate(DELTAS)}
TYPES_UNITE = ["Piquier", "Archer", "Cavalier", "Ouvrier"]
CONSTRUCTIONS = ["rien"] + C.BATIMENTS_ECO
ENTRAINEMENTS = ["rien"] + TYPES_UNITE
BATIMENTS_FORMATEURS = ("QG", "Caserne")
MAX_CANDIDATS = 8
NB_VAR_UNITE = 18
NB_VAR_CANDIDAT = 11
NB_VAR_BATIMENT = 7


def dist(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def plateau_debut(tour_n, tour_n1):
    """Plateau au début du tour N+1 : positions de N+1 (avant que le joueur joue), PV hérités de la fin de N."""
    def pv(camp, k, x):
        avant = tour_n[camp]["unites"].get(k)
        return avant["PV"] if avant else C.UNITES[x["Nom"]]["pv"]

    def pv_bat(camp, k, b):
        avant = tour_n[camp]["batiments"].get(k)
        return avant["PV"] if avant else C.BATIMENTS[b["Nom"]]

    j_unites = {k: {"Nom": u["Nom"], "pos": tuple(u["position"]), "PV": pv("joueur", k, u),
                    "avant": tour_n["joueur"]["unites"].get(k)} for k, u in tour_n1["joueur"]["unites"].items()}
    ia_unites = {k: {"Nom": u["Nom"], "pos": tuple(u["position"]), "PV": pv("IA", k, u)}
                 for k, u in tour_n1["IA"]["unites"].items()}
    ia_bats = {k: {"Nom": b["Nom"], "pos": tuple(b["position"]), "PV": pv_bat("IA", k, b)}
               for k, b in tour_n1["IA"]["batiments"].items()}
    j_bats = {k: {"Nom": b["Nom"], "pos": tuple(b["position"]), "PV": pv_bat("joueur", k, b),
                  "avant": tour_n["joueur"]["batiments"].get(k)} for k, b in tour_n1["joueur"]["batiments"].items()}
    return j_unites, ia_unites, ia_bats, j_bats


def variables_unite(u, ia_unites, ia_bats, j_bats):
    x, y = u["pos"]
    t = [1.0 if u["Nom"] == n else 0.0 for n in TYPES_UNITE]
    avant = u["avant"]
    if avant:
        dx = (avant["Movement"][0] - avant["position"][0]) / 2
        dy = (avant["Movement"][1] - avant["position"][1]) / 2
        a_attaque, a_construit = float(avant["Attaque"] is not None), float(avant["construct"] is not None)
    else:
        dx = dy = a_attaque = a_construit = 0.0
    ennemis = list(ia_unites.values())
    if ennemis:
        e = min(ennemis, key=lambda e: dist(u["pos"], e["pos"]))
        proche = [(e["pos"][0] - x) / 11, (e["pos"][1] - y) / 11, dist(u["pos"], e["pos"]) / 11,
                  e["PV"] / C.UNITES[e["Nom"]]["pv"]]
    else:
        proche = [0.0, 0.0, 1.0, 0.0]
    autour = min(sum(1 for e in ennemis if dist(u["pos"], e["pos"]) <= 2) / 5, 1)
    qg_ia = next((b["pos"] for b in ia_bats.values() if b["Nom"] == "QG"), (6, 11))
    qg_j = next((b["pos"] for b in j_bats.values() if b["Nom"] == "QG"), (6, 0))
    return (t + [u["PV"] / C.UNITES[u["Nom"]]["pv"], x / 11, y / 10, dx, dy, a_attaque, a_construit]
            + proche + [autour, dist(u["pos"], qg_ia) / 11, dist(u["pos"], qg_j) / 11])


def candidats(u, ia_unites, ia_bats):
    """Cibles ennemies atteignables ce tour : distance <= déplacement + portée."""
    caract = C.UNITES[u["Nom"]]
    if caract["portee"] == 0:
        return []
    portee = caract["mvt"] + caract["portee"]
    x, y = u["pos"]
    liste = []
    for k, e in list(ia_unites.items()) + list(ia_bats.items()):
        d = dist(u["pos"], e["pos"])
        if d > portee:
            continue
        est_unite = k in ia_unites
        pv_max = C.UNITES[e["Nom"]]["pv"] if est_unite else C.BATIMENTS[e["Nom"]]
        liste.append((d, k, [(e["pos"][0] - x) / 4, (e["pos"][1] - y) / 4, d / 4, e["PV"] / pv_max,
                             float(est_unite), float(e["Nom"] == "QG"), float(e["Nom"] == "Caserne"),
                             float(e["Nom"] == "Tour")] + [float(e["Nom"] == n) for n in C.TYPES_COMBAT]))
    liste.sort(key=lambda c: c[0])
    return [(k, v) for _, k, v in liste[:MAX_CANDIDATS]]


def masque_deplacement(u):
    caract = C.UNITES[u["Nom"]]
    x, y = u["pos"]
    return [abs(dx) <= caract["mvt"] and abs(dy) <= caract["mvt"]
            and 0 <= x + dx <= C.LARGEUR - 1 and 1 <= y + dy <= C.HAUTEUR - 2 or (dx, dy) == (0, 0)
            for dx, dy in DELTAS]


def variables_batiment(b):
    avant = b["avant"]
    entraine_avant = avant["entraine"] if avant else None
    return ([float(b["Nom"] == "QG"), float(b["Nom"] == "Caserne"), b["PV"] / C.BATIMENTS[b["Nom"]],
             float(entraine_avant is not None)]
            + [float(entraine_avant == n) for n in C.TYPES_COMBAT])


def extraire(tour_n, tour_n1, avec_cibles=True):
    """Toutes les entités du joueur au début du tour N+1, avec (si demandé) ce qu'elles ont fait en N+1."""
    j_unites, ia_unites, ia_bats, j_bats = plateau_debut(tour_n, tour_n1)
    unites, batiments = [], []
    for k, u in j_unites.items():
        cands = candidats(u, ia_unites, ia_bats)
        e = {"id": k, "Nom": u["Nom"], "pos": u["pos"], "variables": variables_unite(u, ia_unites, ia_bats, j_bats),
             "masque": masque_deplacement(u), "candidats": cands}
        if avec_cibles:
            reel = tour_n1["joueur"]["unites"][k]
            d = (max(-2, min(2, reel["Movement"][0] - reel["position"][0])),
                 max(-2, min(2, reel["Movement"][1] - reel["position"][1])))
            e["delta"] = INDEX_DELTA[d]
            ids = [c[0] for c in cands]
            e["attaque"] = 0 if reel["Attaque"] is None else (ids.index(reel["Attaque"]) + 1 if reel["Attaque"] in ids else -1)
            e["construct"] = CONSTRUCTIONS.index(reel["construct"]) if reel["construct"] in CONSTRUCTIONS else 0
            e["arrivee"] = tuple(reel["Movement"])
        unites.append(e)
    for k, b in j_bats.items():
        if b["Nom"] not in BATIMENTS_FORMATEURS:
            continue
        e = {"id": k, "Nom": b["Nom"], "variables": variables_batiment(b)}
        if avec_cibles:
            ent = tour_n1["joueur"]["batiments"][k]["entraine"]
            e["entraine"] = ENTRAINEMENTS.index(ent) if ent in ENTRAINEMENTS else 0
        batiments.append(e)
    return unites, batiments
