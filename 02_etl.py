"""
ÉTAPE 2 — ETL (Extract, Transform, Load)

EXTRACT   : lit data/raw/joueurs.jsonl.gz en flux, un document joueur à la fois
            (format du projet : <ID_Joueur> -> <Partie_X> -> <Tour_X> -> joueur / IA).
TRANSFORM : 1. RGPD      -> l'identifiant joueur est remplacé par un pseudonyme (HMAC-SHA256).
            2. Qualité   -> PV aberrants, champ Movement manquant, déplacements impossibles
                            (téléportation), tours manquants.
            3. Étiquettes -> pour chaque tour, ce que le joueur a fait :
                              mouvement (5 classes) d'après Movement et Attaque de ses unités,
                              gestion (3 classes) d'après entraine (bâtiments) et construct (unités).
            4. Variables -> chaque tour devient un vecteur de 53 nombres entre 0 et 1.
            5. Profil    -> habitudes du joueur sur ses parties PRÉCÉDENTES uniquement (pas de fuite).
LOAD      : - data/processed/dataset.npz          matrices prêtes pour l'entraînement
            - data/processed/mongo_parties.jsonl.gz  1 document = 1 partie (schéma optimisé)
            - resultats/rapport_qualite.json et resultats/volumetrie.json

Convention : la ligne du tour N contient tout ce qui est connu à la fin du tour N ;
sa cible est ce que le joueur fera au tour N+1.
"""
import csv
import gzip
import hashlib
import hmac
import json
import os
from collections import Counter, defaultdict

import numpy as np

import config as C
import entites as E

# Clé secrète de pseudonymisation. En production : variable d'environnement,
# jamais dans le code ni sur Git. Sans la clé, impossible de retrouver le joueur.
CLE_PSEUDO = os.environ.get("MECHAVALON_CLE_PSEUDO", "cle-de-dev-a-changer").encode()

NOMS_VARIABLES = (
    # joueur (14)
    ["j_nb_unites", "j_pv_moyen", "j_avancee", "j_part_gauche", "j_part_centre", "j_part_droite",
     "j_part_piquier", "j_part_archer", "j_part_cavalier", "j_part_ouvrier",
     "j_pertes", "j_part_attaque", "j_batiments_eco", "j_qg_pv"]
    # IA (18)
    + ["ia_nb_unites", "ia_pv_moyen", "ia_avancee", "ia_part_gauche", "ia_part_centre", "ia_part_droite",
       "ia_pertes", "ia_part_attaque", "ia_qg_pv", "ia_tour_pv"]
    + [f"ia_{r}" for r in C.RESSOURCES]
    + [f"ia_objectif_{o}" for o in C.TYPES_OBJECTIF]
    # contexte (4)
    + ["distance_min", "j_part_au_contact", "menace_qg", "tour"]
    # ce que le joueur a fait pendant ce tour (8)
    + [f"tour_{m}" for m in C.MOUVEMENTS] + [f"tour_{g}" for g in C.GESTIONS]
    # profil du joueur sur ses parties précédentes (9)
    + [f"profil_{m}" for m in C.MOUVEMENTS] + [f"profil_{g}" for g in C.GESTIONS] + ["profil_nb_parties"]
)
NB_VARIABLES = len(NOMS_VARIABLES)  # 53
DEBUT_PROFIL = NOMS_VARIABLES.index("profil_" + C.MOUVEMENTS[0])
DEBUT_ACTIONS = NOMS_VARIABLES.index("tour_" + C.MOUVEMENTS[0])


def pseudonymiser(identifiant: str) -> str:
    return "anon_" + hmac.new(CLE_PSEUDO, identifiant.encode(), hashlib.sha256).hexdigest()[:12]


def numero(cle: str) -> int:
    """'Tour_15' -> 15, 'Partie_4' -> 4."""
    return int(cle.rsplit("_", 1)[1])


def dist(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


# ---------------------------------------------------------------------------
# Qualité
# ---------------------------------------------------------------------------
def nettoyer_unite(u, qualite):
    pv_max = C.UNITES[u["Nom"]]["pv"]
    if not 0 <= u["PV"] <= pv_max:
        qualite["pv_aberrants_corriges"] += 1
        u["PV"] = min(max(u["PV"], 0), pv_max)
    if "Movement" not in u or u["Movement"] is None:
        qualite["movement_manquant_complete"] += 1
        u["Movement"] = list(u["position"])
    elif dist(u["position"], u["Movement"]) > C.UNITES[u["Nom"]]["mvt"]:
        qualite["deplacements_impossibles_corriges"] += 1
        u["Movement"] = list(u["position"])


# ---------------------------------------------------------------------------
# Étiquettes : ce que le joueur a fait pendant le tour (règles métier)
# ---------------------------------------------------------------------------
def etiquette_mouvement(unites):
    combat = [u for u in unites.values() if u["Nom"] != "Ouvrier"]
    if any(u["Attaque"] for u in combat):
        return "ATTAQUER"
    en_avant = [u for u in combat if u["Movement"][1] > u["position"][1]]
    lateraux = [u for u in combat if u["Movement"][1] == u["position"][1] and u["Movement"][0] != u["position"][0]]
    if not en_avant and not lateraux:
        return "DEFENDRE"   # repli ou maintien en position
    bougent = en_avant + lateraux
    # couloir visé : là où vont les unités, en prolongeant leur déplacement latéral
    x_vise = sum(u["Movement"][0] + 2 * np.sign(u["Movement"][0] - u["position"][0]) for u in bougent) / len(bougent)
    return C.MOUVEMENTS[C.couloir(x_vise)]


def etiquette_gestion(unites, batiments):
    if any(b["entraine"] for b in batiments.values()):
        return "ENTRAINER"
    if any(u["construct"] for u in unites.values()):
        return "CONSTRUIRE"
    return "RIEN"


# ---------------------------------------------------------------------------
# Variables d'un tour (état en fin de tour + actions du tour)
# ---------------------------------------------------------------------------
def resume_unites(unites, cote):
    vivants = [u for u in unites.values() if not u["Destroy"]]
    n = len(vivants)
    combat = [u for u in unites.values() if u["Nom"] != "Ouvrier"]
    pertes = sum(1 for u in unites.values() if u["Destroy"])
    attaque = sum(1 for u in combat if u["Attaque"]) / len(combat) if combat else 0
    if n == 0:
        base = [0, 0, 0, 0, 0, 0]
    else:
        parts = [0, 0, 0]
        for u in vivants:
            parts[C.couloir(u["Movement"][0])] += 1
        ys = [u["Movement"][1] for u in vivants]
        avancee = sum(ys) / n / (C.HAUTEUR - 2) if cote == "J" else sum(C.HAUTEUR - 1 - y for y in ys) / n / (C.HAUTEUR - 2)
        base = [min(n / 15, 1), sum(u["PV"] / C.UNITES[u["Nom"]]["pv"] for u in vivants) / n,
                min(avancee, 1), parts[0] / n, parts[1] / n, parts[2] / n]
    return base, vivants, min(pertes / 5, 1), attaque


def variables_tour(tour, mouvement, gestion):
    j, ia = tour["joueur"], tour["IA"]
    j_base, j_viv, j_pertes, j_att = resume_unites(j["unites"], "J")
    ia_base, ia_viv, ia_pertes, ia_att = resume_unites(ia["unites"], "IA")
    n = len(j_viv)
    types = [sum(1 for u in j_viv if u["Nom"] == t) / n if n else 0 for t in ("Piquier", "Archer", "Cavalier", "Ouvrier")]
    nb_eco = sum(1 for b in j["batiments"].values() if b["Nom"] in C.BATIMENTS_ECO and not b["destroy"])
    qg_j = next((b["PV"] for b in j["batiments"].values() if b["Nom"] == "QG"), 0)
    qg_ia = next((b["PV"] for b in ia["batiments"].values() if b["Nom"] == "QG"), 0)
    tour_ia = next((b["PV"] for b in ia["batiments"].values() if b["Nom"] == "Tour"), 0)
    obj = [1.0 if any(o["type"] == t for o in ia["Objectif"].values()) else 0.0 for t in C.TYPES_OBJECTIF]
    r = ia["ressources"]

    pos_j = [u["Movement"] for u in j_viv if u["Nom"] != "Ouvrier"]
    pos_ia = [u["Movement"] for u in ia_viv]
    if pos_j and pos_ia:
        dmin = min(dist(a, b) for a in pos_j for b in pos_ia)
        contact = sum(1 for a in pos_j if any(dist(a, b) <= 2 for b in pos_ia)) / len(pos_j)
    else:
        dmin, contact = C.HAUTEUR - 1, 0
    menace = min(sum(1 for b in pos_ia if b[1] <= 3) / 5, 1)

    actions = np.zeros(C.NB_MOUVEMENTS + C.NB_GESTIONS)
    actions[C.MOUVEMENTS.index(mouvement)] = 1
    actions[C.NB_MOUVEMENTS + C.GESTIONS.index(gestion)] = 1

    return (
        j_base + types + [j_pertes, j_att, nb_eco / C.MAX_ECO, qg_j / C.BATIMENTS["QG"]]
        + ia_base + [ia_pertes, ia_att, qg_ia / C.BATIMENTS["QG"], tour_ia / C.BATIMENTS["Tour"]]
        + [min(r[k] / 150, 1) for k in C.RESSOURCES] + obj
        + [dmin / (C.HAUTEUR - 1), contact, menace, int(tour["_num"]) / C.MAX_TOURS]
        + list(actions)
    )


def main():
    C.DOSSIER_PROCESSED.mkdir(parents=True, exist_ok=True)
    C.DOSSIER_RESULTATS.mkdir(parents=True, exist_ok=True)

    verite = {}
    with open(C.DOSSIER_RAW / "verite_terrain.csv", encoding="utf-8") as f:
        for l in csv.DictReader(f):
            verite[(l["joueur"], int(l["partie"]), int(l["tour"]))] = (l["mouvement"], l["gestion"])
    styles = {}
    with open(C.DOSSIER_RAW / "profils_simules.csv", encoding="utf-8") as f:
        for l in csv.DictReader(f):
            styles[l["joueur"]] = C.STYLES.index(l["style"])

    qualite, concordance = Counter(), Counter()
    tailles_joueurs, tailles_parties = [], []
    lignes = defaultdict(list)
    sorties = {k: [] for k in ("X", "y_mvt", "y_ges", "id_partie", "id_joueur", "num_partie", "style", "tour")}
    positions_par_style = np.zeros((len(C.STYLES), C.HAUTEUR, C.LARGEUR))
    nb_partie_global = 0
    joueurs_vus = []
    ent = {k: [] for k in ("u_ligne", "u_var", "u_masque", "u_delta", "u_attaque", "u_construct", "u_ouvrier",
                           "u_pos", "u_arrivee", "c_unite", "c_var", "b_ligne", "b_var", "b_entraine")}

    with gzip.open(C.DOSSIER_RAW / "joueurs.jsonl.gz", "rt", encoding="utf-8") as f, \
            gzip.open(C.DOSSIER_PROCESSED / "mongo_parties.jsonl.gz", "wt", encoding="utf-8") as sortie_mongo:
        for ligne in f:                                            # ---------- EXTRACT (1 joueur)
            tailles_joueurs.append(len(ligne.encode()))
            doc = json.loads(ligne)
            (id_brut, parties), = doc.items()
            joueur = pseudonymiser(id_brut)                         # ---------- RGPD
            index_joueur = len(joueurs_vus)
            joueurs_vus.append(joueur)
            style = styles[id_brut]
            historique = np.zeros(C.NB_MOUVEMENTS + C.NB_GESTIONS)
            for nb_prec, cle_partie in enumerate(sorted(parties, key=numero)):
                num_partie = numero(cle_partie)
                tours = parties[cle_partie]
                tailles_parties.append(len(json.dumps(tours).encode()))
                qualite["parties_lues"] += 1
                numeros = sorted(numero(k) for k in tours)
                manquants = set(range(1, numeros[-1] + 1)) - set(numeros)
                qualite["tours_manquants_reperes"] += len(manquants)

                lignes_partie, etiquettes_partie, tours_mongo = [], [], []
                for n in numeros:
                    tour = tours[f"Tour_{n}"]
                    tour["_num"] = n
                    for camp in ("joueur", "IA"):                   # ---------- QUALITÉ
                        for u in tour[camp]["unites"].values():
                            nettoyer_unite(u, qualite)
                    mvt = etiquette_mouvement(tour["joueur"]["unites"])   # ---------- ÉTIQUETTES
                    ges = etiquette_gestion(tour["joueur"]["unites"], tour["joueur"]["batiments"])
                    vrai = verite[(id_brut, num_partie, n)]
                    concordance["mvt_ok" if mvt == vrai[0] else "mvt_ko"] += 1
                    concordance["ges_ok" if ges == vrai[1] else "ges_ko"] += 1
                    lignes_partie.append(variables_tour(tour, mvt, ges))   # ---------- VARIABLES
                    etiquettes_partie.append((n, mvt, ges))
                    for u in tour["joueur"]["unites"].values():
                        if not u["Destroy"] and u["Nom"] != "Ouvrier":
                            positions_par_style[style, u["Movement"][1], u["Movement"][0]] += 1
                    qualite["tours_valides"] += 1

                    def liste(d):
                        return [{"id": k, **{c: v for c, v in x.items()}} for k, x in d.items()]
                    tours_mongo.append({                           # ---------- LOAD (un tour)
                        "tour": n,
                        "joueur_etat": {"unites": liste(tour["joueur"]["unites"]),
                                        "batiments": liste(tour["joueur"]["batiments"])},
                        "ia_etat": {"ressources": tour["IA"]["ressources"],
                                    "objectifs": liste(tour["IA"]["Objectif"]),
                                    "unites": liste(tour["IA"]["unites"]),
                                    "batiments": liste(tour["IA"]["batiments"])},
                        "mouvement": mvt, "gestion": ges,
                    })

                # ---------- LOAD (schéma optimisé) : 1 document = 1 partie
                # La partie est la maille de lecture : le modèle lit une fenêtre de tours,
                # le profil lit les parties précédentes, personne ne lit un tour isolé.
                # C'est aussi un seau naturellement borné (MAX_TOURS tours au plus), ce qui
                # garde le document très loin de la limite de 16 Mo de MongoDB. Une seule
                # écriture en fin de partie : le document n'est jamais réécrit tour par tour.
                sortie_mongo.write(json.dumps({
                    "joueur": joueur, "partie": num_partie,
                    "nb_tours": len(tours_mongo), "tours": tours_mongo,
                }, ensure_ascii=False) + "\n")

                # ---------- PROFIL : uniquement les parties précédentes
                total = historique.sum()
                profil = np.concatenate([
                    historique[:C.NB_MOUVEMENTS] / max(historique[:C.NB_MOUVEMENTS].sum(), 1),
                    historique[C.NB_MOUVEMENTS:] / max(historique[C.NB_MOUVEMENTS:].sum(), 1),
                    [min(nb_prec / 10, 1)]]) if total else np.zeros(C.NB_MOUVEMENTS + C.NB_GESTIONS + 1)

                for i, (n, mvt, ges) in enumerate(etiquettes_partie):
                    suivant = etiquettes_partie[i + 1] if i + 1 < len(etiquettes_partie) else None
                    connu = suivant is not None and suivant[0] == n + 1   # cible inconnue après un tour manquant
                    sorties["X"].append(np.concatenate([lignes_partie[i], profil]))
                    sorties["y_mvt"].append(C.MOUVEMENTS.index(suivant[1]) if connu else -1)
                    sorties["y_ges"].append(C.GESTIONS.index(suivant[2]) if connu else -1)
                    sorties["id_partie"].append(nb_partie_global)
                    sorties["id_joueur"].append(index_joueur)
                    sorties["num_partie"].append(num_partie)
                    sorties["style"].append(style)
                    sorties["tour"].append(n)
                    if connu:                                       # ---------- NIVEAU 2 : unité par unité
                        ligne_id = len(sorties["X"]) - 1
                        unites, bats = E.extraire(tours[f"Tour_{n}"], tours[f"Tour_{n + 1}"])
                        for u in unites:
                            idx_u = len(ent["u_ligne"])
                            ent["u_ligne"].append(ligne_id)
                            ent["u_var"].append(u["variables"])
                            ent["u_masque"].append(u["masque"])
                            ent["u_delta"].append(u["delta"])
                            ent["u_attaque"].append(u["attaque"])
                            ent["u_construct"].append(u["construct"])
                            ent["u_ouvrier"].append(u["Nom"] == "Ouvrier")
                            ent["u_pos"].append(u["pos"])
                            ent["u_arrivee"].append(u["arrivee"])
                            for _, v in u["candidats"]:
                                ent["c_unite"].append(idx_u)
                                ent["c_var"].append(v)
                        for b in bats:
                            ent["b_ligne"].append(ligne_id)
                            ent["b_var"].append(b["variables"])
                            ent["b_entraine"].append(b["entraine"])
                    historique[C.MOUVEMENTS.index(mvt)] += 1
                    historique[C.NB_MOUVEMENTS + C.GESTIONS.index(ges)] += 1
                nb_partie_global += 1

    X = np.array(sorties["X"], dtype=np.float32)
    assert X.shape[1] == NB_VARIABLES, X.shape
    assert X.min() >= 0 and X.max() <= 1, (X.min(), X.max())
    np.savez_compressed(
        C.DOSSIER_PROCESSED / "dataset.npz", X=X,
        **{k: np.array(v) for k, v in sorties.items() if k != "X"},
        positions_par_style=positions_par_style,
    )
    np.savez_compressed(
        C.DOSSIER_PROCESSED / "entites.npz",
        **{k: np.array(v, dtype=np.float32 if k.endswith("_var") else None) for k, v in ent.items()})
    qualite["unites_a_predire"] = len(ent["u_ligne"])
    qualite["batiments_formateurs_a_predire"] = len(ent["b_ligne"])
    qualite["deplacements_hors_masque"] = int(sum(not m[d] for m, d in zip(ent["u_masque"], ent["u_delta"])))
    qualite["attaques_hors_candidats"] = int(sum(a < 0 for a in ent["u_attaque"]))
    with open(C.DOSSIER_PROCESSED / "variables.json", "w", encoding="utf-8") as f:
        json.dump(NOMS_VARIABLES, f, indent=1, ensure_ascii=False)

    # ---------------------------------------------------------- Rapports
    y_mvt = np.array(sorties["y_mvt"])
    qualite["lignes_sans_cible_tour_suivant"] = int((y_mvt < 0).sum())
    qualite["joueurs_pseudonymises"] = len(joueurs_vus)
    qualite["concordance_regle_mouvement"] = round(concordance["mvt_ok"] / (concordance["mvt_ok"] + concordance["mvt_ko"]), 4)
    qualite["concordance_regle_gestion"] = round(concordance["ges_ok"] / (concordance["ges_ok"] + concordance["ges_ko"]), 4)
    with open(C.DOSSIER_RESULTATS / "rapport_qualite.json", "w", encoding="utf-8") as f:
        json.dump(dict(qualite), f, indent=2, ensure_ascii=False)

    nb_parties, nb_tours = qualite["parties_lues"], qualite["tours_valides"]
    octets_partie = sum(tailles_parties) / nb_parties
    octets_tour = sum(tailles_parties) / nb_tours
    parties_par_mois = 20
    projection = {}
    for nb_j in (1_000, 10_000, 100_000):
        o = nb_j * parties_par_mois * octets_partie
        projection[f"{nb_j}_joueurs"] = {
            "volume_mensuel_Go": round(o / 1e9, 2), "volume_annuel_Go": round(12 * o / 1e9, 1),
            "tours_par_mois_millions": round(nb_j * parties_par_mois * nb_tours / nb_parties / 1e6, 1)}
    volumetrie = {
        "simulation": {
            "joueurs": len(joueurs_vus), "parties": nb_parties, "tours": nb_tours,
            "volume_brut_Mo": round(sum(tailles_joueurs) / 1e6, 1),
            "Ko_par_tour": round(octets_tour / 1e3, 2), "Ko_par_partie": round(octets_partie / 1e3, 1),
            "tours_moyens_par_partie": round(nb_tours / nb_parties, 1),
        },
        "schema_source_1_document_par_joueur": {
            "Ko_moyen_par_joueur_10_parties": round(sum(tailles_joueurs) / len(tailles_joueurs) / 1e3),
            "Ko_max_par_joueur_10_parties": round(max(tailles_joueurs) / 1e3),
            "parties_avant_limite_16_Mo": int(16e6 // octets_partie),
            "remarque": "Le document grossit à chaque partie : limite MongoDB de 16 Mo atteinte pour les joueurs réguliers.",
        },
        "hypothese_projection": f"{parties_par_mois} parties par joueur et par mois",
        "projection": projection,
        "velocite": {
            "ecritures_par_partie": round(nb_tours / nb_parties, 1),
            "ecritures_par_seconde_1000_joueurs_simultanes_1_tour_30s": round(1000 / 30, 1),
            "reentrainement": "après chaque partie (incrémental) + complet chaque nuit",
        },
    }
    with open(C.DOSSIER_RESULTATS / "volumetrie.json", "w", encoding="utf-8") as f:
        json.dump(volumetrie, f, indent=2, ensure_ascii=False)

    print(json.dumps(dict(qualite), indent=2, ensure_ascii=False))
    print(json.dumps(volumetrie["simulation"], indent=2, ensure_ascii=False))
    print(json.dumps(volumetrie["schema_source_1_document_par_joueur"], indent=2, ensure_ascii=False))
    print(f"Dataset : {X.shape[0]} tours x {X.shape[1]} variables -> {C.DOSSIER_PROCESSED / 'dataset.npz'}")


if __name__ == "__main__":
    main()
