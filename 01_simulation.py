"""
ÉTAPE 1 — SIMULATION DE PARTIES (au format de données Mechavalon)

Le jeu n'étant pas jouable, on génère des parties avec des joueurs simulés.
Chaque joueur virtuel a une "personnalité" stable d'une partie à l'autre
(style, couloir préféré, unités préférées, rythme d'entraînement, part de hasard) :
c'est exactement ce que l'IA prédictive doit apprendre.

Sortie : data/raw/joueurs.jsonl.gz
    1 ligne = 1 document = 1 joueur, AU FORMAT DÉFINI DANS LE PROJET :
    { "<ID_Joueur>": { "<Partie_X>": { "<Tour_X>": {
        "joueur": { "unites":    { "<Unite_X>": {Nom, PV, position, Movement, Attaque, Destroy, construct} },
                    "batiments": { "<Bat_X>":   {Nom, PV, position, Attaque, destroy, entraine} } },
        "IA":     { "ressources": {or, bois, pierre, fer, carburant},
                    "Objectif":   { "<marqueur_X>": {type, Cible, Priorite} },
                    "unites": {...}, "batiments": {...} } } } } }

    Signification des champs (un enregistrement = ce qui s'est passé pendant le tour) :
      position  = case [x, y] au début du tour       Movement = case [x, y] à la fin du tour
      Attaque   = id de la cible attaquée, sinon null Destroy  = true si l'unité est tuée pendant le tour
      construct = bâtiment que l'unité construit, sinon null
      entraine  = type d'unité que le bâtiment entraîne, sinon null
    Les ressources du joueur ne sont pas enregistrées (brouillard de guerre).
    Quelques erreurs réalistes sont injectées : l'ETL (étape 2) doit les corriger.

Sorties de contrôle (connues seulement parce qu'on simule, jamais données au modèle) :
    data/raw/verite_terrain.csv  intention réelle de chaque tour (pour vérifier les règles de l'ETL)
    data/raw/profils_simules.csv style de chaque joueur (pour analyser les erreurs par style)
"""
import csv
import gzip
import json
import random

import config as C

U = C.UNITES


def distance(a, b):
    """Distance entre deux cases (approximation simple d'une grille hexagonale)."""
    return max(abs(a["x"] - b["x"]), abs(a["y"] - b["y"]))


def vers(u, cx, cy, pas):
    """Déplace l'unité u de `pas` cases vers (cx, cy)."""
    for _ in range(pas):
        u["x"] = max(0, min(C.LARGEUR - 1, u["x"] + (cx > u["x"]) - (cx < u["x"])))
        u["y"] = max(1, min(C.HAUTEUR - 2, u["y"] + (cy > u["y"]) - (cy < u["y"])))


# ---------------------------------------------------------------------------
# Personnalité d'un joueur simulé
# ---------------------------------------------------------------------------
def creer_profil(rng, index):
    style = rng.choice(C.STYLES)
    preferences = {
        "agressif": [2, 1, 4], "defensif": [2, 4, 1], "flanqueur": [1, 2, 4],
        "economiste": [3, 2, 2], "opportuniste": [2, 2, 3],
    }[style]
    return {
        "id": f"J{index:04d}",
        "style": style,
        "couloir_prefere": rng.choice([0, 2]) if style == "flanqueur" else rng.choice([0, 1, 1, 2]),
        "unites_preferees": [p * rng.uniform(0.6, 1.4) for p in preferences],  # Piquier, Archer, Cavalier
        "rythme_entrainement": rng.randint(3, 6),     # entraîne une unité tous les k tours
        "ouverture": rng.randint(8, 15) if style == "defensif" else rng.randint(2, 5),
        "longueur_vague": rng.randint(2, 5),          # avance n tours puis attaque
        "reactivite": {"agressif": 0.95, "defensif": 0.55, "flanqueur": 0.8,
                       "economiste": 0.6, "opportuniste": 0.9}[style],
        "eco_vise": {"agressif": 1, "defensif": 2, "flanqueur": 1, "economiste": 3, "opportuniste": 2}[style],
        "bruit": round(rng.uniform(0.05, 0.20), 2),   # part d'imprévisibilité
    }


# ---------------------------------------------------------------------------
# Une partie
# ---------------------------------------------------------------------------
class Partie:
    def __init__(self, rng, profil):
        self.rng, self.p = rng, profil
        self.tour = 0
        self.n = 0
        self.difficulte_ia = rng.uniform(0.85, 1.15)
        self.j_unites, self.ia_unites = {}, {}
        self.j_bats = {"J_B1": {"Nom": "QG", "PV": 30, "x": 6, "y": 0},
                       "J_B2": {"Nom": "Caserne", "PV": 20, "x": 4, "y": 0}}
        self.ia_bats = {"IA_B1": {"Nom": "QG", "PV": 30, "x": 6, "y": 11},
                        "IA_B2": {"Nom": "Caserne", "PV": 20, "x": 8, "y": 11},
                        "IA_B3": {"Nom": "Tour", "PV": 15, "x": 4, "y": 11}}
        for nom in ("Piquier", "Piquier", "Ouvrier"):
            self._creer("J", nom)
        for nom in ("Piquier", "Archer"):
            self._creer("IA", nom)
        self.j_or = 10                       # ressource cachée, jamais enregistrée
        self.ia_res = {"or": 10, "bois": 5, "pierre": 5, "fer": 3, "carburant": 2}
        self.a_venir = []                    # unités / bâtiments qui apparaissent au tour suivant
        self.vague = 0
        self.dernier_entrainement = -99
        self.bascule = profil["ouverture"] + rng.randint(-1, 1)

    def _creer(self, camp, nom, x=None, y=None):
        self.n += 1
        if x is None:
            x, y = (5 + self.rng.randint(-1, 1), 1) if camp == "J" else (7 + self.rng.randint(-1, 1), C.HAUTEUR - 2)
        unite = {"Nom": nom, "PV": U[nom]["pv"], "x": x, "y": y}
        (self.j_unites if camp == "J" else self.ia_unites)[f"{camp}_U{self.n}"] = unite

    def combattants(self, camp):
        d = self.j_unites if camp == "J" else self.ia_unites
        return {k: u for k, u in d.items() if u["Nom"] != "Ouvrier"}

    # ---- attaque d'une unité (avec riposte automatique) ----------------------
    def attaquer(self, uid, u, cibles_u, cibles_b, bonus=1.0):
        portee = U[u["Nom"]]["portee"]
        a_portee = [(k, c) for k, c in cibles_u.items() if c["PV"] > 0 and distance(u, c) <= portee]
        if a_portee:
            k, c = min(a_portee, key=lambda kc: kc[1]["PV"])
            c["PV"] -= round(self.rng.randint(*U[u["Nom"]]["degats"]) * bonus)
            if c["PV"] > 0 and distance(u, c) <= U[c["Nom"]]["portee"]:
                u["PV"] -= self.rng.randint(1, 2)          # riposte
            return k
        b_portee = [(k, b) for k, b in cibles_b.items() if b["PV"] > 0 and distance(u, b) <= portee]
        if b_portee:
            k, b = min(b_portee, key=lambda kb: (kb[1]["Nom"] != "QG", kb[1]["PV"]))
            b["PV"] -= round(self.rng.randint(*U[u["Nom"]]["degats"]) * bonus)
            return k
        return None

    # ---- décision du joueur simulé : mouvement + gestion ---------------------
    def decider(self):
        p, rng, t = self.p, self.rng, self.tour
        comb = self.combattants("J")
        menace = any(distance(u, e) <= 2 for u in comb.values() for e in self.ia_unites.values())
        menace_qg = sum(1 for e in self.ia_unites.values() if e["y"] <= 3)
        couloir = C.MOUVEMENTS[p["couloir_prefere"]]
        style = p["style"]

        # -- mouvement --
        if rng.random() < p["bruit"]:
            mouvement = rng.choice(C.MOUVEMENTS)
        elif menace_qg >= 2 and self.j_bats["J_B1"]["PV"] < 20 and rng.random() < 0.8:
            mouvement = "DEFENDRE"
        elif menace and rng.random() < p["reactivite"]:
            mouvement = "ATTAQUER"
        elif style in ("defensif", "economiste") and t < self.bascule + (6 if style == "economiste" else 0):
            mouvement = "DEFENDRE"
        elif style == "opportuniste":
            parts = [0, 0, 0]
            for e in self.ia_unites.values():
                parts[C.couloir(e["x"])] += 1
            faible = min(range(3), key=lambda i: (parts[i], abs(i - 1)))
            mouvement = self._vague(C.MOUVEMENTS[faible])
        else:
            mouvement = self._vague(couloir)

        # -- gestion --
        nb_eco = sum(1 for b in self.j_bats.values() if b["Nom"] in C.BATIMENTS_ECO)
        en_cours = sum(1 for a in self.a_venir if a[0] == "bat")
        if rng.random() < p["bruit"] / 2:
            gestion = rng.choice(C.GESTIONS)
        elif nb_eco + en_cours < p["eco_vise"] and self.j_or >= C.COUT_BATIMENT and t >= 2:
            gestion = "CONSTRUIRE"
        elif t - self.dernier_entrainement >= p["rythme_entrainement"] and self.j_or >= C.COUT_UNITE:
            gestion = "ENTRAINER"
        else:
            gestion = "RIEN"
        return mouvement, gestion

    def _vague(self, couloir):
        """Avance pendant 'longueur_vague' tours, puis attaque."""
        return "ATTAQUER" if self.vague >= self.p["longueur_vague"] else couloir

    # ---- un tour complet -----------------------------------------------------
    def jouer_tour(self):
        rng = self.rng
        self.tour += 1
        for genre, camp, nom, x, y in self.a_venir:          # apparitions prévues au tour précédent
            if genre == "unite":
                self._creer(camp, nom, x, y)
            else:
                self.n += 1
                self.j_bats[f"J_B{self.n}"] = {"Nom": nom, "PV": C.BATIMENTS[nom], "x": x, "y": y}
        self.a_venir = []

        depart = {k: (u["x"], u["y"]) for k, u in {**self.j_unites, **self.ia_unites}.items()}
        attaque, construit, entraine = {}, {}, {}
        mouvement, gestion = self.decider()

        # -- gestion de la base du joueur --
        ouvriers = [k for k, u in self.j_unites.items() if u["Nom"] == "Ouvrier"]
        if gestion == "CONSTRUIRE" and not ouvriers:
            gestion, type_unite, batiment = "ENTRAINER", "Ouvrier", "J_B1"   # le QG forme un ouvrier
        else:
            poids = self.p["unites_preferees"]
            type_unite, batiment = rng.choices(C.TYPES_COMBAT, weights=poids)[0], "J_B2"
        if gestion == "ENTRAINER" and self.j_or >= C.COUT_UNITE:
            self.j_or -= C.COUT_UNITE
            entraine[batiment] = type_unite
            self.dernier_entrainement = self.tour
            self.a_venir.append(("unite", "J", type_unite, 4 + rng.randint(0, 2), 1))
        elif gestion == "CONSTRUIRE" and self.j_or >= C.COUT_BATIMENT and \
                sum(1 for b in self.j_bats.values() if b["Nom"] in C.BATIMENTS_ECO) < C.MAX_ECO:
            self.j_or -= C.COUT_BATIMENT
            existants = [b["Nom"] for b in self.j_bats.values()]
            nom = next((b for b in C.BATIMENTS_ECO if b not in existants), "Scierie")
            construit[ouvriers[0]] = nom
            emplacements = [(1, 0), (9, 0), (11, 0), (2, 0), (10, 0)]
            libres = [e for e in emplacements if all((b["x"], b["y"]) != e for b in self.j_bats.values())]
            x, y = libres[0] if libres else (0, 0)
            self.a_venir.append(("bat", "J", nom, x, y))
        else:
            gestion = "RIEN"

        # -- mouvement des unités de combat du joueur --
        for k, u in self.combattants("J").items():
            mvt = U[u["Nom"]]["mvt"]
            if mouvement == "ATTAQUER":
                cible = self.attaquer(k, u, self.ia_unites, self.ia_bats)
                if cible is None:
                    proche_qg = u["y"] >= C.HAUTEUR - 4 or not self.ia_unites
                    but = self.ia_bats["IA_B1"] if proche_qg else \
                        min(self.ia_unites.values(), key=lambda e: distance(u, e))
                    vers(u, but["x"], but["y"], mvt)
                    cible = self.attaquer(k, u, self.ia_unites, self.ia_bats)
                if cible:
                    attaque[k] = cible
            elif mouvement == "DEFENDRE":
                if u["y"] > 2:
                    vers(u, 6, 1, 1)
            else:
                cx = C.CENTRE_COULOIR[C.MOUVEMENTS.index(mouvement)]
                for _ in range(mvt):
                    if abs(u["x"] - cx) > 1:
                        vers(u, cx, u["y"], 1)
                    else:
                        vers(u, u["x"], C.HAUTEUR - 2, 1)
        self.vague = self.vague + 1 if mouvement.startswith("AVANCER") else 0
        self.j_or += 2 + 2 * sum(1 for b in self.j_bats.values() if b["Nom"] in C.BATIMENTS_ECO)

        # -- tour de l'IA tactique (adversaire scripté) --
        objectifs = self.jouer_ia(attaque, entraine)

        # -- enregistrement du tour au format du projet --
        enreg = self.enregistrer(depart, attaque, construit, entraine, objectifs)
        fin = None
        if self.ia_bats["IA_B1"]["PV"] <= 0:
            fin = "victoire_joueur"
        elif self.j_bats["J_B1"]["PV"] <= 0:
            fin = "victoire_ia"
        for camp in (self.j_unites, self.ia_unites):
            for k in [k for k, u in camp.items() if u["PV"] <= 0]:
                del camp[k]
        for camp in (self.j_bats, self.ia_bats):
            for k in [k for k, b in camp.items() if b["PV"] <= 0 and b["Nom"] != "QG"]:
                del camp[k]
        return enreg, (mouvement, gestion), fin

    def jouer_ia(self, attaque, entraine):
        rng = self.rng
        self.ia_res["or"] += 3
        for r in ("bois", "pierre", "fer", "carburant"):
            self.ia_res[r] += rng.randint(0, 2)
        j_vivants = {k: u for k, u in self.j_unites.items() if u["PV"] > 0}
        objectifs = {"M1": {"type": "attaque", "Cible": [6, 0], "Priorite": 2}}
        avances = [u for u in j_vivants.values() if u["y"] >= 7]
        if avances:
            objectifs["M2"] = {"type": "defense",
                               "Cible": [round(sum(u["x"] for u in avances) / len(avances)),
                                         round(sum(u["y"] for u in avances) / len(avances))],
                               "Priorite": 1}
        faibles = [(k, u) for k, u in j_vivants.items() if u["PV"] <= 0.4 * U[u["Nom"]]["pv"]]
        if faibles:
            k, u = faibles[0]
            objectifs["M3"] = {"type": "harcelement", "Cible": k, "Priorite": 3}

        if self.tour % 4 == 0 and self.ia_res["or"] >= C.COUT_UNITE and len(self.ia_unites) < 10:
            self.ia_res["or"] -= C.COUT_UNITE
            nom = "Archer" if (self.tour // 4) % 2 else "Piquier"
            entraine["IA_B2"] = nom
            self.a_venir.append(("unite", "IA", nom, 7 + rng.randint(-1, 1), C.HAUTEUR - 2))

        for k, e in list(self.ia_unites.items()):
            if e["PV"] <= 0:
                continue
            cible = self.attaquer(k, e, j_vivants, self.j_bats, self.difficulte_ia)
            if cible is None:
                if "M2" in objectifs and rng.random() < 0.7:
                    tx, ty = objectifs["M2"]["Cible"]
                elif j_vivants and rng.random() < 0.6:
                    v = min(j_vivants.values(), key=lambda u: distance(u, e))
                    tx, ty = v["x"], v["y"]
                else:
                    tx, ty = 6, 0
                vers(e, tx, ty, U[e["Nom"]]["mvt"])
                cible = self.attaquer(k, e, j_vivants, self.j_bats, self.difficulte_ia)
            if cible:
                attaque[k] = cible
        tour_def = self.ia_bats.get("IA_B3")
        if tour_def and tour_def["PV"] > 0:
            proches = [(k, u) for k, u in j_vivants.items() if u["PV"] > 0 and distance(u, tour_def) <= 2]
            if proches:
                k, u = proches[0]
                u["PV"] -= 2
                attaque["IA_B3"] = k
        return objectifs

    def enregistrer(self, depart, attaque, construit, entraine, objectifs):
        def unites(d):
            return {k: {"Nom": u["Nom"], "PV": max(u["PV"], 0), "position": list(depart.get(k, (u["x"], u["y"]))),
                        "Movement": [u["x"], u["y"]], "Attaque": attaque.get(k), "Destroy": u["PV"] <= 0,
                        "construct": construit.get(k)} for k, u in d.items()}

        def batiments(d):
            return {k: {"Nom": b["Nom"], "PV": max(b["PV"], 0), "position": [b["x"], b["y"]],
                        "Attaque": attaque.get(k), "destroy": b["PV"] <= 0, "entraine": entraine.get(k)}
                    for k, b in d.items()}

        return {
            "joueur": {"unites": unites(self.j_unites), "batiments": batiments(self.j_bats)},
            "IA": {"ressources": dict(self.ia_res), "Objectif": objectifs,
                   "unites": unites(self.ia_unites), "batiments": batiments(self.ia_bats)},
        }


# ---------------------------------------------------------------------------
# Défauts injectés volontairement (pour que l'ETL ait un vrai travail de qualité)
# ---------------------------------------------------------------------------
def abimer(rng, tours):
    for nom_tour in list(tours):
        if rng.random() < 0.003:
            del tours[nom_tour]                                  # tour perdu (coupure réseau)
            continue
        for u in tours[nom_tour]["joueur"]["unites"].values():
            r = rng.random()
            if r < 0.003:
                u["PV"] = rng.choice([-5, 99])                   # valeur aberrante
            elif r < 0.006:
                del u["Movement"]                                # champ manquant
            elif r < 0.008:
                u["Movement"] = [u["position"][0], min(u["position"][1] + 7, 11)]   # téléportation


def main():
    rng = random.Random(C.GRAINE)
    C.DOSSIER_RAW.mkdir(parents=True, exist_ok=True)
    nb_tours, resultats = 0, {}
    with gzip.open(C.DOSSIER_RAW / "joueurs.jsonl.gz", "wt", encoding="utf-8") as f_json, \
            open(C.DOSSIER_RAW / "verite_terrain.csv", "w", newline="", encoding="utf-8") as f_vt, \
            open(C.DOSSIER_RAW / "profils_simules.csv", "w", newline="", encoding="utf-8") as f_pr:
        vt, pr = csv.writer(f_vt), csv.writer(f_pr)
        vt.writerow(["joueur", "partie", "tour", "mouvement", "gestion"])
        pr.writerow(["joueur", "style"])
        for i in range(C.NB_JOUEURS):
            profil = creer_profil(rng, i + 1)
            pr.writerow([profil["id"], profil["style"]])
            doc = {}
            for k in range(1, C.PARTIES_PAR_JOUEUR + 1):
                partie, tours = Partie(rng, profil), {}
                fin = None
                while fin is None and partie.tour < C.MAX_TOURS:
                    enreg, (mvt, ges), fin = partie.jouer_tour()
                    tours[f"Tour_{partie.tour}"] = enreg
                    vt.writerow([profil["id"], k, partie.tour, mvt, ges])
                resultats[fin or "nul"] = resultats.get(fin or "nul", 0) + 1
                nb_tours += len(tours)
                abimer(rng, tours)
                doc[f"Partie_{k}"] = tours
            f_json.write(json.dumps({profil["id"]: doc}, ensure_ascii=False) + "\n")

    print(f"{C.NB_JOUEURS * C.PARTIES_PAR_JOUEUR} parties simulées, {nb_tours} tours. Résultats : {resultats}")
    print(f"-> {C.DOSSIER_RAW / 'joueurs.jsonl.gz'}")


if __name__ == "__main__":
    main()
