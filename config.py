"""
Paramètres communs à tout le projet (simulation, ETL, entraînement).
Modifier ici plutôt que dans les scripts (ex. : noms des unités et bâtiments du GDD).
"""
from pathlib import Path

RACINE = Path(__file__).resolve().parent
DOSSIER_RAW = RACINE / "data" / "raw"
DOSSIER_PROCESSED = RACINE / "data" / "processed"
DOSSIER_MODELES = RACINE / "modeles"
DOSSIER_RESULTATS = RACINE / "resultats"
DOSSIER_FIGURES = DOSSIER_RESULTATS / "figures"

GRAINE = 42  # reproductibilité : mêmes données et mêmes résultats à chaque lancement

# --- Plateau (version simplifiée de la carte Mechavalon) -------------------
LARGEUR = 12          # colonnes x = 0..11
HAUTEUR = 12          # lignes   y = 0..11 (base joueur en y=0, base IA en y=11)
MAX_TOURS = 100       # match nul après 100 tours (règle du GDD)


def couloir(x: float) -> int:
    """Couloirs : gauche x 0-3, centre x 4-7, droite x 8-11."""
    return 0 if x < 3.5 else (1 if x < 7.5 else 2)


CENTRE_COULOIR = {0: 2, 1: 6, 2: 9}

# --- Unités et bâtiments (noms à aligner sur le GDD) -------------------------
UNITES = {
    #            PV max, portée d'attaque, déplacement par tour, dégâts (min, max)
    "Piquier":  {"pv": 10, "portee": 1, "mvt": 1, "degats": (3, 4)},
    "Archer":   {"pv": 7,  "portee": 2, "mvt": 1, "degats": (2, 3)},
    "Cavalier": {"pv": 12, "portee": 1, "mvt": 2, "degats": (3, 5)},
    "Ouvrier":  {"pv": 5,  "portee": 0, "mvt": 1, "degats": (0, 0)},
}
TYPES_COMBAT = ["Piquier", "Archer", "Cavalier"]
BATIMENTS = {"QG": 30, "Caserne": 20, "Scierie": 12, "Mine": 12, "Raffinerie": 12, "Tour": 15}
BATIMENTS_ECO = ["Scierie", "Mine", "Raffinerie"]
RESSOURCES = ["or", "bois", "pierre", "fer", "carburant"]
TYPES_OBJECTIF = ["attaque", "defense", "harcelement"]

# --- Ce que l'IA prédictive cherche à deviner (au tour suivant) --------------
# Deux sorties, car un même tour mélange déplacement des unités et gestion de la base.
MOUVEMENTS = ["AVANCER_GAUCHE", "AVANCER_CENTRE", "AVANCER_DROITE", "ATTAQUER", "DEFENDRE"]
GESTIONS = ["RIEN", "ENTRAINER", "CONSTRUIRE"]
NB_MOUVEMENTS = len(MOUVEMENTS)
NB_GESTIONS = len(GESTIONS)

STYLES = ["agressif", "defensif", "flanqueur", "economiste", "opportuniste"]

# --- Volume de la simulation -------------------------------------------------
NB_JOUEURS = 150
PARTIES_PAR_JOUEUR = 10

# --- Économie (ressources du joueur cachées : brouillard de guerre) ----------
COUT_UNITE = 6
COUT_BATIMENT = 8
MAX_ECO = 5
