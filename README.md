# Mechavalon — IA prédictive (preuve de concept)

**Bloc RNCP40573 BC04B : construire et implémenter des modèles de big data et d'IA.**

Le jeu Mechavalon est en pause. Ce projet réalise **la partie IA de façon indépendante** : un LSTM
qui lit l'historique d'une partie **au format de données défini pour le jeu** et prédit ce que le
joueur fera au tour suivant. Il a été entraîné sur **des parties simulées**. Il travaille à deux
niveaux :

- **Niveau 1, les intentions** : la stratégie du tour (avancer à gauche, attaquer, entraîner…).
- **Niveau 2, le tour prédit** : pour chaque unité, sa case d'arrivée (`Movement`) et sa cible
  (`Attaque`), et pour chaque bâtiment, ce qu'il entraîne (`entraine`). Le tout dans le format du
  jeu, avec une probabilité pour chaque choix, et une **carte de présence prévue** pour le bot
  tactique.

Le bot tactique réagit à cette prédiction. Elle n'a donc pas besoin d'être parfaite : elle sert à
anticiper, pas à lire dans la tête du joueur.

> À dire clairement au jury : *les données sont simulées*. Le format, l'ETL, la base, le modèle
> et l'évaluation sont ceux qui tourneraient en production. Le jour où le jeu enverra ses
> vraies parties, seule l'étape 1 change.

---

## Auteurs

Projet de fin d'études Mechavalon, réalisé en binôme par Léo Langlois et
Killian Gautheur — Mastère Développeur Big Data et IA.

---

## 1. Lancer le projet

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate     |   Mac/Linux : source .venv/bin/activate
pip install -r requirements.txt

# Clé de pseudonymisation (voir §9). Sans elle, l'ETL retombe sur une valeur de
# développement : suffisant pour rejouer cette démonstration sur des données simulées,
# à ne jamais employer sur de vraies données de joueurs.
#   En générer une : python -c "import secrets; print(secrets.token_hex(32))"
#   Windows        : setx MECHAVALON_CLE_PSEUDO "<les 64 caractères>"   (puis rouvrir le terminal)
#   Mac / Linux    : export MECHAVALON_CLE_PSEUDO="<les 64 caractères>"

python 01_simulation.py       # ~15 s  : 150 joueurs × 10 parties, au format du jeu
python 02_etl.py              # ~35 s  : qualité, RGPD, étiquettes, variables
python 03_entrainement.py              # 3 à 5 min : niveau 1, baselines + 3 itérations du LSTM (CPU, sans GPU)
python 03b_entrainement_niveau2.py     # 4 à 6 min : niveau 2, le tour prédit unité par unité
python 04_visualisations.py            # ~5 s   : graphiques pour les slides
python 05_predire_tour.py              # ~30 s  : un tour prédit en JSON + sa figure (ce que ferait l'API)
```

Tout est reproductible (graine fixée à 42). Les résultats arrivent dans `resultats/` :
- `resultats.md` : tous les tableaux ;
- `figures/` : 8 graphiques au format 16:9 ;
- `metriques.json`, `volumetrie.json` et `rapport_qualite.json`.

Les noms d'unités et de bâtiments (Piquier, Archer, Cavalier, Ouvrier, QG, Caserne, Scierie,
Mine, Raffinerie, Tour) sont dans `config.py`. On peut les remplacer par ceux du GDD.

---

## 2. Les données d'entrée : le format du jeu

Un document par joueur, rangé par partie puis par tour. Un enregistrement de tour décrit **ce qui
s'est passé pendant ce tour** :

```json
{ "<ID_Joueur>": { "<Partie_X>": { "<Tour_X>": {
    "joueur": {
      "unites":    { "<Unite_X>": { "Nom", "PV", "position", "Movement", "Attaque", "Destroy", "construct" } },
      "batiments": { "<Bat_X>":   { "Nom", "PV", "position", "Attaque", "destroy", "entraine" } } },
    "IA": {
      "ressources": { "or", "bois", "pierre", "fer", "carburant" },
      "Objectif":   { "<marqueur_X>": { "type", "Cible", "Priorite" } },
      "unites": { ... }, "batiments": { ... } } } } } }
```

| Champ | Sens |
|---|---|
| `position` | case [x, y] au début du tour |
| `Movement` | case [x, y] à la fin du tour |
| `Attaque` | identifiant de la cible attaquée, sinon `null` |
| `Destroy` / `destroy` | `true` si l'unité ou le bâtiment est détruit pendant le tour |
| `construct` | bâtiment que l'unité construit, sinon `null` |
| `entraine` | type d'unité que le bâtiment entraîne, sinon `null` |

Les ressources du joueur ne sont **jamais** enregistrées : c'est le brouillard de guerre.

---

## 3. Niveau 1 : les intentions (deux sorties)

Dans un même tour, les unités bougent et attaquent, pendant que la caserne entraîne et qu'un
ouvrier construit. Le modèle a donc deux sorties, prédites ensemble pour le **tour suivant** :

| Sortie | Classes | Comment l'ETL la lit dans les données |
|---|---|---|
| **Mouvement** | avancer à gauche, au centre ou à droite, attaquer, défendre | Une unité de combat a une `Attaque` → attaquer. Sinon, des unités vont vers l'avant ou sur le côté → avancer, dans le couloir visé par les `Movement`. Sinon → défendre (repli ou maintien) |
| **Gestion** | rien, entraîner, construire | Un bâtiment a un `entraine` → entraîner. Sinon, une unité a un `construct` → construire. Sinon → rien |

Pourquoi des intentions plutôt que la case exacte de chaque unité ? Le nombre d'unités et la
taille de la carte changent. Une sortie « une case par unité » changerait de taille à chaque tour.
Ces classes forment un espace fixe. Et c'est ce dont l'IA tactique a besoin pour choisir sa
stratégie. Le détail unité par unité est le rôle du niveau 2.

## 3 bis. Niveau 2 : le tour suivant, unité par unité

Au début du tour, le plateau est connu : positions des unités des deux camps, PV, bâtiments. Pour
**chaque** unité et **chaque** bâtiment du joueur, le modèle prédit ce qu'il va faire :

| Champ du format | Ce que le modèle choisit | Comment |
|---|---|---|
| `Movement` | la case d'arrivée | parmi les déplacements possibles de -2 à +2, **masqués** selon la capacité de l'unité (1 case pour un piquier, 2 pour un cavalier) et les bords de la carte |
| `Attaque` | aucune, ou l'une des cibles à portée | chaque ennemi atteignable ce tour (distance ≤ déplacement + portée) reçoit un score. On ajoute un score « aucune », et le plus haut gagne |
| `construct` | rien / Scierie / Mine / Raffinerie | ouvriers uniquement |
| `entraine` | rien / Piquier / Archer / Cavalier / Ouvrier | QG et caserne |

**Comment gérer un nombre d'unités qui change à chaque tour ?** Le LSTM résume la partie en
32 nombres. Ce résumé est ensuite combiné avec la situation de chaque unité (18 variables : type,
PV, position, dernier déplacement, ennemi le plus proche…), et passe dans une petite tête de
prédiction. Cette même tête est appliquée à chaque unité, une par une. Qu'il y ait 2 ou 15
unités, le modèle ne change pas.

**La carte de présence prévue** additionne, case par case, les probabilités d'arrivée de toutes
les unités. Ton bot peut l'utiliser directement, par exemple pour éviter une case où le joueur a
80 % de chances d'arriver, ou pour y préparer une embuscade.

---

## 4. Le programme en un schéma

```
 01_simulation        02_etl                                 03_entrainement               04_visualisations
 joueurs.jsonl.gz ──▶ E : lecture en flux, 1 joueur à la fois ──▶ découpage par joueur ──────▶ figures 16:9
 (format du jeu)      T : qualité, RGPD, 2 étiquettes,          baselines vs LSTM
                          53 variables, profil joueur           3 itérations, métriques
                      L : dataset.npz + MongoDB 1 doc/partie    modeles/lstm_mechavalon.pt
```

**Le modèle :**

```
10 derniers tours × 53 variables ─▶ LSTM 64 ─▶ LSTM 32 ─┬─▶ Softmax 5 : mouvement au tour suivant       (niveau 1)
                                     (dropout 0.2)       ├─▶ Softmax 3 : gestion au tour suivant          (niveau 1)
                                                         │
                     pour chaque unité : résumé (32) + situation de l'unité (18) ─▶ MLP ─┬─▶ Movement : 25 déplacements masqués
                                                                                        ├─▶ Attaque : score de chaque cible + « aucune »
                                                                                        └─▶ construct (ouvriers)
                     pour chaque bâtiment : résumé (32) + situation du bâtiment (7) ─▶ MLP ─▶ entraine          (niveau 2)

Perte = somme des Cross-Entropy de toutes les sorties · Adam · rétropropagation dans le temps · arrêt précoce
```

---

## 5. Étape par étape, et ce que ça valide dans la grille

### Étape 1 : simulation (`01_simulation.py`)
150 joueurs jouent chacun 10 parties, soit environ 96 000 tours. Chaque joueur a une personnalité
stable :
- un **style** : agressif, défensif, flanqueur, économiste ou opportuniste ;
- un **couloir préféré** ;
- des **unités préférées** ;
- un **rythme d'entraînement** : il entraîne une unité tous les k tours ;
- un **nombre de bâtiments économiques visé** ;
- une **durée d'ouverture** et une **longueur de vague** ;
- une **part de hasard** de 5 à 20 %.

L'adversaire est une IA tactique scriptée. Elle se fixe des **Objectifs** (attaque, défense,
harcèlement), entraîne des unités, et sa tour de défense tire. Seule, elle gagne 44 % des parties.

Des défauts réalistes sont injectés pour que l'ETL ait un vrai travail :
- des tours perdus ;
- des PV aberrants ;
- des `Movement` manquants ;
- des déplacements impossibles (téléportations).

### Étape 2 : ETL (`02_etl.py`), compétence *exploiter les données (ETL + RGPD)*
| Phase | Ce qui est fait | Résultat (`rapport_qualite.json`) |
|---|---|---|
| Extract | Lecture en flux, un document joueur à la fois | 150 joueurs, 1 500 parties |
| RGPD | L'identifiant joueur est remplacé par un **HMAC-SHA256** avec clé secrète | 150 joueurs pseudonymisés |
| Qualité | PV ramenés dans les bornes du type d'unité. `Movement` manquant → l'unité n'a pas bougé. Déplacement plus long que la capacité de l'unité → corrigé. Tours manquants repérés | 980 PV, 1 010 `Movement`, 614 téléportations, 274 tours manquants |
| Étiquettes | Règles métier du §3 | gestion : **100 %** de concordance avec la vérité de la simulation, mouvement : **92 %** (voir ci-dessous) |
| Variables | 53 nombres entre 0 et 1 par tour (`data/processed/variables.json`) | |
| Profil | Habitudes du joueur sur ses **parties précédentes uniquement** | aucune fuite d'information |
| Load | `dataset.npz` + `mongo_parties.jsonl.gz` (un document par partie, schéma optimisé) | |

Les 8 % d'écart sur le mouvement ne sont pas des erreurs de la règle. La règle mesure ce que les
unités ont *réellement* fait. Par exemple, un joueur veut avancer mais ses unités sont déjà au
bout de la carte : elles ne bougent pas, et la règle note « défendre ». Ou un joueur ordonne
d'attaquer mais aucun ennemi n'est à portée : ses unités ne font que s'approcher.

**Les 53 variables d'un tour :**
- **Joueur (14) :** nombre d'unités, PV moyens, avancée, répartition gauche/centre/droite,
  part de piquiers, d'archers, de cavaliers et d'ouvriers, pertes du tour, part d'unités qui
  attaquent, bâtiments économiques, PV du QG.
- **IA (18) :** la même chose pour l'IA, plus les PV de sa tour, ses 5 ressources et ses
  objectifs actifs (attaque, défense, harcèlement).
- **Contexte (4) :** distance minimale entre les deux camps, part d'unités au contact, menace
  sur le QG, numéro du tour.
- **Ce que le joueur a fait pendant ce tour (8) :** le mouvement (5) et la gestion (3).
- **Profil (9) :** répartition de ses mouvements (5) et de sa gestion (3) sur ses parties
  précédentes, plus le nombre de parties déjà jouées.

### Étape 3 : entraînement (`03_entrainement.py`), compétences *concevoir et évaluer des modèles* et *schématiser un programme d'IA*

**Découpage par joueur** : 105 joueurs pour l'entraînement, 22 pour la validation, 23 pour le
test. Les joueurs du test n'ont jamais été vus.

**F1 macro sur le test** (chaque classe compte autant) :
| Modèle | Mouvement | Gestion |
|---|---|---|
| Classe majoritaire | 11 % | 28 % |
| Répéter le dernier tour | 45 % | 23 % |
| Markov ordre 1 | 32 % | 28 % |
| MLP sans mémoire | 71 % | 68 % |
| **LSTM v3** | **75 %** | **87 %** |

En exactitude, le LSTM v3 obtient :
- **mouvement : 76 %**, et la bonne réponse figure dans ses 2 premiers choix 92 fois sur 100
  (top-2), pour 20 % au hasard ;
- **gestion : 95 %**, pour 33 % au hasard.

**Itérations, avec les ajustements documentés :**
| Version | Ce qui a changé | Pourquoi | F1 mouvement | F1 gestion |
|---|---|---|---|---|
| v1 | Fenêtre de 5 tours | point de départ | 73 % | 77 % |
| v2 | Fenêtre de **10** tours | un rythme « entraîner tous les 6 tours » ne se voit pas sur 5 tours | 74 % | 86 % |
| v3 | + **profil joueur** + dropout 0.2 | utiliser les parties précédentes, limiter le sur-apprentissage | 75 % | 87 % |

**La preuve que la mémoire compte** : sur la gestion, le MLP sans mémoire plafonne à 68 % de
F1, contre 87 % pour le LSTM. Il voit l'état du tour, mais pas *depuis combien de tours* le
joueur n'a pas entraîné d'unité.

**Coût du modèle (éco-conception)** :
- **43 000 paramètres** (≈ 170 Ko) ;
- **≈ 3 min 20 d'entraînement sur le processeur d'un portable**, sans GPU ;
- **1,6 ms par prédiction**.

### Étape 3b : niveau 2 (`03b_entrainement_niveau2.py`)

Un seul modèle porte les deux niveaux : le même tronc LSTM, les têtes d'intention, et les têtes
par unité et par bâtiment (56 556 paramètres, ≈ 4 min 25 d'entraînement sur CPU, arrêt au plafond de 30 époques). Il est évalué
sur les mêmes 23 joueurs jamais vus, soit 15 287 tours et 52 416 unités.

**Movement : la case d'arrivée de chaque unité**
| Modèle | Case exacte (unités de combat) | Case exacte (toutes unités) | À une case près | Bonne case dans le top-3 |
|---|---|---|---|---|
| Règle « l'unité ne bouge pas » | 27 % | 44 % | 88 % | — |
| Règle « refait son dernier déplacement » | 35 % | 51 % | 81 % | — |
| **LSTM niveau 2** | **75 %** | **81 %** | **94 %** | **95,5 %** |

En moyenne, la case prédite est à 0,26 case de la vraie.

**Attaque : la cible de chaque unité**
| Modèle | Bonne décision (cible ou aucune) | Bonne cible quand le joueur attaque |
|---|---|---|
| Règle « attaque la cible la plus faible à portée » | 80 % | 46 % |
| **LSTM niveau 2** | **88,5 %** | **80 %** |

**entraine et construct**
- **Entraîner ou pas :** F1 de 90 %.
- **Type d'unité entraînée :** juste dans 38 % des cas. Dans la simulation, le type est tiré au
  hasard selon les préférences du joueur, donc imprévisible par nature. C'est une limite à
  assumer.
- **Construire ou pas** (ouvriers) : F1 de 78 %.

**Tour entièrement juste** (toutes les unités sur la bonne case, toutes les attaques et tous les
entraînements justes) : **52 % des tours**. C'est l'indicateur le plus exigeant. Le bot n'en a
pas besoin, puisqu'il s'appuie sur les probabilités.

Les têtes d'intention du même modèle restent au niveau du LSTM v3 : 76 % en mouvement et 94 % en
gestion.

### Étape 5 : prédire un tour (`05_predire_tour.py`)
Ce script fait ce que ferait l'API du jeu. Il relit les données brutes d'un tour, applique l'ETL,
appelle le modèle, et écrit `resultats/exemple_tour_predit.json` (voir §6 bis) ainsi que
`fig7_tour_predit.png`.

### Étape 4 : visualisations (`04_visualisations.py`)
| Figure | Message |
|---|---|
| `fig7_tour_predit.png` | Le tour prédit à côté du tour réel, avec la carte de présence prévue |
| `fig8_niveau2_comparaison.png` | Niveau 2 : case exacte 75 % contre 35 %, bonne cible 80 % contre 46 % |
| `fig1_comparaison_modeles.png` | Le LSTM bat tous les modèles de référence, sur les deux sorties |
| `fig2_courbes_apprentissage.png` | Entraînement et validation restent proches : pas de sur-apprentissage marqué |
| `fig3_matrices_confusion.png` | Mouvement : « Attaquer » est très bien reconnu (94 %). Une avancée est parfois prise pour une attaque. Gestion : « Construire » est parfois manqué (61 %) |
| `fig4_demarrage_a_froid.png` | Le profil joueur fait gagner 1 à 3 points en gestion à partir de la 2ᵉ partie |
| `fig5_positions_par_style.png` | Les styles se voient sur la carte |
| `fig6_mouvements_par_style.png` | Chaque style a sa signature de mouvements |

---

## 6. Exemple concret : ce que l'IA reçoit et ce qu'elle répond

Joueur J0038 (style opportuniste, un joueur du test que le modèle n'a jamais vu), partie 4, tour 20.

**Ce que le jeu enregistre au tour 20** (format du projet, extrait) :
```json
"joueur": {
  "unites": {
    "J_U3":  { "Nom": "Ouvrier",  "PV": 5,  "position": [4, 1], "Movement": [4, 1], "Attaque": null, "Destroy": false, "construct": null },
    "J_U16": { "Nom": "Cavalier", "PV": 12, "position": [6, 5], "Movement": [4, 5], "Attaque": null, "Destroy": false, "construct": null } },
  "batiments": {
    "J_B2": { "Nom": "Caserne", "PV": 20, "position": [4, 0], "Attaque": null, "destroy": false, "entraine": null }, ... } },
"IA": {
  "ressources": { "or": 40, "bois": 28, "pierre": 23, "fer": 26, "carburant": 22 },
  "Objectif": { "M1": { "type": "attaque", "Cible": [6, 0], "Priorite": 2 } },
  "unites": { "IA_U15": { "Nom": "Piquier", "PV": 3, "position": [6, 9], "Movement": [5, 8], ... } }, ... }
```

**Ce que l'ETL en fait (extrait des 53 variables)** :
- cavaliers 0.50, ouvriers 0.50 (2 unités) ;
- PV moyens 1.00, avancée 0.30 ;
- QG de l'IA 0.43 (13 PV sur 30) ;
- objectif d'attaque de l'IA 1, distance minimale 0.27 ;
- **ce tour-ci : avancer à gauche, rien en gestion** ;
- profil : attaque 46 % du temps, entraîne 26 % des tours.

**Ce que le LSTM reçoit** : les tours 11 à 20 sous cette forme, soit une matrice de 10 × 53.
Dans cette séquence, il voit que le joueur a entraîné une unité aux tours 13 et 17.

**Ce qu'il répond pour le tour 21** :
```
Mouvement : avancer à gauche 69 % · avancer au centre 18 % · attaquer 8 % · défendre 3 % · droite 2 %
Gestion   : entraîner 96 % · rien 3 % · construire 2 %
```
**Ce que le joueur a fait au tour 21 :** son cavalier va de [4, 5] à [3, 6] (vers la gauche) et
la caserne entraîne un Cavalier. Les deux prédictions sont justes. Pour la gestion, le modèle a
reconnu le rythme du joueur : une unité tous les 4 tours (13, 17, puis 21).

## 6 bis. Exemple concret du niveau 2 : le tour prédit

Un joueur du test, jamais vu, partie 4. Le modèle prédit le tour 55. Le fichier complet est
`resultats/exemple_tour_predit.json`, et la figure `fig7_tour_predit.png`.

```json
"Tour_55_predit": {
  "unites": {
    "J_U3":  { "Nom": "Ouvrier",  "position": [5, 1],  "Movement": [5, 1],  "proba_Movement": 1.00,
               "Attaque": null,
               "REEL": { "Movement": [5, 1], "Attaque": null } },
    "J_U33": { "Nom": "Archer",   "position": [6, 10], "Movement": [6, 10], "proba_Movement": 0.85,
               "Attaque": "IA_U39", "proba_Attaque": 0.59,
               "REEL": { "Movement": [6, 10], "Attaque": "IA_U39" } },
    "J_U35": { "Nom": "Piquier",  "position": [6, 9],  "Movement": [6, 10], "proba_Movement": 0.46,
               "Attaque": "IA_U39", "proba_Attaque": 0.43,
               "REEL": { "Movement": [6, 9], "Attaque": "IA_U39" } },
    "J_U37": { "Nom": "Archer",   "position": [6, 6],  "Movement": [7, 7],  "proba_Movement": 0.49,
               "Attaque": null,
               "REEL": { "Movement": [7, 7], "Attaque": null } },
    "J_U38": { "Nom": "Piquier",  "position": [5, 2],  "Movement": [6, 3],  "proba_Movement": 0.58,
               "Attaque": null,
               "REEL": { "Movement": [6, 3], "Attaque": null } } },
  "batiments": {
    "J_B1": { "Nom": "QG",      "entraine": null,    "proba_entraine": 1.00, "REEL": { "entraine": null } },
    "J_B2": { "Nom": "Caserne", "entraine": "Archer", "proba_entraine": 0.40, "REEL": { "entraine": "Piquier" } } } }
```

Quatre unités sur cinq sont prédites sur la bonne case, et les deux attaques sont justes, toutes
deux sur `IA_U39`. Deux erreurs, instructives l'une comme l'autre.

Le piquier `J_U35` devait avancer d'une case selon le modèle ; le joueur est resté sur place. Or
rester était la **deuxième hypothèse du modèle, à 40 %**. C'est exactement ainsi que le bot
tactique exploite la prédiction : il reçoit une distribution, pas une certitude, et peut couvrir
les deux cases.

La caserne entraîne un Piquier quand le modèle annonçait un Archer à 40 %. C'est la limite
assumée du niveau 2 : le type d'unité n'est retrouvé que dans 38 % des cas, parce qu'il est tiré
au hasard selon les préférences du joueur simulé.

Le fichier contient aussi la **carte de présence**, une grille 12 × 12 de probabilités que le bot
tactique peut lire directement.

---

## 7. Base de données : du format source au format optimisé (compétence *concevoir une base optimisée*)

**Format source** (celui du §2) : un document par joueur, avec des identifiants comme clés.
- Il grossit à chaque partie : ≈ 113 Ko par partie, donc la **limite MongoDB de 16 Mo est
  atteinte vers 140 parties**, soit environ 7 mois pour un joueur régulier à 20 parties par mois.
- Les clés dynamiques (`<ID_Joueur>`, `<Tour_X>`) ne peuvent pas être indexées.
- Il faut relire tout le document pour obtenir un seul tour.

**Format optimisé** (produit par l'ETL) : **un document par partie**, des listes au lieu de clés
dynamiques.
```json
{ "joueur": "anon_3f9a1c…", "partie": 4, "nb_tours": 64,
  "tours": [
    { "tour": 20,
      "joueur_etat": { "unites": [ { "id": "J_U16", "Nom": "Cavalier", "PV": 12, "position": [6,5], "Movement": [4,5], ... } ],
                       "batiments": [ ... ] },
      "ia_etat": { "ressources": {...}, "objectifs": [ { "id": "M1", "type": "attaque", ... } ], "unites": [...], "batiments": [...] },
      "mouvement": "AVANCER_GAUCHE", "gestion": "RIEN" },
    ...
  ] }
```
```js
db.parties.createIndex({ joueur: 1, partie: 1 })   // historique d'un joueur, dans l'ordre
```

**Pourquoi la partie et pas le tour.** Personne ne lit un tour isolé : le modèle lit une
fenêtre de dix tours, le profil lit les parties précédentes. La maille de lecture est donc la
partie. Découper par tour multiplierait par 64 le nombre d'entrées d'index — 96 097 au lieu de
1 500 sur ce jeu de données — pour aucun gain, et répéterait les clés `joueur` et `partie` dans
chaque document (2,9 % du volume). La partie est de surcroît un *seau* naturellement borné :
les règles du jeu la limitent à `MAX_TOURS` tours. C'est le *bucket pattern* recommandé par
MongoDB.

**Mesuré sur les 1 500 parties produites** : 119,9 Ko par document en moyenne, 250,1 Ko au
maximum, soit **66 fois sous la limite de 16 Mo**. Le format source, lui, l'atteindrait vers
140 parties.

**Pourquoi écrire en fin de partie.** Le moteur de stockage réécrit le document entier à chaque
mise à jour : ajouter les 64 tours un par un ferait écrire environ 4 Mo là où une écriture
unique en écrit 120. L'ETL écrit donc la partie une seule fois, quand elle est terminée. Le
risque est de perdre le journal d'une partie si le serveur tombe en cours de route — acceptable
pour des données d'entraînement ; pour s'en prémunir, on viderait le tampon tous les vingt
tours, ce qui laisse encore vingt fois moins d'entrées d'index qu'un document par tour.

On peut archiver ou partitionner la collection par joueur (sharding).

---

## 8. Big data : les 3V (compétence *analyser une problématique big data*)

Le détail est dans `resultats/volumetrie.json`.

| | Mesure | Projection |
|---|---|---|
| **Volume** | Données brutes : ≈ 1,8 Ko par tour, 113 Ko par partie. Stocké : ≈ 120 Ko par document de partie, 250 Ko au maximum | 10 000 joueurs × 20 parties/mois → **≈ 23 Go/mois, 275 Go/an, 154 M de tours/an** |
| **Vélocité** | ≈ 33 événements par seconde pour 1 000 joueurs simultanés à 1 tour/30 s, groupés en **1 écriture par partie** → ≈ 0,5 insertion/s | Réentraînement après chaque partie |
| **Variété** | Comptes dans MySQL (structuré). Parties en JSON imbriqué de taille variable (le nombre d'unités, de bâtiments et d'objectifs change à chaque tour) dans MongoDB | |

**Stratégie retenue :**
- **Collecte** : le client Godot envoie chaque tour à l'API Spring Boot, qui l'écrit dans MongoDB.
- **ETL** : en Python, après chaque partie, plus un traitement complet chaque nuit.
- **Entraînement** : PyTorch sur CPU.
- **Prédiction** : un service Python appelé par l'API, ou le modèle exporté en ONNX et chargé
  côté Java.
- **Passage à l'échelle, si besoin** : Spark pour l'ETL, sharding MongoDB.

---

## 9. RGPD et AI Act

- **Minimisation** : le format de jeu ne contient aucune donnée d'identité. Email, date de
  naissance et pseudo restent dans MySQL, et ne sont jamais joints aux données d'entraînement.
- **Pseudonymisation** de l'identifiant joueur : HMAC-SHA256 avec une clé secrète lue dans la
  variable d'environnement `MECHAVALON_CLE_PSEUDO` (voir §1), donc stockée hors
  du code. Attention, des données pseudonymisées **restent des données personnelles** : on ne
  parle pas d'anonymisation.
- **Information** : le joueur est informé que son style de jeu sert à adapter la difficulté
  (profilage).
- **Droit à l'effacement** : le bouton « Réinitialiser l'IA » supprime ses tours et son profil.
- **Conservation** : durée limitée.
- **Mineurs** : consentement parental sous 15 ans en France.
- **AI Act** : à mon analyse, un système d'adaptation de la difficulté dans un jeu relève du
  **risque minimal**. Point de vigilance : l'AI Act interdit les techniques manipulatrices qui
  exploitent les vulnérabilités des personnes, par exemple l'âge. Le DDA vise donc un défi
  équilibré, pas l'allongement du temps de jeu.

---

## 10. Limites (à dire soi-même)

- **Données simulées** : de vrais joueurs seront plus imprévisibles. Il faudra mesurer de
  nouveau sur des playtests.
- **Mouvement** : une avancée est parfois prédite comme une attaque, quand des ennemis sont
  proches. **Gestion** : « Construire » est rare (4 % des tours) et parfois manqué. Prochaine
  itération : pondérer la loss par classe.
- **Démarrage à froid** : à la 1ʳᵉ partie, le profil est vide. Le modèle global fait déjà
  ≈ 75 % en mouvement et 91 % en gestion.
- **Niveau 2** : le type d'unité entraînée reste difficile à deviner (38 %), car il est en partie
  dû au hasard. Les attaques de bâtiments (le QG) sont plus souvent manquées que les attaques
  d'unités. La simulation autorise deux unités sur la même case, ce que le vrai jeu interdira
  sûrement : à corriger avant d'utiliser de vraies données.
- **Le DDA** (garder l'IA entre 50 et 60 % de victoires) n'est pas encore branché. Seule, l'IA
  tactique gagne 44 % des parties.

---

## 11. Questions probables du jury

**Pourquoi un LSTM et pas un Transformer ?**
Les séquences sont courtes (10 tours), les données modestes, et il faut prédire en temps réel
sur CPU. Le LSTM léger (43 000 paramètres, 1,6 ms) suffit.

**Comment savez-vous que la mémoire sert à quelque chose ?**
Sur la gestion, le MLP sans mémoire fait 68 % de F1, contre 87 % pour le LSTM. Le rythme
d'entraînement ne se voit qu'avec l'historique.

**Pourquoi deux sorties ?**
Dans un tour, le joueur déplace ses unités ET gère sa base. Ce sont deux décisions différentes.
Un seul tronc LSTM les apprend ensemble, ce qui économise du calcul.

**Pourquoi le F1 macro ?**
Les classes sont déséquilibrées. « Rien » représente 73 % des tours en gestion : prédire
toujours « Rien » donne 73 % d'exactitude, mais seulement 28 % de F1 macro.

**Pourquoi un découpage par joueur ?**
Pour mesurer la performance sur un joueur jamais vu. C'est le vrai cas d'usage, et ça évite
les fuites de données.

**Comment les étiquettes sont-elles obtenues ?**
Par des règles métier sur les champs `Attaque`, `Movement`, `entraine` et `construct`. On les a
vérifiées contre la vérité de la simulation : 100 % de concordance en gestion, 92 % en
mouvement. Les écarts sont expliqués au §5.

**Comment ça s'intégrerait au jeu ?**
Au début du tour du joueur, l'API envoie au modèle les 10 derniers tours et le plateau actuel.
Le modèle renvoie le tour prédit (niveau 2), les intentions (niveau 1) et la carte de présence.
Le bot tactique s'en sert pour son tour : il anticipe où seront les unités, lesquelles vont
attaquer, et si une nouvelle unité arrive.

**Pourquoi la prédiction n'a-t-elle pas besoin d'être parfaite ?**
Le bot réagit à des probabilités, pas à une certitude. Si le modèle donne 53 % pour une case et
40 % pour la voisine, le bot peut couvrir les deux. Un tour prédit juste à 100 % n'est pas
l'objectif. L'objectif est d'anticiper mieux qu'une IA scriptée qui ne regarde pas le joueur.

**Comment gérer un nombre d'unités qui change ?**
Une même petite tête de prédiction est appliquée à chaque unité, avec le résumé de la partie
calculé par le LSTM. Qu'il y ait 2 ou 15 unités, le modèle est le même.

**Comment sait-on qu'une unité ne fait pas un déplacement impossible ?**
Les sorties de déplacement sont **masquées** : seules les cases atteignables par ce type d'unité
et dans la carte peuvent être choisies.

---

## 12. Arborescence

```
config.py                  paramètres communs (noms d'unités et de bâtiments, classes, volumes)
01_simulation.py           génération des parties au format du jeu
02_etl.py                  ETL : qualité, RGPD, étiquettes, variables, profil, MongoDB optimisé
entites.py                 niveau 2 : préparation des données unité par unité (utilisé par l'ETL et la prédiction)
03_entrainement.py         niveau 1 : baselines, LSTM v1/v2/v3 à deux sorties, métriques
03b_entrainement_niveau2.py niveau 2 : LSTM + têtes par unité et par bâtiment, règles de référence, métriques
04_visualisations.py       graphiques
05_predire_tour.py         prédiction d'un tour au format du jeu (ce que ferait l'API)
data/raw/                  joueurs.jsonl.gz (format du jeu) + fichiers de contrôle de la simulation
data/processed/            dataset.npz, entites.npz, variables.json, mongo_parties.jsonl.gz
modeles/                   lstm_mechavalon.pt (niveau 1), mechavalon_niveau2.pt (niveaux 1 + 2)
resultats/                 resultats.md, metriques.json, exemple_tour_predit.json, volumetrie.json,
                           rapport_qualite.json, figures/
```
