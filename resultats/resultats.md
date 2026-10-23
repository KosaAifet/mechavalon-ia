# Résultats — IA prédictive Mechavalon (données simulées)

Test sur **23 joueurs jamais vus** (15287 tours).
Hasard : 20 % pour le mouvement (5 classes), 33 % pour la gestion (3 classes).

## Comparaison des modèles

| Modèle | Mouvement : exactitude | Mouvement : top-2 | Mouvement : F1 macro | Gestion : exactitude | Gestion : F1 macro |
|---|---|---|---|---|---|
| Classe majoritaire | 39.3% | 57.9% | 11.3% | 73.0% | 28.1% |
| Répéter le dernier tour | 43.8% | — | 45.0% | 47.7% | 22.8% |
| Markov ordre 1 | 45.4% | 74.3% | 31.7% | 73.0% | 28.1% |
| MLP sans mémoire | 72.2% | 91.5% | 70.9% | 79.4% | 68.1% |
| LSTM v1 | 74.6% | 91.1% | 72.5% | 85.7% | 77.3% |
| LSTM v2 | 75.2% | 91.6% | 73.6% | 92.9% | 85.5% |
| LSTM v3 | 76.4% | 92.5% | 75.2% | 94.6% | 87.3% |

## Itérations (ajustements documentés)

| Version | Changement | Époques | Durée (s) | Mouvement F1 | Gestion F1 |
|---|---|---|---|---|---|
| LSTM v1 | Fenêtre 5 tours, sans profil joueur | 17 | 45.5 | 72.5% | 77.3% |
| LSTM v2 | Fenêtre allongée à 10 tours | 17 | 103.4 | 73.6% | 85.5% |
| LSTM v3 | Ajout du profil joueur (parties précédentes) + dropout 0.2 | 24 | 200.2 | 75.2% | 87.3% |

## Modèle final : LSTM v3

- Paramètres : 43 272 (≈ 169.0 Ko)
- Latence d'une prédiction (CPU) : 1.574 ms
- Durée d'entraînement (CPU) : 200.2 s

### Sortie mouvement

| Classe | Précision | Rappel | F1 | Nb tours test |
|---|---|---|---|---|
| AVANCER_GAUCHE | 81.1% | 62.4% | 70.5% | 2834 |
| AVANCER_CENTRE | 80.8% | 56.6% | 66.6% | 2751 |
| AVANCER_DROITE | 84.0% | 67.9% | 75.1% | 1671 |
| ATTAQUER | 69.7% | 94.3% | 80.2% | 6013 |
| DEFENDRE | 91.7% | 77.0% | 83.7% | 2018 |

Exactitude selon le nombre de parties déjà jouées (mouvement) :

| Partie n° | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| LSTM v2 (sans profil) | 74.8% | 76.1% | 76.1% | 74.6% | 74.3% | 75.6% | 76.5% | 75.1% | 74.5% | 75.0% |
| LSTM v3 (avec profil) | 75.5% | 77.3% | 76.7% | 76.1% | 75.5% | 76.2% | 77.9% | 76.8% | 75.9% | 76.7% |

Exactitude par style (mouvement) : agressif 87.2%, defensif 72.3%, flanqueur 79.2%, economiste 69.4%, opportuniste 79.5%

### Sortie gestion

| Classe | Précision | Rappel | F1 | Nb tours test |
|---|---|---|---|---|
| RIEN | 95.2% | 98.3% | 96.7% | 11163 |
| ENTRAINER | 92.7% | 89.1% | 90.8% | 3490 |
| CONSTRUIRE | 95.3% | 61.0% | 74.4% | 634 |

Exactitude selon le nombre de parties déjà jouées (gestion) :

| Partie n° | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| LSTM v2 (sans profil) | 91.4% | 93.0% | 92.5% | 93.1% | 92.7% | 92.9% | 92.9% | 93.3% | 94.0% | 93.1% |
| LSTM v3 (avec profil) | 91.2% | 94.4% | 93.7% | 94.7% | 95.0% | 95.2% | 95.6% | 95.6% | 95.9% | 95.2% |

Exactitude par style (gestion) : agressif 95.1%, defensif 95.0%, flanqueur 95.0%, economiste 94.0%, opportuniste 94.8%

# Niveau 2 — le tour suivant prédit unité par unité

Test : 15287 tours, 52416 unités, joueurs jamais vus.

## Movement (case d'arrivée de chaque unité)

| Modèle | Case exacte | Case exacte (unités de combat) | À une case près | Distance moyenne | Case dans le top-3 |
|---|---|---|---|---|---|
| Immobile | 44.2% | 26.9% | 87.9% | 0.679 | — |
| Répéter le dernier déplacement | 50.7% | 35.4% | 81.1% | 0.72 | — |
| LSTM niveau 2 | 80.7% | 74.7% | 94.0% | 0.259 | 95.5% |

## Attaque (cible de chaque unité)

| Modèle | Bonne décision | F1 attaque oui/non | Bonne cible quand il attaque |
|---|---|---|---|
| Attaquer la cible la plus faible à portée | 79.8% | 53.6% | 45.9% |
| LSTM niveau 2 | 88.5% | 75.8% | 80.0% |

## entraine (QG et caserne) et construct (ouvriers)

| Sortie | Modèle | Exactitude | F1 « agit / n'agit pas » | Bon type quand il agit |
|---|---|---|---|---|
| entraine | Toujours « rien » | 88.6% | 0.0% | 0.0% |
| entraine | LSTM niveau 2 | 92.1% | 90.1% | 37.8% |
| construct_ouvriers | Toujours « rien » | 94.9% | 0.0% | 0.0% |
| construct_ouvriers | LSTM niveau 2 | 98.1% | 78.0% | 65.9% |

Tour entièrement juste (toutes les unités et tous les bâtiments) : 51.9%

Modèle complet (niveaux 1 + 2) : 56 556 paramètres, 265.7 s d'entraînement sur CPU.
