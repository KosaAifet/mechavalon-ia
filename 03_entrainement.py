"""
ÉTAPE 3 — ENTRAÎNEMENT ET ÉVALUATION

Le modèle prédit, à partir des 10 derniers tours, ce que le joueur fera au tour suivant :
  - sortie MOUVEMENT (5 classes) : avancer à gauche / au centre / à droite, attaquer, défendre
  - sortie GESTION   (3 classes) : rien, entraîner une unité, construire un bâtiment

1. Découpage PAR JOUEUR : 70 % entraînement, 15 % validation, 15 % test.
   Les joueurs du test n'ont jamais été vus : on mesure la prédiction d'un NOUVEAU joueur.
2. Modèles de référence (baselines), pour chaque sortie :
     - classe majoritaire  : prédit toujours la classe la plus fréquente
     - répétition          : prédit que le joueur refait ce qu'il vient de faire
     - Markov ordre 1      : P(action suivante | action actuelle), sans mémoire longue
     - MLP sans mémoire    : réseau de neurones qui ne voit QUE le dernier tour
3. Trois itérations du LSTM (LSTM 64 -> LSTM 32 -> 2 sorties Softmax ; Cross-Entropy ; Adam ; BPTT) :
     - v1 : fenêtre de 5 tours,  sans profil joueur
     - v2 : fenêtre de 10 tours, sans profil joueur
     - v3 : fenêtre de 10 tours, AVEC profil joueur (parties précédentes) + dropout
4. Métriques sur le test : exactitude, top-2 (mouvement), précision, rappel, F1 macro,
   matrices de confusion, latence.

Sorties : modeles/lstm_mechavalon.pt, resultats/metriques.json, resultats/resultats.md
"""
import json
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch import nn

import config as C

torch.manual_seed(C.GRAINE)
np.random.seed(C.GRAINE)

NOMS = json.loads((C.DOSSIER_PROCESSED / "variables.json").read_text(encoding="utf-8"))
DEBUT_PROFIL = NOMS.index("profil_" + C.MOUVEMENTS[0])
COL_MVT = slice(NOMS.index("tour_" + C.MOUVEMENTS[0]), NOMS.index("tour_" + C.MOUVEMENTS[-1]) + 1)
COL_GES = slice(NOMS.index("tour_" + C.GESTIONS[0]), NOMS.index("tour_" + C.GESTIONS[-1]) + 1)
SORTIES = {"mouvement": C.MOUVEMENTS, "gestion": C.GESTIONS}


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
def charger():
    d = np.load(C.DOSSIER_PROCESSED / "dataset.npz")
    d = {k: d[k] for k in d.files}
    # index de la première ligne de chaque partie (pour construire les fenêtres)
    debut = np.zeros(len(d["id_partie"]), dtype=np.int64)
    for i in range(1, len(debut)):
        debut[i] = debut[i - 1] if d["id_partie"][i] == d["id_partie"][i - 1] else i
    d["debut"] = debut
    return d


def decouper_joueurs(nb_joueurs):
    ordre = np.random.RandomState(C.GRAINE).permutation(nb_joueurs)
    n_train, n_val = int(0.70 * nb_joueurs), int(0.15 * nb_joueurs)
    return set(ordre[:n_train]), set(ordre[n_train:n_train + n_val]), set(ordre[n_train + n_val:])


def fenetres(d, joueurs, longueur, avec_profil):
    """Pour chaque tour avec une cible : les `longueur` derniers tours de la partie (complétés par des 0)."""
    X = d["X"] if avec_profil else d["X"][:, :DEBUT_PROFIL]
    idx = np.where(np.isin(d["id_joueur"], list(joueurs)) & (d["y_mvt"] >= 0))[0]
    seq = np.zeros((len(idx), longueur, X.shape[1]), dtype=np.float32)
    for j, i in enumerate(idx):
        bloc = X[max(d["debut"][i], i - longueur + 1): i + 1]
        seq[j, longueur - len(bloc):] = bloc
    return torch.tensor(seq), torch.tensor(d["y_mvt"][idx]), torch.tensor(d["y_ges"][idx]), idx


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------
class LSTMMechavalon(nn.Module):
    def __init__(self, nb_entrees, dropout=0.0):
        super().__init__()
        self.lstm1 = nn.LSTM(nb_entrees, 64, batch_first=True)   # apprend les comportements
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)           # compresse l'essentiel
        self.dropout = nn.Dropout(dropout)
        self.sortie_mouvement = nn.Linear(32, C.NB_MOUVEMENTS)   # softmax appliqué dans la loss
        self.sortie_gestion = nn.Linear(32, C.NB_GESTIONS)

    def forward(self, x):
        h, _ = self.lstm1(x)
        h, _ = self.lstm2(self.dropout(h))
        h = self.dropout(h[:, -1])                               # dernier pas de temps = dernier tour joué
        return self.sortie_mouvement(h), self.sortie_gestion(h)


class MLPSansMemoire(nn.Module):
    def __init__(self, nb_entrees):
        super().__init__()
        self.tronc = nn.Sequential(nn.Linear(nb_entrees, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.sortie_mouvement = nn.Linear(32, C.NB_MOUVEMENTS)
        self.sortie_gestion = nn.Linear(32, C.NB_GESTIONS)

    def forward(self, x):
        h = self.tronc(x[:, -1])
        return self.sortie_mouvement(h), self.sortie_gestion(h)


def perte(modele, X, ym, yg, fn=nn.CrossEntropyLoss()):
    lm, lg = modele(X)
    return fn(lm, ym) + fn(lg, yg), lm, lg


def entrainer(modele, train, val, nom, max_epoques=25, patience=3):
    optimiseur = torch.optim.Adam(modele.parameters(), lr=1e-3)
    Xtr, ymtr, ygtr = train
    hist = {"perte_train": [], "perte_val": [], "exactitude_val_mouvement": [], "exactitude_val_gestion": []}
    meilleure, meilleur_etat, attente = float("inf"), None, 0
    t0 = time.time()
    for epoque in range(1, max_epoques + 1):
        modele.train()
        ordre = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(Xtr), 256):
            lot = ordre[i:i + 256]
            optimiseur.zero_grad()
            p, _, _ = perte(modele, Xtr[lot], ymtr[lot], ygtr[lot])
            p.backward()                       # rétropropagation dans le temps (BPTT)
            optimiseur.step()
            total += p.item() * len(lot)
        modele.eval()
        with torch.no_grad():
            pv, lm, lg = perte(modele, *val)
        em = (lm.argmax(1) == val[1]).float().mean().item()
        eg = (lg.argmax(1) == val[2]).float().mean().item()
        hist["perte_train"].append(total / len(Xtr))
        hist["perte_val"].append(pv.item())
        hist["exactitude_val_mouvement"].append(em)
        hist["exactitude_val_gestion"].append(eg)
        print(f"  [{nom}] époque {epoque:2d}  perte train {total / len(Xtr):.4f}  perte val {pv.item():.4f}  "
              f"exactitude val : mouvement {em:.3f} / gestion {eg:.3f}")
        if pv.item() < meilleure - 1e-4:
            meilleure, attente = pv.item(), 0
            meilleur_etat = {k: v.clone() for k, v in modele.state_dict().items()}
        else:
            attente += 1
            if attente >= patience:  # arrêt précoce : évite le sur-apprentissage
                print(f"  [{nom}] arrêt précoce (pas d'amélioration depuis {patience} époques)")
                break
    modele.load_state_dict(meilleur_etat)
    hist["duree_s"] = round(time.time() - t0, 1)
    hist["epoques"] = len(hist["perte_val"])
    return hist


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------
def metriques(y_vrai, y_pred, probas=None, topk=None):
    p, r, f1, _ = precision_recall_fscore_support(y_vrai, y_pred, average="macro", zero_division=0)
    m = {"exactitude": round(accuracy_score(y_vrai, y_pred), 4), "precision_macro": round(p, 4),
         "rappel_macro": round(r, 4), "f1_macro": round(f1, 4)}
    if probas is not None and topk:
        top = np.argsort(-probas, axis=1)[:, :topk]
        m[f"top{topk}"] = round(float(np.mean([y in t for y, t in zip(y_vrai, top)])), 4)
    return m


def predire(modele, X):
    modele.eval()
    with torch.no_grad():
        lm, lg = modele(X)
    return torch.softmax(lm, 1).numpy(), torch.softmax(lg, 1).numpy()


def evaluer(modele, te):
    pm, pg = predire(modele, te[0])
    return ({"mouvement": metriques(te[1].numpy(), pm.argmax(1), pm, 2),
             "gestion": metriques(te[2].numpy(), pg.argmax(1), pg)}, pm, pg)


def latence_ms(modele, exemple, repetitions=500):
    modele.eval()
    with torch.no_grad():
        for _ in range(20):
            modele(exemple)
        t0 = time.perf_counter()
        for _ in range(repetitions):
            modele(exemple)
    return round((time.perf_counter() - t0) / repetitions * 1000, 3)


def main():
    C.DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)
    d = charger()
    train_j, val_j, test_j = decouper_joueurs(int(d["id_joueur"].max()) + 1)
    print(f"Joueurs : {len(train_j)} entraînement / {len(val_j)} validation / {len(test_j)} test")
    r = {"decoupage": {"joueurs_train": len(train_j), "joueurs_val": len(val_j), "joueurs_test": len(test_j)},
         "modeles": {}, "iterations": []}

    # ----------------------------- Baselines ---------------------------------
    _, ym_tr, yg_tr, idx_tr = fenetres(d, train_j, 1, False)
    _, ym_te, yg_te, idx_te = fenetres(d, test_j, 1, False)
    r["decoupage"].update({"tours_train": len(idx_tr), "tours_test": len(idx_te)})
    vrais_tr = {"mouvement": ym_tr.numpy(), "gestion": yg_tr.numpy()}
    vrais_te = {"mouvement": ym_te.numpy(), "gestion": yg_te.numpy()}
    cols = {"mouvement": COL_MVT, "gestion": COL_GES}
    base = {"Classe majoritaire": {}, "Répéter le dernier tour": {}, "Markov ordre 1": {}}
    for s, classes in SORTIES.items():
        k = len(classes)
        freq = np.bincount(vrais_tr[s], minlength=k) / len(vrais_tr[s])
        base["Classe majoritaire"][s] = metriques(vrais_te[s], np.full_like(vrais_te[s], freq.argmax()),
                                                  np.tile(freq, (len(vrais_te[s]), 1)), 2 if s == "mouvement" else None)
        actuel_te = d["X"][idx_te, cols[s]].argmax(1)
        base["Répéter le dernier tour"][s] = metriques(vrais_te[s], actuel_te)
        actuel_tr = d["X"][idx_tr, cols[s]].argmax(1)
        trans = np.ones((k, k))
        for a, b in zip(actuel_tr, vrais_tr[s]):
            trans[a, b] += 1
        trans /= trans.sum(1, keepdims=True)
        pr = trans[actuel_te]
        base["Markov ordre 1"][s] = metriques(vrais_te[s], pr.argmax(1), pr, 2 if s == "mouvement" else None)
    r["modeles"].update(base)

    # ----------------------------- MLP sans mémoire ---------------------------
    print("\nMLP sans mémoire (ne voit que le dernier tour, avec profil)")
    tr, va, te = (fenetres(d, j, 1, True) for j in (train_j, val_j, test_j))
    mlp = MLPSansMemoire(tr[0].shape[2])
    h_mlp = entrainer(mlp, tr[:3], va[:3], "MLP")
    r["modeles"]["MLP sans mémoire"] = evaluer(mlp, te)[0]

    # ----------------------------- Itérations LSTM ----------------------------
    configurations = [
        ("LSTM v1", 5, False, 0.0, "Fenêtre 5 tours, sans profil joueur"),
        ("LSTM v2", 10, False, 0.0, "Fenêtre allongée à 10 tours"),
        ("LSTM v3", 10, True, 0.2, "Ajout du profil joueur (parties précédentes) + dropout 0.2"),
    ]
    finaux = {}
    for nom, longueur, profil, dropout, changement in configurations:
        print(f"\n{nom} : {changement}")
        torch.manual_seed(C.GRAINE)
        tr, va, te = (fenetres(d, j, longueur, profil) for j in (train_j, val_j, test_j))
        modele = LSTMMechavalon(tr[0].shape[2], dropout)
        hist = entrainer(modele, tr[:3], va[:3], nom)
        m, pm, pg = evaluer(modele, te)
        r["modeles"][nom] = m
        r["iterations"].append({"version": nom, "changement": changement, "fenetre": longueur,
                                "profil_joueur": profil, "dropout": dropout, "epoques": hist["epoques"],
                                "duree_entrainement_s": hist["duree_s"], **{s: m[s] for s in SORTIES}})
        finaux[nom] = (modele, hist, pm, pg, te)

    # ----------------------------- Analyse du modèle final --------------------
    modele, hist, pm, pg, te = finaux["LSTM v3"]
    idx = te[3]
    pred = {"mouvement": pm.argmax(1), "gestion": pg.argmax(1)}
    vrai = {"mouvement": te[1].numpy(), "gestion": te[2].numpy()}

    def par(cle, valeurs, idx_, p, v):
        return {str(x): round(float((p[d[cle][idx_] == x] == v[d[cle][idx_] == x]).mean()), 4)
                for x in valeurs if (d[cle][idx_] == x).any()}

    _, _, pm2, pg2, te2 = finaux["LSTM v2"]
    parties = range(1, C.PARTIES_PAR_JOUEUR + 1)
    nb_params = sum(p.numel() for p in modele.parameters())
    r["modele_final"] = {
        "nom": "LSTM v3", "parametres": nb_params, "taille_Ko": round(nb_params * 4 / 1024, 1),
        "latence_prediction_ms": latence_ms(modele, te[0][:1]),
        "duree_entrainement_s": hist["duree_s"], "historique": hist, "historique_mlp": h_mlp,
    }
    for s, classes in SORTIES.items():
        r["modele_final"][s] = {
            "rapport_par_classe": classification_report(vrai[s], pred[s], labels=range(len(classes)),
                                                        target_names=classes, output_dict=True, zero_division=0),
            "matrice_confusion": confusion_matrix(vrai[s], pred[s], labels=range(len(classes))).tolist(),
            "exactitude_par_num_partie": par("num_partie", parties, idx, pred[s], vrai[s]),
            "exactitude_par_num_partie_v2_sans_profil": par(
                "num_partie", parties, te2[3], (pm2 if s == "mouvement" else pg2).argmax(1),
                (te2[1] if s == "mouvement" else te2[2]).numpy()),
            "exactitude_par_style": {C.STYLES[int(k)]: v for k, v in
                                     par("style", range(len(C.STYLES)), idx, pred[s], vrai[s]).items()},
        }

    torch.save({"etat": modele.state_dict(), "nb_entrees": te[0].shape[2], "fenetre": 10,
                "mouvements": C.MOUVEMENTS, "gestions": C.GESTIONS, "variables": NOMS},
               C.DOSSIER_MODELES / "lstm_mechavalon.pt")
    with open(C.DOSSIER_RESULTATS / "metriques.json", "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    ecrire_resume(r)
    print(f"\nRésultats -> {C.DOSSIER_RESULTATS / 'resultats.md'}")


def pct(v):
    return f"{v:.1%}" if v is not None else "—"


def ecrire_resume(r):
    L = ["# Résultats — IA prédictive Mechavalon (données simulées)\n",
         f"Test sur **{r['decoupage']['joueurs_test']} joueurs jamais vus** ({r['decoupage']['tours_test']} tours).",
         "Hasard : 20 % pour le mouvement (5 classes), 33 % pour la gestion (3 classes).\n",
         "## Comparaison des modèles\n",
         "| Modèle | Mouvement : exactitude | Mouvement : top-2 | Mouvement : F1 macro | Gestion : exactitude | Gestion : F1 macro |",
         "|---|---|---|---|---|---|"]
    for nom, m in r["modeles"].items():
        L.append(f"| {nom} | {pct(m['mouvement']['exactitude'])} | {pct(m['mouvement'].get('top2'))} | "
                 f"{pct(m['mouvement']['f1_macro'])} | {pct(m['gestion']['exactitude'])} | {pct(m['gestion']['f1_macro'])} |")
    L += ["\n## Itérations (ajustements documentés)\n",
          "| Version | Changement | Époques | Durée (s) | Mouvement F1 | Gestion F1 |", "|---|---|---|---|---|---|"]
    for it in r["iterations"]:
        L.append(f"| {it['version']} | {it['changement']} | {it['epoques']} | {it['duree_entrainement_s']} | "
                 f"{pct(it['mouvement']['f1_macro'])} | {pct(it['gestion']['f1_macro'])} |")
    mf = r["modele_final"]
    L += [f"\n## Modèle final : {mf['nom']}\n",
          f"- Paramètres : {mf['parametres']:,} (≈ {mf['taille_Ko']} Ko)".replace(",", " "),
          f"- Latence d'une prédiction (CPU) : {mf['latence_prediction_ms']} ms",
          f"- Durée d'entraînement (CPU) : {mf['duree_entrainement_s']} s"]
    for s, classes in SORTIES.items():
        L += [f"\n### Sortie {s}\n", "| Classe | Précision | Rappel | F1 | Nb tours test |", "|---|---|---|---|---|"]
        for c in classes:
            x = mf[s]["rapport_par_classe"][c]
            L.append(f"| {c} | {pct(x['precision'])} | {pct(x['recall'])} | {pct(x['f1-score'])} | {int(x['support'])} |")
        e3, e2 = mf[s]["exactitude_par_num_partie"], mf[s]["exactitude_par_num_partie_v2_sans_profil"]
        L += [f"\nExactitude selon le nombre de parties déjà jouées ({s}) :\n",
              "| Partie n° | " + " | ".join(e3) + " |", "|---|" + "---|" * len(e3),
              "| LSTM v2 (sans profil) | " + " | ".join(pct(v) for v in e2.values()) + " |",
              "| LSTM v3 (avec profil) | " + " | ".join(pct(v) for v in e3.values()) + " |",
              f"\nExactitude par style ({s}) : " + ", ".join(f"{k} {pct(v)}" for k, v in mf[s]["exactitude_par_style"].items())]
    (C.DOSSIER_RESULTATS / "resultats.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
