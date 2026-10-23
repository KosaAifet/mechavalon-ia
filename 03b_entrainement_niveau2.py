"""
ÉTAPE 3b — NIVEAU 2 : PRÉDIRE LE TOUR SUIVANT UNITÉ PAR UNITÉ

Même tronc que le niveau 1 (LSTM 64 -> LSTM 32 sur les 10 derniers tours, avec profil joueur),
auquel on ajoute des têtes qui s'appliquent à CHAQUE unité et à CHAQUE bâtiment du joueur :

    résumé de la partie (LSTM, 32) + situation de l'unité (18) -> MLP -> Movement : 25 déplacements (masqués)
                                                                 -> construct : rien / Scierie / Mine / Raffinerie
                                                                 -> Attaque : score de chaque cible à portée + « aucune »
    résumé de la partie (32) + situation du bâtiment (7)        -> entraine : rien / Piquier / Archer / Cavalier / Ouvrier

Les têtes d'intention du niveau 1 sont gardées : un seul modèle sort les deux niveaux.
Le nombre d'unités change à chaque tour : les têtes sont appliquées à chaque unité, une par une.

Sorties : modeles/mechavalon_niveau2.pt et la clé "niveau2" de resultats/metriques.json
"""
import json
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

import config as C
import entites as E

torch.manual_seed(C.GRAINE)
np.random.seed(C.GRAINE)
FENETRE = 10


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
def charger():
    d = np.load(C.DOSSIER_PROCESSED / "dataset.npz")
    d = {k: d[k] for k in d.files}
    e = np.load(C.DOSSIER_PROCESSED / "entites.npz")
    e = {k: e[k] for k in e.files}
    n = len(d["id_partie"])
    debut = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        debut[i] = debut[i - 1] if d["id_partie"][i] == d["id_partie"][i - 1] else i
    d["debut"] = debut
    # index « CSR » : pour chaque ligne, ses unités et bâtiments ; pour chaque unité, ses candidats
    e["u_debut"] = np.searchsorted(e["u_ligne"], np.arange(n))
    e["u_fin"] = np.searchsorted(e["u_ligne"], np.arange(n), side="right")
    e["b_debut"] = np.searchsorted(e["b_ligne"], np.arange(n))
    e["b_fin"] = np.searchsorted(e["b_ligne"], np.arange(n), side="right")
    nu = len(e["u_ligne"])
    e["c_debut"] = np.searchsorted(e["c_unite"], np.arange(nu))
    e["c_fin"] = np.searchsorted(e["c_unite"], np.arange(nu), side="right")
    e["c_rang"] = np.arange(len(e["c_unite"])) - e["c_debut"][e["c_unite"]]
    # le déplacement réellement joué est toujours autorisé (quelques cas limites de bord de carte)
    e["u_masque"][np.arange(nu), e["u_delta"]] = True
    return d, e


def plages(debuts, fins):
    """Concatène les plages [debut, fin) et renvoie (indices, numéro de la plage d'origine)."""
    tailles = fins - debuts
    total = tailles.sum()
    origine = np.repeat(np.arange(len(debuts)), tailles)
    idx = np.arange(total) - np.repeat(np.cumsum(tailles) - tailles, tailles) + np.repeat(debuts, tailles)
    return idx, origine


def sequences(d, lignes):
    seq = np.zeros((len(lignes), FENETRE, d["X"].shape[1]), dtype=np.float32)
    for j, i in enumerate(lignes):
        bloc = d["X"][max(d["debut"][i], i - FENETRE + 1): i + 1]
        seq[j, FENETRE - len(bloc):] = bloc
    return torch.tensor(seq)


def lot(d, e, seq_toutes, pos_seq, lignes):
    """Rassemble tout ce qu'il faut pour un lot de lignes (tours)."""
    u_idx, u_loc = plages(e["u_debut"][lignes], e["u_fin"][lignes])
    c_idx, c_uloc = plages(e["c_debut"][u_idx], e["c_fin"][u_idx])
    b_idx, b_loc = plages(e["b_debut"][lignes], e["b_fin"][lignes])
    return {
        "seq": seq_toutes[pos_seq[lignes]],
        "y_mvt": torch.tensor(d["y_mvt"][lignes]), "y_ges": torch.tensor(d["y_ges"][lignes]),
        "u_loc": torch.tensor(u_loc), "u_var": torch.tensor(e["u_var"][u_idx]),
        "u_masque": torch.tensor(e["u_masque"][u_idx]), "u_delta": torch.tensor(e["u_delta"][u_idx]),
        "u_attaque": torch.tensor(e["u_attaque"][u_idx]), "u_construct": torch.tensor(e["u_construct"][u_idx]),
        "u_ouvrier": torch.tensor(e["u_ouvrier"][u_idx]),
        "c_uloc": torch.tensor(c_uloc), "c_rang": torch.tensor(e["c_rang"][c_idx]),
        "c_var": torch.tensor(e["c_var"][c_idx]),
        "b_loc": torch.tensor(b_loc), "b_var": torch.tensor(e["b_var"][b_idx]),
        "b_entraine": torch.tensor(e["b_entraine"][b_idx]),
        "u_idx": u_idx, "b_idx": b_idx,
    }


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------
class MechavalonNiveau2(nn.Module):
    def __init__(self, nb_entrees, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(nb_entrees, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.sortie_mouvement = nn.Linear(32, C.NB_MOUVEMENTS)        # niveau 1
        self.sortie_gestion = nn.Linear(32, C.NB_GESTIONS)
        self.unite = nn.Sequential(nn.Linear(32 + E.NB_VAR_UNITE, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.deplacement = nn.Linear(64, len(E.DELTAS))                 # niveau 2
        self.construction = nn.Linear(64, len(E.CONSTRUCTIONS))
        self.score_aucune = nn.Linear(64, 1)
        self.score_cible = nn.Sequential(nn.Linear(64 + E.NB_VAR_CANDIDAT, 32), nn.ReLU(), nn.Linear(32, 1))
        self.batiment = nn.Sequential(nn.Linear(32 + E.NB_VAR_BATIMENT, 32), nn.ReLU(),
                                      nn.Linear(32, len(E.ENTRAINEMENTS)))

    def forward(self, b):
        h, _ = self.lstm1(b["seq"])
        h, _ = self.lstm2(self.dropout(h))
        h = self.dropout(h[:, -1])
        sortie = {"mouvement": self.sortie_mouvement(h), "gestion": self.sortie_gestion(h)}
        hu = self.unite(torch.cat([h[b["u_loc"]], b["u_var"]], 1))
        sortie["deplacement"] = self.deplacement(hu).masked_fill(~b["u_masque"], -1e9)
        sortie["construction"] = self.construction(hu)
        attaque = torch.full((len(hu), E.MAX_CANDIDATS + 1), -1e9)
        attaque[:, 0] = self.score_aucune(hu).squeeze(1)
        if len(b["c_uloc"]):
            s = self.score_cible(torch.cat([hu[b["c_uloc"]], b["c_var"]], 1)).squeeze(1)
            attaque = attaque.index_put((b["c_uloc"], b["c_rang"] + 1), s)
        sortie["attaque"] = attaque
        sortie["entraine"] = self.batiment(torch.cat([h[b["b_loc"]], b["b_var"]], 1))
        return sortie


CE = nn.CrossEntropyLoss()


def perte(s, b):
    p = CE(s["mouvement"], b["y_mvt"]) + CE(s["gestion"], b["y_ges"])
    p = p + CE(s["deplacement"], b["u_delta"])
    ok = b["u_attaque"] >= 0
    p = p + CE(s["attaque"][ok], b["u_attaque"][ok])
    if b["u_ouvrier"].any():
        p = p + CE(s["construction"][b["u_ouvrier"]], b["u_construct"][b["u_ouvrier"]])
    if len(b["b_entraine"]):
        p = p + CE(s["entraine"], b["b_entraine"])
    return p


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------
def entrainer(modele, d, e, seq, pos, lignes_train, lignes_val, max_epoques=30, patience=3):
    opt = torch.optim.Adam(modele.parameters(), lr=1e-3)
    hist = {"perte_train": [], "perte_val": []}
    meilleure, etat, attente = float("inf"), None, 0
    val = lot(d, e, seq, pos, lignes_val)
    t0 = time.time()
    for ep in range(1, max_epoques + 1):
        modele.train()
        ordre = np.random.permutation(lignes_train)
        total = 0.0
        for i in range(0, len(ordre), 256):
            b = lot(d, e, seq, pos, ordre[i:i + 256])
            opt.zero_grad()
            p = perte(modele(b), b)
            p.backward()
            opt.step()
            total += p.item() * len(b["seq"])
        modele.eval()
        with torch.no_grad():
            pv = perte(modele(val), val).item()
        hist["perte_train"].append(total / len(ordre))
        hist["perte_val"].append(pv)
        print(f"  [niveau 2] époque {ep:2d}  perte train {total / len(ordre):.4f}  perte val {pv:.4f}")
        if pv < meilleure - 1e-4:
            meilleure, attente = pv, 0
            etat = {k: v.clone() for k, v in modele.state_dict().items()}
        else:
            attente += 1
            if attente >= patience:
                print("  [niveau 2] arrêt précoce")
                break
    modele.load_state_dict(etat)
    hist["duree_s"] = round(time.time() - t0, 1)
    hist["epoques"] = len(hist["perte_val"])
    return hist


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------
def evaluer_regles(e, u_idx, b_idx):
    """Modèles de référence sans apprentissage."""
    delta_reel = e["u_delta"][u_idx]
    immobile = np.full_like(delta_reel, E.INDEX_DELTA[(0, 0)])
    var = e["u_var"][u_idx]
    prec = [E.INDEX_DELTA[(int(round(dx * 2)), int(round(dy * 2)))] for dx, dy in var[:, 7:9]]
    prec = np.array([p if e["u_masque"][i, p] else E.INDEX_DELTA[(0, 0)] for i, p in zip(u_idx, prec)])
    # attaque : la cible la plus faible déjà à portée sans bouger, sinon aucune
    att = np.zeros(len(u_idx), dtype=np.int64)
    for j, i in enumerate(u_idx):
        nom = E.TYPES_UNITE[int(np.argmax(var[j, :4]))]
        portee = C.UNITES[nom]["portee"]
        a, b = e["c_debut"][i], e["c_fin"][i]
        if portee and b > a:
            cv = e["c_var"][a:b]
            ok = np.where(cv[:, 2] * 4 <= portee)[0]
            if len(ok):
                att[j] = ok[np.argmin(cv[ok, 3])] + 1
    return {
        "Immobile": metr_deplacement(e, u_idx, immobile),
        "Répéter le dernier déplacement": metr_deplacement(e, u_idx, prec),
    }, {"Attaquer la cible la plus faible à portée": metr_attaque(e, u_idx, att)}, {
        "Toujours « rien »": metr_classes(e["b_entraine"][b_idx], np.zeros(len(b_idx), dtype=np.int64))}


def cheb(a, b):
    return np.max(np.abs(np.array([E.DELTAS[i] for i in a]) - np.array([E.DELTAS[i] for i in b])), axis=1)


def metr_deplacement(e, u_idx, pred, probas=None):
    vrai = e["u_delta"][u_idx]
    d = cheb(pred, vrai)
    combat = ~e["u_ouvrier"][u_idx]
    m = {"case_exacte": round(float((d == 0).mean()), 4),
         "case_exacte_unites_combat": round(float((d[combat] == 0).mean()), 4),
         "a_une_case_pres": round(float((d <= 1).mean()), 4),
         "distance_moyenne_cases": round(float(d.mean()), 3)}
    if probas is not None:
        top3 = np.argsort(-probas, 1)[:, :3]
        m["case_dans_top3"] = round(float(np.mean([v in t for v, t in zip(vrai, top3)])), 4)
    return m


def metr_attaque(e, u_idx, pred):
    vrai = e["u_attaque"][u_idx]
    ok = vrai >= 0
    vrai, pred = vrai[ok], pred[ok]
    reelles = vrai > 0
    return {"bonne_decision": round(float((pred == vrai).mean()), 4),
            "f1_attaque_oui_non": round(float(f1_score(vrai > 0, pred > 0)), 4),
            "bonne_cible_quand_il_attaque": round(float((pred[reelles] == vrai[reelles]).mean()), 4)}


def metr_classes(vrai, pred):
    m = {"exactitude": round(float((vrai == pred).mean()), 4),
         "f1_macro": round(float(f1_score(vrai, pred, average="macro", zero_division=0)), 4),
         "f1_oui_non": round(float(f1_score(vrai > 0, pred > 0, zero_division=0)), 4)}
    if (vrai > 0).any():
        m["bon_choix_quand_il_agit"] = round(float((pred[vrai > 0] == vrai[vrai > 0]).mean()), 4)
    return m


def main():
    d, e = charger()
    ordre = np.random.RandomState(C.GRAINE).permutation(int(d["id_joueur"].max()) + 1)
    n_train, n_val = int(0.70 * len(ordre)), int(0.15 * len(ordre))
    groupes = {"train": ordre[:n_train], "val": ordre[n_train:n_train + n_val], "test": ordre[n_train + n_val:]}
    lignes = {k: np.where(np.isin(d["id_joueur"], v) & (d["y_mvt"] >= 0))[0] for k, v in groupes.items()}
    toutes = np.concatenate(list(lignes.values()))
    seq = sequences(d, toutes)
    pos = np.zeros(len(d["X"]), dtype=np.int64)
    pos[toutes] = np.arange(len(toutes))
    print(f"Lignes : {len(lignes['train'])} train / {len(lignes['val'])} val / {len(lignes['test'])} test")

    modele = MechavalonNiveau2(d["X"].shape[1])
    hist = entrainer(modele, d, e, seq, pos, lignes["train"], lignes["val"])

    # --------------------------- évaluation sur le test ----------------------
    modele.eval()
    morceaux = {k: [] for k in ("dep", "dep_p", "att", "cons", "ent", "mvt", "ges", "u_idx", "b_idx")}
    with torch.no_grad():
        for i in range(0, len(lignes["test"]), 512):
            b = lot(d, e, seq, pos, lignes["test"][i:i + 512])
            s = modele(b)
            p = torch.softmax(s["deplacement"], 1).numpy()
            morceaux["dep_p"].append(p)
            morceaux["dep"].append(p.argmax(1))
            morceaux["att"].append(s["attaque"].argmax(1).numpy())
            morceaux["cons"].append(s["construction"].argmax(1).numpy())
            morceaux["ent"].append(s["entraine"].argmax(1).numpy())
            morceaux["mvt"].append(s["mouvement"].argmax(1).numpy())
            morceaux["ges"].append(s["gestion"].argmax(1).numpy())
            morceaux["u_idx"].append(b["u_idx"])
            morceaux["b_idx"].append(b["b_idx"])
    r = {k: np.concatenate(v) for k, v in morceaux.items()}
    u_idx, b_idx = r["u_idx"], r["b_idx"]
    ouv = e["u_ouvrier"][u_idx]

    ref_dep, ref_att, ref_ent = evaluer_regles(e, u_idx, b_idx)
    res = {
        "deplacement": {**ref_dep, "LSTM niveau 2": metr_deplacement(e, u_idx, r["dep"], r["dep_p"])},
        "attaque": {**ref_att, "LSTM niveau 2": metr_attaque(e, u_idx, r["att"])},
        "entraine": {**ref_ent, "LSTM niveau 2": metr_classes(e["b_entraine"][b_idx], r["ent"])},
        "construct_ouvriers": {
            "Toujours « rien »": metr_classes(e["u_construct"][u_idx][ouv], np.zeros(int(ouv.sum()), dtype=np.int64)),
            "LSTM niveau 2": metr_classes(e["u_construct"][u_idx][ouv], r["cons"][ouv])},
        "intentions_niveau1_meme_modele": {
            "mouvement": metr_classes(d["y_mvt"][lignes["test"]], r["mvt"]),
            "gestion": metr_classes(d["y_ges"][lignes["test"]], r["ges"])},
    }
    # tour complet juste : toutes les unités sur la bonne case et toutes les décisions justes
    ligne_u = e["u_ligne"][u_idx]
    juste_u = (r["dep"] == e["u_delta"][u_idx]) & (r["att"] == e["u_attaque"][u_idx])
    juste_b = r["ent"] == e["b_entraine"][b_idx]
    faux = set(ligne_u[~juste_u]) | set(e["b_ligne"][b_idx][~juste_b])
    res["tour_entierement_juste"] = round(1 - len(faux) / len(lignes["test"]), 4)
    res["unites_test"] = int(len(u_idx))
    res["tours_test"] = int(len(lignes["test"]))
    nb = sum(p.numel() for p in modele.parameters())
    res["modele"] = {"parametres": nb, "taille_Ko": round(nb * 4 / 1024, 1), "duree_entrainement_s": hist["duree_s"],
                     "epoques": hist["epoques"], "historique": hist}

    torch.save({"etat": modele.state_dict(), "nb_entrees": d["X"].shape[1], "fenetre": FENETRE},
               C.DOSSIER_MODELES / "mechavalon_niveau2.pt")
    chemin = C.DOSSIER_RESULTATS / "metriques.json"
    tout = json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else {}
    tout["niveau2"] = res
    chemin.write_text(json.dumps(tout, indent=2, ensure_ascii=False), encoding="utf-8")
    ecrire_resume(res)
    print(json.dumps({k: v for k, v in res.items() if k != "modele"}, indent=2, ensure_ascii=False))


def pct(v):
    return f"{v:.1%}"


def ecrire_resume(res):
    L = ["\n# Niveau 2 — le tour suivant prédit unité par unité\n",
         f"Test : {res['tours_test']} tours, {res['unites_test']} unités, joueurs jamais vus.\n",
         "## Movement (case d'arrivée de chaque unité)\n",
         "| Modèle | Case exacte | Case exacte (unités de combat) | À une case près | Distance moyenne | Case dans le top-3 |",
         "|---|---|---|---|---|---|"]
    for n, m in res["deplacement"].items():
        L.append(f"| {n} | {pct(m['case_exacte'])} | {pct(m['case_exacte_unites_combat'])} | "
                 f"{pct(m['a_une_case_pres'])} | {m['distance_moyenne_cases']} | "
                 f"{pct(m['case_dans_top3']) if 'case_dans_top3' in m else '—'} |")
    L += ["\n## Attaque (cible de chaque unité)\n", "| Modèle | Bonne décision | F1 attaque oui/non | Bonne cible quand il attaque |",
          "|---|---|---|---|"]
    for n, m in res["attaque"].items():
        L.append(f"| {n} | {pct(m['bonne_decision'])} | {pct(m['f1_attaque_oui_non'])} | {pct(m['bonne_cible_quand_il_attaque'])} |")
    L += ["\n## entraine (QG et caserne) et construct (ouvriers)\n",
          "| Sortie | Modèle | Exactitude | F1 « agit / n'agit pas » | Bon type quand il agit |", "|---|---|---|---|---|"]
    for sortie in ("entraine", "construct_ouvriers"):
        for n, m in res[sortie].items():
            L.append(f"| {sortie} | {n} | {pct(m['exactitude'])} | {pct(m['f1_oui_non'])} | "
                     f"{pct(m.get('bon_choix_quand_il_agit', 0))} |")
    L += [f"\nTour entièrement juste (toutes les unités et tous les bâtiments) : {pct(res['tour_entierement_juste'])}",
          f"\nModèle complet (niveaux 1 + 2) : {res['modele']['parametres']:,}".replace(",", " ")
          + f" paramètres, {res['modele']['duree_entrainement_s']} s d'entraînement sur CPU."]
    chemin = C.DOSSIER_RESULTATS / "resultats.md"
    texte = chemin.read_text(encoding="utf-8") if chemin.exists() else ""
    texte = texte.split("\n# Niveau 2")[0].rstrip() + "\n" + "\n".join(L) + "\n"
    chemin.write_text(texte, encoding="utf-8")


if __name__ == "__main__":
    main()
