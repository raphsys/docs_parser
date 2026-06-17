# Verdict

Ces résultats changent le statut du projet.

On peut maintenant dire :

```text id="b6sdje"
PAGEPRINT / PAGETRANSLATE : validés structurellement sur 10 pages réelles.
Runtime CT2 / OPUS : fonctionne réellement.
Qualité de traduction brute : partielle, instable sur fragments.
Pipeline prêt pour essais contrôlés : oui.
Pipeline prêt pour documents complets sans garde-fous : non.
```

La conclusion importante :

```text id="zkrsta"
Le problème principal n’est plus l’architecture PAGEPRINT/PAGETRANSLATE.
Le problème principal devient la préparation linguistique des unités avant moteur.
```

L’audit montre que le modèle OPUS-MT est bien branché et traduit, mais il reçoit encore certains segments trop fragmentés, césurés ou non documentaires. Donc la prochaine étape n’est pas une nouvelle grande refonte. C’est une **phase de nettoyage pré-traduction + QA moteur**.

---

# 1. Ce que ces résultats valident

L’audit a été fait sur 10 pages réelles, avec `CTranslate2` et le modèle `opus_mt_tc_big_en_fr`, via le script `tools/run_pageprint_pagetranslate_audit.py`. Le pipeline exécuté est bien : PDF natif → `PipelineOrchestrator.run()` → `PAGEPRINT` → `PAGETRANSLATE` + moteur CT2 → audit visuel et fonctionnel. La durée est annoncée à environ 12–40 secondes par page, contre environ 130 secondes par page dans l’ancien essai. 

Les métriques structurelles sont très bonnes :

```text id="x3o8w2"
role_none_translation_units      : 0 / 193
semantic_system_empty_pages      : 0 / 10
generic_coalesced_units          : 0
natural_text_marked_protected    : 0
code_or_command_sent_to_translation : 0
```

C’est le seuil que nous voulions atteindre. L’audit indique explicitement que `PAGETRANSLATE` n’a plus besoin de son coalescer générique et qu’il consomme la sémantique de `PAGEPRINT` au lieu de l’inventer. 

Donc décision ferme :

```text id="r2s73m"
Ne pas refaire PAGEPRINT/PAGETRANSLATE.
Ne pas revenir à selector/coalescer.
Ne pas relancer une refonte conceptuelle.
```

---

# 2. Le vrai changement de phase

Avant, on se demandait :

```text id="xio2c3"
Est-ce que PAGEPRINT donne les bonnes unités ?
Est-ce que PAGETRANSLATE consomme translation_plan ?
Est-ce que le moteur est branché ?
```

Maintenant, la réponse est globalement :

```text id="01xatz"
oui.
```

La nouvelle question est :

```text id="6greoy"
Est-ce que les unités envoyées au moteur sont linguistiquement traduisibles ?
```

Et là, la réponse est :

```text id="9meap5"
pas encore toujours.
```

C’est un changement important. Le pipeline n’est plus en échec architectural ; il est en phase d’**assainissement linguistique pré-moteur**.

---

# 3. Ce qui marche vraiment

## 3.1 Prose propre

Sur les phrases complètes, OPUS-MT produit des traductions correctes :

```text id="vg5n14"
8.3 Deep Learning Architectures for Character Recognition
→ 8.3 Architectures d'apprentissage profond pour la reconnaissance des caractères

8.3.1 Unsupervised Pretraining
→ 8.3.1 Préformation non supervisée

samples from USPS dataset
→ échantillons provenant de l'ensemble de données USPS
```

L’audit dit que sur des phrases complètes, la qualité est bonne, le registre est correct, et les tokens protégés comme les datasets ou sigles sont conservés. 

Donc le moteur n’est pas mauvais en soi.

## 3.2 Structure de sélection

Les rôles sémantiques sont présents :

```text id="4nziyc"
table_body_cell     : 62
body_paragraph      : 59
toc_entry_title     : 32
index_head_term     : 30
table_header_cell   : 4
figure_caption      : 3
list_item           : 3
```

Cela prouve que `PAGEPRINT` fournit désormais des unités typées et que `PAGETRANSLATE` ne traduit plus à l’aveugle. 

---

# 4. Ce qui ne marche pas encore

## 4.1 `functional_status = ko` à cause des publisher marks

L’audit dit :

```text id="skfzg0"
schema_status     : ok
functional_status : ko
```

La raison fonctionnelle est très ciblée :

```text id="sawiu2"
publisher_mark_sent_to_translation = 2
```

Le texte concerné est :

```text id="bjtj72"
Estadísticos e-Books & Papers
```

Il est encore envoyé au traducteur sur Practical SQL p432 et p069. L’audit indique que ce point est le dernier échec structurel explicite du `functional_status`. 

Conclusion :

```text id="zzc2ni"
C’est un P0, mais facile à corriger.
```

Il faut finaliser `publisher_mark_detector`.

---

## 4.2 Les césures cassent la traduction

C’est le défaut le plus dangereux pour la qualité linguistique.

Exemple :

```text id="utxefk"
interest in deep learning, but much research needs to be done to improve the unsu-
```

Le modèle traduit hors sujet :

```text id="3n7lz2"
La recherche sur l'éducation et la formation tout au long de la vie...
```

L’audit précise que les fragments trop courts ou coupés en fin de ligne (`unsu-`, `nor-`) sont envoyés tels quels au moteur, et que ce n’est pas seulement une faute du moteur : ces unités n’auraient pas dû arriver fragmentées. 

Conclusion :

```text id="cjt52x"
La dé-césure devient P0.
```

Sans dé-césure, un bon moteur hallucine.

---

## 4.3 Les cellules de table sont trop fragmentées

Sur p103, le moteur reçoit des segments comme :

```text id="u647bf"
5.7 Challenges and Future Research Direction
interest in deep learning, but much research needs to be done to improve the unsu-
pervised deep learning algorithms.
```

Certaines sorties deviennent répétitives ou incohérentes :

```text id="8q2bxv"
Les défis et les défis et les défis
```

L’audit attribue cette instabilité aux fragments de cellules et à la politique table cellule-par-cellule encore à renforcer. 

Conclusion :

```text id="67lzwz"
Il ne suffit pas de détecter table_body_cell.
Il faut reconstruire des segments de traduction linguistiquement complets dans les cellules.
```

---

## 4.4 `page_role` reste trop général

Les 10 pages ont encore :

```text id="aq7vgs"
page_role = body
```

Même quand les unités montrent clairement :

```text id="bma9u8"
index_head_term = 30
toc_entry_title = 32
```

L’audit signale explicitement que la détection de rôle de page n’est pas encore propagée au niveau page, même si les rôles fins par unité sont bien posés. 

Conclusion :

```text id="077bb3"
Ce n’est pas bloquant pour le moteur, mais c’est bloquant pour les politiques globales.
```

Si une page est index ou TOC, les règles de segmentation, QA, traduction et reconstruction doivent changer.

---

# 5. Analyse de fond

Le pipeline a maintenant trois niveaux de qualité distincts.

## Niveau 1 — Structure documentaire

Statut :

```text id="5c2gn8"
très bon
```

Les métriques fondamentales sont corrigées :

```text id="yl9i1z"
0 role_none
0 semantic_system_empty
0 coalescer générique
0 texte naturel protégé à tort
0 code/commande envoyé
```

## Niveau 2 — Préparation linguistique

Statut :

```text id="79otx3"
insuffisant
```

Problèmes :

```text id="z9o628"
césures
segments trop courts
fragments de table
lignes bibliographiques incomplètes
unités sans contexte suffisant
```

## Niveau 3 — Qualité moteur

Statut :

```text id="dbepxn"
partiel
```

OPUS fonctionne bien sur les phrases complètes, mais échoue sur les fragments. L’audit indique `linguistic_quality_status = ko` sur 8 pages sur 10 et `needs_review = 92 / 193`, soit environ 48 %, surtout concentrés sur les pages denses p103 et p509. 

Donc il ne faut pas accuser le moteur trop vite. Il faut d’abord améliorer ce qu’on lui donne.

---

# 6. Décisions

## Décision 1 — On ne bloque plus sur l’architecture

```text id="j5xvcx"
PAGEPRINT/PAGETRANSLATE validés comme socle.
```

## Décision 2 — On ne lance pas encore un document complet

```text id="aheqaa"
Pas encore de document complet en traduction automatique sans contrôle.
```

Mais on peut lancer :

```text id="wt9oeh"
essais page par page
essais 10 pages
essais ciblés par type de page
```

## Décision 3 — Prochaine version = `rev_07.2` ou `rev_08-pretranslation-cleanup`

Objectif :

```text id="92osrv"
nettoyage pré-traduction avant moteur
```

Pas une refonte globale.

## Décision 4 — La priorité n’est plus CT2

CT2 fonctionne suffisamment pour avoir produit cet audit réel. La priorité immédiate n’est donc pas de refaire `ct2_engine.py`, sauf si des logs montrent des erreurs de décodage.

La priorité devient :

```text id="f1zjxx"
publisher marks
dé-césure
recomposition des cellules/fragments
page_role propagation
QA linguistique plus fine
```

---

# 7. Plan de correction prioritaire

# P0 — Exclure les publisher marks

## Objectif

Ne plus envoyer :

```text id="tp8rwy"
Estadísticos e-Books & Papers
```

au moteur.

## Tâches

* [ ] Renforcer `publisher_mark_builder.py`.
* [ ] Utiliser la répétition multi-pages.
* [ ] Utiliser la position bas de page.
* [ ] Utiliser la faible hauteur de police.
* [ ] Utiliser la distance au corps principal.
* [ ] Ajouter une liste optionnelle de patterns d’artefacts connus :

```text id="sn6nlh"
Estadísticos e-Books & Papers
Downloaded from
Generated by
Scanned by
PDFDrive
```

* [ ] Produire :

```json id="zi35ta"
{
  "role": "publisher_mark",
  "preservation_mode": "exclude_as_artifact",
  "translation_mode": "skip"
}
```

* [ ] Ajouter métrique :

```text id="x8nwpu"
publisher_mark_sent_to_translation = 0
```

## Tests

```text id="t0g2an"
test_repeated_footer_excluded
test_practical_sql_estadisticos_excluded
test_publisher_mark_not_in_translation_plan
```

---

# P0 — Ajouter la dé-césure avant `translation_plan`

## Objectif

Transformer :

```text id="wyg7p9"
unsu-
pervised
nor-
malization
classifica-
tion
```

en :

```text id="8kbiqk"
unsupervised
normalization
classification
```

avant l’appel moteur.

## Où l’implémenter

Le meilleur emplacement :

```text id="7lpscu"
pageprint/semantic_builder.py
ou
pageprint/text_normalization.py
```

Mais attention : il faut conserver les `source_unit_ids` originaux pour la reconstruction.

## Design recommandé

Créer :

```text id="9atqw8"
pageprint/text_postprocessors.py
```

Fonctions :

```python id="i0r4u9"
repair_hyphenation_across_units(units_or_segments)
repair_intra_token_spacing(text)
normalize_pdf_ligatures(text)
```

Sortie segment :

```json id="cq10ud"
{
  "source_text_raw": "interest ... improve the unsu-",
  "source_text": "interest ... improve the unsupervised",
  "normalization_applied": ["dehyphenation"],
  "source_unit_ids": [...]
}
```

## Règles

Dé-césurer seulement si :

```text id="vdwgpm"
fin de ligne = lettre + "-"
début ligne suivante = lettres minuscules
même paragraphe / même cellule / même entrée logique
pas de tiret lexical connu
```

Ne pas casser :

```text id="7pymuv"
state-of-the-art
semi-supervised
end-to-end
VGG-16
F-score
```

## Tests

```text id="uvf1cu"
test_dehyphenates_unsu_pervised
test_dehyphenates_classifica_tion
test_keeps_state_of_the_art
test_keeps_vgg_16
test_preserves_source_unit_ids
```

---

# P0/P1 — Recomposer les cellules de table denses

## Objectif

Ne plus envoyer au moteur des fragments isolés de cellules ou paragraphes de table.

## Problème actuel

Le système a bien `table_body_cell`, mais chaque fragment visuel peut devenir une unité linguistique. Résultat :

```text id="gs9jlc"
interest in deep learning, but much research needs to be done to improve the unsu-
pervised deep learning algorithms.
```

est envoyé en morceaux.

## Correction

Créer une étape :

```text id="bn6ox2"
table_translation_segment_builder
```

ou renforcer `semantic_builder.py`.

Règle :

```text id="bcz5lu"
table_cell visual fragments
→ same cell / same row / same paragraph
→ merged translation segment
```

Un segment de table doit être :

```json id="v70q44"
{
  "role": "table_body_cell",
  "semantic_kind": "table_cell_paragraph",
  "source_text": "interest in deep learning, but much research needs to be done to improve the unsupervised deep learning algorithms.",
  "source_unit_ids": ["..."],
  "cell_id": "tbl_001_r3_c2"
}
```

Pas :

```text id="gni74y"
un fragment par ligne visuelle
```

## Tests

```text id="6j8p2a"
test_table_cell_fragments_merge_before_translation
test_table_cell_dehyphenation
test_table_header_number_preserved
test_no_short_fragment_when_same_cell_continuation_exists
```

---

# P1 — Propager `page_role`

## Objectif

Si une page contient majoritairement :

```text id="bc3g6l"
index_head_term
toc_entry_title
table_body_cell
```

alors `page_role` doit refléter ce type.

## Règles simples

```python id="1x7isd"
if index_head_term_count >= 10 and index_head_term_ratio > 0.4:
    page_role = "index"

if toc_entry_title_count >= 10 and toc_entry_title_ratio > 0.4:
    page_role = "toc"

if table_cell_count >= 10 and table_cell_ratio > 0.5:
    page_role = "table_page"
```

## Pourquoi c’est important

Cela permet :

```text id="ak0xnh"
QA spécifique
segmentation spécifique
reconstruction spécifique
politique de traduction spécifique
audit plus sévère
```

## Tests

```text id="pqpgb5"
test_page_role_index_from_index_head_terms
test_page_role_toc_from_toc_entry_titles
test_page_role_table_page_from_table_cells
```

---

# P1 — Améliorer QA linguistique

## Problème

`needs_review = 92 / 193` est élevé. Il faut classer les raisons.

Aujourd’hui, on sait que c’est lié à :

```text id="pqq4pg"
fragments
numbers
protected tokens
unchanged suspect
hallucination
```

Mais il faut un diagnostic plus fin par unité.

## Ajouter catégories

```text id="54kx2h"
qa_reason:
  source_fragment
  dehyphenation_needed
  too_short_for_mt
  number_mismatch
  protected_token_mismatch
  source_leak
  repeated_output
  hallucination_suspected
  unchanged_suspect
  overflow_risk
```

## Heuristiques utiles

### Répétition

```text id="deskte"
Les défis et les défis et les défis
```

Détecter n-grams répétés.

### Fragment source

```text id="w5p4bk"
termine par "-"
commence par minuscule
pas de verbe
longueur trop courte
```

### Hallucination suspecte

Ratio de longueur trop grand + faible recouvrement terminologique.

## Tests

```text id="e63014"
test_detects_repeated_output
test_detects_hyphenated_fragment
test_detects_too_short_fragment
test_classifies_needs_review_reason
```

---

# 8. À ne pas faire maintenant

Ne pas faire :

```text id="1oz76u"
nouvelle refonte globale
nouvelle architecture
changer translation_plan
revenir au coalescer PAGETRANSLATE
changer de modèle IA immédiatement
lancer document complet sans nettoyage
```

Ne pas conclure :

```text id="4gj7bs"
OPUS est mauvais
```

Le modèle fonctionne sur des phrases propres. Les hallucinations viennent surtout de la segmentation/césure.

---

# 9. Plan d’exécution recommandé

## Sprint 1 — `rev_07.2-cleanup`

Objectif :

```text id="u0q8wj"
functional_status ok + moins d’hallucinations
```

Tâches :

```text id="2boh2p"
1. publisher_mark_detector
2. dehyphenation
3. table cell segment merge
4. page_role propagation
5. QA reasons
```

Relancer le même audit 10 pages.

Critères :

```text id="vo6gy4"
publisher_mark_sent_to_translation = 0
functional_status = ok
needs_review < 50 / 193
linguistic_quality_status ko sur moins de 4 pages
aucune hallucination évidente sur fragments césurés
```

## Sprint 2 — `rev_07.3-real-trial`

Objectif :

```text id="bgl7f8"
essais moteur propres sur 10 pages
```

Tâches :

```text id="c5r423"
1. relancer OPUS sur les 10 pages
2. comparer avant/après
3. auditer les 20 pires unités
4. enrichir translation_memory
5. enrichir terminology
```

Critères :

```text id="3wfmjd"
needs_review < 25 %
number_mismatch proche de 0
protected_token_mismatch proche de 0
sorties répétitives détectées ou éliminées
```

## Sprint 3 — document court

Seulement après :

```text id="i32gs8"
functional_status ok
publisher mark = 0
dé-césure opérationnelle
QA raisons détaillées
```

Lancer :

```text id="q13eg3"
5 pages
puis 10 pages
puis document court
```

---

# 10. Réponse directe : “Alors ?”

Alors :

```text id="kb3jqi"
C’est une excellente nouvelle.
```

Ces résultats montrent que le pipeline a franchi un seuil réel : on n’est plus dans la théorie, le moteur CT2 OPUS tourne sur des pages réelles et le plan de traduction est consommé correctement.

Mais :

```text id="7jg9n8"
On n’est pas encore prêt pour traduction de documents complets.
```

On est prêt pour :

```text id="ap8egv"
essais contrôlés sur lots de pages,
corrections de préparation linguistique,
amélioration QA,
puis montée progressive.
```

La prochaine décision est donc :

```text id="uvkxzk"
On fait une petite rev_07.2 ciblée :
publisher marks + dé-césure + fusion de fragments de table + page_role + QA reasons.
```

Après cette correction, on relance exactement le même audit 10 pages. Si le `functional_status` passe à `ok` et que le `needs_review` baisse nettement, on passe aux essais document court.

