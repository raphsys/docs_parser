# Feuille De Route - Audit Reconstruction En 6 Portes

## Objet

Ce document fixe un protocole fiable, concret et detaille pour auditer toute la chaine :

1. extraction
2. appartenance au bloc
3. qualite des elements extraits
4. traduction
5. placement dans la reconstruction
6. qualite du fond

L'objectif n'est pas d'avoir "une impression correcte".

L'objectif est :

- de decomposer le probleme proprement ;
- d'isoler la cause exacte d'un echec ;
- de produire des preuves a chaque etape ;
- d'exiger `100%` de conformite sur chaque porte ;
- de disposer d'une feuille de route utilisable par plusieurs agents ou outils.

## Position de principe

Le moteur de reconstruction doit respecter l'algorithme cible :

- les blocs restent fixes ;
- les elements extraits sont pris dans l'ordre du bloc ;
- les phrases sont rendues comme phrases completes ;
- les spans / expressions hors phrases sont rendus comme spans / expressions ;
- les nombres sont rendus comme les memes nombres ;
- le placement respecte la geometrie relative ;
- les attributs typographiques sont conserves.

Mais tant que les 6 portes ci-dessous ne sont pas validees, on ne peut pas conclure que la reconstruction est correcte.

En pratique :

- un rendu raté peut venir de l'extraction ;
- un rendu raté peut venir des blocs ;
- un rendu raté peut venir de la segmentation phrase / expression ;
- un rendu raté peut venir de la traduction ;
- un rendu raté peut venir du placement ;
- un rendu raté peut venir du fond.

Le protocole doit separer ces causes.

## Regle generale d'audit

Pour chaque page auditée, il faut produire :

- un verdict par porte
- un score
- une liste des ecarts
- une preuve visuelle
- une preuve structurelle
- une conclusion binaire : `PASS` ou `FAIL`

Un seul `FAIL` signifie :

- la page n'est pas conforme
- l'etape suivante ne doit pas masquer ou compenser artificiellement l'etape precedente

Autrement dit :

- on ne corrige pas un mauvais fond avec un meilleur texte
- on ne corrige pas une mauvaise extraction avec un meilleur placement
- on ne corrige pas une mauvaise appartenance au bloc avec une meilleure traduction

Chaque porte doit etre vraie en elle-meme.

## Vue d'ensemble des 6 portes

### Porte 1 - Extraction

Question :

- Tous les elements textuels visibles de la page originale sont-ils extraits ?

### Porte 2 - Appartenance au bloc

Question :

- Chaque element textuel extrait est-il rattache au bon bloc ?

### Porte 3 - Qualite des elements extraits

Question :

- Les elements extraits sont-ils bien circonscrits et semantiquement complets ?

### Porte 4 - Traduction

Question :

- Chaque element extrait est-il correctement traduit ?

### Porte 5 - Placement dans la reconstruction

Question :

- Tous les elements attendus se retrouvent-ils dans le rendu final, bien places selon l'algorithme ?

### Porte 6 - Fond

Question :

- Le fond sans texte est-il propre, complet, et correctement reutilise au rendu ?

---

## Porte 1 - Extraction

### Question exacte

- Tous les elements textuels visibles de la page originale ont-ils ete extraits ?

### Distinction explicite a auditer

Il faut distinguer :

- texte visible dans la page
- texte detecte par OCR
- texte structurellement extrait en unite exploitable

La Porte 1 ne demande pas seulement :

- "un OCR a vu quelque chose"

Elle demande :

- "chaque contenu textuel visible utile a la reconstruction existe comme unite exploitable en aval"

Autrement dit :

- un texte present seulement dans un gros champ global n'est pas forcement une extraction valide ;
- une extraction n'est utile que si elle peut ensuite etre rattachee, traduite et rendue.

### Ce qui doit etre couvert

- titres
- sous-titres
- paragraphes
- labels
- captions
- items de liste
- textes de tableaux
- annotations
- textes dans figures si textuels et lisibles
- numéros
- leaders et labels de sommaire si structurellement textuels
- texte inline
- mots isoles significatifs

### Ce qui ne doit pas etre considere comme manque d'extraction

- bruit graphique non textuel
- trames decoratives
- micro-artefacts de compression
- texture d'image ne portant pas un contenu textuel exploitable

### Ce qu'il faut verifier

1. Presence exhaustive
   Chaque texte visible doit avoir au moins une unite extraite correspondante.

2. Couverture sans trou
   Aucun morceau de texte lisible ne doit rester sans correspondant.

3. Non duplication abusive
   Un meme texte ne doit pas etre extrait 4 fois sous 4 unites redondantes si cela n'a pas de sens structurel.

4. Exploitabilite structurale
   Le texte extrait doit exister sous une unite exploitable par les portes suivantes.

5. Cohérence multi-niveaux
   Si un texte est visible, il doit etre coherent entre :
   - bloc
   - ligne
   - phrase
   - span/expression

6. Exhaustivite locale
   Dans chaque zone textuelle, l'ensemble des extractions doit recouvrir tout le contenu utile, pas seulement une partie representative.

### Score exige

- `100%`

### Mesures a produire

- `visible_text_units_total`
- `extracted_text_units_total`
- `visible_text_units_covered`
- `visible_text_units_missing`
- `visible_text_units_not_exploitable`
- `visible_text_units_partially_structured`
- `multi_level_consistency_score`
- `coverage_score`

### Preuves attendues

- image originale
- overlays bbox extraction
- tableau : `texte visible -> unite(s) extraites correspondantes`
- tableau : `texte visible -> bloc/ligne/phrase/span`
- vue des zones visibles sans correspondance exploitable

### Types d'echec

- texte visible non extrait
- texte partiellement extrait
- texte extrait sous forme invalide ou vide
- texte vu par OCR mais non transformé en unite exploitable
- texte visible present seulement dans un champ global inutilisable
- incoherence entre niveaux d'extraction pour une meme zone

### Instrumentation a implementer

- export des bboxes source
- revue manuelle assistee
- rapport par zone
- tableau de couverture multi-niveaux
- detecteur `visible but not structurally usable`

### Reflexion approfondie

Cette porte est plus difficile qu'un simple OCR recall.

Le vrai enjeu n'est pas :

- "le texte existe quelque part dans le payload"

Le vrai enjeu est :

- "chaque unite textuelle pertinente de la page visible est presente sous une forme exploitable par le moteur aval"

Un faux positif frequent est :

- l'information semble presente globalement dans le bloc,
- mais elle n'existe pas comme unite exploitable.

Pour la reconstruction, cela revient a une absence d'extraction.

Donc la couverture doit etre evaluee au niveau de l'unite utile, pas seulement au niveau d'un gros `block.text`.

Point critique :

une extraction "presente" mais inutilisable structurellement est une extraction ratee.

La bonne question n'est donc pas seulement :

- "le texte est-il quelque part dans le payload ?"

Mais :

- "le texte est-il present sous une forme utilisable par le reste de la chaine ?"

---

## Porte 2 - Appartenance au bloc

### Question exacte

- Chaque element extrait est-il place dans le bon bloc logique et spatial ?

### Distinction explicite a auditer

Il faut distinguer :

- l'existence d'un bloc
- la qualite de definition du bloc
- le rattachement `element -> bloc`

La Porte 2 ne juge pas encore si le bloc est "excellent" au sens de la Porte 3.
Elle juge si l'unite extraite appartient au bon parent structurel.

Autrement dit :

- Porte 2 = verite d'appartenance
- Porte 3 = qualite du bloc, du cadre rouge et de l'unite

Il faut garder cette separation stricte.

### Ce qu'il faut verifier

1. Bloc parent correct
   Chaque unite textuelle doit appartenir au bloc qui l'encadre reellement dans la page.

2. Pas de fuite inter-bloc
   Un texte d'un bloc ne doit pas etre rattache au bloc voisin.

3. Integrite des blocs
   Un bloc ne doit pas englober du texte qui appartient visuellement a un autre bloc.

4. Rattachement univoque
   Une unite ne doit pas etre implicitement rattachee a plusieurs blocs concurrents.
   En cas d'ambiguite, elle doit etre marquee comme ambiguë, pas acceptee silencieusement.

5. Cohérence avec la structure locale
   Le bloc choisi doit rester coherent avec :
   - la colonne
   - la region
   - l'ordre de lecture
   - le role du texte

6. Cohérence inter-niveaux
   Si une phrase appartient a un bloc, les spans, runs, groupes et lignes associes doivent etre compatibles avec ce bloc.

### Score exige

- `100%`

### Mesures a produire

- `units_correctly_assigned_to_block`
- `units_wrong_block`
- `units_ambiguous_block`
- `units_multi_candidate_block`
- `units_hierarchy_block_inconsistent`
- `block_assignment_score`

### Preuves attendues

- overlay des blocs
- overlay des unites
- tableau : `unit_id -> block_id attendu -> block_id extrait`
- tableau : `unit_id -> candidats bloc -> bloc retenu -> justification`
- vue debug des cas ambigus

### Types d'echec

- unite rattachee au bloc au-dessus
- unite rattachee a un bloc trop large
- bloc qui absorbe deux colonnes ou deux groupes distincts
- unite rattachee au bloc voisin alors que son placement et son voisinage indiquent un autre parent
- phrase rattachee a un bloc mais ses sous-elements a un autre
- unite affectee au premier bloc intersecte sans verification structurelle

### Instrumentation a implementer

- inspecteur bloc + unites
- matrice `unit -> block`
- distance / overlap / containment debugging
- top-k blocs candidats par unite
- justification du bloc retenu
- drapeau `ambiguous_assignment`

### Reflexion approfondie

Cette porte est critique parce qu'un bloc mal defini contamine toute la suite :

- les line templates sont faux
- le reflow est faux
- la traduction peut etre rattachee au mauvais contexte
- le placement final est faux meme si l'algorithme de rendu est bon

Un tres bon moteur de rendu ne peut pas corriger un mauvais bloc parent.

Donc si cette porte n'est pas a `100%`, il ne sert a rien d'optimiser le layout final.

Point fondamental :

le rattachement `element -> bloc` ne doit jamais etre un simple effet secondaire du premier overlap trouve.

Il doit etre un choix explicable, base sur :

- containment
- overlap utile
- proximite
- colonne
- ordre de lecture
- coherence semantique locale

Sinon, des erreurs faibles a ce stade deviennent destructrices au rendu final.

---

## Porte 3 - Qualite des blocs, des cadres rouges et des elements extraits

### Question exacte

- Les blocs, les cadres rouges et les elements extraits sont-ils bien definis, bien circonscrits et semantiquement complets ?

### Ce qu'il faut verifier

1. Qualite des blocs
   Le bloc doit etre bien defini spatialement et structurellement.
   Il ne doit etre ni trop grand, ni trop petit, ni melanger plusieurs zones heterogenes sans justification.

2. Qualite des cadres rouges
   Le cadre rouge de l'element courant doit entourer correctement l'unite extraite.
   Il ne doit pas mordre sur le voisin, ni couper l'unite, ni etre si large qu'il devient ambigu.

3. Circonscription des elements
   La bbox doit correspondre a l'element, sans englober trop ni couper trop.

4. Completude semantique
   Une phrase doit etre complete.
   Une expression doit etre complete.
   Un nombre doit etre complet.

5. Granularite correcte
   On ne doit pas confondre :
   - ligne
   - phrase
   - span
   - groupe semantique

6. Non-chevauchement analytique au niveau du rendu
   Les unites d'analyse peuvent se chevaucher.
   Les unites finales de rendu, elles, ne doivent pas etre ambigues.

### Score exige

- `100%`

### Mesures a produire

- `units_bbox_ok`
- `units_bbox_too_wide`
- `units_bbox_too_narrow`
- `blocks_well_defined`
- `blocks_too_large`
- `blocks_too_fragmented`
- `red_frames_correct`
- `red_frames_ambiguous`
- `semantic_units_complete`
- `semantic_units_fragmented`
- `semantic_units_overlapping_for_render`

### Preuves attendues

- zooms bbox
- overlay des blocs
- overlay des cadres rouges
- table de correspondance phrase / lignes / spans
- export bloc par bloc avec attributs

### Types d'echec

- phrase coupee en deux sans raison
- phrase glissante sur plusieurs lignes qui se chevauche avec sa voisine
- expression inline amputee
- nombre colle a un autre element
- bloc qui absorbe trop d'elements heterogenes
- bloc qui rate une partie evidente de sa zone textuelle
- cadre rouge mal pose sur l'unite

### Instrumentation a implementer

- rapport `block -> phrase -> line -> span`
- audit visuel `block bbox` et `red frame bbox`
- drapeau `renderable_unit` vs `analysis_unit`
- detecteur d'unites chevauchantes au niveau du rendu

### Reflexion approfondie

Cette porte est probablement le pivot central de toute la chaine.

Pourquoi :

- l'extraction peut etre "presente"
- le bloc peut etre "correct"
- mais si les unites internes ne sont pas semantiquement bonnes, la reconstruction restera fausse

Exemple typique :

- phrase analytique chevauchante
- phrase multi-ligne non finalisee
- expression hors phrase mal decoupee
- bloc trop large qui melange plusieurs sous-groupes
- cadre rouge place sur une sous-partie qui n'est pas l'unite complete

Dans ce cas, la reconstruction ne sait pas quoi dessiner.

Le systeme a besoin de deux niveaux distincts :

- unites d'analyse
- unites de rendu final

Cette separation doit etre explicite.

Il faut aussi separer explicitement :

- qualite du bloc bleu
- qualite du cadre rouge
- qualite de l'unite semantique

Car un bloc peut etre mauvais alors que l'unite locale parait correcte,
et inversement un bloc peut etre bon mais le cadre rouge etre mal place.

Les trois doivent etre audites distinctement.

---

## Porte 4 - Traduction

### Question exacte

- Chaque element extrait est-il correctement traduit, element par element ?

### Distinction explicite a auditer

Il faut distinguer :

- presence d'une traduction
- exactitude semantique de la traduction
- respect des exemptions et protections
- compatibilite de la traduction avec le rendu

La Porte 4 ne s'arrete donc pas a :

- "un texte traduit existe"

Elle doit verifier aussi :

- "la bonne unite a la bonne traduction, dans la bonne forme, avec le bon statut de protection"

### Ce qu'il faut verifier

1. Presence de la traduction
   Chaque unite traduisible doit avoir une traduction.

2. Exactitude
   La traduction doit etre correcte semantiquement.

3. Respect des exemptions
   Le code ne doit pas etre traduit.
   Les nombres ne doivent pas etre modifies.
   Les references protegees doivent rester intactes.

4. Non fuite d'anglais
   Une unite traduite ne doit pas rester en anglais sauf cas protege explicite.

5. Fidélité de granularité
   La traduction doit respecter la granularite de l'unite :
   - phrase par phrase
   - expression par expression
   - nombre par nombre

6. Compatibilite avec le rendu
   La traduction produite doit etre compatible avec la reconstruction :
   - pas de pollution inline
   - pas de fusion abusive
   - pas de traduction invalide pour un element protege

7. Traçabilité
   Il doit etre possible de relier explicitement :
   - unite source
   - unite traduite
   - statut de protection
   - regle appliquee

### Score exige

- `100%`

### Mesures a produire

- `translatable_units_total`
- `translated_units_total`
- `missing_translation_units`
- `english_leak_units`
- `protected_units_correctly_preserved`
- `wrong_granularity_translation_units`
- `render_incompatible_translation_units`
- `translation_traceability_score`

### Preuves attendues

- tableau : `source -> translated -> type -> preserve?`
- echantillons de fuite
- tableau : `unit_id -> source_text -> translated_text -> status`
- justification des unites non traduites ou preservees

### Types d'echec

- texte non traduit
- texte mal traduit
- code traduit
- nombre modifie
- phrase traduite comme fragments incoherents
- span hors phrase absorbe dans une autre traduction
- unite preservee sans justification
- unite traduite alors qu'elle devait etre protegee

### Instrumentation a implementer

- export unite source / unite traduite
- detecteur de fuite d'anglais
- detecteur de protection inline
- rapport de granularite de traduction
- rapport de compatibilite traduction/rendu

### Reflexion approfondie

Cette porte doit etre evaluee au niveau des unites extraites, pas du bloc.

Pourquoi :

- un bloc peut paraitre "traduit globalement"
- mais une seule unite ratee invalide la conformite

La traduction doit donc etre auditee a granularite fine.

Point critique :

une traduction peut etre linguistiquement bonne mais operationnellement mauvaise pour la reconstruction.

Exemple :

- bonne idee semantique
- mauvaise granularite
- mauvaise preservation des nombres
- mauvaise gestion des spans proteges

Donc la Porte 4 doit rester branchee sur les contraintes reelles du moteur aval.

---

## Porte 5 - Placement dans la reconstruction

### Question exacte

- Tous les elements extraits et traduits attendus sont-ils presents dans le rendu final, et places selon l'algorithme ?

### Distinction explicite a auditer

Il faut distinguer :

- presence dans le rendu
- placement du rendu
- conformite a l'algorithme
- fidelite typographique

Une page peut avoir :

- une tres bonne couverture
- mais un mauvais placement

ou :

- un placement global propre
- mais un mauvais niveau d'unites rendues

ou encore :

- les bons textes
- mais dans les mauvais slots relatifs.

La Porte 5 doit auditer tout cela explicitement.

### Ce qu'il faut verifier

1. Presence dans le rendu
   Toute unite attendue doit apparaitre dans le PDF reconstruit.

2. Placement conforme
   L'ordre, les ruptures, les continuations et la position relative doivent respecter l'algorithme.

3. Absence de duplication
   Une unite ne doit pas etre rendue plusieurs fois.

4. Absence de collision
   Pas d'overlap texte/texte anormal.

5. Respect des attributs
   Police, taille, style, couleur.

6. Respect des ancrages relatifs
   L'unite doit respecter :
   - son ancrage horizontal
   - son ancrage vertical
   - sa position relative au bloc
   - sa relation relative aux voisins

7. Respect des ruptures et continuites
   Le moteur doit respecter :
   - continuation inline
   - retour a la ligne
   - rupture de paragraphe
   - keep together

8. Respect du niveau de rendu final
   Une unite analytique ne doit pas etre dessinee si seule l'unite finale doit l'etre.
   Il ne doit pas y avoir de duplication par superposition de niveaux.

9. Respect du slot relatif d'origine
   Quand l'algorithme impose un placement local relatif, le moteur doit deposer l'unite dans sa sous-zone correcte.

### Score exige

- `100%`

### Mesures a produire

- `expected_render_units`
- `rendered_units`
- `missing_rendered_units`
- `duplicated_rendered_units`
- `wrong_order_units`
- `wrong_relative_position_units`
- `style_mismatch_units`
- `wrong_anchor_units`
- `wrong_break_behavior_units`
- `wrong_render_level_units`
- `wrong_relative_slot_units`
- `word_overlaps`
- `text_img_collisions`

### Preuves attendues

- PDF reconstruit
- `pdftotext`
- superposition originale / rendu
- debug progressif unite par unite
- tableau `unit_id -> region attendue -> region rendue`
- tableau `unit_id -> style attendu -> style rendu`
- tableau `unit_id -> voisins attendus -> voisins rendus`

### Types d'echec

- unite absente
- unite dupliquee
- unite dans le mauvais slot
- unite rendue avec mauvais style
- unite rendue au mauvais endroit du bloc
- unite inline rendue comme bloc autonome
- phrase rendue par morceaux alors qu'elle devait etre une unite finale complete
- span hors phrase rendu comme s'il appartenait a une phrase
- numero ou label rendu dans un slot relatif faux

### Instrumentation a implementer

- mode progressif page par page
- mapping `unit_id -> draw ops`
- rapport de couverture du rendu
- rapport de style
- rapport d'ancrages et de slots
- diff ordre attendu / ordre rendu
- diff niveau final attendu / niveau effectivement dessine

### Reflexion approfondie

C'est la porte la plus visible, mais elle ne doit pas absorber les fautes des portes precedentes.

Si cette porte echoue, il faut toujours savoir si la cause est :

- mauvaise unite
- mauvaise appartenance
- mauvaise traduction
- mauvais solveur de placement
- mauvais dessin

Cette porte doit donc etre auditee avec une trace fine du pipeline final.

Le point critique ici est :

le moteur peut facilement donner l'illusion qu'il respecte l'algorithme parce qu'il dessine les bons textes dans les bons blocs.

Mais pour etre conforme, il faut aussi :

- le bon niveau d'unite
- le bon ordre
- la bonne continuite
- le bon slot relatif
- le bon style
- la bonne reutilisation du fond

La Porte 5 n'est donc pas seulement :

- "le texte est present"

Elle est :

- "l'unite attendue est dessinee exactement comme l'algorithme l'impose"

---

## Porte 6 - Fond

### Question exacte

- Le fond du document sans les elements extraits est-il propre et bien reutilise au rendu ?

### Distinction explicite a auditer

Il faut distinguer :

- propreté du fond nettoye
- fidelite visuelle du fond
- reutilisation correcte du fond dans le rendu final

La Porte 6 ne demande pas seulement :

- "il n'y a plus de texte"

Elle demande aussi :

- "la page reste visuellement saine une fois les elements extraits retires"

et

- "ce fond est reutilise correctement dans les zones reconstruites"

### Ce qu'il faut verifier

1. Proprete du fond
   Aucun residu textuel d'origine dans les zones nettoyees.

2. Integrite du fond
   Pas d'aplats artificiels non voulus.

3. Restitution correcte
   Les zones extraites doivent se reposer sur un fond coherent.

4. Compatibilite avec le texte rerendu
   Le fond ne doit pas saboter la lisibilite.

5. Cohérence locale des patches
   Les zones nettoyees ne doivent pas introduire :
   - rupture de texture
   - rupture de ligne graphique
   - halo
   - collage artificiel

6. Cohérence globale de page
   Le fond nettoye doit rester coherent avec la page entiere :
   - marges
   - illustrations
   - trames
   - bandes colorees
   - formes de fond

7. Absence de nettoyage redondant
   Si le fond maitre est deja propre, le moteur ne doit pas le degrader par un second effacement local inutile.

### Score exige

- `100%`

### Mesures a produire

- `background_residual_text_regions`
- `background_artifact_regions`
- `background_patch_mismatch_regions`
- `background_local_texture_breaks`
- `background_global_consistency_score`
- `redundant_cleanup_regions`
- `background_clean_score`

### Preuves attendues

- `bg_master`
- comparaison avec original
- crops avant / apres effacement
- comparaison `fond seul / rendu final`
- zoom sur zones nettoyees sensibles

### Types d'echec

- texte anglais encore visible dans le fond
- rectangle blanc artificiel
- patch incoherent
- texture/illustration massacree
- halo autour des zones recomposees
- second effacement qui detruit un fond deja propre
- rupture locale de continuité visuelle

### Instrumentation a implementer

- export du fond seul
- export des zones nettoyees
- heatmap de residues
- rapport de cohérence locale des patches
- diff `bg_master` vs rendu final dans zones reconstruites

### Reflexion approfondie

Le fond est une vraie porte autonome.

Il ne suffit pas de "faire rentrer le texte" si le fond reste sale.

Un fond sale produit :

- ghost text
- halos
- fonds blancs
- impression de page artificielle

Cette porte doit etre verifiee explicitement.

Point critique :

un fond peut etre "propre" au sens OCR et rester mauvais au sens visuel.

Inversement, un fond peut etre globalement bon mais etre re-detruit pendant la reconstruction par :

- whiteout redondant
- patch mal replace
- effacement local excessif

La Porte 6 doit donc verifier :

- le fond avant reconstruction
- le fond pendant la reconstruction
- le fond apres reconstruction

et pas seulement l'etat initial du `bg_master`.

---

## Grille de validation finale

Pour qu'une page soit declaree conforme, il faut :

- Porte 1 = `100%`
- Porte 2 = `100%`
- Porte 3 = `100%`
- Porte 4 = `100%`
- Porte 5 = `100%`
- Porte 6 = `100%`

Sinon :

- la page est `FAIL`
- la cause doit etre attribuee a une ou plusieurs portes

## Ordre de travail recommande

### Phase A - Verite de l'extraction

1. auditer Porte 1
2. auditer Porte 2
3. auditer Porte 3

Pourquoi :

- sans elles, le reste n'est pas interpretable

### Phase B - Verite linguistique

4. auditer Porte 4

Pourquoi :

- on ne peut pas auditer proprement le rendu d'un texte mal traduit

### Phase C - Verite de la reconstruction

5. auditer Porte 5
6. auditer Porte 6

Pourquoi :

- c'est la phase finale

## Outils a produire

### 1. Rapport global par page

Format cible :

- `audit_page_<page>.json`
- `audit_page_<page>.md`

### 2. Overlays d'audit

- original + bboxes
- original + blocs
- fond nettoye
- rendu final
- diff original / rendu

### 3. Tableaux analytiques

- `unit_id`
- `block_id`
- `line_id`
- `type`
- `source_text`
- `translated_text`
- `rendered_text`
- `bbox`
- `attributes`
- `status` par porte

### 4. Debug progressif

Le mode progressif deja produit doit devenir un outil standard de Porte 5.

### 5. Audit du fond

Il faut un export dedie :

- original
- zones extraites
- `bg_master`
- rendu final

## Livrables attendus

### Document de reference

Ce document est la reference de feuille de route.

### Scripts a produire ensuite

- `scripts/audit_reconstruction_pipeline.py`
- `scripts/audit_extraction_blocks.py`
- `scripts/audit_translation_units.py`
- `scripts/audit_render_coverage.py`
- `scripts/audit_background_cleanliness.py`

## Conclusion

La bonne strategie n'est plus :

- "corriger le rendu en aveugle"

La bonne strategie est :

- auditer les 6 portes
- exiger `100%` sur chacune
- ne jamais laisser une etape compenser artificiellement une autre

Le systeme sera considere fiable uniquement quand :

- l'extraction est exhaustive
- les blocs sont corrects
- les unites sont semantiquement saines
- la traduction est correcte
- le placement suit l'algorithme
- le fond est propre

Tant qu'une seule de ces portes n'est pas a `100%`, la page n'est pas validee.
