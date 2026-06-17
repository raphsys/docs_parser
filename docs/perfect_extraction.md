Oui. Pour une reconstruction **WYSIWYG après traduction**, il ne faut pas seulement extraire le texte. Il faut extraire une **description complète de rendu** : page, coordonnées, ordre d’affichage, styles typographiques, objets graphiques, images, tableaux, blocs, relations logiques, contraintes de mise en page et informations utiles à la recomposition après expansion/réduction du texte traduit.

La vérité technique : une reconstruction parfaite est rarement possible si le document est scanné, si les polices ne sont pas embarquées, si le PDF contient du texte vectorisé, ou si le document source a été aplati en image. Mais on peut viser une fidélité très élevée si la première phase d’extraction est conçue comme un **moteur de capture de document**, pas comme un simple OCR.

Les formats comme PDF décrivent une page avec du texte, des polices, des images, des graphiques vectoriels et un modèle de rendu indépendant du périphérique ; il faut donc extraire à la fois la structure logique et les instructions visuelles de rendu. ([developer.adobe.com][1])

---

# 1. Principe fondamental : extraire deux représentations en parallèle

Tu dois produire **deux couches complémentaires**.

## A. Représentation visuelle absolue

C’est la couche WYSIWYG stricte.

Elle décrit :

* où chaque élément apparaît ;
* sa taille exacte ;
* sa couleur ;
* sa police ;
* son empilement ;
* son angle ;
* son opacité ;
* sa forme ;
* ses marges ;
* ses effets ;
* son comportement dans la page.

Cette couche sert à reconstruire **l’apparence originale**.

## B. Représentation logique / sémantique

Elle décrit :

* titres ;
* paragraphes ;
* listes ;
* tableaux ;
* légendes ;
* notes de bas de page ;
* en-têtes ;
* pieds de page ;
* colonnes ;
* ordre de lecture ;
* liens entre texte et images ;
* styles récurrents ;
* hiérarchie documentaire.

Cette couche sert à traduire correctement et à recomposer intelligemment.

Sans la couche visuelle, tu perds la fidélité.
Sans la couche logique, tu traduis mal et tu reconstruis bêtement.

---

# 2. Métadonnées générales du document

À extraire dès le départ.

## 2.1 Informations de fichier

* nom du fichier ;
* extension ;
* type MIME ;
* format détecté réel ;
* taille du fichier ;
* hash du fichier ;
* date de création du fichier ;
* date de modification ;
* nombre de pages ;
* version du format ;
* logiciel producteur ;
* logiciel créateur ;
* compression utilisée ;
* présence de chiffrement ;
* présence de signature numérique ;
* restrictions d’impression ;
* restrictions de copie ;
* restrictions d’édition ;
* présence de formulaire ;
* présence de calques ;
* présence d’annotations ;
* présence de pièces jointes ;
* présence de métadonnées XMP ;
* présence de structure taguée ;
* présence d’accessibilité ;
* présence d’OCR préexistant ;
* présence de texte invisible ;
* présence de texte vectorisé ;
* présence de texte sous image ;
* présence de pages scannées ;
* présence de polices embarquées ;
* présence de polices substituées ;
* présence d’images haute résolution ;
* présence de transparence ;
* présence d’objets vectoriels complexes.

## 2.2 Métadonnées documentaires

* titre ;
* auteur ;
* sujet ;
* mots-clés ;
* langue déclarée ;
* langues détectées ;
* date de création ;
* date de modification ;
* producteur ;
* application source ;
* version de l’application source ;
* identifiant unique ;
* droits d’auteur ;
* classification ;
* statut du document ;
* modèle utilisé ;
* nombre de révisions ;
* historique si disponible ;
* commentaires internes si disponibles.

## 2.3 Profil global du document

* type probable : rapport, article, livre, facture, contrat, présentation, formulaire, courrier, mémoire, brochure, ordonnance, certificat, tableau, formulaire administratif ;
* orientation dominante ;
* style dominant ;
* nombre de colonnes ;
* présence de tableaux ;
* présence de figures ;
* présence de graphiques ;
* présence de formules ;
* présence de notes ;
* présence de références bibliographiques ;
* présence de pagination ;
* présence d’en-tête ;
* présence de pied de page ;
* présence de filigrane ;
* présence de logo ;
* présence de tampon ;
* présence de signature manuscrite ;
* présence d’écriture manuscrite ;
* niveau de complexité visuelle ;
* niveau de complexité linguistique ;
* qualité estimée du document ;
* qualité estimée de l’OCR ;
* score de reconstructibilité.

---

# 3. Caractéristiques de chaque page

Les pages doivent être extraites comme des objets complets. Les standards de mise en page paginée décrivent notamment la taille de page, les marges, l’orientation, les en-têtes et pieds de page ; ces éléments doivent faire partie du modèle d’extraction. ([W3C][2])

## 3.1 Géométrie de page

Pour chaque page :

* numéro physique ;
* numéro logique affiché ;
* label de page ;
* largeur ;
* hauteur ;
* unité de mesure source ;
* unité normalisée ;
* orientation : portrait, paysage, rotation ;
* angle de rotation ;
* format probable : A4, A5, Letter, Legal, personnalisé ;
* MediaBox ;
* CropBox ;
* BleedBox ;
* TrimBox ;
* ArtBox ;
* zone imprimable ;
* zone visible ;
* zone utile ;
* coordonnées d’origine ;
* système de coordonnées ;
* matrice de transformation globale ;
* résolution cible pour rasterisation ;
* DPI natif si image ;
* ratio largeur/hauteur ;
* marges visibles ;
* marges probables ;
* grille de mise en page probable.

## 3.2 Fond de page

* couleur de fond ;
* image de fond ;
* texture ;
* papier scanné ;
* bruit ;
* ombre ;
* dégradé ;
* filigrane ;
* motif ;
* bordure de page ;
* arrière-plan vectoriel ;
* arrière-plan raster ;
* transparence ;
* couches superposées.

## 3.3 Marges

* marge gauche ;
* marge droite ;
* marge haute ;
* marge basse ;
* marge interne ;
* marge externe ;
* marge de reliure ;
* marge d’en-tête ;
* marge de pied ;
* variations par page paire/impaire ;
* variations première page / pages suivantes ;
* alignement du contenu dans la zone utile.

## 3.4 Colonnes

* nombre de colonnes ;
* largeur de chaque colonne ;
* gouttière entre colonnes ;
* limites des colonnes ;
* ordre de lecture des colonnes ;
* colonnes symétriques ou asymétriques ;
* colonne latérale ;
* encadré latéral ;
* rupture de colonne ;
* texte qui traverse plusieurs colonnes ;
* images insérées dans une colonne ;
* titres sur plusieurs colonnes ;
* notes marginales.

## 3.5 Grille de page

* grille verticale ;
* grille horizontale ;
* baseline grid ;
* pas de ligne dominant ;
* alignement des paragraphes sur grille ;
* zones répétitives ;
* zones de contenu ;
* zones flottantes ;
* zones fixes ;
* zones décoratives ;
* zones ignorables ;
* hiérarchie spatiale.

---

# 4. Ordre de rendu et empilement

Très important. Un document n’est pas seulement une liste d’objets ; c’est une pile d’objets rendus dans un ordre.

À extraire :

* ordre réel des objets dans le flux source ;
* ordre visuel sur la page ;
* ordre logique de lecture ;
* z-index reconstruit ;
* objets en arrière-plan ;
* objets au premier plan ;
* objets masqués ;
* objets partiellement masqués ;
* objets recouverts ;
* objets transparents ;
* objets en clipping ;
* objets dans un groupe ;
* objets dans un calque ;
* objets héritant d’un état graphique ;
* objets répétés ;
* objets décoratifs ;
* objets sémantiques.

Pour chaque objet :

* identifiant unique ;
* type ;
* page ;
* parent ;
* enfants ;
* bbox visible ;
* bbox réelle ;
* bbox de clipping ;
* matrice de transformation ;
* rotation ;
* échelle X ;
* échelle Y ;
* cisaillement ;
* translation X ;
* translation Y ;
* opacité ;
* mode de fusion ;
* masque ;
* clip path ;
* ordre de rendu.

---

# 5. Segmentation complète de la page

La segmentation doit se faire à plusieurs niveaux.

## 5.1 Niveau page

* page ;
* zones principales ;
* zones secondaires ;
* arrière-plan ;
* en-tête ;
* pied de page ;
* marge gauche ;
* marge droite ;
* corps principal ;
* blocs flottants ;
* tableaux ;
* images ;
* légendes ;
* notes ;
* signatures ;
* tampons ;
* annotations.

## 5.2 Niveau région

Chaque région doit avoir :

* type de région ;
* bbox ;
* polygone exact ;
* orientation ;
* colonne associée ;
* rôle probable ;
* score de confiance ;
* parent ;
* enfants ;
* ordre de lecture ;
* dépendances ;
* relation avec autres régions.

Types de régions :

* titre principal ;
* sous-titre ;
* paragraphe ;
* liste ;
* tableau ;
* figure ;
* graphique ;
* image ;
* légende ;
* note ;
* citation ;
* encadré ;
* alerte ;
* résumé ;
* signature ;
* cachet ;
* logo ;
* filigrane ;
* formule mathématique ;
* code source ;
* référence bibliographique ;
* pied de page ;
* en-tête ;
* numéro de page ;
* champ de formulaire ;
* case à cocher ;
* ligne de saisie ;
* tampon ;
* texte manuscrit.

## 5.3 Niveau bloc

Pour chaque bloc :

* identifiant ;
* type ;
* bbox ;
* polygone ;
* orientation ;
* angle ;
* largeur ;
* hauteur ;
* coordonnées absolues ;
* coordonnées relatives à la page ;
* coordonnées relatives à la colonne ;
* style dominant ;
* langue dominante ;
* direction d’écriture ;
* nombre de lignes ;
* nombre de mots ;
* nombre de caractères ;
* densité du texte ;
* alignement ;
* indentation ;
* espacements ;
* couleur dominante ;
* hiérarchie probable ;
* rôle documentaire ;
* ordre de lecture ;
* score de confiance.

## 5.4 Niveau ligne

Pour chaque ligne :

* texte brut ;
* texte normalisé ;
* bbox ;
* baseline ;
* ascender line ;
* descender line ;
* cap height line ;
* x-height line ;
* angle ;
* pente ;
* direction ;
* espacement avant ;
* espacement après ;
* hauteur de ligne ;
* interligne réel ;
* alignement ;
* justification ;
* indentation ;
* retrait suspendu ;
* débordement ;
* césure en fin de ligne ;
* espace de fin ;
* ponctuation finale ;
* style dominant ;
* changement de style interne ;
* langue ;
* script ;
* score OCR ;
* score de segmentation.

## 5.5 Niveau mot

Pour chaque mot :

* texte ;
* texte source ;
* texte corrigé ;
* bbox ;
* quadrilatère ;
* baseline ;
* langue ;
* script ;
* police ;
* taille ;
* style ;
* couleur ;
* espacement avant ;
* espacement après ;
* confiance OCR ;
* rôle : mot normal, nombre, unité, abréviation, nom propre, référence, code, URL, email, formule, sigle ;
* ne_pas_traduire ;
* traduisible ;
* sensible à la casse ;
* contient ponctuation ;
* contient diacritiques ;
* contient ligature ;
* contient césure ;
* appartient à une ligne ;
* appartient à un paragraphe ;
* ordre dans la ligne.

## 5.6 Niveau caractère

Obligatoire pour WYSIWYG sérieux.

Pour chaque caractère :

* caractère Unicode ;
* code point ;
* caractère source ;
* caractère normalisé ;
* index dans le mot ;
* index dans la ligne ;
* bbox ;
* quadrilatère ;
* largeur ;
* hauteur ;
* avance horizontale ;
* avance verticale ;
* position X ;
* position Y ;
* baseline ;
* police ;
* taille ;
* couleur ;
* opacité ;
* rotation ;
* style ;
* ligature associée ;
* caractère combinant ;
* diacritique ;
* direction bidi ;
* script ;
* score OCR ;
* origine : texte natif, OCR, correction, reconstruction.

## 5.7 Niveau glyphe

Encore plus précis que le caractère.

À extraire si possible :

* glyph id ;
* nom du glyphe ;
* CID ;
* GID ;
* code caractère original ;
* mapping ToUnicode ;
* Unicode final ;
* police source ;
* sous-police ;
* subset font ;
* advance width ;
* left side bearing ;
* right side bearing ;
* glyph bbox ;
* position glyphique ;
* cluster HarfBuzz ;
* substitution GSUB appliquée ;
* positionnement GPOS appliqué ;
* kerning ;
* ligature ;
* marque combinante ;
* ancrage ;
* offset X ;
* offset Y ;
* scale ;
* rotation ;
* matrice texte ;
* matrice de rendu.

---

# 6. Texte : contenu linguistique et structure

Unicode impose des règles complexes de segmentation, de césure, de direction bidirectionnelle et de coupure de lignes. Pour reconstruire un texte traduit, il faut garder ces informations, pas seulement la chaîne de caractères. ([Unicode][3])

## 6.1 Texte brut

* texte exact extrait ;
* texte visible ;
* texte invisible ;
* texte caché ;
* texte OCR ;
* texte natif ;
* texte alternatif ;
* texte d’annotation ;
* texte de champ formulaire ;
* texte de métadonnée ;
* texte dans image ;
* texte dans graphique ;
* texte vectorisé reconnu ;
* texte manuscrit reconnu ;
* texte barré ;
* texte souligné ;
* texte en exposant ;
* texte en indice ;
* texte vertical ;
* texte incliné ;
* texte courbé ;
* texte sur chemin.

## 6.2 Normalisation textuelle

À conserver en double :

* version brute ;
* version Unicode normalisée NFC ;
* version NFD ;
* version avec ligatures développées ;
* version sans césure ;
* version avec césure conservée ;
* version corrigée OCR ;
* version prête pour traduction ;
* version reconstruite.

Ne jamais écraser le texte source.

## 6.3 Langues

Pour chaque document, page, bloc, ligne, mot :

* langue détectée ;
* langue déclarée ;
* score de langue ;
* alternance de langues ;
* mots étrangers ;
* citations étrangères ;
* noms propres étrangers ;
* langue de l’interface ;
* langue des références ;
* langue des tableaux ;
* langue des légendes ;
* langue des notes.

## 6.4 Scripts

* latin ;
* arabe ;
* hébreu ;
* grec ;
* cyrillique ;
* chinois ;
* japonais ;
* coréen ;
* devanagari ;
* thaï ;
* scripts africains ;
* symboles ;
* emoji ;
* mathématique ;
* phonétique.

## 6.5 Direction d’écriture

* gauche vers droite ;
* droite vers gauche ;
* vertical ;
* mixte ;
* bidirectionnel ;
* nombres dans texte RTL ;
* ponctuation dans texte RTL ;
* parenthèses inversées ;
* segments neutres ;
* ordre logique ;
* ordre visuel.

## 6.6 Segmentation linguistique

* phrases ;
* paragraphes ;
* titres ;
* sous-titres ;
* mots ;
* tokens ;
* ponctuation ;
* abréviations ;
* dates ;
* nombres ;
* unités ;
* noms propres ;
* sigles ;
* acronymes ;
* citations ;
* références ;
* expressions figées ;
* formules ;
* codes ;
* variables ;
* URL ;
* emails ;
* numéros de téléphone ;
* identifiants ;
* montants ;
* pourcentages.

## 6.7 Éléments à ne pas traduire

Marquer explicitement :

* noms propres ;
* noms d’institutions ;
* noms de médicaments ;
* marques ;
* références bibliographiques ;
* citations juridiques ;
* articles de loi ;
* codes ;
* variables ;
* noms de fichiers ;
* chemins informatiques ;
* URL ;
* emails ;
* identifiants ;
* numéros ;
* unités scientifiques ;
* formules ;
* symboles ;
* abréviations non traduisibles ;
* signatures ;
* tampons ;
* logos ;
* textes décoratifs.

---

# 7. Typographie complète

C’est un bloc critique. OpenType ne se limite pas à une police et une taille : les tables GSUB, GPOS, BASE, JSTF et GDEF interviennent dans les substitutions de glyphes, le positionnement, les baselines, la justification et la définition des glyphes. Les fontes variables ajoutent des axes continus comme le poids ou la largeur. ([Microsoft Learn][4])

## 7.1 Identification de police

Pour chaque span de texte :

* nom de police déclaré ;
* nom réel ;
* nom PostScript ;
* famille ;
* sous-famille ;
* style ;
* poids ;
* largeur ;
* pente ;
* version ;
* fabricant ;
* police embarquée ou non ;
* police subset ;
* préfixe de subset ;
* fichier de police extrait ;
* type : TrueType, OpenType, Type1, Type3, CIDFont, CFF, CFF2 ;
* encodage ;
* ToUnicode map ;
* cmap ;
* glyph coverage ;
* fallback probable ;
* police substituée ;
* police équivalente disponible ;
* score d’identification de police ;
* hash de la police ;
* licence si disponible.

## 7.2 Taille et métriques

* font size déclarée ;
* font size visuelle ;
* taille effective après transformation ;
* ascender ;
* descender ;
* cap height ;
* x-height ;
* line gap ;
* em size ;
* units per em ;
* baseline ;
* avance moyenne ;
* largeur moyenne ;
* hauteur moyenne ;
* correction optique ;
* scale horizontal ;
* scale vertical ;
* text rise ;
* superscript offset ;
* subscript offset.

## 7.3 Style typographique

* normal ;
* gras ;
* semi-bold ;
* light ;
* extra-light ;
* black ;
* italique ;
* oblique ;
* small caps ;
* all caps ;
* underline ;
* double underline ;
* overline ;
* strikethrough ;
* shadow ;
* outline ;
* emboss ;
* engrave ;
* highlight ;
* subscript ;
* superscript ;
* couleur ;
* fond de texte ;
* transparence ;
* contour ;
* épaisseur du contour ;
* remplissage ;
* mode de rendu texte.

## 7.4 Espacement typographique

* letter spacing ;
* word spacing ;
* character spacing ;
* tracking ;
* kerning ;
* leading ;
* interligne ;
* espace avant paragraphe ;
* espace après paragraphe ;
* retrait première ligne ;
* retrait gauche ;
* retrait droit ;
* retrait suspendu ;
* tabulations ;
* positions de tabulation ;
* alignement sur grille ;
* justification ;
* expansion/compression ;
* espaces insécables ;
* espaces fines ;
* espaces cadratins ;
* espaces demi-cadratins ;
* espaces multiples ;
* espaces de début ;
* espaces de fin.

## 7.5 Fonctionnalités OpenType

À extraire ou au minimum inférer :

* ligatures standard ;
* ligatures discrétionnaires ;
* ligatures historiques ;
* kerning ;
* petites capitales ;
* capitales titrées ;
* chiffres tabulaires ;
* chiffres proportionnels ;
* chiffres elzéviriens ;
* chiffres alignés ;
* fractions ;
* ordinaux ;
* exposants ;
* indices ;
* variantes stylistiques ;
* alternates ;
* swash ;
* contextual alternates ;
* mark positioning ;
* mark-to-mark ;
* cursive attachment ;
* ruby ;
* vertical alternates ;
* glyph composition ;
* glyph decomposition.

## 7.6 Polices variables

Si police variable :

* axes disponibles ;
* axes utilisés ;
* poids réel ;
* largeur réelle ;
* slant ;
* optical size ;
* grade ;
* italique ;
* paramètres personnalisés ;
* instance nommée ;
* coordonnées de variation ;
* métriques variables ;
* bbox variable.

## 7.7 Rendu typographique

* anti-aliasing observé ;
* hinting ;
* subpixel rendering ;
* stroke adjustment ;
* text rendering mode ;
* fill text ;
* stroke text ;
* fill + stroke ;
* invisible text ;
* clipping text ;
* knockout ;
* overprint ;
* transparence ;
* blend mode.

---

# 8. Styles de paragraphe

Pour chaque paragraphe :

* style nommé si disponible ;
* style inféré ;
* rôle : titre, sous-titre, corps, légende, note, citation ;
* police dominante ;
* taille dominante ;
* couleur dominante ;
* alignement ;
* justification ;
* retrait gauche ;
* retrait droit ;
* retrait première ligne ;
* retrait suspendu ;
* espace avant ;
* espace après ;
* interligne ;
* interligne exact ou proportionnel ;
* tabulations ;
* bordures ;
* fond ;
* indentation de liste ;
* niveau hiérarchique ;
* keep with next ;
* keep lines together ;
* widow/orphan control ;
* césure autorisée ;
* langue ;
* direction ;
* numérotation ;
* puce ;
* lettrine ;
* drop cap ;
* colonnes internes ;
* style suivant ;
* style parent.

---

# 9. Titres et hiérarchie

À extraire :

* titre de niveau 1 ;
* titre de niveau 2 ;
* titre de niveau 3 ;
* titre de niveau 4 ;
* surtitre ;
* sous-titre ;
* intertitre ;
* titre courant ;
* titre de tableau ;
* titre de figure ;
* numérotation du titre ;
* hiérarchie logique ;
* style visuel ;
* position ;
* relation avec contenu suivant ;
* saut de page avant ;
* saut de page après ;
* ancrage ;
* entrée potentielle dans table des matières ;
* ordre dans le document.

---

# 10. Listes

Pour chaque liste :

* type : puce, numérotée, alphabétique, romaine, checklist ;
* niveau ;
* parent ;
* enfants ;
* symbole de puce ;
* police de la puce ;
* taille de la puce ;
* couleur de la puce ;
* position de la puce ;
* retrait de la puce ;
* retrait du texte ;
* tabulation après puce ;
* format de numérotation ;
* valeur de départ ;
* continuité avec liste précédente ;
* redémarrage ;
* espacement entre items ;
* alignement ;
* sous-listes ;
* cases cochées/non cochées ;
* relation item-contenu.

---

# 11. Tableaux

Les tableaux doivent être reconstruits comme des structures, pas comme de simples lignes.

## 11.1 Détection du tableau

* bbox globale ;
* nombre de lignes ;
* nombre de colonnes ;
* présence de cellules fusionnées ;
* présence d’en-tête ;
* présence de pied de tableau ;
* orientation ;
* titre ;
* légende ;
* note de tableau ;
* source ;
* ordre de lecture.

## 11.2 Cellules

Pour chaque cellule :

* ligne ;
* colonne ;
* rowspan ;
* colspan ;
* bbox ;
* polygone ;
* texte ;
* blocs internes ;
* alignement horizontal ;
* alignement vertical ;
* marge interne gauche ;
* marge interne droite ;
* marge interne haute ;
* marge interne basse ;
* couleur de fond ;
* image de fond ;
* bordure haute ;
* bordure basse ;
* bordure gauche ;
* bordure droite ;
* style de bordure ;
* épaisseur de bordure ;
* couleur de bordure ;
* diagonale ;
* rotation du texte ;
* direction du texte ;
* style typographique ;
* format de nombre ;
* format de date ;
* unité ;
* formule ;
* total ;
* cellule vide réelle ;
* cellule vide apparente.

## 11.3 Grille

* lignes visibles ;
* lignes invisibles ;
* colonnes visibles ;
* colonnes invisibles ;
* bordures partielles ;
* espacement entre cellules ;
* largeur de colonne ;
* hauteur de ligne ;
* alignement de la grille ;
* répétition d’en-tête sur pages suivantes ;
* tableau coupé sur plusieurs pages ;
* continuité entre pages.

---

# 12. Images raster

Pour chaque image :

* identifiant ;
* page ;
* bbox ;
* polygone ;
* largeur affichée ;
* hauteur affichée ;
* largeur native ;
* hauteur native ;
* DPI natif ;
* DPI affiché ;
* ratio ;
* rotation ;
* échelle ;
* déformation ;
* recadrage ;
* masque ;
* transparence ;
* profil colorimétrique ;
* espace couleur ;
* bits par canal ;
* compression ;
* format source ;
* image extraite brute ;
* image rendue ;
* image après masque ;
* position dans l’ordre de rendu ;
* relation avec légende ;
* alt text ;
* type probable : photo, logo, scan, signature, cachet, graphique, schéma, icône ;
* OCR dans image ;
* objets détectés ;
* zones textuelles internes ;
* score qualité ;
* flou ;
* bruit ;
* contraste ;
* inclinaison ;
* distorsion perspective.

---

# 13. Graphiques vectoriels

À extraire objet par objet.

## 13.1 Formes simples

* ligne ;
* rectangle ;
* rectangle arrondi ;
* cercle ;
* ellipse ;
* polygone ;
* courbe ;
* chemin ;
* flèche ;
* accolade ;
* trait de séparation ;
* cadre ;
* encadré ;
* bulle ;
* symbole ;
* icône vectorielle.

## 13.2 Propriétés géométriques

* bbox ;
* path complet ;
* points ;
* courbes de Bézier ;
* angle ;
* transformation ;
* clipping ;
* fermeture du chemin ;
* sens du chemin ;
* remplissage ;
* contour.

## 13.3 Style graphique

* couleur de remplissage ;
* couleur de contour ;
* épaisseur de trait ;
* type de trait ;
* pointillé ;
* tirets ;
* extrémités de ligne ;
* jonctions ;
* rayon d’arrondi ;
* opacité ;
* mode de fusion ;
* ombre ;
* dégradé ;
* motif ;
* transparence ;
* surimpression.

## 13.4 Relations

* forme décorative ;
* forme porteuse d’information ;
* encadre un texte ;
* souligne un titre ;
* sépare des sections ;
* appartient à un graphique ;
* appartient à un logo ;
* appartient à un tableau ;
* appartient à un formulaire.

---

# 14. Couleurs et gestion colorimétrique

Pour chaque objet :

* couleur de remplissage ;
* couleur de contour ;
* couleur de texte ;
* couleur de fond ;
* couleur de surlignage ;
* opacité ;
* alpha ;
* espace couleur ;
* RGB ;
* CMYK ;
* Gray ;
* Lab ;
* ICC profile ;
* spot color ;
* couleur nommée ;
* dégradé ;
* motif ;
* surimpression ;
* mode de fusion ;
* transparence ;
* contraste avec fond ;
* couleur normalisée hex ;
* couleur source ;
* équivalence écran ;
* équivalence impression.

---

# 15. Effets visuels

À capturer :

* ombre portée ;
* ombre interne ;
* flou ;
* glow ;
* contour lumineux ;
* emboss ;
* relief ;
* transparence ;
* opacité partielle ;
* dégradé linéaire ;
* dégradé radial ;
* motif ;
* texture ;
* masque ;
* clipping ;
* image masquée par texte ;
* texte en contour ;
* texte en remplissage ;
* texte avec fond ;
* rotation ;
* perspective ;
* déformation ;
* reflet.

---

# 16. En-têtes, pieds de page et éléments récurrents

À extraire :

* en-tête gauche ;
* en-tête centre ;
* en-tête droit ;
* pied gauche ;
* pied centre ;
* pied droit ;
* numéro de page ;
* total de pages ;
* titre courant ;
* nom de chapitre ;
* logo récurrent ;
* ligne de séparation ;
* date ;
* auteur ;
* mention confidentielle ;
* référence documentaire ;
* code document ;
* filigrane ;
* tampon récurrent ;
* différence première page ;
* différence pages paires/impaires ;
* répétition exacte ;
* répétition avec variation ;
* zone à ne pas traduire ;
* zone à traduire.

---

# 17. Notes, références et appels

À extraire :

* appels de note ;
* exposants ;
* notes de bas de page ;
* notes de fin ;
* références croisées ;
* renvois ;
* citations ;
* bibliographie ;
* numéros de référence ;
* hyperliens ;
* ancres ;
* DOI ;
* ISBN ;
* URL ;
* relation appel-note ;
* relation citation-bibliographie ;
* style de citation ;
* ordre des références.

---

# 18. Formules mathématiques et scientifiques

À extraire séparément du texte normal.

## 18.1 Détection

* formule inline ;
* formule bloc ;
* équation numérotée ;
* système d’équations ;
* fraction ;
* racine ;
* exposant ;
* indice ;
* intégrale ;
* somme ;
* matrice ;
* vecteur ;
* symbole grec ;
* unité scientifique ;
* variable ;
* opérateur ;
* constante.

## 18.2 Représentation

* image originale ;
* OCR mathématique ;
* LaTeX inféré ;
* MathML si possible ;
* bbox ;
* baseline ;
* police mathématique ;
* taille ;
* alignement ;
* numéro d’équation ;
* relation avec texte ;
* éléments à ne pas traduire.

---

# 19. Graphiques de données

Pour les graphiques, il faut distinguer l’image du graphique et sa structure.

À extraire :

* type : barre, ligne, camembert, nuage de points, histogramme, radar, carte, diagramme ;
* titre ;
* sous-titre ;
* axes ;
* graduations ;
* labels ;
* légende ;
* séries ;
* couleurs des séries ;
* valeurs si extractibles ;
* unités ;
* échelle linéaire/log ;
* annotations ;
* source ;
* note ;
* bbox ;
* image originale ;
* texte interne OCR ;
* zones non traduisibles ;
* zones traduisibles.

---

# 20. Formulaires

À extraire :

* champs texte ;
* cases à cocher ;
* boutons radio ;
* listes déroulantes ;
* champs date ;
* champs signature ;
* lignes de saisie ;
* libellés ;
* valeurs ;
* options ;
* état coché/non coché ;
* obligatoire/facultatif ;
* nom interne du champ ;
* tooltip ;
* ordre de tabulation ;
* bbox ;
* style ;
* relation label-champ ;
* texte à traduire ;
* valeur à conserver.

---

# 21. Annotations, commentaires et marques

À extraire :

* commentaire ;
* note sticky ;
* surlignage ;
* soulignement ;
* barré ;
* dessin libre ;
* rectangle ;
* flèche ;
* tampon ;
* signature ;
* pièce jointe ;
* lien ;
* annotation invisible ;
* auteur ;
* date ;
* contenu ;
* bbox ;
* couleur ;
* opacité ;
* état ;
* réponse ;
* relation avec texte annoté.

---

# 22. Logos, cachets, signatures et éléments sensibles

À extraire comme objets protégés.

## 22.1 Logos

* bbox ;
* image/vectoriel ;
* texte interne ;
* couleurs ;
* position ;
* taille ;
* répétition ;
* ne_pas_traduire ;
* nom probable de l’organisation ;
* relation avec en-tête.

## 22.2 Cachets

* bbox ;
* couleur ;
* forme ;
* texte OCR ;
* rotation ;
* opacité ;
* image brute ;
* ne_pas_modifier sauf demande ;
* score lisibilité.

## 22.3 Signatures

* bbox ;
* image ;
* manuscrite ou numérique ;
* couleur ;
* transparence ;
* position ;
* relation avec nom/fonction ;
* ne_pas_traduire ;
* ne_pas_altérer.

---

# 23. Documents scannés et OCR

Si page scannée :

## 23.1 Qualité image

* DPI ;
* résolution ;
* couleur/gris/noir-blanc ;
* bruit ;
* flou ;
* contraste ;
* luminosité ;
* compression ;
* skew ;
* rotation ;
* perspective ;
* courbure de page ;
* ombre ;
* pli ;
* tache ;
* fond papier ;
* transparence verso ;
* lignes parasites ;
* tampon superposé ;
* écriture manuscrite.

## 23.2 Prétraitement à documenter

Ne pas seulement appliquer le prétraitement ; il faut enregistrer ce qui a été fait :

* deskew ;
* denoise ;
* binarisation ;
* contraste ;
* sharpening ;
* suppression fond ;
* correction perspective ;
* découpage ;
* upscale ;
* segmentation ;
* OCR utilisé ;
* modèle utilisé ;
* version modèle ;
* langue OCR ;
* seuils ;
* score moyen ;
* zones douteuses.

## 23.3 Résultats OCR

* texte ;
* bbox ligne ;
* bbox mot ;
* bbox caractère si possible ;
* score ;
* alternatives OCR ;
* détection manuscrite ;
* correction proposée ;
* incertitude ;
* conflit entre OCR et texte natif ;
* source retenue ;
* source rejetée.

---

# 24. Ordre de lecture

À extraire explicitement.

Pour chaque bloc :

* reading_order_id ;
* page_order ;
* column_order ;
* region_order ;
* parent_order ;
* next_block_id ;
* previous_block_id ;
* relation logique ;
* relation visuelle ;
* flux principal ;
* flux secondaire ;
* note ;
* légende ;
* encadré ;
* tableau ;
* hors flux ;
* décoratif ;
* à ignorer pour traduction ;
* à conserver.

Il faut distinguer :

* ordre de stockage dans le fichier ;
* ordre de rendu ;
* ordre visuel ;
* ordre humain de lecture ;
* ordre de traduction.

Ces quatre ordres peuvent être différents.

---

# 25. Relations entre objets

Chaque objet doit pouvoir être relié à d’autres.

Relations à extraire :

* titre → paragraphe ;
* image → légende ;
* tableau → titre ;
* tableau → note ;
* appel de note → note ;
* figure → référence dans le texte ;
* champ → libellé ;
* logo → en-tête ;
* numéro de page → pied de page ;
* liste → item ;
* item → sous-item ;
* paragraphe → paragraphe suivant ;
* bloc coupé → continuation page suivante ;
* tableau coupé → continuation page suivante ;
* colonne → bloc ;
* annotation → texte annoté ;
* graphique → légende ;
* équation → numéro ;
* référence → bibliographie.

---

# 26. Contraintes de reconstruction après traduction

C’est souvent oublié. Pourtant, c’est indispensable.

Pour chaque bloc traduisible :

* bbox originale ;
* largeur disponible ;
* hauteur disponible ;
* marge interne ;
* marge externe ;
* possibilité d’expansion horizontale ;
* possibilité d’expansion verticale ;
* possibilité de réduction de police ;
* taille minimale acceptable ;
* interligne minimal ;
* condensation autorisée ;
* hyphenation autorisée ;
* retour à la ligne autorisé ;
* déplacement autorisé ;
* chevauchement interdit ;
* peut repousser le bloc suivant ;
* peut créer une page supplémentaire ;
* peut s’étendre sur plusieurs pages ;
* doit rester même page ;
* doit rester avec titre ;
* doit rester avec image ;
* texte doit être centré ;
* texte doit rester sur une ligne ;
* texte peut être abrégé ;
* texte non traduisible ;
* priorité de fidélité : haute, moyenne, basse.

## Exemple de contraintes utiles

Un titre dans un encadré :

```json
{
  "object_id": "block_145",
  "type": "heading",
  "translatable": true,
  "bbox": [82.1, 140.5, 430.2, 174.8],
  "layout_constraints": {
    "max_width": 348.1,
    "max_height": 34.3,
    "allow_font_shrink": true,
    "min_font_size": 10,
    "allow_line_wrap": true,
    "max_lines": 2,
    "allow_move": false,
    "preserve_center_alignment": true,
    "avoid_overlap": ["logo_01", "table_03"]
  }
}
```

---

# 27. Éléments à classer selon leur traduisibilité

Chaque élément textuel doit recevoir un statut.

## 27.1 Traduisible

* paragraphes ;
* titres ;
* légendes ;
* notes ;
* contenu de tableau ;
* libellés ;
* commentaires ;
* bulles ;
* encadrés ;
* mentions explicatives.

## 27.2 Non traduisible

* noms propres ;
* logos ;
* signatures ;
* cachets ;
* codes ;
* identifiants ;
* références ;
* variables ;
* formules ;
* unités ;
* URLs ;
* emails ;
* noms de fichiers ;
* numéros ;
* dates selon contexte ;
* médicaments selon contexte ;
* noms institutionnels selon contexte.

## 27.3 À traduire avec prudence

* sigles ;
* acronymes ;
* titres d’organismes ;
* articles de loi ;
* noms de programmes ;
* noms de projets ;
* termes techniques ;
* diagnostics ;
* médicaments ;
* concepts juridiques ;
* citations ;
* slogans ;
* poèmes ;
* jeux de mots.

Pour chacun :

* `translation_policy`;
* `preserve_case`;
* `preserve_punctuation`;
* `preserve_line_breaks`;
* `preserve_terms`;
* `glossary_required`;
* `domain`.

---

# 28. Domaines et registre

À détecter par bloc ou document :

* administratif ;
* juridique ;
* médical ;
* scientifique ;
* technique ;
* financier ;
* académique ;
* littéraire ;
* commercial ;
* religieux ;
* informatique ;
* assurance ;
* santé publique ;
* communication ;
* marketing ;
* formulaire ;
* rapport ;
* correspondance.

À extraire aussi :

* niveau de langue ;
* ton ;
* style ;
* formalité ;
* tutoiement/vouvoiement ;
* voix passive/active ;
* terminologie spécialisée ;
* acronymes ;
* conventions institutionnelles.

---

# 29. Styles globaux et styles récurrents

Il faut reconstruire une feuille de styles interne.

## 29.1 Style document

* style corps ;
* style titre 1 ;
* style titre 2 ;
* style titre 3 ;
* style légende ;
* style note ;
* style tableau ;
* style en-tête ;
* style pied ;
* style liste ;
* style citation ;
* style encadré ;
* style formule ;
* style signature.

## 29.2 Détection des styles

Pour chaque style :

* nom inféré ;
* police ;
* taille ;
* poids ;
* couleur ;
* alignement ;
* interligne ;
* espacement avant/après ;
* retrait ;
* bordure ;
* fond ;
* niveau hiérarchique ;
* fréquence ;
* exemples d’utilisation ;
* pages concernées ;
* héritage ;
* exceptions locales.

---

# 30. Cas particuliers à ne pas oublier

## 30.1 Texte vertical

* orientation ;
* glyphes verticaux ;
* rotation caractère par caractère ;
* sens de lecture ;
* ponctuation verticale ;
* police compatible ;
* bbox adaptée.

## 30.2 Texte courbé

* chemin ;
* texte sur path ;
* rayon ;
* angle ;
* position des glyphes ;
* relation avec logo ou tampon.

## 30.3 Texte vectorisé

* chemins vectoriels ;
* tentative de reconnaissance glyphique ;
* image fallback ;
* texte OCR ;
* statut non éditable ;
* score de reconnaissance.

## 30.4 Texte invisible

* texte OCR sous image ;
* texte blanc ;
* texte sans rendu ;
* texte de clipping ;
* texte masqué ;
* texte utile ou parasite.

## 30.5 Superpositions

* texte sur image ;
* texte sur fond coloré ;
* tampon sur texte ;
* signature sur texte ;
* filigrane sous texte ;
* annotation sur texte ;
* objet cachant partiellement un autre.

---

# 31. Représentation recommandée des objets

Chaque objet extrait devrait avoir une structure minimale comme celle-ci :

```json
{
  "id": "obj_000123",
  "type": "text_span",
  "page_id": "page_001",
  "parent_id": "line_0045",
  "role": "body_text",
  "source": "native_pdf",
  "bbox": {
    "x0": 72.14,
    "y0": 184.22,
    "x1": 318.70,
    "y1": 198.40,
    "unit": "pt"
  },
  "quad": [
    [72.14, 184.22],
    [318.70, 184.22],
    [318.70, 198.40],
    [72.14, 198.40]
  ],
  "transform": {
    "matrix": [1, 0, 0, 1, 0, 0],
    "rotation": 0,
    "scale_x": 1,
    "scale_y": 1,
    "shear": 0
  },
  "text": {
    "raw": "Contrôle médical sur site",
    "normalized": "Contrôle médical sur site",
    "language": "fr",
    "script": "Latin",
    "direction": "ltr",
    "translatable": true
  },
  "style": {
    "font_family": "Calibri",
    "font_postscript": "Calibri-Bold",
    "font_size": 12,
    "font_weight": 700,
    "italic": false,
    "underline": false,
    "color": "#000000",
    "opacity": 1,
    "line_height": 14.2,
    "letter_spacing": 0,
    "word_spacing": 0
  },
  "layout": {
    "reading_order": 18,
    "render_order": 256,
    "column": 1,
    "paragraph_id": "para_0012",
    "baseline": 195.1,
    "alignment": "left"
  },
  "constraints": {
    "max_width": 246.56,
    "max_height": 14.18,
    "allow_wrap": true,
    "allow_font_shrink": true,
    "min_font_size": 9,
    "preserve_position": true
  },
  "confidence": {
    "extraction": 0.99,
    "ocr": null,
    "style": 0.95,
    "reading_order": 0.92
  }
}
```

---

# 32. Schéma global minimal de sortie

Ta représentation fidèle devrait ressembler à ceci :

```json
{
  "document": {
    "metadata": {},
    "source_profile": {},
    "pages": [],
    "fonts": [],
    "images": [],
    "styles": [],
    "resources": [],
    "reading_order": [],
    "translation_units": [],
    "layout_constraints": [],
    "quality_report": {}
  }
}
```

Avec, au minimum :

```json
{
  "pages": [
    {
      "id": "page_001",
      "number": 1,
      "width": 595.28,
      "height": 841.89,
      "unit": "pt",
      "rotation": 0,
      "boxes": {
        "media": [],
        "crop": [],
        "trim": [],
        "bleed": []
      },
      "background": {},
      "regions": [],
      "objects": [],
      "reading_order": [],
      "render_order": []
    }
  ]
}
```

---

# 33. Rapport qualité obligatoire

Pour chaque page, produire :

* score extraction texte ;
* score extraction police ;
* score extraction images ;
* score segmentation ;
* score ordre de lecture ;
* score OCR ;
* score reconstruction estimée ;
* zones douteuses ;
* objets non reconnus ;
* polices manquantes ;
* glyphes non mappés ;
* texte sans Unicode ;
* objets vectoriels complexes ;
* tableaux incertains ;
* images basse résolution ;
* chevauchements ;
* risque de mauvaise traduction ;
* risque de mauvaise recomposition.

---

# 34. Priorité d’implémentation pour ta première phase

Tu ne dois pas tout coder au même niveau dès le premier jour. Mais dans la **phase 1**, le schéma doit déjà prévoir tout cela.

## Niveau 1 — indispensable

À implémenter immédiatement :

* pages ;
* dimensions ;
* coordonnées ;
* texte natif ;
* OCR si image ;
* blocs ;
* lignes ;
* mots ;
* bbox ;
* ordre de lecture ;
* ordre de rendu ;
* police ;
* taille ;
* couleur ;
* gras/italique/souligné ;
* paragraphes ;
* images ;
* tableaux simples ;
* formes simples ;
* en-têtes/pieds ;
* numéros de page ;
* styles récurrents ;
* langue ;
* traduisible/non traduisible ;
* contraintes de reconstruction.

## Niveau 2 — très important

À implémenter rapidement :

* glyphes ;
* ToUnicode ;
* ligatures ;
* kerning ;
* baseline ;
* cap height ;
* x-height ;
* tableaux complexes ;
* colonnes ;
* notes ;
* légendes ;
* cachets ;
* signatures ;
* texte dans images ;
* texte vectorisé ;
* formulaires ;
* annotations ;
* équations ;
* calques ;
* transparence ;
* clipping ;
* modes de fusion.

## Niveau 3 — avancé

À implémenter ensuite :

* OpenType GSUB/GPOS complet ;
* polices variables ;
* reconnaissance fine de police ;
* reconstruction de graphiques de données ;
* extraction sémantique avancée ;
* modèles de mise en page adaptatifs ;
* solveur de recomposition ;
* hyphenation multilingue ;
* justification typographique avancée ;
* correction OCR contextuelle ;
* vérification visuelle automatique page originale/page reconstruite.

---

# 35. Les erreurs classiques à éviter

Les erreurs qui détruisent la fidélité :

* extraire seulement le texte ;
* fusionner toutes les sources OCR/PDF sans déduplication ;
* perdre les coordonnées caractère/mot ;
* ignorer la police réelle ;
* ignorer les glyphes ;
* ignorer les ligatures ;
* ignorer l’ordre de rendu ;
* ignorer l’ordre de lecture ;
* traiter les tableaux comme du texte linéaire ;
* traduire les logos ;
* traduire les noms propres ;
* traduire les références ;
* ignorer les zones non traduisibles ;
* ignorer les images contenant du texte ;
* ignorer les signatures/cachets ;
* ignorer les contraintes de bloc ;
* forcer le texte traduit dans la même boîte sans solveur ;
* ne pas conserver le document source brut ;
* ne pas garder les versions intermédiaires ;
* ne pas produire de score de confiance ;
* ne pas signaler les zones incertaines.

---

# 36. Formule simple à retenir

Pour une extraction fidèle, chaque élément doit répondre à ces questions :

1. **Qu’est-ce que c’est ?**
   Texte, image, tableau, forme, formule, signature, logo, note, etc.

2. **Où est-ce ?**
   Page, coordonnées, bbox, polygone, colonne, zone.

3. **Comment est-ce rendu ?**
   Police, taille, couleur, opacité, rotation, style, ordre de rendu.

4. **Comment est-ce lu ?**
   Ordre de lecture, langue, direction, phrase, paragraphe, rôle.

5. **Est-ce traduisible ?**
   Oui, non, partiellement, avec glossaire, avec contrainte.

6. **Comment peut-on le reconstruire après traduction ?**
   Contraintes, largeur disponible, hauteur disponible, possibilité de réduire, déplacer, couper, ajouter une page.

---

# 37. Conclusion opérationnelle

La première phase d’extraction doit sortir une représentation en couches :

```text
Document
 ├── Métadonnées
 ├── Pages
 │    ├── Géométrie
 │    ├── Fond
 │    ├── Régions
 │    ├── Objets visuels
 │    ├── Texte natif
 │    ├── OCR
 │    ├── Images
 │    ├── Formes
 │    ├── Tableaux
 │    ├── Annotations
 │    ├── Ordre de rendu
 │    └── Ordre de lecture
 ├── Polices
 ├── Styles
 ├── Ressources
 ├── Relations
 ├── Unités de traduction
 ├── Contraintes de recomposition
 └── Rapport qualité
```

Le point central est celui-ci : **ta sortie d’extraction ne doit pas être un texte enrichi, mais un modèle complet de document.**
C’est seulement avec ce modèle que tu pourras traduire, recomposer, ajuster et reconstruire un document visuellement fidèle.

[1]: https://developer.adobe.com/document-services/docs/assets/35e4369068f86065372c18787171a17e/PDF_ISO_32000-1.pdf?utm_source=chatgpt.com "Portable document format — Part 1: PDF 1.7"
[2]: https://www.w3.org/TR/css-page-3/?utm_source=chatgpt.com "CSS Paged Media Module Level 3"
[3]: https://www.unicode.org/reports/tr14/?utm_source=chatgpt.com "UAX #14: Unicode Line Breaking Algorithm"
[4]: https://learn.microsoft.com/en-us/typography/opentype/spec/chapter2?utm_source=chatgpt.com "OpenType layout common table formats (OpenType 1.9.1)"

