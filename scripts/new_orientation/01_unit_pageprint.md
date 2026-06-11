ocr_server.py
Python
Analyses décris en détail ce que fait ocr_server.py

Je vais traiter le fichier comme un serveur applicatif complet, pas seulement comme un script OCR. Je vais identifier ses points d’entrée, le pipeline d’extraction, les traitements post-OCR, les exports et les limites techniques visibles dans le code.

Première conclusion nette : ce fichier n’est pas un « serveur OCR » simple. C’est un orchestrateur complet de numérisation documentaire : OCR, extraction native PDF, segmentation sémantique, nettoyage du fond, traduction, reconstruction PDF/HTML et contrôle qualité.

Thought for 1m 13s

Voici l’analyse directe : ocr_server.py est le cœur serveur d’un pipeline WYSIWYG pour documents. Il ne fait pas seulement de l’OCR. Il prend un PDF, une image ou un fichier Office, extrait le texte, la mise en page, les styles, les objets visuels, segmente le contenu en unités sémantiques, prépare la traduction, nettoie l’image de fond, puis permet de reconstruire un PDF/HTML traduit ou non traduit.

1. Rôle général du fichier

Le fichier lance une API FastAPI nommée "IA Document OCR - Stable Precision". Il expose plusieurs routes :

Route	Rôle
POST /ocr	Charger un document et extraire sa structure complète
POST /translate-units	Traduire des phrases/groupes/mots extraits
POST /translate	Traduire une structure documentaire complète
POST /reconstruct	Reconstruire un PDF à partir de la structure extraite/traduite
POST /debug/visual-compare	Comparer visuellement l’original et le PDF reconstruit
POST /export_html	Exporter la structure en HTML
GET /healthz	Vérifier l’état du service

Il sert les fichiers générés via /results, qui pointe vers le dossier ocr_results. Le serveur écoute sur 0.0.0.0:8001 quand le script est lancé directement.

2. Les composants principaux chargés au démarrage

Au démarrage, il instancie plusieurs modules :

RapidOCR() : moteur OCR principal.
DocumentParser() : transforme les mots OCR bruts en blocs/lignes/phrases/spans.
NativePDFExtractor() : extrait le texte natif, images, dessins et zones non textuelles des PDF.
LayoutV2Builder() : reconstruit une structure canonique plus stable.
DocumentReconstructor() : reconstruit un PDF final.
TextRemovalStrategy() : retire le texte de l’image source pour produire un fond propre.
HtmlStyleExporter() : export HTML.
PagePolicyMatrix() : décide ce qui est traduisible, protégé, à préserver, etc.
layout_ai_enricher : enrichissement IA optionnel de la mise en page.

Il définit aussi des variables importantes :

UPLOAD_DIR = "uploads"
CONV_DIR = "converted_pages"
RESULTS_DIR = "ocr_results"
TARGET_DPI = 150
EXTRACTION_AI_ENABLED = os.getenv("DOCS_PARSER_ENABLE_EXTRACTION_AI", "0") == "1"
LAYOUT_OPTIMIZER_ON_TRANSLATION = os.getenv("LAYOUT_OPTIMIZER_ON_TRANSLATION", "0") == "1"
OFFICE_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx", ".odt", ".odp"}

Donc, par défaut, le rendu des pages se fait à 150 DPI, l’IA d’extraction est désactivée sauf variable d’environnement, et les documents Office sont convertis en PDF via LibreOffice/soffice.

3. Ce que fait /ocr

La route /ocr est l’entrée principale. Elle reçoit un fichier, le sauvegarde dans uploads, détecte son extension, puis traite selon trois cas :

Cas 1 — fichier Office

Pour .doc, .docx, .ppt, .pptx, .odt, .odp, le serveur appelle LibreOffice en mode headless pour convertir le fichier en PDF. Ensuite, chaque page du PDF est rendue en image avec PyMuPDF à TARGET_DPI.

Cas 2 — PDF

Chaque page PDF est rendue en image. L’objet page PDF original est transmis à process_page, ce qui permet d’utiliser à la fois :

l’image rendue pour l’OCR ;
le PDF natif pour extraire le texte déjà vectoriel, les images, les dessins, les coordonnées, etc.
Cas 3 — image

L’image est ouverte avec PIL, convertie en RGB, puis passée directement à process_page.

Après traitement, /ocr repasse sur chaque page pour enrichir encore les blocs : marqueurs de ligne, phrases sémantiques, contrats de traduction, spans sémantiques, runs, groupes, extraction hiérarchique et export de fidélité.

4. Le cœur réel : process_page

process_page est la fonction centrale du pipeline. Elle traite une page.

Étape A — sauvegarde de l’image source

La page rendue est sauvegardée comme image src_...png dans ocr_results. Cette image sert ensuite de référence pour la reconstruction et la comparaison visuelle.

Étape B — extraction native PDF

Si la page vient d’un PDF et que force_ai=False, le script appelle :

native_pdf_extractor.extract_page(pdf_page, sx=sx, sy=sy)

Cela récupère :

native_blocks : blocs de texte natif PDF ;
non_text_zones : zones à ne pas nettoyer ;
native_images : images extraites ;
native_drawings : dessins/vectoriels extraits.

C’est important : si un PDF contient déjà du texte exploitable, le serveur évite de dépendre uniquement de l’OCR.

Étape C — OCR du reste de la page

Même quand du texte natif existe, RapidOCR est lancé sur l’image entière :

result, _ = engine_ocr(np.array(img))

Mais le code filtre les résultats OCR qui recouvrent déjà fortement du texte natif. Concrètement, si une boîte OCR recouvre à plus de 50 % un bloc natif, elle est ignorée. Cela évite les doublons entre texte PDF natif et texte OCR.

Étape D — structuration OCR

Les mots OCR bruts deviennent une structure documentaire via :

ocr_structure = parser.parse(raw_ocr, img)

Puis _prune_weak_ocr_lines supprime certaines lignes OCR faibles, notamment des petits fragments suspects ou des queues isolées peu fiables.

Étape E — fusion texte natif + OCR

Le code fusionne les blocs natifs PDF et les blocs OCR avec _dedupe_final_blocks. Cette fonction compare les boîtes, le chevauchement et la similarité textuelle pour éviter de conserver deux fois le même contenu.

Étape F — post-traitement géométrique

Ensuite, _postprocess_blocks applique plusieurs corrections :

fusion de blocs natifs proches si style compatible ;
attachement de petits fragments à des paragraphes ;
regroupement de labels de diagrammes ;
expansion des groupes de labels de figures ;
fusion de labels de diagrammes qui se chevauchent ;
marquage de certains groupes comme background_only.

Ce point est essentiel pour un système WYSIWYG : il ne suffit pas d’avoir le texte, il faut savoir si ce texte est un paragraphe, un titre, une légende, une équation, un label de graphique, etc.

5. Segmentation sémantique

Après la géométrie, le fichier construit plusieurs niveaux de texte :

Niveau	Signification
block	bloc visuel global
line	ligne détectée
phrase	unité visuelle de texte
span	portion typographique avec style
semantic_phrase	phrase reconstituée logiquement
semantic_span	unité textuelle inter-lignes avec style
semantic_run	suite cohérente d’unités
semantic_group	groupe éditorial plus large

Le but est clair : ne pas traduire naïvement ligne par ligne. Le code essaie de reconstituer des phrases complètes même si elles sont coupées sur plusieurs lignes.

Il gère notamment :

les fins de phrase ., !, ?, … ;
les abréviations comme Fig., Dr., Eq. ;
les césures de fin de ligne ;
les titres en majuscules ;
les lignes séparatrices ;
les fractions/formules ;
les blocs atomiques comme labels de diagramme ou légendes ;
les listes et puces.

Il existe aussi un post-traitement sémantique optionnel par LLM ou par P1ExtractionAgent, activé selon les variables d’environnement. Le rôle de cet agent est de corriger les blocs ambigus : frontières de phrases, jonctions par césure, modes de layout, etc.

6. Contrats de traduction

La fonction _annotate_translation_contracts est une pièce majeure. Elle décide, pour chaque bloc, ligne, phrase et span :

si le texte est traduisible ;
quel type d’unité c’est : prose, citation, lien, code, formule, label, etc. ;
quelle stratégie appliquer : traduction normale, conservation exacte, fond seulement, texte ancré, layout contraint ;
si la couverture doit être stricte ;
quel est le contexte structurel parent/enfant ;
quelles sont les relations avec les unités voisines.

Elle utilise PagePolicyMatrix.classify_unit_policy(...), avec le rôle de page, la famille de page, le type documentaire, le type de layout et le profil de style.

En clair : ce fichier prépare le terrain pour éviter des erreurs classiques de traduction documentaire, par exemple traduire une formule, casser une URL, modifier un code, déformer une référence bibliographique ou déplacer un label de graphique.

7. Protection des éléments immuables

Le code détecte certains textes à ne pas re-rendre comme du texte traduisible :

formules ;
symboles grecs ;
références comme [12], (2) ;
acronymes courts ;
formules chimiques ;
morceaux de code monospace ;
labels techniques.

Ces éléments peuvent être découpés en petites images immutable_...png, puis marqués skip_render ou background_only. Le principe : ils restent dans le fond visuel, au lieu d’être retraduits ou redessinés par le moteur de reconstruction.

C’est une bonne idée pour la fidélité visuelle, mais cela peut poser problème si on veut rendre tout le texte éditable.

8. Nettoyage du fond maître

Le pipeline génère un background master : une image de la page sans les textes qui devront être reconstruits.

Le processus est :

collecter les régions textuelles à enlever ;
protéger les zones non textuelles, images, diagrammes, overlays immuables ;
appeler text_removal_strategy.remove(...) ;
sauvegarder :
bg_master_...png ;
mask_master_...png.

Puis, pour les PDF, _erase_uncovered_pdf_words tente d’effacer les mots PDF natifs qui n’ont pas été couverts par les blocs finaux, par exemple certains en-têtes, pieds de page ou titres décoratifs.

Il existe aussi un audit optionnel du fond via P6BackgroundAgent, activé par PIPELINE_AGENT_P6_ENABLE=1. Cet audit produit des informations de qualité, d’artéfacts et de nécessité de retraitement.

9. Export de debug et visualisation

Le fichier produit aussi :

une image vis_...jpg avec rectangles :
bleu = bloc ;
vert = ligne ;
rouge = phrase ;
un texte détaillé display_text listant blocs, lignes, phrases, spans, polices, tailles, couleurs, flags ;
un XML de layout via _write_layout_xml ;
une extraction hiérarchique ;
un export de fidélité fidelity_layout.

L’extraction hiérarchique retourne :

{
  "phrases": [...],
  "groupes_mots": [...],
  "mots": [...],
  "residuels": [...],
  "chiffres": [...],
  "nombres": [...],
  "lettres": [...],
  "symboles": [...],
  "counts": {...}
}

Donc il ne se contente pas de sortir du texte plat : il produit une représentation riche utilisable pour traduction, analyse et reconstruction.

10. Construction layout.v2

Après l’extraction brute, le fichier appelle :

layout_v2_builder.build(page_structure)

Puis, si EXTRACTION_AI_ENABLED=True, il appelle l’enrichisseur IA de layout. Ensuite :

il applique apply_page_extraction_postprocessors;
il reconstruit éventuellement le layout.v2;
il injecte des lignes de table des matières natives avec _inject_native_toc_rows;
il recalcule les phrases, spans, runs et groupes sémantiques.

C’est donc une architecture en deux couches :

extraction brute visuelle/OCR ;
normalisation canonique en structure documentaire stable.
11. Traduction

Le fichier gère deux formes de traduction.

/translate-units

Cette route reçoit des pages avec :

phrases;
groupes_mots;
mots.

Elle traduit chaque unité via _translate_unit_text.

La fonction _translate_unit_text fait plusieurs choses :

normalise les espaces ;
normalise les majuscules espacées ;
préserve les puces ;
évite de traduire les segments protégés ;
détecte domaine et sous-domaine ;
traduit via _translate_text_hierarchical;
restaure les tokens protégés ;
applique une normalisation ;
applique un glossaire métier ;
retente une traduction si le résultat est identique à la source ;
mesure une fuite de langue source dans la traduction et tente une traduction directe si nécessaire.
/translate

Cette route traduit une structure complète. Elle accepte :

une page unique ;
un objet {pages: [...]};
un objet enveloppé dans {structure: ...}.

Elle utilise DocumentTranslator. Si la page est une table des matières layout.v2, elle utilise une méthode spéciale translate_layout_v2; sinon elle utilise translate_page.

12. Reconstruction PDF

La route /reconstruct prend une structure déjà extraite, éventuellement la traduit, puis appelle :

DocumentReconstructor().reconstruct({"pages": pages}, output_path)

Le PDF final est écrit dans :

ocr_results/reconstructed_output.pdf

Si target_lang est fourni, le serveur lance aussi :

analyze_document_coverage;
analyze_rendered_text_coverage;
publication_qa.

Si debug_compare=True, il compare visuellement les pages originales et le PDF reconstruit avec compare_reconstruction. Si export_html=True, il produit aussi reconstructed_output.html.

13. Export HTML et comparaison visuelle

/export_html prend des pages structurées et produit un fichier HTML dans ocr_results.

/debug/visual-compare prend une liste de pages et un PDF reconstruit, retrouve les images sources, puis calcule des métriques de comparaison visuelle. C’est utile pour mesurer la fidélité WYSIWYG.

14. Ce que le fichier cherche à accomplir conceptuellement

Son objectif implicite est le suivant :

Transformer un document en une représentation riche, traduisible et reconstructible, tout en conservant au maximum la fidélité visuelle originale.

Il essaie donc de résoudre plusieurs problèmes difficiles :

OCR fiable : récupérer le texte même quand il n’est pas natif.
Exploitation PDF native : ne pas OCRiser inutilement ce qui existe déjà.
Déduplication : éviter texte natif + OCR doublonné.
Segmentation intelligente : retrouver phrases, paragraphes, titres, captions.
Préservation des éléments sensibles : formules, code, références, diagrammes.
Reconstruction WYSIWYG : produire un fond nettoyé + texte repositionné.
Traduction contrôlée : ne traduire que ce qui doit l’être.
Contrôle qualité : couverture textuelle, comparaison visuelle, QA publication.
15. Points forts

Les vrais points forts du fichier :

Pipeline hybride PDF natif + OCR : c’est supérieur à un OCR pur.
Préoccupation WYSIWYG réelle : il garde les coordonnées, styles, alignements, marges, rôles, backgrounds.
Granularité riche : blocs, lignes, phrases, spans, semantic phrases, semantic runs, semantic groups.
Protection des contenus non traduisibles : formules, code, références, symboles.
Préparation de la reconstruction : background nettoyé + overlays + structure layout.
Contrôle qualité intégré : coverage, visual compare, publication QA.
Prise en charge Office via conversion LibreOffice.
16. Faiblesses et risques

Il y a aussi des problèmes sérieux.

1. Fichier beaucoup trop monolithique

Le fichier fait environ 5 600 lignes et mélange :

serveur FastAPI ;
OCR ;
parsing ;
segmentation ;
layout ;
traduction ;
inpainting ;
QA ;
reconstruction ;
export XML/HTML ;
logique LLM.

C’est difficile à maintenir. Il faudrait découper en modules : routes.py, ocr_pipeline.py, semantic_segmentation.py, translation_pipeline.py, reconstruction_service.py, background_cleaner.py, etc.

2. Routes async, traitements bloquants

Les endpoints FastAPI sont async, mais le code exécute des opérations lourdes synchrones :

OCR ;
rendu PDF ;
LibreOffice ;
OpenCV ;
reconstruction PDF ;
traduction.

Donc le serveur peut bloquer sous charge. Il faudrait passer les traitements lourds en workers, queue, threadpool/processpool, ou service batch.

3. Nom de sortie fixe pour la reconstruction

/reconstruct écrit toujours :

reconstructed_output.pdf

Deux requêtes simultanées peuvent s’écraser. Même problème potentiel pour reconstructed_output.html.

4. Sécurité insuffisante

Pas d’authentification, CORS très permissif pour localhost, upload de fichiers Office/PDF, exécution LibreOffice headless : ce n’est pas acceptable tel quel en production exposée.

5. Nettoyage des fichiers absent ou insuffisant

Les fichiers uploadés, convertis, images sources, backgrounds, masks, visualisations et PDF reconstruits s’accumulent. Il faudrait une politique de nettoyage, TTL, job périodique ou stockage par session.

6. apply_ai_font_matching est désactivée

La fonction retourne un résumé indiquant que l’IA de police est retirée/désactivée. Donc le nom laisse penser à une fonctionnalité active, mais elle ne fait pas réellement de matching de police dans ce fichier.

7. Couplage fort à des méthodes privées

La traduction appelle des méthodes internes du traducteur comme :

tr._translate_text_hierarchical
tr._restore_protected_tokens
tr._normalize_translation
tr._apply_domain_glossary

C’est fragile. Une vraie interface publique serait préférable.

8. Paramètres inutilisés ou ambigus

process_page accepte translate_to, mais dans le code visible il n’est pas réellement exploité. Cela signale une dette technique.

Conclusion

ocr_server.py est le serveur central d’un système de traduction documentaire WYSIWYG. Sa fonction réelle est :

accepter PDF/image/Office ;
convertir/rendre les pages ;
extraire texte natif + OCR ;
fusionner et nettoyer les doublons ;
reconstituer layout, styles, rôles et phrases ;
protéger formules/code/références ;
créer un fond sans texte ;
produire une structure layout.v2;
traduire si demandé ;
reconstruire PDF/HTML ;
contrôler la fidélité.

Techniquement, l’ambition est bonne et cohérente avec ton projet vSense/docs_parser. Mais le fichier est devenu un gros orchestrateur monolithique. Pour le rendre robuste, il faut maintenant le découper, stabiliser les interfaces entre modules, rendre les jobs non bloquants, sécuriser les uploads et isoler les sorties par session/document.

Quels sont les entréees (codes python) appelés par ocr_server pour la lecture/extraction/détection/compréhension/classification des éléments d'une page ?
Thought for 56s

Oui. Dans ocr_server.py, les entrées Python appelées pour lire, extraire, détecter, comprendre et classifier les éléments d’une page se répartissent en plusieurs couches. Le point central est process_page(...), appelé par la route /ocr après conversion éventuelle du fichier en image/page PDF.

1. Entrée principale : /ocr → process_page(...)

Le serveur reçoit le fichier via :

@app.post("/ocr")
async def perform_ocr(...)

Puis il détecte le type de fichier :

if ext in OFFICE_EXTENSIONS:
    converted_pdf = _convert_office_to_pdf(save_path)
    doc = fitz.open(converted_pdf)

elif ext == ".pdf":
    doc = fitz.open(save_path)

else:
    img = Image.open(save_path).convert("RGB")

Ensuite, pour chaque page, il appelle :

process_page(
    img,
    i,
    base_name,
    pdf_page=doc[i],
    force_ai=force_ai,
    font_ai_audit=font_ai_audit,
    text_removal_mode=text_removal_mode,
)

Donc l’entrée réelle du pipeline page est :

def process_page(
    img,
    idx,
    filename,
    pdf_page=None,
    translate_to=None,
    force_ai=False,
    font_ai_audit=False,
    text_removal_mode="default",
    include_debug_layers=False
):

C’est la fonction qui orchestre la lecture, l’extraction, la détection, la compréhension et la classification de la page.

2. Entrées de lecture / conversion documentaire
2.1 _convert_office_to_pdf(input_path)

Utilisée pour les fichiers :

.doc, .docx, .ppt, .pptx, .odt, .odp

Elle appelle LibreOffice ou soffice :

soffice --headless --convert-to pdf

Rôle : transformer un document Office en PDF avant analyse.

2.2 fitz.open(...)

Utilisé pour ouvrir :

doc = fitz.open(save_path)

ou :

doc = fitz.open(converted_pdf)

Rôle : ouvrir le PDF avec PyMuPDF.

2.3 doc[i].get_pixmap(dpi=TARGET_DPI)

Utilisé pour rendre chaque page PDF en image :

pix = doc[i].get_pixmap(dpi=TARGET_DPI)
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

Rôle : convertir la page PDF en image exploitable par OCR/OpenCV/PIL.

2.4 Image.open(...).convert("RGB")

Utilisé si le fichier reçu est déjà une image :

img = Image.open(save_path).convert("RGB")

Rôle : charger une image source directement.

3. Entrées d’extraction brute
3.1 native_pdf_extractor.extract_page(...)

C’est l’entrée pour l’extraction native PDF :

native = native_pdf_extractor.extract_page(pdf_page, sx=sx, sy=sy)
native_blocks = native.get("blocks", [])
non_text_zones = native.get("non_text_zones", [])
native_images = native.get("images", [])
native_drawings = native.get("drawings", [])

Fichier/module appelé :

from native_pdf_extractor import NativePDFExtractor
native_pdf_extractor = NativePDFExtractor()

Rôle :

extraire le texte déjà présent dans le PDF ;
extraire les blocs textuels natifs ;
extraire les images ;
extraire les dessins/vectoriels ;
repérer les zones non textuelles ;
récupérer styles, polices, tailles, couleurs si l’extracteur les fournit.

C’est la meilleure source quand le PDF contient du vrai texte. L’OCR ne devrait venir qu’en complément.

3.2 engine_ocr(np.array(img))

C’est l’entrée OCR principale :

result, _ = engine_ocr(np.array(img))

Déclarée au démarrage :

from rapidocr_onnxruntime import RapidOCR
engine_ocr = RapidOCR()

Rôle :

détecter les zones de texte sur l’image ;
lire le texte ;
retourner les boîtes polygonales ;
retourner le score de confiance OCR.

Ensuite, ocr_server.py transforme chaque résultat OCR en dictionnaire :

raw_ocr.append({
    "label": txt,
    "bbox": bbox,
    "score": float(s)
})

Le code filtre aussi les zones OCR déjà couvertes par du texte natif PDF, pour éviter les doublons.

3.3 parser.parse(raw_ocr, img)

C’est l’entrée de structuration OCR :

ocr_structure = parser.parse(raw_ocr, img) if raw_ocr else []

Déclarée au démarrage :

from structure_extractor import DocumentParser
parser = DocumentParser()

Rôle probable, d’après l’usage dans ocr_server.py :

transformer les mots OCR bruts en blocs ;
regrouper les mots en lignes ;
regrouper les lignes en blocs ;
créer des phrases visuelles ;
créer des spans avec bbox, texte, style, etc.

C’est l’étape qui transforme :

raw_ocr = [{"label": ..., "bbox": ..., "score": ...}]

en structure exploitable :

blocks -> lines -> phrases -> spans
4. Entrées de nettoyage / consolidation après extraction

Après extraction native + OCR, le serveur appelle :

ocr_structure = _prune_weak_ocr_lines(ocr_structure)
final_blocks = _dedupe_final_blocks(native_blocks, ocr_structure)
final_blocks = _postprocess_blocks(final_blocks, img.width, img.height)
4.1 _prune_weak_ocr_lines(ocr_blocks)

Rôle :

supprimer les lignes OCR faibles ;
retirer les petits fragments suspects ;
nettoyer les queues de ligne isolées ;
recalculer la géométrie des blocs via _recompute_ocr_block_geometry.
4.2 _dedupe_final_blocks(native_blocks, ocr_blocks)

Rôle :

fusionner les blocs natifs PDF et OCR ;
supprimer les doublons ;
comparer chevauchement de boîtes ;
comparer similarité textuelle ;
conserver l’OCR seulement si ce n’est pas déjà couvert par le natif.
4.3 _postprocess_blocks(blocks, img_w, img_h)

Cette fonction appelle :

_merge_native_blocks(blocks)
_attach_inline_equation_blocks(merged)
_group_diagram_labels(merged, img_w, img_h)
_expand_diagram_groups(grouped, img_w, img_h)
_merge_overlapping_diagram_labels(grouped)
_enrich_layout_markers(grouped)

Rôle :

fusionner des blocs natifs proches ;
rattacher de petits fragments à des paragraphes ;
détecter/regrouper des labels de diagramme ;
marquer certains groupes comme background_only ;
préparer les lignes pour la reconstruction.

Cette étape est déjà une forme de compréhension géométrique de la page.

5. Entrées de détection spécialisée

Ces fonctions détectent les contenus particuliers.

5.1 _is_equation_like_text(text)

Détecte si un texte ressemble à une formule :

_is_equation_like_text(text)

Critères utilisés :

signes mathématiques ;
fractions ;
symboles grecs ;
opérateurs ;
faible longueur ;
forte densité de caractères symboliques.
5.2 _contains_greek_or_symbol(text)

Détecte :

α β λ Ω ∑ ∫ ∞ ≈ ≠ ≤ ≥ √ ∂ ± × ÷

Rôle : protéger les symboles scientifiques/mathématiques.

5.3 _is_reference_like_text(text)

Détecte les références courtes :

(2)
(i)
[12]
[3-5]

Rôle : éviter de les traduire ou de les reformater abusivement.

5.4 _is_immutable_inline_text(text)

Détecte les fragments qui doivent rester intacts :

formules ;
symboles grecs ;
références ;
formules chimiques ;
acronymes courts ;
puces ;
marqueurs visuels.
5.5 _phrase_is_immutable_programming_code(block, line, phrase)

Détecte le code informatique visible :

_phrase_is_immutable_programming_code(...)

Critères :

unit_type == "code_visible" ;
police monospace ;
présence de motifs de code comme function(...), name=..., stride=..., etc.
5.6 _extract_immutable_overlays(...)

Appelée ici :

immutable_overlays = _extract_immutable_overlays(final_blocks, img, filename, idx)

Rôle :

découper dans l’image source les fragments immuables ;
sauvegarder des mini-images immutable_...png ;
marquer les spans concernés comme skip_render;
mettre certaines phrases en background_only.

C’est crucial pour préserver formules, code, symboles, références ou éléments difficiles à reconstruire typographiquement.

6. Entrées de compréhension de layout
6.1 _enrich_layout_markers(blocks)

Appelée plusieurs fois :

_enrich_layout_markers(final_blocks)

Rôle :

détecter les puces ;
détecter les numérotations ;
calculer line_index;
calculer indent_px;
détecter les ruptures fortes ;
produire render_text_with_breaks;
produire source_layout_mode.

Elle utilise notamment :

_detect_leading_marker(text)
_infer_source_layout_mode(block)
6.2 _infer_source_layout_mode(block)

Classe le mode de flux du bloc :

continuous_paragraph
single_line
list
labels
fixed_labels
caption
paragraphs_with_hard_breaks
lead_in

Rôle : savoir si le texte peut être reflowé, doit garder ses lignes, ou doit rester fixe.

6.3 _annotate_layout(final_blocks, img.width, img.height)

Appelée ici :

layout_meta = _annotate_layout(final_blocks, img.width, img.height)

Elle calcule :

content_bbox;
marges ;
bande d’en-tête ;
bande de pied de page ;
rôle des blocs ;
alignements ;
indentations.

Elle classifie les blocs en rôles :

body
header
footer
title
section_heading
figure_caption
equation_inline
diagram_label
diagram_text_label

Cette fonction est une des fonctions principales de classification visuelle. Elle utilise la position, la largeur, le texte, la présence de motifs comme Figure, les numérotations de sections, et la détection de formules.

6.4 build_page_style_profile(...)

Importée depuis :

from style_profiler import build_page_style_profile

Appelée ainsi :

final_blocks, visual_style_profile = build_page_style_profile(
    final_blocks,
    layout_meta=layout_meta,
    page_width=img.width,
    page_height=img.height,
)

Rôle :

analyser le profil typographique de la page ;
enrichir les styles ;
produire visual_style_profile;
probablement distinguer titres, corps, styles dominants, variations de police.
7. Entrées de segmentation sémantique
7.1 _build_semantic_phrases_for_blocks(final_blocks)

Appelée plusieurs fois :

_build_semantic_phrases_for_blocks(final_blocks)

Elle appelle pour chaque bloc :

_build_semantic_phrases_for_block(block)

Rôle :

reconstituer des phrases complètes ;
éviter la traduction ligne par ligne ;
gérer les phrases sur plusieurs lignes ;
gérer les titres ;
gérer les blocs atomiques ;
gérer les séparateurs ;
gérer les formules ;
gérer les césures.

Elle utilise notamment :

_split_line_text_into_sentence_chunks(...)
_make_semantic_phrase(...)
_semantic_phrase_should_break_on_hard_boundary(...)
_block_is_implicitly_atomic(...)
_line_is_separator(...)
_separator_is_math_fraction(...)
_line_is_all_caps_heading(...)
7.2 _block_is_implicitly_atomic(lines)

Rôle : détecter si un bloc doit être traité ligne par ligne.

Cas typiques :

tableau ;
colonne de valeurs ;
labels de diagramme ;
texte très court répété ;
absence de vraie prose ;
lignes numériques ou symboliques.

Le code donne explicitement l’intention :

"""Detecte si un bloc doit etre traite ligne par ligne...
Analyse les lignes pour identifier colonnes de tableau, labels de diagramme,
valeurs numeriques — sans dependre d'un role preassigne par le layout AI.
"""

Donc cette fonction participe à la compréhension structurelle du bloc.

7.3 _postprocess_blocks_semantic(blocks)

Cette fonction choisit entre deux moteurs de correction sémantique :

if os.environ.get("PIPELINE_AGENT_P1_ENABLE") == "1":
    _p1_agent_postprocess_blocks(blocks)
else:
    _llm_postprocess_blocks(blocks)

Rôle : corriger la segmentation ambiguë.

Si LLM classique :
_llm_postprocess_blocks(blocks)

Il importe :

import llm_semantic_corrector as _corrector

Puis utilise :

_corrector.score_block_ambiguity(b)
_corrector.load_pipeline_if_needed()
_corrector.get_corrections(block, pipe)
_corrector.block_needs_strong_retry(...)
Si agent P1 :
_p1_agent_postprocess_blocks(blocks)

Il importe :

from pipeline_agents import get_agent
from pipeline_agents.p1_extraction import P1ExtractionAgent

Puis utilise :

P1ExtractionAgent.score_block(b)
agent.run(input_data)

Rôle : corriger :

frontières de phrases ;
lignes à traiter comme titres ;
formules ;
blocs atomiques ;
césures ;
modes de layout.

C’est la partie explicitement orientée compréhension par IA/LLM.

8. Entrées de classification traduisible / non traduisible
8.1 _annotate_translation_contracts(blocks, page_context=None)

Appelée ici :

_annotate_translation_contracts(final_blocks)

puis plus tard :

_annotate_translation_contracts(page_structure.get("blocks", []), page_context=page_structure)

Elle appelle :

page_policy_matrix.classify_unit_policy(...)

Déclaré au démarrage :

from page_policy_matrix import PagePolicyMatrix
page_policy_matrix = PagePolicyMatrix()

C’est l’entrée principale de classification fonctionnelle.

Elle annote chaque niveau :

bloc ;
ligne ;
phrase ;
span ;
semantic phrase.

Elle attribue notamment :

translatable
unit_type
translation_strategy
coverage_required
render_policy
expression_semantics
editorial_semantics
structural_context
editorial_relations
expression_relations

Types ou classes possibles visibles dans le code :

citation
reference_link
code_visible
formula
formula_label
plain_text
technical_inline
label

Stratégies possibles visibles :

background_only
exact_preserve
keep_original
layout_constrained
anchored_text

Donc cette fonction ne lit pas la page ; elle comprend les unités extraites et décide comment elles doivent être traitées.

9. Entrées de construction sémantique avancée

Après les contrats de traduction, le serveur construit des couches plus hautes.

9.1 _build_semantic_spans_for_blocks(blocks)

Appelée ici :

_build_semantic_spans_for_blocks(final_blocks)

Rôle :

fusionner des spans visuels en spans sémantiques ;
gérer les continuations entre lignes ;
conserver les styles ;
créer des relations entre spans ;
différencier continuation et changement sémantique.
9.2 _build_semantic_runs_for_blocks(blocks)

Appelée ici :

_build_semantic_runs_for_blocks(final_blocks)

Rôle :

regrouper des unités compatibles ;
créer des semantic_runs;
détecter les classes inline dominantes ;
préserver code, référence, formule comme groupes protégés.
9.3 _build_semantic_groups_for_blocks(blocks)

Appelée ici :

_build_semantic_groups_for_blocks(final_blocks)

Elle utilise :

_classify_semantic_group(units)

Classes visibles :

label_value
technical_group
code_group
reference_group
formula_group
name_version
editorial_group

C’est une couche de compréhension plus haute que la phrase simple. Elle sert à regrouper des éléments qui fonctionnent ensemble : label + valeur, nom + version, groupe technique, etc.

10. Entrées de classification des phrases
_classify_semantic_phrase_kind(block, phrase)

Cette fonction classe une phrase sémantique en :

empty
header_footer
caption
structural
formula
prose

Elle tient compte :

du rôle du bloc ;
des classes inline ;
des formules ;
des références ;
des puces ;
des numérotations ;
des symboles mathématiques ;
des titres ;
des phrases sans ponctuation ;
du nombre de mots ;
du profil de casse.

C’est une fonction importante pour distinguer prose réelle vs élément structurel vs formule.

11. Entrées de table des matières
11.1 _native_toc_rows_from_pdf_page(pdf_page, sx, sy)

Cette fonction lit directement les mots PDF :

words = pdf_page.get_text("words")

Elle essaie de reconnaître une table des matières par scoring :

présence de contents, table of contents, sommaire;
numérotations ;
pages en fin de ligne ;
puces ou marqueurs.

Elle produit des lignes :

toc_entry
toc_title
chapter_heading
section_heading
subentry
page_marker
11.2 _inject_native_toc_rows(page_structure, pdf_page, sx, sy)

Appelée ici :

page_structure = _inject_native_toc_rows(page_structure, pdf_page, sx=sx, sy=sy)

Rôle : injecter une structure de table des matières native si le layout ne l’a pas déjà capturée.

12. Entrées de layout canonique / enrichissement IA
12.1 layout_v2_builder.build(page_structure)

Déclaré au démarrage :

from structure_extractor import LayoutV2Builder
layout_v2_builder = LayoutV2Builder()

Appelé ici :

page_structure = layout_v2_builder.build(page_structure)

Rôle :

construire une structure canonique layout.v2;
stabiliser les données pour traduction/reconstruction ;
probablement ajouter page_role, document_type, layout_type, page_family, etc.
12.2 layout_ai_enricher.enrich(page_structure, img)

Déclaré ainsi :

from layout_ai_enricher import get_layout_ai_enricher
layout_ai_enricher = get_layout_ai_enricher()

Appelé seulement si :

EXTRACTION_AI_ENABLED

Code :

page_structure, layout_ai_info = layout_ai_enricher.enrich(page_structure, img)

Rôle :

enrichissement IA de la mise en page ;
ajout/correction de régions ;
amélioration possible du rôle des éléments ;
correction de classification visuelle.
12.3 apply_page_extraction_postprocessors(page_structure)

Importé depuis :

from page_extraction_postprocessors import apply_page_extraction_postprocessors

Appelé ici :

page_structure, postprocess_info = apply_page_extraction_postprocessors(page_structure)

Rôle :

post-traiter la page après construction layout.v2;
corriger des cas particuliers ;
déclencher une reconstruction layout_v2_builder.build(...) si la structure a changé.
13. Entrées de fond / suppression du texte

Même si ce n’est pas de la compréhension pure, c’est lié à la reconstruction de page.

13.1 _collect_text_regions_for_inpainting(...)

Appelée ici :

text_regions = _collect_text_regions_for_inpainting(
    final_blocks,
    non_text_zones,
    immutable_overlays=immutable_overlays
)

Rôle :

collecter les zones de texte à retirer du fond ;
protéger les zones non textuelles ;
protéger les overlays immuables ;
protéger les diagrammes.
13.2 text_removal_strategy.remove(...)

Déclaré :

from text_removal_strategy import TextRemovalStrategy
text_removal_strategy = TextRemovalStrategy()

Appelé :

clean_bgr, mask, text_removal_debug = text_removal_strategy.remove(
    img,
    text_regions,
    mode=text_removal_mode
)

Rôle :

supprimer le texte de l’image ;
produire le fond maître nettoyé ;
produire le masque.
13.3 _erase_uncovered_pdf_words(...)

Appelée si PDF :

clean_bgr = _erase_uncovered_pdf_words(clean_bgr, pdf_page, text_regions, sx, sy)

Rôle :

effacer les mots PDF natifs qui n’ont pas été couverts par l’extraction principale.
13.4 _p6_audit_background(...)

Appelée ici :

p6_bg_audit = _p6_audit_background(...)

Elle peut charger :

from pipeline_agents import get_agent
agent = get_agent("p6_background")

Rôle : audit IA optionnel du fond nettoyé.

14. Entrées d’export de compréhension
14.1 _build_hierarchical_extraction(blocks)

Appelée ici :

hierarchical_extraction = _build_hierarchical_extraction(final_blocks)

Elle produit :

phrases
groupes_mots
mots
residuels
chiffres
nombres
lettres
symboles
counts

C’est une sortie hiérarchique textuelle, utile pour traduction ou diagnostic.

14.2 _build_fidelity_layout_export(blocks)

Appelée ici :

fidelity_layout = _build_fidelity_layout_export(final_blocks)

Rôle :

produire un export détaillé orienté fidélité ;
conserver blocs, lignes, spans, styles, alignements, bboxes, modes de layout.
14.3 _write_layout_xml(...)

Appelée ici :

layout_xml_path = _write_layout_xml(final_blocks, filename, idx, img.width, img.height)

Rôle : exporter la structure en XML de debug.

15. Résumé en ordre d’appel réel

Voici la chaîne principale simplifiée :

/ocr
  -> perform_ocr(...)
      -> _convert_office_to_pdf(...)        # si Office
      -> fitz.open(...)                     # si PDF / Office converti
      -> doc[i].get_pixmap(...)             # rendu image
      -> Image.open(...)                    # si image directe
      -> process_page(...)

Puis dans process_page(...) :

native_pdf_extractor.extract_page(...)
engine_ocr(np.array(img))
parser.parse(raw_ocr, img)

_prune_weak_ocr_lines(...)
_dedupe_final_blocks(...)
_postprocess_blocks(...)

_enrich_layout_markers(...)
_build_semantic_phrases_for_blocks(...)
_postprocess_blocks_semantic(...)
_annotate_translation_contracts(...)
_build_semantic_spans_for_blocks(...)
_build_semantic_runs_for_blocks(...)
_build_semantic_groups_for_blocks(...)

_extract_immutable_overlays(...)
_annotate_layout(...)
build_page_style_profile(...)

_collect_text_regions_for_inpainting(...)
text_removal_strategy.remove(...)
_erase_uncovered_pdf_words(...)
_p6_audit_background(...)

_build_hierarchical_extraction(...)
_build_fidelity_layout_export(...)
_write_layout_xml(...)

layout_v2_builder.build(...)
layout_ai_enricher.enrich(...)              # si activé
apply_page_extraction_postprocessors(...)
_inject_native_toc_rows(...)

_enrich_layout_markers(...)
_build_semantic_phrases_for_blocks(...)
_postprocess_blocks_semantic(...)
_annotate_translation_contracts(...)
_build_semantic_spans_for_blocks(...)
_build_semantic_runs_for_blocks(...)
_build_semantic_groups_for_blocks(...)
_build_hierarchical_extraction(...)
_build_fidelity_layout_export(...)
16. Les fichiers Python externes réellement appelés pour lecture/extraction/détection/classification

D’après les imports de ocr_server.py, les modules externes importants sont :

from rapidocr_onnxruntime import RapidOCR
from structure_extractor import DocumentParser, LayoutV2Builder
from native_pdf_extractor import NativePDFExtractor
from style_profiler import build_page_style_profile
from page_policy_matrix import PagePolicyMatrix
from page_extraction_postprocessors import apply_page_extraction_postprocessors
from layout_ai_enricher import get_layout_ai_enricher
from text_removal_strategy import TextRemovalStrategy

Et optionnellement, chargés dynamiquement :

import llm_semantic_corrector
from pipeline_agents import get_agent
from pipeline_agents.p1_extraction import P1ExtractionAgent

Donc les vrais fichiers/modules à inspecter ensuite sont, par priorité :

native_pdf_extractor.py : extraction native PDF.
structure_extractor.py : parsing OCR + LayoutV2Builder.
page_policy_matrix.py : classification traduisible / non traduisible.
style_profiler.py : profilage typographique.
page_extraction_postprocessors.py : corrections post-extraction.
layout_ai_enricher.py : enrichissement IA optionnel.
llm_semantic_corrector.py : correction sémantique LLM optionnelle.
pipeline_agents/p1_extraction.py : agent de correction sémantique optionnel.
text_removal_strategy.py : suppression du texte du fond.
pipeline_agents/p6_background.py : audit optionnel du fond.

Conclusion nette : dans ocr_server.py, la lecture brute est faite par PyMuPDF/PIL/RapidOCR, l’extraction structurée par NativePDFExtractor et DocumentParser, la compréhension de layout par _postprocess_blocks, _annotate_layout, LayoutV2Builder, style_profiler, et la classification sémantique/traduction par _annotate_translation_contracts, PagePolicyMatrix, les fonctions semantic_ et éventuellement les agents LLM/P1*.

element_relations.py
Python
element_relations_ai.py
Python
llm_semantic_corrector.py
Python
perfect_document_extractor.py
Python
perfect_extraction_to_reconstruction.py
Python
Qu'en est-il de ces codes d'extractions plus performant? sont-ils intégrés dans l'orchestrateur ocr_server.py ?
Thought for 30s

Oui, mais pas tous. Réponse claire :

Fichier	Intégré dans ocr_server.py ?	Statut réel
llm_semantic_corrector.py	Oui, partiellement	Intégré comme post-processeur sémantique optionnel
element_relations.py	Non	Présent, utile, mais non appelé par l’orchestrateur
element_relations_ai.py	Non	Présent, dépend de element_relations, mais non appelé
perfect_document_extractor.py	Non	Pipeline autonome séparé
perfect_extraction_to_reconstruction.py	Non	Adaptateur expérimental séparé
1. Ce qui est réellement intégré : llm_semantic_corrector.py

Celui-là est bien appelé par ocr_server.py.

Dans ocr_server.py, il y a une fonction :

def _llm_postprocess_blocks(blocks: list) -> None:
    try:
        import llm_semantic_corrector as _corrector
    except ImportError:
        return

Puis elle utilise :

_corrector.score_block_ambiguity(b)
_corrector.load_pipeline_if_needed()
_corrector.get_corrections(block, pipe)
_corrector.block_needs_strong_retry(...)

Donc llm_semantic_corrector.py est intégré, mais pas comme extracteur principal. Il intervient après l’extraction OCR/PDF, pour corriger la segmentation sémantique des blocs ambigus.

Il corrige notamment :

les titres implicites ;
les légendes multi-lignes ;
les formules/code non traduisibles ;
les césures de mots ;
les labels courts de diagramme ;
les découpages intra-ligne quand une ligne contient plusieurs unités sémantiques.

Dans le pipeline process_page, ce correcteur est appelé ici :

_build_semantic_phrases_for_blocks(...)
_postprocess_blocks_semantic(...)
_annotate_translation_contracts(...)

Donc il arrive après la création des blocs, lignes et phrases heuristiques, et avant les contrats de traduction, les semantic spans, runs et groups.

Point important : il est optionnel. Si le modèle n’est pas disponible, ocr_server.py continue avec les heuristiques seules. C’est explicitement prévu dans llm_semantic_corrector.py.

2. element_relations.py : non intégré

element_relations.py contient une fonction intéressante :

def enrich_element_relations(page_data):

Elle prend page_data, parcourt les blocs, ordonne les phrases, crée des relations entre phrases voisines, puis ajoute :

page_data["element_relations"]
page_data["layout"]["element_relations"]

Elle produit notamment :

reading_order;
pair_relations;
visual_relation;
logical_relation;
continuation;
confidence;
ai_review_required.

Mais dans les imports de ocr_server.py, on voit bien les modules appelés actuellement :

NativePDFExtractor
DocumentParser
LayoutV2Builder
PagePolicyMatrix
apply_page_extraction_postprocessors
get_layout_ai_enricher

Il n’y a pas element_relations, pas enrich_element_relations.

Donc : le code existe, mais il n’est pas branché dans l’orchestrateur.

3. element_relations_ai.py : non intégré

element_relations_ai.py est le complément IA de element_relations.py.

Il définit :

class ElementRelationsAIEnricher

avec des hypothèses NLI de type :

"continuation": "the second fragment is a continuation of the previous text"
"new_unit": "the second fragment starts a new textual unit"

et des relations logiques :

same_token_continuation
same_sentence_continuation
same_paragraph_continuation
new_list_item
new_sentence_or_unit
new_structural_unit

Il cherche donc à décider si deux fragments successifs doivent être considérés comme une continuation ou comme une nouvelle unité.

Mais il attend que page_data["element_relations"] existe déjà :

payload = page_data.get("element_relations") or {}
flat_relations = payload.get("flat_relations") or []

Donc il dépend logiquement de element_relations.py.

Lui non plus n’est pas importé dans ocr_server.py. Le fichier fournit pourtant un getter :

def get_element_relations_ai_enricher():

mais ocr_server.py ne l’appelle pas.

Conclusion : non intégré.

4. perfect_document_extractor.py : non intégré, et explicitement séparé

Celui-ci est plus important. Il ne s’agit pas d’un petit enrichisseur, mais d’un pipeline d’extraction autonome.

Le fichier le dit explicitement :

“This module is intentionally separate from the current extraction pipeline.”

Il lit directement un PDF et produit un modèle documentaire plus riche, destiné à être relié plus tard au pipeline existant après validation.

Dans son modèle, il indique aussi :

"uses_current_pipeline_extraction_as_input": False

Donc il ne consomme pas la sortie de ocr_server.py, et ocr_server.py ne l’utilise pas non plus.

Ce qu’il fait :

extraction PDF native directe ;
extraction des polices embarquées ;
extraction texte caractère par caractère ;
extraction images ;
extraction dessins/vectoriels ;
OCR seulement si nécessaire ;
détection de régions spéciales ;
tables ;
formules/code ;
relations ;
contraintes de reconstruction ;
rapport qualité.

Il supporte actuellement les sources PDF uniquement :

if source_path.suffix.lower() != ".pdf":
    raise ValueError("perfect_document_extractor currently supports PDF sources.")

Donc : ce code est probablement plus ambitieux que le pipeline actuel, mais il n’est pas intégré dans ocr_server.py.

5. perfect_extraction_to_reconstruction.py : non intégré, adaptateur expérimental

Ce fichier sert à convertir la sortie de PerfectDocumentExtractor vers le contrat historique du DocumentReconstructor.

Son commentaire d’en-tête est très clair :

“This file is experimental and intentionally does not modify the existing pipeline.”

Il convertit le modèle perfect_extraction vers l’ancien format de reconstruction, tout en transportant les métadonnées riches.

Il importe :

from perfect_document_extractor import ExtractionConfig, PerfectDocumentExtractor

et définit :

class PerfectExtractionReconstructionAdapter

Il précise même dans le payload final :

"does_not_modify_existing_pipeline": True

Donc : il n’est pas intégré à l’orchestrateur actuel ; il est conçu pour faire le pont plus tard.

Conclusion nette

Aujourd’hui, ocr_server.py utilise encore ce pipeline principal :

NativePDFExtractor
RapidOCR
DocumentParser
_postprocess_blocks
LayoutV2Builder
layout_ai_enricher
apply_page_extraction_postprocessors
llm_semantic_corrector
PagePolicyMatrix
semantic_spans / semantic_runs / semantic_groups

Le seul des nouveaux fichiers réellement intégré est :

llm_semantic_corrector.py

Mais uniquement comme correcteur de segmentation sémantique, pas comme moteur d’extraction complet.

Les autres fichiers :

element_relations.py
element_relations_ai.py
perfect_document_extractor.py
perfect_extraction_to_reconstruction.py

sont présents mais non branchés.

Ma lecture technique : tu as actuellement deux architectures parallèles :

Architecture active : ocr_server.py, fonctionnelle, mais héritée et monolithique.
Architecture “perfect” / relations : plus riche, plus propre conceptuellement, mais encore expérimentale et non intégrée.

La prochaine étape logique n’est pas de remplacer brutalement ocr_server.py, mais d’ajouter ces modules dans l’orchestrateur avec des flags :

USE_ELEMENT_RELATIONS=1
USE_ELEMENT_RELATIONS_AI=1
USE_PERFECT_EXTRACTOR=1

et de comparer page par page la qualité d’extraction avant d’en faire le pipeline par défaut.

Que faire alors pour intégrer ces améliorations dans l'orchestrateur et que cela soit profitable au projet ?

Je vais raisonner en mode intégration progressive : ne pas remplacer le pipeline actif d’un coup, mais brancher les nouveaux modules à des points précis, mesurables et réversibles.

Thought for 23s

Il faut intégrer ces améliorations en 3 niveaux, pas tout mélanger immédiatement dans process_page. Le bon schéma est :

brancher les enrichisseurs légers dans le pipeline actif ;
ajouter perfect_document_extractor comme pipeline alternatif contrôlé par flag ;
comparer les deux pipelines avant de remplacer le pipeline historique.

ocr_server.py utilise déjà un pipeline actif : NativePDFExtractor, RapidOCR, DocumentParser, LayoutV2Builder, TextRemovalStrategy, PagePolicyMatrix, layout_ai_enricher, etc. Il ne faut donc pas le casser. Il faut l’enrichir.

1. Intégrer d’abord element_relations.py

C’est l’amélioration la plus simple et la moins risquée.

element_relations.py expose déjà une fonction propre :

def enrich_element_relations(page_data):
    ...

Elle ajoute dans page_data :

page_data["element_relations"]
page_data["layout"]["element_relations"]

avec les relations entre phrases, l’ordre de lecture, les continuités, les ruptures de phrase, les relations logiques et les scores de confiance.

Ce que ça apporte

Actuellement, ton pipeline sait extraire :

bloc → ligne → phrase → span

Mais il ne sait pas toujours si deux phrases voisines sont :

la suite de la même phrase
un nouveau paragraphe
un nouvel item de liste
une rupture structurelle
une phrase coupée par retour ligne

element_relations.py ajoute précisément cette couche :

phrase A → relation → phrase B

Donc c’est utile pour :

améliorer la traduction ;
éviter de traduire ligne par ligne ;
reconstruire les paragraphes correctement ;
gérer les retours à la ligne visuels ;
réduire les erreurs de reflow ;
distinguer paragraphes, listes, titres et légendes.
Où l’intégrer dans ocr_server.py

À intégrer après la construction finale des phrases sémantiques, pas avant.

Dans process_page, tu as déjà cette chaîne :

_enrich_layout_markers(final_blocks)
_build_semantic_phrases_for_blocks(final_blocks)
_postprocess_blocks_semantic(final_blocks)
_annotate_translation_contracts(final_blocks)
_build_semantic_spans_for_blocks(final_blocks)
_build_semantic_runs_for_blocks(final_blocks)
_build_semantic_groups_for_blocks(final_blocks)

Il faut ajouter element_relations après cette étape, lorsque les phrases sont stabilisées.

Patch minimal

En haut de ocr_server.py, ajouter :

USE_ELEMENT_RELATIONS = os.getenv("DOCS_PARSER_USE_ELEMENT_RELATIONS", "1") == "1"
USE_ELEMENT_RELATIONS_AI = os.getenv("DOCS_PARSER_USE_ELEMENT_RELATIONS_AI", "0") == "1"

try:
    from element_relations import enrich_element_relations
except Exception:
    enrich_element_relations = None

try:
    from element_relations_ai import get_element_relations_ai_enricher
except Exception:
    get_element_relations_ai_enricher = None

Puis créer un wrapper propre :

def _apply_element_relations(page_structure: dict) -> dict:
    if not isinstance(page_structure, dict):
        return page_structure

    page_structure.setdefault("extraction_enrichment", {})

    if USE_ELEMENT_RELATIONS and enrich_element_relations is not None:
        try:
            page_structure = enrich_element_relations(page_structure)
            page_structure["extraction_enrichment"]["element_relations"] = {
                "enabled": True,
                "applied": True,
                "error": None,
            }
        except Exception as exc:
            page_structure["extraction_enrichment"]["element_relations"] = {
                "enabled": True,
                "applied": False,
                "error": str(exc),
            }

    if USE_ELEMENT_RELATIONS_AI and get_element_relations_ai_enricher is not None:
        try:
            enricher = get_element_relations_ai_enricher()
            page_structure, info = enricher.enrich(page_structure)
            page_structure["extraction_enrichment"]["element_relations_ai"] = info
        except Exception as exc:
            page_structure["extraction_enrichment"]["element_relations_ai"] = {
                "enabled": True,
                "applied": False,
                "error": str(exc),
            }

    return page_structure

Ensuite, dans process_page, après la dernière reconstruction sémantique de page_structure["blocks"], ajouter :

page_structure = _apply_element_relations(page_structure)

Le meilleur emplacement est après :

_annotate_translation_contracts(page_structure.get("blocks", []), page_context=page_structure)
_build_semantic_spans_for_blocks(page_structure.get("blocks", []))
_build_semantic_runs_for_blocks(page_structure.get("blocks", []))
_build_semantic_groups_for_blocks(page_structure.get("blocks", []))

et avant le return.

2. Intégrer ensuite element_relations_ai.py

element_relations_ai.py ne doit pas être appelé seul. Il dépend d’abord de page_data["element_relations"], car il lit les flat_relations et ne traite que les relations ambiguës ou incertaines.

Donc l’ordre obligatoire est :

page_structure = enrich_element_relations(page_structure)
page_structure, info = get_element_relations_ai_enricher().enrich(page_structure)
Ce que ça apporte

element_relations_ai.py utilise un modèle NLI ONNX pour décider si deux fragments sont :

continuation
nouvelle unité
suite du même token
suite de phrase
suite de paragraphe
nouvel item de liste
nouvelle unité structurelle

Ces hypothèses sont déjà codées dans le fichier.

C’est utile pour les cas que les heuristiques traitent mal :

The model learns
features from data.

ou :

Figure 4.17
A learning rate larger than...

ou encore :

guaran-
teed
Recommandation

Activer par défaut :

DOCS_PARSER_USE_ELEMENT_RELATIONS=1

Mais garder l’IA désactivée au début :

DOCS_PARSER_USE_ELEMENT_RELATIONS_AI=0

Puis l’activer seulement sur documents complexes :

DOCS_PARSER_USE_ELEMENT_RELATIONS_AI=1
ELEMENT_RELATIONS_AI_ENABLE=1

Pourquoi ? Parce que element_relations.py est léger et déterministe. element_relations_ai.py dépend d’un modèle local ONNX, donc il faut vérifier disponibilité, latence et stabilité.

3. Ne pas remplacer tout de suite NativePDFExtractor par PerfectDocumentExtractor

perfect_document_extractor.py est plus ambitieux, mais il est explicitement conçu comme module autonome séparé du pipeline actuel. Le fichier précise qu’il est “intentionally separate from the current extraction pipeline” et qu’il produit un modèle plus riche pouvant être relié plus tard après validation.

Il indique aussi que sa philosophie est :

native_pdf_first_ocr_when_needed_ai_assisted_special_regions

et que :

uses_current_pipeline_extraction_as_input = False

Donc il ne faut pas l’insérer directement au milieu de process_page. Il faut l’intégrer comme mode alternatif d’extraction.

Ajouter un paramètre extraction_mode

Dans /ocr, ajouter un paramètre :

extraction_mode: str = "legacy"

Modes recommandés :

legacy          → pipeline actuel
enhanced        → pipeline actuel + element_relations
perfect         → PerfectDocumentExtractor + adapter
hybrid_compare  → legacy + perfect, avec rapport de comparaison

Exemple :

@app.post("/ocr")
async def perform_ocr(
    file: UploadFile = File(...),
    force_ai: bool = False,
    font_ai_audit: bool = FONT_AI_AUDIT_DEFAULT,
    text_removal_mode: str = "default",
    extraction_mode: str = "legacy",
):
4. Ajouter un chemin perfect séparé

Créer une fonction dédiée :

def _extract_with_perfect_pipeline(pdf_path: str, pages=None) -> dict:
    from perfect_document_extractor import ExtractionConfig, PerfectDocumentExtractor
    from perfect_extraction_to_reconstruction import PerfectExtractionReconstructionAdapter

    extractor = PerfectDocumentExtractor(
        ExtractionConfig(
            target_dpi=TARGET_DPI,
            enable_ocr=True,
            enable_special_regions=True,
            extract_embedded_fonts=True,
        )
    )

    perfect_model = extractor.extract(pdf_path, pages=pages)

    adapter = PerfectExtractionReconstructionAdapter(output_dir=RESULTS_DIR)
    adapted = adapter.adapt(perfect_model)

    return {
        "mode": "perfect",
        "perfect_model": perfect_model,
        "adapted_structure": adapted,
        "pages": adapted.get("pages", []),
        "quality_report": perfect_model.get("quality_report"),
    }

Pourquoi passer par l’adapter ? Parce que perfect_extraction_to_reconstruction.py est justement conçu pour convertir le modèle riche perfect_extraction vers le contrat historique du reconstructeur, tout en gardant les métadonnées complètes.

5. Modifier /ocr pour router vers le bon pipeline

Dans perform_ocr, après sauvegarde/conversion en PDF, ajouter :

if extraction_mode == "perfect":
    if ext not in {".pdf"} and not converted_pdf:
        return JSONResponse(
            status_code=400,
            content={"error": "Le mode perfect supporte actuellement les PDF ou les fichiers Office convertis en PDF."}
        )

    pdf_path = converted_pdf or save_path
    result = _extract_with_perfect_pipeline(pdf_path)
    return JSONResponse(content=result)

Pour les documents Office, ton serveur convertit déjà en PDF via LibreOffice avant traitement. Donc le mode perfect peut fonctionner après conversion Office → PDF.

6. Ajouter un mode hybrid_compare

C’est probablement le plus utile pour ton projet.

Objectif : exécuter les deux pipelines sur le même PDF :

legacy/enhanced
perfect

puis comparer :

nombre de blocs
nombre de lignes
nombre de mots
nombre d’images
nombre de dessins
nombre de tableaux
nombre de régions spéciales
score de reconstructibilité
couverture texte
éléments non traduisibles
formules/code détectés

PerfectDocumentExtractor produit déjà un quality_report avec field_presence_score, mean_page_reconstructibility_score, total_text_blocks, total_images, total_drawings, total_tables, total_special_regions, etc.

Créer :

def _compare_extraction_outputs(legacy_pages: list, perfect_result: dict) -> dict:
    perfect_quality = perfect_result.get("quality_report") or {}
    perfect_pages = perfect_result.get("pages") or []

    return {
        "legacy": {
            "page_count": len(legacy_pages),
            "block_count": sum(len((p.get("structure") or {}).get("blocks") or []) for p in legacy_pages),
            "has_layout_v2": any((p.get("structure") or {}).get("schema_version") == "layout.v2" for p in legacy_pages),
        },
        "perfect": {
            "page_count": len(perfect_pages),
            "block_count": sum(len(p.get("blocks") or []) for p in perfect_pages),
            "quality_report": perfect_quality,
        },
        "decision_hint": "manual_review_required",
    }

Puis dans /ocr :

if extraction_mode == "hybrid_compare":
    legacy_pages = []
    # exécuter le pipeline actuel normalement
    # ...
    perfect_result = _extract_with_perfect_pipeline(pdf_path)

    comparison = _compare_extraction_outputs(legacy_pages, perfect_result)

    return JSONResponse(content={
        "mode": "hybrid_compare",
        "legacy_pages": legacy_pages,
        "perfect_result": perfect_result,
        "comparison": comparison,
    })

Ce mode est fondamental : il permet de voir objectivement si le pipeline “perfect” est vraiment meilleur sur tes documents réels.

7. Faire du pipeline actif un pipeline enhanced

Le mode enhanced doit correspondre à :

pipeline actuel
+ llm_semantic_corrector
+ element_relations
+ element_relations_ai optionnel
+ PagePolicyMatrix
+ postprocessors

llm_semantic_corrector.py est déjà bien orienté “compréhension du texte” : il corrige titres implicites, légendes multi-lignes, formules/code, césures, labels de diagramme et découpages intra-ligne. Il doit rester dans le pipeline actif, mais son usage doit être mieux exposé dans les métadonnées de sortie.

Ajoute dans page_structure :

page_structure.setdefault("pipeline_trace", {})
page_structure["pipeline_trace"]["semantic_corrector"] = {
    "module": "llm_semantic_corrector",
    "enabled": True,
    "stage": "post_semantic_phrase_build",
}

Et pour element_relations :

page_structure["pipeline_trace"]["element_relations"] = {
    "module": "element_relations",
    "enabled": USE_ELEMENT_RELATIONS,
    "stage": "after_semantic_groups",
}

Cela rend les résultats auditables.

8. Ajouter une sortie de diagnostic dans /healthz

Actuellement /healthz doit dire quels modules sont actifs.

Ajouter :

@app.get("/healthz")
def healthcheck():
    element_relations_status = {
        "enabled": USE_ELEMENT_RELATIONS,
        "available": enrich_element_relations is not None,
    }

    element_relations_ai_status = {
        "enabled": USE_ELEMENT_RELATIONS_AI,
        "available": get_element_relations_ai_enricher is not None,
    }

    if get_element_relations_ai_enricher is not None:
        try:
            element_relations_ai_status["runtime"] = get_element_relations_ai_enricher().status()
        except Exception as exc:
            element_relations_ai_status["runtime_error"] = str(exc)

    return {
        "status": "ok",
        "service": "docs-parser",
        "extraction_modes": ["legacy", "enhanced", "perfect", "hybrid_compare"],
        "element_relations": element_relations_status,
        "element_relations_ai": element_relations_ai_status,
        "layout_ai_enabled": EXTRACTION_AI_ENABLED,
        "results_url": "/results",
    }
9. Ordre recommandé d’intégration
Phase 1 — faible risque

Intégrer seulement :

element_relations.py

Valeur immédiate :

meilleure continuité des phrases ;
meilleur reflow ;
meilleure reconstruction des paragraphes ;
aucune dépendance lourde.

Activation :

DOCS_PARSER_USE_ELEMENT_RELATIONS=1
DOCS_PARSER_USE_ELEMENT_RELATIONS_AI=0
Phase 2 — risque moyen

Activer :

element_relations_ai.py

seulement si le modèle ONNX local est disponible.

Activation :

DOCS_PARSER_USE_ELEMENT_RELATIONS=1
DOCS_PARSER_USE_ELEMENT_RELATIONS_AI=1
ELEMENT_RELATIONS_AI_ENABLE=1

Valeur :

meilleure résolution des cas ambigus ;
meilleure distinction continuation / nouvelle unité ;
meilleure préparation de la traduction.
Phase 3 — expérimentation contrôlée

Ajouter :

extraction_mode=perfect

mais sans remplacer le pipeline actif.

Valeur :

extraction PDF native plus riche ;
caractères, polices, images, dessins, tableaux, régions spéciales ;
rapport qualité ;
meilleure base pour WYSIWYG haut niveau.
Phase 4 — comparaison systématique

Ajouter :

extraction_mode=hybrid_compare

C’est là que tu décides si perfect_document_extractor doit devenir le moteur principal.

Phase 5 — bascule progressive

Quand les résultats sont meilleurs sur tes documents types :

legacy → fallback
enhanced → défaut
perfect → défaut pour PDF natifs complexes
10. Architecture cible recommandée

À terme, l’orchestrateur devrait ressembler à ceci :

/ocr
 ├── mode=legacy
 │    └── pipeline actuel
 │
 ├── mode=enhanced
 │    └── pipeline actuel
 │        ├── llm_semantic_corrector
 │        ├── element_relations
 │        └── element_relations_ai optionnel
 │
 ├── mode=perfect
 │    └── PerfectDocumentExtractor
 │        └── PerfectExtractionReconstructionAdapter
 │
 └── mode=hybrid_compare
      ├── enhanced
      ├── perfect
      └── rapport comparatif
11. Ce qu’il ne faut pas faire

Ne fais pas ceci :

final_blocks = PerfectDocumentExtractor(...)

directement dans process_page.

Pourquoi ? Parce que PerfectDocumentExtractor travaille au niveau document PDF complet, pas au niveau simple image/page déjà rendue. Il produit son propre modèle documentaire complet, avec ressources, polices, pages, images, dessins, régions spéciales et rapports qualité.

Ne fais pas non plus :

element_relations_ai.enrich(page_structure)

avant :

enrich_element_relations(page_structure)

car l’IA lit page_data["element_relations"]["flat_relations"].

12. Priorité exacte pour ton projet

La meilleure stratégie est celle-ci :

1. Brancher element_relations.py dans ocr_server.py
2. Ajouter le flag DOCS_PARSER_USE_ELEMENT_RELATIONS
3. Ajouter element_relations_ai.py derrière un flag désactivé par défaut
4. Ajouter extraction_mode="perfect"
5. Ajouter extraction_mode="hybrid_compare"
6. Mesurer les différences
7. Faire de enhanced le mode par défaut
8. Garder perfect pour PDF natifs complexes jusqu’à validation

La valeur immédiate viendra de element_relations.py. La valeur stratégique viendra de perfect_document_extractor.py, mais seulement après comparaison réelle sur tes PDF, car ce module est explicitement expérimental/autonome et non conçu pour modifier immédiatement le pipeline existant.

Il faut aussi normaliser les "unités" d'extraction : utiliser les points plutôt que les pixels

Oui. C’est même un point structurant : l’extraction doit avoir une unité canonique stable. Les pixels doivent devenir une représentation de rendu, pas l’unité source du modèle documentaire.

Thought for 19s

Oui. Il faut le faire. Pour un projet de reconstruction WYSIWYG, l’unité canonique d’extraction doit être le point PDF, pas le pixel.

Le pixel doit rester uniquement une unité de rendu, debug visuel, OCR image, inpainting, masques, backgrounds. Le modèle documentaire, lui, doit parler en points.

1. Pourquoi les points sont meilleurs que les pixels

Un PDF travaille naturellement en points :

1 point = 1/72 inch

Donc une page A4 PDF fait environ :

595 × 842 pt

Alors que ses dimensions en pixels changent selon le DPI :

à 150 DPI → environ 1240 × 1754 px
à 300 DPI → environ 2480 × 3508 px

Si ton extraction est en pixels, alors toute ta structure dépend du DPI choisi au moment du rendu. C’est mauvais pour :

la fidélité WYSIWYG ;
la comparaison entre extracteurs ;
la reconstruction PDF ;
la stabilité des bbox;
la traduction avec reflow ;
la réutilisation des polices ;
les exports HTML/PDF/DOCX ;
le contrôle qualité.

Le fichier perfect_document_extractor.py va déjà dans cette direction : il définit POINTS_PER_INCH = 72.0, extrait la largeur/hauteur de page depuis page.rect, puis déclare dans la géométrie de page unit_source: "pt", unit_normalized: "pt" et coordinate_system: "pymupdf_page_points".

À l’inverse, ocr_server.py convertit actuellement beaucoup de choses vers les pixels : il calcule sx = img.width / pdf_page.rect.width, sy = img.height / pdf_page.rect.height, puis passe ces facteurs à native_pdf_extractor.extract_page(...), pendant que l’OCR retourne directement des bbox en pixels.

Donc oui : il faut corriger cette incohérence.

2. Règle d’architecture à adopter

La règle doit être simple :

Modèle documentaire interne  → points PDF
Rendu image / OCR / masque   → pixels
Conversion                   → seulement aux frontières

Donc chaque objet extrait doit avoir en priorité :

bbox: [x0, y0, x1, y1]      # en points
bbox_unit: "pt"

Et éventuellement, pour debug/rendu image :

bbox_px: [x0, y0, x1, y1]   # en pixels
bbox_px_dpi: 150

Mais il ne faut plus que bbox signifie parfois pixels, parfois points. bbox doit être canonique.

3. Contrat recommandé pour toutes les unités

Chaque niveau doit suivre le même contrat :

{
    "id": "...",
    "level": "block|line|phrase|span|word|char|image|drawing|table|region",
    "bbox": [x0, y0, x1, y1],
    "bbox_unit": "pt",
    "bbox_origin": "top_left",
    "bbox_source": "native_pdf|ocr|layout_ai|postprocess",
    "bbox_confidence": 0.0,
    "bbox_px": [x0, y0, x1, y1],
    "bbox_px_dpi": 150,
}

Pour la page :

{
    "dimensions": {
        "width": 595.276,
        "height": 841.89,
        "unit": "pt",
        "origin": "top_left",
        "render_dpi": 150,
        "render_width_px": 1240,
        "render_height_px": 1754,
        "scale_x_px_per_pt": 2.0833,
        "scale_y_px_per_pt": 2.0833
    }
}

Ce contrat évite les ambiguïtés.

4. Fonctions de conversion à ajouter

Dans ocr_server.py, ajoute un bloc utilitaire centralisé :

POINTS_PER_INCH = 72.0
CANONICAL_UNIT = "pt"

def _scale_from_pdf_page(img, pdf_page):
    if not pdf_page:
        return 1.0, 1.0
    sx = float(img.width) / max(1.0, float(pdf_page.rect.width))
    sy = float(img.height) / max(1.0, float(pdf_page.rect.height))
    return sx, sy


def _bbox_px_to_pt(bbox, sx, sy):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        return [
            round(x0 / max(1e-9, sx), 3),
            round(y0 / max(1e-9, sy), 3),
            round(x1 / max(1e-9, sx), 3),
            round(y1 / max(1e-9, sy), 3),
        ]
    except Exception:
        return None


def _bbox_pt_to_px(bbox, sx, sy):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        return [
            round(x0 * sx, 3),
            round(y0 * sy, 3),
            round(x1 * sx, 3),
            round(y1 * sy, 3),
        ]
    except Exception:
        return None


def _annotate_bbox_units(node, *, bbox_pt=None, bbox_px=None, sx=1.0, sy=1.0, source=None):
    if not isinstance(node, dict):
        return node

    if bbox_pt is None and bbox_px is not None:
        bbox_pt = _bbox_px_to_pt(bbox_px, sx, sy)

    if bbox_px is None and bbox_pt is not None:
        bbox_px = _bbox_pt_to_px(bbox_pt, sx, sy)

    if bbox_pt is not None:
        node["bbox"] = bbox_pt
        node["bbox_unit"] = "pt"
        node["bbox_origin"] = "top_left"

    if bbox_px is not None:
        node["bbox_px"] = bbox_px
        node["bbox_px_dpi"] = TARGET_DPI

    if source:
        node["bbox_source"] = source

    return node

Ensuite, tous les modules doivent utiliser ces fonctions, pas refaire chacun leur propre conversion.

5. Corriger process_page

Actuellement, dans process_page, tu as :

sx = img.width / pdf_page.rect.width if pdf_page else 1.0
sy = img.height / pdf_page.rect.height if pdf_page else 1.0

puis :

native = native_pdf_extractor.extract_page(pdf_page, sx=sx, sy=sy)

et l’OCR produit des bbox en pixels.

Il faut changer la logique ainsi :

sx, sy = _scale_from_pdf_page(img, pdf_page)

page_width_pt = float(pdf_page.rect.width) if pdf_page else round(img.width / sx, 3)
page_height_pt = float(pdf_page.rect.height) if pdf_page else round(img.height / sy, 3)

Puis pour l’OCR :

bbox_px = [
    float(min([p[0] for p in b])),
    float(min([p[1] for p in b])),
    float(max([p[0] for p in b])),
    float(max([p[1] for p in b])),
]

bbox_pt = _bbox_px_to_pt(bbox_px, sx, sy)

raw_ocr.append({
    "label": txt,
    "bbox": bbox_pt,
    "bbox_unit": "pt",
    "bbox_px": bbox_px,
    "bbox_px_dpi": TARGET_DPI,
    "score": float(s),
})

Là, le parser.parse(raw_ocr, img) recevra déjà des coordonnées en points. Mais attention : si DocumentParser suppose que les coordonnées sont en pixels, il faudra lui transmettre aussi le contexte :

ocr_structure = parser.parse(
    raw_ocr,
    img,
    coordinate_unit="pt",
    page_width=page_width_pt,
    page_height=page_height_pt,
    render_width_px=img.width,
    render_height_px=img.height,
    scale_x=sx,
    scale_y=sy,
)

Si tu ne veux pas modifier DocumentParser tout de suite, fais une phase intermédiaire :

parser travaille encore en pixels
puis juste après parser.parse(...)
on convertit toute la structure en points

C’est moins propre, mais plus sûr.

6. Créer une fonction récursive de normalisation

Pour éviter de modifier tout d’un coup, ajoute une fonction qui traverse les blocs/lignes/phrases/spans/mots :

def _normalize_structure_bboxes_to_pt(node, sx, sy):
    if isinstance(node, dict):
        bbox = node.get("bbox")

        # Si l'objet est encore en pixels
        unit = node.get("bbox_unit") or node.get("bbox_source_unit")
        if bbox and unit not in {"pt", "pdf_pt"}:
            bbox_px = list(bbox)
            bbox_pt = _bbox_px_to_pt(bbox_px, sx, sy)

            node["bbox"] = bbox_pt
            node["bbox_unit"] = "pt"
            node["bbox_origin"] = "top_left"
            node["bbox_px"] = bbox_px
            node["bbox_px_dpi"] = TARGET_DPI

        elif bbox and unit in {"pt", "pdf_pt"}:
            node["bbox_unit"] = "pt"
            node["bbox_origin"] = "top_left"
            node.setdefault("bbox_px", _bbox_pt_to_px(bbox, sx, sy))
            node.setdefault("bbox_px_dpi", TARGET_DPI)

        for value in node.values():
            _normalize_structure_bboxes_to_pt(value, sx, sy)

    elif isinstance(node, list):
        for item in node:
            _normalize_structure_bboxes_to_pt(item, sx, sy)

    return node

Puis dans process_page, après :

final_blocks = _postprocess_blocks(final_blocks, img.width, img.height)

ajouter :

final_blocks = _normalize_structure_bboxes_to_pt(final_blocks, sx, sy)

Mais attention : si _postprocess_blocks, _annotate_layout, _extract_immutable_overlays, _collect_text_regions_for_inpainting attendent encore des pixels, il faut retarder la conversion.

Le meilleur compromis est donc :

A. pipeline legacy interne encore en pixels
B. juste avant page_structure final : conversion en points
C. pour les fonctions image/inpainting : utiliser bbox_px
7. Adapter les fonctions qui manipulent l’image

Certaines fonctions doivent rester en pixels :

_extract_immutable_overlays(...)
_collect_text_regions_for_inpainting(...)
text_removal_strategy.remove(...)
_erase_uncovered_pdf_words(...)
visual debug bbox

_erase_uncovered_pdf_words convertit déjà les mots PDF de points vers pixels avec sx et sy. C’est exactement ce modèle qu’il faut généraliser : les opérations image travaillent en pixels, mais seulement à partir d’une conversion explicite.

Il faut donc ajouter une fonction :

def _regions_for_image_processing(blocks, sx, sy):
    regions = []

    def visit(node):
        if isinstance(node, dict):
            bbox_px = node.get("bbox_px")
            if not bbox_px and node.get("bbox_unit") == "pt":
                bbox_px = _bbox_pt_to_px(node.get("bbox"), sx, sy)
            if bbox_px:
                regions.append(bbox_px)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(blocks)
    return regions

Puis pour l’inpainting :

text_regions_px = _regions_for_image_processing(final_blocks, sx, sy)

clean_bgr, mask, text_removal_debug = text_removal_strategy.remove(
    img,
    text_regions_px,
    mode=text_removal_mode
)

Ne donne jamais à TextRemovalStrategy des coordonnées en points. Il travaille sur image, donc pixels.

8. Corriger page_structure["dimensions"]

Actuellement, la structure active utilise surtout les dimensions image. Il faut remplacer par un modèle double :

page_structure["dimensions"] = {
    "width": round(page_width_pt, 3),
    "height": round(page_height_pt, 3),
    "unit": "pt",
    "origin": "top_left",

    "render_width_px": img.width,
    "render_height_px": img.height,
    "render_dpi": TARGET_DPI,
    "scale_x_px_per_pt": round(sx, 6),
    "scale_y_px_per_pt": round(sy, 6),
}

Ne fais plus :

"width": img.width
"height": img.height

comme dimensions principales du document.

Les pixels doivent être secondaires.

9. Harmoniser avec perfect_document_extractor.py

C’est important : perfect_document_extractor.py est déjà plus propre sur ce point. Il extrait la page en points, conserve unit_source: "pt", unit_normalized: "pt" et coordinate_system: "pymupdf_page_points".

Son OCR convertit aussi les résultats OCR image vers les coordonnées page en divisant implicitement par l’échelle : il calcule sx = page_w / img_w, sy = page_h / img_h, puis applique ces facteurs aux points OCR pour obtenir des bbox dans le référentiel page.

Donc la direction correcte est :

faire évoluer ocr_server.py vers la logique de perfect_document_extractor.py

pas l’inverse.

10. Corriger perfect_extraction_to_reconstruction.py

L’adapter actuel reconvertit le modèle perfect vers les pixels historiques :

POINTS_PER_INCH = 72.0
RECON_DPI = 150.0
PT_TO_RECON_PX = RECON_DPI / POINTS_PER_INCH

et sa fonction _bbox_pt_to_px(...) convertit les bbox en pixels.

Il déclare même :

"coordinates": "historical_reconstructor_pixels_150dpi"

et les unités finales comme :

"bbox_source_unit": "px_150dpi"

Ça confirme que ton ancien reconstructeur impose encore les pixels. Il faut donc prévoir deux sorties :

perfect_model_pt          → modèle canonique en points
reconstructor_payload_px  → payload temporaire pour l’ancien reconstructeur

Ne supprime pas la conversion vers pixels tant que DocumentReconstructor n’est pas refactoré. Mais renomme clairement les champs.

Dans l’adapter, au lieu de perdre l’information point, chaque unité doit avoir :

"bbox": bbox_pt,
"bbox_unit": "pt",
"bbox_px": bbox_px,
"bbox_px_dpi": 150,
"reconstructor_bbox": bbox_px,
"reconstructor_unit": "px_150dpi"

L’actuel dimensions de l’adapter garde déjà width_pt, height_pt, source_unit: "pt" et reconstructor_unit: "px_150dpi", mais il met encore width/height en pixels. Il faut inverser la priorité :

"dimensions": {
    "width": round(width_pt, 3),
    "height": round(height_pt, 3),
    "unit": "pt",

    "width_pt": round(width_pt, 3),
    "height_pt": round(height_pt, 3),

    "render_width_px": round(width_px, 3),
    "render_height_px": round(height_px, 3),
    "render_dpi": RECON_DPI,

    "source_unit": "pt",
    "reconstructor_unit": "px_150dpi",
}
11. Normaliser aussi les styles

Les polices doivent être en points aussi.

Dans perfect_document_extractor.py, les styles utilisent déjà :

"size_pt": ...

Dans l’ancien pipeline, tu as souvent :

style["size"]

sans savoir si c’est pixel, point, ou taille estimée OCR.

Il faut imposer :

style = {
    "font_family": "...",
    "font_size_pt": 12.0,
    "font_size_px": 25.0,
    "font_size_unit": "pt",
}

Règle :

font_size_pt = font_size_px / sx

si le style vient d’une image OCR.

Pour un PDF natif, conserver directement la taille PDF en points.

12. Normaliser aussi element_relations.py

element_relations.py calcule des signaux comme :

inline_gap_px
vertical_gap_px
indent_delta_px
block_left_px
block_right_px

Ce n’est pas idéal. Le module parle actuellement en pixels dans ses noms de signaux.

Il faut remplacer ou compléter par :

inline_gap_pt
vertical_gap_pt
indent_delta_pt
block_left_pt
block_right_pt

Pour compatibilité, tu peux garder temporairement les deux :

"signals": {
    "inline_gap_pt": ...,
    "vertical_gap_pt": ...,
    "indent_delta_pt": ...,

    "inline_gap_px": ...,       # legacy/debug seulement
    "vertical_gap_px": ...,
    "indent_delta_px": ...,

    "unit": "pt",
}

Mais la logique principale doit utiliser les points.

13. Stratégie de migration sûre

Ne convertis pas tout brutalement. Fais en 4 étapes.

Étape 1 — Ajouter les métadonnées d’unités

Sans changer la logique :

bbox_unit = "px"
bbox_px = bbox

Puis pour les PDF :

bbox_pt = _bbox_px_to_pt(bbox_px, sx, sy)
Étape 2 — Sortie canonique en points

Dans page_structure, faire :

bbox = bbox_pt
bbox_unit = "pt"
bbox_px = ancienne_bbox
Étape 3 — Adapter les modules image

Tout ce qui touche à l’image utilise explicitement :

bbox_px

Pas bbox.

Étape 4 — Adapter le reconstructeur

À terme, DocumentReconstructor devrait accepter :

coordinates="pt"

et convertir lui-même en pixels seulement pour les opérations raster.

14. Schéma final recommandé
PDF natif
  ↓
PyMuPDF extraction
  ↓
bbox en points
  ↓
OCR image si nécessaire
  ↓
bbox OCR pixels → conversion immédiate en points
  ↓
Document model canonique en points
  ↓
semantic correction / relations / classification en points
  ↓
traduction en points
  ↓
reconstruction PDF directe en points
  ↓
rendu image/debug seulement en pixels
15. Décision technique nette

Oui, il faut normaliser les unités d’extraction en points.

Le contrat à imposer est :

bbox = points
bbox_unit = "pt"
bbox_px = optionnel pour debug/rendu

Et la règle opérationnelle :

aucun module de compréhension, classification, relation, traduction ou layout ne doit dépendre des pixels.
seuls OCR brut, masques, background, inpainting et visualisation travaillent en pixels.

C’est cohérent avec perfect_document_extractor.py, qui est déjà orienté points, et cela corrigera l’un des défauts structurels du pipeline actuel : sa dépendance au DPI de rendu.

page_case_classifier_v2.py
Python
page_extraction_postprocessors.py
Python
page_family_registry.py
Python
page_policy_matrix.py
Python
page_profile_registry.py
Python
special_region_detector.py
Python
structure_extractor.py
Python
style_profiler.py
Python
style_tone_classifier.py
Python
terminology_manager.py
Python
Il y a aussi les fichiers de détection/compréhension/classification de pages, zones, régions etc ... Qui doivent aussi être très importante dans le pipeline au debut (avant ou après extractions des données et caractéristiques). Ces éléments et leurs informations doivent aussi peser fortement dans la suite du pipeline. Comment donc les améliorer ? et les brancher intelligements ? Comment intégrer les données  aux autres données du pipeline au niveau extraction ?
Thought for 25s

Oui. Ces fichiers doivent devenir le système de pilotage du pipeline, pas de simples post-traitements décoratifs.

Actuellement, une partie est déjà branchée, mais pas encore de manière assez stratégique. ocr_server.py construit page_structure, appelle layout_v2_builder.build(page_structure), puis layout_ai_enricher, puis apply_page_extraction_postprocessors, puis reconstruit à nouveau layout_v2 si la structure change. C’est une bonne base, mais il faut transformer cette logique en boucle d’intelligence de page.

L’idée centrale :

La page doit être comprise avant, pendant et après l’extraction.

Pas seulement après.

1. Le problème actuel

Le pipeline actif fait plutôt ceci :

OCR / extraction native
→ blocs / lignes / phrases
→ classification page
→ post-traitements
→ contrats de traduction
→ reconstruction

Mais pour des documents complexes, il faut plutôt :

pré-analyse page / zones / régions
→ extraction guidée par les régions
→ classification page enrichie
→ correction structurelle
→ classification des unités
→ traduction/reconstruction pilotée par les contraintes

Pourquoi ? Parce que la page globale doit influencer les unités locales.

Exemple :

Même fragment court : "Error"

peut être :

un mot normal dans un paragraphe ;
un label de diagramme ;
un axe de graphique ;
une cellule de tableau ;
une légende ;
un terme technique à préserver.

La décision ne doit pas dépendre uniquement du texte "Error". Elle doit dépendre de :

page_family
layout_type
region_type
zone_type
nearby_visual_object
table/chart/formula membership
source_kind
style
position
reading_mode
translation_sensitivity
2. Les fichiers importants et leur rôle réel
PageCaseClassifierV2

Ce fichier produit déjà des signaux très utiles :

layout_tendencies
reading_modes
page_archetype_signals
translation_sensitivity_signals
risk_flags
legacy_bridge
feature_snapshot

Il ne donne donc pas seulement une étiquette de page. Il donne une carte de risques et de modes de lecture : flux linéaire, colonnes, TOC, grille tabulaire, overlay ancré, glossaire, etc.

Il faut donc utiliser sa sortie comme contexte de décision pour tout le reste.

LayoutV2Builder

C’est déjà le bon endroit pour centraliser cette intelligence. Il :

calcule les marges ;
infère les colonnes ;
détecte page_role ;
classe page_family, document_type, layout_type, style_profile ;
ajoute page_case ;
ajoute page_case_v2 ;
construit les TOC ;
enrichit la géométrie relative ;
enrichit les relations entre éléments ;
ajoute les politiques de positionnement ;
construit les descripteurs de layout.

Donc LayoutV2Builder doit devenir le Page Understanding Hub.

page_extraction_postprocessors.py

Ce fichier est déjà piloté par :

layout_type
page_family
document_type

Il applique des post-traitements différents pour :

pages tabulaires ;
pages annotées ;
pages de graphiques ;
formulaires ;
factures ;
reçus ;
diagrammes ;
chart labels.

Il calcule aussi des métriques de fragmentation, puis applique des regroupements adaptés.

Ce fichier ne doit donc pas être appelé une seule fois comme rustine. Il doit faire partie d’une phase de consolidation structurelle.

PagePolicyMatrix

C’est le décideur des politiques d’unité :

unit_type
translatable
translation_strategy
coverage_required
render_policy

Il tient déjà compte de :

page_role
page_family
page_family_group
document_type
layout_type
style_profile
source_kind
role

et il distingue notamment code, formule, citation, lien, label court, label de diagramme, corps narratif, marqueur de liste, TOC, etc.

C’est exactement ce qui doit transformer la compréhension globale de page en décisions locales.

special_region_detector.py

Ce fichier est critique. Il détecte les régions qui ne doivent pas être traitées comme du simple texte : formules, code, zones techniques, régions à préserver, sous-régions à ne pas traduire, etc. Dans perfect_document_extractor.py, il est déjà appelé via detect_special_regions(...), puis les régions sont normalisées, filtrées, validées et fusionnées.

Il faut le brancher aussi dans le pipeline actif de ocr_server.py.

style_profiler.py

Il harmonise les styles, détecte des classes comme body, heading, caption, header, footer, et propage un style résolu aux blocs/spans. C’est important pour la reconstruction, mais aussi pour classifier les rôles.

page_family_registry.py

Ce fichier donne une configuration par famille de page : groupe, style de traduction, ton de traduction, description. Par exemple body_with_figure, body_with_diagram, chart_label_page, table_page, mixed_formula_annotation_page, etc.

Il doit devenir une table de configuration, pas seulement une aide indirecte.

3. Architecture cible : ajouter une couche page_intelligence

Il faut créer un objet unique dans page_structure :

page_structure["page_intelligence"] = {
    "schema_version": "page_intelligence.v1",
    "coordinate_unit": "pt",
    "page_profile": {},
    "page_case": {},
    "page_case_v2": {},
    "regions": [],
    "zones": [],
    "special_regions": [],
    "native_structure": {},
    "reading_modes": {},
    "risk_flags": [],
    "translation_sensitivity": {},
    "extraction_guidance": {},
    "unit_policy_context": {},
}

Ce champ devient la mémoire de compréhension de la page.

Ensuite, les autres blocs héritent de cette intelligence.

Chaque bloc devrait recevoir :

block["page_context"] = {
    "page_family": page_structure["page_family"],
    "layout_type": page_structure["layout_type"],
    "document_type": page_structure["document_type"],
    "style_profile": page_structure["style_profile"],
    "reading_mode": "...",
    "region_memberships": [...],
    "zone_memberships": [...],
    "risk_flags": [...],
}

Chaque phrase ou span peut recevoir en plus :

phrase["region_memberships"] = [...]
phrase["translation_policy"] = {...}
phrase["render_policy"] = "anchored_text|paragraph_flow|fixed_preserve|background_only"
4. Ordre intelligent du pipeline

Il faut séparer le pipeline en trois moments.

Moment A — pré-compréhension avant extraction complète

Objectif : comprendre grossièrement la page avant d’extraire finement.

Entrées :

image rendue
dimensions page
PDF natif si disponible
images PDF
drawings PDF
zones non textuelles
densité visuelle
premiers mots OCR ou natifs rapides

À ce moment, on ne cherche pas encore la phrase parfaite. On cherche :

est-ce une page TOC ?
une page tableau ?
une page schéma ?
une page formulaire ?
une page graphique ?
une page dense deux colonnes ?
une page avec formules/code ?

Créer :

def _build_early_page_intelligence(
    img,
    pdf_page=None,
    native_probe=None,
    native_images=None,
    native_drawings=None,
    sx=1.0,
    sy=1.0,
):
    ...

Sortie :

early_page_intelligence = {
    "stage": "early",
    "page_geometry": {...},
    "visual_complexity": {...},
    "non_text_density": ...,
    "provisional_page_role": ...,
    "provisional_layout_type": ...,
    "provisional_page_family": ...,
    "special_region_candidates": [],
    "extraction_guidance": {
        "prefer_native_text": True,
        "force_ocr_regions": [],
        "avoid_ocr_regions": [],
        "preserve_visual_regions": [],
        "likely_table_regions": [],
        "likely_formula_regions": [],
        "likely_chart_regions": [],
    }
}

Cette sortie doit être disponible avant :

native_pdf_extractor.extract_page(...)
engine_ocr(...)
parser.parse(...)
Moment B — extraction guidée

Ici, les régions détectées au début doivent influencer l’extraction.

Exemples :

Si une région est formula

Alors :

ne pas traduire comme prose
ne pas fusionner dans un paragraphe
préserver visuellement
extraire comme objet spécial
Si une région est table

Alors :

éviter de fusionner les cellules comme un paragraphe
garder les alignements
créer row_groups / column_groups / cell_ids
Si une région est chart

Alors :

labels courts = labels de graphique
ticks numériques = préserver
légendes = ancrées
Si une région est TOC

Alors :

extraire label + page marker
ne pas traduire les numéros
conserver tab stops

Donc il faut appliquer les régions à chaque unité.

Créer :

def _attach_region_memberships(page_structure):
    regions = (
        page_structure.get("special_regions", [])
        + page_structure.get("regions", [])
        + page_structure.get("zones", [])
        + list((page_structure.get("native_structure") or {}).values())
    )

    for block in page_structure.get("blocks", []):
        block["region_memberships"] = _find_overlapping_regions(block, regions)

        for line in block.get("lines", []):
            line["region_memberships"] = _find_overlapping_regions(line, regions)

            for phrase in line.get("phrases", []):
                phrase["region_memberships"] = _find_overlapping_regions(phrase, regions)

                for span in phrase.get("spans", []):
                    span["region_memberships"] = _find_overlapping_regions(span, regions)

La sortie de classification devient alors disponible au niveau le plus fin.

Moment C — compréhension finale après extraction

Après extraction et regroupement, on exécute :

LayoutV2Builder
PageCaseClassifierV2
special_region_detector final
postprocessors
PagePolicyMatrix
semantic relations
translation contracts

Le pipeline devient :

1. lecture source
2. pré-analyse page
3. extraction native/OCR guidée
4. structuration blocs/lignes/phrases
5. style profiling
6. special region detection finale
7. LayoutV2Builder
8. postprocessors
9. LayoutV2Builder à nouveau
10. region memberships
11. PagePolicyMatrix
12. semantic spans/runs/groups
13. traduction/reconstruction
5. Où brancher concrètement dans ocr_server.py

Dans process_page, le bloc actuel construit page_structure, puis appelle :

page_structure = layout_v2_builder.build(page_structure)
...
page_structure, postprocess_info = apply_page_extraction_postprocessors(page_structure)
...
page_structure = layout_v2_builder.build(page_structure)
...
_annotate_translation_contracts(...)

Il faut le remplacer par une orchestration plus claire :

page_structure = _build_initial_page_structure(...)

page_structure = _run_page_understanding_pass(
    page_structure,
    stage="pre_postprocess",
    img=img,
    pdf_page=pdf_page,
    sx=sx,
    sy=sy,
)

page_structure, postprocess_info = apply_page_extraction_postprocessors(page_structure)

if postprocess_info.get("changed") or postprocess_info.get("applied"):
    page_structure = _run_page_understanding_pass(
        page_structure,
        stage="post_postprocess",
        img=img,
        pdf_page=pdf_page,
        sx=sx,
        sy=sy,
    )

page_structure = _attach_region_memberships(page_structure)
page_structure = _propagate_page_intelligence_to_units(page_structure)

_annotate_translation_contracts(
    page_structure.get("blocks", []),
    page_context=page_structure,
)
6. Fonction centrale recommandée

Créer une fonction unique :

def _run_page_understanding_pass(
    page_structure: dict,
    *,
    stage: str,
    img=None,
    pdf_page=None,
    sx=1.0,
    sy=1.0,
):
    page_structure.setdefault("page_intelligence", {})
    page_structure["page_intelligence"].setdefault("passes", [])

    # 1. LayoutV2 / page case / page profile
    page_structure = layout_v2_builder.build(page_structure)

    # 2. Special regions
    page_structure = _apply_special_region_detection(
        page_structure,
        img=img,
        pdf_page=pdf_page,
        sx=sx,
        sy=sy,
    )

    # 3. Region memberships
    page_structure = _attach_region_memberships(page_structure)

    # 4. Consolidated intelligence
    page_structure = _consolidate_page_intelligence(page_structure, stage=stage)

    page_structure["page_intelligence"]["passes"].append({
        "stage": stage,
        "page_role": page_structure.get("page_role"),
        "page_family": page_structure.get("page_family"),
        "layout_type": page_structure.get("layout_type"),
        "document_type": page_structure.get("document_type"),
        "style_profile": page_structure.get("style_profile"),
        "special_region_count": len(page_structure.get("special_regions") or []),
        "region_count": len(page_structure.get("regions") or []),
    })

    return page_structure
7. Brancher special_region_detector.py dans le pipeline actif

Le mode “perfect” l’utilise déjà indirectement, mais le pipeline actif doit l’utiliser aussi. Dans PerfectDocumentExtractor, _detect_special_regions(...) appelle detect_special_regions(...), normalise les régions retournées, ajoute les régions code, filtre les fausses formules, puis fusionne les régions spéciales.

Il faut reproduire cette logique dans ocr_server.py.

Exemple :

def _apply_special_region_detection(page_structure, *, img=None, pdf_page=None, sx=1.0, sy=1.0):
    try:
        from special_region_detector import detect_special_regions
    except Exception as exc:
        page_structure.setdefault("page_intelligence", {})
        page_structure["page_intelligence"]["special_region_detector"] = {
            "available": False,
            "error": str(exc),
        }
        return page_structure

    try:
        enriched, report = detect_special_regions(
            page_structure,
            page_image=img,
            pdf_page=pdf_page,
            sx=sx,
            sy=sy,
        )
    except Exception as exc:
        page_structure.setdefault("page_intelligence", {})
        page_structure["page_intelligence"]["special_region_detector"] = {
            "available": True,
            "applied": False,
            "error": str(exc),
        }
        return page_structure

    special_regions = enriched.get("special_regions") or []
    page_structure["special_regions"] = special_regions
    page_structure.setdefault("layout", {})
    page_structure["layout"]["special_regions"] = special_regions

    page_structure.setdefault("page_intelligence", {})
    page_structure["page_intelligence"]["special_region_detector"] = {
        "available": True,
        "applied": True,
        "report": report,
        "special_region_count": len(special_regions),
    }

    return page_structure

Important : si tu normalises le pipeline en points, alors special_region_detector doit retourner :

{
    "bbox": [...],       # points
    "bbox_unit": "pt",
    "bbox_px": [...],    # pixels uniquement pour masque/debug
}
8. Faire peser les régions dans PagePolicyMatrix

Aujourd’hui, PagePolicyMatrix reçoit surtout :

text
role
source_kind
page_role
page_family
page_family_group
document_type
layout_type
style_profile
fallback_policy

Il faut lui transmettre aussi :

region_memberships
zone_memberships
structure_hints
page_case_v2
reading_modes
risk_flags

Modifier l’appel :

policy = page_policy_matrix.classify_unit_policy(
    text=text,
    role=role,
    source_kind=source_kind,
    page_role=page_role,
    page_family=page_family,
    page_family_group=page_family_group,
    document_type=document_type,
    layout_type=layout_type,
    style_profile=style_profile,
    fallback_policy=fallback_policy,
    region_memberships=unit.get("region_memberships"),
    structure_hints=unit.get("structure_hints"),
    page_case_v2=page_context.get("page_case_v2"),
    page_intelligence=page_context.get("page_intelligence"),
)

Puis dans PagePolicyMatrix, ajouter en priorité haute :

if _has_region(region_memberships, "formula"):
    return {
        "unit_type": "formula",
        "translatable": False,
        "translation_strategy": "exact_preserve",
        "coverage_required": "strict",
        "render_policy": "fixed_preserve",
    }

if _has_region(region_memberships, "code"):
    return {
        "unit_type": "code_visible",
        "translatable": False,
        "translation_strategy": "exact_preserve",
        "coverage_required": "strict",
        "render_policy": "anchored_text",
    }

if _has_region(region_memberships, "table_cell"):
    return {
        "unit_type": "table_cell_text",
        "translatable": True,
        "translation_strategy": "layout_constrained",
        "coverage_required": "strict",
        "render_policy": "anchored_text",
    }

if _has_region(region_memberships, "chart_tick"):
    return {
        "unit_type": "chart_tick_label",
        "translatable": False,
        "translation_strategy": "exact_preserve",
        "coverage_required": "strict",
        "render_policy": "fixed_preserve",
    }

Ainsi, la région devient plus forte que le simple texte.

9. Améliorer les postprocessors

page_extraction_postprocessors.py fait déjà beaucoup : métrique de fragmentation, regroupement des fragments de tableau, regroupement des pages annotées, structure de graphique, structure native table/annotation/chart, etc.

Mais il faut améliorer trois points.

9.1 Ne pas seulement modifier les blocs : produire un rapport décisionnel

Chaque fusion ou regroupement doit produire :

{
    "operation": "merge_blocks",
    "reason": "table_row_fragment",
    "source_block_ids": [...],
    "target_block_id": "...",
    "confidence": 0.82,
    "signals": {...}
}

Ajouter :

page_data["postprocess_decisions"] = decisions

Pourquoi ? Pour déboguer les erreurs de regroupement.

9.2 Garder les anciennes unités

Avant une fusion :

block["merged_from"] = [...]
block["pre_merge_bboxes"] = [...]

Ne jamais perdre les anciennes coordonnées.

9.3 Utiliser page_case_v2.risk_flags

Si risk_flags contient :

toc_row_fragmentation
grid_alignment_loss
anchor_attachment_drift
translation_overflow
key_value_pair_breakage

alors les postprocessors doivent adapter leur agressivité.

Par exemple :

if has_risk("grid_alignment_loss"):
    avoid_paragraph_merging = True
    preserve_cell_boundaries = True
10. Utiliser page_case_v2 comme poids de décision

PageCaseClassifierV2 produit des reading_modes :

linear_flow
columnar_flow
anchored_overlay_flow
toc_row_flow
tabular_grid_flow
glossary_pair_flow

Ces valeurs doivent peser sur les décisions.

Exemple :

reading_modes = page_structure["page_case_v2"]["reading_modes"]

if reading_modes["tabular_grid_flow"] > 0.65:
    page_structure["extraction_guidance"]["preserve_grid"] = True
    page_structure["translation_mode"] = "cell_constrained"

if reading_modes["anchored_overlay_flow"] > 0.55:
    page_structure["extraction_guidance"]["preserve_anchors"] = True
    page_structure["translation_mode"] = "anchored_labels"

if reading_modes["linear_flow"] > 0.65:
    page_structure["translation_mode"] = "paragraph_reflow"

if reading_modes["toc_row_flow"] > 0.80:
    page_structure["translation_mode"] = "toc_row_layout"

C’est ce que j’appelle faire peser l’intelligence de page.

11. Créer un graphe d’appartenance

Il ne faut pas seulement stocker les régions en liste. Il faut créer un graphe :

page_structure["layout_graph"] = {
    "nodes": [
        {"id": "page", "type": "page"},
        {"id": "region_1", "type": "table"},
        {"id": "block_3", "type": "block"},
        {"id": "phrase_3_1", "type": "phrase"},
    ],
    "edges": [
        {"source": "page", "target": "region_1", "relation": "contains"},
        {"source": "region_1", "target": "block_3", "relation": "contains"},
        {"source": "block_3", "target": "phrase_3_1", "relation": "contains"},
        {"source": "label_1", "target": "chart_1", "relation": "annotates"},
    ]
}

Ce graphe sert ensuite à :

la traduction ;
la reconstruction ;
la validation ;
la visualisation debug ;
les corrections manuelles futures.
12. Intégration aux données d’extraction

Chaque unité doit avoir trois couches.

Couche 1 — extraction brute
{
    "text": "...",
    "bbox": [...],
    "style": {...},
    "source": "native_pdf|ocr",
    "confidence": ...
}
Couche 2 — compréhension
{
    "object_type": "text_block|formula|code|table_cell|chart_label|caption",
    "object_class": "editorial|technical|visual|tabular",
    "role": "body|section_heading|figure_caption|axis_label",
    "region_memberships": [...],
    "structure_hints": {...},
    "page_context": {...}
}
Couche 3 — politique de traitement
{
    "translatable": True,
    "translation_strategy": "paragraph_flow|layout_constrained|exact_preserve",
    "render_policy": "paragraph_flow|anchored_text|fixed_preserve|background_only",
    "coverage_required": "strict|normal|optional",
    "reconstruction_constraints": {...}
}

L’erreur actuelle dans beaucoup de pipelines WYSIWYG est de mélanger ces trois couches. Il faut les séparer.

13. Schéma de données recommandé

Ajouter à chaque bloc :

block.update({
    "extraction": {
        "source": block.get("source"),
        "source_kind": block.get("source_kind"),
        "confidence": block.get("ocr_confidence_mean") or block.get("confidence"),
    },
    "understanding": {
        "role": block.get("role"),
        "object_type": block.get("object_type"),
        "object_class": block.get("object_class"),
        "page_family": page_structure.get("page_family"),
        "layout_type": page_structure.get("layout_type"),
        "region_memberships": block.get("region_memberships", []),
        "structure_hints": block.get("structure_hints", {}),
    },
    "policy": {
        "translatable": block.get("translatable"),
        "translation_strategy": block.get("translation_strategy"),
        "render_policy": block.get("render_policy"),
        "coverage_required": block.get("coverage_required"),
    }
})

Cela permet à tous les modules suivants de lire proprement les décisions.

14. Correction importante sur l’intégration actuelle

Dans une réponse précédente, j’ai dit que element_relations.py n’était pas intégré directement à ocr_server.py. C’est vrai au niveau des imports directs de ocr_server.py, mais le fichier est en réalité déjà appelé indirectement par LayoutV2Builder dans structure_extractor.py, avec aussi element_relations_ai, positioning_policy, element_rulesets, relative_geometry et les descripteurs de layout.

Donc la bonne correction est :

element_relations n’est pas importé directement par ocr_server.py,
mais il est déjà branché via LayoutV2Builder.

C’est une bonne chose. Il faut maintenant faire pareil pour special_region_detector, page_case_v2, page_policy_matrix, style_profiler, terminology_manager et les régions : les relier dans une couche commune.

15. Ordre final recommandé

Voici l’ordre que je recommande pour ton pipeline :

A. Préparation
   - ouvrir PDF/image
   - calculer dimensions en points
   - rendre image seulement pour OCR/debug

B. Pré-analyse page
   - images natives
   - drawings natifs
   - zones non textuelles
   - special_region candidates
   - page profile provisoire

C. Extraction
   - native PDF extraction
   - OCR si nécessaire
   - DocumentParser
   - style extraction

D. Fusion
   - dédoublonnage natif/OCR
   - blocs/lignes/phrases/spans
   - normalisation bbox en points

E. Compréhension page
   - LayoutV2Builder
   - PageCaseClassifier
   - PageCaseClassifierV2
   - page_family_registry
   - layout descriptors

F. Détection régions finale
   - special_region_detector
   - table/chart/formula/code regions
   - region memberships

G. Consolidation
   - page_extraction_postprocessors
   - rebuild LayoutV2Builder
   - attach region memberships
   - layout graph

H. Politiques
   - PagePolicyMatrix
   - translation contracts
   - render policies
   - reconstruction constraints

I. Sémantique
   - semantic phrases
   - LLM semantic corrector
   - semantic spans/runs/groups
   - element relations

J. Traduction / reconstruction
   - style/tone/terminology
   - layout constrained translation
   - reconstruction WYSIWYG
16. Priorité de développement

Ordre pratique :

1. Créer page_intelligence dans page_structure.
2. Normaliser toutes les bboxes en points + garder bbox_px pour image/debug.
3. Brancher special_region_detector dans ocr_server.py.
4. Ajouter region_memberships aux blocs/lignes/phrases/spans.
5. Faire lire region_memberships par PagePolicyMatrix.
6. Faire produire postprocess_decisions par page_extraction_postprocessors.
7. Faire de LayoutV2Builder le hub officiel de page understanding.
8. Ajouter layout_graph.
9. Intégrer terminology/style_tone seulement après stabilisation des unités.

La décision la plus importante : les classifications de page, zones et régions doivent devenir des données d’entrée pour la traduction et la reconstruction, pas de simples informations de sortie. Elles doivent influencer les fusions, les politiques de traduction, les zones protégées, les modes de rendu, les contraintes d’overflow et les décisions de reflow.

A partir de tout cela, nous devrions donc construire une première tête d'unité pour notre pipeline WYSIWYG que nous appelerons INPUT_DATA. Son rôle est donc ''de disposer de toutes les données analytiques, visuelles, sémantiques, caratéristiques, de fonds, de formes, etc ..." sur la page, d'organiser toutes ces informations en une structure qui sera après consommé par le reste du pipeline (TRADUCTION, RECONSTRUCTION, MISE_EN_FORME_, etc ...)
Tu vois un peu ce que je veux faire ?

Oui. Tu es en train de définir le contrat canonique d’entrée du pipeline : une structure unique, riche et normalisée, produite par l’extraction, puis consommée sans ambiguïté par traduction, reconstruction et contrôle qualité.

Thought for 23s

Oui. Je vois très bien.

Tu veux créer une première tête canonique du pipeline, appelée INPUT_DATA, qui ne sera pas seulement “le résultat OCR”, mais le modèle complet de compréhension de la page.

Autrement dit :

INPUT_DATA = tout ce que le pipeline sait de la page,
avant traduction, reconstruction, mise en forme et contrôle qualité.

Ce sera le contrat d’entrée unique consommé par :

TRADUCTION
RECONSTRUCTION
MISE_EN_FORME
LAYOUT_OPTIMIZER
QA / COMPARAISON
EXPORT HTML / PDF / DOCX
1. Définition exacte de INPUT_DATA

INPUT_DATA doit être une structure normalisée qui contient :

1. la source
2. la page
3. les unités extraites
4. les styles
5. les régions
6. les relations
7. les classifications
8. les politiques de traduction/reconstruction
9. les fonds / images / masques
10. les contraintes WYSIWYG
11. la qualité / confiance / provenance

Donc INPUT_DATA n’est pas un simple JSON OCR. C’est une représentation documentaire complète.

2. Son rôle dans le pipeline

Son rôle est de transformer ceci :

PDF / image / Word / PPT

en ceci :

modèle analytique canonique de page

Puis le reste du pipeline n’a plus besoin de refaire l’analyse de base.

Le pipeline devient :

SOURCE
  ↓
EXTRACTION + ANALYSE + CLASSIFICATION
  ↓
INPUT_DATA
  ↓
TRADUCTION
  ↓
RECONSTRUCTION
  ↓
QA / EXPORT

INPUT_DATA est donc le point de stabilité du projet.

Même si plus tard tu changes l’OCR, le moteur PDF, le détecteur de régions ou le classificateur, le reste du pipeline continue à consommer le même contrat.

3. Pourquoi c’est nécessaire

Aujourd’hui, plusieurs modules produisent déjà des informations utiles, mais elles sont dispersées :

LayoutV2Builder ajoute page_role, page_family, document_type, layout_type, style_profile, page_case, page_case_v2, les relations d’éléments, la géométrie relative, les politiques de positionnement et les descripteurs de layout.
PageCaseClassifierV2 produit des signaux de lecture, de densité, d’archétype, de sensibilité à la traduction et de risques comme toc_row_fragmentation, grid_alignment_loss, anchor_attachment_drift, etc.
PagePolicyMatrix décide pour chaque unité si elle est traduisible, à préserver, à rendre comme texte ancré, comme flux paragraphe, comme élément fixe, etc.
special_region_detector.py détecte les formules, codes, régions techniques et sous-régions à préserver visuellement.
style_profiler.py construit un profil de style, harmonise les styles, détecte heading/body/caption/header/footer et propage les styles résolus aux spans.
page_family_registry.py associe chaque famille de page à un style et ton de traduction.

Le problème est que ces informations doivent être rassemblées, hiérarchisées et rendues consommables.

C’est précisément le rôle de INPUT_DATA.

4. Principe fondamental

INPUT_DATA doit être organisé en couches.

Pas un gros dictionnaire désordonné.

Structure recommandée :

INPUT_DATA
├── document
├── page
├── assets
├── extraction
├── units
├── regions
├── page_intelligence
├── style_system
├── semantic
├── relations
├── policies
├── reconstruction_constraints
├── translation_context
├── quality
└── provenance
5. Schéma général de INPUT_DATA

Voici une première forme solide :

INPUT_DATA = {
    "schema_version": "input_data.v1",
    "kind": "page_input_data",

    "document": {
        "document_id": "...",
        "source_path": "...",
        "file_name": "...",
        "file_type": "pdf|image|docx|pptx",
        "page_count": 0,
        "detected_document_type": "scientific_paper|report|form|mixed_unknown",
        "language": {
            "source_lang": "en",
            "target_lang": "fr",
            "detected_languages": [],
        },
    },

    "page": {
        "page_index": 0,
        "page_number": 1,
        "page_role": "body|toc|cover|references|unknown",

        "geometry": {
            "width": 595.276,
            "height": 841.89,
            "unit": "pt",
            "origin": "top_left",

            "render_width_px": 1240,
            "render_height_px": 1754,
            "render_dpi": 150,
            "scale_x_px_per_pt": 2.0833,
            "scale_y_px_per_pt": 2.0833,
        },

        "rotation": 0,
        "orientation": "portrait|landscape",
        "format_probable": "A4|Letter|custom",
    },

    "assets": {
        "source_image_path": "...",
        "source_image_url": "...",
        "background_path": "...",
        "mask_master_path": "...",
        "visual_debug_path": "...",

        "background": {
            "type": "cleaned_background|source_page_render",
            "text_removed": True,
            "inpainting_mode": "default|strict|none",
            "quality": {},
        },

        "immutable_overlays": [],
    },

    "extraction": {
        "pipeline": "legacy|enhanced|perfect|hybrid_compare",
        "source_priority": ["native_pdf", "ocr", "layout_ai", "heuristic"],
        "native_pdf_available": True,
        "ocr_used": True,
        "ocr_engine": "RapidOCR",
        "target_dpi": 150,

        "raw_sources": {
            "native_blocks": [],
            "ocr_blocks": [],
            "images": [],
            "drawings": [],
            "non_text_zones": [],
        },
    },

    "units": [],
    "regions": [],
    "page_intelligence": {},
    "style_system": {},
    "semantic": {},
    "relations": {},
    "policies": {},
    "reconstruction_constraints": {},
    "translation_context": {},
    "quality": {},
    "provenance": {},
}
6. L’unité centrale : units

Le cœur de INPUT_DATA, ce sont les unités.

Il ne faut pas seulement avoir :

blocks → lines → phrases

Il faut une liste canonique d’unités, chaque unité ayant un niveau :

page
region
block
line
phrase
span
word
char
image
drawing
table
cell
formula
code
overlay

Exemple :

unit = {
    "unit_id": "p001_block_003_phrase_002",
    "level": "phrase",
    "parent_id": "p001_block_003_line_001",
    "children_ids": [],

    "content": {
        "text": "The model learns features from data.",
        "raw_text": "The model learns features from data.",
        "normalized_text": "The model learns features from data.",
        "translated_text": None,
        "language": "en",
    },

    "geometry": {
        "bbox": [72.0, 145.2, 312.5, 158.7],
        "bbox_unit": "pt",
        "bbox_origin": "top_left",

        "bbox_px": [150, 303, 651, 331],
        "bbox_px_dpi": 150,

        "baseline": None,
        "rotation": 0,
        "reading_order_index": 42,
        "render_order_index": 42,
    },

    "visual": {
        "style": {
            "font_family": "Times New Roman",
            "font_size_pt": 10.5,
            "font_size_px": 21.8,
            "color": "#000000",
            "fill_color": "#000000",
            "background_color": None,
            "flags": {
                "bold": False,
                "italic": False,
                "underline": False,
                "serif": True,
                "monospace": False,
                "uppercase": False,
            },
        },
        "style_class": "Body_01",
        "style_confidence": 0.91,
    },

    "extraction": {
        "source": "native_pdf|ocr|merged",
        "source_kind": "native_span|ocr_phrase",
        "confidence": 0.97,
        "ocr_confidence_mean": None,
        "native_confidence": 1.0,
        "dedupe_status": "kept|merged|discarded",
    },

    "understanding": {
        "role": "body|section_heading|figure_caption|diagram_label|table_cell",
        "object_type": "narrative_text|short_label|formula|code|table_cell",
        "object_class": "editorial|technical|visual|tabular",
        "page_family": "body_text_two_column",
        "layout_type": "double_column",
        "document_type": "scientific_paper",
        "region_memberships": [],
        "structure_hints": {},
        "semantic_kind": "prose|formula|caption|reference|code",
    },

    "policy": {
        "translatable": True,
        "translation_strategy": "paragraph_flow|layout_constrained|exact_preserve",
        "render_policy": "paragraph_flow|anchored_text|fixed_preserve|background_only",
        "coverage_required": "strict|normal|optional",
        "preserve_exact_text": False,
        "preserve_visual": False,
    },

    "relations": {
        "previous_unit_id": "...",
        "next_unit_id": "...",
        "parent_region_id": "...",
        "flow_to_next": {},
        "flow_from_previous": {},
    },

    "constraints": {
        "allow_reflow": True,
        "allow_line_wrap": True,
        "preserve_bbox": False,
        "preserve_alignment": True,
        "preserve_font_size": True,
        "preserve_color": True,
        "allow_horizontal_expansion": True,
        "allow_vertical_expansion": True,
        "overflow_risk": 0.35,
    },

    "provenance": {
        "created_by": "native_pdf_extractor|DocumentParser|LayoutV2Builder",
        "postprocessed_by": [],
        "decision_trace": [],
    },
}

Cette structure est fondamentale.

Elle évite que la traduction ou la reconstruction devine quoi faire. Tout est déjà préparé.

7. page_intelligence

C’est la couche qui résume la compréhension globale de la page.

Elle doit reprendre les sorties de LayoutV2Builder, PageCaseClassifierV2, page_family_registry, style_profiler, etc.

"page_intelligence": {
    "page_role": "body",
    "page_family": "body_with_figure",
    "page_family_group": "body_with_diagram",
    "document_type": "manual_guide",
    "layout_type": "annotated_page",
    "style_profile": "editorial_visual",

    "reading_modes": {
        "linear_flow": 0.35,
        "columnar_flow": 0.12,
        "anchored_overlay_flow": 0.82,
        "toc_row_flow": 0.0,
        "tabular_grid_flow": 0.0,
        "glossary_pair_flow": 0.0,
    },

    "layout_tendencies": {
        "columnarity": "single",
        "text_density": "medium",
        "visual_density": "high",
        "table_density": "low",
        "lineation": "fragmented",
    },

    "translation_sensitivity": {
        "line_break_sensitivity": 0.74,
        "overflow_risk": 0.51,
        "anchoring_sensitivity": 0.91,
        "grid_alignment_sensitivity": 0.18,
        "lexical_preservation_bias": 0.37,
    },

    "risk_flags": [
        {
            "code": "anchor_attachment_drift",
            "severity": 0.91,
        }
    ],

    "decision_context": {
        "default_translation_strategy": "layout_constrained",
        "default_render_policy": "anchored_text",
        "preserve_anchors": True,
        "preserve_grid": False,
        "allow_paragraph_reflow": False,
    },
}

Cette couche doit peser fortement sur les unités.

Exemple :

Si anchored_overlay_flow est élevé,
alors les labels courts ne doivent pas être traités comme du texte courant.
8. regions

Les régions sont aussi centrales.

Une région peut être :

body_region
header_region
footer_region
table_region
table_cell
figure_region
chart_region
formula_region
code_region
caption_region
annotation_region
toc_region
background_region
non_text_zone

Exemple :

region = {
    "region_id": "p001_region_formula_002",
    "region_type": "formula",
    "role": "formula_region",
    "bbox": [220.4, 310.2, 355.9, 342.6],
    "bbox_unit": "pt",
    "bbox_px": [459, 646, 741, 714],
    "source": "special_region_detector|pdf_glyph|layout_ai|heuristic",
    "confidence": 0.93,

    "members": {
        "block_ids": [],
        "line_ids": [],
        "phrase_ids": [],
        "span_ids": [],
    },

    "policy": {
        "translatable": False,
        "translation_strategy": "exact_preserve",
        "render_policy": "fixed_preserve",
        "must_preserve_visual": True,
        "must_exclude_from_translation_flow": True,
    },

    "constraints": {
        "preserve_bbox": True,
        "preserve_as_overlay": True,
        "allow_reflow": False,
    },
}

Ensuite chaque unité reçoit :

"region_memberships": [
    {
        "region_id": "p001_region_formula_002",
        "region_type": "formula",
        "overlap_ratio": 0.91,
        "membership_role": "inside",
    }
]

C’est ainsi que les régions influencent vraiment le pipeline.

9. relations

INPUT_DATA doit aussi contenir les relations.

Relations utiles :

contains
belongs_to
flows_to
continues
new_sentence_after
same_paragraph
same_table_row
same_table_column
caption_of
label_of
legend_of
axis_of
annotates
overlaps
near
aligned_with
same_style_as

Exemple :

"relations": {
    "schema_version": "relations.v1",

    "reading_order": [
        "p001_block_001",
        "p001_block_002",
        "p001_block_003",
    ],

    "edges": [
        {
            "source_id": "p001_phrase_010",
            "target_id": "p001_phrase_011",
            "relation": "continues_wrapped_line",
            "logical_relation": "same_paragraph_continuation",
            "confidence": 0.86,
            "source": "element_relations",
        },
        {
            "source_id": "p001_caption_001",
            "target_id": "p001_image_001",
            "relation": "caption_of",
            "confidence": 0.78,
            "source": "layout_ai|heuristic",
        }
    ],
}

Cette couche est essentielle pour reconstruire.

10. policies

policies est le pont entre compréhension et action.

Le pipeline ne doit pas seulement savoir :

ceci est une formule

Il doit savoir :

donc on ne traduit pas,
on préserve visuellement,
on rend comme overlay fixe,
on exige une couverture stricte.

Exemple :

"policies": {
    "default_policy": {
        "translation_strategy": "layout_constrained",
        "render_policy": "anchored_text",
        "coverage_required": "strict",
    },

    "unit_policies": {
        "p001_phrase_010": {
            "unit_type": "narrative_body",
            "translatable": True,
            "translation_strategy": "paragraph_flow",
            "render_policy": "paragraph_flow",
            "coverage_required": "normal",
        },
        "p001_formula_001": {
            "unit_type": "formula",
            "translatable": False,
            "translation_strategy": "exact_preserve",
            "render_policy": "fixed_preserve",
            "coverage_required": "strict",
        },
    },
}

PagePolicyMatrix doit alimenter cette partie.

11. style_system

Il faut sortir du simple style individuel.

Il faut un système de styles :

"style_system": {
    "schema_version": "style_system.v1",

    "global_styles": {
        "Body_01": {
            "semantic_type": "body",
            "font_family": "Times New Roman",
            "font_size_pt": 10.5,
            "color": "#000000",
            "line_height_pt": 12.8,
            "flags": {
                "bold": False,
                "italic": False,
                "serif": True,
            },
        },
        "Heading1_01": {
            "semantic_type": "heading",
            "heading_level": 1,
            "font_family": "Arial",
            "font_size_pt": 18.0,
            "color": "#111111",
            "flags": {
                "bold": True,
            },
        },
    },

    "page_style_profile": {},
    "dominant_body_style_id": "Body_01",
    "heading_hierarchy": [],
    "caption_style_id": "Caption_01",
}

Les unités ne répètent pas tout. Elles référencent :

"style_class": "Body_01"

Mais elles gardent aussi leur style exact si nécessaire.

12. translation_context

Cette couche prépare la traduction.

"translation_context": {
    "source_lang": "en",
    "target_lang": "fr",

    "document_domain": "technical",
    "document_subdomain": "deep_learning",
    "translation_style": "technique",
    "translation_tone": "didactique",

    "terminology": {
        "domain": "technical",
        "subdomain": "deep_learning",
        "locked_terms": [],
        "preferred_terms": [],
        "reserved_terms": [],
    },

    "protected_tokens": [],
    "non_translatable_units": [],
    "layout_constrained_units": [],
    "paragraph_flow_units": [],
}

TerminologyManager peut alimenter cette couche, car il sait résoudre les termes selon domaine, sous-domaine, langue source, langue cible et rôle documentaire.

StyleToneClassifier peut aussi contribuer, mais il doit rester secondaire par rapport à la famille de page et au type documentaire.

13. reconstruction_constraints

C’est la partie que le reconstructeur doit consommer.

"reconstruction_constraints": {
    "page": {
        "preserve_page_size": True,
        "preserve_background": True,
        "preserve_margins": True,
        "coordinate_unit": "pt",
    },

    "layout": {
        "preserve_columns": True,
        "preserve_table_grid": False,
        "preserve_anchors": True,
        "allow_global_reflow": False,
    },

    "overflow": {
        "strategy": "fit_text|reduce_font|expand_box|line_wrap|manual_review",
        "max_font_reduction_ratio": 0.12,
        "allow_vertical_growth": True,
    },

    "visual_preservation": {
        "preserve_formulas_as_images": True,
        "preserve_code_blocks": True,
        "preserve_images": True,
        "preserve_drawings": True,
    },
}

Le reconstructeur ne doit pas deviner. Il lit ces contraintes.

14. quality

Chaque INPUT_DATA doit pouvoir dire :

est-ce fiable ?
où sont les risques ?
qu’est-ce qui doit être vérifié ?
"quality": {
    "overall_confidence": 0.87,

    "extraction_quality": {
        "native_text_coverage": 0.91,
        "ocr_confidence_mean": 0.84,
        "block_fragmentation_score": 0.22,
        "style_confidence_mean": 0.76,
    },

    "layout_quality": {
        "classification_confidence": {},
        "region_detection_confidence": 0.81,
        "reading_order_confidence": 0.88,
    },

    "risks": [
        {
            "code": "anchor_attachment_drift",
            "severity": 0.91,
            "affected_units": [],
        }
    ],

    "manual_review_required": False,
}
15. provenance

Très important pour déboguer.

"provenance": {
    "created_by": "ocr_server.py",
    "pipeline_version": "docs_parser.input_data.v1",
    "modules": {
        "native_pdf_extractor": "enabled",
        "rapidocr": "enabled",
        "document_parser": "enabled",
        "layout_v2_builder": "enabled",
        "page_case_classifier_v2": "enabled",
        "special_region_detector": "enabled",
        "page_policy_matrix": "enabled",
        "style_profiler": "enabled",
        "element_relations": "enabled",
    },

    "decision_trace": [
        {
            "stage": "page_classification",
            "decision": "layout_type=annotated_page",
            "confidence": 0.82,
        },
        {
            "stage": "unit_policy",
            "unit_id": "p001_phrase_010",
            "decision": "translation_strategy=layout_constrained",
            "reason": "inside annotated page",
        }
    ],
}

Sans provenance, tu ne pourras pas corriger les erreurs finement.

16. Où générer INPUT_DATA dans ocr_server.py

Il faut le générer après extraction + compréhension, mais avant traduction/reconstruction.

Ordre recommandé :

process_page()
  ↓
extraction native / OCR
  ↓
fusion / déduplication
  ↓
style profiling
  ↓
semantic phrase building
  ↓
LayoutV2Builder
  ↓
special_region_detector
  ↓
postprocessors
  ↓
LayoutV2Builder rebuild
  ↓
region memberships
  ↓
PagePolicyMatrix
  ↓
semantic spans/runs/groups
  ↓
build_INPUT_DATA()
  ↓
return INPUT_DATA

Donc tu crées une fonction :

def build_input_data(
    *,
    page_structure: dict,
    source_context: dict,
    extraction_context: dict,
    assets_context: dict,
) -> dict:
    ...

Et process_page retourne :

return {
    "page": idx + 1,
    "input_data": INPUT_DATA,
    "structure": page_structure,  # temporaire pour compatibilité
    "visual_url": vis_url,
}

À terme, structure peut disparaître ou devenir une vue dérivée de INPUT_DATA.

17. Différence entre page_structure et INPUT_DATA

page_structure actuel est un objet historique.

INPUT_DATA doit être le nouveau contrat.

Différence :

page_structure
= structure de travail interne actuelle

INPUT_DATA
= contrat canonique stabilisé pour tout le pipeline

Donc au début :

page_structure → build_input_data() → INPUT_DATA

Plus tard :

tous les modules consomment directement INPUT_DATA
18. Nom recommandé

Tu peux garder INPUT_DATA.

Mais techniquement, je recommande de le versionner :

"schema_version": "input_data.v1"

Et dans le code :

class InputDataBuilder:
    def build_page_input_data(...):
        ...

Fichier recommandé :

input_data_builder.py

Ou plus explicite :

canonical_input_builder.py

Mais si tu veux un nom simple et central :

input_data.py
19. Le vrai rôle de INPUT_DATA

Formulation précise :

INPUT_DATA est la représentation canonique, normalisée et enrichie d’une page,
contenant toutes les données nécessaires à la traduction, la reconstruction,
la mise en forme et la validation WYSIWYG.

Il contient :

ce qui a été vu
ce qui a été compris
ce qui doit être traduit
ce qui doit être préservé
ce qui doit être reconstruit
ce qui est risqué
ce qui est fiable
ce qui doit rester fixe
ce qui peut être reformulé ou reflow

C’est donc la tête d’unité d’entrée du pipeline.

20. Version minimale à implémenter d’abord

Ne commence pas par tout.

La première version doit contenir obligatoirement :

INPUT_DATA_V1_REQUIRED = {
    "schema_version",
    "document",
    "page",
    "assets",
    "units",
    "regions",
    "page_intelligence",
    "style_system",
    "relations",
    "policies",
    "translation_context",
    "reconstruction_constraints",
    "quality",
    "provenance",
}

Et pour chaque unité :

UNIT_REQUIRED = {
    "unit_id",
    "level",
    "parent_id",
    "children_ids",
    "content",
    "geometry",
    "visual",
    "extraction",
    "understanding",
    "policy",
    "relations",
    "constraints",
    "provenance",
}

C’est la base solide.

21. Conclusion

Oui, l’idée est correcte et même nécessaire.

Tu veux créer une couche INPUT_DATA qui sert de socle intelligent entre l’extraction et les traitements aval.

La bonne définition est :

INPUT_DATA = page analysée + unités normalisées + régions + styles + sémantique
             + relations + politiques + contraintes + qualité + provenance.

Et son objectif est :

empêcher la traduction et la reconstruction de travailler à l’aveugle.

C’est cette couche qui va rendre le pipeline WYSIWYG robuste, contrôlable et extensible.

Et encore ? Complète avec ce que je n'ai pas vu prévu imaginé ... pour améliorer et figé le concept. On aura notre première partie de ocr_server (ocr_server sera coupé en plusieurs unités et non plus en un seul code long non maintenable).
J'attends alors ton apport et tes idéees innovantes.
Il faut aussi un nom pour cela

Je vais figer le concept comme une vraie brique logicielle : rôle, nom, frontières, schéma, modules, responsabilités, erreurs à éviter, et innovations à intégrer dès la V1.

Thought for 22s

Oui. Il faut maintenant figer le concept comme une brique indépendante du serveur. Ce ne doit pas être seulement une variable INPUT_DATA. Ce doit être une tête de pipeline complète, avec un nom, un contrat, des sous-modules, des validateurs et une sortie stable.

Je propose de l’appeler :

PAGEPRINT

Nom complet :

PAGEPRINT — Canonical Page Intelligence Input

Nom technique dans le code :

pageprint

Nom du schéma :

pageprint.input.v1

Pourquoi ce nom ? Parce que cette brique doit produire l’empreinte complète d’une page : géométrie, fond, texte, images, styles, régions, rôles, sémantique, risques, relations, politiques, contraintes et provenance.

Tu peux garder INPUT_DATA comme nom fonctionnel, mais le concept logiciel devrait être :

PAGEPRINT = le générateur de INPUT_DATA
INPUT_DATA = la sortie canonique de PAGEPRINT
1. Définition figée
PAGEPRINT est la première tête du pipeline WYSIWYG.

Son rôle est de transformer une page source en une représentation canonique,
normalisée, enrichie, vérifiable et consommable par tous les modules aval :
traduction, reconstruction, mise en forme, optimisation de layout, QA et export.

Donc :

PDF / IMAGE / DOCX / PPTX
        ↓
     PAGEPRINT
        ↓
  INPUT_DATA canonique
        ↓
TRADUCTION / RECONSTRUCTION / MISE_EN_FORME / QA
2. Ce que tu avais déjà bien vu

Tu avais déjà identifié les grandes dimensions :

données analytiques
données visuelles
données sémantiques
caractéristiques de page
fonds
formes
styles
zones
régions
relations
unités d’extraction

C’est juste. Mais il faut ajouter plusieurs dimensions que tu n’avais pas encore explicitement figées.

3. Ce qu’il faut ajouter au concept
3.1 Une couche de contrat, pas seulement une couche de données

INPUT_DATA ne doit pas être un simple dictionnaire rempli par hasard.

Il doit être un contrat strict.

Chaque module aval doit pouvoir dire :

Je sais exactement où lire :
- le texte à traduire ;
- les éléments à préserver ;
- les bboxes en points ;
- les bboxes pixels de debug ;
- les styles ;
- les régions ;
- les contraintes ;
- les risques ;
- les relations ;
- les politiques de rendu.

Donc PAGEPRINT doit produire un objet validé par schéma :

{
    "schema_version": "pageprint.input.v1",
    "coordinate_unit": "pt",
    "page": {},
    "assets": {},
    "units": [],
    "regions": [],
    "relations": {},
    "policies": {},
    "constraints": {},
    "quality": {},
    "provenance": {}
}

Ce contrat est plus important que l’algorithme lui-même.

3.2 Une couche de normalisation d’unités

Tu as déjà identifié qu’il faut utiliser les points plutôt que les pixels. Il faut aller plus loin.

Toutes les grandeurs doivent être normalisées :

bbox            → points PDF
bbox_px         → pixels de rendu seulement
font_size       → font_size_pt
line_height     → line_height_pt
margin          → pt
column_width    → pt
spacing         → pt
stroke_width    → pt si vectoriel, px si raster mais annoté
color           → hex normalisé #rrggbb
rotation        → degrés
confidence      → 0.0 à 1.0
reading_order   → entier stable

Donc PAGEPRINT doit avoir un module :

unit_normalizer.py

Responsabilité :

convertir, nettoyer, borner, typer et annoter toutes les unités.
3.3 Une couche de résolution de conflits

C’est très important.

Tu vas avoir plusieurs sources qui se contredisent :

Native PDF dit : bloc A
OCR dit : bloc B
Layout AI dit : région table
Special detector dit : formule
Page classifier dit : page annotée
Postprocessor dit : fusionner les blocs

Il faut donc un module qui décide :

quelle source gagne ?
dans quel contexte ?
avec quelle confiance ?

Créer :

evidence_resolver.py

Il doit produire pour chaque unité :

"evidence": {
    "sources": [
        {
            "source": "native_pdf",
            "claim": "text_block",
            "confidence": 0.98
        },
        {
            "source": "ocr",
            "claim": "text_block",
            "confidence": 0.86
        },
        {
            "source": "special_region_detector",
            "claim": "formula_region",
            "confidence": 0.91
        }
    ],
    "resolved_as": "formula",
    "resolution_rule": "special_region_over_text_when_overlap_gt_0.65",
    "confidence": 0.91
}

Sans cette couche, le pipeline restera fragile.

3.4 Une couche de graphe documentaire

Tu as prévu les relations, mais il faut les figer comme un graphe.

Une page n’est pas seulement une liste de blocs. C’est un graphe :

page contient region
region contient block
block contient line
line contient phrase
phrase contient span
caption décrit image
label annote schéma
cell appartient à table
phrase continue phrase précédente

Donc PAGEPRINT doit produire :

"graph": {
    "nodes": [],
    "edges": []
}

Relations essentielles :

contains
belongs_to
flows_to
continues
same_paragraph
same_list
same_table
same_row
same_column
caption_of
label_of
legend_of
axis_of
annotates
overlaps
near
aligned_with
style_similar_to

element_relations.py et element_relations_ai.py vont déjà dans cette direction, et LayoutV2Builder les appelle via structure_extractor.py. Il faut maintenant formaliser leur sortie dans un graphe stable.

3.5 Une couche de mémoire de décision

Chaque décision importante doit être traçable.

Exemple :

Pourquoi ce bloc est une légende ?
Pourquoi cette phrase n’est pas traduite ?
Pourquoi cette zone est une formule ?
Pourquoi ces deux fragments ont été fusionnés ?

Ajouter :

"decision_trace": [
    {
        "stage": "special_region_detection",
        "target_id": "unit_042",
        "decision": "object_type=formula",
        "reason": "math_symbols_and_pdf_glyph_formula_candidate",
        "confidence": 0.93
    }
]

Cela te permettra de déboguer proprement les erreurs.

3.6 Une couche de prévision des risques aval

PAGEPRINT ne doit pas seulement décrire la page. Il doit prédire les problèmes qui vont arriver après traduction.

Exemples :

traduction plus longue que le texte source
débordement probable
perte d’alignement tabulaire
décrochage d’un label de schéma
confusion légende / paragraphe
risque de fusion abusive
risque de rupture TOC

PageCaseClassifierV2 produit déjà des signaux comme translation_sensitivity_signals, reading_modes et risk_flags. Il faut les compiler dans une couche :

"downstream_risks": {
    "translation_overflow": 0.72,
    "anchor_drift": 0.88,
    "grid_alignment_loss": 0.14,
    "toc_fragmentation": 0.0,
    "manual_review_required": False
}

C’est innovant parce que l’extraction prépare déjà la reconstruction.

3.7 Une couche de politiques compilées

PagePolicyMatrix décide déjà si une unité doit être traduite, préservée, ancrée ou rendue en flux paragraphe. Mais sa sortie doit être centralisée.

Créer :

policy_compiler.py

Son rôle :

prendre page_intelligence + regions + unit roles + source_kind + style + risks,
et produire une politique finale par unité.

Exemple :

"compiled_policy": {
    "translation": {
        "enabled": True,
        "strategy": "layout_constrained",
        "max_expansion_ratio": 1.25
    },
    "rendering": {
        "mode": "anchored_text",
        "preserve_bbox": True,
        "allow_font_reduction": True
    },
    "preservation": {
        "preserve_text_exactly": False,
        "preserve_visual_exactly": False
    }
}

La politique devient un ordre exécutable pour les modules aval.

3.8 Une couche de budget de transformation

C’est une idée importante.

Chaque unité doit savoir jusqu’où elle peut changer.

Exemple :

"transformation_budget": {
    "max_text_expansion_ratio": 1.30,
    "max_font_reduction_ratio": 0.12,
    "max_bbox_growth_x_pt": 8.0,
    "max_bbox_growth_y_pt": 12.0,
    "allow_line_wrap": True,
    "allow_reflow": False,
    "allow_reposition": False,
    "must_keep_alignment": True
}

Pourquoi ? Parce que la traduction va souvent allonger le texte. Le reconstructeur doit savoir quoi faire.

Pour une cellule de tableau :

"allow_reflow": False
"must_keep_alignment": True
"max_font_reduction_ratio": 0.18

Pour un paragraphe normal :

"allow_reflow": True
"allow_vertical_growth": True

Pour une formule :

"allow_reflow": False
"preserve_visual_exactly": True

C’est ce qui relie extraction et reconstruction.

3.9 Une couche de stratégie de traduction par unité

Ne laisse pas le traducteur deviner.

Chaque unité doit avoir :

"translation_directive": {
    "mode": "translate|preserve|copy|terminology_only|skip",
    "style": "technique",
    "tone": "didactique",
    "domain": "deep_learning",
    "preserve_terms": [],
    "forbidden_changes": [],
    "context_before": "...",
    "context_after": "..."
}

page_family_registry.py donne déjà des styles et tons par famille de page, par exemple page avec figure, page tableau, page diagramme, page technique, etc. TerminologyManager peut ensuite ajouter domaine, sous-domaine, termes verrouillés et termes préférés.

3.10 Une couche de fonds et couches visuelles

Pour le WYSIWYG, il ne suffit pas d’avoir du texte.

Il faut décomposer la page en couches :

background_layer
image_layer
drawing_layer
text_layer
overlay_layer
mask_layer
debug_layer

Dans INPUT_DATA :

"visual_layers": {
    "background": {
        "path": "...",
        "type": "cleaned_background",
        "text_removed": True
    },
    "source_render": {
        "path": "...",
        "dpi": 150
    },
    "masks": {
        "text_mask": "...",
        "formula_mask": "...",
        "image_mask": "..."
    },
    "overlays": [
        {
            "overlay_id": "formula_001",
            "type": "preserve_visual",
            "bbox": [...],
            "source": "special_region_detector"
        }
    ]
}

special_region_detector.py produit déjà des candidats de formules/code/régions techniques, avec sous-régions à préserver. Il faut les raccorder directement à cette logique de couches.

4. Ce que PAGEPRINT ne doit pas faire

Très important pour couper ocr_server.py.

PAGEPRINT ne doit pas :

traduire
réécrire
optimiser la mise en page traduite
produire le PDF final
gérer les endpoints HTTP
faire de l’export utilisateur

Il doit seulement produire :

INPUT_DATA

Donc sa frontière est nette :

Entrée  : page source + ressources + extraction brute
Sortie  : INPUT_DATA canonique
5. Architecture logicielle proposée

Il faut couper ocr_server.py en modules.

Structure recommandée :

docs_parser/
│
├── server/
│   ├── api.py
│   ├── routes_ocr.py
│   ├── routes_translate.py
│   ├── routes_reconstruct.py
│   └── health.py
│
├── pipelines/
│   ├── orchestrator.py
│   ├── legacy_pipeline.py
│   ├── enhanced_pipeline.py
│   ├── perfect_pipeline.py
│   └── hybrid_compare_pipeline.py
│
├── pageprint/
│   ├── __init__.py
│   ├── builder.py
│   ├── schema.py
│   ├── normalizer.py
│   ├── unit_factory.py
│   ├── region_index.py
│   ├── evidence_resolver.py
│   ├── graph_builder.py
│   ├── policy_compiler.py
│   ├── constraint_compiler.py
│   ├── quality_assessor.py
│   ├── provenance.py
│   ├── validators.py
│   └── serializers.py
│
├── extractors/
│   ├── native_pdf_extractor.py
│   ├── ocr_extractor.py
│   ├── office_converter.py
│   ├── image_extractor.py
│   └── visual_attribute_extractor.py
│
├── intelligence/
│   ├── page_case_classifier_v2.py
│   ├── page_policy_matrix.py
│   ├── page_profile_registry.py
│   ├── page_family_registry.py
│   ├── special_region_detector.py
│   ├── style_profiler.py
│   ├── element_relations.py
│   └── element_relations_ai.py
│
├── postprocessors/
│   ├── page_extraction_postprocessors.py
│   ├── semantic_corrector.py
│   ├── table_postprocessor.py
│   ├── chart_postprocessor.py
│   └── toc_postprocessor.py
│
├── translation/
│   ├── translator.py
│   ├── terminology_manager.py
│   └── style_tone_classifier.py
│
├── reconstruction/
│   ├── reconstructor.py
│   ├── layout_optimizer.py
│   └── html_exporter.py
│
└── qa/
    ├── visual_compare.py
    ├── input_data_audit.py
    └── reconstruction_audit.py

Le serveur devient mince. Il ne contient plus l’intelligence.

6. Le vrai pipeline après découpage
API /ocr
  ↓
PipelineOrchestrator
  ↓
SourceLoader
  ↓
ExtractionEngine
  ↓
PageUnderstanding
  ↓
PAGEPRINT Builder
  ↓
INPUT_DATA
  ↓
JSON response ou stockage

Dans le code :

input_data = PagePrintBuilder().build(
    source_context=source_context,
    extraction_result=extraction_result,
    page_structure=page_structure,
    assets=assets,
)
7. Les sous-modules de PAGEPRINT
7.1 schema.py

Contient les définitions.

PAGEPRINT_SCHEMA_VERSION = "pageprint.input.v1"

REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "document",
    "page",
    "assets",
    "units",
    "regions",
    "relations",
    "page_intelligence",
    "policies",
    "constraints",
    "quality",
    "provenance",
}
7.2 unit_factory.py

Transforme les anciens blocs/lignes/phrases/spans en unités canoniques.

Entrée :

page_structure["blocks"]

Sortie :

units[]

Chaque unité reçoit :

unit_id
level
parent_id
children_ids
content
geometry
visual
extraction
understanding
policy
constraints
quality
provenance
7.3 region_index.py

Centralise les régions.

Il fusionne :

regions
special_regions
non_text_zones
images
drawings
tables
charts
formulas
code regions
layout_ai regions

Puis calcule les appartenances :

unit → region
region → unit
7.4 evidence_resolver.py

Résout les conflits entre :

native_pdf
ocr
layout_ai
special_region_detector
heuristics
postprocessors
LLM/ONNX

Il produit une décision traçable.

7.5 graph_builder.py

Construit :

reading graph
containment graph
layout graph
semantic graph
visual attachment graph
7.6 policy_compiler.py

Compile les décisions de :

PagePolicyMatrix
page_case_v2
region_memberships
style_profile
document_type
layout_type
risk_flags

en politiques finales.

PagePolicyMatrix est déjà bien placé pour décider des stratégies de traduction et de rendu selon le rôle, le type d’unité, le type de page, le layout et le profil de style.

7.7 constraint_compiler.py

Produit les contraintes WYSIWYG :

preserve_bbox
allow_reflow
allow_wrap
allow_font_scaling
preserve_alignment
preserve_grid
preserve_anchor
preserve_visual
7.8 quality_assessor.py

Produit :

score de confiance
risques
fragments suspects
unités faibles
zones à revoir

Il utilise notamment les métriques de fragmentation déjà calculées dans page_extraction_postprocessors.py.

7.9 validators.py

Valide que INPUT_DATA est exploitable :

toutes les bboxes sont en points
chaque unit_id est unique
chaque parent_id existe
chaque region_id existe
chaque policy est complète
chaque unité traduisible a du texte
chaque unité non traduisible a une raison
chaque bbox a une surface positive
8. Innovation importante : INPUT_DATA doit être multi-vues

La même page doit pouvoir être vue de plusieurs manières.

Vue hiérarchique
page → regions → blocks → lines → phrases → spans
Vue plate
units[]
Vue graphe
nodes + edges
Vue traduction
translation_units[]
Vue reconstruction
render_units[]
Vue debug
evidence + quality + decisions

Donc dans INPUT_DATA :

"views": {
    "hierarchical": {},
    "flat_units": [],
    "translation_units": [],
    "render_units": [],
    "debug_units": []
}

Cela évite que chaque module aval reconstruise sa propre vue.

9. Innovation : unit_lifecycle

Chaque unité doit avoir un cycle de vie.

"lifecycle": {
    "created_at_stage": "native_extraction",
    "current_stage": "pageprint_final",
    "status": "active",
    "merged_from": [],
    "split_from": None,
    "superseded_by": None,
    "discard_reason": None
}

Pourquoi ? Parce que les postprocessors fusionnent, découpent et corrigent. Si tu perds l’historique, tu ne peux plus auditer.

10. Innovation : confidence_vector

Un seul score de confiance ne suffit pas.

Chaque unité doit avoir un vecteur :

"confidence": {
    "text": 0.97,
    "bbox": 0.92,
    "style": 0.73,
    "role": 0.81,
    "reading_order": 0.88,
    "region_membership": 0.76,
    "translation_policy": 0.91,
    "overall": 0.84
}

Cela permet de traiter différemment :

texte fiable mais style incertain
bbox fiable mais rôle incertain
rôle fiable mais ordre de lecture incertain
11. Innovation : semantic_pressure

Pour chaque unité, tu peux calculer la pression sémantique.

C’est-à-dire : à quel point une erreur sur cette unité va dégrader le document.

"semantic_pressure": {
    "importance": 0.82,
    "reason": "section_heading",
    "translation_priority": "high",
    "layout_priority": "high",
    "preservation_priority": "medium"
}

Exemples :

titre de chapitre        → pression élevée
numéro de page           → pression faible mais préservation stricte
formule                  → pression élevée, traduction interdite
label de schéma          → pression élevée, ancrage strict
paragraphe courant       → pression moyenne, reflow permis
12. Innovation : layout_freedom

Chaque unité doit savoir son degré de liberté.

"layout_freedom": {
    "x": "fixed|elastic|free",
    "y": "fixed|elastic|free",
    "width": "fixed|elastic",
    "height": "fixed|elastic",
    "font_size": "fixed|shrink_allowed|adaptive",
    "line_breaks": "preserve|adaptive|reflow",
    "anchor": "none|region|visual_object|page"
}

C’est supérieur à un simple render_policy.

Exemple :

table_cell:
x fixed, y fixed, width fixed, height elastic faible

body_paragraph:
x fixed, y elastic, width fixed, height elastic

diagram_label:
x fixed, y fixed, width elastic faible, anchor visual_object

formula:
tout fixed
13. Innovation : translation_expansion_model

Avant de traduire, on peut prévoir l’expansion probable.

Pour EN → FR, un texte peut s’allonger de 10 à 35 % selon domaine.

Ajouter :

"translation_forecast": {
    "expected_length_ratio": 1.22,
    "overflow_probability": 0.64,
    "recommended_strategy": "font_shrink_then_wrap",
    "fallback_strategy": "manual_review"
}

Cela permet au reconstructeur de ne pas découvrir les problèmes trop tard.

14. Innovation : visual_anchor_map

Pour les schémas, figures, graphiques, labels, légendes :

"visual_anchor_map": [
    {
        "text_unit_id": "unit_label_004",
        "target_visual_id": "image_001",
        "anchor_type": "label_of",
        "anchor_side": "left",
        "anchor_point": [120.4, 330.2],
        "distance_pt": 18.5,
        "confidence": 0.79
    }
]

C’est nécessaire pour les pages annotées. PageCaseClassifierV2 détecte déjà les flux de type anchored_overlay_flow, et page_extraction_postprocessors.py contient des logiques d’annotation autour d’illustrations, chart labels, table fragments, etc.

15. Innovation : render_contract

Chaque unité doit recevoir un contrat final de rendu.

"render_contract": {
    "mode": "paragraph_flow",
    "target_layer": "text_layer",
    "background_handling": "erase_and_redraw",
    "text_box": {
        "bbox": [...],
        "unit": "pt",
        "baseline_policy": "preserve_baseline_grid"
    },
    "font_policy": {
        "family": "preserve_or_substitute",
        "size": "adaptive_within_budget",
        "color": "preserve"
    },
    "overflow_policy": {
        "first": "wrap",
        "second": "shrink_font",
        "third": "expand_height",
        "final": "flag_review"
    }
}

La reconstruction ne doit plus interpréter des champs dispersés. Elle lit render_contract.

16. Innovation : translation_contract

De même pour la traduction.

"translation_contract": {
    "mode": "translate",
    "strategy": "layout_constrained",
    "context_scope": "paragraph",
    "preserve_terms": [],
    "protected_patterns": [
        "formula",
        "url",
        "code",
        "page_number"
    ],
    "max_output_chars": 140,
    "style": "technique",
    "tone": "didactique",
    "domain": "technical"
}

Cela évite que la traduction casse le layout.

17. Innovation : audit_overlay

Prévoir dès maintenant une sortie visuelle de debug.

PAGEPRINT doit pouvoir générer une superposition :

bbox des unités
couleurs par type
régions spéciales
liens caption/image
ordre de lecture
unités traduisibles / non traduisibles
risques overflow

Dans INPUT_DATA :

"debug": {
    "audit_overlay_path": "...",
    "audit_overlay_url": "...",
    "layers": [
        "units",
        "regions",
        "reading_order",
        "policies",
        "risks"
    ]
}
18. Innovation : replayable_pipeline

Chaque INPUT_DATA doit permettre de rejouer une partie du pipeline.

Ajouter :

"replay": {
    "source_file_hash": "...",
    "page_render_hash": "...",
    "config_hash": "...",
    "module_versions": {},
    "random_seed": None,
    "environment_flags": {}
}

Cela rend les bugs reproductibles.

19. Innovation : compatibility_views

Pendant la migration, les anciens modules attendent page_structure.

Donc PAGEPRINT doit fournir :

"compatibility": {
    "legacy_page_structure": {},
    "reconstructor_payload_v1": {},
    "translator_payload_v1": {}
}

Mais attention : ce sont des vues dérivées, pas la source de vérité.

Source de vérité = INPUT_DATA
Vues legacy = export temporaire
20. Le schéma final figé

La V1 de INPUT_DATA devrait ressembler à ceci :

INPUT_DATA = {
    "schema_version": "pageprint.input.v1",
    "input_id": "...",

    "document": {},
    "page": {},

    "assets": {},
    "visual_layers": {},

    "units": [],
    "regions": [],
    "graph": {},
    "relations": {},

    "page_intelligence": {},
    "style_system": {},
    "semantic_system": {},

    "policies": {},
    "constraints": {},

    "translation_context": {},
    "reconstruction_context": {},

    "views": {
        "hierarchical": {},
        "translation_units": [],
        "render_units": [],
        "debug_units": []
    },

    "quality": {},
    "risks": {},
    "provenance": {},
    "debug": {},
    "compatibility": {}
}
21. Le nom des sous-parties

Je recommande ces noms :

PAGEPRINT Builder          → construit INPUT_DATA
PAGEPRINT Schema           → définit le contrat
PAGEPRINT Normalizer       → normalise unités, points, styles
PAGEPRINT Region Index     → indexe régions/zones/objets
PAGEPRINT Evidence Resolver→ résout conflits entre sources
PAGEPRINT Graph Builder    → construit les graphes
PAGEPRINT Policy Compiler  → compile politiques
PAGEPRINT Constraint Engine→ compile contraintes WYSIWYG
PAGEPRINT Quality Auditor  → évalue confiance et risques
PAGEPRINT Legacy Adapter   → produit les vues anciennes
22. Les noms possibles

Voici plusieurs noms possibles, classés.

Meilleur choix
PAGEPRINT

Avantages :

court
mémorisable
évoque l’empreinte complète de la page
adapté au WYSIWYG
peut devenir un vrai module logiciel

Nom Python :

pageprint

Classe :

PagePrintBuilder

Schéma :

pageprint.input.v1
Autres bons noms
PageDNA

Très parlant : le génome de la page. Mais un peu métaphorique.

DocGenome

Bon pour document entier, moins précis pour page.

PageCore

Très technique, mais moins distinctif.

InputCore

Clair, mais trop générique.

SensePage

Cohérent avec vSense, mais moins précis.

PageMind

Bon pour compréhension, mais trop IA-marketing.

DocSense Core

Cohérent avec ton projet vSense, mais plus long.

23. Mon choix final

Je recommande :

PAGEPRINT

Avec cette convention :

# package
docs_parser/pageprint/

# object
INPUT_DATA

# schema
pageprint.input.v1

# builder
PagePrintBuilder

# main function
build_pageprint_input_data(...)

Phrase de définition :

PAGEPRINT produit l’empreinte canonique d’une page, sous forme d’INPUT_DATA,
afin que les modules de traduction, reconstruction, mise en forme et QA
travaillent sur une représentation unique, normalisée, auditable et riche.
24. Découpage pratique de l’ancien ocr_server.py

L’ancien ocr_server.py doit devenir seulement :

serveur HTTP
upload
routing
appel du pipeline
retour JSON

Il ne doit plus contenir :

classification
postprocessing
sémantique
règles page
politiques
normalisation
relations
inpainting
reconstruction
traduction

Découpage recommandé :

ocr_server.py
↓ devient
server/api.py

Puis :

process_page()
↓ devient
pipelines/enhanced_pipeline.py

Et la fin de process_page, qui fabrique la structure finale, devient :

pageprint/builder.py
25. Pipeline cible avec PAGEPRINT
1. SourceLoader
   - lit PDF/image/Office
   - convertit Office en PDF si besoin
   - calcule hash/source metadata

2. PageRenderer
   - rend la page en image
   - garde DPI, scale, bbox_px

3. RawExtractors
   - NativePDFExtractor
   - OCRExtractor
   - VisualAttributeExtractor
   - Image/Drawing extractor

4. PageUnderstanding
   - LayoutV2Builder
   - PageCaseClassifierV2
   - StyleProfiler
   - SpecialRegionDetector
   - Postprocessors

5. PAGEPRINT
   - normalise
   - résout conflits
   - fabrique units
   - indexe regions
   - construit graph
   - compile policies
   - compile constraints
   - évalue quality
   - produit INPUT_DATA

6. Downstream
   - Translator
   - Reconstructor
   - LayoutOptimizer
   - QA
26. Ce qu’il faut faire dès maintenant

Priorité de développement :

1. Créer docs_parser/pageprint/schema.py
2. Créer docs_parser/pageprint/builder.py
3. Créer docs_parser/pageprint/normalizer.py
4. Créer docs_parser/pageprint/unit_factory.py
5. Créer docs_parser/pageprint/region_index.py
6. Créer docs_parser/pageprint/policy_compiler.py
7. Créer docs_parser/pageprint/validators.py
8. Modifier ocr_server.py pour appeler PagePrintBuilder
9. Retourner input_data dans /ocr
10. Garder structure legacy temporairement dans compatibility
27. V1 minimale à figer

La première V1 ne doit pas tout faire parfaitement. Elle doit seulement garantir le contrat.

V1 obligatoire :

document
page
assets
units
regions
page_intelligence
style_system
relations
policies
constraints
quality
provenance
compatibility

V1 non obligatoire mais prévue :

semantic_pressure
translation_forecast
visual_anchor_map
audit_overlay
replay
confidence_vector avancé
layout_freedom avancé
28. Résumé net

Ton idée doit devenir ceci :

PAGEPRINT est la première tête du pipeline WYSIWYG.

Elle remplace le vrac d’ocr_server.py par une unité maintenable qui produit
INPUT_DATA : une empreinte canonique, normalisée, riche et auditable de chaque page.

INPUT_DATA devient la source unique de vérité pour traduction, reconstruction,
mise en forme, optimisation, QA et export.

Le choix stratégique est bon. Le point à ne pas manquer est celui-ci :

PAGEPRINT ne doit pas seulement stocker les données extraites.
Il doit aussi stocker les décisions, les conflits, les preuves, les politiques,
les contraintes, les risques et les vues de compatibilité.

C’est cela qui rendra le pipeline robuste, pas seulement plus détaillé.
