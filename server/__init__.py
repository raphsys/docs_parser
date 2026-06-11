"""Server — couche HTTP mince du clone objectif.

Le serveur ne contient plus l'intelligence : seulement upload, routing,
appel du pipeline et retour JSON. Toute l'intelligence vit dans pipelines/
et pageprint/.
"""
