• Le frontend actuel est un simple fichier statique, frontend/index.html, pas
  une app avec package.json.

  Pour l’utiliser correctement:

  1. Démarre le backend sur 8001:

  .docs-parser/bin/python ocr_server.py

  ou

  uvicorn ocr_server:app --host 0.0.0.0 --port 8001

  2. Sers le frontend en local:

  cd frontend
  python3 -m http.server 8000

  3. Ouvre:

  http://127.0.0.1:8000

  Point important: le bouton appelle POST /pipeline/run dans
  frontend/index.html:439, donc si tu ouvres le fichier en file://, ça ne
  marchera pas. Il faut soit le servir sur le même origin que l’API, soit
  ajouter un proxy, soit modifier l’URL du fetch vers
  http://127.0.0.1:8001/pipeline/run.
  
  
  
  OU BIEN
  
  
• Le frontend se lance maintenant avec une seule commande backend:

  .docs-parser/bin/python ocr_server.py

  Puis ouvre:

  http://127.0.0.1:8001/

  La racine redirige vers l’UI frontend/index.html, et le bouton appelle bien
  POST /pipeline/run sur le même origin, donc pas de problème de CORS ni de
  fetch cassé.

  J’ai aussi vérifié que ocr_server.py compile.

  
  
  
  
  
  

