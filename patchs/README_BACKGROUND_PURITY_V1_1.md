# Background Purity v1.1

Correction du contrat : le background ne doit pas être vide de tout contenu source.

Il doit conserver les visuels non textuels :
- images ;
- diagrammes/formes/flèches ;
- graphes non textuels ;
- illustrations.

Il doit effacer :
- tout texte, traduisible ou non ;
- numéros de page ;
- titres/en-têtes/pieds de page ;
- labels/captions ;
- formules/équations/math ;
- code.

Application :

```bash
cd ~/Mes_Projets/docs_parser
rm -rf patchs
unzip -o ~/Téléchargements/patchs_background_purity_v1_1.zip
bash patchs/apply_all_patches.sh
```
