# PUBREADY — BASELINE (avant refonte pubready)

Source: `results/show10_mission2` · pages: 10 · **avg publication_ready_score: 0.96**
ready stricts: 8/10 · ok/review/ko: 10/0/0

| page | status | ready | score | overlap | typo | leak | ko |
|---|---|---|---|---|---|---|---|
|  in Deep Learnin_p0140 | ok | False | 0.8 | 1.0 | 0.9 | low | 0 |
| l SQL A Beginner_p0051 | ok | True | 1.0 | 1.0 | 1.0 | low | 0 |
| l SQL A Beginner_p0133 | ok | True | 1.0 | 1.0 | 1.0 | low | 0 |
| l SQL A Beginner_p0180 | ok | True | 1.0 | 1.0 | 1.0 | low | 0 |
| l SQL A Beginner_p0457 | ok | True | 1.0 | 1.0 | 1.0 | low | 0 |
| l SQL A Beginner_p0505 | ok | True | 1.0 | 1.0 | 1.0 | low | 0 |
| _docintelligence_p0192 | ok | False | 0.8 | 1.0 | 1.0 | low | 0 |
| _docintelligence_p0337 | ok | True | 1.0 | 1.0 | 1.0 | low | 0 |
| _docintelligence_p0406 | ok | True | 1.0 | 1.0 | 1.0 | low | 0 |
| _docintelligence_p0463 | ok | True | 1.0 | 1.0 | 1.0 | low | 0 |

## Note
Baseline figée AVANT l'instrumentation pubready et la refonte composition/legacy. Sert de référence anti-régression: le nouveau pipeline ne doit pas faire pire que ces scores.
Le détail granulaire (bloc→phrase→dimension) sera produit une fois pubready intégré à l'orchestrateur (M11).
