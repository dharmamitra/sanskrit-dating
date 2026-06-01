# Sanskrit Dating

Category-free **relative chronology of Sanskrit texts**, inferred from intertextual
citation structure together with stylometric signal (morphology + function-word
*n*-grams), and resolved into absolute dates with a Gibbs sampler anchored on a set
of externally-dated works.

## 🗺️ Interactive chronology map

An interactive, searchable timeline of the dated corpus is published via GitHub Pages:

**→ https://dharmamitra.github.io/sanskrit-dating/**

The map (`sanskrit_chronology_interactive.html`) lets you hover for per-work dates and
notes, search by title / author / anchor text, and zoom into any period. Static
renderings are also included (`sanskrit_chronology.png`, `sanskrit_chronology_all.png`).

## Approach

1. **Intertextual graph** — parallel passages between works (`matches/`) are extracted
   into a directed citation/relation graph (`extract_edges.py`, `extract_relations.py`,
   `orient.py`, `mst.py`). Direction of borrowing is the key signal; an undirected graph
   alone does not date.
2. **Stylometric signal** — morphology and function-word *n*-grams give a genre-largely-
   independent "composition-style" clock (`extract_morph.py`, `extract_ngrams.py`,
   `extract_pos.py`, `date_morph*.py`). Raw vocabulary turned out to be too weak a clock
   on its own (Sanskrit is normatively frozen).
3. **Anchoring & inference** — known/argued dates are encoded as priors and order
   constraints (`manual_constraints.tsv`, `researched_anchors.tsv`, `chronbmm_priors.tsv`,
   `vedic_anchors.tsv`, `dcs_anchors.tsv`). A Gibbs sampler propagates dates over the
   graph subject to these constraints (`date_gibbs*.py`, `date_gibbs_full.py`).
4. **Visualization** — `visualize*.py` render the results to PNG and the interactive HTML.

## Repository layout

| Path | What it is |
|------|------------|
| `extract_*.py` | Feature/edge/relation extraction from the corpus |
| `dating*.py`, `date_*.py` | Dating models (morphology, directional graph, Gibbs sampler) |
| `visualize*.py` | Render PNG + interactive HTML chronology |
| `*.tsv` | Constraints, anchors, and **dated results** (e.g. `dated_gibbs_full.tsv`) |
| `meta.json`, `metadata.tsv`, `text-information.json`, `text_info/` | Work metadata |
| `sanskrit_chronology_interactive.html` | The interactive map (served by Pages) |
| `*.png` | Static chronology renderings |

### Key result files

- `dated_gibbs_full.tsv` — final dated works from the full Gibbs run (input to the map)
- `dated_chunks_works.tsv` — per-work date estimates from chunk-level dating
- `researched_anchors.tsv`, `manual_constraints.tsv` — the anchoring evidence base

## Data not included

Large, regenerable artifacts are **not** version-controlled (see `.gitignore`):

- `matches/` — the raw parallel-passage corpus (~2.2 GB)
- `*.pkl` — extracted feature stores (`vocab.pkl`, `chunks_fw.pkl`, `fwgrams.pkl`, …)
- `chunks_dense.tsv` — dense chunk matrix (~62 MB)

These are produced by the `extract_*.py` scripts from the source corpus and feature stores.

## Reproducing

```bash
# 1. extract features + intertextual edges from the corpus (regenerates the .pkl / matrix files)
python extract_chunks.py
python extract_morph.py
python extract_edges.py
python extract_relations.py

# 2. run the dating model
python date_gibbs_full.py        # -> dated_gibbs_full.tsv

# 3. build the visualizations
python visualize.py              # -> sanskrit_chronology.png
python visualize_interactive.py  # -> sanskrit_chronology_interactive.html
```
