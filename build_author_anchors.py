#!/usr/bin/env python3
"""Author-floruit anchors from the Sanskrit prosopography.

The Gibbs model only knew dates that were researched per work. An author's floruit was never passed
on to their other works, so a short text by a well-dated author (e.g. Vasubandhu's Pañcaskandhaka)
was dated from stylometry alone. This script writes one soft `anchor` per work from the floruit of
its author, as recorded in the catalog `authors` field (filled from sanskrit-prosopography).

Rules
- slot: attribution == 'secure', role is an authorial role (not translator/editor/quoted/...)
- person: status OK, confidence >= MIN_CONF, both floruit bounds, width <= MAX_WIDTH
- the catalog dates an e-text as its root work (mūla/kārikā/sūtra/author) even when commentaries are
  included, so primary-role slots give the anchor (all of them must be secure and usable)
- only commentator slots and no root text in "Works in this e-text": a pure commentary, dated by
  the commentator; root + commentary: the commentator only gives a not_after for the root
- in the Buddhist shelves (SA_T*, SA_K*) primary "authors" with a floruit after 1850 are modern
  restorers (Sanskrit restorations from Tibetan/Pali) and are skipped
- works with a curated anchor (literature/researched/manual/... files) are skipped here; date_gibbs_full.py
  loads this file FIRST, so curated anchors would override it anyway. Disagreements are reported.

Usage: build_author_anchors.py <SA_files.json> <persons_atii_schema.tsv> [out.tsv] [report.tsv]
"""
import csv, json, re, sys

MIN_CONF = 0.75
MAX_WIDTH = 300
NON_AUTHOR = ('translat', 'editor', 'quoted', 'included extract')
CURATED = ['literature_anchors.tsv', 'dcs_anchors.tsv', 'researched_anchors.tsv', 'chronbmm_priors.tsv',
           'manual_constraints.tsv', 'new_texts_anchors.tsv', 'tibetan_translation_constraints.tsv']

cat_p, pers_p = sys.argv[1], sys.argv[2]
out_p = sys.argv[3] if len(sys.argv) > 3 else 'author_floruit_anchors.tsv'
rep_p = sys.argv[4] if len(sys.argv) > 4 else 'author_floruit_report.tsv'

P = {r['id']: r for r in csv.DictReader(open(pers_p, encoding='utf-8'), delimiter='\t')}
cat = json.load(open(cat_p, encoding='utf-8'))

curated = {}
for f in CURATED:
    try:
        lines = open(f, encoding='utf-8').read().splitlines()[1:]
    except FileNotFoundError:
        continue
    for line in lines:
        p = (line.split('\t') + [''] * 4)[:5]
        if p[0] == 'anchor':
            try:
                curated[p[1]] = (float(p[2]), float(p[3]), f)
            except ValueError:
                pass
taq = {}
for line in open('translation_taq.tsv', encoding='utf-8').read().splitlines()[1:]:
    p = line.split('\t')
    if len(p) > 2 and p[2]:
        try:
            taq[p[0]] = float(p[2])
        except ValueError:
            pass


def usable(pid):
    r = P.get(pid)
    if not r or r['status'] != 'OK':
        return None
    try:
        nb, na, conf = float(r['floruit_not_before']), float(r['floruit_not_after']), float(r['confidence'] or 0)
    except ValueError:
        return None
    if conf < MIN_CONF or na < nb or na - nb > MAX_WIDTH:
        return None
    return nb, na, r


PRIMARY = {'author', 'mūla', 'kārikā', 'sūtra', 'root text', 'compiler', 'redactor'}
MODERN = 1850   # in the Buddhist shelves (SA_T*/SA_K*) a later "author" is a modern restorer, not a dating cue


def kind(role):
    head = re.split(r'[,(]', (role or '').lower())[0].strip()
    return 'primary' if head in PRIMARY else 'commentary'


def has_root(e):
    m = re.search(r'\*\*Works in this e-text:?\*\*(.*?)(?=\n\n\*\*|\n## |$)', e.get('raw_metadata') or '', re.S)
    return bool(m and re.search(r'\(root (text|work)\b|\(mūla\b|\(kārikā\b|\(sūtra\b', m.group(1)))


rows, report = [], []
for e in cat:
    fn = e['filename']
    slots = [a for a in e.get('authors') or [] if not any(k in (a.get('role') or '').lower() for k in NON_AUTHOR)]
    prim = [a for a in slots if kind(a.get('role')) == 'primary']
    comm = [a for a in slots if kind(a.get('role')) == 'commentary']
    anchor = None
    restor = bool(re.match(r'SA_[TK]\d', fn))   # Buddhist shelves: modern "authors" are restorers
    if prim:
        # the e-text is dated as its root/primary work: every primary slot must be secure and usable
        us = [usable(a['id']) if a.get('attribution') == 'secure' else None for a in prim]
        if all(us):
            nb, na = min(u[0] for u in us), max(u[1] for u in us)
            if na - nb <= MAX_WIDTH and (na <= MODERN or not restor):
                anchor = (nb, na, ', '.join(f"{u[2]['label']} ({u[2]['id']}, {a.get('role')})" for u, a in zip(us, prim)))
            elif na > MODERN and restor:
                report.append([fn, 'modern_primary_author_skipped', f'{nb:.0f}-{na:.0f}', '', us[-1][2]['label']])
    elif comm and not has_root(e):
        # a pure commentary e-text: dated by its commentator(s)
        us = [usable(a['id']) if a.get('attribution') == 'secure' else None for a in comm]
        if all(us):
            nb, na = min(u[0] for u in us), max(u[1] for u in us)
            if na - nb <= MAX_WIDTH and (na <= MODERN or not restor):
                anchor = (nb, na, ', '.join(f"{u[2]['label']} ({u[2]['id']}, {a.get('role')})" for u, a in zip(us, comm)))
    # commentators in a root+commentary e-text: the root is older than its earliest secure commentator
    bound = None
    if prim or has_root(e):
        cu = [(usable(a['id']), a) for a in comm if a.get('attribution') == 'secure']
        cu = [(u, a) for u, a in cu if u]
        if cu:
            u, a = min(cu, key=lambda x: x[0][1])
            bound = (u[1], f"{u[2]['label']} ({u[2]['id']}, {a.get('role')})")
    if fn in curated:
        cnb, cna, f = curated[fn]
        if anchor and (cna < anchor[0] - 50 or cnb > anchor[1] + 50):
            report.append([fn, 'curated_disagrees', f'{anchor[0]:.0f}-{anchor[1]:.0f}', f'{cnb:.0f}-{cna:.0f} ({f})', anchor[2]])
        if bound and cnb > bound[0] + 50:
            report.append([fn, 'curated_after_commentator', f'<= {bound[0]:.0f}', f'{cnb:.0f}-{cna:.0f} ({f})', bound[1]])
        continue
    if anchor:
        if fn in taq and taq[fn] < anchor[0]:
            report.append([fn, 'translation_taq_before_floruit', f'{anchor[0]:.0f}-{anchor[1]:.0f}', f'taq {taq[fn]:.0f}', anchor[2]])
        else:
            rows.append(['anchor', fn, f'{anchor[0]:.0f}', f'{anchor[1]:.0f}',
                         f'author floruit: {anchor[2]} | sanskrit-prosopography'])
    elif bound:
        rows.append(['not_after', fn, f'{bound[0]:.0f}', '',
                     f'root text older than its commentator: {bound[1]} | sanskrit-prosopography'])

with open(out_p, 'w', encoding='utf-8') as fh:
    fh.write('type\targ1\targ2\targ3\tnote\n')
    for x in rows:
        fh.write('\t'.join(x) + '\n')
with open(rep_p, 'w', encoding='utf-8') as fh:
    fh.write('work\tissue\tfloruit\tother\tperson\n')
    for x in report:
        fh.write('\t'.join(x) + '\n')
print(f'{len(rows)} author anchors -> {out_p}; {len(report)} reported -> {rep_p}', file=sys.stderr)
