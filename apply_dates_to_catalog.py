#!/usr/bin/env python3
"""Write dated_gibbs_full.tsv results into SA_files.json (not_before/not_after + '## Date estimate').

Only entries whose numbers change are touched. The section format is date_section() in
dharmanexus-sanskrit utils/provenance/apply.py. JSON layout (indent 2, ensure_ascii False) is kept.

Usage: apply_dates_to_catalog.py <dated_gibbs_full.tsv> <SA_files.json> [changes.tsv]
"""
import json, re, sys

dated_p, cat_p = sys.argv[1], sys.argv[2]
chg_p = sys.argv[3] if len(sys.argv) > 3 else None
D = {}
for line in open(dated_p, encoding='utf-8').read().splitlines()[1:]:
    p = line.split('\t')
    D[p[0]] = p


def year(v):
    y = round(float(v))
    return f'{-y} BCE' if y < 0 else f'{y} CE'


def section(r):
    return ('## Date estimate ([Statistical Model](https://dharmamitra.github.io/sanskrit-dating/'
            'sanskrit_chronology_interactive.html))\n\n'
            f"**95% credible interval:** {year(r[7])} – {year(r[8])}\n\n"
            f"**Posterior median:** {year(r[6])}\n")


SEC = re.compile(r'## Date estimate \(\[Statistical Model\]\([^)]*\)\)\n\n\*\*95% credible interval:\*\* [^\n]*\n\n'
                 r'\*\*Posterior median:\*\* [^\n]*\n')
src = open(cat_p, encoding='utf-8').read()
cat = json.loads(src)
changes = []
for e in cat:
    r = D.get(e['filename'])
    if not r:
        continue
    nb, na = round(float(r[7])), round(float(r[8]))
    rm = e.get('raw_metadata') or ''
    new_sec = section(r)
    m = SEC.search(rm)
    if m:
        rm2 = rm[:m.start()] + new_sec + rm[m.end():]
    elif '## Date estimate' in rm:
        print('unparsed date section:', e['filename'], file=sys.stderr)
        continue
    else:
        rm2 = rm.rstrip('\n') + '\n\n' + new_sec if rm else rm
    if e.get('not_before') == nb and e.get('not_after') == na and rm2 == rm:
        continue
    changes.append([e['filename'], str(e.get('not_before')), str(e.get('not_after')), str(nb), str(na), r[6], r[1]])
    e['not_before'], e['not_after'] = nb, na
    if rm:
        e['raw_metadata'] = rm2
open(cat_p, 'w', encoding='utf-8').write(json.dumps(cat, ensure_ascii=False, indent=2) + ('\n' if src.endswith('\n') else ''))
if chg_p:
    with open(chg_p, 'w', encoding='utf-8') as fh:
        fh.write('filename\told_nb\told_na\tnew_nb\tnew_na\tnew_median\tsource\n')
        for c in changes:
            fh.write('\t'.join(c) + '\n')
print(f'{len(changes)} entries updated', file=sys.stderr)
