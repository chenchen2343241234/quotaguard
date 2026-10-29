"""Render the four paper figures from portable numerical summaries."""
import argparse
import json
from statistics import mean, stdev
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'figures'
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.2})
WIDTH = 3.33
BLUE, ORANGE, GREY = ('#1f5fa8', '#d9731a', '#7a7a7a')

def load(name):
    """Read portable numerical tables in the shape required by each plot."""
    rows = json.loads((REPO / 'results' / (name + '.json')).read_text(encoding='utf-8'))
    if name == 'candidate_coverage':
        report = {'domains': {}}
        for row in rows:
            if row['candidate_rule'] != 'union':
                continue
            mode = report['domains'].setdefault(row['catalogue'], {'text_modes': {}})['text_modes'].setdefault(row['text_mode'], {'limits': {}})
            mode['limits'][str(row['neighbor_limit'])] = {'union': {'coverage': row['coverage']}}
        return report
    if name == 'paired_comparisons':
        names = {'candidate_graph-public_groups': 'cg-lg', 'candidate_graph-candidate_graph_raw_accountant': 'cg-cgcons'}
        return {'descriptive': {names[row['comparison']] + '/H1/batch_mrr/' + row['direction']: {'differences': row['differences']} for row in rows if row['field'] == 'CS' and row['epsilon'] == 10 and (row['negatives'] == 'hard') and (row['metric'] == 'batch_mrr') and (row['comparison'] in names)}}
    if name == 'construction_scaling':
        return {'rows': rows}
    if name == 'per_run_metrics':
        names = {'candidate_graph': 'cg', 'public_groups': 'lg'}
        means = {epsilon: {} for epsilon in (1, 4, 10)}
        for epsilon in means:
            for direction in ('CN-US', 'US-CN'):
                for structure, label in names.items():
                    selected = sorted((row for row in rows if row['field'] == 'CS' and row['epsilon'] == epsilon and (row['structure'] == structure) and (row['direction'] == direction) and (row['negatives'] == 'hard')), key=lambda row: row['seed'])
                    if len(selected) != 5:
                        raise ValueError('Each curve point requires five training seeds')
                    means[epsilon][label + '/H1/' + direction + '/batch_mrr'] = mean((row['batch_mrr'] for row in selected))
        return means
    raise ValueError('Unknown numerical summary: ' + name)

def coverage():
    report = load('candidate_coverage')
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    limits = ['16', '64', '256']
    for mode, colour, label in (('title', GREY, 'Titles'), ('title_abstract', BLUE, 'Titles + abstracts')):
        for domain, marker, style in (('CN', 'o', '-'), ('US', 's', '--')):
            values = [100 * report['domains'][domain]['text_modes'][mode]['limits'][L]['union']['coverage'] for L in limits]
            ax.plot([int(L) for L in limits], values, style, marker=marker, color=colour, markersize=3.5, label=f'{label}, {domain}')
    ax.set_xscale('log', base=2)
    ax.set_xticks([16, 64, 256], ['16', '64', '256'])
    ax.set_xlabel('Neighbours per entity $L$')
    ax.set_ylabel('Relations among\ncandidates (%)')
    ax.set_ylim(0, 75)
    ax.legend(frameon=False, ncol=2, loc='upper left', handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    return fig

def decomposition():
    report = load('paired_comparisons')
    parts = [('cg-cgcons', 'Accounting', ORANGE), ('cgcons-lg', 'Structure', BLUE), ('cg-lg', 'Total', 'black')]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.6), sharey=True)
    for ax, direction, title in zip(axes, ('CN-US', 'US-CN'), ('CN$\\to$US', 'US$\\to$CN')):
        for y, (key, label, colour) in enumerate(parts):
            if key == 'cgcons-lg':
                cg, cons = (report['descriptive'][f'{k}/H1/batch_mrr/{direction}']['differences'] for k in ('cg-lg', 'cg-cgcons'))
                diffs = [a - b for a, b in zip(cg, cons)]
            else:
                diffs = report['descriptive'][f'{key}/H1/batch_mrr/{direction}']['differences']
            m = 100 * mean(diffs)
            half = 100 * 2.7764451051977934 * stdev(diffs) / len(diffs) ** 0.5
            ax.barh(y, m, color=colour, alpha=0.25 if key != 'cg-lg' else 0.12, height=0.6)
            ax.errorbar(m, y, xerr=half, fmt='none', ecolor=colour, capsize=2, linewidth=1)
            ax.scatter([100 * d for d in diffs], [y] * len(diffs), s=6, color=colour, zorder=3)
        ax.axvline(0, color=GREY, linewidth=0.6)
        ax.set_title(title, fontsize=8)
        ax.set_xlim(0, 2.6)
    axes[0].set_yticks(range(len(parts)), [p[1] for p in parts])
    axes[0].invert_yaxis()
    fig.supxlabel('Batch MRR gain of the candidate graph (points)', fontsize=8, y=0.02)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    return fig

def scaling():
    rows = load('construction_scaling')['rows']
    n = [r['n'] for r in rows]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    for key, label, colour, marker in (('tfidf_knn_seconds', 'Neighbour search', BLUE, 'o'), ('bmatching_seconds', 'Greedy selection', ORANGE, 's'), ('predict_seconds', 'Pair scoring', GREY, '^')):
        ax.plot(n, [r[key] for r in rows], '-', marker=marker, color=colour, markersize=3.5, label=label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([125000.0, 250000.0, 500000.0, 1000000.0], ['125k', '250k', '500k', '1M'])
    ax.minorticks_off()
    ax.set_ylim(0.2, 1000000.0)
    ax.set_xlabel('Public entities')
    ax.set_ylabel('Time (s)')
    ax.legend(frameon=False, loc='upper left', ncol=2)
    fig.tight_layout(pad=0.3)
    return fig

def privacy():
    means = load('per_run_metrics')
    eps = [1, 4, 10]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.7), sharey=False)
    for ax, direction, title in zip(axes, ('CN-US', 'US-CN'), ('CN$\\to$US', 'US$\\to$CN')):
        for kind, label, colour, marker in (('cg', 'Candidate graph', BLUE, 'o'), ('lg', 'Public groups', GREY, 's')):
            ax.plot(eps, [100 * means[e][f'{kind}/H1/{direction}/batch_mrr'] for e in eps], '-', marker=marker, color=colour, markersize=3.5, label=label)
        ax.set_xscale('log')
        ax.set_xticks(eps, ['1', '4', '10'])
        ax.minorticks_off()
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('$\\varepsilon$')
    axes[0].set_ylabel('Batch MRR ($\\times100$)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.86))
    return fig
FIGURES = {'coverage': coverage, 'decomposition': decomposition, 'scaling': scaling, 'privacy': privacy}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='*')
    names = parser.parse_args().only or list(FIGURES)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        fig = FIGURES[name]()
        fig.savefig(OUT / f'{name}.pdf')
        plt.close(fig)
        print(OUT / f'{name}.pdf')
if __name__ == '__main__':
    main()
