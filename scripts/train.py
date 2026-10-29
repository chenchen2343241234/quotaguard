import argparse
import sys
from collections import deque
from contextlib import contextmanager
import copy
if sys.platform != 'win32':
    import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
_RUNNER_LOCK_FD = None
_COORDINATOR_STOP = threading.Event()
PRIVATE_ARMS = frozenset({'R1', 'H1', 'P1', 'normalized_uniform', 'normalized_hard', 'B1'})
ARMS = PRIVATE_ARMS | {'R0', 'H0'}

def _watch_parent():
    expected = os.environ.get('QUOTAGUARD_PARENT_PID')
    if expected is None:
        return
    expected = int(expected)

    def monitor():
        while os.getppid() == expected:
            time.sleep(0.1)
        os.killpg(os.getpid(), signal.SIGKILL)
    if os.getppid() != expected:
        os.killpg(os.getpid(), signal.SIGKILL)
    threading.Thread(target=monitor, daemon=True).start()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)

def calibrate_sigma(*, epsilon, delta, steps, gamma, kdeg, tolerance=1e-07):
    from opacus.accountants import RDPAccountant
    from quotaguard.train.dp import entity_sample_rate
    version = importlib.metadata.version('opacus')
    if version != '1.5.4':
        raise ValueError(f'Expected Opacus 1.5.4, found {version}')
    if not (math.isfinite(epsilon) and epsilon > 0 and (0 < delta < 1) and isinstance(steps, int) and (steps > 0) and (0 < tolerance < 1)):
        raise ValueError('Invalid calibration arguments')
    q = entity_sample_rate(gamma, kdeg)

    def account(sigma):
        accountant = RDPAccountant()
        accountant.history = [(sigma, q, steps)]
        return accountant.get_epsilon(delta)
    low, high = (0.0, 1.0)
    while account(high) > epsilon:
        high *= 2
        if high > 100000000.0:
            raise ValueError('Cannot bracket sigma')
    while high - low > tolerance * max(1.0, high):
        middle = (high + low) / 2
        if account(middle) <= epsilon:
            high = middle
        else:
            low = middle
    return dict(sigma=high, epsilon=account(high), delta=delta, sample_rate=q, steps=steps, accountant='Opacus RDPAccountant', accountant_version=version, sigma_infeasible_lower=low, sigma_tolerance=tolerance)

def prepare(config, source, target):
    if config.get('stable_truncation') is True:
        from quotaguard.train.stable import prepare_stable
        return prepare_stable(config, source, target)
    from quotaguard.train.data import load_domain, clip_degree
    cfg = copy.deepcopy(config)
    cfg.update(source=source, target=target)
    data = load_domain(cfg['data_dir'], source, 'train', role='training')
    clipped = clip_degree(data.relations, len(data.ids), cfg['kdeg'], seed=cfg['clip_seed'])
    n = len(clipped)
    if n <= 1:
        raise ValueError('At least two clipped training relations required')
    if 'train_fraction' in cfg:
        import shutil
        from fractions import Fraction
        import torch
        from quotaguard.train.data import read_manifest_tsv
        p, fraction_seed = (cfg['train_fraction'], cfg['train_fraction_seed'])
        if type(p) is not float or not 0 < p < 1 or type(fraction_seed) is not int or (cfg.get('noise_calibration') != 'mog') or ('caseB' in cfg) or ('public_train_size_override' in cfg) or ('delta_rule_v2' not in cfg):
            raise ValueError('train_fraction requires 0<p<1, an integer seed, MoG case A and delta_rule_v2')
        root = Path(cfg['data_dir']).resolve()
        manifest_bytes = (root / 'manifest.json').read_bytes()
        manifest = json.loads(manifest_bytes)
        relative = f'{source}/splits/train.tsv'
        rows = read_manifest_tsv(root, manifest, relative)
        row_of = {tuple(pair): i for i, pair in enumerate(data.relations.tolist())}
        if len(rows) != len(data.relations) or len(row_of) != len(rows):
            raise ValueError('Source train rows do not match loaded relations')
        k = math.floor(Fraction(repr(p)) * n)
        if k <= 1:
            raise ValueError('train_fraction leaves fewer than two training relations')
        chosen = torch.randperm(n, generator=torch.Generator().manual_seed(fraction_seed))[:k].sort().values
        subset = clipped[chosen]
        payload = ''.join(('\t'.join(rows[row_of[tuple(pair)]]) + '\n' for pair in subset.tolist())).encode()
        record = dict(train_fraction=p, train_fraction_seed=fraction_seed, source=source, kdeg=cfg['kdeg'], clip_seed=cfg['clip_seed'], raw_train_relations=len(data.relations), clipped_relations_before=n, train_relations_after=k, rule='k=floor(p*clipped count); torch.randperm(clipped count, seed)[:k] sorted; clipped order kept', source_data_dir=str(root), source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), train_tsv_sha256=hashlib.sha256(payload).hexdigest())
        derived_manifest = copy.deepcopy(manifest)
        derived_manifest['sha256'][relative] = record['train_tsv_sha256']
        if isinstance(derived_manifest.get('splits', {}).get(source), dict):
            derived_manifest['splits'][source]['train'] = k
        derived_manifest['train_fraction_derivation'] = record
        body = (json.dumps(derived_manifest, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        folder = Path(os.environ.get('QUOTAGUARD_OUTPUT_DIR', cfg['output_dir'])).resolve() / 'train_fraction_data' / f'{source}_{record["source_manifest_sha256"][:12]}_k{cfg["kdeg"]}_c{cfg["clip_seed"]}_p{p!r}_s{fraction_seed}'
        if not folder.exists():
            temporary = folder.with_name(folder.name + f'.tmp{os.getpid()}')
            shutil.rmtree(temporary, ignore_errors=True)
            for name in manifest['sha256']:
                path = temporary / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if name == relative:
                    path.write_bytes(payload)
                else:
                    path.symlink_to(root / name)
            (temporary / 'manifest.json').write_bytes(body)
            temporary.rename(folder)
        if (folder / 'manifest.json').read_bytes() != body:
            raise ValueError('Existing train_fraction dataset differs from this derivation')
        derived = load_domain(folder, source, 'train', role='training')
        if derived.ids != data.ids or not torch.equal(derived.relations, subset) or (not torch.equal(clip_degree(derived.relations, len(derived.ids), cfg['kdeg'], seed=cfg['clip_seed']), subset)):
            raise ValueError('Training loader does not reproduce the train_fraction subset')
        record['derived_data_dir'] = str(folder)
        cfg.update(data_dir=str(folder), train_fraction_record=record)
        n = k
    clipped_n = n
    if 'public_train_size_override' in cfg:
        bound = cfg['public_train_size_override']
        bound = bound[source] if isinstance(bound, dict) else bound
        if type(bound) is not int or bound < n:
            raise ValueError('Clipped training relation count exceeds public_train_size_override')
        n = bound
    cfg.update(public_train_size=n, gamma=min(1.0, cfg['expected_batch_size'] / n), delta=1 / n)
    if cfg.get('noise_calibration') == 'inbatch':
        cfg['delta'] = 1 / clipped_n
    extra = {}
    universe = len(data.ids)
    if 'entity_universe_override' in cfg:
        universe = cfg['entity_universe_override'][source]
        if type(universe) is not int or not 5 <= universe <= len(data.ids):
            raise ValueError('entity_universe_override must be an integer public lower bound on the source entity count')
    if cfg.get('noise_calibration') == 'caseB':
        from quotaguard.train.samplers import bounded_parameters
        bounds = cfg['caseB']
        if bounds.get('derive_from_source_train') is True:
            m_bar, N_min = (n, universe - 3)
        else:
            m_bar, N_min = (bounds['m_bar'], bounds['N_min'])
        if 'public_train_size_override' in cfg and m_bar != n:
            raise ValueError('Case B m_bar must equal public_train_size_override')
        if type(m_bar) is not int or type(N_min) is not int or m_bar < n or (N_min < 2) or (N_min > len(data.ids) - 2):
            raise ValueError('Data exceed configured Case B public scale bounds')
        R = bounded_parameters(cfg['bounded'])[0] if 'bounded' in cfg else math.exp(2 / cfg['tau_sampler'])
        rho = R / (R + N_min - 1)
        cfg['caseB'] = dict(m_bar=m_bar, N_min=N_min, rho=rho)
        cfg.update(public_train_size=m_bar, gamma=min(1.0, cfg['expected_batch_size'] / m_bar), delta=1 / m_bar)
        extra = dict(k_neg=cfg['k_neg'], **cfg['caseB'])
        if cfg.get('arm_overrides'):
            extra['tolerance'] = 1e-10
    if 'delta_rule_v2' in cfg:
        entities = universe
        if 'delta_entities_override' in cfg:
            entities = cfg['delta_entities_override'][source]
            if type(entities) is not int or entities < len(data.ids):
                raise ValueError('delta_entities_override must be an integer >= actual source entity count')
        cfg['delta'] = 1 / entities ** cfg['delta_exponent']
        cfg['delta_rule_entities'] = entities
    cfg['normalization_source'] = 'fixed clipped source train relation count * gamma'
    if 'public_train_size_override' in cfg:
        cfg['normalization_source'] = 'fixed public_train_size_override * gamma'
    if 'train_fraction' in cfg:
        cfg['normalization_source'] = 'fixed train_fraction subset of clipped source train relation count * gamma'
        cfg['train_fraction_record'].update(public_train_size=cfg['public_train_size'], gamma=cfg['gamma'], public_M=cfg['gamma'] * cfg['public_train_size'], delta=cfg['delta'])
    calibration = calibrate_sigma
    if cfg.get('noise_calibration') == 'mog':
        from quotaguard.train.dp import calibrate_mog
        calibration = calibrate_mog
    elif cfg.get('noise_calibration') == 'inbatch_amp':
        from quotaguard.train.dp import calibrate_inbatch_amp
        calibration = calibrate_inbatch_amp
    elif cfg.get('noise_calibration') == 'caseB':
        from quotaguard.train.dp import calibrate_caseb
        calibration = calibrate_caseb
    if cfg.get('noise_calibration') == 'inbatch':
        from quotaguard.train.dp import calibrate_inbatch
        cfg['calibration'] = calibrate_inbatch(epsilon=cfg['epsilon_budget'], delta=cfg['delta'], steps=cfg['steps'])
    else:
        cfg['calibration'] = calibration(epsilon=cfg['epsilon_budget'], delta=cfg['delta'], steps=cfg['steps'], gamma=cfg['gamma'], kdeg=cfg['kdeg'], **extra)
    cfg['sigma'] = cfg['calibration']['sigma']
    cfg['manifest_sha256'] = hashlib.sha256((Path(cfg['data_dir']) / 'manifest.json').read_bytes()).hexdigest()
    return cfg

def analysis_pairs(cfg):
    analysis = cfg.get('analysis', {})
    primary = analysis.get('primary_pair', ['H1', 'R1'])
    descriptive = analysis.get('descriptive_pairs', [['H0', 'R0']])
    if not isinstance(descriptive, list):
        raise ValueError('descriptive_pairs must be a list of arm pairs')
    pairs = [primary, *descriptive]
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2 or any((not isinstance(arm, str) or arm not in cfg['arms'] for arm in pair)) or (pair[0] == pair[1]):
            raise ValueError('Analysis pairs require two distinct configured arms')
    if len({tuple(pair) for pair in pairs}) != len(pairs):
        raise ValueError('Duplicate analysis pair')
    return (primary, descriptive)

def _paired_directions(directions):
    return isinstance(directions, list) and len(directions) == 2 and all((isinstance(d, list) and len(d) == 2 and all((isinstance(x, str) and x for x in d)) for d in directions)) and (directions[0][0] != directions[0][1]) and (directions[1] == directions[0][::-1])

def validate_config(cfg, mode):
    from quotaguard.train.model import validate_model_options
    if 'stable_truncation' in cfg:
        from quotaguard.train.stable import validate_stable_options
        validate_stable_options(cfg)
    validate_model_options(cfg.get('model', {}))
    if {'negative_sampler', 'bm25_topn', 'bm25'} & set(cfg):
        from quotaguard.train.train import validate_screen_options
        if mode not in {'confirm', 'diagnose'}:
            raise ValueError('bm25_topn requires confirm or diagnose mode')
        for arm in cfg['arms']:
            validate_screen_options(dict(cfg, **cfg.get('arm_overrides', {}).get(arm, {})), arm)
    if {'text_fields', 'max_length'} & set(cfg):
        raise ValueError('text_fields and max_length belong under model')
    if 'abstract' in cfg.get('model', {}).get('text_fields', ()):
        if mode != 'diagnose' or cfg.get('exploratory') is not True:
            raise ValueError('Abstract text_fields requires exploratory=true and diagnose mode')
        from quotaguard.train.data import abstract_missing_rates
        domains = sorted({domain for pair in cfg['directions'] for domain in pair})
        rates = abstract_missing_rates(cfg['data_dir'], domains)
        recorded = json.loads((Path(cfg['data_dir']) / 'manifest.json').read_text())['domains']
        if cfg.get('abstract_missing_rate') != rates or any((recorded[d].get('abstract_missing_rate', rates[d]) != rates[d] for d in domains)):
            raise ValueError(f'abstract_missing_rate must equal the data_dir rates {rates}')
    elif 'abstract_missing_rate' in cfg:
        raise ValueError('abstract_missing_rate requires abstract text_fields')
    if 'delta_rule_v2' in cfg or 'delta_exponent' in cfg:
        exponent = cfg.get('delta_exponent')
        if cfg.get('delta_rule_v2') != 'entity_power' or type(exponent) not in (int, float) or (not math.isfinite(exponent)) or (exponent <= 1):
            raise ValueError('delta_rule_v2 requires entity_power with a finite delta_exponent > 1')
        if cfg.get('stable_truncation') or any((dict(cfg, **o).get('noise_calibration', cfg.get('noise_calibration')) not in {'mog', 'caseB'} for o in [{}, *cfg.get('arm_overrides', {}).values()])):
            raise ValueError('delta_rule_v2 supports MoG case A and case B without stable truncation')
    if 'train_fraction' in cfg or 'train_fraction_seed' in cfg:
        p = cfg.get('train_fraction')
        if type(p) is not float or not math.isfinite(p) or (not 0 < p < 1):
            raise ValueError('train_fraction must be a float with 0 < p < 1')
        if type(cfg.get('train_fraction_seed')) is not int:
            raise ValueError('train_fraction requires a fixed integer train_fraction_seed')
        if mode != 'diagnose' or cfg.get('exploratory') is not True:
            raise ValueError('train_fraction requires exploratory=true and diagnose mode')
        overrides = cfg.get('arm_overrides', {})
        if cfg.get('stable_truncation') or 'caseB' in cfg or 'public_train_size_override' in cfg or (not isinstance(overrides, dict)) or any((not isinstance(o, dict) or 'caseB' in o or dict(cfg, **o).get('noise_calibration') != 'mog' for o in [{}, *overrides.values()])):
            raise ValueError('train_fraction supports MoG case A only (no caseB, stable_truncation or public_train_size_override)')
        if 'delta_rule_v2' not in cfg:
            raise ValueError('train_fraction requires delta_rule_v2 (delta independent of p)')
        if not isinstance(cfg.get('output_dir'), str) or not cfg['output_dir']:
            raise ValueError('train_fraction requires output_dir for the derived training data')
    if 'ann' in cfg:
        import quotaguard.train.train as train_module
        from quotaguard.train.samplers import ann_parameters
        params = ann_parameters(cfg['ann'])
        if cfg.get('exploratory') is not True or mode not in {'screen', 'diagnose'}:
            raise ValueError('ann requires exploratory=true and screen or diagnose mode')
        if cfg['arms'] != ['H1']:
            raise ValueError('ann supports the H1 hard arm only')
        if params['rebuild_every'] not in (None, cfg.get('refresh_every')):
            raise ValueError('ann rebuild_every must equal refresh_every (index follows the cache)')
        candidates = [{}, *cfg.get('arm_overrides', {}).values(), *(v.get('overrides', {}) for v in cfg.get('variants', []))]
        if cfg.get('stable_truncation') or cfg.get('model', {}).get('kind') == 'residual_e5' or any((not isinstance(o, dict) or {'bounded', 'caseB', 'hard_filter_margin', 'hard_mix'} & set(dict(cfg, **o)) or dict(cfg, **o).get('noise_calibration') != 'mog' for o in candidates)):
            raise ValueError('ann is case A only: MoG, no bounded/caseB, filtering, mixing, stable truncation or residual query caches')
        if getattr(train_module, 'ANN_FORWARDING', False) is not True:
            raise ValueError("ann needs Trainer to forward cfg['ann'] to sample_negatives (train.py integration pending); refusing a silent exact run")
    if cfg.get('learning_rate') == 'unset':
        raise ValueError('unset: fill learning_rate from source-only screen before running')
    if 'hard_cache' in cfg:
        from quotaguard.train.train import validate_screen_options
        for arm in cfg['arms']:
            validate_screen_options(dict(cfg, phase=mode), arm)
    if 'full_corpus_eval' in cfg and type(cfg['full_corpus_eval']) is not bool:
        raise ValueError('full_corpus_eval must be a boolean')
    for key in ('full_corpus_query_block_size', 'full_corpus_candidate_block_size'):
        if key in cfg and (type(cfg[key]) is not int or cfg[key] <= 0):
            raise ValueError(f'{key} must be a positive integer')
    if 'public_train_size_override' in cfg:
        if mode != 'confirm' and (mode != 'diagnose' or cfg.get('exploratory') is not True):
            raise ValueError('public_train_size_override requires exploratory=true and diagnose mode, or confirm mode')
        bound = cfg['public_train_size_override']
        values = bound.values() if isinstance(bound, dict) else [bound]
        if isinstance(bound, dict) and set(bound) != {s for s, _ in cfg['directions']} or any((type(value) is not int or value <= 0 for value in values)):
            raise ValueError('public_train_size_override requires positive integers for every source domain')
    if 'entity_universe_override' in cfg:
        if mode != 'confirm' and (mode != 'diagnose' or cfg.get('exploratory') is not True):
            raise ValueError('entity_universe_override requires exploratory=true and diagnose mode, or confirm mode')
        bound = cfg['entity_universe_override']
        if not isinstance(bound, dict) or set(bound) != {s for s, _ in cfg['directions']} or any((type(value) is not int or value < 5 for value in bound.values())):
            raise ValueError('entity_universe_override requires an integer >= 5 for every source domain')
        overrides = cfg.get('arm_overrides', {})
        if cfg.get('stable_truncation') or 'train_fraction' in cfg or (not isinstance(overrides, dict)) or (not isinstance(cfg.get('arms'), list)) or any((not isinstance(overrides.get(arm, {}), dict) or dict(cfg, **overrides.get(arm, {})).get('noise_calibration') not in {'mog', 'caseB'} or (dict(cfg, **overrides.get(arm, {})).get('noise_calibration') == 'caseB' and dict(cfg, **overrides.get(arm, {})).get('caseB') != {'derive_from_source_train': True}) for arm in cfg['arms'])):
            raise ValueError('entity_universe_override requires MoG or case B with derive_from_source_train for every arm (no stable truncation or train_fraction)')
        if 'delta_entities_override' not in cfg and any((dict(cfg, **overrides.get(arm, {})).get('noise_calibration') == 'mog' for arm in cfg['arms'])):
            raise ValueError('MoG entity_universe_override requires delta_entities_override so delta uses a separate public upper bound')
    if 'delta_entities_override' in cfg:
        bound = cfg['delta_entities_override']
        if mode not in {'confirm', 'diagnose'} or cfg.get('delta_rule_v2') != 'entity_power' or (not isinstance(bound, dict)) or (set(bound) != {s for s, _ in cfg['directions']}) or any((type(v) is not int or v < 1 for v in bound.values())):
            raise ValueError('delta_entities_override requires entity_power and a positive integer per source in confirm or diagnose mode')
    if cfg.get('noise_calibration', 'opacus') not in {'opacus', 'mog', 'caseB', 'inbatch', 'inbatch_amp'}:
        raise ValueError('Unknown noise calibration')
    if cfg.get('noise_calibration') == 'caseB' and mode not in {'screen', 'confirm', 'diagnose'}:
        raise ValueError('Case B requires screen, confirm or diagnose mode')
    seeds = cfg['training_seeds']
    if not seeds or len(seeds) != len(set(seeds)) or cfg['clip_seed'] == cfg['eval_seed'] or {cfg['clip_seed'], cfg['eval_seed']} & set(seeds):
        raise ValueError('Training, clipping and evaluation seeds must be distinct')
    if not 0 <= cfg['max_restarts'] <= 2:
        raise ValueError('At most two restarts')
    if min(cfg['run_timeout_seconds'], cfg['wall_time_limit_seconds'], cfg['process_gpu_hours_limit']) <= 0:
        raise ValueError('Positive timeout/cost limits required')
    if isinstance(cfg['epsilon_budget'], bool) or cfg['epsilon_budget'] not in (1, 2, 4, 8, 10) or type(cfg['steps']) is not int or (cfg['steps'] < 1) or (mode not in {'confirm', 'diagnose'} and cfg['steps'] != 2000):
        raise ValueError('Requires epsilon in {1, 2, 4, 8, 10} and positive integer steps (T=2000 outside confirm/diagnose)')
    if cfg['epsilon_budget'] not in (1, 4, 10) and (mode != 'diagnose' or cfg.get('exploratory') is not True):
        raise ValueError('Epsilon outside {4, 10} requires exploratory=true and diagnose mode')
    arms = cfg['arms']
    if not isinstance(arms, list) or not arms or any((not isinstance(arm, str) or arm not in ARMS for arm in arms)) or (len(set(arms)) != len(arms)):
        raise ValueError('arms must be a nonempty unique subset of R0, H0, R1, H1, P1, structure, training, B1')
    if cfg.get('model', {}).get('kind') == 'residual_e5':
        from quotaguard.train.train import validate_screen_options
        for arm in arms:
            validate_screen_options(cfg, arm)
    if 'arm_overrides' in cfg:
        from quotaguard.train.samplers import bounded_parameters
        from quotaguard.train.train import validate_screen_options
        overrides = cfg['arm_overrides']
        if mode not in {'confirm', 'diagnose'} or not isinstance(overrides, dict) or set(overrides) - set(arms):
            raise ValueError('arm_overrides requires confirm or diagnose mode and configured arms')
        for arm in arms:
            override = overrides.get(arm, {})
            if not isinstance(override, dict) or set(override) - {'bounded', 'noise_calibration', 'caseB', 'learning_rate', 'negative_sampler', 'bm25_topn', 'bm25'}:
                raise ValueError('Invalid arm_overrides keys')
            if 'learning_rate' in override:
                lr = override['learning_rate']
                if type(lr) is not float or not math.isfinite(lr) or lr <= 0:
                    raise ValueError('arm_overrides learning_rate must be a positive finite float')
            candidate = dict(cfg, **override)
            validate_screen_options(dict(candidate, phase='diagnose' if mode == 'diagnose' else 'train'), arm)
            if candidate.get('noise_calibration') == 'caseB':
                bounds = candidate.get('caseB')
                if not isinstance(bounds, dict) or not (bounds == {'derive_from_source_train': True} or (set(bounds) == {'m_bar', 'N_min'} and all((type(bounds[k]) is int and bounds[k] >= 2 for k in bounds)))):
                    raise ValueError('Case B requires source-train derivation or explicit public bounds')
                if arm == 'R1' and bounded_parameters(candidate['bounded'])[0] != 1:
                    raise ValueError('Case B uniform arm requires bounded R=1')
    if 'stable_truncation' in cfg:
        for arm in arms:
            validate_stable_options(dict(cfg, **cfg.get('arm_overrides', {}).get(arm, {})), arm)
    if 'B1' in arms or cfg.get('noise_calibration') in {'inbatch', 'inbatch_amp'}:
        from quotaguard.train.train import validate_screen_options
        for arm in arms:
            candidate = dict(cfg, **cfg.get('arm_overrides', {}).get(arm, {}))
            validate_screen_options(dict(candidate, phase=mode), arm)
    if mode == 'confirm' and cfg.get('noise_calibration') == 'caseB' and (not cfg.get('arm_overrides')):
        raise ValueError('Case B confirmation requires arm_overrides for per-arm calibration')
    if mode == 'confirm':
        if len(seeds) != 5 or not _paired_directions(cfg['directions']) or cfg['analysis'].get('delta_min', 0.02) not in (0.01, 0.02):
            raise ValueError('Confirmation requires five seeds, both directions and delta_min in {.01, .02}')
        analysis_pairs(cfg)
    elif mode == 'explore':
        if len(seeds) != 1 or cfg['directions'] != [['CN', 'US']]:
            raise ValueError('Exploration requires one seed and CN→US only')
        if cfg['learning_rates'] != [0.0001, 0.0003, 0.001] or cfg['sampler_temperatures'] != [0.05, 0.1, 0.2]:
            raise ValueError('Exploration grid must match configuration')
    elif mode == 'screen':
        from quotaguard.train.train import validate_screen_options
        if cfg.get('exploratory') is not True or cfg['directions'] != [['CN', 'US']]:
            raise ValueError('Screen requires exploratory=true and CN→US only')
        if any((type(seed) is not int for seed in seeds)):
            raise ValueError('Screen requires fixed integer training seeds')
        variants = cfg.get('variants')
        if not isinstance(variants, list) or not variants:
            raise ValueError('Screen requires variants')
        names = set()
        allowed = {'hard_filter_margin', 'optimizer', 'adambc_floor', 'hard_mix', 'learning_rate', 'noise_calibration', 'bounded', 'tau_sampler'}
        for variant in variants:
            name = variant.get('name')
            if not isinstance(name, str) or not name or (not name.isascii()) or (not name.replace('_', '').isalnum()) or (name in names):
                raise ValueError('Screen variant names must be unique safe identifiers')
            names.add(name)
            if variant.get('arm') not in cfg['arms']:
                raise ValueError('Screen variant arm must be configured')
            overrides = variant.get('overrides')
            if not isinstance(overrides, dict) or set(overrides) - allowed:
                raise ValueError('Invalid screen overrides')
            candidate = dict(cfg, **overrides)
            if not math.isfinite(candidate['learning_rate']) or candidate['learning_rate'] <= 0:
                raise ValueError('Positive finite learning rate required')
            validate_screen_options(dict(candidate, phase='screen'), variant['arm'])
    elif mode == 'diagnose':
        if cfg.get('exploratory') is not True:
            raise ValueError('Diagnosis requires exploratory=true')
        single = 'abstract' in cfg.get('model', {}).get('text_fields', ()) and cfg['directions'] == [['CN', 'US']]
        if not (_paired_directions(cfg['directions']) or single):
            raise ValueError('Diagnosis requires both directions of one domain pair')
        if any((type(seed) is not int for seed in seeds)):
            raise ValueError('Diagnosis requires fixed integer training seeds')
        if cfg.get('label_shuffle') and type(cfg.get('label_shuffle_seed')) is not int:
            raise ValueError('label_shuffle_seed must be a fixed integer')
        if cfg.get('noise_calibration') == 'caseB' and (not cfg.get('arm_overrides')):
            from quotaguard.train.train import validate_screen_options
            for arm in arms:
                validate_screen_options(dict(cfg, phase='diagnose'), arm)

def task_for(cfg, arm, seed, *, phase='train', suffix=''):
    run = copy.deepcopy(cfg)
    run.update(arm=arm, seed=seed, phase=phase, retain_records=False)
    name = f'{run["source"]}-{run["target"]}_{arm}_s{seed}{suffix}'
    run['run_id'] = name
    run['config_hash'] = digest(run)
    return run

def completed(folder, task):
    path = folder / 'result.json'
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    if result.get('config_hash') != task['config_hash']:
        raise ValueError(f'Result/config mismatch: {folder}')
    if task['phase'] == 'screen':
        return result.get('status') == 'complete' and result.get('exploratory') is True and (result.get('target_test_accessed') is False) and ('target_test' not in result) and ('source_validation' in result) and (folder / 'checkpoint.pt').exists()
    if task['phase'] == 'diagnose':
        return result.get('status') == 'complete' and result.get('exploratory') is True and (folder / 'checkpoint.pt').exists() and ('source_validation' in result) and ('target_test' in result) and (not task.get('log_negative_stats') or (folder / 'negative_stats.json').exists())
    return result.get('status') == 'complete' and (task['phase'] != 'train' or (folder / 'checkpoint.pt').exists())

def diagnose_metric(folder, name, task, evaluate):
    path, intent = (folder / f'{name}.json', folder / f'{name}_access_started.json')
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get('config_hash') != task['config_hash']:
            raise ValueError('Diagnostic metric/config mismatch')
        return saved['metric']
    if intent.exists():
        raise RuntimeError('Interrupted diagnostic access; automatic repeat prohibited')
    write_json(intent, dict(config_hash=task['config_hash'], exploratory=True, started=time.time()))
    metric = dict(evaluate(), exploratory=True)
    write_json(path, dict(config_hash=task['config_hash'], exploratory=True, metric=metric))
    return metric

def schedule(tasks, output, gpus, per_gpu, cfg):
    if per_gpu < 1 or not gpus or len(gpus) != len(set(gpus)):
        raise ValueError('Require unique GPUs and per-gpu >= 1')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'schedule.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = deque((task for task in tasks if not completed(output / task['run_id'], task)))
        slots = [(gpu, slot) for gpu in gpus for slot in range(per_gpu)]
        active, failures = ({}, [])
        started = time.time()
        budget_path = output / 'budget.json'
        budget = json.loads(budget_path.read_text()) if budget_path.exists() else {'process_seconds': 0.0, 'wall_seconds': 0.0}
        ledgers = {path.parent: json.loads(path.read_text()) for path in output.glob('*/attempts.json')}
        unfinished = [attempt for attempts in ledgers.values() for attempt in attempts if 'finished' not in attempt]
        if unfinished or budget.get('active_processes', 0):
            baseline = budget.get('updated_at', min((a['started'] for a in unfinished), default=started))
            budget['wall_seconds'] += max(0.0, started - baseline)
        for folder, attempts in ledgers.items():
            changed = False
            for attempt in attempts:
                if 'finished' not in attempt:
                    attempt.update(finished=started, exit_code=None, interrupted=True, recovery_accounting='conservative through coordinator restart')
                    changed = True
            if changed:
                write_json(folder / 'attempts.json', attempts)
        accounted = time.monotonic()

        def save_budget():
            nonlocal accounted
            now, wall_now = (time.monotonic(), time.time())
            budget['wall_seconds'] += now - accounted
            accounted = now
            charged = sum((max(0.0, a.get('finished', wall_now) - a['started']) for attempts in ledgers.values() for a in attempts))
            budget['process_seconds'] = max(budget['process_seconds'], charged)
            budget.update(updated_at=wall_now, active_processes=len(active))
            write_json(budget_path, budget)
        try:
            save_budget()
            while pending or active:
                if _COORDINATOR_STOP.is_set():
                    raise RuntimeError('Coordinator shutdown requested')
                save_budget()
                if budget['wall_seconds'] > cfg['wall_time_limit_seconds'] or budget['process_seconds'] / 3600 > cfg['process_gpu_hours_limit']:
                    raise RuntimeError('Predeclared wall/process-GPU budget exhausted')
                for slot in slots:
                    if slot in active or not pending:
                        continue
                    task = pending.popleft()
                    folder = output / task['run_id']
                    folder.mkdir(parents=True, exist_ok=True)
                    write_json(folder / 'task.json', task)
                    attempts = ledgers.setdefault(folder, [])
                    if len(attempts) >= 1 + cfg['max_restarts']:
                        failures.append(task['run_id'])
                        continue
                    attempts.append({'started': time.time(), 'gpu': slot[0]})
                    write_json(folder / 'attempts.json', attempts)
                    log = (folder / f'attempt-{len(attempts)}.log').open('a')
                    env = dict(os.environ, QUOTAGUARD_DEVICE=f'cuda:{slot[0]}', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS=str(cfg['cpu_threads']), QUOTAGUARD_PARENT_PID=str(os.getpid()))
                    command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(folder / 'task.json')]
                    inherited = (lease.fileno(),) + ((_RUNNER_LOCK_FD,) if _RUNNER_LOCK_FD is not None else ())
                    try:
                        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True, pass_fds=inherited)
                    except BaseException:
                        log.close()
                        attempts[-1].update(finished=time.time(), exit_code=None, launch_failed=True)
                        write_json(folder / 'attempts.json', attempts)
                        raise
                    active[slot] = (proc, task, folder, log, time.monotonic(), attempts)
                for slot, (proc, task, folder, log, launch, attempts) in list(active.items()):
                    timed_out = time.monotonic() - launch > cfg['run_timeout_seconds']
                    if timed_out and proc.poll() is None:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait()
                    code = proc.poll()
                    if code is None:
                        continue
                    log.close()
                    attempts[-1].update(finished=time.time(), exit_code=code, timed_out=timed_out)
                    write_json(folder / 'attempts.json', attempts)
                    del active[slot]
                    if code == 0 and completed(folder, task):
                        print(f'complete {task["run_id"]}', flush=True)
                    elif len(attempts) < 1 + cfg['max_restarts']:
                        pending.append(task)
                    else:
                        failures.append(task['run_id'])
                save_budget()
                if active:
                    time.sleep(0.2)
            if failures:
                raise RuntimeError(f'Training incomplete: {failures}; target report forbidden')
        finally:
            for proc, _, folder, log, _, attempts in active.values():
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                log.close()
                attempts[-1].update(finished=time.time(), exit_code=proc.poll(), interrupted=True)
                write_json(folder / 'attempts.json', attempts)
            active.clear()
            save_budget()
    return time.time() - started

def worker(task_path):
    _watch_parent()
    import torch
    from quotaguard.train.train import run_training
    from quotaguard.train.model import checkpoint_payload
    task_path = Path(task_path)
    cfg = json.loads(task_path.read_text())
    folder = task_path.parent
    torch.set_num_threads(cfg['cpu_threads'])
    cfg['device'] = os.environ.get('QUOTAGUARD_DEVICE', 'cuda:0')
    torch.cuda.set_device(cfg['device'])
    progress = {'phase': 'initializing', 'step': 0}
    stop = threading.Event()
    started = time.time()

    def heartbeat():
        while not stop.wait(30):
            print(json.dumps(dict(progress, elapsed_seconds=time.time() - started)), flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    timing = []
    relation_digest = hashlib.sha256()

    def record(row):
        progress.update(phase='training', step=row['step'] + 1)
        timing.append({k: v for k, v in row.items() if k != 'relations'})
        relation_digest.update(json.dumps(row['relations']).encode())

    def ready():
        if cfg['phase'] == 'benchmark':
            (folder / 'ready').touch()
            progress['phase'] = 'benchmark_barrier'
            while not Path(cfg['barrier']).exists():
                time.sleep(0.1)
        progress['phase'] = 'training'
    try:
        model, result = run_training(cfg, cfg['arm'], on_step=record, on_ready=ready, retain_records=False)
        result.pop('records', None)
        if cfg.get('log_negative_stats'):
            write_json(folder / 'negative_stats.json', result.pop('negative_stats'))
            result['negative_stats_path'] = str((folder / 'negative_stats.json').resolve())
        result.update(status='complete', config_hash=cfg['config_hash'], run_id=cfg['run_id'], source=cfg['source'], target=cfg['target'], seed=cfg['seed'], clip_seed=cfg['clip_seed'], eval_seed=cfg['eval_seed'], sigma=cfg['sigma'] if cfg['arm'] in PRIVATE_ARMS else 0.0, delta=cfg['delta'], calibration=cfg['calibration'], accountant_version=importlib.metadata.version('opacus'), training_seconds=sum((r['seconds'] for r in timing)), training_finished_at=time.time(), batch_sequence_sha256=relation_digest.hexdigest(), target_test_accessed=False, gpu_name=torch.cuda.get_device_name(), device=cfg['device'])
        if cfg.get('noise_calibration') in {'caseB', 'inbatch', 'inbatch_amp'}:
            result['accountant_version'] = '1'
        if cfg['phase'] in {'train', 'screen'}:
            from quotaguard.train.model import build_tokenizer
            from quotaguard.train.evaluate import evaluate_source_validation
            progress['phase'] = 'source_validation'
            tokenizer = build_tokenizer(cfg['model'])
            result['source_validation'] = evaluate_source_validation(model, tokenizer, cfg)
            trainable = checkpoint_payload(model, cfg)
            torch.save(trainable, folder / 'checkpoint.tmp')
            (folder / 'checkpoint.tmp').replace(folder / 'checkpoint.pt')
            if cfg['phase'] == 'screen':
                result.update(exploratory=True, variant=cfg['variant'])
        elif cfg['phase'] == 'diagnose':
            from quotaguard.train.model import build_tokenizer
            from quotaguard.train.evaluate import evaluate_source_validation, evaluate_target_test
            if cfg.get('exploratory') is not True:
                raise ValueError('Diagnosis requires exploratory=true')
            trainable = checkpoint_payload(model, cfg)
            torch.save(trainable, folder / 'checkpoint.tmp')
            (folder / 'checkpoint.tmp').replace(folder / 'checkpoint.pt')
            tokenizer = build_tokenizer(cfg['model'])
            progress['phase'] = 'source_validation'
            result['source_validation'] = diagnose_metric(folder, 'source_validation', cfg, lambda: evaluate_source_validation(model, tokenizer, cfg))
            progress['phase'] = 'target_test'
            result['target_test'] = diagnose_metric(folder, 'target', cfg, lambda: evaluate_target_test(model, tokenizer, cfg, final_report=True))
            result.update(exploratory=True, target_test_accessed=True)
        result['elapsed_seconds'] = time.time() - started
        write_json(folder / 'steps.json', timing)
        write_json(folder / 'result.json', result)
    finally:
        stop.set()

def paired_interval(differences, confidence=0.95):
    from scipy.stats import t
    if len(differences) < 2 or not all((math.isfinite(d) for d in differences)):
        raise ValueError('Need at least two finite run-level paired differences')
    mean = statistics.mean(differences)
    half = float(t.ppf((1 + confidence) / 2, len(differences) - 1)) * statistics.stdev(differences) / math.sqrt(len(differences))
    return dict(differences=differences, n=len(differences), mean=mean, confidence=confidence, lower=mean - half, upper=mean + half)

def paired_analysis(rows, cfg):
    primary, descriptive = analysis_pairs(cfg)
    delta_min = cfg.get('analysis', {}).get('delta_min', 0.02)
    primary_name = '-'.join(primary)
    index = {}
    for row in rows:
        key = (row['source'], row['target'], row['arm'], row['seed'])
        if key in index:
            raise ValueError('Duplicate run in paired analysis')
        index[key] = row['target_test']['mrr']
    expected = {(s, t, a, seed) for s, t in cfg['directions'] for a in cfg['arms'] for seed in cfg['training_seeds']}
    if set(index) != expected:
        raise ValueError('Missing or unexpected runs in paired analysis')
    conditions = []
    for source, target in cfg['directions']:
        comparisons = {}
        for a, b in [primary, *descriptive]:
            differences = [index[source, target, a, seed] - index[source, target, b, seed] for seed in cfg['training_seeds']]
            ci = paired_interval(differences)
            ci['seeds'] = cfg['training_seeds']
            ci['state'] = 'supported' if ci['lower'] >= delta_min else 'not_supported' if ci['upper'] < delta_min else 'undetermined'
            ci['descriptive_only'] = [a, b] != primary
            comparisons[f'{a}-{b}'] = ci
        rejection_interval = paired_interval(comparisons[primary_name]['differences'], 1 - 0.1 / len(cfg['directions']))
        comparisons[primary_name]['global_rejection_upper'] = rejection_interval['upper']
        conditions.append(dict(source=source, target=target, comparisons=comparisons))
    main = [r['comparisons'][primary_name] for r in conditions]
    state = 'not_supported' if any((c['global_rejection_upper'] < delta_min for c in main)) else 'supported' if all((c['lower'] >= delta_min for c in main)) else 'undetermined'
    return dict(state=state, conditions=conditions, delta_min=delta_min, statistical_unit='complete paired training run', scope=cfg.get('analysis', {}).get('scope', 'epsilon=10; OpenAlex; CN→US and US→CN'), fourth_state='invalid_input: integrity/precondition failure')

def full_corpus_analysis(rows, cfg):
    primary, descriptive = analysis_pairs(cfg)
    metrics = ('mrr', 'recall_at_10', 'recall_at_100')
    index = {}
    for row in rows:
        key = (row['source'], row['target'], row['arm'], row['seed'])
        if key in index:
            raise ValueError('Duplicate run in full-corpus analysis')
        values = row['target_test'].get('full_corpus', {})
        if not isinstance(values, dict) or any((type(values.get(name)) not in (int, float) or not math.isfinite(values[name]) or (not 0 <= values[name] <= 1) for name in metrics)) or type(values.get('n')) is not int or (values['n'] <= 0):
            raise ValueError('Missing or invalid full-corpus metrics')
        index[key] = values
    expected = {(s, t, a, seed) for s, t in cfg['directions'] for a in cfg['arms'] for seed in cfg['training_seeds']}
    if set(index) != expected:
        raise ValueError('Missing or unexpected runs in full-corpus analysis')
    conditions = []
    for source, target in cfg['directions']:
        comparisons = {}
        for a, b in [primary, *descriptive]:
            intervals = {}
            for metric in metrics:
                differences = [index[source, target, a, seed][metric] - index[source, target, b, seed][metric] for seed in cfg['training_seeds']]
                intervals[metric] = dict(paired_interval(differences), seeds=cfg['training_seeds'])
            comparisons[f'{a}-{b}'] = intervals
        conditions.append(dict(source=source, target=target, comparisons=comparisons))
    return dict(secondary_only=True, statistical_unit='complete paired training run', conditions=conditions)

@contextmanager
def final_budget(output, folder, cfg):
    path = Path(output) / 'budget.json'
    budget = json.loads(path.read_text()) if path.exists() else {'process_seconds': 0.0, 'wall_seconds': 0.0}
    allowance = min(cfg['run_timeout_seconds'], cfg['wall_time_limit_seconds'] - budget['wall_seconds'], cfg['process_gpu_hours_limit'] * 3600 - budget['process_seconds'])
    if allowance <= 0:
        raise RuntimeError('Final-report budget exhausted before target access')
    reserved = dict(budget, process_seconds=budget['process_seconds'] + allowance, wall_seconds=budget['wall_seconds'] + allowance)
    write_json(path, reserved)
    start = time.monotonic()
    stop = threading.Event()

    def heartbeat():
        with (folder / 'final_report.log').open('a') as log:
            while not stop.wait(30):
                log.write(json.dumps({'phase': 'final_report', 'elapsed_seconds': time.monotonic() - start}) + '\n')
                log.flush()

    def timeout(signum, frame):
        raise RuntimeError('Final-report timeout/budget exhausted; target access will not repeat')
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, allowance)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        stop.set()
        thread.join()
        elapsed = time.monotonic() - start
        budget['process_seconds'] += elapsed
        budget['wall_seconds'] += elapsed
        write_json(path, budget)

def final_report(tasks, output, cfg, gpu):
    from quotaguard.train.evaluate import evaluate_target_test
    from quotaguard.train.model import build_model
    import torch
    output = Path(output)
    if not all((completed(output / t['run_id'], t) for t in tasks)):
        raise RuntimeError('All training must finish before any target access')
    report_path = output / 'final_report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get('config_hash') != digest(cfg):
            raise ValueError('Final report/config mismatch')
        return report
    primary, _ = analysis_pairs(cfg)
    paired_fields = ('initial_parameter_sha256', 'batch_sequence_sha256', 'clipped_train_size')
    caseb_pair = bool(cfg.get('arm_overrides')) and all((arm in PRIVATE_ARMS and cfg['arm_overrides'].get(arm, {}).get('noise_calibration', cfg.get('noise_calibration')) == 'caseB' for arm in primary))
    if all((arm in PRIVATE_ARMS for arm in primary)):
        if not caseb_pair:
            paired_fields += ('epsilon', 'sigma')
    for source, target in cfg['directions']:
        for seed in cfg['training_seeds']:
            pair = [json.loads((output / t['run_id'] / 'result.json').read_text()) for t in tasks if (t['source'], t['target'], t['seed']) == (source, target, seed) and t['arm'] in primary]
            if len(pair) != 2 or {row['arm'] for row in pair} != set(primary) or any((k not in pair[0] or k not in pair[1] or pair[0][k] != pair[1][k] for k in paired_fields)) or (caseb_pair and (not all((isinstance(row.get('epsilon'), (int, float)) and math.isfinite(row['epsilon']) for row in pair)) or not math.isclose(pair[0]['epsilon'], pair[1]['epsilon'], rel_tol=0.0, abs_tol=1e-06))):
                raise ValueError('Invalid paired training: ' + '-'.join(primary))
    torch.set_num_threads(cfg['cpu_threads'])
    rows = []
    for task in tasks:
        folder = output / task['run_id']
        target_path, intent = (folder / 'target.json', folder / 'target_access_started.json')
        row = json.loads((folder / 'result.json').read_text())
        if target_path.exists():
            target = json.loads(target_path.read_text())
            if target['config_hash'] != task['config_hash']:
                raise ValueError('Target/config mismatch')
        else:
            if intent.exists():
                raise RuntimeError('Interrupted target access; automatic repeat prohibited')
            with final_budget(output, folder, cfg):
                torch.backends.cuda.matmul.allow_tf32 = task.get('tf32', False)
                torch.backends.cudnn.allow_tf32 = task.get('tf32', False)
                model, tokenizer = build_model(task['model'], seed=task['seed'])
                weights = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
                if task.get('model', {}).get('kind') == 'residual_e5':
                    from quotaguard.train.model import residual_checkpoint_weights
                    weights = residual_checkpoint_weights(weights, task, dimension=model.A.shape[0])
                names = {n for n, p in model.named_parameters() if p.requires_grad}
                if set(weights) != names:
                    raise ValueError('Invalid trainable checkpoint keys')
                model.load_state_dict(weights, strict=False)
                model.to(f'cuda:{gpu}')
                write_json(intent, {'config_hash': task['config_hash'], 'started': time.time()})
                metric = evaluate_target_test(model, tokenizer, task, final_report=True)
                target = dict(config_hash=task['config_hash'], metric=metric)
                write_json(target_path, target)
                del model
                torch.cuda.empty_cache()
        rows.append(dict(row, target_test=target['metric'], target_test_accessed=True))
    report = dict(paired_analysis(rows, cfg), runs=rows, config_hash=digest(cfg))
    if cfg.get('full_corpus_eval') is True:
        report['full_corpus'] = full_corpus_analysis(rows, cfg)
    write_json(report_path, report)
    return report

def select_best(rows, parameter):
    return min(rows, key=lambda r: (-r['mrr'], r[parameter] if parameter == 'learning_rate' else -r[parameter]))

def diagnose_report(tasks, output, cfg, gpu):
    from quotaguard.train.evaluate import evaluate_target_test
    from quotaguard.train.model import build_model
    from quotaguard.train.train import parameter_digest
    import torch
    output = Path(output)
    if not all((completed(output / task['run_id'], task) for task in tasks)):
        raise RuntimeError('All diagnostic runs must finish before reporting')
    path = output / 'diagnose_report.json'
    if path.exists():
        report = json.loads(path.read_text())
        if report.get('config_hash') != digest(cfg):
            raise ValueError('Diagnostic report/config mismatch')
        return report
    rows = [json.loads((output / task['run_id'] / 'result.json').read_text()) for task in tasks]
    initial = []
    if cfg.get('label_shuffle'):
        torch.set_num_threads(cfg['cpu_threads'])
        seen = set()
        for task in tasks:
            key = (task['source'], task['target'], task['seed'])
            if key in seen:
                continue
            seen.add(key)
            folder = output / f'initial_{task["source"]}-{task["target"]}_s{task["seed"]}'
            folder.mkdir(parents=True, exist_ok=True)

            def evaluate_initial():
                with final_budget(output, folder, cfg):
                    torch.backends.cuda.matmul.allow_tf32 = task.get('tf32', False)
                    torch.backends.cudnn.allow_tf32 = task.get('tf32', False)
                    model, tokenizer = build_model(task['model'], seed=task['seed'])
                    initial_digest = parameter_digest(model)
                    if any((row['initial_parameter_sha256'] != initial_digest for row in rows if (row['source'], row['target'], row['seed']) == key)):
                        raise ValueError('Initial diagnostic model does not match training initialization')
                    model.to(f'cuda:{gpu}')
                    metric = evaluate_target_test(model, tokenizer, task, final_report=True)
                    return dict(metric, initial_parameter_sha256=initial_digest)
            metric = diagnose_metric(folder, 'target', task, evaluate_initial)
            initial.append(dict(source=key[0], target=key[1], seed=key[2], exploratory=True, target_test=metric, metric_path=str((folder / 'target.json').resolve())))
    report = dict(exploratory=True, config_hash=digest(cfg), runs=rows, initial_models=initial, target_test_accessed=True, interpretation='Diagnostic only; no target-based selection or confirmatory claim')
    write_json(path, report)
    return report

def screen_report(tasks, output, cfg):
    rows = []
    for task in tasks:
        folder = output / task['run_id']
        if not completed(folder, task):
            raise RuntimeError('Screen training/source validation incomplete')
        row = json.loads((folder / 'result.json').read_text())
        if not math.isfinite(row['source_validation']['mrr']):
            raise ValueError('Nonfinite source validation MRR')
        rows.append(row)
    variants = []
    for variant in cfg['variants']:
        selected = [r for r in rows if r['variant'] == variant['name']]
        values = [r['source_validation']['mrr'] for r in selected]
        if len(values) != len(cfg['training_seeds']):
            raise ValueError('Incomplete screen variant')
        variants.append(dict(name=variant['name'], arm=variant['arm'], n=len(values), mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else 0.0, diagnostics_by_seed={str(r['seed']): r.get('screen_diagnostics', {}) for r in selected}))
    report = dict(exploratory=True, target_test_accessed=False, config_hash=digest(cfg), metric='source_validation.mrr', std_ddof=1, runs=rows, variants=variants)
    write_json(output / 'screen_report.json', report)
    return report

def experiment(cfg, mode, output, gpus, per_gpu):
    if mode == 'screen':
        validate_config(cfg, mode)
        if per_gpu != 1:
            raise ValueError('Screen requires one process per GPU')
        tasks, bases = ([], {})
        source, target = cfg['directions'][0]
        for variant in cfg['variants']:
            candidate = dict(cfg, **variant['overrides'])
            method = candidate.get('noise_calibration', 'opacus')
            calibration_key = (method, variant['name']) if method == 'caseB' else method
            if calibration_key not in bases:
                bases[calibration_key] = prepare(candidate, source, target)
            prepared = dict(bases[calibration_key], **variant['overrides'], variant=variant['name'])
            for key in {'hard_filter_margin', 'optimizer', 'adambc_floor', 'hard_mix', 'learning_rate', 'noise_calibration', 'bounded', 'tau_sampler'}:
                if key in candidate:
                    prepared[key] = copy.deepcopy(candidate[key])
                else:
                    prepared.pop(key, None)
            tasks.extend((task_for(prepared, variant['arm'], seed, phase='screen', suffix=f'_{variant["name"]}') for seed in cfg['training_seeds']))
        schedule(tasks, output, gpus, per_gpu, cfg)
        screen_report(tasks, output, cfg)
        return
    if mode in {'confirm', 'diagnose'} and cfg.get('arm_overrides'):
        validate_config(cfg, mode)
        tasks = []
        arm_learning_rates = any(('learning_rate' in o for o in cfg['arm_overrides'].values()))
        for source, target in cfg['directions']:
            case_a_privacy = None
            for arm in cfg['arms']:
                candidate = dict(cfg, **cfg['arm_overrides'].get(arm, {}))
                prepared = prepare(candidate, source, target)
                if arm_learning_rates and candidate.get('noise_calibration', 'opacus') in {'opacus', 'mog'}:
                    privacy = tuple((prepared[k] for k in ('sigma', 'gamma', 'delta')))
                    assert case_a_privacy is None or privacy == case_a_privacy, 'Case A/MoG arms must share sigma, gamma and delta'
                    case_a_privacy = privacy
                tasks.extend((task_for(prepared, arm, seed, phase='diagnose' if mode == 'diagnose' else 'train') for seed in cfg['training_seeds']))
        schedule(tasks, output, gpus, per_gpu, cfg)
        report = diagnose_report if mode == 'diagnose' else final_report
        report(tasks, output, cfg, gpus[0])
        return
    prepared = [prepare(cfg, s, t) for s, t in cfg['directions']]
    if mode == 'diagnose':
        tasks = [task_for(c, arm, seed, phase='diagnose') for c in prepared for arm in cfg['arms'] for seed in cfg['training_seeds']]
        schedule(tasks, output, gpus, per_gpu, cfg)
        diagnose_report(tasks, output, cfg, gpus[0])
        return
    if mode == 'confirm':
        tasks = [task_for(c, arm, seed) for c in prepared for arm in cfg['arms'] for seed in cfg['training_seeds']]
        schedule(tasks, output, gpus, per_gpu, cfg)
        final_report(tasks, output, cfg, gpus[0])
        return
    base, seed = (prepared[0], cfg['training_seeds'][0])
    choices = {}
    for arm, parameter, grid in [('R0', 'learning_rate', cfg['learning_rates']), ('H0', 'tau_sampler', cfg['sampler_temperatures'])]:
        tasks = []
        for value in grid:
            candidate = dict(base, **choices)
            candidate[parameter] = value
            tasks.append(task_for(candidate, arm, seed, suffix=f'_{parameter}-{value}'))
        schedule(tasks, output, gpus, per_gpu, cfg)
        rows = [{parameter: t[parameter], 'mrr': json.loads((output / t['run_id'] / 'result.json').read_text())['source_validation']['mrr']} for t in tasks]
        choices[parameter] = select_best(rows, parameter)[parameter]
        write_json(output / f'selection_{arm}.json', dict(exploratory=True, target_test_accessed=False, candidates=rows, selected=choices.copy()))
    write_json(output / 'selection.json', dict(exploratory=True, target_test_accessed=False, selected=choices, config_hash=digest(cfg)))

def benchmark(cfg, output, gpus):
    from concurrent.futures import ThreadPoolExecutor
    if len(gpus) != 1 and len(gpus) < 3:
        raise ValueError('Benchmark needs one GPU (serial K trials) or at least three (parallel)')
    cfg = dict(cfg, steps=100)
    prepared = prepare(cfg, 'CN', 'US')

    def trial(k, gpu):
        folder = output / f'K{k}'
        folder.mkdir(parents=True, exist_ok=True)
        barrier = folder / 'go'
        tasks = [task_for(dict(prepared, barrier=str(barrier)), 'H1', cfg['training_seeds'][0], phase='benchmark', suffix=f'_worker{i}') for i in range(k)]
        if (folder / 'throughput.json').exists():
            return json.loads((folder / 'throughput.json').read_text())
        if barrier.exists() or any(((folder / t['run_id']).exists() for t in tasks)):
            raise RuntimeError('Partial benchmark: use a new output directory for a synchronized trial')
        with ThreadPoolExecutor(max_workers=1) as executor:
            job = executor.submit(schedule, tasks, folder, [gpu], k, cfg)
            started = time.monotonic()
            while not all(((folder / t['run_id'] / 'ready').exists() for t in tasks)):
                if job.done():
                    job.result()
                    raise RuntimeError('Benchmark exited before barrier')
                if time.monotonic() - started > cfg['run_timeout_seconds']:
                    raise RuntimeError('Benchmark initialization timeout')
                time.sleep(0.2)
            released = time.time()
            barrier.touch()
            job.result()
        rows = [json.loads((folder / t['run_id'] / 'result.json').read_text()) for t in tasks]
        seconds = max((r['training_finished_at'] for r in rows)) - released
        result = dict(k=k, gpu=gpu, processes=k, steps_per_process=100, total_steps=100 * k, wall_seconds=seconds, steps_per_second=100 * k / seconds, clipped_train_size=prepared['public_train_size'], gamma=prepared['gamma'], sigma=prepared['sigma'], calibration=prepared['calibration'], arm='H1', refresh_every=cfg['refresh_every'], target_test_accessed=False, config_hash=digest(prepared), runs=rows)
        write_json(folder / 'throughput.json', result)
        return result
    if len(gpus) == 1:
        rows = [trial(k, gpus[0]) for k in [1, 2, 3]]
    else:
        with ThreadPoolExecutor(max_workers=3) as executor:
            jobs = [executor.submit(trial, k, gpu) for k, gpu in zip([1, 2, 3], gpus)]
            rows = [job.result() for job in jobs]
    best = max(rows, key=lambda r: (r['steps_per_second'], -r['k']))
    report = dict(trials=rows, best_k=best['k'], target_test_accessed=False, trial_layout='same GPU, sequential K' if len(gpus) == 1 else 'separate GPUs, parallel K', caveat='H1 100-step engineering trials include one full refresh. Parallel K trials can have GPU/load confounding. No formal-run performance claim.')
    write_json(output / 'benchmark.json', report)
    print(json.dumps({'best_k': best['k'], 'throughput': [(r['k'], r['steps_per_second']) for r in rows]}), flush=True)

def main():
    global _RUNNER_LOCK_FD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--mode', choices=['explore', 'confirm', 'benchmark', 'diagnose', 'screen'])
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--per-gpu', type=int, default=1)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not args.config or not args.mode:
        parser.error('--config and --mode are required')
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg['data_dir'] = os.environ.get('QUOTAGUARD_DATA_DIR', cfg['data_dir'])
    output = Path(os.environ.get('QUOTAGUARD_OUTPUT_DIR', cfg['output_dir'])).resolve() / args.mode
    gpus = [int(g) for g in args.gpus.split(',')]
    validate_config(cfg, args.mode)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _RUNNER_LOCK_FD = lock.fileno()
        _COORDINATOR_STOP.clear()
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def terminated(signum, frame):
            _COORDINATOR_STOP.set()
            raise RuntimeError('Coordinator terminated by SIGTERM')
        signal.signal(signal.SIGTERM, terminated)
        try:
            fingerprint = dict(config=cfg, mode=args.mode)
            snapshot = output / 'configuration.json'
            if snapshot.exists() and json.loads(snapshot.read_text()) != fingerprint:
                raise ValueError('Output directory belongs to a different configuration')
            write_json(snapshot, fingerprint)
            if args.mode == 'benchmark':
                benchmark(cfg, output, gpus)
            else:
                experiment(cfg, args.mode, output, gpus, args.per_gpu)
        except Exception as exc:
            write_json(output / 'failure.json', dict(state='undetermined' if isinstance(exc, RuntimeError) else 'invalid_input', error=str(exc), time=time.time()))
            raise
        finally:
            _RUNNER_LOCK_FD = None
            signal.signal(signal.SIGTERM, previous_sigterm)
if __name__ == '__main__':
    main()
