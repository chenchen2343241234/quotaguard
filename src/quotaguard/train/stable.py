import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import torch
from .dp import CaseBAccountant, MoGAccountant, calibrate_caseb, calibrate_mog

def stable_noise(sigma):
    result = math.sqrt(2) * sigma
    target_squared = 2 * Fraction.from_float(sigma) ** 2
    while Fraction.from_float(result) ** 2 < target_squared or math.nextafter(result / math.sqrt(2), 0.0) < sigma:
        result = math.nextafter(result, math.inf)
    return result
PUBLIC_OPTION_KEYS = ('stable_scale', 'stable_delta', 'zero_insertion')

def uses_public_options(config):
    return any((k in config for k in PUBLIC_OPTION_KEYS))

def public_scale(bounds):
    if 'scale' in bounds:
        return (bounds['scale'], bounds['delta'])
    return (bounds['public_train_size'], 1 / bounds['public_train_size'])

def validate_stable_options(config, arm=None):
    if 'stable_truncation' in config and type(config['stable_truncation']) is not bool:
        raise ValueError('stable_truncation must be a boolean')
    if config.get('stable_truncation') is not True:
        if any((k in config for k in PUBLIC_OPTION_KEYS)):
            raise ValueError('stable_scale/stable_delta/zero_insertion require stable_truncation')
        return
    if config.get('noise_calibration') not in {'mog', 'caseB'}:
        raise ValueError('stable_truncation requires MoG case A or case B')
    if arm is not None and arm not in {'R1', 'H1'}:
        raise ValueError('stable_truncation supports R1/H1 only')
    if any((k in config for k in ('hard_filter_margin', 'hard_mix', 'public_train_size_override'))) or config.get('label_shuffle') or config.get('log_negative_stats'):
        raise ValueError('stable_truncation forbids graph-dependent extras/size overrides')
    if config.get('stable_scale', 'quota_sum') not in {'quota_sum', 'public_expected'}:
        raise ValueError('stable_scale must be quota_sum or public_expected')
    if config.get('stable_delta', 'inverse_scale') not in {'inverse_scale', 'entity_power'}:
        raise ValueError('stable_delta must be inverse_scale or entity_power')
    if 'zero_insertion' in config and type(config['zero_insertion']) is not bool:
        raise ValueError('zero_insertion must be a boolean')

def public_bounds(config, data):
    from quotaguard.data.buckets import PublicBuckets
    root = Path(config['data_dir'])
    payload = (root / 'manifest.json').read_bytes()
    manifest_hash = hashlib.sha256(payload).hexdigest()
    if not config.get('data_manifest_sha256') and (not config.get('manifest_sha256')):
        print('Dataset manifest pin omitted; verifying file and structure checksums')
    for key in ('data_manifest_sha256', 'manifest_sha256'):
        if key in config and config[key] != manifest_hash:
            raise ValueError('Stable dataset manifest checksum mismatch')
    manifest = json.loads(payload)
    spec = manifest.get('stable_truncation', {})
    if manifest.get('status') != 'passed' or spec.get('version') != 1 or spec.get('kdeg') != config['kdeg']:
        raise ValueError('Missing/incompatible stable bucket manifest')
    info = spec['domains'][config['source']]
    path = root / info['artifact']
    if hashlib.sha256(path.read_bytes()).hexdigest() != info['artifact_sha256']:
        raise ValueError('Public bucket checksum mismatch')
    buckets = PublicBuckets.load(path)
    summary = buckets.validate()
    if buckets.kdeg != config['kdeg'] or buckets.scheme != spec['scheme'] or buckets.seed != spec['seed'] or (info['public_train_size'] != summary['quota_sum']) or (info['public_entities'] != len(buckets.ids)) or (info['N_min'] != len(buckets.ids) - 3) or (summary['quota_sum'] <= 1):
        raise ValueError('Public scales must equal configured bucket structure')
    index = {v: i for i, v in enumerate(buckets.ids)}
    if not set(data.ids) <= set(index) or len(data.ids) < len(index) - 1 or (config['noise_calibration'] == 'mog' and data.ids != buckets.ids):
        raise ValueError('Training candidate universe violates public bounds')
    if len(data.ids) == len(index) and data.ids == buckets.ids:
        mapped = data.relations.numpy()
    else:
        mapped = torch.tensor([index[v] for v in data.ids])[data.relations].numpy()
    if len(buckets.truncate(mapped)) != len(mapped):
        raise ValueError('Stable training input is not already bucket-truncated')
    bounds = dict(public_train_size=summary['quota_sum'], public_entities=len(buckets.ids), N_min=len(buckets.ids) - 3, artifact_sha256=info['artifact_sha256'], kdeg=buckets.kdeg, seed=buckets.seed, scheme=buckets.scheme, derivation='B=sum_b beta_b; one quota per edge; fixed public universe')
    if not uses_public_options(config):
        if 'stable_public_bounds' in config and config['stable_public_bounds'] != bounds:
            raise ValueError('Dispatched public bounds changed')
        return (bounds, manifest_hash)
    if config.get('zero_insertion') is True:
        import numpy as np
        sizes = np.bincount(buckets.bucket, minlength=len(buckets.quotas))
        if not (np.all(sizes == 1) and np.all(buckets.quotas == 1)):
            raise ValueError('zero_insertion requires singleton pools with unit quota')
        bounds['zero_insertion'] = True
    rule = config.get('stable_scale', 'quota_sum')
    if rule == 'public_expected':
        scale = info.get('public_expected_size')
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 1 or (not info.get('public_expected_derivation')):
            raise ValueError('public_expected scale missing from stable manifest')
        bounds.update(scale=float(scale), scale_rule=rule, scale_derivation=info['public_expected_derivation'])
    else:
        bounds.update(scale=float(summary['quota_sum']), scale_rule=rule)
    if config.get('stable_delta', 'inverse_scale') == 'entity_power':
        bounds.update(delta=len(buckets.ids) ** (-1.1), delta_rule='public entities ** -1.1')
    else:
        bounds.update(delta=1 / bounds['scale'], delta_rule='1/scale')
    if 'stable_public_bounds' in config and config['stable_public_bounds'] != bounds:
        raise ValueError('Dispatched public bounds changed')
    return (bounds, manifest_hash)

class StableAccountant:

    def __init__(self, sigma, gamma, kdeg, *, method, total_steps, **bounds):
        if method not in {'mog', 'caseB'} or type(total_steps) is not int or total_steps < 1:
            raise ValueError('Invalid stable accounting schedule')
        self.sigma, self.total_steps, self.steps = (sigma, total_steps, 0)
        self.method = method
        inner_sigma = math.nextafter(sigma / math.sqrt(2), 0.0)
        if method == 'caseB':
            self.inner = CaseBAccountant(inner_sigma, gamma, kdeg, total_steps=2 * total_steps, **bounds)
        else:
            self.inner = MoGAccountant(inner_sigma, gamma, kdeg)

    def step(self):
        if self.steps >= self.total_steps:
            raise ValueError('Stable accountant exceeds configured step budget')
        self.inner.step()
        self.inner.step()
        self.steps += 1

    def report(self, delta):
        inner = self.inner.report(delta)
        result = dict(inner)
        result.update(accountant='stable_' + inner['accountant'], sigma=self.sigma, sum_noise_over_C=self.sigma * self.inner.kdeg, steps=self.steps, total_steps=self.total_steps, accounting_steps=self.inner.steps, accounting_total_steps=2 * self.total_steps, accounting_sigma=self.inner.sigma, stable_truncation=True, rdp_forward=[2 * x for x in inner['rdp_forward']], rdp_reverse=[2 * x for x in inner['rdp_reverse']], rdp_direction_scope='single_actual_step_sufficient_bound', privacy_adjacency='raw graph; configured public bucket quotas', composition='2 original steps at sigma/sqrt(2) per actual step', original_accountant=inner)
        return result

    def epsilon(self, delta):
        return self.report(delta)['epsilon']

def prepare_stable(config, source, target):
    from .data import load_domain, clip_degree
    from .samplers import bounded_parameters
    cfg = copy.deepcopy(config)
    cfg.update(source=source, target=target)
    validate_stable_options(cfg)
    data = load_domain(cfg['data_dir'], source, 'train', role='training')
    bounds, manifest_hash = public_bounds(cfg, data)
    clipped = clip_degree(data.relations, len(data.ids), cfg['kdeg'], seed=cfg['clip_seed'])
    if not torch.equal(clipped, data.relations):
        raise ValueError('Stable training requires identity degree clipping')
    size, delta = public_scale(bounds)
    capacity = bounds['public_train_size']
    cfg.update(stable_public_bounds=bounds, manifest_sha256=manifest_hash, public_train_size=size, gamma=min(1.0, cfg['expected_batch_size'] / size), delta=delta, normalization_source='fixed public scale (' + bounds['scale_rule'] + ') * gamma' if uses_public_options(cfg) else 'fixed public sum of bucket quotas * gamma')
    if uses_public_options(cfg):
        cfg['stable_capacity'] = capacity
    method, extra = (cfg['noise_calibration'], {})
    calibration = calibrate_mog
    if method == 'caseB':
        R = bounded_parameters(cfg['bounded'])[0] if 'bounded' in cfg else math.exp(2 / cfg['tau_sampler'])
        N_min = bounds['N_min']
        if N_min < 2:
            raise ValueError('Case B public universe is too small')
        cfg['caseB'] = dict(m_bar=capacity, N_min=N_min, rho=R / (R + N_min - 1))
        extra = dict(k_neg=cfg['k_neg'], **cfg['caseB'])
        calibration = calibrate_caseb
    if bounds.get('zero_insertion'):
        original = calibration(epsilon=cfg['epsilon_budget'], delta=cfg['delta'], steps=cfg['steps'], gamma=cfg['gamma'], kdeg=cfg['kdeg'], tolerance=1e-10, **extra)
        cfg['sigma'] = original['sigma']
        if method == 'caseB':
            accountant = CaseBAccountant(cfg['sigma'], cfg['gamma'], cfg['kdeg'], delta=cfg['delta'], total_steps=cfg['steps'], **extra)
        else:
            accountant = MoGAccountant(cfg['sigma'], cfg['gamma'], cfg['kdeg'])
        for _ in range(cfg['steps']):
            accountant.step()
        cfg['calibration'] = dict(accountant.report(cfg['delta']), calibration_T=original, method='zero_insertion_' + original['method'], nominal_epsilon=cfg['epsilon_budget'])
        return cfg
    original = calibration(epsilon=cfg['epsilon_budget'], delta=cfg['delta'], steps=2 * cfg['steps'], gamma=cfg['gamma'], kdeg=cfg['kdeg'], tolerance=1e-10, **extra)
    cfg['sigma'] = stable_noise(original['sigma'])
    accountant = StableAccountant(cfg['sigma'], cfg['gamma'], cfg['kdeg'], method=method, total_steps=cfg['steps'], **dict(delta=cfg['delta'], **extra) if method == 'caseB' else {})
    for _ in range(cfg['steps']):
        accountant.step()
    cfg['calibration'] = dict(accountant.report(cfg['delta']), calibration_2T=original, method='stable_' + original['method'], nominal_epsilon=cfg['epsilon_budget'])
    return cfg
