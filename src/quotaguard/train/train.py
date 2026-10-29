import hashlib
from importlib.metadata import version
import math
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from .data import clip_degree, load_domain, shuffle_labels, tokenize_titles
from .dp import CaseBAccountant, InBatchAccountant, InBatchAmplifiedAccountant, MoGAccountant, PrivacyAccountant, batch_gradients, inbatch_gradients
from .model import ResidualE5, build_model, encode_entities
from .samplers import bounded_parameters, fixed_permutation, generator_for, poisson_batch, sample_negatives
ANN_FORWARDING = True
ARMS = {'R0': (False, 'uniform'), 'H0': (False, 'hard'), 'R1': (True, 'uniform'), 'H1': (True, 'hard'), 'P1': (True, 'hard_perm'), 'normalized_uniform': (True, 'nodedp_wor'), 'normalized_hard': (True, 'nodedp_bwor'), 'B1': (True, 'inbatch')}

def parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def summarize(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    if not values.numel():
        return dict(count=0, mean=None, quantiles=None)
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    return dict(count=values.numel(), mean=values.mean().item(), quantiles=dict(zip(map(str, levels), values.quantile(torch.tensor(levels, dtype=torch.float64)).tolist())))

class NegativeStats:

    def __init__(self, model, tokens, relations, data, config):
        python_state, numpy_state = (random.getstate(), np.random.get_state())
        modes = [(module, module.training) for module in model.modules()]
        device = next(model.parameters()).device
        try:
            with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []), torch.no_grad():
                self.reference = F.normalize(encode_entities(model, tokens, config['encode_batch_size'], config.get('bf16', False)).detach().cpu(), dim=-1)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for module, training in modes:
                module.training = training
        self.ids = data.ids
        self.counts = torch.zeros(len(data.ids), dtype=torch.long)
        self.neighbors = [set() for _ in data.ids]
        for a, b in relations.tolist():
            self.neighbors[a].add(b)
            self.neighbors[b].add(a)
        self.steps = []

    @torch.no_grad()
    def record(self, step, batch, negatives, distributions, positions=None, *, compute_cosine=True):
        batch, negatives = (batch.cpu(), negatives.cpu())
        anchors = batch[:, :1].expand_as(negatives)
        if positions is not None:
            anchors = batch.gather(1, positions.cpu())
        self.counts += torch.bincount(negatives.flatten(), minlength=len(self.ids))
        cosine = (self.reference[anchors] * self.reference[negatives]).sum(dim=-1) if compute_cosine else None
        collisions = sum((n in self.neighbors[a] for a, n in zip(anchors.flatten().tolist(), negatives.flatten().tolist())))
        distribution = None
        if distributions:
            distribution = {name: summarize(column) for name, column in zip(('entropy', 'effective_support', 'max_probability'), zip(*distributions))}
        self.steps.append(dict(step=step, batch_size=len(batch), distribution=distribution, cosine_similarity=summarize(cosine) if cosine is not None else None, training_relation_collisions=collisions, self_collisions=int((anchors == negatives).sum()), sampled_relation_endpoint_collisions=int(((negatives == batch[:, :1]) | (negatives == batch[:, 1:])).sum())))

    def result(self):
        counts = self.counts.double()
        total, n = (int(self.counts.sum()), len(counts))
        top = max(1, math.ceil(n * 0.01))
        gini = ((2 * torch.arange(1, n + 1, dtype=torch.float64) - n - 1) * counts.sort().values).sum().item() / (n * total) if total else None
        return dict(schema_version=1, exploratory=True, reference='initial model; public source titles only; fixed before training', reference_sha256=hashlib.sha256(self.reference.numpy().tobytes()).hexdigest(), anchor='column 0; structure uses the endpoint selected for each negative loss', collision_graph='undirected clipped source training relations only', quantile_levels=[0.0, 0.25, 0.5, 0.75, 1.0], steps=self.steps, entity_ids=list(self.ids), entity_negative_counts=self.counts.tolist(), total_draws=total, top_one_percent_entities=top, top_one_percent_share=counts.topk(top).values.sum().item() / total if total else None, gini=gini, distinct_entities=int((counts > 0).sum()), training_relation_collisions=sum((s['training_relation_collisions'] for s in self.steps)), self_collisions=sum((s['self_collisions'] for s in self.steps)), sampled_relation_endpoint_collisions=sum((s['sampled_relation_endpoint_collisions'] for s in self.steps)))

def validate_screen_options(config, arm):
    if 'stable_truncation' in config:
        from .stable import validate_stable_options
        validate_stable_options(config, arm)
    private, kind = ARMS[arm]
    if {'negative_sampler', 'bm25_topn', 'bm25'} & set(config):
        from .bm25 import bm25_parameters
        if config.get('negative_sampler') != 'bm25_topn':
            raise ValueError('negative_sampler must be bm25_topn when BM25 options are present')
        bm25_parameters(config)
        if arm != 'H1' or config.get('noise_calibration') != 'mog' or config.get('stable_truncation') or (config.get('model', {}).get('kind') == 'residual_e5') or any((key in config for key in ('bounded', 'caseB', 'ann', 'hard_filter_margin', 'hard_mix', 'hard_cache'))):
            raise ValueError('bm25_topn requires H1 with MoG notion A and no other sampler options')
    if config.get('model', {}).get('kind') == 'residual_e5':
        if arm not in {'R1', 'H1'} or config.get('noise_calibration') != 'mog':
            raise ValueError('residual_e5 requires R1/H1 with MoG Case A')
        if config.get('hard_cache', 'dynamic') != 'dynamic' or any((key in config for key in ('bounded', 'hard_filter_margin', 'hard_mix'))):
            raise ValueError('residual_e5 requires unmodified dynamic hard sampling')
    if config.get('hard_cache', 'dynamic') not in {'dynamic', 'static'}:
        raise ValueError('hard_cache must be dynamic or static')
    if config.get('hard_cache') == 'static' and arm != 'H1':
        raise ValueError('Static hard_cache requires H1')
    if arm == 'B1' or config.get('noise_calibration') in {'inbatch', 'inbatch_amp'}:
        if arm != 'B1' or config.get('noise_calibration') not in {'inbatch', 'inbatch_amp'}:
            raise ValueError('B1 requires noise_calibration=inbatch or inbatch_amp, exclusively')
        amplified = config.get('noise_calibration') == 'inbatch_amp'
        phases = {'screen', 'diagnose'} if amplified else {'diagnose'}
        if config.get('exploratory') is not True or config.get('phase') not in phases:
            raise ValueError('B1 requires exploratory=true and ' + ('screen/diagnose phase' if amplified else 'diagnose phase'))
        if amplified:
            if config.get('label_shuffle'):
                raise ValueError('inbatch_amp forbids label_shuffle: deletion coupling would change')
            lr = config.get('learning_rate')
            if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
                raise ValueError('inbatch_amp requires a positive learning_rate; resolve unset first')
        if config.get('log_negative_stats'):
            raise ValueError('B1 does not support independent-negative statistics')
    diag_every = config.get('caseB_diag_every', 1)
    if type(diag_every) is not int or diag_every < 1:
        raise ValueError('caseB_diag_every must be a positive integer')
    if config.get('optimizer', 'adamw') not in {'adamw', 'adambc'}:
        raise ValueError('Unknown optimizer')
    if config.get('optimizer') == 'adambc' and (not private):
        raise ValueError('AdamBC requires a private arm')
    floor = config.get('adambc_floor', 1e-12)
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError('AdamBC floor must be finite and positive')
    if 'hard_filter_margin' in config or 'hard_mix' in config:
        if kind != 'hard':
            raise ValueError('Filtering/mixing requires the hard arm')
    if 'hard_filter_margin' in config and (not math.isfinite(config['hard_filter_margin'])):
        raise ValueError('Filter margin must be finite')
    if 'hard_mix' in config:
        mix = config['hard_mix']
        if not isinstance(mix, dict) or set(mix) != {'start', 'end'} or any((not 0 <= mix[k] <= 1 for k in ('start', 'end'))):
            raise ValueError('hard_mix requires start/end probabilities')
    if 'bounded' in config:
        bounded_parameters(config['bounded'])
        if kind not in {'hard', 'uniform'}:
            raise ValueError('bounded requires hard or uniform arm')
        if any((key in config for key in ('hard_filter_margin', 'hard_mix'))):
            raise ValueError('bounded forbids filtering/mixing')
    if config.get('noise_calibration', 'opacus') not in {'opacus', 'mog', 'caseB', 'inbatch', 'inbatch_amp'}:
        raise ValueError('Unknown noise calibration')
    if config.get('noise_calibration') == 'mog' and (not private):
        raise ValueError('MoG calibration requires a private arm')
    if config.get('noise_calibration') == 'caseB':
        if not private or kind not in {'hard', 'uniform'}:
            raise ValueError('Case B requires a private hard/uniform arm')
        if any((key in config for key in ('hard_filter_margin', 'hard_mix'))):
            raise ValueError('Case B forbids filtering/mixing')
        if 'label_shuffle' in config and (not (config.get('exploratory') is True and config.get('phase') == 'diagnose')):
            raise ValueError('Case B label shuffle requires exploratory=true and diagnose phase')
        if 'bounded' not in config and kind == 'uniform':
            raise ValueError('Case B uniform must use bounded R=1')
        if not math.isfinite(config['tau_sampler']) or config['tau_sampler'] <= 0:
            raise ValueError('Invalid Case B sampler temperature')

class AdamBC(torch.optim.Optimizer):

    def __init__(self, params, *, lr, noise_variance, floor=1e-12, betas=(0.9, 0.999), eps=1e-08):
        if not math.isfinite(noise_variance) or noise_variance < 0 or (not math.isfinite(floor)) or (floor <= 0):
            raise ValueError('Invalid AdamBC variance/floor')
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, noise_variance=noise_variance, floor=floor))
        self.floored = self.coordinates = 0

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
                state['step'] += 1
                t = state['step']
                m, v = (state['exp_avg'], state['exp_avg_sq'])
                m.lerp_(p.grad, 1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                corrected = v / (1 - b2 ** t) - group['noise_variance']
                self.floored += int((corrected < group['floor']).sum())
                self.coordinates += corrected.numel()
                denom = corrected.clamp_min(group['floor']).sqrt().add_(group['eps'])
                p.addcdiv_(m, denom, value=-group['lr'] / (1 - b1 ** t))

    def diagnostics(self):
        return dict(adambc_floored_coordinates=self.floored, adambc_coordinates=self.coordinates, adambc_floor_fraction=self.floored / self.coordinates if self.coordinates else 0.0, adambc_floor=self.param_groups[0]['floor'], adambc_noise_variance=self.param_groups[0]['noise_variance'])

class Trainer:

    def __init__(self, model, tokens, data, config, arm):
        if data.role != 'training' or data.split != 'train' or data.domain != config['source']:
            raise ValueError('Training only accepts source train relations')
        if arm not in ARMS:
            raise ValueError('Unknown arm')
        self.private, self.kind = ARMS[arm]
        self.model, self.tokens, self.data, self.config = (model, tokens, data, config)
        self.relations = clip_degree(data.relations, len(data.ids), config['kdeg'], seed=config['clip_seed'])
        if config.get('stable_truncation') is True:
            from .stable import public_bounds, public_scale
            bounds, _ = public_bounds(config, data)
            size, delta = public_scale(bounds)
            if not torch.equal(self.relations, data.relations) or config['public_train_size'] != size or config['delta'] != delta or (config['gamma'] != min(1.0, config['expected_batch_size'] / size)):
                raise ValueError('Stable training public scales/clipping changed')
        self.label_shuffle = None
        if config.get('label_shuffle', False):
            self.relations, self.label_shuffle = shuffle_labels(self.relations, seed=config['label_shuffle_seed'])
        self.M = config['gamma'] * config['public_train_size']
        if self.M <= 0 or config['refresh_every'] < 1:
            raise ValueError('Invalid public normalization/refresh interval')
        validate_screen_options(config, arm)
        self.bm25_table = None
        if config.get('negative_sampler') == 'bm25_topn':
            from .bm25 import BM25TopN, bm25_parameters
            topn, params = bm25_parameters(config)
            self.bm25_table = BM25TopN(data.titles, topn=topn, bm25=params)
            self.kind = 'bm25_topn'
        if 'bounded' in config:
            self.kind = 'uniform' if config['bounded']['R'] == 1 else 'bounded_hard'
        parameters = [p for p in model.parameters() if p.requires_grad]
        if config.get('optimizer', 'adamw') == 'adambc':
            self.optimizer = AdamBC(parameters, lr=config['learning_rate'], noise_variance=(config['sigma'] * config['C'] * (2 if self.kind == 'inbatch' else config['kdeg']) / self.M) ** 2, floor=config.get('adambc_floor', 1e-12))
        else:
            self.optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=0.0)
        self.screen_diagnostics = {}
        self.noise_generator = generator_for(config['seed'], 'gaussian')
        self.permutation = fixed_permutation(len(data.ids), seed=config['seed'])
        if config.get('stable_truncation') is True:
            from .stable import StableAccountant
            extra = {}
            if config['noise_calibration'] == 'caseB':
                extra = dict(k_neg=config['k_neg'], delta=config['delta'], **config['caseB'])
                R = config['bounded']['R'] if 'bounded' in config else math.exp(2 / config['tau_sampler'])
                if extra['m_bar'] != bounds['public_train_size'] or extra['N_min'] != bounds['N_min'] or extra['rho'] != R / (R + extra['N_min'] - 1) or (len(self.relations) > extra['m_bar']):
                    raise ValueError('Stable Case B public bounds changed')
            if bounds.get('zero_insertion'):
                if config['noise_calibration'] == 'caseB':
                    self.accountant = CaseBAccountant(config['sigma'], config['gamma'], config['kdeg'], total_steps=config['steps'], **extra)
                else:
                    self.accountant = MoGAccountant(config['sigma'], config['gamma'], config['kdeg'])
            else:
                self.accountant = StableAccountant(config['sigma'], config['gamma'], config['kdeg'], method=config['noise_calibration'], total_steps=config['steps'], **extra)
        elif config.get('noise_calibration') == 'inbatch':
            self.accountant = InBatchAccountant(config['sigma'])
        elif config.get('noise_calibration') == 'inbatch_amp':
            self.accountant = InBatchAmplifiedAccountant(config['sigma'], config['gamma'], config['kdeg'])
        elif config.get('noise_calibration') == 'caseB':
            bounds = config['caseB']
            if len(self.relations) > bounds['m_bar'] or len(data.ids) - 2 < bounds['N_min']:
                raise ValueError('Data exceed configured Case B public scale bounds')
            R = config['bounded']['R'] if 'bounded' in config else math.exp(2 / config['tau_sampler'])
            if bounds['rho'] + 1e-15 < R / (R + bounds['N_min'] - 1):
                raise ValueError('Case B rho underestimates sampler probability')
            self.accountant = CaseBAccountant(config['sigma'], config['gamma'], config['kdeg'], k_neg=config['k_neg'], delta=config['delta'], total_steps=config['steps'], **bounds)
        else:
            accountant = MoGAccountant if config.get('noise_calibration') == 'mog' else PrivacyAccountant
            self.accountant = accountant(config['sigma'], config['gamma'], config['kdeg'])
        self.cache = None
        self.step_number = 0
        self.negative_stats = NegativeStats(model, tokens, self.relations, data, config) if config.get('log_negative_stats', False) else None
        self.caseb_reference = F.normalize(encode_entities(model, tokens, config['encode_batch_size'], config.get('bf16', False)).detach().cpu(), dim=-1) if config.get('noise_calibration') == 'caseB' else None
        if config.get('hard_cache') == 'static':
            self.cache = encode_entities(model, tokens, config['encode_batch_size'], config.get('bf16', False))
        self.query_cache = None
        if isinstance(model, ResidualE5):
            self.cache = encode_entities(model, tokens, config['encode_batch_size'])
            self.tokens = {'input_ids': torch.arange(len(data.ids), device=self.cache.device)[:, None], 'cached_embeddings': self.cache}

    def synchronize(self):
        device = next(self.model.parameters()).device
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def step(self):
        cfg, step = (self.config, self.step_number)
        self.model.train()
        self.synchronize()
        started = time.perf_counter()
        refresh_seconds = 0.0
        if self.kind in {'hard', 'hard_perm', 'bounded_hard'} and cfg.get('hard_cache', 'dynamic') == 'dynamic' and (step % cfg['refresh_every'] == 0):
            before = time.perf_counter()
            if isinstance(self.model, ResidualE5):
                with torch.no_grad():
                    self.query_cache = self.model.query_embeddings(self.cache).detach()
            else:
                self.cache = encode_entities(self.model, self.tokens, cfg['encode_batch_size'], cfg.get('bf16', False))
            self.synchronize()
            refresh_seconds = time.perf_counter() - before
        indices = poisson_batch(self.relations, cfg['gamma'], seed=cfg['seed'], step=step)
        batch = self.relations[indices]
        caseb = self.caseb_reference is not None
        diagnostic_step = not caseb or step % cfg.get('caseB_diag_every', 1) == 0
        distributions = [] if diagnostic_step and (self.negative_stats is not None or caseb) else None
        maxima = [] if caseb else None
        sampling = {}
        if self.bm25_table is not None:
            sampling['bm25_table'] = self.bm25_table
        if caseb:
            sampling['cosine_bound'] = True
            sampling['probability_maxima'] = maxima
        if 'bounded' in cfg:
            sampling['bounded'] = cfg['bounded']
        if 'hard_filter_margin' in cfg:
            sampling['hard_filter_margin'] = cfg['hard_filter_margin']
        if 'hard_mix' in cfg:
            mix = cfg['hard_mix']
            fraction = step / max(1, cfg['steps'] - 1)
            sampling['hard_probability'] = mix['start'] + (mix['end'] - mix['start']) * fraction
        if 'ann' in cfg:
            sampling['ann'] = cfg['ann']
        if sampling:
            sampling['diagnostics'] = self.screen_diagnostics
        if self.query_cache is not None:
            sampling['query_embeddings'] = self.query_cache
        negatives = None if self.kind == 'inbatch' else sample_negatives(batch[:, 0], batch[:, 1], self.kind, cfg['k_neg'], n_entities=len(self.data.ids), seed=cfg['seed'], step=step, embeddings=self.cache, tau=cfg['tau_sampler'], permutation=self.permutation, **sampling, **{'distribution_stats': distributions} if distributions is not None else {})
        negative_endpoints = None
        if self.kind in {'nodedp_wor', 'nodedp_bwor'}:
            negatives, negative_endpoints = negatives
        if self.negative_stats is not None:
            self.negative_stats.record(step, batch, negatives, distributions, negative_endpoints, compute_cosine=diagnostic_step)
        if caseb and len(batch):
            diagnostics = self.screen_diagnostics
            if diagnostic_step:
                cosine = (self.caseb_reference[batch[:, :1]] * self.caseb_reference[negatives]).sum(-1)
                for key, value in {'reference_cosine_sum': cosine.sum().item(), 'reference_cosine_count': cosine.numel(), 'distribution_relations': len(distributions), 'effective_support_sum': sum((row[1] for row in distributions))}.items():
                    diagnostics[key] = diagnostics.get(key, 0) + value
            maximum = max(maxima)
            if maximum > cfg['caseB']['rho'] * (1 + 1e-06):
                raise ArithmeticError('Sampler exceeds configured Case B probability bound')
            diagnostics['max_probability'] = max(diagnostics.get('max_probability', 0), maximum)
        if self.kind == 'inbatch':
            gradients = inbatch_gradients(self.model, self.tokens, batch, C=cfg['C'], sigma=cfg['sigma'], M=self.M, tau_loss=cfg['tau_loss'], noise_generator=self.noise_generator, bf16=cfg.get('bf16', False))
        else:
            gradients = batch_gradients(self.model, self.tokens, batch, negatives, C=cfg['C'], sigma=cfg['sigma'], kdeg=cfg['kdeg'], M=self.M, tau_loss=cfg['tau_loss'], private=self.private, noise_generator=self.noise_generator, chunk_size=cfg.get('gradient_chunk_size', 16), bf16=cfg.get('bf16', False), negative_endpoints=negative_endpoints)
        self.optimizer.zero_grad(set_to_none=True)
        for p, gradient in zip((p for p in self.model.parameters() if p.requires_grad), gradients):
            p.grad = gradient
        self.optimizer.step()
        if self.private:
            self.accountant.step()
        self.step_number += 1
        self.synchronize()
        return {'step': step, 'seconds': time.perf_counter() - started, 'refresh_seconds': refresh_seconds, 'batch_size': len(batch), 'relations': batch.tolist()}

def run_training(config, arm, *, on_step=None, on_ready=None, retain_records=True):
    for name in ('seed', 'clip_seed', 'eval_seed'):
        if type(config[name]) is not int:
            raise ValueError(f'{name} must be a fixed integer')
    torch.backends.cuda.matmul.allow_tf32 = config.get('tf32', False)
    torch.backends.cudnn.allow_tf32 = config.get('tf32', False)
    data = load_domain(config['data_dir'], config['source'], 'train', role='training')
    model, tokenizer = build_model(config['model'], seed=config['seed'])
    model.to(config['device'])
    tokens = {k: v.to(config['device']) for k, v in tokenize_titles(data.titles, tokenizer).items()}
    trainer = Trainer(model, tokens, data, config, arm)
    initial_digest = parameter_digest(model)
    if next(model.parameters()).device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    records = []
    if on_ready is not None:
        on_ready()
    for _ in range(config['steps']):
        record = trainer.step()
        if retain_records:
            records.append(record)
        if on_step is not None:
            on_step(record)
    peak = torch.cuda.max_memory_allocated() if next(model.parameters()).is_cuda else 0
    result = {'arm': arm, 'initial_parameter_sha256': initial_digest, 'clipped_train_size': len(trainer.relations), 'public_M': trainer.M, 'peak_allocated_bytes': peak, 'peak_reserved_bytes': torch.cuda.max_memory_reserved() if peak else 0, 'source_entities': len(data.ids), 'records': records, 'seed': config['seed'], 'clip_seed': config['clip_seed'], 'eval_seed': config['eval_seed'], 'sigma': config['sigma'], 'delta': config['delta'], 'accountant': 'Opacus RDPAccountant', 'accountant_version': version('opacus'), 'epsilon': trainer.accountant.epsilon(config['delta']) if trainer.private else None}
    if config.get('noise_calibration') in {'mog', 'caseB', 'inbatch', 'inbatch_amp'}:
        result.update(trainer.accountant.report(config['delta']))
        result.update(nominal_epsilon=config['epsilon_budget'], calibration=config['calibration'])
    if trainer.screen_diagnostics or isinstance(trainer.optimizer, AdamBC):
        diagnostics = dict(trainer.screen_diagnostics)
        if diagnostics.get('bounded_candidates'):
            diagnostics['lower_saturation_fraction'] = diagnostics['bounded_lower'] / diagnostics['bounded_candidates']
            diagnostics['upper_saturation_fraction'] = diagnostics['bounded_upper'] / diagnostics['bounded_candidates']
        if diagnostics.get('distribution_relations'):
            diagnostics['effective_support_mean'] = diagnostics['effective_support_sum'] / diagnostics['distribution_relations']
            diagnostics['reference_cosine_mean'] = diagnostics['reference_cosine_sum'] / diagnostics['reference_cosine_count']
            diagnostics['reference'] = 'initial model; fixed source titles; comparison cosine convention'
            diagnostics['release_scope'] = 'Diagnostic output outside the private release'
        if diagnostics.get('filter_relations'):
            diagnostics['filter_fraction'] = diagnostics['filtered_candidates'] / diagnostics['filter_candidates']
            diagnostics['filter_fallback_fraction'] = diagnostics['filter_fallbacks'] / diagnostics['filter_relations']
        if isinstance(trainer.optimizer, AdamBC):
            diagnostics.update(trainer.optimizer.diagnostics())
        result['screen_diagnostics'] = diagnostics
    if trainer.label_shuffle is not None:
        result['label_shuffle'] = trainer.label_shuffle
    if trainer.negative_stats is not None:
        result['negative_stats'] = trainer.negative_stats.result()
    return (model, result)
