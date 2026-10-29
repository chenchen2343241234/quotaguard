import math
from functools import lru_cache
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from opacus.accountants import RDPAccountant

def _validate_negative_endpoints(negative_endpoints, shape):
    if negative_endpoints is not None and (negative_endpoints.shape != shape or negative_endpoints.dtype != torch.long or ((negative_endpoints < 0) | (negative_endpoints > 1)).any()):
        raise ValueError('Negative endpoint positions must be 0/1 long tensors matching negatives')

def _relation_scores(z, tau_loss, negative_endpoints):
    if negative_endpoints is None:
        return z[0:1] @ z[1:].T / tau_loss
    positive = z[0:1] @ z[1:2].T
    anchors = z.index_select(0, negative_endpoints.to(z.device))
    negative = (anchors * z[2:]).sum(dim=-1).unsqueeze(0)
    return torch.cat((positive, negative), dim=1) / tau_loss

def relation_loss(model, tokens, relation, negatives, tau_loss, *, negative_endpoints=None):
    if tau_loss <= 0:
        raise ValueError('Loss temperature must be positive')
    _validate_negative_endpoints(negative_endpoints, negatives.shape)
    ids = torch.cat((relation, negatives)).to(tokens['input_ids'].device)
    z = model(**{k: v[ids] for k, v in tokens.items()})
    scores = _relation_scores(z, tau_loss, negative_endpoints)
    return F.cross_entropy(scores, torch.zeros(1, dtype=torch.long, device=scores.device))

def relation_gradients(model, tokens, relation, negatives, tau_loss, *, negative_endpoints=None):
    parameters = [p for p in model.parameters() if p.requires_grad]
    loss = relation_loss(model, tokens, relation, negatives, tau_loss, negative_endpoints=negative_endpoints)
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, grads)]

def per_relation_gradients(model, item_tokens, tau_loss, *, bf16=False, negative_endpoints=None):
    if tau_loss <= 0:
        raise ValueError('Loss temperature must be positive')
    items = item_tokens['input_ids']
    _validate_negative_endpoints(negative_endpoints, (items.shape[0], items.shape[1] - 2))
    parameters = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p.detach() for n, p in model.named_parameters() if not p.requires_grad}
    buffers = dict(model.named_buffers())
    device_type = next(model.parameters()).device.type

    def loss(trainable, one_item, endpoints=None):
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=bf16):
            z = functional_call(model, ({**frozen, **trainable}, buffers), (), one_item)
        z = z.float()
        with torch.autocast(device_type, enabled=False):
            scores = _relation_scores(z, tau_loss, endpoints)
            return F.cross_entropy(scores, torch.zeros(1, dtype=torch.long, device=z.device))
    if negative_endpoints is None:
        gradients = vmap(grad(loss), in_dims=(None, 0), randomness='error')(parameters, item_tokens)
    else:
        gradients = vmap(grad(loss), in_dims=(None, 0, 0), randomness='error')(parameters, item_tokens, negative_endpoints)
    return [g.detach().float() for g in gradients.values()]

def batch_gradients(model, tokens, relations, negatives, *, C, sigma, kdeg, M, tau_loss, private, noise_generator, chunk_size=16, bf16=False, negative_endpoints=None):
    if M <= 0 or C <= 0 or sigma < 0 or (kdeg < 1) or (len(relations) != len(negatives)):
        raise ValueError('Invalid gradient mechanism parameters')
    if chunk_size < 1:
        raise ValueError('Gradient chunk size must be positive')
    _validate_negative_endpoints(negative_endpoints, negatives.shape)
    parameters = [p for p in model.parameters() if p.requires_grad]
    total = [torch.zeros_like(p, dtype=torch.float32) for p in parameters]
    ids = torch.cat((relations, negatives), dim=1).to(tokens['input_ids'].device)
    for start in range(0, len(relations), chunk_size):
        selected = ids[start:start + chunk_size]
        grads = per_relation_gradients(model, {k: v[selected] for k, v in tokens.items()}, tau_loss, bf16=bf16, negative_endpoints=None if negative_endpoints is None else negative_endpoints[start:start + chunk_size])
        if private:
            norm = torch.linalg.vector_norm(torch.stack([g.flatten(1).norm(dim=1) for g in grads]), dim=0)
            scale = (C / norm.clamp_min(torch.finfo(norm.dtype).tiny)).clamp(max=1.0)
        else:
            scale = torch.ones(len(selected), device=total[0].device)
        for dest, grad in zip(total, grads):
            dest.add_((grad * scale.reshape(-1, *[1] * (grad.ndim - 1))).sum(dim=0))
    for grad in total:
        if private and sigma:
            noise = torch.randn(grad.shape, dtype=grad.dtype, device=noise_generator.device, generator=noise_generator).to(grad.device)
            grad.add_(noise, alpha=sigma * C * kdeg)
        grad.div_(M)
    return total

def entity_sample_rate(gamma, kdeg):
    if not 0 < gamma <= 1 or kdeg < 1:
        raise ValueError('Invalid gamma/Kdeg')
    return 1.0 if gamma == 1 else -math.expm1(kdeg * math.log1p(-gamma))

def inbatch_loss(model, tokens, relations, tau_loss, *, bf16=False):
    if not math.isfinite(tau_loss) or tau_loss <= 0 or len(relations) == 0:
        raise ValueError('In-batch loss requires a positive temperature and nonempty batch')
    device = tokens['input_ids'].device
    relations = relations.to(device)
    ids, inverse = torch.unique(relations.flatten(), return_inverse=True)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
        z = model(**{k: v[ids] for k, v in tokens.items()})
    with torch.autocast(device.type, enabled=False):
        endpoints = z.float()[inverse].reshape(len(relations), 2, -1)
        logits = endpoints[:, 0] @ endpoints[:, 1].T / tau_loss
        excluded = (relations[:, 1][None, :] == relations[:, 0, None]) | (relations[:, 1][None, :] == relations[:, 1, None])
        excluded.fill_diagonal_(False)
        return F.cross_entropy(logits.masked_fill(excluded, -torch.inf), torch.arange(len(relations), device=device), reduction='sum')

def inbatch_gradients(model, tokens, relations, *, C, sigma, M, tau_loss, noise_generator, bf16=False):
    if not (math.isfinite(C) and C > 0 and math.isfinite(sigma) and (sigma >= 0) and math.isfinite(M) and (M > 0) and math.isfinite(tau_loss) and (tau_loss > 0)):
        raise ValueError('Invalid in-batch gradient mechanism parameters')
    parameters = [p for p in model.parameters() if p.requires_grad]
    total = [torch.zeros_like(p, dtype=torch.float32) for p in parameters]
    if len(relations):
        loss = inbatch_loss(model, tokens, relations, tau_loss, bf16=bf16)
        grads = torch.autograd.grad(loss, parameters, allow_unused=True)
        total = [zero if g is None else g.detach().float() for zero, g in zip(total, grads)]
        norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in total]))
        if not torch.isfinite(norm):
            raise ArithmeticError('Nonfinite in-batch gradient norm')
        scale = (C / norm.clamp_min(torch.finfo(norm.dtype).tiny)).clamp(max=1.0)
        for gradient in total:
            gradient.mul_(scale)
    for gradient in total:
        if sigma:
            noise = torch.randn(gradient.shape, dtype=gradient.dtype, device=noise_generator.device, generator=noise_generator).to(gradient.device)
            gradient.add_(noise, alpha=2 * C * sigma)
        gradient.div_(M)
    return total

class PrivacyAccountant:

    def __init__(self, sigma, gamma, kdeg):
        if sigma < 0:
            raise ValueError('Negative noise')
        self.sigma = sigma
        self.sample_rate = entity_sample_rate(gamma, kdeg)
        self.accountant = RDPAccountant()

    def step(self):
        self.accountant.step(noise_multiplier=self.sigma, sample_rate=self.sample_rate)

    def epsilon(self, delta):
        if not 0 < delta < 1:
            raise ValueError('delta must be in (0, 1)')
        return self.accountant.get_epsilon(delta)
MOG_ORDERS = tuple([1 + x / 10 for x in range(1, 100)] + list(range(12, 64)))
ACCOUNTING_REFERENCE = 'paper, Section 4'

class InBatchAccountant:
    orders = MOG_ORDERS

    def __init__(self, sigma, *, steps=0):
        if not (math.isfinite(sigma) and sigma >= 0 and (type(steps) is int) and (steps >= 0)):
            raise ValueError('Invalid in-batch accounting arguments')
        self.sigma, self.steps = (sigma, steps)

    def step(self):
        self.steps += 1

    def report(self, delta):
        import numpy as np
        if not 0 < delta < 1:
            raise ValueError('delta must be in (0, 1)')
        orders = np.asarray(self.orders)
        rdp = np.zeros(len(orders)) if not self.steps else orders * (self.steps / (2 * self.sigma ** 2)) if self.sigma else np.full(len(orders), np.inf)
        eps = rdp + np.log((orders - 1) / orders) - (math.log(delta) + np.log(orders)) / (orders - 1)
        best = int(np.argmin(eps))
        return dict(accountant='inbatch_gaussian_v1', accountant_version='1', epsilon=max(0.0, float(eps[best])) if self.steps else 0.0, delta=delta, sigma=self.sigma, steps=self.steps, sensitivity_over_C=2.0, sum_noise_over_C=2 * self.sigma, sampling_amplification=False, orders=list(self.orders), best_order=self.orders[best], rdp=rdp.tolist(), conversion='RDP conversion, Theorem 20 (Opacus form)')

    def epsilon(self, delta):
        return self.report(delta)['epsilon']

def calibrate_inbatch(*, epsilon, delta, steps, tolerance=1e-07):
    if not (math.isfinite(epsilon) and epsilon > 0 and (0 < delta < 1) and (type(steps) is int) and (steps > 0) and (0 < tolerance < 1)):
        raise ValueError('Invalid in-batch calibration arguments')

    def account(sigma):
        return InBatchAccountant(sigma, steps=steps)
    low, high = (0.0, 1.0)
    while account(high).epsilon(delta) > epsilon:
        high *= 2
        if high > 100000000.0:
            raise ValueError('Cannot bracket in-batch noise')
    while high - low > tolerance * max(1.0, high):
        mid = (low + high) / 2
        if account(mid).epsilon(delta) <= epsilon:
            high = mid
        else:
            low = mid
    result = account(high).report(delta)
    result.update(method='inbatch_gaussian_v1', nominal_epsilon=epsilon, sigma_infeasible_lower=low, sigma_tolerance=tolerance)
    return result

class InBatchAmplifiedAccountant:
    orders = MOG_ORDERS

    def __init__(self, sigma, gamma, kdeg, *, steps=0):
        if not (math.isfinite(sigma) and sigma >= 0 and (0 < gamma <= 1) and (type(kdeg) is int) and (kdeg >= 1) and (type(steps) is int) and (steps >= 0)):
            raise ValueError('Invalid amplified in-batch accounting arguments')
        self.sigma, self.gamma, self.kdeg, self.steps = (sigma, gamma, kdeg, steps)
        self.sample_rate = entity_sample_rate(gamma, kdeg)

    def step(self):
        self.steps += 1

    def report(self, delta):
        from importlib.metadata import version
        import numpy as np
        from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
        if not 0 < delta < 1:
            raise ValueError('delta must be in (0, 1)')
        rdp = compute_rdp(q=self.sample_rate, noise_multiplier=self.sigma, steps=self.steps, orders=self.orders) if self.steps else np.zeros(len(self.orders))
        epsilon, best_order = get_privacy_spent(orders=self.orders, rdp=rdp, delta=delta)
        return dict(accountant='inbatch_amp_case_A_v1', accountant_version='1', opacus_version=version('opacus'), epsilon=epsilon if self.steps else 0.0, delta=delta, sigma=self.sigma, steps=self.steps, gamma=self.gamma, kdeg=self.kdeg, sample_rate=self.sample_rate, sensitivity_over_C=2.0, sum_noise_over_C=2 * self.sigma, sampling_amplification=True, adjacency='entity deletion after degree clipping', orders=list(self.orders), best_order=best_order, rdp=rdp.tolist(), directions=['P||Q', 'Q||P'], conversion='RDP conversion, Theorem 20 (Opacus form)', numerical_certification='Opacus floating point; use certify() for interval bounds')

    def epsilon(self, delta):
        return self.report(delta)['epsilon']

    def certify(self, delta, *, alpha=None, precision=192, **integration_options):
        from flint import arb, ctx
        from .dp_certify import ball, certify_order, enclosure
        if self.sigma <= 0 or type(precision) is not int or precision < 128:
            raise ValueError('Certification requires positive sigma and precision >= 128')
        if alpha is None:
            alpha = self.report(delta)['best_order']
        with ctx.workprec(precision):
            absent = (1 - ball(self.gamma)) ** self.kdeg
            q = 1 - absent
            result = certify_order([(arb(1), [absent, q])], self.sigma, alpha, self.steps, delta, **integration_options)
            result.update(accountant='inbatch_amp_case_A_v1', sigma=self.sigma, gamma=self.gamma, kdeg=self.kdeg, steps=self.steps, sample_rate=enclosure(q), precision_bits=precision, numerical_certification='Arb/Acb via dp_certify.certify_order')
            return result

def calibrate_inbatch_amp(*, epsilon, delta, steps, gamma, kdeg, tolerance=1e-07):
    if not (math.isfinite(epsilon) and epsilon > 0 and (0 < delta < 1) and (type(steps) is int) and (steps > 0) and (0 < tolerance < 1)):
        raise ValueError('Invalid amplified in-batch calibration arguments')

    def account(sigma):
        return InBatchAmplifiedAccountant(sigma, gamma, kdeg, steps=steps)
    low, high = (0.0, 1.0)
    while account(high).epsilon(delta) > epsilon:
        high *= 2
        if high > 100000000.0:
            raise ValueError('Cannot bracket amplified in-batch noise')
    while high - low > tolerance * max(1.0, high):
        mid = (low + high) / 2
        if account(mid).epsilon(delta) <= epsilon:
            high = mid
        else:
            low = mid
    result = account(high).report(delta)
    result.update(method='inbatch_amp_case_A_v1', nominal_epsilon=epsilon, sigma_infeasible_lower=low, sigma_tolerance=tolerance)
    return result

@lru_cache(maxsize=2)
def _mog_quadrature_rule(points):
    from numpy.polynomial.legendre import leggauss
    return leggauss(points)

def _mog_log_integral(power, means, log_weights, points):
    import numpy as np
    from scipy.special import logsumexp
    left_mode, right_mode = sorted((0.0, power * means[-1]))
    left, right = (left_mode - 12.0, right_mode + 12.0)
    panels = math.ceil(right - left)
    if panels * points > 2000000:
        raise ValueError('MoG integration range exceeds numerical resource limit')
    nodes, weights = _mog_quadrature_rule(points)
    width = (right - left) / panels
    z = (left + width * (np.arange(panels)[:, None] + (nodes + 1) / 2)).ravel()

    def integrand(z):
        log_ratio = logsumexp(log_weights[:, None] + means[:, None] * z - 0.5 * means[:, None] ** 2, axis=0)
        return -0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) + power * log_ratio
    integral = logsumexp(integrand(z).reshape(panels, points) + np.log(weights * width / 2))
    ends = integrand(np.array([left, right]))
    tails = ends - np.log([left_mode - left, right - right_mode])
    return float(logsumexp([integral, *tails]))

def _mog_integer_log_moments(log_weights, means, max_order):
    import numpy as np
    from scipy.special import logsumexp
    coefficients = np.array([0.0])
    terms = log_weights - 0.5 * means ** 2
    moments = {}
    for n in range(1, max_order + 1):
        updated = np.full(len(coefficients) + len(terms) - 1, -np.inf)
        for i, term in enumerate(terms):
            updated[i:i + len(coefficients)] = np.logaddexp(updated[i:i + len(coefficients)], coefficients + term)
        coefficients = updated
        sums = np.arange(len(coefficients)) * means[1]
        moments[n] = float(logsumexp(coefficients + 0.5 * sums ** 2))
    return moments

@lru_cache(maxsize=64)
def _mog_single_step(s, gamma, kdeg):
    import numpy as np
    from scipy.special import gammaln, xlogy, xlog1py
    if s == 0:
        return (tuple((math.inf for _ in MOG_ORDERS)),) * 2
    if gamma == 1:
        gaussian = tuple((a * (kdeg / s) ** 2 / 2 for a in MOG_ORDERS))
        return (gaussian, gaussian)
    i = np.arange(kdeg + 1)
    log_weights = gammaln(kdeg + 1) - gammaln(i + 1) - gammaln(kdeg - i + 1) + xlogy(i, gamma) + xlog1py(kdeg - i, -gamma)
    means = i / s
    exact = _mog_integer_log_moments(log_weights, means, int(max(MOG_ORDERS)))
    forward, reverse = ([], [])
    for alpha in MOG_ORDERS:
        for power, dest in ((alpha, forward), (1 - alpha, reverse)):
            coarse = _mog_log_integral(power, means, log_weights, 16)
            fine = _mog_log_integral(power, means, log_weights, 32)
            if not math.isfinite(fine) or not math.isclose(coarse, fine, rel_tol=1e-10, abs_tol=1e-10):
                raise ArithmeticError('MoG quadrature did not converge')
            if power == alpha and float(alpha).is_integer():
                reference = exact[int(alpha)]
                if not math.isclose(fine, reference, rel_tol=1e-10, abs_tol=1e-10):
                    raise ArithmeticError('MoG integer finite-sum cross-check failed')
                fine = max(fine, reference)
            dest.append(max(0.0, fine + 5e-12) / (alpha - 1))
    return (tuple(forward), tuple(reverse))

class MoGAccountant:
    orders = MOG_ORDERS

    def __init__(self, sigma, gamma, kdeg, *, steps=0):
        if not (math.isfinite(sigma) and sigma >= 0 and (0 < gamma <= 1) and (type(kdeg) is int) and (kdeg >= 1) and (type(steps) is int) and (steps >= 0) and math.isfinite(sigma * kdeg)):
            raise ValueError('Invalid MoG accounting arguments')
        self.sigma, self.gamma, self.kdeg = (sigma, gamma, kdeg)
        self.steps = steps
        self.forward, self.reverse = _mog_single_step(sigma * kdeg, gamma, kdeg)

    def step(self):
        self.steps += 1

    def report(self, delta):
        import numpy as np
        if not 0 < delta < 1:
            raise ValueError('delta must be in (0, 1)')
        orders = np.asarray(self.orders)
        rdp = self.steps * np.maximum(self.forward, self.reverse) if self.steps else np.zeros(len(orders))
        eps = rdp + np.log((orders - 1) / orders) - (math.log(delta) + np.log(orders)) / (orders - 1)
        best = int(np.argmin(eps))
        return dict(accountant='mog_case_A_v1', accountant_version='1', accounting_reference=ACCOUNTING_REFERENCE, epsilon=max(0.0, float(eps[best])) if self.steps else 0.0, delta=delta, steps=self.steps, sigma=self.sigma, sum_noise_over_C=self.sigma * self.kdeg, gamma=self.gamma, kdeg=self.kdeg, orders=list(self.orders), best_order=self.orders[best], directions=['P||Q', 'Q||P'], rdp_forward=list(self.forward), rdp_reverse=list(self.reverse), rdp=rdp.tolist(), rdp_direction_scope='single_step', conversion='RDP conversion, Theorem 20 (Opacus form)', integration='log-domain composite Gauss-Legendre 16/32; analytic tail bounds', integer_cross_check='finite sum (21), all integer grid orders', numerical_certification='floating point; not interval-certified')

    def epsilon(self, delta):
        return self.report(delta)['epsilon']

def calibrate_mog(*, epsilon, delta, steps, gamma, kdeg, tolerance=1e-07):
    if not (math.isfinite(epsilon) and epsilon > 0 and (0 < delta < 1) and (type(steps) is int) and (steps > 0) and (0 < gamma <= 1) and (type(kdeg) is int) and (kdeg >= 1) and (0 < tolerance < 1)):
        raise ValueError('Invalid MoG calibration arguments')

    def account(s):
        return MoGAccountant(s / kdeg, gamma, kdeg, steps=steps)
    low, high = (0.0, 1.0)
    while account(high).epsilon(delta) > epsilon:
        high *= 2
        if high > 100000000.0:
            raise ValueError('Cannot bracket MoG noise')
    while high - low > tolerance * max(1.0, high):
        mid = (low + high) / 2
        if account(mid).epsilon(delta) <= epsilon:
            high = mid
        else:
            low = mid
    result = account(high).report(delta)
    result.update(method='mog_case_A_v1', nominal_epsilon=epsilon, sum_noise_infeasible_lower=low, noise_tolerance=tolerance)
    return result
CASEB_ORDERS = tuple((2 + i / 4 for i in range(25)))

@lru_cache(maxsize=32)
def _caseb_distribution(gamma, kdeg, k_neg, rho, m_bar, delta, total_steps):
    import numpy as np
    from scipy.special import logsumexp
    from scipy.stats import binom
    eta = -math.expm1(k_neg * math.log1p(-rho))
    allowance = delta / (100 * total_steps)
    b_star = min(m_bar, 512)
    while binom.sf(b_star, m_bar, gamma) > allowance:
        b_star = min(m_bar, max(b_star + 1, b_star * 2))
    if b_star > 4096:
        raise ValueError('Case B outer support exceeds numerical resource limit')
    j_star = 0
    while binom.sf(j_star, b_star, eta) > allowance:
        j_star += 1
    ell = np.arange(b_star + 1)
    log_b = binom.logpmf(ell, m_bar, gamma)
    log_kappa = binom.logpmf(np.arange(j_star + 1)[None, :], ell[:, None], eta)
    log_F = logsumexp(log_kappa, axis=1)
    beta_L = float(binom.sf(b_star, m_bar, gamma))
    beta_J = float(np.exp(logsumexp(log_b + binom.logsf(j_star, ell, eta))))
    beta = float(np.nextafter((beta_L + beta_J) * (1 + 1e-10), np.inf))
    if beta >= 1 or -math.expm1(total_steps * math.log1p(-beta)) >= delta:
        raise ValueError('Case B tail budget exhausts delta')
    outer = log_b + log_F
    outer -= logsumexp(outer)
    log_pi = binom.logpmf(np.arange(kdeg + 1), kdeg, gamma)
    weights = np.full((b_star + 1, kdeg + 2 * j_star + 1), -np.inf)
    for i in range(kdeg + 1):
        for j in range(j_star + 1):
            weights[:, i + 2 * j] = np.logaddexp(weights[:, i + 2 * j], log_pi[i] + log_kappa[:, j] - log_F)
    return (outer, weights, dict(b_star=b_star, j_star=j_star, eta=eta, beta_per_step_upper=beta, beta_L=beta_L, beta_J=beta_J, tail_selection='b*=min(m_bar,512), doubled until sf<=delta/(100T); smallest j* with Bin(b*,eta).sf<=delta/(100T)'))

def _caseb_integrals(s, outer, weights, points):
    import numpy as np
    from scipy.special import logsumexp
    means = np.arange(weights.shape[1]) / s
    powers = np.array([*CASEB_ORDERS, *(1 - a for a in CASEB_ORDERS)])
    left = min(powers) * means[-1] - 12
    right = max(powers) * means[-1] + 12
    panels = math.ceil(right - left)
    if panels * points * len(outer) > 20000000:
        raise ValueError('Case B quadrature exceeds numerical resource limit')
    nodes, rule = _mog_quadrature_rule(points)
    width = (right - left) / panels
    z = (left + width * (np.arange(panels)[:, None] + (nodes + 1) / 2)).ravel()
    z = np.r_[z, left, right]
    log_ratio = np.full((len(outer), len(z)), -np.inf)
    for d, mean in enumerate(means):
        log_ratio = np.logaddexp(log_ratio, weights[:, d, None] + mean * z - mean * mean / 2)
    normal = -z * z / 2 - math.log(2 * math.pi) / 2
    log_rule = np.tile(np.log(rule * width / 2), panels)
    results = []
    for power in powers:
        values = normal + power * log_ratio
        integral = logsumexp(values[:, :-2] + log_rule, axis=1)
        low_mode, high_mode = sorted((0.0, power * means[-1]))
        tails = values[:, -2:] - np.log([low_mode - left, right - high_mode])
        integrals = logsumexp(np.column_stack((integral, tails)), axis=1)
        results.append(float(logsumexp(outer + integrals)))
    return np.asarray(results).reshape(2, -1)

def _caseb_integer_moments(s, outer, weights):
    import numpy as np
    from scipy.special import logsumexp
    terms = weights - 0.5 * (np.arange(weights.shape[1]) / s) ** 2
    coeff = np.zeros((len(outer), 1))
    moments = {}
    for order in range(1, int(max(CASEB_ORDERS)) + 1):
        updated = np.full((len(outer), coeff.shape[1] + terms.shape[1] - 1), -np.inf)
        for d in range(terms.shape[1]):
            updated[:, d:d + coeff.shape[1]] = np.logaddexp(updated[:, d:d + coeff.shape[1]], coeff + terms[:, d, None])
        coeff = updated
        per_ell = logsumexp(coeff + 0.5 * (np.arange(coeff.shape[1]) / s) ** 2, axis=1)
        moments[order] = float(logsumexp(outer + per_ell))
    return moments

@lru_cache(maxsize=64)
def _caseb_single_step(s, gamma, kdeg, k_neg, rho, m_bar, delta, total_steps):
    import numpy as np
    outer, weights, _ = _caseb_distribution(gamma, kdeg, k_neg, rho, m_bar, delta, total_steps)
    if s == 0:
        return (tuple((math.inf for _ in CASEB_ORDERS)),) * 2
    coarse = _caseb_integrals(s, outer, weights, 16)
    fine = _caseb_integrals(s, outer, weights, 32)
    if not np.isfinite(fine).all() or not np.allclose(coarse, fine, rtol=1e-10, atol=1e-10):
        raise ArithmeticError('Case B quadrature did not converge')
    exact = _caseb_integer_moments(s, outer, weights)
    for idx, alpha in enumerate(CASEB_ORDERS):
        if alpha.is_integer():
            reference = exact[int(alpha)]
            if not math.isclose(fine[0, idx], reference, rel_tol=1e-10, abs_tol=1e-10):
                raise ArithmeticError('Case B integer finite-sum cross-check failed')
            fine[0, idx] = max(fine[0, idx], reference)
    rdp = np.maximum(0.0, np.maximum(fine, coarse) + 5e-12) / (np.array(CASEB_ORDERS) - 1)
    return (tuple(rdp[0]), tuple(rdp[1]))

class CaseBAccountant:
    orders = CASEB_ORDERS

    def __init__(self, sigma, gamma, kdeg, *, k_neg, rho, m_bar, N_min, delta, total_steps, steps=0):
        if not (math.isfinite(sigma) and sigma >= 0 and math.isfinite(sigma * kdeg) and (0 < gamma <= 1) and (0 < rho < 1) and (0 < delta < 1) and all((type(v) is int and v >= 1 for v in (kdeg, k_neg, m_bar, N_min, total_steps))) and (type(steps) is int) and (0 <= steps <= total_steps)):
            raise ValueError('Invalid Case B accounting arguments')
        self.sigma, self.gamma, self.kdeg = (sigma, gamma, kdeg)
        self.k_neg, self.rho, self.m_bar, self.N_min = (k_neg, rho, m_bar, N_min)
        self.delta, self.total_steps, self.steps = (delta, total_steps, steps)
        _, _, self.tail = _caseb_distribution(gamma, kdeg, k_neg, rho, m_bar, delta, total_steps)
        self.forward, self.reverse = _caseb_single_step(sigma * kdeg, gamma, kdeg, k_neg, rho, m_bar, delta, total_steps)

    def step(self):
        if self.steps >= self.total_steps:
            raise ValueError('Case B exceeds configured step budget')
        self.steps += 1

    def report(self, delta=None):
        import numpy as np
        if delta is not None and delta != self.delta:
            raise ValueError('Case B delta must match configured tail calibration')
        log_good = self.steps * math.log1p(-self.tail['beta_per_step_upper'])
        tail_delta = -math.expm1(log_good)
        good_mass = math.exp(log_good)
        delta_G = (self.delta - tail_delta) / good_mass
        orders = np.array(self.orders)
        rdp = self.steps * np.maximum(self.forward, self.reverse) if self.steps else np.zeros(len(orders))
        eps = rdp + np.log((orders - 1) / orders) - (math.log(delta_G) + np.log(orders)) / (orders - 1)
        best = int(np.argmin(eps))
        return dict(accountant='caseB_v1', accountant_version='1', accounting_reference=ACCOUNTING_REFERENCE, epsilon=max(0.0, float(eps[best])) if self.steps else 0.0, delta=self.delta, sigma=self.sigma, sum_noise_over_C=self.sigma * self.kdeg, gamma=self.gamma, kdeg=self.kdeg, k_neg=self.k_neg, rho=self.rho, m_bar=self.m_bar, N_min=self.N_min, total_steps=self.total_steps, steps=self.steps, orders=list(self.orders), best_order=self.orders[best], rdp_forward=list(self.forward), rdp_reverse=list(self.reverse), rdp=rdp.tolist(), rdp_direction_scope='single_step_conditional_good_branch', directions=['P||Q', 'Q||P'], delta_budget=dict(total=self.delta, tail=tail_delta, good_mass=good_mass, conditional=delta_G, good_contribution=good_mass * delta_G, **self.tail), conversion='equations (19)-(20), RDP conversion', integration='log-domain composite Gauss-Legendre 16/32; analytic tail upper bounds', integer_cross_check='finite sum, every integer grid order; outer mixture retained', numerical_certification='floating point upward guards; not interval-certified')

    def epsilon(self, delta=None):
        return self.report(delta)['epsilon']

@lru_cache(maxsize=32)
def calibrate_caseb(*, epsilon, delta, steps, gamma, kdeg, k_neg, rho, m_bar, N_min, tolerance=1e-07):
    if not math.isfinite(epsilon) or epsilon <= 0 or (not 0 < tolerance < 1):
        raise ValueError('Invalid Case B calibration arguments')

    def account(s):
        return CaseBAccountant(s / kdeg, gamma, kdeg, k_neg=k_neg, rho=rho, m_bar=m_bar, N_min=N_min, delta=delta, total_steps=steps, steps=steps)
    low, high = (0.0, 1.0)
    while account(high).epsilon() > epsilon:
        low, high = (high, high * 2)
        if high > 100000000.0:
            raise ValueError('Cannot bracket Case B noise')
    while high - low > tolerance * max(1.0, high):
        mid = (low + high) / 2
        if account(mid).epsilon() <= epsilon:
            high = mid
        else:
            low = mid
    result = account(high).report()
    result.update(method='caseB_v1', nominal_epsilon=epsilon, sum_noise_infeasible_lower=low, noise_tolerance=tolerance)
    return result
