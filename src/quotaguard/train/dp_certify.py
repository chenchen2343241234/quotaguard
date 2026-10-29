from fractions import Fraction
from math import isfinite
from flint import acb, acb_poly, arb

def ball(value):
    if isinstance(value, float) and (not isfinite(value)):
        raise ValueError('Nonfinite input')
    return arb(value)

def _fraction(value):
    return Fraction(str(value.fmpq()))

def decimal_endpoint(value, *, upper, digits=30):
    if not value.is_finite():
        raise ArithmeticError('Cannot serialize a nonfinite certificate')
    endpoint = _fraction(value.upper() if upper else value.lower())
    scaled = endpoint * 10 ** digits
    n = -(-scaled.numerator // scaled.denominator) if upper else scaled.numerator // scaled.denominator
    sign = '-' if n < 0 else ''
    integer, remainder = divmod(abs(n), 10 ** digits)
    return f'{sign}{integer}.{remainder:0{digits}d}'

def enclosure(value):
    return {'lower': decimal_endpoint(value, upper=False), 'upper': decimal_endpoint(value, upper=True), 'ball': value.str(60, more=True)}

def binomial_prefix(n, probability, stop):
    p = ball(probability)
    if type(n) is not int or n < 0 or type(stop) is not int or (stop < 0):
        raise ValueError('Invalid binomial support')
    if not (p >= 0 and p <= 1):
        raise ValueError('Invalid probability')
    stop = min(stop, n)
    if p == 1:
        return [arb(int(i == n)) for i in range(stop + 1)]
    values = [(1 - p) ** n]
    for i in range(stop):
        values.append(values[-1] * (n - i) * p / ((i + 1) * (1 - p)))
    return values

def case_a(gamma, kdeg):
    return ([(arb(1), binomial_prefix(kdeg, gamma, kdeg))], {})

def case_b(*, gamma, kdeg, k_neg, rho, m_bar, delta, total_steps, b_star, j_star):
    if not 0 < ball(rho) < 1 or min(kdeg, k_neg, m_bar, total_steps) < 1:
        raise ValueError('Invalid Case B parameters')
    if not (0 <= b_star <= min(m_bar, 4096) and 0 <= j_star <= b_star):
        raise ValueError('Invalid Case B cutoffs')
    allowance = ball(delta) / (100 * total_steps)
    eta = 1 - (1 - ball(rho)) ** k_neg
    candidate = min(m_bar, 512)
    while True:
        tail = 1 - sum(binomial_prefix(m_bar, gamma, candidate), arb(0))
        if tail <= allowance:
            break
        if not tail > allowance or candidate >= 4096:
            raise ArithmeticError('Unresolved/unsupported outer cutoff')
        candidate = min(m_bar, max(candidate + 1, candidate * 2))
    if candidate != b_star:
        raise ValueError('Recorded b_star disagrees with certified selection')
    j_probs = binomial_prefix(b_star, eta, j_star)
    if not 1 - sum(j_probs, arb(0)) <= allowance:
        raise ArithmeticError('Recorded j_star fails certified tail allowance')
    if j_star and (not 1 - sum(j_probs[:-1], arb(0)) > allowance):
        raise ArithmeticError('Recorded j_star is not certified minimal')
    outer_pmf = binomial_prefix(m_bar, gamma, b_star)
    pi = binomial_prefix(kdeg, gamma, kdeg)
    branches, masses = ([], [])
    beta_j = arb(0)
    for ell, b_l in enumerate(outer_pmf):
        kappa = binomial_prefix(ell, eta, j_star)
        f_l = sum(kappa, arb(0))
        if not f_l > 0:
            raise ArithmeticError('Unresolved conditional mass')
        weights = [arb(0) for _ in range(kdeg + 2 * min(ell, j_star) + 1)]
        for i, p_i in enumerate(pi):
            for j, p_j in enumerate(kappa):
                weights[i + 2 * j] += p_i * p_j / f_l
        masses.append(b_l * f_l)
        branches.append(weights)
        beta_j += b_l * (1 - f_l)
    good = sum(masses, arb(0))
    beta = 1 - good
    if not (good > 0 and beta >= 0 and (beta < 1)):
        raise ArithmeticError('Unresolved Case B good-event probability')
    return (list(zip([mass / good for mass in masses], branches)), {'b_star': b_star, 'j_star': j_star, 'eta': eta, 'beta_L': 1 - sum(outer_pmf, arb(0)), 'beta_J': beta_j, 'beta': beta, 'one_step_good_mass': good})

def integrand(weights, s, power):
    s, p = (ball(s), ball(power))
    if not s > 0 or not weights or any((not w >= 0 for w in weights)):
        raise ValueError('Invalid Gaussian mixture')
    if not sum(weights, arb(0)).contains(1):
        raise ValueError('Mixture weights do not enclose normalization')
    polynomial = acb_poly([w * (-arb(d * d) / (2 * s * s)).exp() for d, w in enumerate(weights)])
    normalizer = 1 / (2 * arb.pi()).sqrt()

    def evaluate(z, analytic):
        ratio = polynomial((z / s).exp())
        return (-z * z / 2).exp() * normalizer * ratio.pow(p, analytic=analytic)
    return evaluate

def integration_bounds(weights, s, power, margin):
    if type(margin) is not int or margin < 1:
        raise ValueError('Tail margin must be a positive integer')
    extreme = ball(power) * (len(weights) - 1) / ball(s)
    low, high = (extreme.min(arb(0)), extreme.max(arb(0)))
    lo = _fraction(low.lower())
    hi = _fraction(high.upper())
    left = lo.numerator // lo.denominator - margin
    right = -(-hi.numerator // hi.denominator) + margin
    return (left, right, low, high)

def tail_bounds(weights, s, power, left, right):
    extreme = ball(power) * (len(weights) - 1) / ball(s)
    low, high = (extreme.min(arb(0)), extreme.max(arb(0)))
    if not (ball(left) < low and ball(right) > high):
        raise ValueError('Endpoints do not surround all possible stationary points')
    f = integrand(weights, s, power)
    return (f(acb(left), False).real / (low - left), f(acb(right), False).real / (right - high))

def moment(weights, s, power, *, margin=12, tolerance='1e-24', eval_limit=200000):
    left, right, _, _ = integration_bounds(weights, s, power, margin)
    f = integrand(weights, s, power)
    core = acb.integral(f, left, right, rel_tol=arb(tolerance), abs_tol=arb(tolerance), eval_limit=eval_limit)
    if not core.is_finite() or not core.imag.contains(0):
        raise ArithmeticError('Acb could not enclose the real finite integral')
    tail_left, tail_right = tail_bounds(weights, s, power, left, right)
    tail = (tail_left + tail_right).upper()
    full = core.real + arb(0).union(tail)
    if not full > 0:
        raise ArithmeticError('Integral enclosure does not prove positivity')
    return (full, {'left': left, 'right': right, 'finite_integral_radius': core.real.rad(), 'tail_upper': tail})

def certify_order(branches, s, alpha, steps, delta, *, beta=None, margin=12, tolerance='1e-24', eval_limit=200000):
    a, d = (ball(alpha), ball(delta))
    if not a > 1 or not 0 < d < 1 or type(steps) is not int or (steps < 0):
        raise ValueError('Invalid RDP conversion parameters')
    if not branches or not sum((w for w, _ in branches), arb(0)).contains(1):
        raise ValueError('Invalid outer mixture normalization')
    if any((not w >= 0 for w, _ in branches)):
        raise ValueError('Negative outer weight')
    good = arb(1) if beta is None else (1 - beta.upper()) ** steps
    conditional = (d - (1 - good)) / good
    if not (good > 0 and conditional > 0 and (conditional < 1)):
        raise ArithmeticError('Certified tail exhausts delta')
    directions = {}
    rdps = []
    for label, power in (('P||Q', a), ('Q||P', 1 - a)):
        total, finite_error, tails = (arb(0), arb(0), arb(0))
        leftmost, rightmost = (None, None)
        for outer, weights in branches:
            value, error = moment(weights, s, power, margin=margin, tolerance=tolerance, eval_limit=eval_limit)
            total += outer * value
            finite_error += outer * error['finite_integral_radius']
            tails += outer * error['tail_upper']
            leftmost = error['left'] if leftmost is None else min(leftmost, error['left'])
            rightmost = error['right'] if rightmost is None else max(rightmost, error['right'])
        rdp = (total.log() / (a - 1)).nonnegative_part()
        if not rdp.is_finite():
            raise ArithmeticError('Nonfinite Renyi enclosure')
        rdps.append(rdp)
        directions[label] = {'moment': enclosure(total), 'single_step_rdp': enclosure(rdp), 'weighted_finite_integral_radius': enclosure(finite_error), 'weighted_tail_upper': enclosure(tails), 'endpoint_range': [leftmost, rightmost]}
    composed = steps * rdps[0].max(rdps[1])
    epsilon = (composed + ((a - 1) / a).log() - (conditional.log() + a.log()) / (a - 1)).nonnegative_part() if steps else arb(0)
    return {'alpha': str(alpha), 'directions': directions, 'branch_count': len(branches), 'composed_rdp': enclosure(composed), 'conditional_delta': enclosure(conditional), 'tail_delta_upper': enclosure(1 - good), 'epsilon': enclosure(epsilon)}
