from __future__ import annotations

from typing import Dict, List, Tuple

from .fib import app_ratio, closeness_to_set, external_retracement_ratio, retracement_ratio
from .models import Pivot, RuleCheck


def _direction_from_pivots(pivots: List[Pivot]) -> str:
    return 'bull' if pivots[-1].price > pivots[0].price else 'bear'


def evaluate_impulse_window(pivots: List[Pivot]) -> Tuple[bool, List[RuleCheck], Dict[str, float]]:
    if len(pivots) != 6:
        raise ValueError('Impulse window must have 6 pivots')

    direction = _direction_from_pivots(pivots)
    kinds = [p.kind for p in pivots]
    prices = [p.price for p in pivots]

    if direction == 'bull':
        expected_kinds = ['low', 'high', 'low', 'high', 'low', 'high']
        w1 = prices[1] - prices[0]
        w2 = prices[1] - prices[2]
        w3 = prices[3] - prices[2]
        w4 = prices[3] - prices[4]
        w5 = prices[5] - prices[4]
        overlap = prices[4] <= prices[1]
        truncation = prices[5] <= prices[3]
    else:
        expected_kinds = ['high', 'low', 'high', 'low', 'high', 'low']
        w1 = prices[0] - prices[1]
        w2 = prices[2] - prices[1]
        w3 = prices[2] - prices[3]
        w4 = prices[4] - prices[3]
        w5 = prices[4] - prices[5]
        overlap = prices[4] >= prices[1]
        truncation = prices[5] >= prices[3]

    checks: List[RuleCheck] = []
    checks.append(RuleCheck('alternating pivot kinds', kinds == expected_kinds, f'expected={expected_kinds}, got={kinds}', 1.0))
    checks.append(RuleCheck('wave 1 positive', w1 > 0, f'w1={w1:.4f}', 1.5))
    checks.append(RuleCheck('wave 2 retraces < 100% of wave 1', 0 < w2 < w1, f'w2={w2:.4f}, w1={w1:.4f}', 2.0))
    checks.append(RuleCheck('wave 3 positive', w3 > 0, f'w3={w3:.4f}', 1.5))
    checks.append(RuleCheck('wave 4 positive', w4 > 0, f'w4={w4:.4f}', 1.0))
    checks.append(RuleCheck('wave 5 positive', w5 > 0, f'w5={w5:.4f}', 1.5))
    checks.append(RuleCheck('wave 3 not shortest', w3 >= min(w1, w5), f'w1={w1:.4f}, w3={w3:.4f}, w5={w5:.4f}', 2.5))
    checks.append(RuleCheck('wave 4 no overlap with wave 1', not overlap, f'overlap={overlap}', 2.5))
    checks.append(RuleCheck('wave 5 not truncated', not truncation, f'truncation={truncation}', 1.0))

    hard_pass = all(c.passed for c in checks if c.weight >= 1.5)

    r2 = retracement_ratio(prices[0], prices[1], prices[2])
    r3 = app_ratio(prices[0], prices[1], prices[2], prices[3])
    r4 = retracement_ratio(prices[2], prices[3], prices[4])
    r5_from_w4 = external_retracement_ratio(prices[3], prices[4], prices[5])
    r5_from_w1 = app_ratio(prices[0], prices[1], prices[4], prices[5])
    net13 = abs(prices[3] - prices[0])
    r5_net13 = abs(prices[5] - prices[4]) / net13 if net13 else 0.0

    soft = {
        'w2_ret': r2,
        'w3_app': r3,
        'w4_ret': r4,
        'w5_ext_w4': r5_from_w4,
        'w5_app_w1': r5_from_w1,
        'w5_app_net13': r5_net13,
        'truncation_flag': float(truncation),
        'fib_score_w2': closeness_to_set(r2, [0.5, 0.618, 0.786], tol=0.18),
        'fib_score_w3': closeness_to_set(r3, [0.618, 1.0, 1.618, 2.618], tol=0.30),
        'fib_score_w4': closeness_to_set(r4, [0.236, 0.382, 0.5, 0.618], tol=0.18),
        'fib_score_w5': max(
            closeness_to_set(r5_from_w4, [1.272, 1.618, 2.618], tol=0.35),
            closeness_to_set(r5_from_w1, [0.618, 1.0, 1.618], tol=0.35),
            closeness_to_set(r5_net13, [0.382, 0.618, 1.0], tol=0.30),
        ),
    }
    return hard_pass, checks, soft


def evaluate_zigzag_window(pivots: List[Pivot]) -> Tuple[bool, List[RuleCheck], Dict[str, float]]:
    if len(pivots) != 4:
        raise ValueError('Zigzag window must have 4 pivots')

    direction = _direction_from_pivots(pivots)
    prices = [p.price for p in pivots]
    kinds = [p.kind for p in pivots]
    if direction == 'bull':
        expected = ['low', 'high', 'low', 'high']
        a = prices[1] - prices[0]
        b = prices[1] - prices[2]
        c = prices[3] - prices[2]
    else:
        expected = ['high', 'low', 'high', 'low']
        a = prices[0] - prices[1]
        b = prices[2] - prices[1]
        c = prices[2] - prices[3]

    checks: List[RuleCheck] = []
    checks.append(RuleCheck('alternating pivot kinds', kinds == expected, f'expected={expected}, got={kinds}', 1.0))
    checks.append(RuleCheck('wave A positive', a > 0, f'a={a:.4f}', 1.5))
    checks.append(RuleCheck('wave B positive', b > 0, f'b={b:.4f}', 1.0))
    checks.append(RuleCheck('wave C positive', c > 0, f'c={c:.4f}', 1.5))
    r_b = retracement_ratio(prices[0], prices[1], prices[2])
    r_c = app_ratio(prices[0], prices[1], prices[2], prices[3])
    sharp = 1.0 if r_b <= 0.5 else 0.0
    deep = 1.0 if r_b >= 0.618 else 0.0
    soft = {
        'b_ret': r_b,
        'c_app': r_c,
        'zigzag_variant_sharp': sharp,
        'zigzag_variant_deep': deep,
        'fib_score_b': closeness_to_set(r_b, [0.382, 0.5, 0.618, 0.786], tol=0.20),
        'fib_score_c': closeness_to_set(r_c, [1.0, 1.272, 1.618, 2.618], tol=0.35),
    }
    hard_pass = all(c.passed for c in checks if c.weight >= 1.5)
    return hard_pass, checks, soft


def evaluate_flat_window(pivots: List[Pivot]) -> Tuple[bool, List[RuleCheck], Dict[str, float]]:
    if len(pivots) != 4:
        raise ValueError('Flat window must have 4 pivots')
    direction = _direction_from_pivots(pivots)
    prices = [p.price for p in pivots]
    kinds = [p.kind for p in pivots]
    if direction == 'bull':
        expected = ['low', 'high', 'low', 'high']
        a = prices[1] - prices[0]
        b = prices[1] - prices[2]
        c = prices[3] - prices[2]
        b_exceeds_a_origin = prices[2] <= prices[0]
        c_breaks_b = prices[3] > prices[1]
    else:
        expected = ['high', 'low', 'high', 'low']
        a = prices[0] - prices[1]
        b = prices[2] - prices[1]
        c = prices[2] - prices[3]
        b_exceeds_a_origin = prices[2] >= prices[0]
        c_breaks_b = prices[3] < prices[1]
    r_b = retracement_ratio(prices[0], prices[1], prices[2])
    r_c = app_ratio(prices[0], prices[1], prices[2], prices[3])
    regular_score = 0.55 * closeness_to_set(r_b, [0.9, 1.0], tol=0.18) + 0.45 * closeness_to_set(r_c, [1.0], tol=0.22)
    expanded_score = 0.55 * closeness_to_set(r_b, [1.0, 1.236], tol=0.24) + 0.45 * closeness_to_set(r_c, [1.236, 1.618], tol=0.30)
    running_score = 0.60 * closeness_to_set(r_b, [1.0, 1.236], tol=0.24) + 0.40 * closeness_to_set(r_c, [0.618, 0.786, 1.0], tol=0.22)
    if b_exceeds_a_origin:
        running_score += 0.10
    if c_breaks_b:
        expanded_score += 0.08
    checks = [
        RuleCheck('alternating pivot kinds', kinds == expected, f'expected={expected}, got={kinds}', 1.0),
        RuleCheck('wave A positive', a > 0, f'a={a:.4f}', 1.5),
        RuleCheck('wave B deep retrace', r_b >= 0.786, f'b_ret={r_b:.4f}', 1.5),
        RuleCheck('wave C positive', c > 0, f'c={c:.4f}', 1.5),
        RuleCheck('expanded/running flat behavior allowed', b_exceeds_a_origin or r_b >= 0.9, f'b_exceeds_a_origin={b_exceeds_a_origin}, b_ret={r_b:.4f}', 1.0),
    ]
    best_variant = max(
        [('regular_flat', regular_score), ('expanded_flat', expanded_score), ('running_flat', running_score)],
        key=lambda x: x[1],
    )
    soft = {
        'b_ret': r_b,
        'c_app': r_c,
        'flat_variant_regular_score': regular_score,
        'flat_variant_expanded_score': expanded_score,
        'flat_variant_running_score': running_score,
        'flat_variant': best_variant[0],
        'fib_score_b': max(closeness_to_set(r_b, [0.786, 0.9, 1.0, 1.236], tol=0.25), 0.0),
        'fib_score_c': closeness_to_set(r_c, [0.618, 1.0, 1.236, 1.618], tol=0.35),
    }
    hard_pass = all(c.passed for c in checks if c.weight >= 1.5)
    return hard_pass, checks, soft


def evaluate_double_zigzag_window(pivots: List[Pivot]) -> Tuple[bool, List[RuleCheck], Dict[str, float]]:
    if len(pivots) != 6:
        raise ValueError('Double zigzag window must have 6 pivots')
    direction = _direction_from_pivots(pivots)
    prices = [p.price for p in pivots]
    kinds = [p.kind for p in pivots]
    if direction == 'bull':
        expected = ['low', 'high', 'low', 'high', 'low', 'high']
        w = prices[1] - prices[0]
        x = prices[1] - prices[2]
        y = prices[3] - prices[2]
        x2 = prices[3] - prices[4]
        z = prices[5] - prices[4]
    else:
        expected = ['high', 'low', 'high', 'low', 'high', 'low']
        w = prices[0] - prices[1]
        x = prices[2] - prices[1]
        y = prices[2] - prices[3]
        x2 = prices[4] - prices[3]
        z = prices[4] - prices[5]

    r_x1 = retracement_ratio(prices[0], prices[1], prices[2])
    r_y = app_ratio(prices[0], prices[1], prices[2], prices[3])
    r_x2 = retracement_ratio(prices[2], prices[3], prices[4])
    r_z = app_ratio(prices[2], prices[3], prices[4], prices[5])
    checks = [
        RuleCheck('alternating pivot kinds', kinds == expected, f'expected={expected}, got={kinds}', 1.0),
        RuleCheck('W positive', w > 0, f'w={w:.4f}', 1.5),
        RuleCheck('X positive retrace', x > 0 and 0.2 <= r_x1 <= 0.9, f'x={x:.4f}, x_ret={r_x1:.4f}', 1.5),
        RuleCheck('Y positive', y > 0, f'y={y:.4f}', 1.5),
        RuleCheck('second X positive retrace', x2 > 0 and 0.2 <= r_x2 <= 0.9, f'x2={x2:.4f}, x2_ret={r_x2:.4f}', 1.5),
        RuleCheck('Z positive', z > 0, f'z={z:.4f}', 1.5),
    ]
    soft = {
        'x1_ret': r_x1,
        'y_app': r_y,
        'x2_ret': r_x2,
        'z_app': r_z,
        'fib_score_x1': closeness_to_set(r_x1, [0.382, 0.5, 0.618], tol=0.22),
        'fib_score_y': closeness_to_set(r_y, [0.618, 1.0, 1.272], tol=0.30),
        'fib_score_x2': closeness_to_set(r_x2, [0.382, 0.5, 0.618], tol=0.22),
        'fib_score_z': closeness_to_set(r_z, [0.618, 1.0, 1.272, 1.618], tol=0.30),
    }
    hard_pass = all(c.passed for c in checks if c.weight >= 1.5)
    return hard_pass, checks, soft


def evaluate_triangle_window(pivots: List[Pivot]) -> Tuple[bool, List[RuleCheck], Dict[str, float]]:
    if len(pivots) != 6:
        raise ValueError('Triangle window must have 6 pivots')
    prices = [p.price for p in pivots]
    highs = [p.price for p in pivots if p.kind == 'high']
    lows = [p.price for p in pivots if p.kind == 'low']
    checks = [
        RuleCheck('at least three highs and lows', len(highs) >= 3 and len(lows) >= 3, f'highs={len(highs)}, lows={len(lows)}', 1.0),
        RuleCheck('contracting highs', highs[0] > highs[1] > highs[2] if len(highs) >= 3 else False, str(highs), 1.5),
        RuleCheck('rising lows', lows[0] < lows[1] < lows[2] if len(lows) >= 3 else False, str(lows), 1.5),
    ]
    hard_pass = all(c.passed for c in checks if c.weight >= 1.5)
    retrs = []
    for i in range(1, len(prices) - 1):
        prev = abs(prices[i] - prices[i - 1])
        nxt = abs(prices[i + 1] - prices[i])
        retrs.append(nxt / prev if prev else 0.0)
    avg_ret = sum(retrs) / len(retrs) if retrs else 0.0
    soft = {'avg_leg_ret': avg_ret, 'fib_score': closeness_to_set(avg_ret, [0.618, 0.786], tol=0.22) if retrs else 0.0}
    return hard_pass, checks, soft
