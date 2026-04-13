"""Kelly Criterion bankroll management."""


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """Calculate the full Kelly fraction.

    f* = (bp - q) / b
    where b = decimal_odds - 1, p = model_prob, q = 1 - p

    Returns the fraction of bankroll to wager (can be negative = no bet).
    """
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    p = model_prob
    q = 1 - p
    f = (b * p - q) / b
    return max(0.0, f)


def fractional_kelly(model_prob: float, decimal_odds: float,
                     fraction: float = 0.25) -> float:
    """Calculate fractional Kelly.

    Common fractions:
        0.25 = quarter Kelly (conservative, recommended)
        0.50 = half Kelly (moderate)
        1.00 = full Kelly (aggressive, high variance)
    """
    return kelly_fraction(model_prob, decimal_odds) * fraction


def kelly_to_units(kelly_frac: float, bankroll: float,
                   unit_size: float = 100.0,
                   max_bet_pct: float = 0.05) -> float:
    """Convert Kelly fraction to bet units.

    Args:
        kelly_frac: Kelly fraction (from fractional_kelly)
        bankroll: Total bankroll in dollars
        unit_size: Size of 1 unit in dollars
        max_bet_pct: Maximum single bet as fraction of bankroll

    Returns: Number of units to wager (capped at max_bet_pct of bankroll)
    """
    bet_dollars = kelly_frac * bankroll
    max_dollars = bankroll * max_bet_pct
    bet_dollars = min(bet_dollars, max_dollars)
    if unit_size <= 0:
        return 0.0
    return round(bet_dollars / unit_size, 2)
