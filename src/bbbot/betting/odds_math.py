"""Odds conversion, vig removal, and EV calculation."""


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    if american >= 100:
        return (american / 100) + 1
    else:
        return (100 / abs(american)) + 1


def decimal_to_american(decimal_odds: float) -> float:
    """Convert decimal odds to American odds."""
    if decimal_odds >= 2.0:
        return (decimal_odds - 1) * 100
    else:
        return -100 / (decimal_odds - 1)


def american_to_implied(american: float) -> float:
    """Convert American odds to raw implied probability (includes vig)."""
    if american >= 100:
        return 100 / (american + 100)
    else:
        return abs(american) / (abs(american) + 100)


def remove_vig_power(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove vig using the power method (shin method approximation).

    Returns fair probabilities that sum to 1.0.
    """
    total = prob_a + prob_b
    if total <= 0:
        return 0.5, 0.5
    return prob_a / total, prob_b / total


def remove_vig_from_odds(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Given two American odds, return vig-free fair probabilities."""
    imp_a = american_to_implied(odds_a)
    imp_b = american_to_implied(odds_b)
    return remove_vig_power(imp_a, imp_b)


def calculate_ev(model_prob: float, decimal_odds: float) -> float:
    """Calculate expected value as a percentage of stake.

    EV = (prob * net_payout) - (1 - prob) * 1
    Returns: EV as a decimal (0.05 = 5% edge)
    """
    net_payout = decimal_odds - 1
    return (model_prob * net_payout) - (1 - model_prob)


def calculate_ev_american(model_prob: float, american_odds: float) -> float:
    """Calculate EV given American odds."""
    return calculate_ev(model_prob, american_to_decimal(american_odds))


def calculate_clv(closing_odds: float, odds_taken: float) -> float:
    """Calculate Closing Line Value.

    Positive CLV means you got a better price than the market closed at.
    Returns the CLV in implied probability points.
    """
    closing_implied = american_to_implied(closing_odds)
    taken_implied = american_to_implied(odds_taken)
    return closing_implied - taken_implied
