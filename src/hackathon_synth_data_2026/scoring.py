import math


def calculate_final_score(
    report,
    *,
    weights=None,
):
    """
    Combine utility, fidelity, and privacy results into ranking points.

    Expected report structure:
        report["utility"]["trtr"]["pr_auc"]
        report["utility"]["tstr"]["pr_auc"]
        report["fidelity"]["score"]
        report["privacy"]["summary"]

    Higher is better. Utility is uncapped, so final points may exceed 100.
    """
    names = ("utility", "fidelity", "privacy")

    if weights is None:
        weights = dict.fromkeys(names, 1 / 3)

    if set(weights) != set(names):
        raise ValueError(
            "Weights must contain utility, fidelity, and privacy"
        )

    def number(value, label):
        if value is None or isinstance(value, (bool, str)):
            raise ValueError(f"{label} must be a finite number")

        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{label} must be a finite number"
            ) from exc

        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")

        return value

    def unit_score(value, label):
        value = number(value, label)

        if not 0 <= value <= 1:
            raise ValueError(f"{label} must be between 0 and 1")

        return value

    weights = {
        name: number(weights[name], f"{name} weight")
        for name in names
    }

    if any(value < 0 for value in weights.values()):
        raise ValueError("Weights cannot be negative")

    if not math.isclose(
        sum(weights.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Weights must sum to 1")

    utility = report["utility"]
    fidelity = report["fidelity"]
    privacy = report["privacy"]["summary"]

    ap_trtr = unit_score(
        utility["trtr"]["pr_auc"],
        "TRTR average precision",
    )
    ap_tstr = unit_score(
        utility["tstr"]["pr_auc"],
        "TSTR average precision",
    )

    if ap_trtr == 0:
        raise ValueError(
            "TRTR average precision must be positive"
        )

    fidelity_score = unit_score(
        fidelity["score"],
        "Fidelity score",
    )

    if fidelity.get("undefined_components"):
        raise ValueError(
            "Fidelity contains undefined score components"
        )

    if privacy.get("technically_complete") is not True:
        raise ValueError(
            "Privacy evaluation is not technically complete"
        )

    if privacy.get("fallback_score", False):
        raise ValueError(
            "Cannot rank a privacy error-fallback score"
        )

    if privacy.get("valid_for_ranking") is not True:
        raise ValueError(
            "Privacy scoring has not been approved for ranking"
        )

    privacy_score = unit_score(
        privacy["privacy_score"],
        "Privacy score",
    )

    scores = {
        "utility": ap_tstr / ap_trtr,
        "fidelity": fidelity_score,
        "privacy": privacy_score,
    }

    contributions = {
        name: 100 * weights[name] * scores[name]
        for name in names
    }

    return {
        "final_points": sum(contributions.values()),
        "scores": scores,
        "weights": weights,
        "point_contributions": contributions,
        "utility_reference": {
            "ap_trtr": ap_trtr,
            "ap_tstr": ap_tstr,
            "ap_difference_tstr_minus_trtr": ap_tstr - ap_trtr,
        },
    }