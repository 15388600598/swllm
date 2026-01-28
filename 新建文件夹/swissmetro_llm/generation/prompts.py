"""
Prompt templates for LLM-based choice generation.

This module provides prompt construction functions for three strategies:
1. Choice prediction: LLM directly outputs CHOICE (1, 2, or 3)
2. Utility estimation: LLM outputs utility values for softmax sampling
3. Residual adjustment: LLM outputs preference residuals to adjust baseline MNL
"""

from typing import Any
import pandas as pd


def make_prompt(row: pd.Series, cot: bool = False) -> str:
    """Create prompt for direct choice prediction.

    Args:
        row: DataFrame row with traveler and alternative attributes
        cot: If True, add chain-of-thought reasoning hints

    Returns:
        Formatted prompt string
    """
    extra = ""
    if cot:
        extra = (
            "\nReasoning rubric (do internally): compare time first, then cost, "
            "then headway/comfort; avoid obviously dominated options."
        )

    return f"""You are simulating a discrete choice in Swissmetro. Output ONLY a JSON object that matches the schema.

Alternatives (encode choice as integer):
1=TRAIN, 2=SWISSMETRO, 3=CAR

Traveler attributes (coded):
MALE={int(row.MALE)}, AGE={int(row.AGE)}, INCOME={int(row.INCOME)}, PURPOSE={int(row.PURPOSE)}, GA={int(row.GA)}.

Level-of-service:
TRAIN: TT={float(row.TRAIN_TT)}, CO={float(row.TRAIN_CO)}, HE={float(row.TRAIN_HE)}
SM:    TT={float(row.SM_TT)},    CO={float(row.SM_CO)},    HE={float(row.SM_HE)}, SEATS={float(row.SM_SEATS)}
CAR:   TT={float(row.CAR_TT)},   CO={float(row.CAR_CO)}

Instruction:
- Choose the most preferred alternative for this traveler.
- Do NOT add any special preference for GA beyond its cost implication (no GA preference term).
- Return JSON with key CHOICE only.{extra}"""


def make_prompt_u(row: pd.Series, cot: bool = False) -> str:
    """Create prompt for utility estimation.

    Args:
        row: DataFrame row with traveler and alternative attributes
        cot: If True, add chain-of-thought reasoning hints

    Returns:
        Formatted prompt string
    """
    extra = ""
    if cot:
        extra = "\nReason internally; DO NOT output reasoning."

    return f"""You are simulating utilities for a Swissmetro discrete choice. Output ONLY a JSON object with utilities.

Alternatives:
1=TRAIN, 2=SWISSMETRO (SM), 3=CAR

Traveler attributes (coded):
MALE={int(row.MALE)}, AGE={int(row.AGE)}, INCOME={int(row.INCOME)}, PURPOSE={int(row.PURPOSE)}, GA={int(row.GA)}.

Level-of-service:
TRAIN: TT={float(row.TRAIN_TT)}, CO={float(row.TRAIN_CO)}, HE={float(row.TRAIN_HE)}
SM:    TT={float(row.SM_TT)},    CO={float(row.SM_CO)},    HE={float(row.SM_HE)}, SEATS={float(row.SM_SEATS)}
CAR:   TT={float(row.CAR_TT)},   CO={float(row.CAR_CO)}

Instruction:
- Estimate utilities U_TRAIN, U_SM, U_CAR for this traveler.
- Higher utility = stronger preference.
- Do NOT add any special GA preference term beyond cost.
- Return JSON with keys U_TRAIN, U_SM, U_CAR (all numbers).{extra}"""


def make_prompt_residual(row: pd.Series, cot: bool = False) -> str:
    """Create prompt for residual/preference adjustment.

    Args:
        row: DataFrame row with traveler and alternative attributes
        cot: If True, add chain-of-thought reasoning hints

    Returns:
        Formatted prompt string
    """
    extra = ""
    if cot:
        extra = "\n(Think internally: compare time and cost; output JSON only.)"

    return f"""You will provide a small preference residual for Swissmetro choice.
Output ONLY strict JSON matching the schema. Each residual must be within [-1, 1].

Alternatives:
1=TRAIN, 2=SWISSMETRO, 3=CAR

Traveler attributes (coded):
MALE={int(row.MALE)}, AGE={int(row.AGE)}, INCOME={int(row.INCOME)}, PURPOSE={int(row.PURPOSE)}, GA={int(row.GA)}.

Level-of-service:
TRAIN: TT={float(row.TRAIN_TT)}, CO={float(row.TRAIN_CO)}, HE={float(row.TRAIN_HE)}
SM:    TT={float(row.SM_TT)},    CO={float(row.SM_CO)},    HE={float(row.SM_HE)}, SEATS={float(row.SM_SEATS)}
CAR:   TT={float(row.CAR_TT)},   CO={float(row.CAR_CO)}

Task:
- Return residuals that slightly adjust preferences but do NOT override realistic trade-offs.
- Do NOT add any special GA preference term.
- Keep values small. {extra}
"""


# =============================================================================
# Advanced Prompt Strategies (for future GoT/CoT experiments)
# =============================================================================

def make_prompt_cot_structured(row: pd.Series) -> str:
    """Create structured chain-of-thought prompt.

    Uses explicit step-by-step reasoning structure.

    Args:
        row: DataFrame row with traveler and alternative attributes

    Returns:
        Formatted prompt string
    """
    return f"""You are simulating a discrete choice in Swissmetro. Analyze step by step, then output JSON.

Alternatives: 1=TRAIN, 2=SWISSMETRO (SM), 3=CAR

Traveler Profile:
- MALE={int(row.MALE)}, AGE={int(row.AGE)}, INCOME={int(row.INCOME)}
- PURPOSE={int(row.PURPOSE)} (1=commute, 2=shopping, 3=business, 4=leisure, ...)
- GA={int(row.GA)} (1=has General Abonnement, 0=no)

Level-of-service Comparison:
              TRAIN        SM          CAR
Travel Time:  {float(row.TRAIN_TT):>8.1f}  {float(row.SM_TT):>8.1f}  {float(row.CAR_TT):>8.1f}
Cost:         {float(row.TRAIN_CO):>8.1f}  {float(row.SM_CO):>8.1f}  {float(row.CAR_CO):>8.1f}
Headway:      {float(row.TRAIN_HE):>8.1f}  {float(row.SM_HE):>8.1f}  N/A

Reasoning Steps (internal):
1. Compare travel times - which is fastest?
2. Compare costs (if GA=1, TRAIN effectively costs 0)
3. Consider traveler income level for price sensitivity
4. Consider trip purpose for time sensitivity
5. Weigh trade-offs and pick best option

Output ONLY: {{"CHOICE": <1, 2, or 3>}}"""


def make_prompt_got_causal(row: pd.Series) -> str:
    """Create Graph of Thought style prompt with causal structure.

    Models decision as a causal graph with explicit dependencies.

    Args:
        row: DataFrame row with traveler and alternative attributes

    Returns:
        Formatted prompt string
    """
    return f"""Analyze the discrete choice using this causal structure:

[Decision Tree]
├── Time Sensitivity
│   ├── If PURPOSE=business (3): weight TT heavily, prefer fastest
│   └── If PURPOSE=leisure (4+): balance TT with comfort
├── Budget Constraint
│   ├── If INCOME<=2 (low): prioritize cost strongly
│   └── If INCOME>=3 (high): comfort matters more than cost
├── GA Ownership
│   └── If GA=1: TRAIN_CO effectively 0, favors TRAIN
└── Final Choice = alternative with best weighted utility

Traveler: MALE={int(row.MALE)}, AGE={int(row.AGE)}, INCOME={int(row.INCOME)}, PURPOSE={int(row.PURPOSE)}, GA={int(row.GA)}

Alternatives:
TRAIN: TT={float(row.TRAIN_TT)}, CO={float(row.TRAIN_CO)}, HE={float(row.TRAIN_HE)}
SM:    TT={float(row.SM_TT)},    CO={float(row.SM_CO)},    HE={float(row.SM_HE)}, SEATS={float(row.SM_SEATS)}
CAR:   TT={float(row.CAR_TT)},   CO={float(row.CAR_CO)}

Apply the causal rules and output ONLY: {{"CHOICE": <1, 2, or 3>}}"""


# =============================================================================
# Prompt Strategy Registry
# =============================================================================

PROMPT_STRATEGIES = {
    "choice_basic": lambda row: make_prompt(row, cot=False),
    "choice_cot": lambda row: make_prompt(row, cot=True),
    "choice_cot_structured": make_prompt_cot_structured,
    "choice_got_causal": make_prompt_got_causal,
    "utility_basic": lambda row: make_prompt_u(row, cot=False),
    "utility_cot": lambda row: make_prompt_u(row, cot=True),
    "residual_basic": lambda row: make_prompt_residual(row, cot=False),
    "residual_cot": lambda row: make_prompt_residual(row, cot=True),
}


def get_prompt_fn(strategy: str):
    """Get prompt function by strategy name.

    Args:
        strategy: One of the registered strategy names

    Returns:
        Prompt function that takes a row and returns a string
    """
    if strategy not in PROMPT_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(PROMPT_STRATEGIES.keys())}")
    return PROMPT_STRATEGIES[strategy]
