"""
Global configuration for Swissmetro LLM project.

This module contains all global constants, column definitions,
JSON schemas, and default parameters used across the package.
"""

from typing import Dict, List, Any
import os

# =============================================================================
# Data Column Definitions
# =============================================================================

# Columns used for demographic combination analysis
COMBO_COLS: List[str] = ["MALE", "AGE", "INCOME", "PURPOSE", "GA"]

# Required columns for synthetic data generation and evaluation
REQUIRED_COLS: List[str] = [
    "ID", "GA", "MALE", "AGE", "INCOME", "PURPOSE", "CHOICE",
    "TRAIN_TT", "SM_TT", "CAR_TT",
    "TRAIN_CO", "SM_CO", "CAR_CO",
    "TRAIN_HE", "SM_HE", "SM_SEATS"
]

# Alternative names
ALTERNATIVES: Dict[int, str] = {
    1: "TRAIN",
    2: "SM",  # SwissMetro
    3: "CAR"
}

# =============================================================================
# LLM Model Configuration
# =============================================================================

DEFAULT_MODEL: str = "gpt-4o-mini"
REPAIR_MODEL: str = "gpt-4o"  # Stronger model for repair stage
BATCH_ENDPOINT: str = "/v1/responses"
BATCH_COMPLETION_WINDOW: str = "24h"
DEFAULT_POLL_INTERVAL: int = 15  # seconds

# =============================================================================
# JSON Schemas for Structured Output
# =============================================================================

CHOICE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "CHOICE": {"type": "integer", "enum": [1, 2, 3]}
    },
    "required": ["CHOICE"],
    "additionalProperties": False
}

UTILITY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "U_TRAIN": {"type": "number"},
        "U_SM": {"type": "number"},
        "U_CAR": {"type": "number"}
    },
    "required": ["U_TRAIN", "U_SM", "U_CAR"],
    "additionalProperties": False
}

RESIDUAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "dU_TRAIN": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "dU_SM": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "dU_CAR": {"type": "number", "minimum": -1.0, "maximum": 1.0}
    },
    "required": ["dU_TRAIN", "dU_SM", "dU_CAR"],
    "additionalProperties": False
}

# Format configurations for OpenAI API
def get_format_config(schema_type: str) -> Dict[str, Any]:
    """Get format configuration for OpenAI structured output.

    Args:
        schema_type: One of 'choice', 'utility', 'residual'

    Returns:
        Format configuration dict for the API
    """
    schemas = {
        "choice": ("choice_schema", CHOICE_SCHEMA),
        "utility": ("utility_schema", UTILITY_SCHEMA),
        "residual": ("residual_schema", RESIDUAL_SCHEMA),
    }

    if schema_type not in schemas:
        raise ValueError(f"Unknown schema type: {schema_type}. Must be one of {list(schemas.keys())}")

    name, schema = schemas[schema_type]
    return {
        "type": "json_schema",
        "name": name,
        "strict": True,
        "schema": schema
    }

# =============================================================================
# MNL Model Configuration
# =============================================================================

# Optimization parameters
MNL_OPTIMIZER_CONFIG: Dict[str, Any] = {
    "method": "L-BFGS-B",
    "options": {
        "maxiter": 20000,
        "maxfun": 20000,
        "ftol": 1e-6,
        "gtol": 1e-5,
    }
}

# Bootstrap configuration
DEFAULT_BOOTSTRAP_B: int = 30
DEFAULT_BOOTSTRAP_SEED: int = 2025

# Parameter names for Step 1 MNL (6 parameters)
STEP1_PARAM_NAMES: List[str] = [
    "B_TT", "B_CO", "B_HE", "B_SEATS", "ASC_SM", "ASC_CAR"
]

# =============================================================================
# Synthetic Generation Configuration
# =============================================================================

DEFAULT_SYNTH_SIZE: int = 2000
DEFAULT_P_UNSEEN: float = 0.2  # Probability of sampling unseen demographic combinations
DEFAULT_TAU: float = 1.0  # Softmax temperature
DEFAULT_LAMBDA: float = 0.3  # Residual perturbation strength

# =============================================================================
# Paths and Environment
# =============================================================================

def get_data_dir() -> str:
    """Get the data directory path.

    First checks for SWISSMETRO_DATA_DIR environment variable,
    then falls back to default locations.
    """
    env_path = os.environ.get("SWISSMETRO_DATA_DIR")
    if env_path and os.path.exists(env_path):
        return env_path

    # Default fallback paths
    default_paths = [
        "./data",
        "../data",
        os.path.expanduser("~/swissmetro_data"),
    ]

    for path in default_paths:
        if os.path.exists(path):
            return path

    return "./data"  # Return default even if doesn't exist

def get_output_dir() -> str:
    """Get the output directory for generated files."""
    env_path = os.environ.get("SWISSMETRO_OUTPUT_DIR")
    if env_path:
        os.makedirs(env_path, exist_ok=True)
        return env_path

    default_path = "./output"
    os.makedirs(default_path, exist_ok=True)
    return default_path
