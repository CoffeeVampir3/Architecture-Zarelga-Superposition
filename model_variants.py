"""Model variants -- the single source of truth for which model we build.

A *variant* is a fully-specified ``ModelConfig`` factory. Both the trainer (main.py)
and inference (inference.py) build their model by naming a variant, so they are
guaranteed to instantiate the identical architecture from one declaration -- there is
no second place where modeling decisions can drift out of sync.

Division of responsibility:
  * ``modeling/model_config.py`` owns the *numerical decisions* and the schema: every
    architectural knob, its type, its default value, and the validation. It says what
    a model *can* be.
  * this file owns the *structural outline*: which knobs a given named model overrides
    away from the schema defaults (here, the Engram placement), composed into one
    config. It says what our model *is*. Anything not set falls through to the
    model_config defaults.

The only per-run inputs are tokenizer-derived (``vocab_size``, ``pad_token_id``);
everything architectural is fixed by the variant.
"""

from modeling.model_config import ModelConfig, EngramSettings


def base_moe(vocab_size: int, pad_token_id: int) -> ModelConfig:
    """Current model: the 8-layer fine-grained MoE (numeric spec = model_config
    defaults) with two Engram conditional-memory branches at layers 2 and 4,
    straddling the middle global / full-attention layer. Their hashed n-gram tables
    are sized at 4x the schema default rows/head, with per-head importance weighting
    (a second hash gates each head's contribution). Token Superposition Training off.
    """
    return ModelConfig(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        engram=EngramSettings(
            layers=(2, 4), rows_per_head=65536, importance_weighting=True,
            # Experiment: no learned gating. alpha is frozen at alpha_init (0.1) instead
            # of trained. head_norm caps each per-head table lookup at unit L2 norm
            # (shrink-only; zeros stay zero) so the memory magnitude stays bounded now
            # that the learned gate isn't there to attenuate it.
            head_norm=True, learned_gate=False,
        ),
    )


VARIANTS = {
    "base_moe": base_moe,
}

DEFAULT_VARIANT = "base_moe"


def build_config(vocab_size: int, pad_token_id: int,
                 variant: str = DEFAULT_VARIANT) -> ModelConfig:
    """Instantiate a named variant's ModelConfig. Shared by the trainer and inference."""
    try:
        factory = VARIANTS[variant]
    except KeyError:
        raise KeyError(
            f"unknown model variant {variant!r}; known variants: {sorted(VARIANTS)}"
        ) from None
    return factory(vocab_size, pad_token_id)
