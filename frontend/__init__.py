"""Front-end: Text → Structured Numerical Tensor.

This is the pure-code front-end that takes text prompts and converts them
into a fixed-size 688-dim float tensor that a back-end diffusion / flow model
consumes as conditioning.

Pipeline:
  prompt → parser → layout → encoder → tensor

The output tensor bypasses the need for text tokens in the generative model.
It provides structured hierarchical IDs (WordNet synsets), spatial positions,
part-level attributes (color, texture, position), and scene context.

Usage:
  from frontend.encoder import prompt_to_tensor
  tensor, layout_info = prompt_to_tensor("a brown dog sitting under a tree")
  # tensor.shape == (688,)
  # Feed tensor into your back-end diffusion model as conditioning
"""

from . import knowledge_base
from . import attributes
from . import parser
from . import layout
from . import encoder
from . import schema
from .encoder import prompt_to_tensor

__all__ = [
    "knowledge_base",
    "attributes",
    "parser",
    "layout",
    "encoder",
    "schema",
    "prompt_to_tensor",
]
