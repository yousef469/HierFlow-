"""Back-end: Converts 688-dim tensors to images via flow matching.

The front-end produces a 688-dim numerical contract (no text tokens).
This module learns to render images conditioned on that tensor.
"""

from .model import ConditionalUNet
