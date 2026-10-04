"""Frozen encoders behind one interface (encoders) and the small decoder head (decoder).

Copied from part two for the scoring step, with the encoders of part three added.
"""

LAST = "last"      # layer label of an encoder scored on its final output
