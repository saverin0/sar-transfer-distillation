"""The scoring step of part two, copied for part three, plus the aerial domain.

    prepared.py    the compact on-disk format of a prepared dataset (SAR float16, aerial uint8)
    s1input.py     the fixed rules from the stored channels to the encoder input + skip image
    hub.py         fetch a converted SAR dataset from its private Hugging Face copy
    run.py         frozen features -> linear probe + small decoder -> metrics
    aerial.py      the FLAIR-HUB aerial patches of part three in the prepared format

The converters of the four SAR datasets stay in part two; their converted copies are the data.
"""
