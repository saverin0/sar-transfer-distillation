"""Distillation of one student from several frozen satellite teachers.

records    shared data records (one Sentinel-2 cell, one Sentinel-1 cell, one pair, one skip)
majortom   index of the Major-TOM cells and single-cell reads over HTTP
pairing    the pairing rules, the offset rule and the band shift rule
aerial     the FLAIR-HUB aerial sets (RGB + NIR at 0.4 m)
pilotdata  the pilot fetch in chunks, the held-out split, channel statistics, pairs in RAM
teachers   the frozen teachers behind one interface
student    the student with its three input layers, its heads and the losses
phis       PHI-S standardisation of the teachers' patch features
train      the distillation loop, checkpoints and the held-out loss
stage0     the measurements and checks of stage 0
score      the scoring of students and baselines with part two's run step
"""
