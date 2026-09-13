# Comparison identity review

Ranking previously compared truncated PA checkpoint hashes and rounded reference
gains. Two distinct experiments could therefore appear comparable. The comparison
key now uses complete hashes, exact stored gains, raw and processed data identity,
reference rules and the evaluated sample interval.

When present, frozen signal metadata, declared RF conditions and measurement
processing/calibration also participate in the key. Missing fields remain readable
on historical results. A real measured result without declared output power cannot
establish physical power matching; synthetic/mock results retain their explicit
evidence classification.

This change only determines whether results may be ranked. It does not recompute
NMSE, ACPR, EVM or other metrics, alter their definitions, change data splits or
relax tolerances. The frozen metric golden tests and explicit identity regression
tests are the acceptance checks for this separate scientific-path PR. Physical
measurement validation remains an external requirement.
