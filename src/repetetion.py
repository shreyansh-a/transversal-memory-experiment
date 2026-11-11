import numpy as np
from steane_gate.util import _Circuit as Circuit
from steane_gate.util import get_schedule
import stim

def circuit_transversal(distance=3, rounds=5, basis='Z'):
    """
    Constructs the circuit for memory experiment on the repetetion code of
    given distance for given rounds, following the standard convention:
    basis='Z' prepares and measures logical Z, i.e. the bit-flip code
    (stabilizers Z_iZ_{i+1}, protects against X errors);
    basis='X' prepares and measures logical X, i.e. the phase-flip code
    (stabilizers X_iX_{i+1}, protects against Z errors).
    """
    if basis not in ('X', 'Z'):
        raise ValueError("Invalid basis specified.")

    data = list(range(distance))
    ancilla = list(range(distance, distance*2))
    # basis 'X' is the basis 'Z' circuit conjugated by H on every qubit
    meas = 'M' if basis == 'Z' else 'MX'

    circuit = Circuit(distance*2)
    if basis == 'X':
        # prepare the data block in |+_L> of the phase-flip code
        circuit.op('H', data)
        circuit.tick()

    prev_round = []  # stores the measurement indices of the previous round

    for r in range(rounds):
        if r > 0:
            # ancilla block still holds last round's outcomes
            circuit.op('R', ancilla)
            circuit.tick()

        # encode the ancilla block in |+_L> = GHZ state of the bit-flip code
        circuit.op('H', [ancilla[0]])
        circuit.tick()
        cnots = [(ancilla[0], a) for a in ancilla[1:]]
        for pairs in get_schedule(cnots):
            circuit.cx(pairs)
            circuit.tick()
        if basis == 'X':
            # rotate to |+_L> of the phase-flip code
            circuit.op('H', ancilla)
            circuit.tick()

        # transversal CNOT: data -> ancilla ('Z'), ancilla -> data ('X')
        circuit.cx([(q, a) if basis == 'Z' else (a, q) for q, a in zip(data, ancilla)])
        circuit.tick()

        current_round = circuit.op(meas, ancilla)
        circuit.tick()

        # Each ancilla outcome carries a fresh random GHZ bit, so only
        # neighbouring parities (the stabilizers) are deterministic.
        # Compare each parity with the same parity from the previous round.
        for i in range(distance-1):
            indices = [current_round[i], current_round[i+1]]
            if prev_round:
                indices += [prev_round[i], prev_round[i+1]]
            circuit.detector(indices, (i, r))
        prev_round = current_round

    # final data measurement: data parities compared with the last syndrome round
    final = circuit.op(meas, data)
    for i in range(distance-1):
        circuit.detector([final[i], final[i+1], prev_round[i], prev_round[i+1]], (i, rounds))
    circuit.observable([final[0]])
    return circuit.get_circuit()
    
def circuit_standard(distance=3, rounds=5, basis='Z'):
    """
    Constructs a repetetiion circuit for memory experiment using
    standard syndrome extraction using one ancilla for each stabilizer.
    """
    circuit = Circuit(distance*2 - 1)
    
    ...
    
    return circuit.get_circuit()