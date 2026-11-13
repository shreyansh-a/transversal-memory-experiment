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
        for pair in cnots:
            circuit.cx([pair])
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
    Constructs a repetetion circuit for memory experiment using
    standard syndrome extraction using one ancilla for each stabilizer.
    """
    circuit = Circuit(distance*2 - 1)
    
    # the ancillas are placed between the data qubits whose parity they measure
    
    if basis not in ('X', 'Z'):
        raise ValueError("Invalid basis specified.")

    data = list(range(0,distance*2-1,2))
    ancilla = list(range(1, distance*2-1,2))
    
    # basis 'X' is the basis 'Z' circuit conjugated by H on every qubit
    meas = 'M' if basis == 'Z' else 'MX'
    
    if basis == 'X':
        circuit.op('H', data)
        circuit.tick()
    
    prev_round = []
    for r in range(rounds):
        # resetting ancillas for each measurement round; done in round 0 too,
        # since the initial reset leaves them in |0>, not |+> for basis 'X'
        circuit.op('R' if basis=='Z' else 'RX', ancilla)
        circuit.tick()

        # measuring the syndromes and storing the measurement ids;
        # ancilla k sits between data[k] (left) and data[k+1] (right)
        cnots = list(zip(data[:-1],ancilla)) if basis=='Z' else list(zip(ancilla, data[:-1]))
        circuit.cx(cnots)
        circuit.tick()
        cnots = list(zip(data[1:], ancilla)) if basis=='Z' else list(zip(ancilla, data[1:]))
        circuit.cx(cnots)
        circuit.tick()
        
        curr_round = circuit.op(meas, ancilla)
        circuit.tick()
        
        for i in range(len(ancilla)):
            indices = [curr_round[i]]
            if prev_round:
                indices += [prev_round[i]]
            circuit.detector(indices, (i, r))
        
        prev_round = curr_round
        
    final = circuit.op(meas, data)
    circuit.tick()
    for i in range(distance-1):
        indices = [final[i], final[i+1], prev_round[i]]
        circuit.detector(indices, (i, rounds))
    circuit.observable([final[0]])
    return circuit.get_circuit()