import numpy as np
from steane_gate.util import _Circuit as Circuit
from steane_gate.util import get_schedule
import stim

def circuit_transversal(distance=3, rounds=5, basis='X'):
    """
    Constructs the circuit for memory experiment on bit-flip(if basis='X')
    or phase-flip(if basis='Z') repetetiion code of given distance for
    given rounds.
    """
    circuit = Circuit(distance*2)
    if basis == 'X':
        # encode the measurement block in \instate{+} state
        circuit.op('H', [distance])
        circuit.tick()
        cnots = [(distance, i) for i in range(distance+1, distance*2)]
        circuit.cx(get_schedule(cnots))
    elif basis == 'Z':
        # prepare the circuit for phase-flip protection
        circuit.op('H', [i for i in range(distance)])
    else:
        raise ValueError("Invalid basis specified.")
    measurements = [(i, i+distance) if basis=='X' else (i+distance, i) for i in range(distance)]
    # transversal measurements
    circuit.cx(measurements)
    
    ...
    
    return circuit.get_circuit()
    
def circuit_standard(distance=3, rounds=5, basis='X'):
    """
    Constructs a repetetiion circuit for memory experiment using
    standard syndrome extraction using one ancilla for each stabilizer.
    """
    circuit = Circuit(distance*2 - 1)
    
    ...
    
    return circuit.get_circuit()