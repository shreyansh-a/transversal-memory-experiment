import numpy as np
from steane_gate.util import _Circuit as Circuit
from steane_gate.util import get_schedule
import stim

def ghz_layers(qubits, method='bisect'):
    """
    Returns the CNOT layers, as lists of (control, target) pairs, that copy
    qubits[0] onto the rest of `qubits` to prepare a GHZ state.

    method='ladder': qubits[0] fans out one CNOT per layer (depth n-1).
    method='bisect': depth ceil(log2 n). Each qubit owns a contiguous range
        [lo, hi) of the block and copies itself to the midpoint, so every
        subtree is a contiguous range. A fault during preparation then flips
        a contiguous block of ancilla outcomes and only a few stabilizer
        parities, like the ladder does. A breadth-first fan-out (0->1,
        then 0->2 and 1->3, ...) has the same depth but spreads faults to
        scattered qubits, which flip almost every parity and badly hurt decoding.
    """
    qubits = list(qubits)
    if method == 'ladder':
        return [[(qubits[0], q)] for q in qubits[1:]]
    if method != 'bisect':
        raise ValueError("Invalid GHZ preparation method specified.")
    layers, intervals = [], [(0, len(qubits))]
    while any(hi - lo > 1 for lo, hi in intervals):
        layer, split = [], []
        for lo, hi in intervals:
            if hi - lo > 1:
                mid = (lo + hi) // 2
                layer.append((qubits[lo], qubits[mid]))
                split += [(lo, mid), (mid, hi)]
            else:
                split.append((lo, hi))
        layers.append(layer)
        intervals = split
    return layers


def circuit_transversal(distance=3, rounds=5, basis='Z', ghz='ladder', blocks=1):
    """
    Constructs the circuit for memory experiment on the repetetion code of
    given distance for given rounds, following the standard convention:
    basis='Z' prepares and measures logical Z, i.e. the bit-flip code
    (stabilizers Z_iZ_{i+1}, protects against X errors);
    basis='X' prepares and measures logical X, i.e. the phase-flip code
    (stabilizers X_iX_{i+1}, protects against Z errors).

    Each round prepares an ancilla block in the logical |+> state (a GHZ
    state), couples it to the data with a transversal CNOT and measures it.

    ghz:    'ladder' (depth d-1) or 'bisect' (depth ceil(log2 d)) GHZ
            preparation, see ghz_layers().
    blocks: number of ancilla blocks used in rotation. With blocks > 1 the
            GHZ states for later rounds are prepared while the current round
            runs, so a new round can start every ceil(L / blocks) ticks
            instead of every L ticks (L = length of one block's cycle).
            Less time between rounds means less idle noise on the data.
            Uses distance * (1 + blocks) qubits.
    """
    if basis not in ('X', 'Z'):
        raise ValueError("Invalid basis specified.")
    if blocks < 1:
        raise ValueError("Need at least one ancilla block.")

    data = list(range(distance))
    ancillas = [list(range(distance * (1 + b), distance * (2 + b))) for b in range(blocks)]
    # basis 'X' is the basis 'Z' circuit conjugated by H on every qubit:
    # resets swap R <-> RX and every CNOT is reversed
    meas = 'M' if basis == 'Z' else 'MX'
    root_reset, other_reset = ('RX', 'R') if basis == 'Z' else ('R', 'RX')

    def cnot(c, t):
        return (c, t) if basis == 'Z' else (t, c)

    circuit = Circuit(distance * (1 + blocks))
    if basis == 'X':
        # prepare the data block in |+_L> of the phase-flip code
        circuit.op('H', data)
        circuit.tick()

    # One block's cycle: reset (root straight into |+>, which merges the
    # usual R + H), GHZ layers, transversal CNOT, measurement.
    num_layers = len(ghz_layers(ancillas[0], ghz))
    cycle = num_layers + 3
    period = max(1, -(-cycle // blocks))   # ticks between consecutive rounds
    schedule = [[] for _ in range((rounds - 1) * period + cycle)]
    for r in range(rounds):
        anc = ancillas[r % blocks]
        start = r * period
        schedule[start].append(('reset', anc))
        for i, layer in enumerate(ghz_layers(anc, ghz)):
            schedule[start + 1 + i].append(('cx', [cnot(c, t) for c, t in layer]))
        schedule[start + cycle - 2].append(('cx', [cnot(q, a) for q, a in zip(data, anc)]))
        schedule[start + cycle - 1].append(('measure', anc))

    prev_round = []  # stores the measurement indices of the previous round
    r = 0
    for tick_ops in schedule:
        # gates in the same tick act on disjoint qubits; measurements go
        # last so that the detectors can follow them directly
        tick_ops.sort(key=lambda op: op[0] == 'measure')
        for kind, arg in tick_ops:
            if kind == 'reset':
                circuit.op(root_reset, arg[:1])
                circuit.op(other_reset, arg[1:])
            elif kind == 'cx':
                circuit.cx(arg)
            else:
                current_round = circuit.op(meas, arg)
                # Each ancilla outcome carries a fresh random GHZ bit, so only
                # neighbouring parities (the stabilizers) are deterministic.
                # Compare each parity with the same parity from the previous round.
                for i in range(distance-1):
                    indices = [current_round[i], current_round[i+1]]
                    if prev_round:
                        indices += [prev_round[i], prev_round[i+1]]
                    circuit.detector(indices, (i, r))
                prev_round = current_round
                r += 1
        circuit.tick()

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