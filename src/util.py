import deltakit_stim
import stim
from deltakit.circuit import Circuit as DeltakitCircuit
from deltakit.circuit import GateLayer

_ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS"}


def to_deltakit(circuit: stim.Circuit) -> DeltakitCircuit:
    """
    Converts a stim circuit into a deltakit circuit with one GateLayer per TICK.

    deltakit's own Circuit.from_stim_circuit puts every stim instruction in
    its own layer, so gates of different types in the same TICK (e.g. a reset
    on one block while another block is measured) end up in separate layers.
    The noise models then add spurious idle noise. Here all gates between
    two TICKs (which must act on disjoint qubits) form a single layer.
    Detectors and observables keep their position before or after that layer.
    """
    layers = []

    def flush(before, gates, after):
        for part in (before, None, after):
            if part is None:
                if gates:
                    parsed = DeltakitCircuit.from_stim_circuit(deltakit_stim.Circuit(str(gates)))
                    layers.append(GateLayer([g for layer in parsed.layers for g in layer.gates]))
            elif part:
                layers.extend(DeltakitCircuit.from_stim_circuit(deltakit_stim.Circuit(str(part))).layers)

    before, gates, after = stim.Circuit(), stim.Circuit(), stim.Circuit()
    for inst in circuit.flattened():
        if inst.name == "TICK":
            flush(before, gates, after)
            before, gates, after = stim.Circuit(), stim.Circuit(), stim.Circuit()
        elif inst.name in _ANNOTATIONS:
            (after if len(gates) else before).append(inst)
        else:
            gates.append(inst)
    flush(before, gates, after)
    return DeltakitCircuit(layers)
