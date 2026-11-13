"""Circuit layer (task 4): schematic -> SPICE, deck assembly, ngspice batch runner."""

from magsim.circuit.netlist import build_deck, coil_vectors, export_spice, load_circuit, pwl
from magsim.circuit.ngspice import NgspiceError, ngspice_available, parse_wrdata, run_ngspice

__all__ = [
    "NgspiceError",
    "build_deck",
    "coil_vectors",
    "export_spice",
    "load_circuit",
    "ngspice_available",
    "parse_wrdata",
    "pwl",
    "run_ngspice",
]
