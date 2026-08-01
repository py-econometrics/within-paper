"""One record per solver configuration, and every name and style derived from it.

A configuration is identified by three things a reader cares about: the package
that ran it, the algorithm it used, and the preconditioner it applied. Every
label in the paper is one of those three fields rendered differently, so they
are formatted here rather than written out per call site.

The dictionary keys are the raw names that appear in the result files. Those are
internal and never change, so renaming a label never changes which cell a table
reads.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    package: str
    algorithm: str
    preconditioner: str
    colour: str
    marker: str
    linestyle: str = "-"


# Python configurations share a green family and are separated by marker and
# dash pattern; R fixest is blue; the Julia packages are gold.
METHODS = {
    "rust-map": Method("PyFixest", "MAP", "none", "#166534", "o"),
    "rust-map-matched": Method("PyFixest", "MAP", "none; matched", "#166534", "o"),
    "within-off": Method("PyFixest", "LSMR", "none", "#65a30d", "X", "--"),
    "within-diagonal": Method("PyFixest", "LSMR", "diagonal", "#0f766e", "^", "-."),
    "within-additive": Method("PyFixest", "LSMR", "factor-pair", "#16a34a", "D"),
    "within": Method("PyFixest", "LSMR", "factor-pair", "#16a34a", "D"),
    "fixest": Method("fixest (R)", "MAP", "accelerated", "#2563eb", "s"),
    "FEM.jl": Method("FEM.jl", "LSMR", "diagonal", "#ca8a04", "P"),
    "GLFEM.jl": Method("GLFEM.jl", "LSMR", "diagonal", "#ca8a04", "P"),
}

# The tolerance benchmark, the sweep result files, and the figures each name
# these configurations differently. One table maps every spelling onto the
# canonical key, so a caller never has to hold a second mapping of its own.
ALIASES = {
    # Tolerance benchmark arm names.
    "lsmr_off": "within-off",
    "lsmr_diagonal": "within-diagonal",
    "lsmr_additive": "within-additive",
    "pyfixest_map": "rust-map",
    "r_fixest": "fixest",
    "julia_fem": "FEM.jl",
    # Backend names as the sweep drivers record them in the result files.
    "pyfixest (rust-map)": "rust-map",
    "pyfixest (rust-map, matched)": "rust-map-matched",
    "pyfixest (within)": "within",
    "pyfixest (within-off)": "within-off",
    "pyfixest (within-diagonal)": "within-diagonal",
    "pyfixest (within-additive)": "within-additive",
    "fixest-map": "fixest",
    "FEM.jl (lsmr)": "FEM.jl",
    # The crossover figure draws one series for both Julia packages, which
    # share a style. Its own label is set at the call site.
    "Julia": "FEM.jl",
}


def resolve(key: str) -> str:
    """Map any known alias onto its canonical result-file key."""
    return ALIASES.get(key, key)


def method(key: str) -> Method:
    """The record for a canonical key or any of its aliases."""
    return METHODS[resolve(key)]


def style(key: str) -> tuple[str, str]:
    """Colour and marker, as matplotlib takes them."""
    record = method(key)
    return record.colour, record.marker


def linestyle(key: str) -> str:
    return method(key).linestyle


def _fields(key: str) -> tuple[str, str, str]:
    record = method(key)
    return record.package, record.algorithm, record.preconditioner


def legend_label(key: str) -> str:
    """Three stacked lines, for a legend outside the axes or a table header."""
    return "\n".join(_fields(key))


def inline_label(key: str) -> str:
    """One line, for a legend that sits inside a panel."""
    package, algorithm, preconditioner = _fields(key)
    return f"{package} — {algorithm} ({preconditioner})"


def table_header(key: str) -> str:
    """Three lines as Typst markup, for a generated table header."""
    return legend_label(key).replace("\n", " #linebreak() ")


# Keyed by canonical name only. paper_results looks table cells up here with
# `.get(cell, cell)`, so adding aliases would let an ordinary cell that happens
# to read "Julia" be rewritten as a method label.
METHOD_TABLE_HEADER = {key: table_header(key) for key in METHODS}
