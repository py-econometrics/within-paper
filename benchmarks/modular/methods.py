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

# Solver-shaped aliases, for figures that key on algorithm rather than package.
METHOD_STYLE = {
    "map": (METHODS["rust-map"].colour, METHODS["rust-map"].marker),
    "lsmr_none": (METHODS["within-off"].colour, METHODS["within-off"].marker),
    "lsmr_diagonal": (
        METHODS["within-diagonal"].colour,
        METHODS["within-diagonal"].marker,
    ),
    "lsmr_factor_pair": (METHODS["within"].colour, METHODS["within"].marker),
    "fixest": (METHODS["fixest"].colour, METHODS["fixest"].marker),
    "fem": (METHODS["FEM.jl"].colour, METHODS["FEM.jl"].marker),
}

METHOD_LINESTYLE = {
    "map": METHODS["rust-map"].linestyle,
    "lsmr_none": METHODS["within-off"].linestyle,
    "lsmr_diagonal": METHODS["within-diagonal"].linestyle,
    "lsmr_factor_pair": METHODS["within"].linestyle,
    "fixest": METHODS["fixest"].linestyle,
    "fem": METHODS["FEM.jl"].linestyle,
}


def _fields(key: str) -> tuple[str, str, str]:
    method = METHODS[key]
    return method.package, method.algorithm, method.preconditioner


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


METHOD_LEGEND_LABEL = {key: legend_label(key) for key in METHODS}
METHOD_INLINE_LABEL = {key: inline_label(key) for key in METHODS}
METHOD_TABLE_HEADER = {key: table_header(key) for key in METHODS}


# The tolerance benchmark names its arms after solver and preconditioner rather
# than after the result-file backend. One alias table here keeps that experiment
# from restating the labels.
ALIASES = {
    "lsmr_off": "within-off",
    "lsmr_diagonal": "within-diagonal",
    "lsmr_additive": "within-additive",
    "pyfixest_map": "rust-map",
    "r_fixest": "fixest",
    "julia_fem": "FEM.jl",
}


def resolve(key: str) -> str:
    """Map any known alias onto its canonical result-file key."""
    return ALIASES.get(key, key)
