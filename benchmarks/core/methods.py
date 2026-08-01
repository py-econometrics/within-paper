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

# Spellings recorded by runs made before the names were unified on 2026-08-01.
#
# Nothing in the codebase produces any of these any more: every driver records
# the canonical key above, and a test enforces it. They stay only so the
# existing result CSVs, which are untracked and cannot be regenerated cheaply,
# still render. Delete this table after the next full sweep.
#
# Note how many of them named the same thing. "rust-map" alone was spelled
# five ways, one per driver that happened to write it.
LEGACY_SPELLINGS = {
    "pyfixest (rust-map)": "rust-map",
    "pyfixest-map": "rust-map",
    "pyfixest_map": "rust-map",
    "rust": "rust-map",
    "pyfixest (rust-map, matched)": "rust-map-matched",
    "pyfixest (within)": "within",
    "pyfixest-within": "within",
    "pyfixest (within-off)": "within-off",
    "lsmr_off": "within-off",
    "pyfixest (within-diagonal)": "within-diagonal",
    "lsmr_diagonal": "within-diagonal",
    "pyfixest (within-additive)": "within-additive",
    "lsmr_additive": "within-additive",
    "fixest-map": "fixest",
    "fixest-fepois": "fixest",
    "r_fixest": "fixest",
    "FEM.jl (lsmr)": "FEM.jl",
    "FixedEffectModels": "FEM.jl",
    "julia_fem": "FEM.jl",
    "Julia": "FEM.jl",
    "glfixedeffectmodels.jl": "GLFEM.jl",
}

ALIASES = LEGACY_SPELLINGS


def canonical(key: str) -> str | None:
    """The canonical key for a recorded backend name, or None if unknown.

    Matching is exact. paper_results used to do this by substring, which is how
    "PyFixest MAP" mapped to R fixest: "pyfi(xest)" contains "fixest". Nothing
    in the tracked results hits that case today, but the class of bug is only
    avoidable by matching whole names.

    Returning None rather than raising keeps a result file that mentions an
    unregistered backend from aborting a render; the row is skipped, and a test
    asserts every backend in the tracked results is registered.
    """
    resolved = ALIASES.get(key, key)
    return resolved if resolved in METHODS else None


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
