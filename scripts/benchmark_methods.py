"""Paper labels and styles for canonical benchmark backend names."""

METHODS = {
    "rust-map": ("PyFixest MAP", "#166534", "o", "-"),
    "within": ("PyFixest within", "#16a34a", "D", "-"),
    "within-reuse": ("within (reuse)", "#16a34a", "D", "-"),
    "within-rebuild": ("within (rebuild)", "#15803d", "d", "--"),
    "within-off": ("LSMR, none", "#65a30d", "X", "--"),
    "within-diagonal": ("LSMR, diagonal", "#0f766e", "^", "-."),
    "within-additive": ("LSMR, factor-pair", "#16a34a", "D", "-"),
    "fixest": ("fixest", "#2563eb", "s", "-"),
    "FEM.jl": ("FEM.jl", "#ca8a04", "P", "-"),
    "GLFEM.jl": ("GLFEM.jl", "#ca8a04", "P", "-"),
}
