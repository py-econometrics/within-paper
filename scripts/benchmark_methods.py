"""Paper labels and styles for canonical benchmark backend names."""

METHODS = {
    "rust-map": ("PyFixest MAP", "#166534", "o", "-"),
    "within": ("PyFixest LSMR, factor-pair", "#16a34a", "D", "-"),
    "within-off": ("PyFixest LSMR, none", "#65a30d", "X", "--"),
    "within-diagonal": ("PyFixest LSMR, diagonal", "#0f766e", "^", "-."),
    "within-additive": ("PyFixest LSMR, factor-pair", "#16a34a", "D", "-"),
    "fixest": ("fixest", "#2563eb", "s", "-"),
    "FEM.jl": ("FEM.jl", "#ca8a04", "P", "-"),
    "GLFEM.jl": ("GLFEM.jl", "#ca8a04", "P", "-"),
}
