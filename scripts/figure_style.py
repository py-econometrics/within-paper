"""Canonical colors, markers, and line styles for the paper's method figures."""

METHOD_STYLE = {
    # Python / PyFixest: a green family, differentiated further by marker and line.
    "map": ("#166534", "o"),
    "lsmr_none": ("#65a30d", "X"),
    "lsmr_diagonal": ("#0f766e", "^"),
    "lsmr_factor_pair": ("#16a34a", "D"),
    # R fixest: blue.
    "fixest": ("#2563eb", "s"),
    # FEM.jl and GLFixedEffectModels.jl: yellow / gold.
    "fem": ("#ca8a04", "P"),
}

METHOD_LINESTYLE = {
    "map": "-",
    "lsmr_none": "--",
    "lsmr_diagonal": "-.",
    "lsmr_factor_pair": "-",
    "fixest": "-",
    "fem": "-",
}
