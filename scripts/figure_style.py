"""Canonical public labels, colors, markers, and line styles for paper methods."""

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

# These are reader-facing names.  Raw result-file names remain internal keys, so
# regenerating a table never changes its data lookup.
METHOD_LABEL = {
    "rust-map": "PyFixest — MAP — none",
    "rust-map-matched": "PyFixest — MAP — none; matched accuracy",
    "fixest": "fixest (R) — MAP — accelerated",
    "FEM.jl": "FEM.jl — LSMR — diagonal",
    "GLFEM.jl": "GLFEM.jl — LSMR — diagonal",
    "within": "PyFixest — LSMR — factor-pair",
    "within-off": "PyFixest — LSMR — none",
    "within-diagonal": "PyFixest — LSMR — diagonal",
    "within-additive": "PyFixest — LSMR — factor-pair",
}

# Compact multi-line versions for figure legends and table headers.
METHOD_LEGEND_LABEL = {
    "rust-map": "PyFixest\nMAP\nnone",
    "rust-map-matched": "PyFixest\nMAP\nnone; matched",
    "fixest": "fixest (R)\nMAP\naccelerated",
    "FEM.jl": "FEM.jl\nLSMR\ndiagonal",
    "GLFEM.jl": "GLFEM.jl\nLSMR\ndiagonal",
    "within": "PyFixest\nLSMR\nfactor-pair",
    "within-off": "PyFixest\nLSMR\nnone",
    "within-diagonal": "PyFixest\nLSMR\ndiagonal",
    "within-additive": "PyFixest\nLSMR\nfactor-pair",
}

# One-line variants for legends that sit inside a panel.  They keep the same
# package → algorithm → preconditioner order without covering the data.
METHOD_INLINE_LABEL = {
    "rust-map": "PyFixest — MAP (none)",
    "rust-map-matched": "PyFixest — MAP (none; matched)",
    "fixest": "fixest (R) — MAP (accelerated)",
    "FEM.jl": "FEM.jl — LSMR (diagonal)",
    "GLFEM.jl": "GLFEM.jl — LSMR (diagonal)",
    "within": "PyFixest — LSMR (factor-pair)",
    "within-off": "PyFixest — LSMR (none)",
    "within-diagonal": "PyFixest — LSMR (diagonal)",
    "within-additive": "PyFixest — LSMR (factor-pair)",
}

METHOD_TABLE_HEADER = {
    key: value.replace("\n", " #linebreak() ")
    for key, value in METHOD_LEGEND_LABEL.items()
}
