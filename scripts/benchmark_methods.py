"""Paper labels and styles for canonical benchmark backend names."""

METHODS = {
    "rust-map": ("PyFixest MAP", "#166534", "o", "-"),
    "within": ("PyFixest within", "#16a34a", "D", "-"),
    "within-off": ("LSMR, none", "#65a30d", "X", "--"),
    "within-diagonal": ("LSMR, diagonal", "#0f766e", "^", "-."),
    "within-additive": ("LSMR, factor-pair", "#16a34a", "D", "-"),
    "fixest": ("fixest", "#2563eb", "s", "-"),
    "FEM.jl": ("FEM.jl", "#ca8a04", "P", "-"),
    "GLFEM.jl": ("GLFEM.jl", "#ca8a04", "P", "-"),
}


def style(key: str) -> tuple[str, str]:
    return METHODS[key][1:3]


def linestyle(key: str) -> str:
    return METHODS[key][3]


def legend_label(key: str) -> str:
    return METHODS[key][0]


def inline_label(key: str) -> str:
    return METHODS[key][0]


def table_header(key: str) -> str:
    label = METHODS[key][0]
    return label.replace(" ", " #linebreak() ", 1) if " " in label else label


METHOD_TABLE_HEADER = {key: table_header(key) for key in METHODS}
