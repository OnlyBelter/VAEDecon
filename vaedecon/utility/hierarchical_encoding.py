SORTED_CELL_TYPES = [
    "NK",
    "Non-plasma B cells",
    "Plasma B cells",
    "CD4 T",
    "CD8 T (GZMK high)",
    "CD8 T effector",
    "Double-neg-like T",
    "Monocytes",
    "Macrophages",
    "DC",
    "Mast Cells",
    "Neutrophils",
    "Endothelial Cells",
    "CAFs",
    "Myofibroblasts",
    "Cancer Cells",
]

HIERARCHICAL_ENCODING = {
    "NK":                 [1, 0, 0, 0, 0, 0, 0, 0],  # Lymphoid cells
    "Non-plasma B cells": [1, 0, 0, 0, 1, 0, 0, 0],  # B cells
    "Plasma B cells":     [1, 0, 0, 0, 1, 0, 0, 0],
    "CD4 T":              [1, 0, 0, 0, 0, 1, 0, 0],  # T cells
    "CD8 T (GZMK high)":  [1, 0, 0, 0, 0, 1, 0, 0],
    "CD8 T effector":     [1, 0, 0, 0, 0, 1, 0, 0],
    "Double-neg-like T":  [1, 0, 0, 0, 0, 1, 0, 0],
    "Monocytes":          [0, 1, 0, 0, 0, 0, 1, 0],  # Monocytic cells
    "Macrophages":        [0, 1, 0, 0, 0, 0, 1, 0],
    "DC":                 [0, 1, 0, 0, 0, 0, 0, 0],  # Myeloid cells
    "Mast Cells":         [0, 1, 0, 0, 0, 0, 0, 0],
    "Neutrophils":        [0, 1, 0, 0, 0, 0, 0, 0],
    "Endothelial Cells":  [0, 0, 1, 0, 0, 0, 0, 0],  # Stromal cells
    "CAFs":               [0, 0, 1, 0, 0, 0, 0, 1],  # Fibroblasts
    "Myofibroblasts":     [0, 0, 1, 0, 0, 0, 0, 1],
    "Cancer Cells":       [0, 0, 0, 1, 0, 0, 0, 0],
}
