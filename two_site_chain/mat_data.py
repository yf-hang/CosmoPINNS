import torch

_DTYPE = torch.float32
_SHAPE = (4, 4)


def _dense_from_entries(entries):
    mat = torch.zeros(_SHAPE, dtype=_DTYPE)
    if not entries:
        return mat
    rows, cols, vals = zip(*entries)
    mat[list(rows), list(cols)] = torch.tensor(vals, dtype=_DTYPE)
    return mat


a1 = _dense_from_entries([
    (0, 0, 1.0), (0, 1, -1.0),
    (2, 2, 1.0),
    (2, 3, -1.0),
])

a2 = _dense_from_entries([
    (0, 0, 1.0), (0, 2, -1.0),
    (1, 1, 1.0),
    (1, 3, -1.0),
])

a3 = _dense_from_entries([
    (0, 1, 1.0),
    (1, 1, 1.0),
])

a4 = _dense_from_entries([
    (0, 2, 1.0),
    (2, 2, 1.0),
])

a5 = _dense_from_entries([
    (1, 3, 1.0),
    (2, 3, 1.0),
    (3, 3, 2.0),
])


# -------- eps = 0 --------
a1_eps0 = _dense_from_entries([
    (0, 1, -1.0),
    (2, 3, -1.0),
])

a2_eps0 = _dense_from_entries([
    (0, 2, -1.0),
    (1, 3, -1.0),
])

a3_eps0 = _dense_from_entries([
    (0, 1, 1.0),
])

a4_eps0 = _dense_from_entries([
    (0, 2, 1.0),
])

a5_eps0 = _dense_from_entries([
    (1, 3, 1.0),
    (2, 3, 1.0),
])


__all__ = [
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a1_eps0",
    "a2_eps0",
    "a3_eps0",
    "a4_eps0",
    "a5_eps0",
]

