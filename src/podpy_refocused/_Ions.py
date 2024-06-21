# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
from enum import Enum

class Ion(Enum):
    H_I    = "h1"
    C_II   = "c2"
    C_III  = "c3"
    C_IV   = "c4"
    N_V    = "n5"
    O_I    = "o1"
    O_VI   = "o6"
    Si_II  = "si2"
    Si_III = "si3"
    Si_IV  = "si4"

    def __str__(self) -> str:
        return self.value
