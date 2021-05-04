import numpy as np
from aicsshparam import shtools

class MeshToolKit():

    @staticmethod
    def get_mesh_from_series(row, alias, lmax):
        coeffs = np.zeros((2, lmax, lmax), dtype=np.float32)
        for l in range(lmax):
            for m in range(l + 1):
                try:
                    # Cosine SHE coefficients
                    coeffs[0, l, m] = row[
                        [f for f in row.keys() if f"{alias}_shcoeffs_L{l}M{m}C" in f]
                    ]
                    # Sine SHE coefficients
                    coeffs[1, l, m] = row[
                        [f for f in row.keys() if f"{alias}_shcoeffs_L{l}M{m}S" in f]
                    ]
                # If a given (l,m) pair is not found, it is assumed to be zero
                except: pass
        mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs)
        return mesh
