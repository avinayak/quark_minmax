
import numpy as np
import struct

def encode(mat):
    n_rows = len(mat)
    n_columns = len(mat[0])
    shape = struct.pack('>II', n_rows, n_columns)
    return shape + mat.tobytes()

def decode(data,sh):
    n_rows, n_columns = struct.unpack('>II', data[:8])
    mat = np.frombuffer(data, dtype=np.int8, offset=8)
    mat.shape = (n_rows, n_columns)
    return mat