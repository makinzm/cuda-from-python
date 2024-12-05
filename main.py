import numpy as np
import ctypes

# 共有ライブラリをロード
lib = ctypes.cdll.LoadLibrary('./libvector_add.so')

# 関数の引数と戻り値の型を指定
lib.vector_add.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    ctypes.c_int
]
lib.vector_add.restype = None

# データの準備
N = 1000000
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
c = np.zeros(N, dtype=np.float32)

# CUDA関数の呼び出し
lib.vector_add(a, b, c, N)

# 結果の検証
if np.allclose(c, a + b):
    print("計算結果は正しいです。")
else:
    print("計算結果に誤りがあります。")

