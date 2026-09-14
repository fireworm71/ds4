"""Minimal GGUF tensor extractor: walk the tensor table, read named tensors.

Handles F32, F16, Q8_0 (block 32: f16 scale + 32 int8). Reads only the
requested tensors via seek, never the whole file.
"""
import struct, numpy as np

T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
FIXED = {T_U8: 'B', T_I8: 'b', T_U16: 'H', T_I16: 'h', T_U32: 'I',
         T_I32: 'i', T_F32: 'f', T_BOOL: 'B', T_U64: 'Q', T_I64: 'q', T_F64: 'd'}
GGML_F32, GGML_F16, GGML_Q8_0 = 0, 1, 8
TYPE_SIZE = {GGML_F32: (1, 4), GGML_F16: (1, 2), GGML_Q8_0: (32, 34)}

class GGUF:
    def __init__(self, path):
        self.f = open(path, 'rb')
        f = self.f
        assert f.read(4) == b'GGUF'
        ver, n_tensors, n_kv = struct.unpack('<IQQ', f.read(20))
        self.align = 32
        for _ in range(n_kv):
            k = self._s()
            (t,) = struct.unpack('<I', f.read(4))
            v = self._val(t)
            if k == 'general.alignment':
                self.align = int(v)
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._s()
            (nd,) = struct.unpack('<I', f.read(4))
            dims = struct.unpack('<%dQ' % nd, f.read(8 * nd))
            ttype, off = struct.unpack('<IQ', f.read(12))
            self.tensors[name] = (dims, ttype, off)
        pos = f.tell()
        self.data0 = (pos + self.align - 1) // self.align * self.align

    def _s(self):
        (n,) = struct.unpack('<Q', self.f.read(8))
        return self.f.read(n).decode('utf-8', 'replace')

    def _val(self, t):
        f = self.f
        if t in FIXED:
            return struct.unpack('<' + FIXED[t], f.read(struct.calcsize(FIXED[t])))[0]
        if t == T_STR:
            return self._s()
        if t == T_ARR:
            (et,) = struct.unpack('<I', f.read(4))
            (n,) = struct.unpack('<Q', f.read(8))
            if et == T_STR:
                return [self._s() for _ in range(n)]
            f.seek(n * struct.calcsize(FIXED[et]), 1)
            return None
        raise ValueError(t)

    def read(self, name):
        dims, ttype, off = self.tensors[name]
        n = 1
        for d in dims:
            n *= d
        blk, bsz = TYPE_SIZE[ttype]
        nbytes = n // blk * bsz
        self.f.seek(self.data0 + off)
        raw = self.f.read(nbytes)
        if ttype == GGML_F32:
            a = np.frombuffer(raw, dtype='<f4').astype(np.float64)
        elif ttype == GGML_F16:
            a = np.frombuffer(raw, dtype='<f2').astype(np.float64)
        else:  # Q8_0
            rec = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
            scale = rec[:, :2].copy().view('<f2').astype(np.float64).reshape(-1, 1)
            q = rec[:, 2:].copy().view(np.int8).astype(np.float64)
            a = (q * scale).reshape(-1)
        # GGUF dims are [ne0(fastest), ne1, ...]; row-major matrix = (ne1, ne0)
        if len(dims) == 2:
            return a.reshape(dims[1], dims[0])
        return a
