"""Read a torch.save .pt archive without torch (contiguous tensors only)."""
import zipfile, pickle, io
import numpy as np

_DT = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64,
    "HalfStorage": np.float16, "LongStorage": np.int64,
    "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


def load_pt(path):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]

    def rebuild(storage, offset, size, stride, *_):
        dt = storage["dtype"]
        raw = z.read(f"{root}/data/{storage['key']}")
        a = np.frombuffer(raw, dtype=dt)
        size = tuple(int(s) for s in size)
        n = int(np.prod(size)) if size else 1
        a = a[offset:offset + n]
        return a.reshape(size) if size else a[0]

    def mk(mod, name):
        if name == "_rebuild_tensor_v2":
            return rebuild
        if name == "OrderedDict":
            return dict

        class Stub:
            def __init__(s, *a, **k):
                s.args = a
        Stub.__name__ = name
        return Stub

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            return mk(mod, name)

        def persistent_load(self, pid):
            # pid = ('storage', StorageTypeClass, key, location, numel)
            st = pid[1]
            nm = getattr(st, "__name__", str(st))
            return {"key": pid[2], "numel": pid[4],
                    "dtype": _DT.get(nm, np.float32)}

    return U(io.BytesIO(z.read(f"{root}/data.pkl"))).load()


if __name__ == "__main__":
    import sys
    d = load_pt(sys.argv[1] if len(sys.argv) > 1 else "2026-06-29.pt")
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            rng = f"[{v.min():.4g}, {v.max():.4g}]" if v.size and v.dtype != np.bool_ else ""
            print(f"  {k:16s} {str(v.shape):18s} {str(v.dtype):9s} {rng}")
        else:
            print(f"  {k:16s} {type(v)}")
