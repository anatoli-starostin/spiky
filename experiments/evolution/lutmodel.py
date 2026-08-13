"""Shared toy-LUT model — a Python port of lut_spiking.js buildModel/lutBits,
so the evolution experiments use the SAME seed-0 table as the visualiser.
Binary input enumeration: x in {-1,+1}^6 -> 64 fully-enumerable inputs."""


def mulberry32(seed):
    s = [seed & 0xffffffff]

    def rng():
        s[0] = (s[0] + 0x6d2b79f5) & 0xffffffff
        t = s[0]
        t = (t ^ (t >> 15))
        t = (t * (t | 1)) & 0xffffffff
        t ^= (t + ((t ^ (t >> 7)) * (t | 61) & 0xffffffff)) & 0xffffffff
        t &= 0xffffffff
        return ((t ^ (t >> 14)) & 0xffffffff) / 4294967296
    return rng


def make_normal(rng):
    spare = [None]

    def n():
        if spare[0] is not None:
            v = spare[0]
            spare[0] = None
            return v
        while True:
            u = 2 * rng() - 1
            v = 2 * rng() - 1
            s = u * u + v * v
            if 0 < s < 1:
                break
        m = (-2.0 * __import__('math').log(s) / s) ** 0.5
        spare[0] = v * m
        return u * m
    return n


def build_model(seed=0, D=6, K=3, Dout=4):
    rng = mulberry32(seed)
    N = make_normal(rng)
    W = [[N() for _ in range(D)] for _ in range(K)]
    b = [N() for _ in range(K)]
    V = [[N() for _ in range(Dout)] for _ in range(1 << K)]
    return {"W": W, "b": b, "V": V, "D": D, "K": K, "Dout": Dout}


def lut_bits(m, x):
    return [1 if (sum(m["W"][k][i] * x[i] for i in range(m["D"])) + m["b"][k]) > 0 else 0
            for k in range(m["K"])]


def bits_to_row(bits):
    r = 0
    for bb in bits:
        r = (r << 1) | bb
    return r


def all_inputs(D=6):
    """the 64 binary inputs x in {-1,+1}^D, fully enumerable."""
    out = []
    for code in range(1 << D):
        out.append([1 if (code >> (D - 1 - i)) & 1 else -1 for i in range(D)])
    return out


def lut_spec(m):
    """the complete input->output specification: for each of the 64 inputs, the
    LUT row index and stored output vector. This IS the exact fitness oracle."""
    spec = []
    for x in all_inputs(m["D"]):
        bits = lut_bits(m, x)
        r = bits_to_row(bits)
        spec.append({"x": x, "bits": bits, "row": r, "out": m["V"][r]})
    return spec
