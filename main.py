#!/usr/bin/env python3
"""
Aurora Vortex -- procedural generative art (make.fun "coolest art" entry).

STANDARD LIBRARY ONLY. No Pillow, no numpy: the image is computed in a
plain bytearray pixel buffer and written as a PNG with a hand-rolled
zlib/struct chunk encoder, so the program runs in a bare Python 3.13
container.

A seeded value-noise flow field drives thousands of luminous strokes that
swirl around an off-centre vortex above a glowing core. A low-resolution
glow buffer (nebula + stroke halos) is box-blurred and upscaled; sharp
core strokes, a golden bloom, stars, vignette and film grain finish it.

Deterministic: fixed seeds everywhere, byte-identical output every run.
"""
import binascii
import math
import os
import random
import struct
import zlib

SEED = 20261113
N = 1024                     # output size (square)
GLOW = 512                   # low-res glow buffer size

# ------------------------------------------------------------ PNG writer --
def write_png_palette(path, w, h, buf):
    """Palette PNG (color type 3): posterize to 5 bits per channel, keep
    the 256 most frequent colors, map the rest to their nearest palette
    color. Truecolor filtering still left this noisy particle field at
    2.0 MB; indexed color brings it under the 1 MiB cap."""
    stride = w * 3

    def chunk(typ, data):
        return (struct.pack(">I", len(data)) + typ + data +
                struct.pack(">I", binascii.crc32(typ + data) & 0xFFFFFFFF))

    counts = {}
    keys = bytearray(w * h * 2)  # not used; placeholder replaced below
    pix_keys = [0] * (w * h)
    for i in range(w * h):
        r = buf[i * 3] >> 3
        g = buf[i * 3 + 1] >> 3
        b = buf[i * 3 + 2] >> 3
        k = (r << 10) | (g << 5) | b
        pix_keys[i] = k
        counts[k] = counts.get(k, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:256]
    palette = []
    index_of = {}
    for idx, (k, _) in enumerate(top):
        r = (k >> 10) & 31
        g = (k >> 5) & 31
        b = k & 31
        palette.append((((r << 3) | (r >> 2)), ((g << 3) | (g >> 2)), ((b << 3) | (b >> 2))))
        index_of[k] = idx
    pal_set = set(index_of)
    # Nearest palette color for every posterized key not in the palette.
    nearest = {}
    pal_keys = [k for k, _ in top]
    for k in counts:
        if k in pal_set:
            continue
        r = (k >> 10) & 31
        g = (k >> 5) & 31
        b = k & 31
        best, bestd = 0, 1 << 30
        for pk in pal_keys:
            dr = ((pk >> 10) & 31) - r
            dg = ((pk >> 5) & 31) - g
            db = (pk & 31) - b
            d = dr * dr + dg * dg + db * db
            if d < bestd:
                bestd, best = d, pk
        nearest[k] = index_of[best]
    lookup = [0] * 32768
    for k, idx in index_of.items():
        lookup[k] = idx
    for k, idx in nearest.items():
        lookup[k] = idx

    raw = bytearray()
    prev = bytes(w)
    for y in range(h):
        row = bytes(lookup[pix_keys[y * w + x]] for x in range(w))
        cands = [(0, row),
                 (1, bytes((row[i] - (row[i - 1] if i else 0)) & 0xFF for i in range(w))),
                 (2, bytes((row[i] - prev[i]) & 0xFF for i in range(w)))]

        def cost(b):
            return sum(x if x < 128 else 256 - x for x in b)

        ftype, fdata = min(cands, key=lambda c: cost(c[1]))
        raw.append(ftype)
        raw += fdata
        prev = row

    plte = bytearray()
    for r, g, b in palette:
        plte += bytes((r, g, b))
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0))
    png += chunk(b"PLTE", bytes(plte))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def write_png(path, w, h, buf):
    stride = w * 3

    def chunk(typ, data):
        return (struct.pack(">I", len(data)) + typ + data +
                struct.pack(">I", binascii.crc32(typ + data) & 0xFFFFFFFF))

    # Per-row adaptive filtering (None/Sub/Up) by the usual minimum
    # sum-of-absolute-signed-bytes heuristic, then zlib level 9: the
    # particle field is noisy, and this keeps the PNG under the
    # competition's 1 MiB cap without changing a single pixel.
    raw = bytearray()
    prev = bytes(stride)
    for y in range(h):
        row = bytes(buf[y * stride:(y + 1) * stride])
        cands = []
        cands.append((0, row))
        sub = bytearray(stride)
        for i in range(stride):
            left = row[i - 3] if i >= 3 else 0
            sub[i] = (row[i] - left) & 0xFF
        cands.append((1, bytes(sub)))
        up = bytes((row[i] - prev[i]) & 0xFF for i in range(stride))
        cands.append((2, up))

        def cost(b):
            return sum(x if x < 128 else 256 - x for x in b)

        ftype, fdata = min(cands, key=lambda c: cost(c[1]))
        raw.append(ftype)
        raw += fdata
        prev = row

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)

# ---------------------------------------------------------------- noise ---
class ValueNoise:
    def __init__(self, seed, cell=56):
        self.rng = random.Random(seed)
        self.cell = cell
        self.grid = {}

    def _val(self, ix, iy):
        key = (ix, iy)
        v = self.grid.get(key)
        if v is None:
            v = self.rng.random()
            self.grid[key] = v
        return v

    def noise(self, x, y):
        gx, gy = x / self.cell, y / self.cell
        ix, iy = math.floor(gx), math.floor(gy)
        fx, fy = gx - ix, gy - iy
        sx = fx * fx * (3 - 2 * fx)
        sy = fy * fy * (3 - 2 * fy)
        a = self._val(ix, iy)
        b = self._val(ix + 1, iy)
        c = self._val(ix, iy + 1)
        d = self._val(ix + 1, iy + 1)
        return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy

    def fbm(self, x, y, octaves=3):
        total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
        for _ in range(octaves):
            total += amp * self.noise(x * freq, y * freq)
            norm += amp
            amp *= 0.5
            freq *= 2.1
        return total / norm

# -------------------------------------------------------------- palette ---
def lerp(c1, c2, t):
    return (int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t))

_STOPS = [
    (0.00, (64, 32, 140)), (0.25, (150, 40, 160)), (0.45, (235, 90, 120)),
    (0.62, (250, 180, 70)), (0.80, (60, 220, 190)), (1.00, (64, 32, 140)),
]

def palette(t):
    t %= 1.0
    for i in range(len(_STOPS) - 1):
        t0, c0 = _STOPS[i]
        t1, c1 = _STOPS[i + 1]
        if t0 <= t <= t1:
            return lerp(c0, c1, (t - t0) / (t1 - t0))
    return _STOPS[-1][1]

# --------------------------------------------------------- buffer utils ---
def box_blur(buf, w, h, radius):
    """Separable box blur on an RGB bytearray, in place."""
    tmp = bytearray(len(buf))
    # horizontal
    for y in range(h):
        row = y * w * 3
        for c in range(3):
            acc = 0
            for k in range(-radius, radius + 1):
                xx = min(w - 1, max(0, k))
                acc += buf[row + xx * 3 + c]
            for x in range(w):
                tmp[row + x * 3 + c] = acc // (2 * radius + 1)
                xa = min(w - 1, max(0, x - radius))
                xb = min(w - 1, max(0, x + radius + 1))
                acc += buf[row + xb * 3 + c] - buf[row + xa * 3 + c]
    # vertical
    for x in range(w):
        for c in range(3):
            acc = 0
            for k in range(-radius, radius + 1):
                yy = min(h - 1, max(0, k))
                acc += tmp[(yy * w + x) * 3 + c]
            for y in range(h):
                buf[(y * w + x) * 3 + c] = acc // (2 * radius + 1)
                ya = min(h - 1, max(0, y - radius))
                yb = min(h - 1, max(0, y + radius + 1))
                acc += tmp[(yb * w + x) * 3 + c] - tmp[(ya * w + x) * 3 + c]

def upscale_add(dst, src, sw, sh, gain):
    """Bilinear-upscale src (sw x sh) and additively blend into dst (N x N)."""
    scale = sw / N
    for y in range(N):
        sy = (y + 0.5) * scale - 0.5
        y0 = int(math.floor(sy))
        fy = sy - y0
        y0c = min(sh - 1, max(0, y0))
        y1c = min(sh - 1, max(0, y0 + 1))
        for x in range(N):
            sx = (x + 0.5) * scale - 0.5
            x0 = int(math.floor(sx))
            fx = sx - x0
            x0c = min(sw - 1, max(0, x0))
            x1c = min(sw - 1, max(0, x0 + 1))
            i00 = (y0c * sw + x0c) * 3
            i01 = (y0c * sw + x1c) * 3
            i10 = (y1c * sw + x0c) * 3
            i11 = (y1c * sw + x1c) * 3
            di = (y * N + x) * 3
            for c in range(3):
                v = (src[i00 + c] * (1 - fx) * (1 - fy) +
                     src[i01 + c] * fx * (1 - fy) +
                     src[i10 + c] * (1 - fx) * fy +
                     src[i11 + c] * fx * fy) * gain
                nv = dst[di + c] + int(v)
                dst[di + c] = 255 if nv > 255 else nv

# ----------------------------------------------------------------- main ---
def main():
    noise = ValueNoise(SEED)
    cx, cy = N * 0.62, N * 0.40
    buf = bytearray(N * N * 3)

    # ---- background: vertical gradient + radial glow toward the vortex
    top, mid, glowc = (10, 8, 30), (24, 14, 58), (58, 30, 92)
    maxd = N * 0.95
    for y in range(N):
        base = lerp(mid, top, min(1.0, y / N * 1.2))
        dy = y - cy
        row = y * N * 3
        for x in range(N):
            d = math.hypot(x - cx, dy) / maxd
            g = max(0.0, 1.0 - d)
            g = g * g * g
            i = row + x * 3
            buf[i] = int(base[0] + (glowc[0] - base[0]) * g)
            buf[i + 1] = int(base[1] + (glowc[1] - base[1]) * g)
            buf[i + 2] = int(base[2] + (glowc[2] - base[2]) * g)

    # ---- flow field ------------------------------------------------------
    rng = random.Random(SEED + 7)
    glow = bytearray(GLOW * GLOW * 3)
    particles = []
    n_particles = 2400
    for p in range(n_particles):
        if p % 4 == 0:      # bias some starts into the sparse lower-right
            x = rng.uniform(N * 0.45, N)
            y = rng.uniform(N * 0.5, N)
        else:
            x = rng.uniform(0, N)
            y = rng.uniform(0, N)
        d0 = math.hypot(x - cx, y - cy) / (N * 0.75)
        hue = (d0 * 0.9 + noise.fbm(x * 0.6, y * 0.6) * 0.25) % 1.0
        col = palette(hue)
        fade = rng.uniform(0.5, 1.0)
        steps = rng.randint(60, 190)
        pts = [(x, y)]
        for _s in range(steps):
            n = noise.fbm(x, y)
            ang = n * math.pi * 4.2
            dx, dy = x - cx, y - cy
            dist = math.hypot(dx, dy) + 1e-6
            swirl = 1.35 * math.exp(-dist / (N * 0.55))
            tang = math.atan2(dy, dx) + math.pi / 2
            wgt = min(swirl, 0.85)
            ang = ang * (1 - wgt) + tang * wgt
            x += math.cos(ang) * 3.5
            y += math.sin(ang) * 3.5
            if not (-20 <= x <= N + 20 and -20 <= y <= N + 20):
                break
            pts.append((x, y))
        if len(pts) >= 8:
            particles.append((pts, col, fade))

    # nebula underlay into the glow buffer (huge soft palette clouds)
    neb_rng = random.Random(SEED + 31)
    gs = GLOW / N
    for _ in range(26):
        nx = neb_rng.uniform(-0.1, 1.1) * GLOW
        ny = neb_rng.uniform(-0.1, 1.1) * GLOW
        nr = neb_rng.uniform(0.16, 0.42) * GLOW
        ncol = palette(neb_rng.random())
        x0, x1 = max(0, int(nx - nr)), min(GLOW - 1, int(nx + nr))
        y0, y1 = max(0, int(ny - nr * 0.7)), min(GLOW - 1, int(ny + nr * 0.7))
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                q = ((xx - nx) / nr) ** 2 + ((yy - ny) / (nr * 0.7)) ** 2
                if q < 1.0:
                    a = (1.0 - q) * 0.10
                    i = (yy * GLOW + xx) * 3
                    glow[i] = min(255, glow[i] + int(ncol[0] * a))
                    glow[i + 1] = min(255, glow[i + 1] + int(ncol[1] * a))
                    glow[i + 2] = min(255, glow[i + 2] + int(ncol[2] * a))

    # stroke halos into the glow buffer (gentle: this layer saturates fast)
    for pts, col, fade in particles:
        a = 0.055 * fade
        sr, sg, sb = int(col[0] * a), int(col[1] * a), int(col[2] * a)
        for (x, y) in pts[::2]:
            gx, gy = int(x * gs), int(y * gs)
            if 1 <= gx < GLOW - 1 and 1 <= gy < GLOW - 1:
                for oy in (-1, 0, 1):
                    base = ((gy + oy) * GLOW + gx) * 3
                    for ox in (-1, 0, 1):
                        i = base + ox * 3
                        glow[i] = min(255, glow[i] + sr)
                        glow[i + 1] = min(255, glow[i + 1] + sg)
                        glow[i + 2] = min(255, glow[i + 2] + sb)
    box_blur(glow, GLOW, GLOW, 8)
    box_blur(glow, GLOW, GLOW, 8)
    upscale_add(buf, glow, GLOW, GLOW, 0.9)

    # ---- sharp core strokes, alpha-over blended (bounded, never blows out)
    for pts, col, fade in particles:
        a_num = int(95 * fade)          # of 255
        ia = 255 - a_num
        cr, cg, cb = col
        px, py = pts[0]
        for (x, y) in pts[1:]:
            seg = max(abs(x - px), abs(y - py))
            nseg = max(1, int(seg / 1.2))
            for k in range(nseg + 1):
                ix = int(px + (x - px) * k / nseg)
                iy = int(py + (y - py) * k / nseg)
                if 1 <= ix < N - 1 and 1 <= iy < N - 1:
                    i = (iy * N + ix) * 3
                    for di in (i, i + 3, i + N * 3):
                        buf[di] = (cr * a_num + buf[di] * ia) // 255
                        buf[di + 1] = (cg * a_num + buf[di + 1] * ia) // 255
                        buf[di + 2] = (cb * a_num + buf[di + 2] * ia) // 255
            px, py = x, y

    # ---- golden bloom at the core
    R = 125
    x0, x1 = max(0, int(cx - R)), min(N - 1, int(cx + R))
    y0, y1 = max(0, int(cy - R)), min(N - 1, int(cy + R))
    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            d = math.hypot(xx - cx, yy - cy) / R
            if d < 1.0:
                a = (1.0 - d) ** 2.2 * 0.55
                i = (yy * N + xx) * 3
                buf[i] = min(255, buf[i] + int(255 * a))
                buf[i + 1] = min(255, buf[i + 1] + int(196 * a))
                buf[i + 2] = min(255, buf[i + 2] + int(92 * a))
    # hot heart + cross sparkle
    for yy in range(int(cy - 8), int(cy + 9)):
        for xx in range(int(cx - 8), int(cx + 9)):
            if math.hypot(xx - cx, yy - cy) <= 8:
                i = (yy * N + xx) * 3
                buf[i], buf[i + 1], buf[i + 2] = 255, 246, 220
    for k in range(1, 61):
        a = int(130 * (1 - k / 61))
        for (xx, yy) in [(int(cx + k), int(cy)), (int(cx - k), int(cy)),
                         (int(cx), int(cy + k)), (int(cx), int(cy - k))]:
            i = (yy * N + xx) * 3
            buf[i] = min(255, buf[i] + int(255 * a / 255))
            buf[i + 1] = min(255, buf[i + 1] + int(226 * a / 255))
            buf[i + 2] = min(255, buf[i + 2] + int(160 * a / 255))

    # ---- stars
    for _ in range(240):
        x, y = rng.uniform(0, N), rng.uniform(0, N)
        ix, iy = int(x), int(y)
        if 0 <= ix < N and 0 <= iy < N:
            a = rng.uniform(0.25, 0.75)
            i = (iy * N + ix) * 3
            buf[i] = min(255, buf[i] + int(235 * a))
            buf[i + 1] = min(255, buf[i + 1] + int(240 * a))
            buf[i + 2] = min(255, buf[i + 2] + int(255 * a))

    # ---- vignette + film grain in one final pass
    grain = random.Random(SEED + 99).randbytes(N * N)
    ccx, ccy = N / 2, N / 2
    for y in range(N):
        dy = (y - ccy) / (N * 0.85)
        row = y * N * 3
        for x in range(N):
            dx = (x - ccx) / (N * 0.85)
            d = math.sqrt(dx * dx + dy * dy)
            t = min(1.0, max(0.0, (d - 0.55) / 0.7))
            f = 1.0 - 0.55 * (t * t * (3 - 2 * t))
            g = (grain[y * N + x] % 15) - 7
            i = row + x * 3
            buf[i] = max(0, min(255, int(buf[i] * f) + g))
            buf[i + 1] = max(0, min(255, int(buf[i + 1] * f) + g))
            buf[i + 2] = max(0, min(255, int(buf[i + 2] * f) + g))

    outdir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(outdir, exist_ok=True)
    write_png_palette(os.path.join(outdir, "entry.png"), N, N, buf)
    print("saved entry.png", N, "x", N)

if __name__ == "__main__":
    main()
