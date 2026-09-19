"""Renders the procedural ambience beds into sounds/.

Everything here is synthesised from noise and oscillators, so there is nothing to license:
the output is original work and goes in as CC0. Needs numpy. Run from the repo root:

    python scripts/gen_ambience.py [name ...]

Deterministic (fixed seeds), so re-running rewrites byte-identical files. Every bed is a loop:
the tail is crossfaded back over the head, so the seam is inaudible.
"""
import sys
import wave
from pathlib import Path

import numpy as np

SR = 48000
OUT = Path(__file__).resolve().parent.parent / "sounds"


def rng(seed):
    return np.random.default_rng(seed)


def fft_filter(x, lo=None, hi=None, slope=None):
    # brickwall-with-soft-edges band filter in the frequency domain, optionally tilted
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    g = np.ones_like(f)
    if lo:
        g *= 1.0 / (1.0 + (lo / np.maximum(f, 1e-3)) ** 6)
    if hi:
        g *= 1.0 / (1.0 + (np.maximum(f, 1e-3) / hi) ** 6)
    if slope is not None:
        g *= (np.maximum(f, 20.0) / 1000.0) ** slope
    return np.fft.irfft(spec * g, len(x))


def pink(r, n):
    return fft_filter(r.standard_normal(n), slope=-0.5)


def slow(r, n, rate, seed_len=None):
    # smooth random 0..1 curve that wanders at roughly `rate` Hz
    pts = max(4, int(n / SR * rate) + 3)
    v = r.random(pts)
    xs = np.linspace(0, pts - 1, n)
    i = xs.astype(int)
    t = xs - i
    t = t * t * (3 - 2 * t)
    return v[i] * (1 - t) + v[np.minimum(i + 1, pts - 1)] * t


def loopify(x, seconds, xf=1.5):
    n = int(seconds * SR)
    k = int(xf * SR)
    head = x[:k]
    tail = x[n : n + k]
    fade = np.linspace(0, 1, k)
    out = x[:n].copy()
    out[:k] = head * fade + tail * (1 - fade)
    return out


def normalise(x, rms=None, peak=0.6):
    x = x - x.mean()
    if rms:
        x = x * (rms / max(np.sqrt((x ** 2).mean()), 1e-9))
    m = np.abs(x).max()
    if m > peak:
        x = x * (peak / m)
    return x


def write(name, x):
    OUT.mkdir(exist_ok=True)
    pcm = np.clip(x * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(OUT / f"{name}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"{name}.wav  {len(x) / SR:.1f}s")


def place(buf, clip, at):
    # add a clip at sample offset `at`, wrapping around the loop end
    n = len(buf)
    idx = (np.arange(len(clip)) + at) % n
    np.add.at(buf, idx, clip)


def echo(x, taps):
    out = x.copy()
    for delay, gain in taps:
        d = int(delay * SR)
        out[d:] += x[: len(x) - d] * gain
    return out


# --- beds ---------------------------------------------------------------------------------


def surf():
    r = rng(11)
    secs = 40
    n = (secs + 2) * SR
    t = np.arange(n) / SR
    bed = fft_filter(pink(r, n), hi=2500) * (0.35 + 0.35 * slow(r, n, 0.15))
    wash_src = fft_filter(r.standard_normal(n), lo=350, hi=5200)
    rumble_src = fft_filter(r.standard_normal(n), hi=180)
    env = np.zeros(n)
    when = 3.0
    while when < secs + 6:
        rise = r.uniform(2.0, 3.2)
        fall = r.uniform(4.0, 6.5)
        amp = r.uniform(0.6, 1.0)
        for off in (0.0, float(secs)):  # second copy makes the loop seam continuous
            s = t - (when - off)
            up = np.clip(s / rise, 0, 1)
            e = np.where(s < 0, 0.0, np.where(s < rise, np.sin(up * np.pi / 2) ** 2,
                                              np.exp(-(s - rise) / (fall / 3.2))))
            env += amp * e
        when += r.uniform(7.0, 11.5)
    env = np.clip(env, 0, 1.4)
    # foam fizz rides the back of each wave
    fizz = fft_filter(r.standard_normal(n), lo=3000, hi=9000) * np.clip(np.gradient(env) * -60, 0, 1) * env
    x = bed * 0.6 + wash_src * env ** 1.4 * 0.9 + rumble_src * env * 1.2 + fizz * 0.35
    return normalise(loopify(x, secs), rms=0.075)


def gull_call(r, f0, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    glide = np.where(u < 0.32, 0.86 + 1.3 * u, 1.276 - 0.85 * np.maximum(u - 0.32, 0.0) ** 0.9)
    vib = 1 + 0.03 * np.sin(2 * np.pi * 28 * t)
    f = f0 * glide * vib
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = sum(np.sin(k * ph) / k ** 1.1 for k in range(1, 9))
    s += 0.35 * r.standard_normal(n) * 0.25  # breath
    s = fft_filter(s, lo=500, hi=6500)
    env = np.minimum(u / 0.06, 1.0) * np.exp(-2.6 * u)
    return s * env


def gulls():
    r = rng(23)
    secs = 32
    n = (secs + 2) * SR
    buf = np.zeros(n)
    for start in (3.0, 12.5, 21.0, 27.5):
        f0 = r.uniform(950, 1350)
        calls = r.integers(2, 5)
        at = start
        for _ in range(calls):
            dur = r.uniform(0.32, 0.55)
            c = gull_call(r, f0 * r.uniform(0.94, 1.06), dur) * r.uniform(0.6, 1.0)
            place(buf, c * 0.5, int(at * SR))
            at += dur * r.uniform(0.9, 1.25)
    buf = echo(buf, [(0.09, 0.28), (0.21, 0.14), (0.45, 0.07)])
    wind = fft_filter(pink(r, n), hi=900) * 0.05
    return normalise(loopify(buf + wind, secs), peak=0.55)


def chirp(r, kind):
    if kind == 0:  # tweet: rising sweep
        d = r.uniform(0.07, 0.14)
        n = int(d * SR)
        u = np.linspace(0, 1, n)
        f = r.uniform(2800, 3600) + u * r.uniform(900, 1700)
        env = np.sin(np.pi * u) ** 0.6
    elif kind == 1:  # trill
        d = r.uniform(0.35, 0.7)
        n = int(d * SR)
        u = np.linspace(0, 1, n)
        f = r.uniform(4200, 5600) + 300 * np.sin(2 * np.pi * 3 * u)
        env = np.sin(np.pi * u) ** 0.4 * (0.55 + 0.45 * np.sin(2 * np.pi * 26 * u * d) ** 2)
    elif kind == 2:  # whistle: two falling tones
        d = r.uniform(0.32, 0.5)
        n = int(d * SR)
        u = np.linspace(0, 1, n)
        top = r.uniform(2300, 2900)
        f = np.where(u < 0.5, top - 260 * u, top * 0.78 - 300 * (u - 0.5))
        env = np.sin(np.pi * u) ** 0.7
    else:  # warble: stepwise wandering pitch
        d = r.uniform(0.5, 0.95)
        n = int(d * SR)
        steps = r.uniform(2400, 5200, int(d / 0.045) + 2)
        f = np.repeat(steps, int(0.045 * SR) + 1)[:n]
        f = fft_filter(f, hi=90) + 0.0
        u = np.linspace(0, 1, n)
        env = np.sin(np.pi * u) ** 0.5
    ph = 2 * np.pi * np.cumsum(f) / SR
    return (np.sin(ph) + 0.18 * np.sin(2 * ph)) * env


def birds():
    r = rng(37)
    secs = 34
    n = (secs + 2) * SR
    buf = np.zeros(n)
    at = 1.0
    while at < secs:
        kind = int(r.integers(0, 4))
        reps = int(r.integers(1, 5)) if kind != 3 else 1
        gain = r.uniform(0.25, 0.7)
        pos = at
        for _ in range(reps):
            c = chirp(r, kind) * gain
            place(buf, c, int(pos * SR))
            pos += len(c) / SR + r.uniform(0.06, 0.18)
        at = pos + r.uniform(0.8, 3.6)
    buf = echo(buf, [(0.07, 0.3), (0.16, 0.18), (0.34, 0.09)])
    air = fft_filter(pink(r, n), lo=400, hi=3500) * 0.02
    return normalise(loopify(buf * 0.5 + air, secs), peak=0.5)


def desert():
    r = rng(41)
    secs = 30
    n = (secs + 2) * SR
    gust = slow(r, n, 0.22) ** 1.6
    body = fft_filter(pink(r, n), lo=120, hi=900) * (0.25 + 0.9 * gust)
    whistle = fft_filter(r.standard_normal(n), lo=850, hi=1500) * gust ** 3 * 0.55
    sand = fft_filter(r.standard_normal(n), lo=4200, hi=11000) * (0.02 + 0.10 * gust) * (0.5 + slow(r, n, 6))
    low = fft_filter(r.standard_normal(n), hi=110) * (0.2 + 0.5 * gust)
    x = body + whistle + sand + low
    return normalise(loopify(x, secs), rms=0.06)


def cicadas():
    r = rng(53)
    secs = 24
    n = (secs + 2) * SR
    t = np.arange(n) / SR
    out = np.zeros(n)
    for i in range(5):
        f = r.uniform(3700, 5600)
        rate = r.uniform(38, 62)
        on = slow(r, n, r.uniform(0.18, 0.32))
        gate = np.clip((on - 0.42) * 5, 0, 1)
        am = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * rate * t + r.uniform(0, 6))) * 0.8
        car = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * f * 1.5 * t)
        out += car * am * gate * r.uniform(0.15, 0.3)
    out = fft_filter(out, lo=2500, hi=8000)
    out += fft_filter(r.standard_normal(n), lo=2500, hi=7000) * 0.012
    return normalise(loopify(out, secs), rms=0.045)


def leaves():
    r = rng(67)
    secs = 20
    n = (secs + 2) * SR
    gust = 0.25 + 0.75 * slow(r, n, 0.25) ** 1.4
    hiss = fft_filter(r.standard_normal(n), lo=1400, hi=7500)
    # crackle: random short decaying bursts, denser in a gust
    imp = (r.random(n) < (0.0009 + 0.0016 * gust)).astype(float)
    k = np.exp(-np.arange(int(0.012 * SR)) / (0.0025 * SR))
    crack = np.convolve(imp, k)[:n] * fft_filter(r.standard_normal(n), lo=2000, hi=9000) * 4
    x = hiss * gust * 0.5 + crack * gust
    return normalise(loopify(x, secs), rms=0.05)


def frogs():
    r = rng(79)
    secs = 30
    n = (secs + 2) * SR
    buf = np.zeros(n)
    for frog in range(3):
        base = r.uniform(300, 540)
        at = r.uniform(0.5, 4.0)
        while at < secs:
            for _ in range(int(r.integers(2, 6))):
                d = r.uniform(0.11, 0.2)
                m = int(d * SR)
                t = np.arange(m) / SR
                f = base * (1 + 0.1 * np.sin(2 * np.pi * 9 * t)) * (1.0 - 0.15 * t / d)
                ph = 2 * np.pi * np.cumsum(f) / SR
                c = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph))
                c *= 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 42 * t)) * 0.7
                c *= np.sin(np.pi * t / d) ** 0.8
                place(buf, c * 0.35, int(at * SR))
                at += d + r.uniform(0.04, 0.12)
            at += r.uniform(2.0, 6.5)
    buf = fft_filter(buf, lo=200, hi=3500)
    lap = fft_filter(pink(r, n), hi=1400) * 0.02
    return normalise(loopify(echo(buf, [(0.13, 0.2), (0.29, 0.1)]) + lap, secs), peak=0.5)


def drips():
    r = rng(83)
    secs = 30
    n = (secs + 2) * SR
    buf = np.zeros(n)
    at = 0.8
    while at < secs:
        d = 0.25
        m = int(d * SR)
        t = np.arange(m) / SR
        f = 1500 * np.exp(-t * 9) + 600
        ph = 2 * np.pi * np.cumsum(f) / SR
        c = np.sin(ph) * np.exp(-t * 26) * r.uniform(0.25, 0.7)
        c += np.sin(2 * np.pi * (700 + 200 * r.random()) * t) * np.exp(-t * 50) * 0.2
        place(buf, c, int(at * SR))
        at += r.uniform(1.4, 5.0)
    buf = echo(buf, [(0.11, 0.45), (0.23, 0.3), (0.41, 0.2), (0.7, 0.12), (1.1, 0.06)])
    return normalise(loopify(buf, secs), peak=0.55)


def reef():
    r = rng(97)
    secs = 30
    n = (secs + 2) * SR
    hum = fft_filter(pink(r, n), hi=260) * (0.4 + 0.6 * slow(r, n, 0.2))
    imp = (r.random(n) < 0.0012).astype(float)
    k = np.exp(-np.arange(int(0.004 * SR)) / (0.0009 * SR))
    snap = np.convolve(imp, k)[:n] * fft_filter(r.standard_normal(n), lo=3500, hi=12000) * 5
    buf = np.zeros(n)
    at = 0.3
    while at < secs:
        d = r.uniform(0.04, 0.09)
        m = int(d * SR)
        t = np.arange(m) / SR
        f = r.uniform(280, 520) * np.exp(t / d * 1.1)
        c = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / d) * r.uniform(0.1, 0.35)
        place(buf, c, int(at * SR))
        at += r.exponential(0.16)
    x = hum * 0.9 + snap * 0.25 + buf * 0.35
    return normalise(loopify(x, secs), rms=0.05)


BEDS = {
    "amb_surf": surf,
    "amb_gulls": gulls,
    "amb_birds": birds,
    "amb_desert": desert,
    "amb_cicadas": cicadas,
    "amb_leaves": leaves,
    "amb_frogs": frogs,
    "amb_drips": drips,
    "amb_reef": reef,
}

if __name__ == "__main__":
    want = sys.argv[1:] or list(BEDS)
    for name in want:
        write(name, BEDS[name]())
