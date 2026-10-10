#!/usr/bin/env python3
"""Builds the four recorded ambience beds from CC0 recordings (the sources are credited in README.md).

    python scripts/build_ambience_recordings.py <folder of decoded 16-bit mono 48 kHz wavs> [out dir]

The sources are short one-shots and loops, not beds, so each bed is assembled: waves are chained
with crossfades in a shuffled order, gull calls are scattered over a quiet floor, the insect
loops are tiled and layered at different lengths so they never line up, and every bed gets its
tail crossfaded over its head so it loops without a click. Each is scaled to the RMS of the
synthesised bed it replaces (scripts/gen_ambience.py), so the mix the ambience director was tuned
against keeps its balance. Deterministic: same inputs, same bytes. Standard library only.
"""
import array
import math
import random
import sys
import wave
from pathlib import Path

RATE = 48000


def read(path):
    with wave.open(str(path), 'rb') as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == RATE, path
        a = array.array('h')
        a.frombytes(w.readframes(w.getnframes()))
    return [v / 32768.0 for v in a]


def write(path, samples):
    a = array.array('h', (max(-32768, min(32767, int(round(v * 32767.0)))) for v in samples))
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(a.tobytes())


def rms(x):
    return math.sqrt(sum(v * v for v in x) / max(1, len(x)))


def peak(x):
    return max((abs(v) for v in x), default=0.0)


def fade_pair(n):
    """Equal-power crossfade curves: (rising, falling)."""
    return ([math.sin(0.5 * math.pi * i / n) for i in range(n)],
            [math.cos(0.5 * math.pi * i / n) for i in range(n)])


def chain(pieces, xf):
    """Concatenate with an equal-power crossfade of xf samples between neighbours."""
    up, down = fade_pair(xf)
    out = list(pieces[0])
    for p in pieces[1:]:
        base = len(out) - xf
        for i in range(xf):
            out[base + i] = out[base + i] * down[i] + p[i] * up[i]
        out.extend(p[xf:])
    return out


def make_loop(x, xf):
    """Crossfade the tail over the head so the end runs straight back into the start."""
    n = len(x)
    up, down = fade_pair(xf)
    out = x[:n - xf]
    for i in range(xf):
        out[i] = x[i] * up[i] + x[n - xf + i] * down[i]
    return out


def tile(x, length, xf):
    """Repeat x with crossfaded joins until it is at least `length` long, then cut to length."""
    out = list(x)
    while len(out) < length + xf:
        out = chain([out, x], xf)
    return out[:length]


def fade_edges(x, n):
    x = list(x)
    for i in range(min(n, len(x))):
        g = i / n
        x[i] *= g
        x[len(x) - 1 - i] *= g
    return x


def scale_to(x, target_rms, ceiling=0.93):
    r = rms(x)
    g = target_rms / r if r > 0 else 1.0
    p = peak(x) * g
    if p > ceiling:
        g *= ceiling / p
    return [v * g for v in x]


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('sounds')
    old = out  # the synthesised beds being replaced are read for their RMS if still there
    rnd = random.Random(20260920)

    def target(name, fallback):
        p = old / name
        return rms(read(p)) if p.exists() else fallback

    # ---- surf: the four waves of one beach, shuffled and chained ----------------------------
    waves = [read(src / f'wave_0{i}_cc0-18363__jasinski__alkaibeach.wav') for i in (1, 2, 3, 4)]
    order, last = [], -1
    for _ in range(12):
        k = rnd.choice([i for i in range(4) if i != last])
        order.append(k)
        last = k
    pieces = [[v * rnd.uniform(0.72, 1.0) for v in waves[k]] for k in order]
    surf = make_loop(chain(pieces, int(1.1 * RATE)), int(2.0 * RATE))
    write(out / 'amb_surf.wav', scale_to(surf, target('amb_surf.wav', 0.06)))

    # ---- gulls: calls scattered over near silence -------------------------------------------
    calls = [read(src / f'gull{i}.wav') for i in range(1, 8)] + [read(src / 'gullbig2.wav')]
    length = 56 * RATE
    gulls = [0.0] * length
    t = 2.5
    idx = 0
    while t < 52.0:
        c = calls[(idx * 3 + rnd.randrange(3)) % len(calls)]
        c = fade_edges([v * rnd.uniform(0.45, 1.0) for v in c], int(0.03 * RATE))
        start = int(t * RATE)
        for i, v in enumerate(c):
            if start + i < length:
                gulls[start + i] += v
        t += len(c) / RATE + rnd.uniform(3.5, 8.5)
        idx += 1
    write(out / 'amb_gulls.wav', scale_to(gulls, target('amb_gulls.wav', 0.02)))

    # ---- cave drips: one 20 s recording, looped ---------------------------------------------
    drips = make_loop(read(src / 'atmosbasement.mp3_.wav'), int(1.5 * RATE))
    write(out / 'amb_drips.wav', scale_to(drips, target('amb_drips.wav', 0.03)))

    # ---- cicadas: three insect loops at three lengths, layered ------------------------------
    n = 34 * RATE
    layers = [(read(src / 'swamp_cicada_2.wav'), 0.9, 0.0), (read(src / 'swamp_cicada_1.wav'), 0.7, 2.3),
              (read(src / 'swamp_bug_1.wav'), 0.35, 5.1)]
    mix = [0.0] * n
    for x, gain, offset in layers:
        t = tile(x, n + int(offset * RATE), int(0.4 * RATE))[int(offset * RATE):]
        for i in range(n):
            mix[i] += t[i] * gain
    cic = make_loop(mix, int(1.5 * RATE))
    write(out / 'amb_cicadas.wav', scale_to(cic, target('amb_cicadas.wav', 0.05)))

    for name in ('amb_surf', 'amb_gulls', 'amb_drips', 'amb_cicadas'):
        x = read(out / f'{name}.wav')
        print(f'{name}: {len(x) / RATE:.1f} s, rms {rms(x):.4f}, peak {peak(x):.3f}')


if __name__ == '__main__':
    main()
