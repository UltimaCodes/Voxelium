# sounds/

Missing files are not an error. The engine just stays quiet for whatever you haven't
supplied, so it runs fine with this directory empty.

## What ships here

Footsteps only, from [Fantozzi's Footsteps (Grass/Sand & Stone)](https://opengameart.org/content/fantozzis-footsteps-grasssand-stone)
by Fantozzi, released **CC0**. Twelve FLAC clips: six sand and six stone. Sand doubles for
grass, dirt and snow; stone doubles for gravel and the ores.

**Break and place sounds are still missing** — that pack has none, so those events are
silent until you drop something in. `break.wav` and `place.wav` alone are enough to cover
every block.

Only commit audio that may be redistributed — CC0 or equivalent. No Minecraft audio.

## Format

`wav`, `mp3` or `flac`. (`ogg` would need stb_vorbis wired into miniaudio, so it is
rejected rather than failing silently at load.)

Short clips. Everything is decoded fully into memory on first use and cached.

## Naming

The filename stem is the sound's name. A **trailing digit marks a variant** — one is
picked at random per play, which is what stops footsteps sounding like a machine gun:

```
step_grass1.wav   step_grass2.wav   step_grass3.wav   ->  all register as "step_grass"
```

Names that END in a number are the awkward case, because block sounds are looked up by
numeric id: `break_3` is grass, and stripping the trailing digit as a variant index would
turn it into `break_`, which nothing asks for. So the rule is:

```
step_grass3   ->  step_grass    ordinary variant
break_3       ->  break_3       digits after '_' are the block id, kept
step_1-2      ->  step_1        '-' marks a variant of a numbered name
```

Use the `-` form when you want several clips for one block id, which is how the bundled
footsteps are named (`step_1-1.flac` … `step_1-6.flac` for stone).

## What the engine looks for

Block sounds are tried by numeric block id first, then fall back to a generic name, so you
can be as specific or as lazy as you like:

| Event | Tried first | Falls back to |
|---|---|---|
| Break a block | `break_<id>` | `break` |
| Place a block | `place_<id>` | `place` |
| Footstep (walk mode) | `step_<id>` | `step` |

`<id>` is the block id from `src/world/Chunk.hpp`:

| id | block | | id | block |
|---|---|---|---|---|
| 1 | stone | | 7 | leaves |
| 2 | dirt | | 8 | sand |
| 3 | grass | | 9 | snow |
| 4 | glow | | 10 | planks |
| 5 | water | | 11 | bricks |
| 6 | wood | | | |

So `break.wav` alone gives every block the same break sound; adding `break_3.wav` gives
grass its own and leaves the rest on the generic one.

Master volume is in the ESC settings menu and is saved with the rest of your settings.
