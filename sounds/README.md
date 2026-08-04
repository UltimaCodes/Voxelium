# sounds/

Drop your own audio files here. **Nothing ships with the engine** and nothing is
downloaded at build time — same rule as the block textures. No Minecraft audio is bundled
or fetched, so please use clips you have the rights to.

Missing files are not an error. The engine just stays quiet for whatever you haven't
supplied, so it runs fine with this directory empty.

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
