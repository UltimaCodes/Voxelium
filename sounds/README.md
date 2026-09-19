# sounds/

Missing files are not an error. The engine stays quiet for whatever you haven't supplied, so
it runs fine with this directory empty — and everything here can be replaced or deleted
without touching a line of code.

**Only CC0 or equivalent may be committed here. No Minecraft audio, ever.** Everything
bundled is listed below with its source.

## Provenance

| What | Source | Author | Licence |
|---|---|---|---|
| `step*.flac`, `step_<id>-*.flac` | [Fantozzi's Footsteps (Grass/Sand & Stone)](https://opengameart.org/content/fantozzis-footsteps-grasssand-stone) | Fantozzi | CC0 |
| `break_stone*`, `break_wood*`, `break_glass*`, `break_gravel*`, `break*`, `place_stone*`, `place_wood*`, `place_glass*`, `place*`, `land_hard*`, `thunder2` | [75 CC0 breaking / falling / hit SFX](https://opengameart.org/content/75-cc0-breaking-falling-hit-sfx) | rubberduck | CC0 |
| `step_wood*`, `step_water*`, `break_stone3-4`, `place_stone2`, `jump*`, `amb_deep`, `amb_wind`, `thunder1` | [100 CC0 SFX #2](https://opengameart.org/content/100-cc0-sfx-2) | rubberduck | CC0 |
| `splash_in*`, `splash_out*`, `swim*`, `bubble*`, `break_water*`, `amb_underwater`, `amb_rain`, `amb_water` | [40 CC0 water / splash / slime SFX](https://opengameart.org/content/40-cc0-water-splash-slime-sfx) | rubberduck | CC0 |
| `amb_cave` | [Loopable Dungeon Ambience](https://opengameart.org/content/loopable-dungeon-ambience) | JaggedStone | CC0 |
| `amb_night` | [Crickets Ambient Noise](https://opengameart.org/content/crickets-ambient-noise-loopable) | Wolfgang_ (rec. Ted Kerr) | CC0 |
| `amb_day` | [Forest Ambience](https://opengameart.org/content/forest-ambience) | TinyWorlds | CC0 |
| `step_grass*`, `break_grass*`, `break_sand*`, `break_dirt*`, `break_snow*` | derived from the Fantozzi footsteps above (filtered and re-levelled) | — | CC0 |
| `amb_surf`, `amb_gulls`, `amb_birds`, `amb_desert`, `amb_cicadas`, `amb_leaves`, `amb_frogs`, `amb_drips`, `amb_reef` | synthesised from noise and oscillators by [`scripts/gen_ambience.py`](../scripts/gen_ambience.py) - nothing sampled, so nothing to license | this repo | CC0 |

The bundled clips were converted from their original `ogg`/`mp3` to 16-bit mono 48 kHz wav,
trimmed, peak-normalised and faded at both ends. The beds additionally have their tail
crossfaded back over their head so the loop point is seamless.

The nine synthesised beds (surf, gulls, birds, desert wind, cicadas, leaf rustle, frogs, cave
drips, reef) are deterministic - `python scripts/gen_ambience.py` rewrites them byte for byte -
so if you would rather have recordings, delete the file and drop your own in under the same
name; the ambience director asks for them by name and stays quiet for any that are missing.

The soft materials — grass, sand, dirt, snow — have no dedicated recording in any of these
packs. Their break sounds are the bundled footsteps, filtered per material. That is not a
shortcut: digging sand and walking on sand are the same material making the same noise, and
it is how Minecraft's own dig sounds relate to its step sounds.

## Format

`wav`, `mp3` or `flac`. (`ogg` would need stb_vorbis wired into miniaudio, so it is rejected
rather than failing silently at load. Convert it first — that is what was done here.)

Everything is decoded **once at startup** into memory and held as PCM for the life of the
process, so playing a sound never touches the disk. The decodes run across the job system, so
a large folder does not stall the loading screen.

Keep clips short. They are all resident at once, uncompressed, as 32-bit float mono at
48 kHz — about **192 KB per second of audio**. The bundled set is 144 files and **72 MB**
of PCM, and most of that is the seventeen looping beds. If that matters to you, the beds
are the thing to shorten.

## Naming

The filename stem is the sound's name. A **trailing digit marks a variant** — one is picked
at random per play, which is what stops footsteps sounding like a machine gun:

```
step_grass1.wav   step_grass2.wav   step_grass3.wav   ->  all register as "step_grass"
```

Names that END in a number are the awkward case, because block sounds are also looked up by
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

### Blocks

Three tiers, narrowest first — so you can be as specific or as lazy as you like:

| Event | Tried first | Then | Then | Last resort |
|---|---|---|---|---|
| Break a block | `break_<id>` | `break_<material>` | `break` | the material's own step, pitched **down** |
| Place a block | `place_<id>` | `place_<material>` | `place` | the material's own step, pitched **up** |
| Footstep | `step_<id>` | `step_<material>` | `step` | — |

**The material tier is the one that matters.** This engine has no per-voxel metadata, so it
stores data in the block id itself: water is seven ids and a lamp is twenty-five. Keying
sounds on the id alone meant twenty-five identical copies of one file, so in practice almost
nothing ever matched and everything fell through to the generic clip.

| material | block ids |
|---|---|
| `stone` | stone, bricks, coal ore, iron ore, and anything added later |
| `dirt` | dirt |
| `grass` | grass, leaves, grass tufts, flowers |
| `sand` | sand |
| `snow` | snow |
| `wood` | wood, planks |
| `glass` | glow, all 25 lamp colours |
| `gravel` | gravel |
| `water` | water source and all 7 flowing levels |

Leaves and foliage are in the `grass` group on purpose — a canopy and a lawn are the same
rustle, and it is one clip instead of three.

`<id>` is the block id from `src/world/Chunk.hpp`:

| id | block | | id | block |
|---|---|---|---|---|
| 1 | stone | | 9 | snow |
| 2 | dirt | | 10 | planks |
| 3 | grass | | 11 | bricks |
| 4 | glow | | 12 | coal ore |
| 5 | water (source) | | 13 | iron ore |
| 6 | wood | | 21 | gravel |
| 7 | leaves | | 47 | grass tuft |
| 8 | sand | | 48 | flower |

with 14–20 the flowing water levels and 22–46 the lamp colours.

### The player

| Name | When |
|---|---|
| `jump` | leaving the ground while still moving up. Walking off a ledge is silent |
| `land_hard` | landing from more than 3 blocks, **on top of** the material's own step |
| `splash_in` / `splash_out` | breaking the water surface, either way. Volume scales with how fast you hit it |
| `swim` | every 1.5 blocks of travel while in water |
| `bubble` | irregularly, while your head is under |

A landing under 0.55 blocks is just the next footstep — stepping off a block is a stride, not
a drop. Above that the material's own step plays louder and lower with the distance fallen,
and past 3 blocks `land_hard` is layered over it so landing on sand and on stone still sound
different.

### Ambience beds

Looping clips on the ambience bus, each crossfaded by where the player is. Supply as many or
as few as you like; a missing bed is simply never heard.

| Name | Comes up when |
|---|---|
| `amb_day` | outdoors, in daylight |
| `amb_night` | outdoors, after dusk |
| `amb_wind` | open sky, louder the higher you are |
| `amb_cave` | enclosed **and** below the surface — both, so an overhang is not a cave |
| `amb_deep` | far underground, layered under `amb_cave` |
| `amb_underwater` | head submerged. Replaces the surface beds rather than joining them |
| `amb_rain` | precipitation, heard through an opening but not through rock |
| `amb_water` | near a body of water |
| `thunder` | occasionally during heavy rain, 2D and rolled off by how much sky you can see |

Enclosure is read off the voxel **sky light** at your ear rather than by casting rays, which
costs one chunk lookup a frame — the light BFS has already worked it out.

## Mixing

Three buses, each with its own slider in the ESC menu under **audio**, beneath a master:

| Bus | What rides it |
|---|---|
| world | anything with a position — footsteps, breaking, placing, splashes |
| interface | menu and UI sounds, never spatialised |
| ambience | the looping beds and thunder |

World sounds attenuate with distance (full volume within 2 blocks, silent by 48) and are
muffled two ways: when solid blocks sit between the sound and your ear (one raycast per
sound event, with its own slider — set it to 0 to turn it off), and when your head is
underwater. Submerging runs the world and ambience buses through a low-pass that sweeps from
18 kHz down to 700 Hz over about 150 ms. The interface bus stays dry, because a menu click is
not in the water with you.

All of it is saved with the rest of your settings.
