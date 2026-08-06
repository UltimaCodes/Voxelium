# block textures

16x16 PNGs here override the procedural tiles the engine generates at startup. Any that
are missing fall back to the generated pixel-art, so deleting one is a valid way to compare
the two. Bigger images are nearest-downsampled to 16x16.

## what ships here

The bundled set is from [16x16 Block Texture Set](https://opengameart.org/content/16x16-block-texture-set)
by ARoachIFoundOnMyPillow, released **CC0** (public domain dedication). CC0 asks for
nothing in return, but recording where a file came from is worth doing anyway — nobody
should have to guess later whether this folder is safe to redistribute.

The pack's names differ from the ones the loader wants, so they were renamed on the way in.
The mapping, in case you want to swap an individual tile back:

| here | from the pack |
|---|---|
| `stone.png` | `stone_generic.png` |
| `sand.png` | `sand_ugly.png` |
| `wood_side.png` / `wood_top.png` | `oak_log_side.png` / `oak_log_top.png` |
| `leaves.png` | `oak_leaves.png` |
| `planks.png` | `oak_planks.png` |
| `bricks.png` | `cobblestone_bricks.png` |
| `coal_ore.png` | `stone_generic_ore_nuggets.png` |
| `iron_ore.png` | `stone_generic_ore_crystalline.png` |
| `glow.png` | `amethyst.png` |

`dirt`, `grass_top`, `grass_side`, `snow` and `gravel` kept their names.

## filenames the loader looks for

    stone.png  dirt.png  grass_top.png  grass_side.png  sand.png
    wood_side.png  wood_top.png  leaves.png  glow.png  snow.png
    planks.png  bricks.png  coal_ore.png  iron_ore.png  gravel.png

## what is not overridable yet

Three groups are generated only and have no filename, because they are built from tables
the engine also reads for something else:

- **25 lamp swatches** — generated from `blocks::kLampRGB`, the same table that drives the
  light each lamp emits. A file here would let the face and its light disagree.
- **2 cross-plane foliage tiles** (grass tuft, flower) — these need an alpha channel,
  because the engine alpha-cuts them.
- **5 break-progress crack stages** — each stage has to contain the previous stage's
  cracks, or the animation reads as a new pattern per step instead of accumulating damage.

Adding loader paths for any of these is easy enough; it just has not been needed.

Only commit textures that may be redistributed — CC0 or an equivalent licence.
