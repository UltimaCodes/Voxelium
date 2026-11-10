#pragma once

#include "world/Chunk.hpp"

namespace vx::blocks {

// What the block picker shows: a name and a tab. Ids that only exist as data (the shallower water
// steps) are left out of it.
enum class Tab { Building, Natural, Plants, Light, Liquid, Count };

inline constexpr const char* tabName(Tab t) {
    switch (t) {
        case Tab::Building: return "Building";
        case Tab::Natural: return "Natural";
        case Tab::Plants: return "Plants";
        case Tab::Light: return "Light";
        case Tab::Liquid: return "Liquid";
        default: return "";
    }
}

inline constexpr bool inPicker(BlockId id) {
    return id != Air && id < kBlockCount && !(id >= WaterL1 && id <= WaterL7);
}

inline constexpr Tab tabOf(BlockId id) {
    if (id == Water)
        return Tab::Liquid;
    if (id == Glow || isLamp(id) || id == GlowCoral)
        return Tab::Light;
    if (isCross(id) || isLeafBlock(id) || id == Cactus || id == Coconut || (id >= BrainCoral && id <= BubbleCoral))
        return Tab::Plants;
    if (isPlanks(id) || id == Bricks || id == Cobblestone || id == MossyCobble || id == StoneBricks || id == Glass ||
        isLogBlock(id) || isTerracotta(id) || id == Sandstone)
        return Tab::Building;
    return Tab::Natural;
}

inline constexpr const char* blockName(BlockId id) {
    if (isLamp(id))
        return "Lamp";
    switch (id) {
        case Stone: return "Stone";
        case Dirt: return "Dirt";
        case Grass: return "Grass Block";
        case Glow: return "Glow Block";
        case Water: return "Water";
        case Wood: return "Oak Log";
        case Leaves: return "Oak Leaves";
        case Sand: return "Sand";
        case Snow: return "Snow";
        case Planks: return "Oak Planks";
        case Bricks: return "Bricks";
        case CoalOre: return "Coal Ore";
        case IronOre: return "Iron Ore";
        case Gravel: return "Gravel";
        case GrassTuft: return "Grass";
        case Flower: return "Dandelion";
        case BeachSand: return "Beach Sand";
        case Sandstone: return "Sandstone";
        case Cactus: return "Cactus";
        case DeadBush: return "Dead Bush";
        case PalmWood: return "Palm Log";
        case PalmLeaves: return "Palm Leaves";
        case Coconut: return "Coconut";
        case Seagrass: return "Seagrass";
        case Kelp: return "Kelp";
        case BrainCoral: return "Brain Coral";
        case TubeCoral: return "Tube Coral";
        case FireCoral: return "Fire Coral";
        case HornCoral: return "Horn Coral";
        case BubbleCoral: return "Bubble Coral";
        case CoralFanPink: return "Pink Coral Fan";
        case CoralFanBlue: return "Blue Coral Fan";
        case CoralFanOrange: return "Orange Coral Fan";
        case Shell: return "Shell";
        case GlowCoral: return "Glow Coral";
        case SpruceWood: return "Spruce Log";
        case BirchWood: return "Birch Log";
        case SpruceLeaves: return "Spruce Leaves";
        case BirchLeaves: return "Birch Leaves";
        case TallGrass: return "Tall Grass";
        case Fern: return "Fern";
        case DryGrass: return "Dry Grass";
        case DuneGrass: return "Dune Grass";
        case JungleGrass: return "Jungle Grass";
        case FlowerRed: return "Rose";
        case FlowerBlue: return "Geranium";
        case FlowerPurple: return "Viola";
        case FlowerWhite: return "White Dandelion";
        case FlowerOrange: return "Tulip";
        case MushroomBrown: return "Brown Mushroom";
        case MushroomRed: return "Red Mushroom";
        case Reed: return "Reeds";
        case FireflyBush: return "Firefly Bush";
        case CactusFlower: return "Cactus Flower";
        case LeafLitter: return "Leaf Litter";
        case Pebbles: return "Pebbles";
        case LilyPad: return "Lily Pad";
        case Moss: return "Moss";
        case ForestFloor: return "Forest Floor";
        case JungleFloor: return "Jungle Floor";
        case SavannaGrass: return "Savanna Grass";
        case SnowyGrass: return "Snowy Grass";
        case DryDirt: return "Dry Dirt";
        case Permafrost: return "Permafrost";
        case Mud: return "Mud";
        case RedSand: return "Red Sand";
        case Clay: return "Clay";
        case Ice: return "Ice";
        case Terracotta: return "Terracotta";
        case TerracottaRed: return "Red Terracotta";
        case TerracottaYellow: return "Yellow Terracotta";
        case TerracottaBrown: return "Brown Terracotta";
        case TerracottaWhite: return "White Terracotta";
        case Granite: return "Granite";
        case Andesite: return "Andesite";
        case Slate: return "Slate";
        case AcaciaWood: return "Acacia Log";
        case AcaciaLeaves: return "Acacia Leaves";
        case JungleWood: return "Jungle Log";
        case JungleLeaves: return "Jungle Leaves";
        case BerryBush: return "Berry Bush";
        case WoodX:
        case WoodZ: return "Oak Log (lying)";
        case SpruceWoodX:
        case SpruceWoodZ: return "Spruce Log (lying)";
        case BirchWoodX:
        case BirchWoodZ: return "Birch Log (lying)";
        case Cobblestone: return "Cobblestone";
        case MossyCobble: return "Mossy Cobblestone";
        case StoneBricks: return "Stone Bricks";
        case Glass: return "Glass";
        case SprucePlanks: return "Spruce Planks";
        case BirchPlanks: return "Birch Planks";
        case AcaciaPlanks: return "Acacia Planks";
        case JunglePlanks: return "Jungle Planks";
        default: return "Block";
    }
}

} // namespace vx::blocks
