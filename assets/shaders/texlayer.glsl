// Block id + face dir -> texture-array layer. Shared by chunk.vert and shadow.vert: the
// shadow pass has to alpha-test cutout blocks, so it needs the same mapping, and a second
// copy would drift into shadows sampling the wrong tile.
// Layers whose alpha punches holes in a solid block (leaves, palm fronds, glass). Cross-plane
// foliage is cut out by its dir instead. Keep in step with texLayer below.
bool isCutoutLayer(uint layer) {
    return layer == 7u || layer == 55u || layer == 73u || layer == 74u || layer == 124u || layer == 127u ||
           layer == 128u || layer == 135u || (layer >= 150u && layer <= 155u);
}
// the emitters: glow, the 25 lamp swatches, glow coral, glow berries, the glow mushroom, crystal
bool isEmitterLayer(uint layer) { return layer == 4u || (layer >= 15u && layer <= 39u) || layer == 68u || (layer >= 147u && layer <= 149u); }
// grass and ferns, which take the climate's colour like the grass they grow from
bool isGrassLayer(uint layer) { return layer == 40u || layer == 140u || layer == 141u || (layer >= 75u && layer <= 79u); }
// leaf litter, pebbles, lily pads and petals: lying flat rather than standing crossed
bool isCarpet(uint block) { return (block >= 87u && block <= 89u) || block == 149u; }

// block id + face dir -> texture-array layer. mirrors BlockTextures layer order.
uint texLayer(uint block, uint dir) {
    bool cap = dir == 2u || dir == 3u;
    if (block == 1u) return 0u;                              // stone
    if (block == 2u) return 1u;                              // dirt
    if (block == 3u) return dir == 2u ? 2u : (dir == 3u ? 1u : 3u); // grass top/bottom/side
    if (block == 4u) return 4u;                              // glow
    if (block == 6u) return cap ? 6u : 5u;                   // wood top/side
    if (block == 7u) return 7u;                              // leaves
    if (block == 8u) return 8u;                              // sand
    if (block == 9u) return 9u;                              // snow
    if (block == 10u) return 10u;                            // planks
    if (block == 11u) return 11u;                            // bricks
    if (block == 12u) return 12u;                            // coal ore
    if (block == 13u) return 13u;                            // iron ore
    if (block == 21u) return 14u;                            // gravel
    if (block >= 22u && block <= 46u) return block - 7u;     // 25 lamp swatches -> 15..39
    if (block == 47u) return 40u;                            // grass tuft
    if (block == 48u) return 41u;                            // flower
    if (block == 49u) return 47u;                            // beach sand
    if (block == 50u) return cap ? 49u : 48u;                // sandstone top/side
    if (block == 51u) return cap ? 51u : 50u;                // cactus
    if (block == 52u) return 52u;                            // dead bush
    if (block == 53u) return cap ? 54u : 53u;                // palm trunk
    if (block == 54u) return 55u;                            // palm leaves
    if (block == 55u) return 56u;                            // coconut
    // seagrass, kelp, five corals, three fans, shell, glow coral: 56..67 -> 57..68
    if (block >= 56u && block <= 67u) return block + 1u;
    if (block == 68u) return cap ? 70u : 69u;                // spruce log
    if (block == 69u) return cap ? 72u : 71u;                // birch log
    if (block == 70u) return 73u;                            // spruce leaves
    if (block == 71u) return 74u;                            // birch leaves
    if (block == 72u) return 75u;                            // tall grass (2 variants)
    if (block == 73u) return 77u;                            // fern (3)
    if (block == 74u) return 80u;                            // dry grass (3)
    if (block == 75u) return 83u;                            // dune grass (3)
    if (block >= 76u && block <= 90u) return block + 10u;    // jungle grass .. moss
    if (block == 91u) return dir == 2u ? 101u : (dir == 3u ? 1u : 102u);   // forest floor
    if (block == 92u) return dir == 2u ? 103u : (dir == 3u ? 1u : 104u);   // jungle floor
    if (block == 93u) return dir == 2u ? 105u : (dir == 3u ? 108u : 106u); // savanna grass
    if (block == 94u) return dir == 2u ? 9u : (dir == 3u ? 1u : 107u);    // snowy grass
    if (block >= 95u && block <= 100u) return block + 13u;   // dry dirt .. ice
    if (block >= 101u && block <= 108u) return block + 13u;  // terracottas, granite, andesite, slate
    if (block == 109u) return cap ? 123u : 122u;             // acacia log
    if (block == 110u) return 124u;
    if (block == 111u) return cap ? 126u : 125u;             // jungle log
    if (block == 112u) return 127u;
    if (block == 113u) return 128u;                          // berry bush
    // logs lying down: the rings face along their axis, the bark runs along it
    if (block >= 114u && block <= 119u) {
        uint s = (block - 114u) / 2u;                        // oak, spruce, birch
        uint ring = s == 0u ? 6u : (s == 1u ? 70u : 72u);
        uint bark = s == 0u ? 5u : (s == 1u ? 69u : 71u);
        uint turned = 129u + s;
        if ((block & 1u) == 0u)                               // along x
            return dir <= 1u ? ring : turned;
        return dir >= 4u ? ring : (dir <= 1u ? turned : bark);
    }
    if (block >= 120u && block <= 127u) return block + 12u;  // cobble .. jungle planks
    if (block == 128u) return 142u;                          // bedrock
    if (block >= 129u && block <= 135u) return block + 14u;  // the cave blocks, dripstone .. crystal
    if (block >= 136u && block <= 141u) return block + 14u;  // cherry, blossom, the autumn leaves, mangrove roots
    if (block >= 142u && block <= 144u) {                    // cherry, redwood and mangrove logs
        uint side = 156u + (block - 142u) * 2u;
        return cap ? side + 1u : side;
    }
    if (block >= 145u && block <= 148u) return block + 17u;  // mushroom stem and caps, bamboo
    if (block >= 149u && block <= 151u) return block + 18u;  // petals, lavender, pink tulip
    return 0u;
}

// Plants with more than one look: which tile a given cell shows.
uint variantLayer(uint block, uint layer, uint h) {
    if (block == 47u) {
        uint v = h % 3u;
        return v == 0u ? layer : 139u + v;                   // 40, 140, 141
    }
    if (block == 72u) return layer + h % 2u;
    if (block >= 73u && block <= 75u) return layer + h % 3u;
    if (block == 148u) return layer + (h % 3u == 0u ? 1u : 0u); // bamboo: now and then a leafy joint
    return layer;
}
