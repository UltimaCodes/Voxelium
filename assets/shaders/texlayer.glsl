// Block id + face dir -> texture-array layer. Shared by chunk.vert and shadow.vert: the
// shadow pass has to alpha-test cutout blocks, so it needs the same mapping, and a second
// copy would drift into shadows sampling the wrong tile.
// block id + face dir -> texture-array layer. mirrors BlockTextures layer order.
uint texLayer(uint block, uint dir) {
    if (block == 1u) return 0u;                              // stone
    if (block == 2u) return 1u;                              // dirt
    if (block == 3u) return dir == 2u ? 2u : (dir == 3u ? 1u : 3u); // grass top/bottom/side
    if (block == 4u) return 4u;                              // glow
    if (block == 6u) return (dir == 2u || dir == 3u) ? 6u : 5u;     // wood top/side
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
    return 0u;
}
