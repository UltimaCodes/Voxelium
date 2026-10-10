#pragma once

#include "world/Chunk.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace vx {

// Chunk <-> bytes. The on-disk form is run-length over the block grid in index order,
// not the in-memory palette layout: voxel worlds are dominated by runs (air columns,
// stone layers), so RLE crushes them, an all-air chunk is ~4 bytes, a terrain chunk a
// few hundred, with zero external dependency. Counts and ids are LEB128 varints.
//
// Light is NOT stored: it re-seeds through the normal pipeline on load, same as a freshly
// generated chunk. Keeps the format to pure content and the loader identical to gen.
namespace ChunkSerializer {

// Appends the RLE encoding of `chunk` to `out`. Returns bytes written.
std::size_t serialize(const Chunk& chunk, std::vector<std::uint8_t>& out);

// Rebuilds `chunk` from `bytes`. Returns false on malformed input.
bool deserialize(std::span<const std::uint8_t> bytes, Chunk& chunk);

} // namespace ChunkSerializer

} // namespace vx
