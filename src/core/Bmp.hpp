#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

namespace vx {

// Writes a 24-bit uncompressed BMP. `bgr` is bottom-up, tightly packed rows — exactly
// what glReadPixels(GL_BGR, GL_UNSIGNED_BYTE) produces with GL_PACK_ALIGNMENT = 1, so a
// framebuffer readback round-trips with zero conversion and zero image dependencies.
bool writeBmp24(const std::filesystem::path& path, int width, int height,
                std::span<const std::uint8_t> bgr);

} // namespace vx
