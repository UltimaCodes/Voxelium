#include "core/Bmp.hpp"

#include "core/Log.hpp"

#include <fstream>
#include <vector>

namespace vx {

// Explicit little-endian byte writes: no packed-struct pragmas, no endianness assumption.
static void put16(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(std::uint8_t(v & 0xFFu));
    out.push_back(std::uint8_t((v >> 8) & 0xFFu));
}

static void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    put16(out, v & 0xFFFFu);
    put16(out, v >> 16);
}

bool writeBmp24(const std::filesystem::path& path, int width, int height,
                std::span<const std::uint8_t> bgr) {
    if (width <= 0 || height <= 0 ||
        bgr.size() != std::size_t(width) * std::size_t(height) * 3u) {
        VX_ERROR("writeBmp24: pixel data size does not match %dx%d", width, height);
        return false;
    }

    // BMP rows are padded to 4-byte multiples.
    const std::uint32_t tightRow = std::uint32_t(width) * 3u;
    const std::uint32_t paddedRow = (tightRow + 3u) & ~3u;
    const std::uint32_t pixelBytes = paddedRow * std::uint32_t(height);

    std::vector<std::uint8_t> header;
    header.reserve(54);
    put16(header, 0x4D42u);          // 'BM'
    put32(header, 54u + pixelBytes); // total file size
    put32(header, 0);                // reserved
    put32(header, 54);               // offset to pixel data
    put32(header, 40);               // BITMAPINFOHEADER size
    put32(header, std::uint32_t(width));
    put32(header, std::uint32_t(height)); // positive height = bottom-up rows
    put16(header, 1);                     // color planes
    put16(header, 24);                    // bits per pixel
    put32(header, 0);                     // BI_RGB, uncompressed
    put32(header, pixelBytes);
    put32(header, 2835); // 72 DPI in pixels/meter
    put32(header, 2835);
    put32(header, 0);
    put32(header, 0);

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        VX_ERROR("writeBmp24: cannot open %s", path.string().c_str());
        return false;
    }

    file.write(reinterpret_cast<const char*>(header.data()), std::streamsize(header.size()));
    const std::uint8_t pad[3] = {0, 0, 0};
    for (int y = 0; y < height; ++y) {
        file.write(reinterpret_cast<const char*>(bgr.data() + std::size_t(y) * tightRow),
                   std::streamsize(tightRow));
        if (paddedRow != tightRow)
            file.write(reinterpret_cast<const char*>(pad), std::streamsize(paddedRow - tightRow));
    }
    return file.good();
}

} // namespace vx
