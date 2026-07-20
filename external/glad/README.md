# glad — vendored GL loader

Generated output of [glad2](https://github.com/Dav1dde/glad) v2.0.8 for **OpenGL 4.6 core,
no extensions**. Everything the engine needs (DSA, MultiDrawIndirect, buffer storage,
compute, anisotropic filtering, KHR_debug) is core in 4.6, so no extensions are requested;
that keeps `gl.h` at ~200 KB instead of ~5 MB.

The generated files are vendored so builds are deterministic and offline. Never hand-edit
`include/` or `src/` — regenerate instead:

    python -m pip install glad2==2.0.8
    python -m glad --api "gl:core=4.6" "--extensions=" --reproducible --out-path external/glad c

If an extension is ever needed (e.g. `GL_EXT_texture_compression_s3tc`), add it to
`--extensions=...` as a comma-separated list, regenerate, and update this file.
