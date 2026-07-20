#pragma once

namespace vx {

// Installs the KHR_debug message callback (core in GL 4.3+). Wired from day one because
// GL errors without it are silent; the M2 debug HUD routes these into the on-screen log.
// Requires a current GL context.
void installGlDebugCallback();

} // namespace vx
