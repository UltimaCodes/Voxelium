#pragma once

namespace vx {

// Installs a top-level exception filter that writes a .dmp minidump to crashes/ before the
// process dies, so a crash report is a file to attach rather than "it closed, no idea why."
// Windows-only (SetUnhandledExceptionFilter + DbgHelp); a no-op call on any other platform.
void installCrashHandler();

} // namespace vx
