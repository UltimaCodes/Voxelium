<#
.SYNOPSIS
  Builds a distributable Voxelium zip: the Release exe, its MinGW runtime DLLs, and the
  assets/sounds it reads at startup, nothing else. No build system installs on its own;
  this only stages and zips what cmake --build already produced.

.PARAMETER Config
  Which build tree under build/mingw to package. Defaults to Release.

.PARAMETER MingwBin
  Directory holding libgcc_s_seh-1.dll / libstdc++-6.dll / libwinpthread-1.dll. Defaults to
  the WinLibs GCC install this project's own dev loop uses (see docs/NEXT.md's PATH notes);
  override if your toolchain lives somewhere else.

.EXAMPLE
  cmake --build build/mingw --config Release
  .\scripts\package.ps1
#>
param(
    [string]$Config = "Release",
    [string]$MingwBin = "C:\Users\ryaan\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root "build\mingw\$Config\voxelium.exe"
if (-not (Test-Path $exe)) {
    throw "$exe does not exist, build it first: cmake --build build/mingw --config $Config"
}

$stage = Join-Path $root "dist\voxelium"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null

Copy-Item $exe $stage
foreach ($dll in @("libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll")) {
    $src = Join-Path $MingwBin $dll
    if (-not (Test-Path $src)) {
        throw "$src not found, pass -MingwBin pointing at your GCC's bin directory"
    }
    Copy-Item $src $stage
}
Copy-Item (Join-Path $root "assets") (Join-Path $stage "assets") -Recurse
# Only the CC0 clips (credited in README.md), which is exactly what .gitignore already lets
# through, so mirroring that filter here keeps this script from quietly shipping something
# that was never meant to be redistributed.
New-Item -ItemType Directory -Path (Join-Path $stage "sounds") -Force | Out-Null
Copy-Item (Join-Path $root "sounds\*.flac") (Join-Path $stage "sounds") -ErrorAction SilentlyContinue
Copy-Item (Join-Path $root "sounds\*.wav") (Join-Path $stage "sounds") -ErrorAction SilentlyContinue
Copy-Item (Join-Path $root "README.md") $stage
Copy-Item (Join-Path $root "LICENSE") $stage

$zip = Join-Path $root "dist\voxelium-win64.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "$stage\*" -DestinationPath $zip

$sizeMb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Output "packaged: $zip ($sizeMb MB)"
