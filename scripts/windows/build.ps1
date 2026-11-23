<#
.SYNOPSIS
Builds and tests the presets win-cpu-strict and win-cpu on Windows with the MinGW-w64 GCC, CMake
and Ninja of CLion (phase 07, part 4, decision W1).

.DESCRIPTION
Run from a PowerShell 5.1 window (no administrator rights, no modules), after
scripts/windows/stage.sh has copied the commit (DEST = C:\Users\Simone\sbmc-win):

  powershell -NoProfile -ExecutionPolicy Bypass -File DEST\src\scripts\windows\build.ps1

The sources are the copy of stage.sh, DEST\src (two folders above this script); the builds go to
DEST\build-<preset> (the binaryDir of the presets) and the logs to DEST\logs:
<preset>_configure.log, <preset>_build.log, <preset>_ctest.log and build_summary.txt, whose lines
are also written on the console:

  toolchain clion=<folder> gcc=<first line of g++ --version>
  cmake=<first line of cmake --version> ninja=<version>
  source git_hash=<git_hash of DEST\src\.sbmc_build_info>
  preset <name> configure=<exit> build=<exit> warnings=<lines> tests=<passed>/<total>
    failed=<n> skipped=<n> ctest=<exit>     (one line per preset)
  WIN_BUILD: PASS|FAIL

PASS: both presets configured, built and tested with exit status 0, no line with "warning" in
the build logs, at least one test and no test failed. The toolchain: -ClionDir, or the newest
folder CLion* of $env:ProgramFiles\JetBrains, then of $env:LOCALAPPDATA\Programs (Toolbox), with
bin\mingw\bin (g++), bin\cmake\win\x64\bin (cmake, ctest) and bin\ninja\win\x64 (ninja), put
first in the PATH of this process only.

Exit status: 0 for PASS, 1 for FAIL or a missing tool (the summary says why).
#>
param(
  [string]$ClionDir = '',
  [string]$Dest = ''
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$Src = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Dest) {
  $Dest = Split-Path -Parent $Src
}
$Logs = Join-Path $Dest 'logs'
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$Summary = New-Object System.Collections.Generic.List[string]

function Write-Summary([string]$Line) {
  Write-Host $Line
  $Summary.Add($Line)
}

function Save-Summary {
  [System.IO.File]::WriteAllLines((Join-Path $Logs 'build_summary.txt'), $Summary.ToArray())
}

function Stop-Build([string]$Message) {
  Write-Summary "ERROR: $Message"
  Write-Summary 'WIN_BUILD: FAIL'
  Save-Summary
  exit 1
}

# A native command with its stdout and stderr in a log file (ASCII); returns its exit status.
function Invoke-Logged([string]$Log, [string]$Exe, [string[]]$Arguments) {
  $previous = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    & $Exe @Arguments 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $Log -Encoding ascii
    $code = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previous
  }
  return $code
}

function Find-Clion {
  if ($ClionDir) {
    if (-not (Test-Path -LiteralPath $ClionDir)) {
      Stop-Build "-ClionDir $ClionDir does not exist"
    }
    return (Resolve-Path -LiteralPath $ClionDir).Path
  }
  $roots = @()
  if ($env:ProgramFiles) {
    $roots += (Join-Path $env:ProgramFiles 'JetBrains')
  }
  if ($env:LOCALAPPDATA) {
    $roots += (Join-Path $env:LOCALAPPDATA 'Programs')
  }
  foreach ($root in $roots) {
    if (-not (Test-Path -LiteralPath $root)) {
      continue
    }
    $found = @(Get-ChildItem -LiteralPath $root -Directory -Filter 'CLion*' |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'bin\mingw\bin\g++.exe') } |
        Sort-Object -Property Name -Descending)
    if ($found.Count -gt 0) {
      return $found[0].FullName
    }
  }
  Stop-Build 'no CLion folder with bin\mingw\bin\g++.exe: pass -ClionDir <folder of CLion>'
}

$Clion = Find-Clion
$MinGw = Join-Path $Clion 'bin\mingw\bin'
$CMakeBin = Join-Path $Clion 'bin\cmake\win\x64\bin'
$NinjaBin = Join-Path $Clion 'bin\ninja\win\x64'
$Gxx = Join-Path $MinGw 'g++.exe'
$CMake = Join-Path $CMakeBin 'cmake.exe'
$CTest = Join-Path $CMakeBin 'ctest.exe'
$Ninja = Join-Path $NinjaBin 'ninja.exe'
foreach ($tool in @($Gxx, $CMake, $CTest, $Ninja)) {
  if (-not (Test-Path -LiteralPath $tool)) {
    Stop-Build "missing $tool"
  }
}
$env:PATH = "$MinGw;$CMakeBin;$NinjaBin;$env:PATH"
Write-Host "sources: $Src"
Write-Host "builds:  $Dest\build-<preset>"
Write-Host "logs:    $Logs"
Write-Host "g++:     $Gxx"
Write-Host "cmake:   $CMake"
Write-Host "ninja:   $Ninja"

$gccVersion = @(& $Gxx --version)[0]
$cmakeVersion = @(& $CMake --version)[0]
$ninjaVersion = @(& $Ninja --version)[0]
Write-Summary "toolchain clion=$Clion gcc=$gccVersion"
Write-Summary "cmake=$cmakeVersion ninja=$ninjaVersion"
$infoFile = Join-Path $Src '.sbmc_build_info'
$sourceHash = 'unknown'
if (Test-Path -LiteralPath $infoFile) {
  foreach ($line in (Get-Content -LiteralPath $infoFile)) {
    if ($line -match '^git_hash=([0-9a-f]{40})$') {
      $sourceHash = $Matches[1]
    }
  }
}
Write-Summary "source git_hash=$sourceHash"

$pass = $true
Push-Location -LiteralPath $Src
try {
  foreach ($preset in @('win-cpu-strict', 'win-cpu')) {
    Write-Host "--- $preset"
    $configureLog = Join-Path $Logs "${preset}_configure.log"
    $configure = Invoke-Logged $configureLog $CMake @('--preset', $preset)
    $build = -1
    $warnings = -1
    $test = -1
    $total = 0
    $failed = 0
    $skipped = 0
    if ($configure -eq 0) {
      $buildLog = Join-Path $Logs "${preset}_build.log"
      $build = Invoke-Logged $buildLog $CMake @('--build', '--preset', $preset)
      $warnings = @(Select-String -LiteralPath $buildLog -Pattern 'warning' -SimpleMatch).Count
      if ($build -eq 0) {
        $testLog = Join-Path $Logs "${preset}_ctest.log"
        $test = Invoke-Logged $testLog $CTest @('--preset', $preset)
        $result = @(Select-String -LiteralPath $testLog `
            -Pattern '(\d+)% tests passed, (\d+) tests failed out of (\d+)')
        if ($result.Count -gt 0) {
          $groups = $result[$result.Count - 1].Matches[0].Groups
          $failed = [int]$groups[2].Value
          $total = [int]$groups[3].Value
        }
        $skipPattern = '\*\*\*Skipped|\*\*\*Not Run'
        $skipped = @(Select-String -LiteralPath $testLog -Pattern $skipPattern).Count
      }
    }
    $passed = $total - $failed
    $format = 'preset {0} configure={1} build={2} warnings={3} '
    $format += 'tests={4}/{5} failed={6} skipped={7} ctest={8}'
    Write-Summary ($format -f $preset, $configure, $build, $warnings, $passed, $total, $failed,
      $skipped, $test)
    $built = ($configure -eq 0) -and ($build -eq 0) -and ($warnings -eq 0)
    $tested = ($test -eq 0) -and ($total -gt 0) -and ($failed -eq 0)
    if (-not ($built -and $tested)) {
      $pass = $false
    }
  }
} finally {
  Pop-Location
}

if ($pass) {
  Write-Summary 'WIN_BUILD: PASS'
} else {
  Write-Summary 'WIN_BUILD: FAIL'
}
Save-Summary
if ($pass) {
  exit 0
}
exit 1
