<#
.SYNOPSIS
Builds and tests the presets win-cpu-strict and win-cpu on Windows with the MinGW-w64 GCC, CMake
and Ninja of CLion (phase 07, part 4, decision W1; part 4b, decision W5: dynamic link, with the
runtime DLLs of the toolchain next to the executables).

.DESCRIPTION
Run from a PowerShell 5.1 window (no administrator rights, no modules), after
scripts/windows/stage.sh has copied the commit (DEST = C:\Users\Simone\sbmc-win):

  powershell -NoProfile -ExecutionPolicy Bypass -File DEST\src\scripts\windows\build.ps1

The sources are the copy of stage.sh, DEST\src (two folders above this script); the builds go to
DEST\build-<preset> (the binaryDir of the presets), configured from a new cache at every run
(cmake --fresh, so every run builds from scratch), and the logs to DEST\logs:
<preset>_configure.log, <preset>_build.log, <preset>_ctest.log, <preset>_runtime.log and
build_summary.txt, whose lines are also written on the console:

  toolchain clion=<folder> gcc=<first line of g++ --version>
  cmake=<first line of cmake --version> ninja=<version>
  source git_hash=<git_hash of DEST\src\.sbmc_build_info>
  preset <name> configure=<exit> build=<exit> warnings=<lines> tests=<passed>/<total>
    failed=<n> skipped=<n> ctest=<exit>     (one line per preset)
  runtime <name> dlls=<DLL,...> standalone=PASS|FAIL     (one line per preset)
  WIN_BUILD: PASS|FAIL

The runtime line (after a successful build): the DLLs imported by apps\sbmc_run.exe and
tests\tools\sbmc_sa_thresholds.exe (objdump -p of CLion), followed through the imports of the
DLLs found next to them; dlls= lists those next to the executables. standalone=PASS when every
imported DLL is next to the executable, in System32 or an API set (api-ms-win-*, ext-ms-*), and
then sbmc_run.exe --version exits with 0 with a PATH of System32 and Windows only (no folder of
CLion). The details go to <preset>_runtime.log. sbmc_run.exe is started only when no DLL is
missing, to avoid the dialog of Windows about a missing DLL.

PASS: both presets configured, built and tested with exit status 0, no line with "warning" in
the build logs, at least one test, no test failed, standalone=PASS. The toolchain: -ClionDir, or
the newest folder CLion* of $env:ProgramFiles\JetBrains, then of $env:LOCALAPPDATA\Programs
(Toolbox), with bin\mingw\bin (g++, objdump), bin\cmake\win\x64\bin (cmake, ctest) and
bin\ninja\win\x64 (ninja), put first in the PATH of this process only.

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

# One item of "& <native command> 2>&1" as the line it stands for. PowerShell 5.1 wraps every
# line of stderr in an ErrorRecord, and "$item" of an empty one is the name of the exception
# (System.Management.Automation.RemoteException); its Exception.Message is the line itself, empty
# for an empty line.
function ConvertTo-LogLine($Item) {
  if ($Item -is [System.Management.Automation.ErrorRecord]) {
    if (($null -ne $Item.Exception) -and ($null -ne $Item.Exception.Message)) {
      return [string]$Item.Exception.Message
    }
    return ''
  }
  return [string]$Item
}

# A native command with its stdout and stderr in a log file (ASCII); returns its exit status.
function Invoke-Logged([string]$Log, [string]$Exe, [string[]]$Arguments) {
  $previous = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    & $Exe @Arguments 2>&1 | ForEach-Object { ConvertTo-LogLine $_ } |
      Out-File -FilePath $Log -Encoding ascii
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
$Objdump = Join-Path $MinGw 'objdump.exe'
$CMake = Join-Path $CMakeBin 'cmake.exe'
$CTest = Join-Path $CMakeBin 'ctest.exe'
$Ninja = Join-Path $NinjaBin 'ninja.exe'
foreach ($tool in @($Gxx, $Objdump, $CMake, $CTest, $Ninja)) {
  if (-not (Test-Path -LiteralPath $tool)) {
    Stop-Build "missing $tool"
  }
}
$SystemDir = Join-Path $env:SystemRoot 'System32'
$CleanPath = "$SystemDir;$env:SystemRoot;$(Join-Path $SystemDir 'Wbem')"
$BuildRoot = Split-Path -Parent $Src
$env:PATH = "$MinGw;$CMakeBin;$NinjaBin;$env:PATH"
Write-Host "sources: $Src"
Write-Host "builds:  $BuildRoot\build-<preset>"
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

# The DLLs imported by a PE file (objdump -p, lines "DLL Name: <name>"); $null when objdump fails.
function Get-DllImport([string]$File) {
  $previous = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    $lines = @(& $Objdump -p $File 2>$null)
    $code = $LASTEXITCODE
  } catch {
    $lines = @()
    $code = -1
  } finally {
    $ErrorActionPreference = $previous
  }
  if ($code -ne 0) {
    return $null
  }
  $names = New-Object System.Collections.Generic.List[string]
  foreach ($line in $lines) {
    if ("$line" -match 'DLL Name:\s*(\S+)') {
      $names.Add($Matches[1])
    }
  }
  return ,$names.ToArray()
}

# The runtime check of one preset: the imports of the executables, followed through the DLLs
# found next to them, then sbmc_run.exe --version with the clean PATH. Writes the log and
# returns the runtime line of the summary.
function Test-Runtime([string]$Preset, [string]$Log) {
  $buildDir = Join-Path $BuildRoot "build-$Preset"
  $exes = @((Join-Path $buildDir 'apps\sbmc_run.exe'),
    (Join-Path $buildDir 'tests\tools\sbmc_sa_thresholds.exe'))
  $report = New-Object System.Collections.Generic.List[string]
  $found = @{}
  $missing = New-Object System.Collections.Generic.List[string]
  foreach ($exe in $exes) {
    $report.Add("executable $exe")
    if (-not (Test-Path -LiteralPath $exe)) {
      $missing.Add("(no $exe)")
      continue
    }
    $dir = Split-Path -Parent $exe
    $seen = @{}
    $queue = New-Object System.Collections.Generic.Queue[string]
    $queue.Enqueue($exe)
    while ($queue.Count -gt 0) {
      $file = $queue.Dequeue()
      $imports = Get-DllImport $file
      if ($null -eq $imports) {
        $missing.Add("(objdump failed on $file)")
        continue
      }
      $report.Add("  imports of $(Split-Path -Leaf $file): $($imports -join ' ')")
      foreach ($name in $imports) {
        $key = $name.ToLowerInvariant()
        if ($seen.ContainsKey($key)) {
          continue
        }
        $seen[$key] = $true
        $path = Join-Path $dir $name
        $apiSet = $key.StartsWith('api-ms-win-') -or $key.StartsWith('ext-ms-')
        if (Test-Path -LiteralPath $path) {
          $found[$key] = $name
          $queue.Enqueue($path)
        } elseif ($apiSet -or (Test-Path -LiteralPath (Join-Path $SystemDir $name))) {
          $report.Add("  system: $name")
        } else {
          $missing.Add($name)
          $report.Add("  MISSING: $name (not next to the executable, not in $SystemDir)")
        }
      }
    }
  }
  $dlls = @($found.Values | Sort-Object)
  $versionCode = 'not run'
  if ($missing.Count -eq 0) {
    $runExe = $exes[0]
    $output = @()
    $previousPath = $env:PATH
    $previous = $ErrorActionPreference
    $env:PATH = $CleanPath
    $ErrorActionPreference = 'Continue'
    try {
      $output = @(& $runExe --version 2>&1 | ForEach-Object { ConvertTo-LogLine $_ })
      $versionCode = $LASTEXITCODE
    } catch {
      $output = @("not started: $($_.Exception.Message)")
      $versionCode = 'not started'
    } finally {
      $ErrorActionPreference = $previous
      $env:PATH = $previousPath
    }
    $report.Add("PATH=$CleanPath")
    $report.Add("sbmc_run.exe --version: exit $versionCode")
    foreach ($line in $output) {
      $report.Add("  $line")
    }
  }
  $standalone = 'FAIL'
  if (($missing.Count -eq 0) -and ($versionCode -eq 0)) {
    $standalone = 'PASS'
  } else {
    $report.Add("missing: $($missing -join ' '); --version: $versionCode")
  }
  [System.IO.File]::WriteAllLines($Log, $report.ToArray())
  return "runtime $Preset dlls=$($dlls -join ',') standalone=$standalone"
}

$pass = $true
Push-Location -LiteralPath $Src
try {
  foreach ($preset in @('win-cpu-strict', 'win-cpu')) {
    Write-Host "--- $preset"
    $configureLog = Join-Path $Logs "${preset}_configure.log"
    # --fresh: a new cache at every run, so that no value of an earlier configure survives (the
    # -static of F07-R4 stays in CMakeCache.txt when the presets no longer set it).
    $configure = Invoke-Logged $configureLog $CMake @('--preset', $preset, '--fresh')
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
    $runtime = "runtime $preset dlls=- standalone=FAIL"
    if (($configure -eq 0) -and ($build -eq 0)) {
      $runtime = Test-Runtime $preset (Join-Path $Logs "${preset}_runtime.log")
    }
    Write-Summary $runtime
    $built = ($configure -eq 0) -and ($build -eq 0) -and ($warnings -eq 0)
    $tested = ($test -eq 0) -and ($total -gt 0) -and ($failed -eq 0)
    $standalone = $runtime.EndsWith('standalone=PASS')
    if (-not ($built -and $tested -and $standalone)) {
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
