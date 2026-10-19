[CmdletBinding()]
param(
    [string]$Destination,
    [switch]$SkipInstall,
    [switch]$Push
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Install Git for Windows before running this script.'
}

$manifestPath = Join-Path $PSScriptRoot 'package.json'
$bundlePath = Join-Path $PSScriptRoot 'simulation-config.bundle'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $bundlePath -PathType Leaf)) {
    throw 'Extract the complete delivery ZIP. Keep package.json and simulation-config.bundle beside this script.'
}
$package = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$repoUrl = 'https://github.com/AhmedAbdullahAbbasi/dsh-ray-tracing-3.git'
$branch = 'refactor/simulation-config'
if ($package.repository -ne $repoUrl -or $package.branch -ne $branch) {
    throw 'Package repository or branch does not match this setup script.'
}
if ((Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash -ine $package.bundle_sha256) {
    throw 'Bundle checksum mismatch. Download and extract the package again.'
}
if (-not $Destination) {
    $desktop = [Environment]::GetFolderPath([Environment+SpecialFolder]::DesktopDirectory)
    if (-not $desktop) { throw 'Windows Desktop could not be located. Supply -Destination explicitly.' }
    $Destination = Join-Path $desktop 'dsh-simulation-config'
}
$Destination = [IO.Path]::GetFullPath($Destination)

if (Test-Path -LiteralPath $Destination) {
    # Never repurpose or clean an existing personal checkout.
    $existingHead = & git -C $Destination rev-parse HEAD 2>$null
    if ($LASTEXITCODE -ne 0 -or $existingHead -ne $package.tip) {
        throw "Destination already exists and is not this exact package checkout: $Destination. Choose a new -Destination."
    }
    $existingBranch = & git -C $Destination branch --show-current
    if ($LASTEXITCODE -ne 0 -or $existingBranch -ne $branch) {
        throw 'Existing destination is on another branch. Select a new destination or switch it deliberately yourself.'
    }
    $existingOrigin = & git -C $Destination remote get-url --push origin
    if ($LASTEXITCODE -ne 0 -or $existingOrigin -ne $repoUrl) {
        throw 'Existing destination has a different origin. Nothing was changed.'
    }
    Write-Host "Reusing the exact package checkout at $Destination"
} else {
    $parent = Split-Path -Parent $Destination
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    & git clone --config core.autocrlf=false --config core.longpaths=true --single-branch --branch development/post-reflex-improvements $repoUrl $Destination
    if ($LASTEXITCODE -ne 0) { throw 'Clone failed. Check GitHub access; any partial destination was left intact.' }
    & git -C $Destination cat-file -e ($package.base + '^{commit}')
    if ($LASTEXITCODE -ne 0) { throw 'Published prerequisite commit is missing; no older checkpoint was substituted.' }
    & git -C $Destination bundle verify $bundlePath
    if ($LASTEXITCODE -ne 0) { throw 'Git bundle verification failed.' }
    & git -C $Destination fetch $bundlePath ('refs/heads/' + $branch + ':refs/heads/' + $branch)
    if ($LASTEXITCODE -ne 0) { throw 'Could not import the exact improvement history.' }
    & git -C $Destination switch $branch
    if ($LASTEXITCODE -ne 0) { throw 'Could not select the new refactor branch.' }
}

$actualHead = & git -C $Destination rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $actualHead -ne $package.tip) {
    throw 'Checkout tip does not match the delivery manifest.'
}

if (-not $SkipInstall) {
    $python = Join-Path $Destination '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3 -m venv (Join-Path $Destination '.venv')
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            & python -m venv (Join-Path $Destination '.venv')
        } else {
            throw 'Install Python 3.10 or newer, then rerun this script.'
        }
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated Python environment.' }
    }
    & $python -c 'import sys; sys.exit(sys.version_info < (3, 10))'
    if ($LASTEXITCODE -ne 0) { throw 'Unsupported Python version.' }
    & $python -m pip install -e "${Destination}[dev,plots]"
    if ($LASTEXITCODE -ne 0) { throw 'Package installation failed. Checkout is intact; rerun after resolving the dependency error.' }
    Push-Location -LiteralPath $Destination
    try {
        & $python -m unittest tests.test_run_config tests.test_configured_run -v
        if ($LASTEXITCODE -ne 0) { throw 'Configured-run tests failed. No automatic push was performed.' }
        & $python -m dsh.command check (Join-Path $Destination 'configs\smoke_legacy.toml')
        if ($LASTEXITCODE -ne 0) { throw 'Legacy-material configuration check failed.' }
        & $python -m dsh.command check (Join-Path $Destination 'configs\smoke_wd01.toml')
        if ($LASTEXITCODE -ne 0) { throw 'Declared-material configuration check failed.' }
    } finally {
        Pop-Location
    }
}

if ($Push) {
    & git -C $Destination push --set-upstream origin ('refs/heads/' + $branch + ':refs/heads/' + $branch)
    if ($LASTEXITCODE -ne 0) { throw 'Push failed. Your local checkout remains ready; check GitHub authentication or branch conflicts.' }
    $remoteLine = & git -C $Destination ls-remote --heads origin ('refs/heads/' + $branch)
    if ($LASTEXITCODE -ne 0 -or -not $remoteLine -or ($remoteLine -split '\s+')[0] -ne $package.tip) {
        throw 'Remote branch could not be verified at the expected commit.'
    }
    Write-Host "Published and verified $branch at $($package.tip)"
}

Write-Host "Ready: $Destination"
Write-Host "Branch: $branch"
Write-Host 'Example: .\.venv\Scripts\dsh.exe run configs\smoke_legacy.toml'
Write-Host 'Your earlier project directory was not modified.'
