[CmdletBinding()]
param(
    [string]$RepoPath,
    [switch]$Push
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $RepoPath) {
    $desktop = [Environment]::GetFolderPath([Environment+SpecialFolder]::DesktopDirectory)
    if (-not $desktop) { throw 'Supply -RepoPath for your Desktop checkout.' }
    $RepoPath = Join-Path $desktop 'dsh-simulation-config'
}
$RepoPath = (Resolve-Path -LiteralPath $RepoPath).Path
$python = Join-Path $RepoPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'The isolated Python environment is missing. Finish the original Desktop setup first.'
}
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'package.json') -Raw | ConvertFrom-Json
$bundle = Join-Path $PSScriptRoot 'windows-smoke.bundle'
if ($manifest.branch -ne 'refactor/simulation-config') { throw 'Unexpected package branch.' }
if ((Get-FileHash -LiteralPath $bundle -Algorithm SHA256).Hash -ine $manifest.bundle_sha256) {
    throw 'Bundle checksum mismatch. Extract the ZIP again.'
}
$branch = & git -C $RepoPath branch --show-current
if ($LASTEXITCODE -ne 0 -or $branch -ne $manifest.branch) {
    throw 'Select refactor/simulation-config before applying this package.'
}
& git -C $RepoPath diff --quiet HEAD --
if ($LASTEXITCODE -ne 0) { throw 'Tracked files have local changes. Commit or preserve those changes before applying this update.' }
$head = & git -C $RepoPath rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Cannot read the current commit.' }
if ($head -eq $manifest.base) {
    & git -C $RepoPath bundle verify $bundle
    if ($LASTEXITCODE -ne 0) { throw 'Bundle verification failed.' }
    & git -C $RepoPath fetch $bundle ('refs/heads/' + $manifest.branch + ':refs/remotes/dsh-delivery/windows-smoke')
    if ($LASTEXITCODE -ne 0) { throw 'Could not import the validation update.' }
    & git -C $RepoPath merge --ff-only $manifest.tip
    if ($LASTEXITCODE -ne 0) { throw 'Fast-forward failed. No reset or force operation was attempted.' }
} elseif ($head -ne $manifest.tip) {
    throw 'Checkout differs from the expected milestone. Nothing was overwritten; report the current commit before applying.'
}

& $python (Join-Path $RepoPath 'scripts\validate_configured_smoke.py') --repo $RepoPath
if ($LASTEXITCODE -ne 0) { throw 'Smoke validation failed. Send the smoke_report.zip printed above for diagnosis.' }

if ($Push) {
    $origin = & git -C $RepoPath remote get-url --push origin
    if ($LASTEXITCODE -ne 0 -or $origin -notmatch '^(https://github\.com/|git@github\.com:|ssh://git@github\.com/)AhmedAbdullahAbbasi/dsh-ray-tracing-3(?:\.git)?/?$') {
        throw 'Origin must point to your AhmedAbdullahAbbasi/dsh-ray-tracing-3 repository.'
    }
    & git -C $RepoPath push --set-upstream origin $manifest.branch
    if ($LASTEXITCODE -ne 0) { throw 'Push failed; local validation reports are intact.' }
    $remoteLine = & git -C $RepoPath ls-remote --heads origin ('refs/heads/' + $manifest.branch)
    if ($LASTEXITCODE -ne 0 -or -not $remoteLine -or ($remoteLine -split '\s+')[0] -ne $manifest.tip) {
        throw 'Remote branch could not be verified at the validation-tool checkpoint.'
    }
}
Write-Host 'Windows smoke run finished. Send the printed smoke_report.zip for review.'
