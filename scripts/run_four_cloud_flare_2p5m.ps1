<#
Run a 2.5-million-photon, one-hour hard-state flare through a four-cloud FITS cube.
Photon energies are sampled continuously over 2-10 keV from E^(-Gamma).
The three output images integrate observer arrival days [3,4), [6,7), [9,10)
by default; their start times are three days apart. Run from a PowerShell
terminal in the checkout containing the integrated 2-10 keV tables.

Example:
  .\scripts\run_four_cloud_flare_2p5m.ps1 -CloudFits 'C:\data\realistic_nh_cube_500asec_10kpc_dz0p05.fits' -SourceDistanceKpc 10.5
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$CloudFits,
    [ValidateRange(0.001, 100000)]
    [double]$SourceDistanceKpc = 10.5,
    [ValidateRange(1, 52)]
    [int]$FirstSnapshotDay = 3,
    [ValidateRange(1, 2500000)]
    [int]$ChunkSize = 512,
    [ValidateRange(1, 128)]
    [int]$MaxInteractions = 16,
    [double]$PhotonIndex = 1.7,
    [double]$PhotonFlux2to10 = 0.038,
    [string]$OutputDir = 'outputs/flare_2p5m_four_cloud_2_10'
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$cloudPath = (Resolve-Path -LiteralPath $CloudFits -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath $cloudPath -PathType Leaf)) {
    throw "CloudFits must be a FITS file: $cloudPath"
}
if ($FirstSnapshotDay + 7 -ge 60) {
    throw 'The third one-day snapshot must end before the 60-day arrival limit.'
}

Push-Location $repositoryRoot
try {
    $chosenOutput = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
        $OutputDir
    } else {
        Join-Path (Get-Location).Path $OutputDir
    }
    $productDir = [System.IO.Path]::GetFullPath($chosenOutput)
    $sourceDistanceArgument = $SourceDistanceKpc.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    $photonIndexArgument = $PhotonIndex.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    $photonFluxArgument = $PhotonFlux2to10.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    if ([double]::IsNaN($PhotonIndex) -or [double]::IsInfinity($PhotonIndex) -or
        [double]::IsNaN($PhotonFlux2to10) -or [double]::IsInfinity($PhotonFlux2to10) -or
        $PhotonFlux2to10 -le 0) {
        throw 'PhotonIndex must be finite and PhotonFlux2to10 must be finite and positive.'
    }
    New-Item -ItemType Directory -Force -Path $productDir | Out-Null
    $npzPath = Join-Path $productDir 'flare_2p5m_full.npz'
    $fitsPath = Join-Path $productDir 'flare_2p5m_full.fits'
    $snapshotDir = Join-Path $productDir 'snapshots'
    $secondDay = $FirstSnapshotDay + 3
    $thirdDay = $FirstSnapshotDay + 6
    $firstEnd = $FirstSnapshotDay + 1
    $secondEnd = $secondDay + 1
    $thirdEnd = $thirdDay + 1

    python -m unittest tests.test_material_v2 tests.test_source tests.test_npz_output tests.test_fits_output tests.test_flare_snapshots -v
    if ($LASTEXITCODE -ne 0) { throw 'Material/source/output preflight tests failed.' }

    python -m scripts.run_dsh_v1 `
        --materials 2-10 `
        --source-model constant-flare `
        --source-spectrum hard-state-powerlaw `
        --photon-index $photonIndexArgument `
        --total-2-10-photon-flux $photonFluxArgument `
        --cloud-fits "$cloudPath" `
        --source-distance-kpc $sourceDistanceArgument `
        --packets 2500000 `
        --chunk-size $ChunkSize `
        --max-interactions $MaxInteractions `
        --seed 2026 `
        --arrival-time-edges-days 0 $FirstSnapshotDay $firstEnd $secondDay $secondEnd $thirdDay $thirdEnd 60 `
        --output "$npzPath" `
        --fits-output "$fitsPath"
    if ($LASTEXITCODE -ne 0) { throw 'The 2.5-million-photon run failed.' }

    python -m scripts.extract_flare_snapshots `
        --input-fits "$fitsPath" `
        --output-dir "$snapshotDir" `
        --first-day $FirstSnapshotDay `
        --separation-days 3 `
        --exposure-days 1 `
        --expected-packets 2500000
    if ($LASTEXITCODE -ne 0) { throw 'Snapshot validation/extraction failed.' }

    Write-Host "Success. Three time-stamped FITS images and a manifest: $snapshotDir"
    Write-Host "Full simulation: $fitsPath"
    Write-Host "Reproducibility archive: $npzPath"
} finally {
    Pop-Location
}
