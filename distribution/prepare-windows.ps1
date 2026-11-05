# Installe dubGPUStack dans distribution\Windows11.
# Chaque fichier est copié depuis le dépôt ou le toolkit CUDA 12 s'il est déjà là,
# sinon téléchargé depuis la Release GitHub décrite par manifest.json.
#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

$Distribution = $PSScriptRoot
$Root = Split-Path -Parent $Distribution
$Dest = Join-Path $Distribution 'Windows11'
$ManifestPath = Join-Path $Distribution 'manifest.json'

function Read-Manifest {
    $raw = [System.IO.File]::ReadAllText($ManifestPath)
    return $raw | ConvertFrom-Json
}

function Ensure-Parent([string]$Path) {
    $parent = Split-Path -Parent $Path
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
}

function Copy-Payload([string]$Source, [string]$DestPath) {
    Ensure-Parent $DestPath
    $sourceLen = (Get-Item -LiteralPath $Source).Length
    if ((Test-Path -LiteralPath $DestPath) -and ((Get-Item -LiteralPath $DestPath).Length -eq $sourceLen)) {
        Write-Host "deja en place  $(Split-Path -Leaf $DestPath)"
        return
    }
    Write-Host "copie  $Source"
    Copy-Item -LiteralPath $Source -Destination $DestPath -Force
}

function Get-RemoteLength([string]$Url) {
    $headers = & curl.exe -sI -L --fail $Url
    if ($LASTEXITCODE -ne 0) { return [int64]-1 }
    $length = [int64]-1
    foreach ($line in @($headers)) {
        if ($line -match '(?i)^content-length:\s*(\d+)') {
            $length = [int64]$Matches[1]
        }
    }
    return $length
}

function Save-Url([string]$Url, [string]$DestPath) {
    Ensure-Parent $DestPath
    $remote = Get-RemoteLength $Url
    if ($remote -ge 0 -and (Test-Path -LiteralPath $DestPath) -and ((Get-Item -LiteralPath $DestPath).Length -eq $remote)) {
        Write-Host "deja en place  $(Split-Path -Leaf $DestPath)"
        return $true
    }
    Write-Host "telechargement  $Url"
    $partial = "$DestPath.partial"
    & curl.exe -L --fail --retry 3 --retry-delay 2 -o $partial $Url
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
        return $false
    }
    if (Test-Path -LiteralPath $DestPath) { Remove-Item -LiteralPath $DestPath -Force }
    Move-Item -LiteralPath $partial -Destination $DestPath -Force
    return $true
}

function Find-CudaFile([string]$Name) {
    $toolkit = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path -LiteralPath $toolkit) {
        $versions = @(Get-ChildItem -LiteralPath $toolkit -Directory | Where-Object { $_.Name -like 'v12*' } | Sort-Object Name -Descending)
        foreach ($version in $versions) {
            foreach ($rel in @('bin', 'lib\x64', 'bin\x64')) {
                $candidate = Join-Path $version.FullName (Join-Path $rel $Name)
                if (Test-Path -LiteralPath $candidate) { return $candidate }
            }
        }
    }
    foreach ($extra in @('C:\Program Files\NVIDIA\CUDNN', 'C:\Program Files\NVIDIA GPU Computing Toolkit\cuDNN')) {
        if (-not (Test-Path -LiteralPath $extra)) { continue }
        $hit = Get-ChildItem -LiteralPath $extra -Recurse -Filter $Name -File -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    $onPath = Get-Command $Name -ErrorAction SilentlyContinue
    if ($onPath -and $onPath.Source) { return $onPath.Source }
    return $null
}

function Install-Asset($Item, [string]$Repo, [string]$Tag) {
    $destPath = Join-Path $Dest ($Item.dest -replace '/', '\')
    $local = $null
    if ($Item.local) {
        $candidate = Join-Path $Root ($Item.local -replace '/', '\')
        if (Test-Path -LiteralPath $candidate) { $local = $candidate }
    }
    if (-not $local -and $Item.cuda) {
        $local = Find-CudaFile $Item.cuda
    }
    if ($local) {
        Copy-Payload $local $destPath
        return $true
    }
    $asset = [uri]::EscapeDataString([string]$Item.asset)
    $url = "https://github.com/$Repo/releases/download/$Tag/$asset"
    if (Save-Url $url $destPath) { return $true }
    Write-Host "introuvable  $($Item.asset)"
    return $false
}

function Install-VcRuntime([string]$DestDir) {
    $names = @(
        'msvcp140.dll',
        'msvcp140_1.dll',
        'msvcp140_atomic_wait.dll',
        'vcruntime140.dll',
        'vcruntime140_1.dll'
    )
    $system32 = Join-Path $env:SystemRoot 'System32'
    $missing = New-Object System.Collections.Generic.List[string]
    foreach ($name in $names) {
        $target = Join-Path $DestDir $name
        $src = Join-Path $system32 $name
        if (Test-Path -LiteralPath $src) {
            Copy-Payload $src $target
        } elseif (-not (Test-Path -LiteralPath $target)) {
            $missing.Add($name)
        }
    }
    if ($missing.Count -eq 0) { return }

    $tmp = Join-Path $env:TEMP 'dubGPUStack-vcredist'
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    $exe = Join-Path $tmp 'vc_redist.x64.exe'
    $extract = Join-Path $tmp 'extract'
    if (-not (Test-Path -LiteralPath $exe)) {
        Write-Host 'telechargement  https://aka.ms/vs/17/release/vc_redist.x64.exe'
        & curl.exe -L --fail --retry 3 --retry-delay 2 -o $exe 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
        if ($LASTEXITCODE -ne 0) { throw 'telechargement du redistribuable Visual C++ impossible' }
    }
    if (Test-Path -LiteralPath $extract) { Remove-Item -LiteralPath $extract -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $extract | Out-Null
    Expand-VcRedistCabinets $exe $extract

    foreach ($name in $missing) {
        $found = Find-VcDll $extract $name
        if (-not $found) { throw "$name introuvable dans le redistribuable Visual C++" }
        Copy-Payload $found (Join-Path $DestDir $name)
    }
}

function Expand-VcRedistCabinets([string]$Exe, [string]$Extract) {
    $data = [System.IO.File]::ReadAllBytes($Exe)
    $latin = [System.Text.Encoding]::GetEncoding(28591)
    $text = $latin.GetString($data)
    $index = 0
    $written = 0
    while (($at = $text.IndexOf('MSCF', $index)) -ge 0) {
        $index = $at + 4
        if ($at + 28 -gt $data.Length) { continue }
        $size = [int][BitConverter]::ToUInt32($data, $at + 8)
        $major = $data[$at + 25]
        if ($major -ne 1 -or $size -lt 64 -or ($at + $size) -gt $data.Length) { continue }
        $slice = New-Object byte[] $size
        [Array]::Copy($data, $at, $slice, 0, $size)
        $cab = Join-Path $Extract ("cab{0}.cab" -f $written)
        [System.IO.File]::WriteAllBytes($cab, $slice)
        $dest = Join-Path $Extract ("cab{0}" -f $written)
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        & expand.exe '-F:*' $cab $dest | Out-Null
        $written++
    }
    if ($written -eq 0) { throw 'aucune archive dans vc_redist.x64.exe' }

    $nested = @(Get-ChildItem -LiteralPath $Extract -Recurse -File)
    foreach ($file in $nested) {
        $stream = [System.IO.File]::OpenRead($file.FullName)
        $header = New-Object byte[] 4
        $read = $stream.Read($header, 0, 4)
        $stream.Close()
        if ($read -lt 4 -or $header[0] -ne 77 -or $header[1] -ne 83 -or $header[2] -ne 67 -or $header[3] -ne 70) { continue }
        if ($file.Name -like 'Windows*' -or $file.Name -like 'WSUSSCAN*') { continue }
        $dest = Join-Path $file.DirectoryName ($file.BaseName + '_x')
        if (Test-Path -LiteralPath $dest) { continue }
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        & expand.exe '-F:*' $file.FullName $dest | Out-Null
    }
}

function Find-VcDll([string]$Root, [string]$Name) {
    $exact = Get-ChildItem -LiteralPath $Root -Recurse -Filter $Name -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($exact) { return $exact.FullName }
    $amd64 = Get-ChildItem -LiteralPath $Root -Recurse -Filter ($Name + '_amd64') -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($amd64) { return $amd64.FullName }
    return $null
}

$manifest = Read-Manifest
if (-not (Test-Path -LiteralPath $Dest)) {
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
}

$missing = New-Object System.Collections.Generic.List[string]
foreach ($item in $manifest.windows) {
    if (-not (Install-Asset $item $manifest.repo $manifest.tag)) {
        $missing.Add([string]$item.asset)
    }
}
Install-VcRuntime $Dest

if ($missing.Count -gt 0) {
    Write-Host ("Fichiers introuvables : " + ($missing -join ', '))
    exit 1
}
Write-Host "Installation Windows prete : $Dest"
