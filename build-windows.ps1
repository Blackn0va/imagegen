<#
.SYNOPSIS
    Baut StreamForge Studio als PyInstaller-Bundle (--onedir) für Windows.

.DESCRIPTION
    Ablauf: Build-Venv anlegen -> Abhängigkeiten installieren -> optional
    Modell vorladen -> PyInstaller -> CUDA-Laufzeit neben die .exe kopieren
    -> Lizenzhinweise beilegen -> Ergebnis prüfen.

    Der Build läuft bewusst in einem eigenen Venv, damit die Entwicklungs-
    umgebung nicht mit den GPU-Wheels vermischt wird.

    Hinweis: Skript in einer UTF-8-fähigen Shell ausführen (PowerShell 7+),
    sonst werden die Umlaute in den Meldungen falsch dargestellt.

.PARAMETER Python
    Pfad zum Python-Interpreter (Vorgabe: py -3.13/3.12/3.11, sonst python).

.PARAMETER Model
    Kurznamen der Modelle, die mitgebündelt werden (Vorgabe: sdxl-base und
    realesrgan-x4 zum Vergrößern). Landet in dist\<Name>\data\models; das
    Bundle läuft dann portabel.

.PARAMETER WithCuda
    NVIDIA-Laufzeit installieren und neben die .exe legen (Vorgabe: $true).

.PARAMETER SkipModelDownload
    Kein Modell vorladen – für schnelle Testbauten.

.PARAMETER NoGui
    Nur die Kommandozeilen-Variante bauen (ohne tkinter).

.PARAMETER Clean
    Vorherige Artefakte (build, dist, Venv) vorher entfernen.

.PARAMETER FfmpegDir
    Ordner mit einem LGPL-ffmpeg-Build (ffmpeg.exe, ffprobe.exe). Wird nach
    dist\<Name>\tools\ffmpeg kopiert. WICHTIG: keinen GPL-Build verwenden.

.EXAMPLE
    .\build-windows.ps1 -Clean
.EXAMPLE
    .\build-windows.ps1 -WithCuda:$false -SkipModelDownload -NoGui
#>

[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$Name = "StreamForge",
    [string[]]$Model = @("sdxl-base", "realesrgan-x4"),
    [bool]$WithCuda = $true,
    [switch]$SkipModelDownload,
    [switch]$NoGui,
    [switch]$Clean,
    [string]$FfmpegDir = "",
    [bool]$WithFfmpeg = $true,
    [string]$FfmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl.zip",
    [string]$CudaIndexUrl = "https://download.pytorch.org/whl/cu126",
    # Fertige CPU-Wheels fuer llama-cpp-python. Ohne die muesste pip aus
    # Quelltext bauen (CMake + MSVC noetig).
    [string]$LlamaWheelIndex = "https://abetlen.github.io/llama-cpp-python/whl/cpu",
    # Vollstaendige URL eines CUDA-Wheels fuer llama-cpp-python. Leer = CPU.
    # Der offizielle Index hat CUDA nur bis Python 3.12; fuer neuere Fassungen
    # gibt es Wheels aus Fremdquellen. Bewusst als URL statt fest verdrahtet:
    # sie muss zur Python-Fassung UND zur CUDA-Fassung des Rechners passen.
    # Nach dem Bau mit 'streamforge chat --info' pruefen, ob wirklich "GPU"
    # gemeldet wird - sonst rechnet der Chat still auf der CPU weiter.
    [string]$LlamaCudaWheel = "",
    # Vorgabe an: '-Clean' allein soll ein vollständiges Programm bauen.
    # Auf Rechnern ohne AMD-/Intel-Grafik kostet das nur Platz, nichts sonst.
    [bool]$WithOnnx = $true,
    # Chat/Code-Writer über llama.cpp (GGUF). Ebenfalls Vorgabe an.
    [bool]$WithChat = $true,
    [bool]$WithVoiceRuntime = $false,
    [string]$VoiceRuntimeDir = "",
    [switch]$Console
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = $PSScriptRoot
$VenvDir = Join-Path $Root ".build-venv"
$DistDir = Join-Path $Root "dist"
$BuildDir = Join-Path $Root "build"
$SpecFile = Join-Path $Root "packaging\app.spec"
$Target = Join-Path $DistDir $Name

function Write-Step($Message) {
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Write-Note($Message) {
    Write-Host "    $Message" -ForegroundColor DarkGray
}

function Resolve-Python {
    param([string]$Preferred)
    if ($Preferred) {
        if (-not (Test-Path $Preferred)) { throw "Python nicht gefunden: $Preferred" }
        return $Preferred
    }
    $launcher = (Get-Command py -ErrorAction SilentlyContinue)
    if ($launcher) {
        foreach ($version in @("3.13", "3.12", "3.11")) {
            & py "-$version" --version *> $null
            if ($LASTEXITCODE -eq 0) { return "py -$version" }
        }
    }
    $fallback = (Get-Command python -ErrorAction SilentlyContinue)
    if ($fallback) { return $fallback.Source }
    throw "Kein Python 3.11+ gefunden. Mit -Python <pfad> angeben."
}

function Install-Ffmpeg {
    <#
      Holt einen LGPL-ffmpeg-Build und legt ihn neben die .exe.

      ffmpeg läuft als eigenständiges Programm in einem eigenen Prozess –
      es wird nicht gelinkt. Damit bleibt die Anwendung proprietär, solange
      der Build KEINE GPL-Bestandteile enthält. Genau das wird hier am
      Binary geprüft, nicht am Dateinamen: ein Build mit --enable-gpl
      (libx264/libx265) würde die GPL auf die gesamte Auslieferung
      erstrecken und bricht den Bau ab.
    #>
    param([string]$Url, [string]$Target, [string]$CacheDir)

    New-Item -ItemType Directory -Force $CacheDir | Out-Null
    $zip = Join-Path $CacheDir "ffmpeg-lgpl.zip"
    if (-not (Test-Path $zip)) {
        Write-Note "lade ffmpeg: $Url"
        $old = $ProgressPreference
        $ProgressPreference = "SilentlyContinue"   # sonst extrem langsam
        try {
            Invoke-WebRequest -Uri $Url -OutFile $zip -UseBasicParsing
        } finally {
            $ProgressPreference = $old
        }
    } else {
        Write-Note "nutze zwischengespeicherten Download: $zip"
    }

    $extract = Join-Path $CacheDir "x"
    if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
    Expand-Archive -Path $zip -DestinationPath $extract -Force

    $binary = Get-ChildItem -Path $extract -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
    if (-not $binary) { throw "ffmpeg.exe im Archiv nicht gefunden: $zip" }

    # Lizenz-Gegenprobe am Binary
    $config = & $binary.FullName -hide_banner -version 2>&1 | Out-String
    foreach ($flag in @("--enable-gpl", "--enable-nonfree")) {
        if ($config -match [regex]::Escape($flag)) {
            throw ("ffmpeg-Build enthält $flag – damit wäre die gesamte Anwendung " +
                   "an die GPL gebunden. LGPL-Build verwenden (Variante 'win64-lgpl').")
        }
    }
    $version = ($config -split "`n")[0].Trim()
    Write-Note "geprüft, kein GPL/nonfree: $version"

    $binDir = Join-Path $Target "bin"
    New-Item -ItemType Directory -Force $binDir | Out-Null
    # ffplay wird nicht gebraucht – spart rund 110 MB.
    foreach ($name in @("ffmpeg.exe", "ffprobe.exe")) {
        $file = Get-ChildItem -Path $extract -Recurse -Filter $name | Select-Object -First 1
        if ($file) { Copy-Item $file.FullName -Destination $binDir -Force }
    }
    $license = Get-ChildItem -Path $extract -Recurse -Filter "LICENSE.txt" | Select-Object -First 1
    if ($license) { Copy-Item $license.FullName -Destination $Target -Force }

    $hash = (Get-FileHash (Join-Path $binDir "ffmpeg.exe") -Algorithm SHA256).Hash.ToLower()
    Set-Content -Encoding utf8 -Path (Join-Path $Target "HERKUNFT.txt") -Value @"
ffmpeg – LGPL-Build, als eigenständiges Programm mitgeliefert
$version
Quelle: $Url
Lizenz: LGPL-3.0-or-later (Build ohne --enable-gpl und ohne --enable-nonfree)
Bewusst NICHT enthalten: libx264, libx265, libxvid (GPL)
SHA-256 ffmpeg.exe: $hash
Quelltext der verwendeten Fassung auf Anfrage – siehe THIRD-PARTY-NOTICES.md
"@
    $size = (Get-ChildItem -Recurse $Target | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Note ("ffmpeg eingebettet: {0} ({1:N0} MB)" -f $Target, $size)
}

function Invoke-Checked {
    param([string]$File, [string[]]$Arguments, [string]$What)
    Write-Note "$File $($Arguments -join ' ')"
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$What fehlgeschlagen (Exit $LASTEXITCODE)." }
}

function Get-PythonLine {
    <#
      Eine einzelne Ausgabezeile von Python holen.

      '& python -c ...' liefert ALLE Zeilen als Array zurueck - und manche
      Pakete (llama_cpp) drucken beim Import Meldungen. Ein Array als Pfad
      weitergereicht bricht spaeter mit "Cannot convert System.Object[]".
      Deshalb: letzte nicht-leere Zeile, getrimmt, als String.
    #>
    param([string]$Python, [string]$Code)
    $roh = & $Python -c $Code 2>$null
    if ($null -eq $roh) { return "" }
    # Das @(...) MUSS um das Ergebnis von Where-Object stehen. Bei genau
    # einer Zeile gibt Where-Object eine Zeichenkette zurueck statt eines
    # Arrays - [-1] liefert dann den letzten BUCHSTABEN statt der Zeile.
    $zeilen = @($roh | Where-Object { "$_".Trim() -ne "" })
    if ($zeilen.Count -eq 0) { return "" }
    return "$($zeilen[$zeilen.Count - 1])".Trim()
}

function Invoke-Optional {
    <#
      Fuer Bestandteile, ohne die die Anwendung laeuft: Chat, ONNX/OpenVINO.
      Ein Fehlschlag darf den ganzen Bau nicht abbrechen - sonst kostet eine
      fehlende Build-Umgebung fuer ein Nebenteil die komplette .exe. Gibt
      $true zurueck, wenn es geklappt hat.
    #>
    param([string]$File, [string[]]$Arguments, [string]$What)
    Write-Note "$File $($Arguments -join ' ')"
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$What fehlgeschlagen (Exit $LASTEXITCODE) - wird uebersprungen."
        return $false
    }
    return $true
}

# ---------------------------------------------------------------------------
Write-Step "Vorbereitung"
if ($Clean) {
    foreach ($path in @($DistDir, $BuildDir, $VenvDir)) {
        if (Test-Path $path) {
            Write-Note "entferne $path"
            Remove-Item -Recurse -Force $path
        }
    }
}

$PythonCmd = Resolve-Python -Preferred $Python
Write-Note "Interpreter: $PythonCmd"

# ---------------------------------------------------------------------------
Write-Step "Build-Umgebung"
if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    if ($PythonCmd -like "py -*") {
        $parts = $PythonCmd.Split(" ")
        Invoke-Checked -File $parts[0] -Arguments @($parts[1], "-m", "venv", $VenvDir) -What "Venv anlegen"
    } else {
        Invoke-Checked -File $PythonCmd -Arguments @("-m", "venv", $VenvDir) -What "Venv anlegen"
    }
} else {
    Write-Note "Venv ist bereits vorhanden"
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) { throw "Venv-Python fehlt: $VenvPython" }

Invoke-Checked -File $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "wheel") -What "pip aktualisieren"

# WICHTIG: Der CUDA-Stack muss VOR den Basis-Abhängigkeiten installiert
# werden. Sonst zieht pip zuerst das CPU-Wheel von PyPI, sieht danach
# "torch>=2.6,<3" als erfüllt an und überspringt das cu-Wheel – das Bundle
# hätte dann CUDA-DLLs, aber ein torch, das keine GPU findet.
# --index-url (nicht --extra-index-url), damit PyPI nicht als Quelle dient.
if ($WithCuda) {
    Write-Step "NVIDIA-Laufzeit installieren (vor der Basis)"
    Invoke-Checked -File $VenvPython -Arguments @(
        "-m", "pip", "install", "--index-url", $CudaIndexUrl,
        "torch", "torchvision", "torchaudio"
    ) -What "torch mit CUDA"
    Invoke-Checked -File $VenvPython -Arguments @(
        "-m", "pip", "install", "-r", (Join-Path $Root "requirements-cuda.txt"),
        "--extra-index-url", $CudaIndexUrl
    ) -What "CUDA-Laufzeitbibliotheken"
} else {
    Write-Note "CUDA übersprungen – der Build läuft auf CPU und DirectML."
}

Invoke-Checked -File $VenvPython -Arguments @("-m", "pip", "install", "-r", (Join-Path $Root "requirements.txt")) -What "Basis-Abhängigkeiten"

# ONNX/OpenVINO muss BEIM BAUEN ins Venv. Ein fertiges Bundle bringt seinen
# eigenen Python mit und sieht nichts, was hinterher per 'pip install' in ein
# System-Python gelegt wird.
if ($WithOnnx) {
    Write-Step "ONNX- und OpenVINO-Laufzeit installieren"
    $onnxOk = Invoke-Optional -File $VenvPython -Arguments @(
        "-m", "pip", "install", "optimum[onnxruntime]", "optimum[openvino]", "openvino"
    ) -What "optimum, OpenVINO"
    if (-not $onnxOk) {
        $WithOnnx = $false
        Write-Warning ("ONNX/OpenVINO wird nicht mitgeliefert. Bild und Chat laufen " +
                       "trotzdem; AMD-/Intel-GPU und NPU bleiben ungenutzt.")
    }
    $ovInfo = Get-PythonLine -Python $VenvPython -Code "import openvino,sys;c=openvino.Core();sys.stdout.write(openvino.__version__+'|'+','.join(c.available_devices))"
    Write-Note "OpenVINO im Bau-Venv: $ovInfo"
} else {
    Write-Note "ONNX/OpenVINO übersprungen. Für AMD-/Intel-GPU oder NPU: -WithOnnx `$true"
}

# Chat/Code-Writer: llama.cpp über llama-cpp-python (GGUF). Laut Messungen
# rund doppelt so schnell wie OpenVINO für Sprachmodelle auf Intel-CPUs.
if ($WithChat) {
    Write-Step "Chat-Laufzeit installieren (llama.cpp)"
    # llama-cpp-python liegt auf PyPI nur als Quelltext - pip wuerde CMake und
    # die MSVC-Build-Tools brauchen und ohne die abbrechen. Zuerst deshalb der
    # offizielle Wheel-Index mit fertig gebauten CPU-Fassungen; erst wenn der
    # nichts hergibt, der Weg ueber PyPI.
    if ($LlamaCudaWheel) {
        Write-Note "CUDA-Wheel angefordert: $LlamaCudaWheel"
        $chatOk = Invoke-Optional -File $VenvPython -Arguments @(
            "-m", "pip", "install", $LlamaCudaWheel
        ) -What "llama-cpp-python (CUDA-Wheel)"
        if ($chatOk) {
            # ggml-cuda.dll ist gegen cudart/cublas gelinkt und wird ohne die
            # stillschweigend uebersprungen - der Chat rechnet dann auf der CPU,
            # ohne dass irgendwo ein Fehler steht. Die DLLs von torch daneben
            # legen; ggml sucht Abhaengigkeiten im eigenen Ordner.
            $libDir = Get-PythonLine -Python $VenvPython -Code "import llama_cpp,os,sys;sys.stdout.write(os.path.join(os.path.dirname(llama_cpp.__file__),'lib'))"
            $sitePkgs = Get-PythonLine -Python $VenvPython -Code "import sysconfig,sys;sys.stdout.write(sysconfig.get_paths()['purelib'])"
            if ($libDir -and (Test-Path $libDir)) {
                $kopiert = 0
                foreach ($muster in @("nvidia\*in\cudart64_*.dll", "nvidia\*in\cublas*64_*.dll",
                                      "torch\lib\cudart64_*.dll", "torch\lib\cublas*64_*.dll")) {
                    $pfad = Join-Path $sitePkgs $muster
                    foreach ($dll in @(Get-ChildItem -Path $pfad -ErrorAction SilentlyContinue)) {
                        Copy-Item -LiteralPath $dll.FullName -Destination "$libDir" -Force -ErrorAction SilentlyContinue
                        $kopiert++
                    }
                }
                Write-Note "CUDA-Laufzeit neben ggml gelegt: $kopiert Datei(en)"
            }
            $gpuInfo = Get-PythonLine -Python $VenvPython -Code "import sys;sys.path.insert(0,'.');from app import pipeline_chat;sys.stdout.write(str(pipeline_chat.gpu_offload_possible()))"
            if ($gpuInfo -notmatch "True") {
                Write-Warning ("Das CUDA-Wheel meldet KEINE GPU-Unterstuetzung ($gpuInfo). " +
                               "Der Chat wuerde auf der CPU rechnen. Passt die Wheel-Fassung " +
                               "zu Python und zur CUDA-Fassung des Rechners?")
            } else {
                Write-Note "llama.cpp mit GPU-Unterstuetzung."
            }
        }
    } else {
        $chatOk = Invoke-Optional -File $VenvPython -Arguments @(
            "-m", "pip", "install", "llama-cpp-python",
            "--extra-index-url", $LlamaWheelIndex,
            "--prefer-binary"
        ) -What "llama-cpp-python (fertiges Wheel, CPU)"
    }

    if (-not $chatOk) {
        Write-Note "Kein fertiges Wheel gefunden - Versuch ueber PyPI (baut aus Quelltext)."
        $chatOk = Invoke-Optional -File $VenvPython -Arguments @(
            "-m", "pip", "install", "llama-cpp-python", "--prefer-binary"
        ) -What "llama-cpp-python (aus Quelltext)"
    }

    if ($chatOk) {
        $llamaInfo = Get-PythonLine -Python $VenvPython -Code "import llama_cpp,sys;sys.stdout.write(getattr(llama_cpp,'__version__','unbekannt'))"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "llama_cpp laesst sich nicht importieren: $llamaInfo"
            $chatOk = $false
        } else {
            Write-Note "llama-cpp-python im Bau-Venv: $llamaInfo"
        }
    }

    if (-not $chatOk) {
        $WithChat = $false
        Write-Warning ("Chat wird ohne Laufzeit gebaut. Die Anwendung startet und " +
                       "meldet den fehlenden Teil im Klartext. Zum Nachruesten " +
                       "'Visual Studio Build Tools' und CMake installieren, dann " +
                       "erneut bauen.")
    }
} else {
    Write-Note "Chat übersprungen. Nachrüsten mit -WithChat `$true"
}

if ($WithCuda) {
    # Gegenprobe: ein CPU-Wheel an dieser Stelle wäre ein stiller Fehlbau.
    # Ueber Get-PythonLine, damit eine Warnung beim torch-Import die Pruefung
    # nicht in ein Array verwandelt - '-match' auf einem Array verhaelt sich
    # anders und koennte einen CPU-Fehlbau durchlassen.
    $torchInfo = Get-PythonLine -Python $VenvPython -Code "import torch,sys;sys.stdout.write(f'{torch.__version__}|{torch.version.cuda}')"
    Write-Note "torch: $torchInfo"
    if ($torchInfo -match '\+cpu' -or $torchInfo -match '\|None') {
        throw ("CPU-Wheel von torch installiert ($torchInfo). Das Bundle würde trotz " +
               "CUDA-DLLs auf der CPU rechnen. Venv mit -Clean neu aufbauen oder " +
               "-CudaIndexUrl auf eine Fassung zeigen lassen, die dieses torch anbietet.")
    }
}

# ---------------------------------------------------------------------------
Write-Step "Modell vorladen"
if ($SkipModelDownload) {
    Write-Note "übersprungen (-SkipModelDownload)"
} elseif (-not $Model) {
    Write-Note "kein Modell angegeben"
} else {
    $StageRoot = Join-Path $BuildDir "stage-models"
    New-Item -ItemType Directory -Force $StageRoot | Out-Null
    foreach ($ModelKey in $Model) {
        if (-not $ModelKey) { continue }
        Invoke-Checked -File $VenvPython -Arguments @(
            "-m", "app", "--data-dir", $StageRoot, "models", "download", $ModelKey
        ) -What "Modell-Download ($ModelKey)"
    }
}

# ---------------------------------------------------------------------------
Write-Step "PyInstaller"

# Schutz der Nutzerdaten: Im Portable-Modus liegen Modelle, Konfiguration und
# Ausgaben in dist\<Name>\data – teils zweistellige GB. PyInstaller räumt den
# Ausgabeordner mit --noconfirm ab. Deshalb data\ vorher wegtragen und danach
# zurücklegen.
$DataStash = ""
$DataDir = Join-Path $Target "data"
if (Test-Path $DataDir) {
    $DataStash = Join-Path $BuildDir ("data-stash-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
    New-Item -ItemType Directory -Force (Split-Path $DataStash) | Out-Null
    Write-Note "sichere Nutzerdaten: $DataDir -> $DataStash"
    Move-Item -Path $DataDir -Destination $DataStash -Force
}

$env:SF_ROOT = $Root
$env:SF_NAME = $Name
$env:SF_ENTRY = "run_app.py"
$env:SF_NOGUI = if ($NoGui) { "1" } else { "0" }
$env:SF_WITHCUDA = if ($WithCuda) { "1" } else { "0" }
$env:SF_WITHONNX = if ($WithOnnx) { "1" } else { "0" }
$env:SF_WITHCHAT = if ($WithChat) { "1" } else { "0" }
$env:SF_CONSOLE = if ($Console) { "1" } else { "0" }

try {
    Invoke-Checked -File $VenvPython -Arguments @(
        "-m", "PyInstaller", "--noconfirm", "--clean",
        "--distpath", $DistDir, "--workpath", (Join-Path $BuildDir "pyi"),
        $SpecFile
    ) -What "PyInstaller"
} finally {
    foreach ($key in @("SF_ROOT", "SF_NAME", "SF_ENTRY", "SF_NOGUI", "SF_WITHCUDA", "SF_CONSOLE")) {
        Remove-Item "Env:\$key" -ErrorAction SilentlyContinue
    }
    # Nutzerdaten zurücklegen – auch wenn der Build fehlgeschlagen ist.
    if ($DataStash -and (Test-Path $DataStash)) {
        New-Item -ItemType Directory -Force $Target | Out-Null
        if (Test-Path $DataDir) {
            Write-Warning "Neues data\ vorhanden – gesicherte Daten liegen unter $DataStash"
        } else {
            Move-Item -Path $DataStash -Destination $DataDir -Force
            Write-Note "Nutzerdaten zurückgelegt: $DataDir"
        }
    }
}

if (-not (Test-Path $Target)) { throw "Ausgabeordner fehlt: $Target" }

# ---------------------------------------------------------------------------
Write-Step "CUDA-Laufzeit neben die .exe legen"
if ($WithCuda) {
    $CudaOut = Join-Path $Target "cuda"
    New-Item -ItemType Directory -Force $CudaOut | Out-Null
    $NvidiaRoot = Join-Path $VenvDir "Lib\site-packages\nvidia"
    $copied = 0
    if (Test-Path $NvidiaRoot) {
        Get-ChildItem -Path $NvidiaRoot -Recurse -Filter *.dll -ErrorAction SilentlyContinue |
            ForEach-Object {
                Copy-Item $_.FullName -Destination $CudaOut -Force
                $copied++
            }
    }
    $TorchLib = Join-Path $VenvDir "Lib\site-packages\torch\lib"
    if (Test-Path $TorchLib) {
        Get-ChildItem -Path $TorchLib -Filter "cu*.dll" -ErrorAction SilentlyContinue |
            ForEach-Object {
                Copy-Item $_.FullName -Destination $CudaOut -Force
                $copied++
            }
    }
    Write-Note "$copied DLL(s) nach $CudaOut kopiert"
    if ($copied -eq 0) {
        Write-Warning "Keine CUDA-DLLs gefunden – das Bundle läuft nur auf CPU/DirectML."
    }
} else {
    Write-Note "übersprungen"
}

# ---------------------------------------------------------------------------
Write-Step "Beilagen"
foreach ($file in @("THIRD-PARTY-NOTICES.md", "AGB.md", "MODELS.md", "README.md")) {
    $source = Join-Path $Root $file
    if (Test-Path $source) {
        Copy-Item $source -Destination $Target -Force
        Write-Note "kopiert: $file"
    } else {
        Write-Warning "$file fehlt – Abnahmekriterium 7 verlangt vollständige Hinweise."
    }
}

$FfmpegOut = Join-Path $Target "tools\ffmpeg"
if ($FfmpegDir) {
    # Eigener Build vorgegeben – wird übernommen und ebenfalls geprüft.
    if (-not (Test-Path $FfmpegDir)) { throw "FfmpegDir nicht gefunden: $FfmpegDir" }
    New-Item -ItemType Directory -Force $FfmpegOut | Out-Null
    Copy-Item (Join-Path $FfmpegDir "*") -Destination $FfmpegOut -Recurse -Force
    $own = Get-ChildItem -Path $FfmpegOut -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
    if ($own) {
        $cfg = & $own.FullName -hide_banner -version 2>&1 | Out-String
        if ($cfg -match "--enable-gpl" -or $cfg -match "--enable-nonfree") {
            throw ("Der unter -FfmpegDir angegebene Build enthält GPL- oder nonfree-Teile. " +
                   "Für die Auslieferung einen LGPL-Build verwenden.")
        }
        Write-Note "eigener ffmpeg-Build übernommen und geprüft"
    }
} elseif ($WithFfmpeg) {
    Install-Ffmpeg -Url $FfmpegUrl -Target $FfmpegOut -CacheDir (Join-Path $BuildDir "ffmpeg-dl")
} else {
    Write-Note "kein ffmpeg mitgeliefert – Video/Vertonung meldet das im Klartext."
}

# Modell mitgeben
$StageModels = Join-Path $BuildDir "stage-models\models"
if ((-not $SkipModelDownload) -and (Test-Path $StageModels)) {
    $DataModels = Join-Path $Target "data\models"
    New-Item -ItemType Directory -Force $DataModels | Out-Null
    Copy-Item (Join-Path $StageModels "*") -Destination $DataModels -Recurse -Force
    Write-Note "Modell(e) nach $DataModels kopiert"
}

# Klon-Laufzeit (Chatterbox) als getrennte Umgebung mitliefern.
# Bewusst nicht in der Hauptumgebung: chatterbox-tts verlangt torch 2.6 ohne
# CUDA-Build und ältere diffusers/transformers – zusammen installiert wäre
# die GPU-Beschleunigung für Bild und Video weg.
if ($WithVoiceRuntime) {
    $VoiceSrc = if ($VoiceRuntimeDir) { $VoiceRuntimeDir } else { Join-Path $Root ".voice-venv" }
    if (-not (Test-Path $VoiceSrc)) {
        throw ("Klon-Laufzeit nicht gefunden: $VoiceSrc. Erst anlegen mit " +
               "'python -m venv .voice-venv' und 'pip install chatterbox-tts', " +
               "oder -VoiceRuntimeDir angeben.")
    }
    $VoiceOut = Join-Path $Target "tools\voice-runtime"
    Write-Note "kopiere Klon-Laufzeit (mehrere GB) …"
    New-Item -ItemType Directory -Force $VoiceOut | Out-Null
    Copy-Item (Join-Path $VoiceSrc "*") -Destination $VoiceOut -Recurse -Force
    Copy-Item (Join-Path $Root "packaging\voice_worker.py") -Destination $VoiceOut -Force
    $voiceSize = (Get-ChildItem -Recurse $VoiceOut | Measure-Object -Property Length -Sum).Sum / 1GB
    Write-Note ("Klon-Laufzeit eingebettet: {0:N1} GB" -f $voiceSize)
} else {
    Write-Note "ohne Klon-Laufzeit – Klonstimmen fallen auf die Standardstimme zurück."
}

# Portable-Marker IMMER schreiben. PyInstaller räumt den Ausgabeordner ab;
# fehlt die Datei danach, sucht die Anwendung ihre Modelle plötzlich in
# %LOCALAPPDATA% und findet das mitgelieferte data\ nicht mehr.
Set-Content -Encoding utf8 -Path (Join-Path $Target "portable.txt") -Value @"
Portable-Modus: Modelle, Konfiguration und Ausgaben liegen im Unterordner data\.
Diese Datei löschen, damit stattdessen %LOCALAPPDATA%\StreamForge genutzt wird.
"@
Write-Note "portable.txt geschrieben"

# ---------------------------------------------------------------------------
Write-Step "Ergebnis"
$exePath = Join-Path $Target "$Name.exe"
if (-not (Test-Path $exePath)) { throw ".exe fehlt: $exePath" }

$size = (Get-ChildItem -Recurse $Target | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host ("Bundle:   {0}" -f $Target)
Write-Host ("Größe:    {0:N2} GB" -f $size)
Write-Host ("Start:    {0}" -f $exePath)
Write-Host ("Test:     {0} info" -f $exePath)
Write-Host ""
Write-Host "Abnahme vor der Auslieferung:" -ForegroundColor Yellow
Write-Host "  1. Rechner ohne Python und ohne NVIDIA-Treiber -> läuft auf CPU, Meldung erklärt warum"
Write-Host "  2. Rechner mit NVIDIA-GPU -> Backend-Anzeige zeigt CUDA ohne Zutun"
Write-Host "  3. Modell-Download mitten im Laden abbrechen -> keine .incomplete-Dateien"
Write-Host "  4. Zweiten Start versuchen -> 'läuft bereits' und Ende"
Write-Host "  5. Laufenden Auftrag abbrechen -> Prozess lebt weiter"
Write-Host "  6. ffmpeg oder Modell entfernen -> verständliche Meldung statt Stacktrace"
Write-Host "  7. THIRD-PARTY-NOTICES.md liegt im Auslieferungsordner"
