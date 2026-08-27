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
    Neu bauen: entfernt den PyInstaller-Arbeitsordner und das fertige
    Bundle, damit nichts aus dem letzten Lauf uebernommen wird.

    Was NICHT geloescht wird, weil es nur Download-Zeit kostet:
      dist\<Name>\data    Modelle, Konfiguration, Ausgaben
      .build-venv          die installierten Pakete (torch allein ~8 GB)
      build\stage-models   vorgeladene Modelle
      build\ffmpeg-dl      der ffmpeg-Download

    Fuer jedes davon gibt es einen eigenen Schalter, siehe unten.

.PARAMETER PurgeData
    Zusaetzlich die Nutzerdaten loeschen. Das entfernt auch alle
    heruntergeladenen Modelle - je nach Auswahl zweistellige GB, die neu
    geladen werden muessen.

.PARAMETER FreshVenv
    Das Bau-Venv neu aufsetzen. Noetig, wenn eine Abhaengigkeit
    durcheinander ist; kostet den vollen pip-Download.

.PARAMETER PurgeCache
    Die Zwischenspeicher leeren (vorgeladene Modelle, ffmpeg-Download).

.PARAMETER WithPiper
    Piper-Sprachausgabe mitliefern (Stimme Thorsten, schnell auf der CPU).
    Vorgabe AUS: piper-tts steht unter GPL-3.0 und bettet espeak-ng ein.
    Eine damit gebaute Fassung darf nicht weitergegeben oder verkauft
    werden. Fuer den privaten Betrieb ist das ohne Folgen.

.PARAMETER LlamaCudaWheel
    URL eines CUDA-Wheels fuer llama-cpp-python. Ohne das rechnet der Chat
    auf der CPU - gemessen rund zehnmal langsamer. Die URL wird in
    .llama-cuda-wheel.txt gemerkt und beim naechsten Bau von selbst
    genommen; einmal angeben genuegt.

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
    # Loescht ZUSAETZLICH die Nutzerdaten (Modelle, Konfiguration,
    # Ausgaben). Ohne diesen Schalter ueberleben sie auch ein -Clean.
    [switch]$PurgeData,
    # Setzt das Bau-Venv neu auf. Kostet den kompletten pip-Download
    # (torch mit CUDA allein rund 8 GB).
    [switch]$FreshVenv,
    # Wirft die Zwischenspeicher weg: vorgeladene Modelle und den
    # ffmpeg-Download. Beides wird danach neu geladen.
    [switch]$PurgeCache,
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
    # Paketindex mit den CUDA-Bauten von llama-cpp-python (Seite des
    # Autors) und die Fassung, die dort nachweislich laeuft.
    #
    # Gemessen auf einem i9-10850K (Comet Lake, kein AVX-512) mit einer
    # RTX 4070 Ti, gleiches Modell, gleicher Index:
    #
    #   0.3.35   Modell laden -> 0xc000001d, Absturz
    #   0.3.30   Modell in 1,6 s, Antwort in 0,2 s
    #
    # Beide melden llama_supports_gpu_offload() == True; der Unterschied
    # zeigt sich erst beim LADEN. 0.3.35 nutzt Befehlssaetze, die aeltere
    # CPUs nicht kennen.
    #
    # Deshalb die Fassung festnageln statt "neueste". Der Bau prueft
    # danach mit einem echten Modell nach und faellt sonst auf den
    # CPU-Bau zurueck.
    [string]$LlamaCudaIndex = "https://abetlen.github.io/llama-cpp-python/whl/cu124",
    [string]$LlamaCudaVersion = "0.3.30",
    # Discord-Bot fuers Telefonieren im Sprachkanal.
    [bool]$WithDiscord = $true,
    # Piper-Sprachausgabe (Stimme Thorsten). Vorgabe AUS: piper-tts steht
    # unter GPL-3.0. Fuer den privaten Betrieb unproblematisch, fuer eine
    # Weitergabe nicht - deshalb muss man es ausdruecklich verlangen.
    [switch]$WithPiper,
    # Vorgabe an: '-Clean' allein soll ein vollständiges Programm bauen.
    # Auf Rechnern ohne AMD-/Intel-Grafik kostet das nur Platz, nichts sonst.
    [bool]$WithOnnx = $true,
    # Chat/Code-Writer über llama.cpp (GGUF). Ebenfalls Vorgabe an.
    [bool]$WithChat = $true,
    # Telefonieren: Spracherkennung (faster-whisper) und Mikrofon/Ton
    # (sounddevice). Ebenfalls Vorgabe an.
    [bool]$WithCall = $true,
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
# Liegengebliebenes Zwischenlager melden statt still zu uebergehen.
#
# Wird ein Lauf zwischen "wegtragen" und "zurueckgeben" abgebrochen (Strg+C,
# Stromausfall), stehen die Modelle noch da - nur am falschen Ort. Ohne diesen
# Hinweis sucht man sie vergeblich und laedt zweistellige GB neu.
$verwaist = @(Get-ChildItem -Path $Root -Filter ".data-stash-*" -Directory -Force -ErrorAction SilentlyContinue)
foreach ($rest in $verwaist) {
    Write-Warning ("Nutzerdaten aus einem abgebrochenen Lauf liegen unter {0}. " -f $rest.FullName +
                   "Nach dem Bau nach dist\$Name\data zurueckschieben oder loeschen.")
}

function Move-Tree {
    <#
    .SYNOPSIS
        Ein Verzeichnis am Stueck verschieben.
    .DESCRIPTION
        Move-Item arbeitet bei Verzeichnissen rekursiv und bricht an
        Pfaden jenseits von 260 Zeichen MITTEN im Umbau ab. Die
        Klon-Laufzeit bringt solche Pfade mit
        (onnx/backend/test/data/node/...). Zurueck blieb ein auf zwei
        Orte zerrissener Datenbestand - die halbe Konfiguration im
        Stash, die Modelle noch am alten Platz.

        [System.IO.Directory]::Move benennt nur um: ein Vorgang auf
        Dateisystemebene, unteilbar, ohne Pfadlaengenproblem und ohne
        Zeitverlust bei 23 GB. Nur wenn das nicht geht - etwa ueber
        Laufwerksgrenzen - wird auf robocopy /MOVE ausgewichen.

        Scheitert beides, wird geworfen: lieber ein abgebrochener Bau
        als geloeschte Modelle und Stimmen.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Quelle,
        [Parameter(Mandatory = $true)][string]$Ziel
    )
    try {
        [System.IO.Directory]::Move($Quelle, $Ziel)
        return
    } catch {
        Write-Note "Umbenennen ging nicht ($($_.Exception.Message)) - kopiere"
    }
    & robocopy $Quelle $Ziel /E /MOVE /NFL /NDL /NJH /NJS /NP /R:1 /W:1 | Out-Null
    if ($LASTEXITCODE -ge 8) {
        throw "Verschieben fehlgeschlagen (robocopy $LASTEXITCODE): $Quelle -> $Ziel"
    }
}

function Merge-Tree {
    <#
    .SYNOPSIS
        Einen gesicherten Baum in einen vorhandenen zurueckfuehren.
    .DESCRIPTION
        Die Quelle gewinnt bei Namensgleichheit, alles im Ziel, was die
        Quelle nicht kennt, bleibt stehen. Danach wird die Quelle
        entfernt.

        robocopy statt Move-Item/Copy-Item: die Klon-Laufzeit enthaelt
        Pfade jenseits von 260 Zeichen (onnx/backend/test/data/node/...),
        an denen beide mit WinError 145 scheitern.

        robocopy meldet Erfolg mit Exit-Codes 0-7; alles ab 8 ist ein
        echter Fehler.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Quelle,
        [Parameter(Mandatory = $true)][string]$Ziel,
        [string]$Was = "Daten"
    )
    & robocopy $Quelle $Ziel /E /NFL /NDL /NJH /NJS /NP /R:1 /W:1 | Out-Null
    if ($LASTEXITCODE -ge 8) {
        throw "$Was liessen sich nicht zurueckfuehren (robocopy $LASTEXITCODE). Sie liegen weiter unter $Quelle."
    }
    Remove-Tree $Quelle
    Write-Note "$Was zusammengeführt: $Ziel"
}

function Get-DirSize {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return 0 }
    try {
        $summe = (Get-ChildItem -Recurse -File $Path -ErrorAction SilentlyContinue |
                  Measure-Object -Property Length -Sum).Sum
        return [double]($summe / 1GB)
    } catch { return 0 }
}

function Remove-Tree {
    param([string]$Path, [string]$What)
    if (-not (Test-Path $Path)) { return }
    $gb = Get-DirSize $Path
    if ($gb -ge 0.1) {
        Write-Note ("entferne {0} ({1:N1} GB): {2}" -f $What, $gb, $Path)
    } else {
        Write-Note "entferne ${What}: $Path"
    }
    try {
        Remove-Item -Recurse -Force $Path -ErrorAction Stop
    } catch {
        # Tiefe Pfade sprengen die 260-Zeichen-Grenze, und Remove-Item
        # bricht mitten drin ab ("Das Verzeichnis ist nicht leer").
        # robocopy gegen ein leeres Verzeichnis zu spiegeln ist der
        # uebliche Weg dafuer - es kommt mit langen Pfaden zurecht.
        Write-Note "tiefe Pfade - zweiter Versuch ueber robocopy"
        $leer = Join-Path $env:TEMP ("sf-leer-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
        New-Item -ItemType Directory -Force $leer | Out-Null
        try {
            robocopy $leer $Path /MIR /NFL /NDL /NJH /NJS /NP /R:1 /W:1 | Out-Null
            Remove-Item -Recurse -Force $Path -ErrorAction Stop
        } finally {
            Remove-Item -Recurse -Force $leer -ErrorAction SilentlyContinue
        }
    }
}

# Aufraeumen heisst: Bauartefakte weg, Geladenes bleibt.
#
# '-Clean' hiess frueher "build, dist und Venv loeschen". Das klingt harmlos,
# vernichtete aber drei Sammlungen, die nur Zeit kosten: die pip-Pakete im
# Venv (torch mit CUDA allein rund 8 GB), den Modell-Zwischenspeicher unter
# build\stage-models (SDXL 6,6 GB) und den ffmpeg-Download. Ein "nur neu
# bauen" wurde damit zu ueber 15 GB Download. Jetzt wird nur entfernt, was
# der neue Bau ohnehin neu erzeugt; fuer alles andere gibt es je einen
# eigenen Schalter.
$PyiWork = Join-Path $BuildDir "pyi"
$StageCache = Join-Path $BuildDir "stage-models"
$FfmpegCache = Join-Path $BuildDir "ffmpeg-dl"

# Verwaiste Sicherungen aus einem abgebrochenen Lauf einsammeln.
#
# Bricht ein Bau zwischen "data zur Seite" und "data zurueck" ab, blieben
# die Daten frueher in einem .data-stash-XXXX liegen - und die Anwendung
# startete mit leerem Datenordner. Fuer den Bediener sieht das aus, als
# seien Modelle und Einstellungen geloescht worden.
#
# Gesucht wird an BEIDEN Stellen, an denen der Bau Daten wegtraegt:
# -Clean legt sie unter <Repo>\.data-stash-XXXX ab, der
# PyInstaller-Schritt unter <Repo>\build\data-stash-XXXX. Frueher war
# nur die erste bekannt - und die zweite ist genau die, an der es
# schiefging: nach einem abgebrochenen Bau lagen dort 23,8 GB Modelle,
# ein angelerntes Stimmprofil und secrets.json.
$StashOrte = @($Root, (Join-Path $Root "build")) | Where-Object { Test-Path $_ }
foreach ($paar in @(
    @{ Muster = ".data-stash-*"; Unter = "data"; Was = "Nutzerdaten" },
    @{ Muster = "data-stash-*"; Unter = "data"; Was = "Nutzerdaten" },
    @{ Muster = ".tools-stash-*"; Unter = "tools"; Was = "Werkzeuge" },
    @{ Muster = "tools-stash-*"; Unter = "tools"; Was = "Werkzeuge" }
)) {
foreach ($alt in @($StashOrte | ForEach-Object {
    Get-ChildItem -Path $_ -Directory -Filter $paar.Muster -EA SilentlyContinue
})) {
    $ziel = Join-Path $Target $paar.Unter
    if (Test-Path $ziel) {
        Write-Warning ("Verwaiste Sicherung gefunden: {0}. Der Datenordner ist bereits da - " +
                       "bitte von Hand vergleichen, es wird nichts ueberschrieben." -f $alt.FullName)
    } else {
        Write-Note "Verwaiste Sicherung wird zurueckgelegt: $($alt.FullName)"
        $null = New-Item -ItemType Directory -Force -Path $Target
        Move-Tree -Quelle $alt.FullName -Ziel $ziel
    }
}
}

$CleanStash = ""
if ($Clean) {
    # Nutzerdaten aus dem fertigen Bundle in Sicherheit bringen, bevor das
    # Bundle faellt.
    $DataDirVorher = Join-Path $Target "data"

    # Laeuft die Anwendung noch? Sie haelt ihren Datenordner offen, und
    # Move-Item scheitert dann MITTEN im Umbau - mit halb geleertem
    # Bundle und gestrandeten Daten.
    $laeuft = @(Get-Process -Name $Name -ErrorAction SilentlyContinue)
    if ($laeuft.Count -gt 0) {
        throw ("$Name laeuft noch (PID $($laeuft[0].Id)). Bitte erst beenden - " +
               "ein laufendes Programm haelt seinen Datenordner, und der Bau " +
               "bricht sonst mitten im Umbau ab.")
    }

    if ((Test-Path $DataDirVorher) -and (-not $PurgeData)) {
        $CleanStash = Join-Path $Root (".data-stash-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
        Write-Note ("sichere Nutzerdaten ({0:N1} GB): {1}" -f (Get-DirSize $DataDirVorher), $DataDirVorher)
        Move-Tree -Quelle $DataDirVorher -Ziel $CleanStash
    } elseif ($PurgeData -and (Test-Path $DataDirVorher)) {
        Write-Warning "-PurgeData: Nutzerdaten und Modelle werden geloescht."
    }

    # tools\ ebenso in Sicherheit bringen.
    #
    # Darin liegen ffmpeg und die 5 GB Klon-Laufzeit. Deren
    # onnx-Testdaten haben so tiefe Pfade, dass Remove-Item beim Loeschen
    # von dist\ abbricht - genau daran ist -Clean gescheitert. Und
    # neu kopieren muesste man sie ohnehin nicht: nichts davon stammt
    # von PyInstaller.
    $CleanToolsStash = ""
    $ToolsDirVorher = Join-Path $Target "tools"
    if ((Test-Path $ToolsDirVorher) -and (-not $PurgeData)) {
        $CleanToolsStash = Join-Path $Root (".tools-stash-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
        Write-Note ("sichere Werkzeuge ({0:N1} GB): {1}" -f (Get-DirSize $ToolsDirVorher), $ToolsDirVorher)
        Move-Tree -Quelle $ToolsDirVorher -Ziel $CleanToolsStash
    }

    # Ab hier MUSS zurueckgelegt werden, egal was dazwischen passiert.
    try {
        Remove-Tree $PyiWork "PyInstaller-Arbeitsordner"
        Remove-Tree $DistDir "fertiges Bundle"
    } finally {
        if ($CleanStash -and (Test-Path $CleanStash)) {
            $null = New-Item -ItemType Directory -Force -Path $Target
            Move-Tree -Quelle $CleanStash -Ziel (Join-Path $Target "data")
            Write-Note "Nutzerdaten zurueckgelegt."
            $CleanStash = ""
        }
        if ($CleanToolsStash -and (Test-Path $CleanToolsStash)) {
            $null = New-Item -ItemType Directory -Force -Path $Target
            Move-Tree -Quelle $CleanToolsStash -Ziel (Join-Path $Target "tools")
            Write-Note "Werkzeuge zurueckgelegt."
            $CleanToolsStash = ""
        }
    }

    if (-not $FreshVenv) {
        Write-Note "Venv bleibt bestehen (-FreshVenv setzt es neu auf)."
    }
    if ((-not $PurgeCache) -and ((Test-Path $StageCache) -or (Test-Path $FfmpegCache))) {
        Write-Note "Zwischenspeicher bleiben bestehen (-PurgeCache leert sie)."
    }
}

if ($FreshVenv) {
    Remove-Tree $VenvDir "Bau-Venv"
}
if ($PurgeCache) {
    Remove-Tree $StageCache "Modell-Zwischenspeicher"
    Remove-Tree $FfmpegCache "ffmpeg-Zwischenspeicher"
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

    # Zuletzt benutztes CUDA-Wheel wiederverwenden.
    #
    # Ohne das laeuft der Chat auf der CPU, sobald jemand den Schalter
    # vergisst - und der Unterschied ist rund das Zehnfache an
    # Antwortzeit.
    #
    # Frueher stand hier, die offiziellen CUDA-Indizes fuehrten fuer
    # Python 3.13 keine Wheels. Das war ein Trugschluss: gesucht wurde
    # nach "cp313", das Wheel traegt aber gar keine Python-Bindung.
    # Nachgemessener Trockenlauf gegen den cu124-Index:
    #   llama_cpp_python-0.3.35-py3-none-win_amd64.whl   482,7 MB
    # "py3-none" laeuft auf 3.13, und 482 MB statt rund 10 MB ist der Bau
    # mit CUDA. Deshalb wird der Index jetzt von selbst genommen.
    $LlamaMerker = Join-Path $Root ".llama-cuda-wheel.txt"
    if ((-not $LlamaCudaWheel) -and (Test-Path $LlamaMerker)) {
        $gemerkt = (Get-Content $LlamaMerker -Raw -ErrorAction SilentlyContinue).Trim()
        if ($gemerkt) {
            $LlamaCudaWheel = $gemerkt
            Write-Note "CUDA-Wheel aus dem letzten Bau uebernommen."
            Write-Note "  (loeschen: $LlamaMerker)"
        }
    }
    if ($LlamaCudaWheel) {
        try {
            Set-Content -Path $LlamaMerker -Value $LlamaCudaWheel -Encoding UTF8
        } catch {
            Write-Note "Wheel-URL nicht merkbar: $($_.Exception.Message)"
        }
    }
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
                # cublasLt ist eine EIGENE DLL und wird von cublas gebraucht. Fehlt
                # sie, scheitert schon der Import von llama_cpp - und PyInstaller
                # packt dann stillschweigend gar nichts ein.
                foreach ($muster in @("nvidia\*\bin\cudart64_*.dll", "nvidia\*\bin\cublas64_*.dll",
                                      "nvidia\*\bin\cublasLt64_*.dll",
                                      "torch\lib\cudart64_*.dll", "torch\lib\cublas64_*.dll",
                                      "torch\lib\cublasLt64_*.dll")) {
                    $pfad = Join-Path $sitePkgs $muster
                    foreach ($dll in @(Get-ChildItem -Path $pfad -ErrorAction SilentlyContinue)) {
                        Copy-Item -LiteralPath $dll.FullName -Destination "$libDir" -Force -ErrorAction SilentlyContinue
                        $kopiert++
                    }
                }
                Write-Note "CUDA-Laufzeit neben ggml gelegt: $kopiert Datei(en)"
            }
            # ERST pruefen, ob das Paket ueberhaupt fuer sich allein
            # importierbar ist. pipeline_chat legt vorher die DLL-Pfade
            # zurecht und verdeckt damit genau den Fehler, der spaeter im
            # Bau zuschlaegt: PyInstaller importiert OHNE diese Hilfe, und
            # ein fehlgeschlagener Import liefert dort 0 Module - ohne
            # Fehlermeldung, aber mit leerem Buendel.
            $roh = Get-PythonLine -Python $VenvPython -Code "import llama_cpp,sys;sys.stdout.write('OK')"
            if ($roh -notmatch "OK") {
                Write-Warning ("llama_cpp ist NICHT fuer sich allein importierbar. " +
                               "PyInstaller wuerde dann nichts einpacken und das " +
                               "fertige Programm meldet 'llama_cpp nicht ladbar'. " +
                               "Fehlen cudart64/cublas64/cublasLt64 neben llama.dll?")
            }

            # Ein Import beweist NICHTS.
            #
            # Gemessen: ein CUDA-Wheel importierte sauber, meldete die
            # Grafikkarte - und starb beim Laden eines Modells mit
            # 0xc000001d (STATUS_ILLEGAL_INSTRUCTION), weil es
            # Befehlssaetze nutzt, die diese CPU nicht kennt. Die
            # betroffene Rechenschleife laeuft erst beim Laden.
            # Deshalb hier ein echter Ladeversuch, wenn ein Modell da ist.
            $probeCode = @'
import glob, os, sys
wurzel = os.environ.get("SF_PROBE_MODELS", "")
alle = glob.glob(os.path.join(wurzel, "**", "*.gguf"), recursive=True) if wurzel else []
# mmproj ist der BILDTEIL, kein Modell - als Modell geladen gibt es
# einen ValueError, und der sah wie ein Absturz des Wheels aus.
kand = [k for k in alle if "mmproj" not in os.path.basename(k).lower()]
if not kand:
    sys.stdout.write("KEINMODELL"); raise SystemExit(0)
try:
    from llama_cpp import Llama
    Llama(model_path=min(kand, key=os.path.getsize), n_ctx=256, n_gpu_layers=99, verbose=False)
    sys.stdout.write("LAEDT")
except BaseException as exc:
    sys.stdout.write("CRASH:" + type(exc).__name__)
'@
            $probeDatei = Join-Path $env:TEMP "sf-llama-probe.py"
            Set-Content -Path $probeDatei -Value $probeCode -Encoding UTF8
            $modellOrdner = Join-Path $Target "data\models\chat"
            if (Test-Path $modellOrdner) {
                $env:SF_PROBE_MODELS = $modellOrdner
                $probe = Get-PythonLine -Python $VenvPython -Code "exec(open(r'$probeDatei',encoding='utf-8').read())"
                Remove-Item Env:\SF_PROBE_MODELS -ErrorAction SilentlyContinue
                # Leere Antwort zaehlt als Fehlschlag: bei einem harten
                # Abbruch (STATUS_ILLEGAL_INSTRUCTION) stirbt der Prozess,
                # ohne dass Python noch etwas schreiben koennte.
                if (($probe -match "CRASH") -or ([string]::IsNullOrWhiteSpace($probe))) {
                    Write-Warning ("Das CUDA-Wheel laedt KEIN Modell ($probe). Auf dieser CPU " +
                                   "fehlen die noetigen Befehlssaetze. Es wird der CPU-Bau genommen.")
                    $chatOk = Invoke-Optional -File $VenvPython -Arguments @(
                        "-m", "pip", "install", "llama-cpp-python",
                        "--extra-index-url", $LlamaWheelIndex,
                        "--prefer-binary", "--force-reinstall", "--no-deps"
                    ) -What "llama-cpp-python (Rueckfall auf CPU)"
                } elseif ($probe -match "LAEDT") {
                    Write-Note "Ladeversuch mit einem echten Modell erfolgreich."
                }
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
    } elseif ($WithCuda -and $LlamaCudaIndex) {
        # Kein festes Wheel angegeben, aber CUDA gewuenscht: den Index des
        # Autors nehmen. Schlaegt das fehl, bleibt der CPU-Weg darunter.
        $paket = if ($LlamaCudaVersion) { "llama-cpp-python==$LlamaCudaVersion" }
                 else { "llama-cpp-python" }
        Write-Note "CUDA-Bau: $paket ueber $LlamaCudaIndex"
        $chatOk = Invoke-Optional -File $VenvPython -Arguments @(
            "-m", "pip", "install", $paket,
            "--extra-index-url", $LlamaCudaIndex,
            "--prefer-binary", "--force-reinstall", "--no-deps"
        ) -What "llama-cpp-python (CUDA ueber Index)"
        if ($chatOk) {
            # Dieselbe DLL-Beistellung wie beim festen Wheel: ggml-cuda.dll
            # ist gegen cudart/cublas gelinkt und wird ohne die still
            # uebersprungen.
            $libDir = Get-PythonLine -Python $VenvPython -Code "import llama_cpp,os,sys;sys.stdout.write(os.path.join(os.path.dirname(llama_cpp.__file__),'lib'))"
            $sitePkgs = Get-PythonLine -Python $VenvPython -Code "import sysconfig,sys;sys.stdout.write(sysconfig.get_paths()['purelib'])"
            if ($libDir -and (Test-Path $libDir)) {
                $kopiert = 0
                # cublasLt ist eine EIGENE DLL und wird von cublas gebraucht. Fehlt
                # sie, scheitert schon der Import von llama_cpp - und PyInstaller
                # packt dann stillschweigend gar nichts ein.
                foreach ($muster in @("nvidia\*\bin\cudart64_*.dll", "nvidia\*\bin\cublas64_*.dll",
                                      "nvidia\*\bin\cublasLt64_*.dll",
                                      "torch\lib\cudart64_*.dll", "torch\lib\cublas64_*.dll",
                                      "torch\lib\cublasLt64_*.dll")) {
                    $pfad = Join-Path $sitePkgs $muster
                    foreach ($dll in @(Get-ChildItem -Path $pfad -ErrorAction SilentlyContinue)) {
                        Copy-Item -LiteralPath $dll.FullName -Destination "$libDir" -Force -ErrorAction SilentlyContinue
                        $kopiert++
                    }
                }
                Write-Note "CUDA-Laufzeit neben ggml gelegt: $kopiert Datei(en)"
            }
            # ERST pruefen, ob das Paket ueberhaupt fuer sich allein
            # importierbar ist. pipeline_chat legt vorher die DLL-Pfade
            # zurecht und verdeckt damit genau den Fehler, der spaeter im
            # Bau zuschlaegt: PyInstaller importiert OHNE diese Hilfe, und
            # ein fehlgeschlagener Import liefert dort 0 Module - ohne
            # Fehlermeldung, aber mit leerem Buendel.
            $roh = Get-PythonLine -Python $VenvPython -Code "import llama_cpp,sys;sys.stdout.write('OK')"
            if ($roh -notmatch "OK") {
                Write-Warning ("llama_cpp ist NICHT fuer sich allein importierbar. " +
                               "PyInstaller wuerde dann nichts einpacken und das " +
                               "fertige Programm meldet 'llama_cpp nicht ladbar'. " +
                               "Fehlen cudart64/cublas64/cublasLt64 neben llama.dll?")
            }

            # Ein Import beweist NICHTS.
            #
            # Gemessen: ein CUDA-Wheel importierte sauber, meldete die
            # Grafikkarte - und starb beim Laden eines Modells mit
            # 0xc000001d (STATUS_ILLEGAL_INSTRUCTION), weil es
            # Befehlssaetze nutzt, die diese CPU nicht kennt. Die
            # betroffene Rechenschleife laeuft erst beim Laden.
            # Deshalb hier ein echter Ladeversuch, wenn ein Modell da ist.
            $probeCode = @'
import glob, os, sys
wurzel = os.environ.get("SF_PROBE_MODELS", "")
alle = glob.glob(os.path.join(wurzel, "**", "*.gguf"), recursive=True) if wurzel else []
# mmproj ist der BILDTEIL, kein Modell - als Modell geladen gibt es
# einen ValueError, und der sah wie ein Absturz des Wheels aus.
kand = [k for k in alle if "mmproj" not in os.path.basename(k).lower()]
if not kand:
    sys.stdout.write("KEINMODELL"); raise SystemExit(0)
try:
    from llama_cpp import Llama
    Llama(model_path=min(kand, key=os.path.getsize), n_ctx=256, n_gpu_layers=99, verbose=False)
    sys.stdout.write("LAEDT")
except BaseException as exc:
    sys.stdout.write("CRASH:" + type(exc).__name__)
'@
            $probeDatei = Join-Path $env:TEMP "sf-llama-probe.py"
            Set-Content -Path $probeDatei -Value $probeCode -Encoding UTF8
            $modellOrdner = Join-Path $Target "data\models\chat"
            if (Test-Path $modellOrdner) {
                $env:SF_PROBE_MODELS = $modellOrdner
                $probe = Get-PythonLine -Python $VenvPython -Code "exec(open(r'$probeDatei',encoding='utf-8').read())"
                Remove-Item Env:\SF_PROBE_MODELS -ErrorAction SilentlyContinue
                # Leere Antwort zaehlt als Fehlschlag: bei einem harten
                # Abbruch (STATUS_ILLEGAL_INSTRUCTION) stirbt der Prozess,
                # ohne dass Python noch etwas schreiben koennte.
                if (($probe -match "CRASH") -or ([string]::IsNullOrWhiteSpace($probe))) {
                    Write-Warning ("Das CUDA-Wheel laedt KEIN Modell ($probe). Auf dieser CPU " +
                                   "fehlen die noetigen Befehlssaetze. Es wird der CPU-Bau genommen.")
                    $chatOk = Invoke-Optional -File $VenvPython -Arguments @(
                        "-m", "pip", "install", "llama-cpp-python",
                        "--extra-index-url", $LlamaWheelIndex,
                        "--prefer-binary", "--force-reinstall", "--no-deps"
                    ) -What "llama-cpp-python (Rueckfall auf CPU)"
                } elseif ($probe -match "LAEDT") {
                    Write-Note "Ladeversuch mit einem echten Modell erfolgreich."
                }
            }
            $gpuInfo = Get-PythonLine -Python $VenvPython -Code "import sys;sys.path.insert(0,'.');from app import pipeline_chat;sys.stdout.write(str(pipeline_chat.gpu_offload_possible()))"
            if ($gpuInfo -notmatch "True") {
                Write-Warning ("Der CUDA-Bau meldet KEINE GPU-Unterstuetzung ($gpuInfo). " +
                               "Der Chat wuerde auf der CPU rechnen.")
            } else {
                Write-Note "llama.cpp mit GPU-Unterstuetzung (ueber Index)."
            }
        } else {
            Write-Note "CUDA-Index ohne Erfolg - es wird der CPU-Bau genommen."
            $chatOk = Invoke-Optional -File $VenvPython -Arguments @(
                "-m", "pip", "install", "llama-cpp-python",
                "--extra-index-url", $LlamaWheelIndex,
                "--prefer-binary"
            ) -What "llama-cpp-python (fertiges Wheel, CPU)"
        }
    } else {
        $chatOk = Invoke-Optional -File $VenvPython -Arguments @(
            "-m", "pip", "install", "llama-cpp-python",
            "--extra-index-url", $LlamaWheelIndex,
            "--prefer-binary"
        ) -What "llama-cpp-python (fertiges Wheel, CPU)"

        # Karte da, aber Chat auf der CPU: das ist fast immer ein
        # vergessener Schalter, kein Wunsch.
        if ($WithCuda) {
            Write-Warning ("Der Chat wurde OHNE GPU gebaut und rechnet auf der CPU " +
                           "(gemessen rund zehnmal langsamer). Fuer die Grafikkarte " +
                           "einmalig ein CUDA-Wheel angeben:")
            Write-Note "  .\build-windows.ps1 -Clean -LlamaCudaWheel `"<URL zum cp313-CUDA-Wheel>`""
            Write-Note "  Die URL wird gemerkt und beim naechsten Bau von selbst genommen."
        }
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

# Telefonieren: Spracherkennung und Audio-Ein/Ausgabe. Beides optional -
# ohne sie startet die Anwendung und meldet den fehlenden Teil im Klartext.
if ($WithCall) {
    Write-Step "Telefon-Laufzeit installieren (Spracherkennung, Audio)"
    $callOk = Invoke-Optional -File $VenvPython -Arguments @(
        "-m", "pip", "install", "faster-whisper", "sounddevice"
    ) -What "faster-whisper, sounddevice"
    if ($callOk) {
        $sttInfo = Get-PythonLine -Python $VenvPython -Code "import faster_whisper,sys;sys.stdout.write(getattr(faster_whisper,'__version__','?'))"
        Write-Note "faster-whisper im Bau-Venv: $sttInfo"
    } else {
        $WithCall = $false
        Write-Warning "Telefonieren wird ohne Laufzeit gebaut. Chat und Bild laufen trotzdem."
    }
} else {
    Write-Note "Telefonieren übersprungen. Nachrüsten mit -WithCall `$true"
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

# Piper: schnelle deutsche Sprachausgabe (Stimme Thorsten).
#
# ACHTUNG GPL-3.0. piper-tts bettet espeak-ng ein; wird es in denselben
# Prozess geladen, erfasst die GPL nach verbreiteter Auslegung die ganze
# Anwendung. Fuer den privaten Betrieb ohne Weitergabe ist das ohne
# Folgen - eine gebaute Fassung mit Piper darf aber NICHT weitergegeben
# oder verkauft werden. Deshalb Vorgabe aus und ein Warnhinweis.
if ($WithPiper) {
    Write-Step "Piper-Sprachausgabe installieren (GPL-3.0)"
    Write-Warning ("piper-tts steht unter GPL-3.0 und bettet espeak-ng ein. " +
                   "Diese gebaute Fassung darf damit NICHT weitergegeben oder " +
                   "verkauft werden.")
    $piperOk = Invoke-Optional -File $VenvPython -Arguments @(
        "-m", "pip", "install", "piper-tts", "--prefer-binary"
    ) -What "piper-tts (GPL-3.0)"
    if ($piperOk) {
        $pv = Get-PythonLine -Python $VenvPython -Code "import piper,sys;sys.stdout.write('vorhanden')"
        Write-Note "piper: $pv"
        Write-Note "Stimme laden: streamforge models download piper"
    }
}

# Discord: der Bot fuer Gespraeche im Sprachkanal.
#
# Was mitkommt: discord.py (MIT), PyNaCl (Apache-2.0), davey (MIT),
# audioop-lts (PSF-2.0) und die libopus-DLL (BSD-3-Clause), die discord.py
# im Wheel mitbringt. Alles kommerziell unbedenklich, Namensnennung
# gehoert in THIRD-PARTY-NOTICES.md.
if ($WithDiscord) {
    Write-Step "Discord-Laufzeit installieren"
    $discordOk = Invoke-Optional -File $VenvPython -Arguments @(
        "-m", "pip", "install", "discord.py[voice]", "discord-ext-voice-recv",
        "--prefer-binary"
    ) -What "discord.py mit Sprachunterstuetzung"
    if ($discordOk) {
        $dver = Get-PythonLine -Python $VenvPython -Code "import discord,sys;sys.stdout.write(discord.__version__)"
        Write-Note "discord.py im Bau-Venv: $dver"
        # libopus muss ladbar sein, sonst bleibt der Bot stumm.
        $opus = Get-PythonLine -Python $VenvPython -Code "import discord,sys;sys.stdout.write(str(discord.opus._load_default()))"
        if ($opus -match "True") {
            Write-Note "libopus ladbar."
        } else {
            Write-Warning "libopus laedt nicht ($opus). Der Bot koennte stumm bleiben."
        }
        # davey loest die Ende-zu-Ende-Verschluesselung der Sprachkanaele.
        # Ohne davey spricht der Bot zwar, hoert aber nichts - und zwar
        # ohne Fehlermeldung. Deshalb hier pruefen, nicht erst im Gespraech.
        $dave = Get-PythonLine -Python $VenvPython -Code "import davey,sys;sys.stdout.write('v%s' % davey.DAVE_PROTOCOL_VERSION if hasattr(davey.DaveSession,'decrypt') else 'ohne-decrypt')"
        if ($dave -match "^v\d") {
            Write-Note "davey ladbar ($dave) - Bot hoert in normalen Sprachkanaelen mit."
        } else {
            Write-Warning "davey fehlt oder kann nicht entschluesseln ($dave). Der Bot bliebe taub."
        }
    } else {
        Write-Note "Ohne Discord - der Telefon-Bot meldet das im Klartext."
    }
}

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
    Move-Tree -Quelle $DataDir -Ziel $DataStash
}

# tools\ ebenso. PyInstaller raeumt sein Ausgabeverzeichnis leer, und die
# Klon-Laufzeit darin sind rund 5 GB mit sehr tiefen Pfaden
# (onnx/backend/test/data/node/...). Das Loeschen scheitert dort mit
# "WinError 145: Das Verzeichnis ist nicht leer" und bricht den ganzen
# Bau ab.
#
# Nebenbei erspart das die 5 GB, die sonst bei JEDEM Bau neu kopiert
# wurden - nichts davon stammt von PyInstaller.
$ToolsStash = ""
$ToolsDir = Join-Path $Target "tools"
if (Test-Path $ToolsDir) {
    $ToolsStash = Join-Path $BuildDir ("tools-stash-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
    New-Item -ItemType Directory -Force (Split-Path $ToolsStash) | Out-Null
    Write-Note ("sichere Werkzeuge ({0:N1} GB): $ToolsDir" -f (Get-DirSize $ToolsDir))
    Move-Tree -Quelle $ToolsDir -Ziel $ToolsStash
}

$env:SF_ROOT = $Root
$env:SF_NAME = $Name
$env:SF_ENTRY = "run_app.py"
$env:SF_NOGUI = if ($NoGui) { "1" } else { "0" }
$env:SF_WITHCUDA = if ($WithCuda) { "1" } else { "0" }
$env:SF_WITHONNX = if ($WithOnnx) { "1" } else { "0" }
$env:SF_WITHCHAT = if ($WithChat) { "1" } else { "0" }
$env:SF_WITHCALL = if ($WithCall) { "1" } else { "0" }
$env:SF_WITHDISCORD = if ($WithDiscord) { "1" } else { "0" }
$env:SF_WITHPIPER = if ($WithPiper) { "1" } else { "0" }
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
            # PyInstaller hat selbst ein data\ angelegt - die vorgeladenen
            # Modelle als Erstausstattung. Frueher wurde hier nur gewarnt
            # und der Stash blieb liegen: 23,8 GB Modelle, das angelernte
            # Stimmprofil und secrets.json in einem Ordner unter build\,
            # den niemand sucht. Also zusammenfuehren.
            #
            # Die gesicherten Nutzerdaten gewinnen bei Namensgleichheit;
            # was nur in der Erstausstattung steckt, bleibt erhalten.
            Write-Note "führe gesicherte Nutzerdaten mit der Erstausstattung zusammen"
            Merge-Tree -Quelle $DataStash -Ziel $DataDir -Was "Nutzerdaten"
        } else {
            Move-Tree -Quelle $DataStash -Ziel $DataDir
            Write-Note "Nutzerdaten zurückgelegt: $DataDir"
        }
    }
    # Werkzeuge ebenso – auch nach einem Fehlschlag. Ohne das läge die
    # Klon-Laufzeit in build\ und die Anwendung fände sie nicht mehr.
    if ($ToolsStash -and (Test-Path $ToolsStash)) {
        New-Item -ItemType Directory -Force $Target | Out-Null
        if (Test-Path $ToolsDir) {
            # Wie bei den Nutzerdaten: zusammenfuehren statt liegenlassen.
            # Bleibt die Klon-Laufzeit im Stash, findet die Anwendung sie
            # nicht mehr und meldet "Laufzeit nicht eingerichtet".
            Write-Note "führe gesicherte Werkzeuge mit den neuen zusammen"
            Merge-Tree -Quelle $ToolsStash -Ziel $ToolsDir -Was "Werkzeuge"
        } else {
            Move-Tree -Quelle $ToolsStash -Ziel $ToolsDir
            Write-Note "Werkzeuge zurückgelegt: $ToolsDir"
        }
    }
}

if (-not (Test-Path $Target)) { throw "Ausgabeordner fehlt: $Target" }

# Portable-Marker SOFORT schreiben, nicht erst am Ende.
#
# Er entscheidet, wo die Anwendung ihre Daten sucht. Bricht der Bau
# danach ab, steht sonst ein lauffaehiges Programm ohne Marker da - und
# es legt einen ZWEITEN Datenbestand unter %LOCALAPPDATA% an. Fuer den
# Bediener sieht das aus, als seien Modelle und Stimmprofile weg.
Set-Content -Encoding utf8 -Path (Join-Path $Target "portable.txt") -Value @"
Portable-Modus: Modelle, Konfiguration und Ausgaben liegen im Unterordner data\.
Diese Datei löschen, damit stattdessen %LOCALAPPDATA%\StreamForge genutzt wird.
"@

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
        # Selbst anlegen statt abbrechen. Ein Schalter, der etwas
        # mitliefern soll, muss es auch beschaffen koennen - sonst
        # scheitert der Bau an einem Schritt, den niemand kennt.
        if ($VoiceRuntimeDir) {
            throw "Angegebene Klon-Laufzeit nicht gefunden: $VoiceRuntimeDir"
        }
        Write-Note "Klon-Laufzeit fehlt - wird angelegt (mehrere GB, dauert)."
        Invoke-Checked -File $VenvPython -Arguments @(
            "-m", "app", "voice-runtime", "install", "--target", $VoiceSrc
        ) -What "Klon-Laufzeit einrichten"
    }
    $VoiceOut = Join-Path $Target "tools\voice-runtime"
    $VoicePy = Join-Path $VoiceOut "Scripts\python.exe"
    # NICHT nur auf python.exe pruefen: eine halb kopierte Laufzeit (aus
    # einem abgebrochenen Bau) hat die auch, und dann fehlt chatterbox -
    # die Anwendung meldet danach "nicht eingerichtet", obwohl 4,8 GB
    # dort liegen. Auf das Paket pruefen, auf das es ankommt.
    $VoiceChatterbox = Join-Path $VoiceOut "Lib\site-packages\chatterbox"
    if ((Test-Path $VoicePy) -and (Test-Path $VoiceChatterbox)) {
        # Schon da (aus der Sicherung vor PyInstaller). Erneut mehrere GB
        # zu kopieren kostet nur Zeit und Plattenlast.
        Write-Note "Klon-Laufzeit liegt bereits – nur der Arbeiter wird aufgefrischt."
        Copy-Item (Join-Path $Root "packaging\voice_worker.py") -Destination $VoiceOut -Force
        $voiceSize = (Get-ChildItem -Recurse $VoiceOut -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
        Write-Note ("Klon-Laufzeit vorhanden: {0:N1} GB" -f $voiceSize)
    } else {
        # KEIN return: das Skript ist keine Funktion. Ein return
        # wuerde den ganzen Bau hier beenden - portable.txt und die
        # Abnahmeliste kaemen nie.
        Write-Note "kopiere Klon-Laufzeit (mehrere GB) …"
        New-Item -ItemType Directory -Force $VoiceOut | Out-Null

        # robocopy statt Copy-Item -Recurse.
        #
        # Gemessen: Copy-Item kopierte 4,7 GB und liess chatterbox aus -
        # ohne Fehler, ohne Warnung. Die Anwendung meldete danach
        # "chatterbox fehlt", obwohl der Ordner voll aussah. Dieselbe
        # Kopie mit robocopy ergab 5,0 GB samt chatterbox.
        #
        # robocopy kommt mit langen Pfaden zurecht und wiederholt bei
        # Fehlern. Rueckgabecodes unter 8 sind Erfolg.
        $rc = 0
        try {
            robocopy $VoiceSrc $VoiceOut /E /NFL /NDL /NJH /NJS /NP /R:2 /W:2 | Out-Null
            $rc = $LASTEXITCODE
        } catch {
            throw "robocopy fehlgeschlagen: $($_.Exception.Message)"
        }
        if ($rc -ge 8) { throw "Kopieren der Klon-Laufzeit fehlgeschlagen (robocopy $rc)." }

        Copy-Item (Join-Path $Root "packaging\voice_worker.py") -Destination $VoiceOut -Force

        # Gegenprobe: ohne chatterbox ist die Laufzeit wertlos, und der
        # Fehler faellt sonst erst beim ersten Sprechversuch auf.
        if (-not (Test-Path (Join-Path $VoiceOut "Lib\site-packages\chatterbox"))) {
            throw ("Die kopierte Klon-Laufzeit enthaelt kein chatterbox. " +
                   "Quelle pruefen: $VoiceSrc")
        }

        $voiceSize = (Get-ChildItem -Recurse $VoiceOut -EA SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum / 1GB
        Write-Note ("Klon-Laufzeit eingebettet und geprueft: {0:N1} GB" -f $voiceSize)
    }
} else {
    Write-Note ("ohne Klon-Laufzeit - selbst angelernte Stimmen sind dann nicht " +
                "nutzbar. Mitliefern mit -WithVoiceRuntime `$true (mehrere GB), " +
                "oder spaeter in der Anwendung unter 'Stimme anlernen' einrichten.")
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
