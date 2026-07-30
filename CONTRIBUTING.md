# Mitarbeit

Interner Leitfaden. Das Projekt ist proprietär (siehe `LICENSE`); Beiträge
von außen sind nicht vorgesehen.

## Umgebung einrichten

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU-Pfad – **Reihenfolge einhalten**, sonst installiert pip das CPU-Wheel und
betrachtet `torch>=2.6,<3` danach als erfüllt:

```powershell
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
pip install -r requirements-cuda.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Klonstimmen laufen in einer **getrennten** Umgebung
(`streamforge voice-runtime install`). Niemals `chatterbox-tts` in die
Hauptumgebung installieren – es stuft torch und diffusers herunter.

## Vor jedem Commit

```powershell
python -m compileall -q app tests run_app.py
python tests\smoke.py          # 46 Prüfungen, ohne Netz und ohne GPU
ruff check app tests           # falls installiert
```

Wer die Oberfläche angefasst hat, baut zusätzlich jede Seite einmal auf
(die Seiten werden erst beim ersten Aufruf erzeugt, Fehler zeigen sich
sonst erst beim Nutzer).

## Code-Regeln

- **Kommentare auf Deutsch, Bezeichner auf Englisch.** Umlaute ausschreiben,
  kein `ae`/`oe`/`ue`.
- Kommentare erklären das *Warum*, nicht das *Was*. Besonders wertvoll:
  Hinweise auf Fallstricke, die schon einmal Zeit gekostet haben.
- Keine Geheimnisse, Tokens oder Schlüssel im Quelltext.
- Fehlerbehandlung ausdrücklich; kein leeres `except`. Fremdbibliotheks-
  Fehler über `accel.clean_error()` säubern, bevor sie angezeigt werden.
- Sicherheits- und lizenzkritische Pfade sind **fail-closed**: im Zweifel
  nicht laden, sondern verständlich absagen.
- Konfiguration ist unveränderlich (`@dataclass(frozen=True)`), Varianten
  über `with_values()`.
- Lange Läufe brauchen `should_stop()` in der Schleife und Fortschritt über
  Rückrufe – niemals die Oberfläche blockieren.

## Modelle aufnehmen

Vor jeder Aufnahme die Lizenz prüfen und in `models.py` dokumentieren:

| Stufe | Bedeutung |
|---|---|
| `ALLOWED` | kommerziell klar erlaubt, darf Vorgabe sein |
| `CONDITIONAL` | erlaubt mit Auflage (Umsatzgrenze, Namensnennung, Registrierung) |
| `DENIED` | nicht kommerziell – Download wird verweigert |

Danach `MODELS.md` und `THIRD-PARTY-NOTICES.md` nachziehen. Im Zweifel das
Modell weglassen und nachfragen.

## Bauen

```powershell
.\build-windows.ps1 -SkipModelDownload            # schneller Testbau
.\build-windows.ps1 -Clean                        # Vollbau
.\build-windows.ps1 -WithVoiceRuntime $true       # mit Klonstimmen (5,2 GB)
```

Das Skript ist **UTF-8 mit BOM** – ohne BOM liest Windows PowerShell 5.1 es
als ANSI und bricht am ersten Gedankenstrich mit einem Parserfehler ab.

## Commits

Deutsch, Conventional Commits:

```
feat: Klonstimmen über getrennte Laufzeit
fix: CPU-Wheel im GPU-Build verhindern
docs: Lizenzlage zu Piper festhalten
chore: Abhängigkeiten festnageln
```

Direkt auf `main` nur nach Absprache.
