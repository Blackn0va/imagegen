# Sicherheit

## Schwachstelle melden

Bitte **nicht** öffentlich melden, sondern an [SICHERHEIT-E-MAIL].
Sinnvolle Angaben: betroffene Fassung (steht unter *Hardware → Bericht
kopieren*), Schritte zum Nachstellen, erwartetes und tatsächliches Verhalten.

Rückmeldung erfolgt innerhalb von [FRIST] Werktagen.

## Was die Anwendung tut – und was nicht

- Sie verarbeitet Eingaben, Modelle und Ergebnisse **ausschließlich lokal**.
  Es werden keine Inhalte an den Anbieter übertragen.
- Netzzugriff findet nur beim Modellbezug statt (Hugging Face). Der
  Offline-Modus schaltet auch das ab.
- TLS-Prüfung läuft über den Windows-Zertifikatspeicher (`truststore`), mit
  `certifi` als Rückfallebene. Zertifikatsprüfung wird nie abgeschaltet.

## Sicherheitsrelevante Entwurfsentscheidungen

- **Fail-closed**: Fehlt eine Zustimmung oder ein Nachweis, wird die
  Funktion nicht ausgeführt – nicht stillschweigend weitergelaufen. Betrifft
  proprietäre Laufzeiten, Stimmklonen und gesperrte Modelle.
- **Einwilligungs-Nachweis für Stimmen** enthält eine Prüfsumme des
  Wortlauts. Wird der Text nachträglich verändert, gilt der Nachweis als
  ungültig und das Profil wird gesperrt.
- **Downloads** landen erst in `.part`-Dateien und werden dann umbenannt.
  Ein Abbruch hinterlässt keine halbe Datei, die später als vollständiges
  Modell gilt.
- **Einzelinstanz-Sperre** über einen benannten Kernel-Mutex, den das System
  auch nach einem Absturz aufräumt.
- **Klonstimmen** laufen in einem getrennten Prozess mit eigener Umgebung.

## Personenbezogene Daten im Datenverzeichnis

Diese Dateien enthalten Angaben zu Personen und gehören **nicht** in ein
Repository oder in einen Fehlerbericht:

- `<daten>\voices\**` – Sprachaufnahmen und Einwilligungs-Nachweise
  (Name der sprechenden Person, Zweck, Datum)
- `<daten>\consent.json` – erteilte Zustimmungen mit Zeitstempel
- `<daten>\logs\*.log` – kann Prompts und Dateipfade enthalten

`.gitignore` schließt sie aus. Vor dem Versand von Protokollen bitte
durchsehen.

## Bekannte Einschränkungen

- Erzeugte Inhalte werden nicht auf Rechtsverstöße geprüft. Die
  Verantwortung liegt beim Nutzer (siehe `AGB.md`).
- Der Einwilligungs-Nachweis dokumentiert eine Einwilligung, er ersetzt sie
  nicht.
- Modelle stammen von Dritten und werden nicht auf Hintertüren geprüft.
  Deshalb werden ausschließlich `safetensors` geladen und keine
  Pickle-basierten Formate.
