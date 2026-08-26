# Allgemeine Geschäftsbedingungen und Endnutzer-Lizenzvertrag

Für die Anwendung **StreamForge Studio**. Diese Bedingungen regeln, wer sie
nutzen darf, wozu, und wer für die erzeugten Inhalte einsteht. Sie sind beim
ersten Start zu bestätigen; ohne Zustimmung wird die Anwendung nicht genutzt.

## 1. Geltungsbereich

Diese Bedingungen gelten für die ausgelieferte, ausführbare Fassung von
StreamForge Studio samt aller mitgelieferten Bestandteile. Für den Quelltext
gilt zusätzlich die Datei `LICENSE`. Ergänzend gelten `MODELS.md` (Modelle und
ihre Auflagen), `THIRD-PARTY-NOTICES.md` und `SECURITY.md`. Für Bestandteile
Dritter gehen deren Lizenzen diesen Bedingungen vor.

## 2. Anbieter und Kontakt

- Anbieter: Blackn0va
- Website: https://streamwizard.de
- Anschrift: [Anschrift eintragen]
- E-Mail: [E-Mail-Adresse eintragen]

Angaben in eckigen Klammern sind vor einer Weitergabe an Dritte auszufüllen.

## 3. Was die Anwendung tut

StreamForge Studio erzeugt Bilder, Video, Sprache und Text mit KI-Modellen,
dazu Bildbearbeitung (Vergrößern, Umarbeiten, Inpainting, Einfärben),
Diamond-Painting-Vorlagen, einen Chat und eine Telefonie-Funktion.

- Eingaben, Modelle und Ergebnisse werden **ausschließlich lokal** auf dem
  Rechner des Nutzers verarbeitet. Es werden keine Inhalte an den Anbieter oder
  an einen Dienstleister übertragen. Es gibt kein Konto und keine Telemetrie.
- Netzzugriff findet nur beim Bezug von Modellen statt; diese werden von
  **Hugging Face** heruntergeladen. Der Offline-Modus schaltet auch das ab.
- Dafür gelten die Bedingungen von Hugging Face und die des jeweiligen
  Modell-Repos. Bei zugangsbeschränkten Repos stimmt der Nutzer dort selbst zu
  und verwendet einen eigenen Zugangsschlüssel.

## 4. Nutzungsrecht

Die Anwendung ist **proprietär**. Sie wird privat betrieben und nicht verkauft
oder vermietet. Erlaubt ist die Nutzung auf den eigenen Geräten des Nutzers.
Ohne vorherige schriftliche Zustimmung des Anbieters nicht erlaubt sind:

- Weitergabe, Verkauf, Vermietung oder Veröffentlichung der Anwendung, auch
  auszugsweise,
- Bereitstellen als Dienst für Dritte,
- Zurückentwickeln oder Umgehen der eingebauten Sperren, insbesondere der
  Inhaltssperre und der Einwilligungspflicht für Stimmen.

Beim ersten Start wird die Freischaltung **private Nutzung** gesetzt. Sie
erlaubt Modelle, deren Lizenz die kommerzielle Nutzung einschränkt oder
ausschließt. Daraus folgt:

- Sie gilt nur für private, nicht-kommerzielle Nutzung. Sobald mit den
  Ergebnissen Geld verdient wird – Verkauf, Auftragsarbeit, Werbung, Streaming
  mit Einnahmen –, greift die Beschränkung wieder.
- **Mit dieser Freischaltung darf die Anwendung nicht weitergegeben und nicht
  verkauft werden.** Die eingeschränkten Modelle wären sonst Teil eines
  kommerziellen Produkts.

Die Freischaltung wird mit Zeitpunkt und Fassung protokolliert und lässt sich
einzeln widerrufen (`streamforge licenses revoke private-use`).

## 5. Modelle und deren Lizenzen

Die Lizenzen der verwendeten Modelle bleiben bestehen, samt Namensnennung,
Weitergabeverboten und Nutzungsbeschränkungen.

- **FLUX.1-dev** ist nicht-kommerziell lizenziert; nur FLUX.1-schnell
  (Apache-2.0) ist kommerziell nutzbar.
- Modelle unter **CreativeML Open RAIL-M / RAIL++-M** (SD 1.5, SDXL und darauf
  aufbauende Feinabstimmungen) tragen die Nutzungsbeschränkungen aus Anhang A.
  Diese sind einzuhalten und bei Weitergabe von Ergebnissen weiterzugeben.
- Modelle unter der **Stability AI Community License** sind nur bis zu einer
  Umsatzgrenze und nach Registrierung kommerziell nutzbar.
- Einzelne Modelle sind gesperrt, weil ihre Lizenz die kommerzielle Nutzung
  ausschließt. Setzt ein Modell ein Wasserzeichen in die Ausgabe, darf dieses
  nicht entfernt werden.

Lizenzstufe und Auflagen je Modell zeigen `streamforge models table` und
`MODELS.md`. Vor einer Veröffentlichung von Ergebnissen prüft der Nutzer die
jeweils geltende Lizenzlage selbst; Anbieter ändern ihre Lizenzen.

## 6. Stimmen anlernen und Stimmklonen

Für **jede** angelernte Stimme muss eine dokumentierte Einwilligung der
sprechenden Person vorliegen, mit Name, Zweck und Datum.

- Stimmen realer Personen dürfen ohne deren Einwilligung nicht angelernt und
  nicht verwendet werden.
- Ergebnisse dürfen nicht als echte Äußerung dieser Person ausgegeben werden.
- Die Einwilligung ist jederzeit widerrufbar; danach ist das Stimmprofil zu
  löschen. Das Löschen ist der Widerrufsweg in der Anwendung.
- Ohne gültigen Nachweis wird ein Profil nicht angelernt und nicht benutzt,
  auch nicht in der Telefonie-Funktion. Diese Sperre ist nicht abschaltbar.

Der Nachweis dokumentiert eine Einwilligung, er ersetzt sie nicht. Bei fremden
Stimmen sind zusätzlich Schriftform, Zweckbindung, Vergütung und Widerruf zu
klären.

## 7. Telefonie-Funktion und Mikrofonaufnahmen

Die Telefonie-Funktion nimmt Sprache über das Mikrofon auf, erkennt sie lokal
und antwortet gesprochen. Alles bleibt auf dem Rechner.

- Aufnahmen und Mitschrift landen im Ausgabeverzeichnis
  (`output\telefonate\<Zeitstempel>\`), die Mitschrift nach jedem Zug; erkannte
  Code-Blöcke zusätzlich als eigene Dateien.
- Der Nutzer ist dafür verantwortlich, Gesprächspartner vor der Aufnahme zu
  informieren und ihr Einverständnis einzuholen.
- In Deutschland ist das heimliche Aufnehmen des nichtöffentlich gesprochenen
  Wortes nach **§ 201 StGB** strafbar. Andere Länder haben eigene Regeln.
- Aufnahmen und Mitschriften sind zu löschen, wenn sie nicht mehr nötig sind.

## 8. Verbotene Nutzung

Nicht zulässig ist es, mit der Anwendung Inhalte zu erzeugen, zu speichern oder
zu verbreiten, die

- Rechte Dritter verletzen – Urheberrecht, Marken-, Persönlichkeits- oder
  Datenschutzrechte,
- sexualisierte Darstellungen Minderjähriger zeigen oder andeuten,
- Personen verleumden, herabwürdigen oder falsche Tatsachen über sie behaupten,
- über die Identität realer Personen täuschen (Deepfakes in Bild, Video oder
  Stimme),
- zu Straftaten oder Gewalt aufrufen oder sonst rechtswidrig sind.

Die Anwendung enthält dazu eine nicht abschaltbare Textsperre: Aufträge, die
Begriffe für Minderjährige mit sexuellen Begriffen verbinden, werden vor dem
Laden des Modells abgelehnt, im Prompt wie im Negativ-Prompt. Das ist eine
Untergrenze, kein Vollschutz; geprüft wird der Text, nicht das Bild. Die
Inhaltsprüfung der Modelle für Nacktheit und Erotik ist standardmäßig
abgeschaltet und wieder einschaltbar; Charaktere im Chat ändern nur den Ton,
alle Sperren gelten unverändert weiter.

## 9. Kennzeichnung erzeugter Inhalte

Erzeugte Bilder, Videos, Sprachaufnahmen und Texte sind als KI-erzeugt zu
kennzeichnen, wo das Recht es verlangt – insbesondere bei Veröffentlichung, bei
Inhalten mit realen Personen und bei allem, was für eine echte Aufnahme
gehalten werden könnte. Die Anwendung kennzeichnet nichts automatisch.

## 10. Verantwortung für Ergebnisse

Der Nutzer trägt die Verantwortung für alle Inhalte, die er erzeugt, speichert,
veröffentlicht oder weitergibt – auch für die Wahl von Eingaben und Modellen.
Die Anwendung prüft Ergebnisse nicht auf Rechtsverstöße; der Anbieter sieht sie
nicht und hat keinen Zugriff darauf.

## 11. Daten auf dem Gerät

An den Anbieter werden keine personenbezogenen Daten übertragen. Im
Datenverzeichnis liegen aber Dateien mit Personenbezug, für die der Nutzer
verantwortlich ist: `voices\**` (Sprachaufnahmen und Einwilligungs-Nachweise),
`consent.json` (Zustimmungen mit Zeitstempel) und `logs\*.log` (Prompts,
Dateipfade). Sie gehören weder in ein Repository noch in einen Fehlerbericht.

## 12. Gewährleistung und Haftung

Die Anwendung wird **wie besehen** bereitgestellt, ohne Gewähr für
Verfügbarkeit, Eignung für einen bestimmten Zweck oder Fehlerfreiheit.

- Es besteht keine Zusage, dass ein Ergebnis brauchbar, richtig, vollständig
  oder frei von Rechten Dritter ist. Modelle erfinden Inhalte.
- Fehlt eine Voraussetzung – Modell, Zustimmung, Laufzeit –, schaltet die
  Anwendung auf einen Ersatzpfad um und nennt den Grund. Ein Anspruch auf eine
  bestimmte Ausgabequalität besteht nicht; genannte Geschwindigkeiten sind
  Messwerte, keine Zusage.
- Die Haftung ist auf Vorsatz und grobe Fahrlässigkeit beschränkt; für
  Datenverlust haftet der Anbieter nicht. Die Haftung für Schäden aus der
  Verletzung des Lebens, des Körpers oder der Gesundheit sowie nach dem
  Produkthaftungsgesetz bleibt unberührt.

## 13. Zustimmung, Widerruf und Änderungen

Die Zustimmung erfolgt beim ersten Start und deckt die übrigen
Lizenz-Komponenten mit ab, weil diese Bedingungen deren Auflagen im Wortlaut
nennen. Jede Komponente wird einzeln protokolliert und ist einzeln widerrufbar;
ein Widerruf hält, bis die Bedingungen erneut bestätigt werden. Ändert sich ihr
Wortlaut, ist erneut zuzustimmen. Die Einwilligung je angelernter Stimme bleibt
davon unberührt – sie wird pro Profil erhoben.

## 14. Anwendbares Recht

Es gilt deutsches Recht. Gerichtsstand, soweit zulässig vereinbar:
[Gerichtsstand eintragen]. Ist eine Bestimmung unwirksam, bleiben die übrigen
wirksam.

## 15. Fassung

Fassung vom 26.08.2026. Die Anwendung bildet die Fassungskennung zur Laufzeit
aus einer Prüfsumme dieses Textes und zeigt sie im Zustimmungsdialog an.

---

**Hinweis:** Dieser Text ersetzt keine Rechtsberatung. Vor einer Weitergabe der
Anwendung an Dritte – und erst recht vor einem Verkauf – sind diese Bedingungen
anwaltlich zu prüfen und die Platzhalter in eckigen Klammern auszufüllen.
