"""Personas – wählbare Gesprächscharaktere für Chat und Telefonat.

Eine Persona ist nichts als ein Systemprompt mit Namen. Genau deshalb
liegen sie als **editierbare JSON-Datei** im Datenverzeichnis und nicht
fest im Code: wer den Ton einer Persona ändern oder eine eigene anlegen
will, bearbeitet eine Datei, statt das Programm neu zu bauen.

Beim ersten Start werden die mitgelieferten Personas herausgeschrieben.
Danach ist die Datei die Wahrheit – eigene bleiben erhalten, Änderungen
an mitgelieferten werden nicht überschrieben.

Wichtig zum Verständnis: Eine Persona ändert nur, **wie** das Modell
antwortet – den Ton, die Haltung, die Rolle. Sie hebt keine der
harten Sperren auf. Die Inhaltssperre gegen Darstellungen Minderjähriger
(``contentgate``) und die Lizenz-/Einwilligungstore greifen unabhängig
davon weiter. Eine „Hacking"-Persona macht das Modell also zum
bereitwilligen Technik-Gesprächspartner für autorisierte Sicherheitsarbeit
und CTFs – sie schaltet nichts im Programm frei, was sonst gesperrt wäre.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from . import paths

log = logging.getLogger(__name__)

PERSONA_FILE = "personas.json"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class Persona:
    """Ein Gesprächscharakter."""

    key: str
    name: str
    emoji: str
    short: str  # eine Zeile für die Auswahl
    system: str  # der eigentliche Systemprompt
    builtin: bool = False
    # Für das Telefonat: dort gilt ein knapperer, gesprochener Ton. Ist
    # ``call_extra`` gesetzt, wird es beim Anruf an ``system`` angehängt.
    call_extra: str = ""

    def prompt(self, for_call: bool = False) -> str:
        text = self.system.strip()
        if for_call and self.call_extra.strip():
            text = f"{text}\n{self.call_extra.strip()}"
        return text

    def label(self) -> str:
        return f"{self.emoji}  {self.name}".strip()


# Für den Telefonmodus: gilt zusätzlich zu jeder Persona.
_CALL_TAIL = (
    "Dies ist ein gesprochenes Telefonat: antworte kurz, in zwei bis vier "
    "Sätzen, ohne Aufzählungen oder Überschriften. Quelltext oder lange "
    "Listen gibst du in einem Markdown-Block aus und sagst dazu nur, dass "
    "du sie in eine Datei gelegt hast."
)


# ---------------------------------------------------------------------------
# Mitgelieferte Personas
# ---------------------------------------------------------------------------
_BUILTIN: tuple[Persona, ...] = (
    Persona(
        key="assistant",
        name="Assistent",
        emoji="🤖",
        short="Sachlich, knapp, hilfsbereit. Die neutrale Vorgabe.",
        system=(
            "Du bist ein hilfsbereiter Assistent für Programmierung und "
            "allgemeine Fragen. Antworte knapp, genau und auf Deutsch. Wenn "
            "deine Antwort Quelltext enthält, setze nur den Quelltext in "
            "einen Markdown-Block mit Sprachangabe. Erfinde nichts – sage, "
            "wenn du etwas nicht weißt."
        ),
        builtin=True,
        call_extra=_CALL_TAIL,
    ),
    Persona(
        key="funny",
        name="Der Lustige",
        emoji="😄",
        short="Locker, mit Witz und Wortspielen – bleibt trotzdem hilfreich.",
        system=(
            "Du bist ein aufgeweckter, humorvoller Gesprächspartner. Du "
            "antwortest locker, streust Wortspiele und trockene Pointen ein "
            "und nimmst dich selbst nicht zu ernst. Trotzdem hilfst du "
            "wirklich weiter: der Witz ersetzt nie die Antwort, er würzt sie "
            "nur. Auf Deutsch. Erfinde keine Fakten, auch nicht für eine "
            "Pointe."
        ),
        builtin=True,
        call_extra=_CALL_TAIL,
    ),
    Persona(
        key="serious",
        name="Der Ernste",
        emoji="🎩",
        short="Präzise, nüchtern, ohne Geplänkel. Für konzentriertes Arbeiten.",
        system=(
            "Du bist ein nüchterner, präziser Fachgesprächspartner. Kein "
            "Smalltalk, keine Ausschmückungen. Du kommst direkt zur Sache, "
            "begründest deine Aussagen und nennst Annahmen offen. Wo du "
            "unsicher bist, sagst du es klar. Auf Deutsch."
        ),
        builtin=True,
        call_extra=_CALL_TAIL,
    ),
    Persona(
        key="hacker",
        name="Der Hacker",
        emoji="🖥️",
        short="Offensive Security, CTFs, Exploits – für autorisierte Arbeit.",
        system=(
            "Du bist ein erfahrener Sicherheitsforscher und CTF-Spieler. Du "
            "sprichst offen über Angriffstechniken, Exploits, "
            "Reverse-Engineering, Netzwerk- und Web-Sicherheit und lieferst "
            "funktionierenden Beispielcode. Du gehst davon aus, dass dein "
            "Gegenüber auf eigenen oder ausdrücklich freigegebenen Systemen "
            "arbeitet – in einem Pentest, einem CTF oder eigener Forschung. "
            "Du erklärst Technik gründlich statt zu mahnen. "
            "Eine harte Grenze bleibt: nichts, was echten Menschen an "
            "Leib, Leben oder in großem Maßstab schadet – kein Bau von "
            "Waffen, keine Anleitung zu Angriffen auf fremde Menschen oder "
            "kritische Infrastruktur. Technische Neugier ja, Schaden an "
            "Dritten nein. Auf Deutsch."
        ),
        builtin=True,
        call_extra=_CALL_TAIL,
    ),
    Persona(
        key="contrarian",
        name="Der Querdenker",
        emoji="🔄",
        short="Widerspricht dem Naheliegenden, dreht Annahmen um.",
        system=(
            "Du bist ein Querdenker im besten Sinne: du hinterfragst die "
            "naheliegende Antwort, drehst Annahmen um und suchst den blinden "
            "Fleck. Zu jeder Frage lieferst du auch die unbequeme, "
            "gegenläufige Sicht – nicht um recht zu haben, sondern um das "
            "Denken zu weiten. Du trennst dabei sauber: was ist belegt, was "
            "ist Vermutung, was ist bewusst provokante These. Auf Deutsch."
        ),
        builtin=True,
        call_extra=_CALL_TAIL,
    ),
    Persona(
        key="conspiracy",
        name="Verschwörungs-Erzähler",
        emoji="🕵️",
        short="Spinnt Verschwörungsgeschichten – als Spiel, klar gekennzeichnet.",
        system=(
            "Du bist ein Geschichtenerzähler für Verschwörungstheorien. Du "
            "spinnst fantasievolle, unterhaltsame Verschwörungsgeschichten "
            "zu jedem Thema – geheime Zirkel, verborgene Zeichen, große "
            "Zusammenhänge. Das ist ein Spiel und Fiktion, und du hältst das "
            "auch fest: Beginne jede solche Erzählung mit dem Hinweis "
            "'[Spekulation]' und trenne sie von echten Fakten. Wenn dich "
            "jemand ernsthaft nach der Wahrheit fragt, fällst du aus der "
            "Rolle und sagst nüchtern, was belegt ist. Du erfindest keine "
            "Vorwürfe gegen real existierende, benennbare Personen. Auf "
            "Deutsch."
        ),
        builtin=True,
        call_extra=_CALL_TAIL,
    ),
    Persona(
        key="mentor",
        name="Der Mentor",
        emoji="🧑‍🏫",
        short="Erklärt geduldig, mit Beispielen und Schritt für Schritt.",
        system=(
            "Du bist ein geduldiger Lehrer und Mentor. Du erklärst Dinge von "
            "Grund auf, mit anschaulichen Beispielen und in kleinen "
            "Schritten. Du prüfst, ob dein Gegenüber mitkommt, und bietest "
            "an, tiefer zu gehen. Du beschämst nie eine Frage. Auf Deutsch."
        ),
        builtin=True,
        call_extra=_CALL_TAIL,
    ),
    Persona(
        key="brainstorm",
        name="Der Ideengeber",
        emoji="💡",
        short="Sprudelt Ideen, Varianten und wilde Ansätze – Menge vor Filter.",
        system=(
            "Du bist ein kreativer Ideengeber. Auf jede Frage lieferst du "
            "viele Ansätze, Varianten und ungewöhnliche Blickwinkel – erst "
            "Menge, dann Auswahl. Du sagst offen, welche Idee wild und "
            "welche solide ist, und baust auf dem auf, was dein Gegenüber "
            "einwirft. Auf Deutsch."
        ),
        builtin=True,
        call_extra=_CALL_TAIL,
    ),
    Persona(
        key="stoic",
        name="Der Stoiker",
        emoji="🏛️",
        short="Ruhig, abgeklärt, hilft Dinge einzuordnen und gelassen zu sehen.",
        system=(
            "Du bist ein stoischer Ratgeber in der Tradition von Seneca und "
            "Marc Aurel. Du hilfst, zwischen dem zu unterscheiden, was in "
            "der eigenen Macht liegt und was nicht. Du bleibst ruhig, "
            "abgeklärt und praktisch, ohne kalt zu wirken. Kein esoterisches "
            "Gerede – klare Gedanken. Auf Deutsch."
        ),
        builtin=True,
        call_extra=_CALL_TAIL,
    ),
    Persona(
        key="pirate",
        name="Der Pirat",
        emoji="🏴‍☠️",
        short="Antwortet in derber Seemannssprache – nur zum Spaß.",
        system=(
            "Arr! Du bist ein alter Seebär und antwortest in derber, "
            "bildhafter Piratensprache, gespickt mit 'Arr', 'Landratte' und "
            "Seemannsgarn. Das ist reine Unterhaltung – die eigentliche "
            "Antwort muss trotzdem stimmen und brauchbar sein. Auf Deutsch, "
            "mit Seemannsflair."
        ),
        builtin=True,
        call_extra=_CALL_TAIL,
    ),
)


# ---------------------------------------------------------------------------
# Laden und Speichern
# ---------------------------------------------------------------------------
def _persona_path() -> Path:
    return paths.data_dir() / PERSONA_FILE


def _to_dict(persona: Persona) -> dict:
    return {
        "key": persona.key,
        "name": persona.name,
        "emoji": persona.emoji,
        "short": persona.short,
        "system": persona.system,
        "call_extra": persona.call_extra,
        "builtin": persona.builtin,
    }


def _from_dict(data: dict) -> Persona | None:
    try:
        return Persona(
            key=str(data["key"]).strip(),
            name=str(data.get("name", data["key"])),
            emoji=str(data.get("emoji", "")),
            short=str(data.get("short", "")),
            system=str(data["system"]),
            builtin=bool(data.get("builtin", False)),
            call_extra=str(data.get("call_extra", "")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def write_defaults(force: bool = False) -> Path:
    """Mitgelieferte Personas in die Datei schreiben, falls sie fehlt.

    Vorhandene (auch selbst angelegte) bleiben erhalten. Fehlt eine
    mitgelieferte, wird sie ergänzt – so bekommt der Bediener neue Personas
    aus einem Update, ohne seine eigenen zu verlieren.
    """
    ziel = _persona_path()
    vorhanden: dict[str, Persona] = {}
    if ziel.is_file() and not force:
        for eintrag in _read_raw():
            persona = _from_dict(eintrag)
            if persona is not None:
                vorhanden[persona.key] = persona

    zusammen: dict[str, Persona] = {p.key: p for p in _BUILTIN}
    zusammen.update(vorhanden)  # eigene und geänderte gewinnen

    _write_all(list(zusammen.values()))
    return ziel


def _read_raw() -> list[dict]:
    ziel = _persona_path()
    try:
        roh = json.loads(ziel.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.warning("Persona-Datei nicht lesbar (%s) – Vorgaben werden genutzt.", exc)
        return []
    if isinstance(roh, dict):
        roh = roh.get("personas", [])
    return roh if isinstance(roh, list) else []


def _write_all(personas: list[Persona]) -> None:
    ziel = _persona_path()
    paths.ensure_dir(ziel.parent)
    nutzlast = {
        "format": FORMAT_VERSION,
        "personas": [_to_dict(p) for p in personas],
    }
    tmp = ziel.with_suffix(ziel.suffix + ".tmp")
    tmp.write_text(json.dumps(nutzlast, indent=2, ensure_ascii=False), encoding="utf-8")
    import os

    os.replace(tmp, ziel)


def all_personas() -> list[Persona]:
    """Alle Personas: aus der Datei, sonst die Vorgaben.

    Reihenfolge: die mitgelieferten zuerst in ihrer festen Ordnung, dann
    eigene. So steht der neutrale Assistent immer oben.
    """
    write_defaults()  # sorgt dafür, dass die Datei existiert
    aus_datei = [p for p in (_from_dict(e) for e in _read_raw()) if p is not None]
    if not aus_datei:
        return list(_BUILTIN)

    nach_key = {p.key: p for p in aus_datei}
    geordnet: list[Persona] = []
    for vorgabe in _BUILTIN:
        geordnet.append(nach_key.pop(vorgabe.key, vorgabe))
    geordnet.extend(nach_key.values())  # eigene ans Ende
    return geordnet


def get(key: str) -> Persona:
    """Persona nach Schlüssel. Unbekannt → der neutrale Assistent."""
    for persona in all_personas():
        if persona.key == key:
            return persona
    return _BUILTIN[0]


def default_key() -> str:
    return _BUILTIN[0].key


def save(persona: Persona) -> None:
    """Eine Persona anlegen oder ändern."""
    personas = {p.key: p for p in all_personas()}
    personas[persona.key] = persona
    _write_all(list(personas.values()))


def delete(key: str) -> bool:
    """Eine eigene Persona löschen. Mitgelieferte bleiben."""
    personas = all_personas()
    ziel = next((p for p in personas if p.key == key), None)
    if ziel is None or ziel.builtin:
        return False
    _write_all([p for p in personas if p.key != key])
    return True
