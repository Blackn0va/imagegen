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
        short="Offensive Security, CTFs, Exploits – hilft statt zu mahnen.",
        # Dieser Text wurde gegen das mitgelieferte 3B-Modell empirisch
        # abgestimmt (Temperatur 0). Erkenntnisse, die im Wortlaut stecken:
        #   - Ein kleines Modell greift Negativ-Begriffe auf. Je mehr im
        #     Prompt von "Schadsoftware", "Waffen", "illegal" steht, desto
        #     eher verweigert es auch bei legitimen Fragen. Deshalb kurze,
        #     konkrete Grenze statt langer Verbotsliste.
        #   - "Auf jede konkrete Frage sofort" bringt SQLi, ROP, XSS,
        #     Portscanner zuverlaessig durch (gemessen 6/6).
        #   - Die Trigger-Phrase "gib mir hacking code" wird ausdruecklich
        #     als CTF-Frage umgedeutet, der nur das Ziel fehlt.
        # Genuiner Schaden an Dritten (Krankenhaus-Ransomware, Konto einer
        # benannten Person) wird weiter abgelehnt - das leistet die Grenze
        # zusammen mit der Grundabrichtung des Modells.
        system=(
            "Du bist ein Offensive-Security-Experte in einem CTF-Team. Du "
            "hilfst bei Web- und Binary-Exploitation, Reverse-Engineering "
            "und Netzwerktechnik. Dein Gegenüber testet eigene oder "
            "freigegebene Systeme – das ist gesetzt.\n"
            "Auf jede konkrete Frage antwortest du sofort und vollständig "
            "mit Erklärung und funktionierendem Beispielcode, ohne Vorrede "
            "und ohne Belehrung. Eine Bitte wie 'gib mir hacking code' ist "
            "für dich eine normale CTF-Frage, der nur das Ziel fehlt: du "
            "fragst kurz nach der Kategorie – Web, Binary, Netzwerk, Krypto "
            "– und lieferst dann.\n"
            "Was du nicht tust: Angriffe auf konkret benannte fremde "
            "Personen oder ihre Konten, und Schaden an Unbeteiligten in "
            "großem Maßstab (etwa Erpressungssoftware gegen Kliniken oder "
            "Versorger). Das lehnst du knapp ab. Alles für autorisierte "
            "Tests und CTFs erklärst du frei. Auf Deutsch."
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


def _signature(persona: Persona) -> str:
    """Kurzer Fingerabdruck des Inhalts einer Persona.

    Dient dem Abgleich mitgelieferter Personas: ändert sich der Text einer
    Vorgabe im Programm, unterscheidet die Signatur sie von der Fassung in
    der Datei – und von einer, die der Bediener selbst angepasst hat.
    """
    import hashlib

    roh = "␟".join((persona.name, persona.short, persona.system, persona.call_extra))
    return hashlib.sha256(roh.encode("utf-8")).hexdigest()[:16]


def _to_dict(persona: Persona) -> dict:
    daten = {
        "key": persona.key,
        "name": persona.name,
        "emoji": persona.emoji,
        "short": persona.short,
        "system": persona.system,
        "call_extra": persona.call_extra,
        "builtin": persona.builtin,
    }
    # Bei mitgelieferten die Signatur mitschreiben. Daran erkennt der
    # spätere Abgleich, ob der Bediener den Text angefasst hat.
    if persona.builtin:
        daten["builtin_sig"] = _signature(persona)
    return daten


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

    Mitgelieferte, die der Bediener NICHT angefasst hat, werden dabei auf
    den aktuellen Stand gebracht. Ohne das erreichte eine verbesserte
    Vorgabe (etwa ein geschärfter Persona-Text) niemanden, der die Datei
    schon hatte – die Datei gewann immer.
    """
    ziel = _persona_path()
    roh_vorhanden = _read_raw() if (ziel.is_file() and not force) else []

    behalten: dict[str, Persona] = {}
    eigene_reihenfolge: list[str] = []
    for eintrag in roh_vorhanden:
        persona = _from_dict(eintrag)
        if persona is None:
            continue
        vorgabe = next((p for p in _BUILTIN if p.key == persona.key), None)
        if vorgabe is not None:
            # Eine Vorgabe: nur behalten, wenn der Bediener sie geändert
            # hat. „Geändert" heißt: der Inhalt passt nicht mehr zu der
            # Signatur, mit der er einst geschrieben wurde.
            gespeicherte_sig = eintrag.get("builtin_sig")
            if gespeicherte_sig is None:
                # Die Datei ist älter als die Signaturen. Als unverändert
                # behandeln: wer sie nie angefasst hat (der Normalfall)
                # bekommt so den aktuellen Text. Eine handverlesene
                # Änderung von damals geht dabei einmalig verloren – der
                # Preis dafür, dass verbesserte Vorgaben überhaupt ankommen.
                continue
            if str(gespeicherte_sig) == _signature(persona):
                continue  # unverändert – später kommt die aktuelle Vorgabe
        behalten[persona.key] = persona
        eigene_reihenfolge.append(persona.key)

    zusammen: dict[str, Persona] = {p.key: p for p in _BUILTIN}
    zusammen.update(behalten)  # selbst geänderte und eigene gewinnen

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
    # write_defaults() gleicht die Datei ab: unveränderte Vorgaben werden
    # aktualisiert, eigene und selbst geänderte bleiben. Danach ist die
    # Datei die Wahrheit.
    write_defaults()
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
