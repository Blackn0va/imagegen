"""Inhaltssperre für die Bilderzeugung.

Die Anwendung kann Inhalte für Erwachsene erzeugen (siehe Lizenz-Komponente
``nsfw``). Zwei Dinge bleiben davon unberührt und sind **nicht** über die
Konfiguration abschaltbar:

1. **Keine Darstellung Minderjähriger in sexuellem Zusammenhang.** Das ist
   in Deutschland nach § 184b StGB strafbar, auch wenn die Darstellung
   vollständig computererzeugt ist, und ebenso in den meisten anderen
   Absatzmärkten. Für eine verkaufte Anwendung wäre schon die Möglichkeit
   ein untragbares Risiko – für Anbieter wie Kunde.
2. **Keine sexualisierte Darstellung realer Personen ohne deren
   Einwilligung.** Dafür gibt es keine technische Prüfung; es steht als
   Auflage in der Lizenz-Komponente und in den AGB.

Was dieses Modul leistet und was nicht:

  * Es prüft **Text** – Prompt und Negativ-Prompt – auf Kombinationen aus
    Minderjährigkeits- und Sexualbegriffen und verweigert den Auftrag.
  * Es hängt Schutzbegriffe an den Negativ-Prompt, damit das Modell in
    diese Richtung gar nicht erst zieht.
  * Es ist **keine** Bildprüfung und keine Garantie. Es ist eine Untergrenze.
    Die Verantwortung für das Erzeugte bleibt beim Bediener; das steht so
    auch in den Auflagen.

Fail-closed: im Zweifel wird abgelehnt, nicht erzeugt.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Begriffe
# ---------------------------------------------------------------------------
# Begriffe, die praktisch ausschließlich Minderjährige bezeichnen. Sie
# blockieren allein – ohne dass ein Sexualbegriff dazukommen muss.
ALWAYS_BLOCKED: tuple[str, ...] = (
    "loli", "lolis", "lolicon", "shota", "shotas", "shotacon", "jailbait",
    "preteen", "preteens", "pre teen", "toddlercon", "kinderporno",
    "kinderpornografie", "kinderpornographie", "childporn", "child porn",
    "child pornography",
)

# Hinweise auf Minderjährige. Blockieren in Verbindung mit einem
# Sexualbegriff. Bewusst NICHT enthalten: "girl", "boy", "mädchen",
# "junge", "young" – diese Wörter stehen in erwachsenen Prompts ständig
# und würden die Sperre zur Dauerbremse machen, ohne Schutz zu bringen.
MINOR_TERMS: tuple[str, ...] = (
    # Englisch
    "child", "children", "childlike", "kid", "kids", "toddler", "toddlers",
    "infant", "infants", "baby", "babies", "newborn", "minor", "minors",
    "underage", "under age", "teen", "teens", "teenage", "teenager",
    "teenagers", "adolescent", "adolescents", "juvenile", "schoolgirl",
    "schoolboy", "elementary school", "middle school", "grade school",
    "kindergarten", "nursery", "prepubescent", "pubescent", "lolita",
    "little sister", "little brother",
    # Deutsch
    "kind", "kindes", "kindern", "kita", "vorschule",
)

# Deutsche Zusammensetzungen: hier wird nur der Wortanfang geprüft, damit
# "kinderzimmer", "schülerinnen" oder "minderjährige" mitgehen. Die
# Einträge sind so gewählt, dass sie nicht in gängigen englischen Wörtern
# stecken – anders als das bloße "kind", das auch in "kindness" steht und
# deshalb oben als ganzes Wort geprüft wird.
MINOR_PREFIXES: tuple[str, ...] = (
    "kinder", "kindlich", "kleinkind", "säugling", "saeugling",
    "minderjährig", "minderjaehrig", "jugendlich", "schülerin",
    "schuelerin", "schüler", "schueler", "grundschul", "vorschul",
    "vorpubertär", "vorpubertaer", "halbwüchsig", "halbwuechsig",
)

# Sexualbegriffe. Für sich genommen erlaubt (das ist ja der Sinn der
# Freigabe) – nur die Kombination mit einem Begriff oben ist gesperrt.
SEXUAL_TERMS: tuple[str, ...] = (
    # Englisch – als ganzes Wort geprüft
    "nude", "nudes", "nudity", "naked", "nsfw", "explicit", "porn",
    "porno", "hentai", "erotic", "erotica", "sex", "sexy", "topless",
    "bottomless", "strip", "stripping", "lingerie", "penis", "vagina",
    "vulva", "labia", "anus", "breast", "breasts", "boobs", "tits",
    "nipple", "nipples", "areola", "cleavage", "orgasm", "fellatio",
    "cunnilingus", "intercourse", "cum", "semen", "bdsm", "fetish",
    "seductive", "provocative", "spread legs",
    # Deutsch – als ganzes Wort geprüft
    "busen", "brustwarzen", "schambereich", "schamlippen", "scheide",
    "oben ohne", "dessous", "geschlechtsverkehr",
)

# Sexualbegriffe, bei denen der Wortanfang genügt. Nötig für die deutsche
# Beugung ("nackt" -> "nacktes", "erotisch" -> "erotische") und für
# englische Formen wie "undressing". Bewusst NICHT dabei: "anal" (steckt in
# "analysis", "analog") und "breast" als Anfang (steckt in
# "breastfeeding") – solche Treffer würden zusammen mit einem Altersbegriff
# harmlose Motive sperren.
SEXUAL_PREFIXES: tuple[str, ...] = (
    "nackt", "erotisch", "erotik", "sexuell", "sexualis", "sexualiz",
    "pornograf", "pornograph", "unbekleidet", "entkleidet", "ausgezogen",
    "verführerisch", "verfuehrerisch", "undress", "masturbat", "genital",
)

# "12 years old", "9-jährig", "14 jahre alt" …
_AGE_PATTERNS = (
    re.compile(r"\b(\d{1,2})\s*(?:\+)?\s*(?:years?|yrs?|yo)\b[ -]*(?:old)?"),
    re.compile(r"\b(\d{1,2})\s*[- ]?\s*(?:jahre|jahren|jährig\w*|jaehrig\w*)\b"),
    re.compile(r"\bage\s*[:=]?\s*(\d{1,2})\b"),
    re.compile(r"\balter\s*[:=]?\s*(\d{1,2})\b"),
)

ADULT_AGE = 18

# Wird bei freigegebener NSFW-Erzeugung an den Negativ-Prompt gehängt.
PROTECTIVE_NEGATIVE = (
    "child, children, kid, toddler, infant, baby, teen, teenager, "
    "underage, minor, loli, shota, childlike body"
)


class BlockedContent(RuntimeError):
    """Auftrag abgelehnt. Die Meldung ist für den Bediener bestimmt.

    ``expected`` sagt der Warteschlange, dass dies eine bewusste Ablehnung
    ist und kein Programmfehler – sie schreibt dann eine Warnung statt
    eines Stacktrace ins Protokoll (siehe jobs.py).
    """

    expected = True


# ---------------------------------------------------------------------------
# Prüfung
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    """Kleinschreibung, Akzente weg, Trennzeichen zu Leerzeichen.

    Damit greifen die Listen auch bei ``ch1ld``-Schreibweisen mit
    Sonderzeichen, ``t_e_e_n`` oder ``Kind-lich``. Vollständig lässt sich
    das nicht abfangen – es ist eine Hürde, kein Schloss.
    """
    lowered = unicodedata.normalize("NFKD", text.lower())
    # Umlaute bleiben erhalten (ä ö ü stehen in den deutschen Begriffen),
    # deshalb nur kombinierende Zeichen entfernen, die nicht dazugehören.
    lowered = "".join(ch for ch in lowered if not unicodedata.combining(ch) or ch in "̈")
    lowered = unicodedata.normalize("NFC", lowered)
    # Ziffern-Ersatzschreibweisen zurückführen
    for wrong, right in (("0", "o"), ("1", "i"), ("3", "e"), ("4", "a"), ("5", "s"), ("@", "a")):
        lowered = lowered.replace(wrong, right)
    return re.sub(r"[^a-zäöüß0-9]+", " ", lowered)


def _contains(haystack: str, needles: tuple[str, ...], prefix: bool = False) -> list[str]:
    """Begriffe suchen. Vorgabe: ganzes Wort.

    Ohne die hintere Wortgrenze träfe "kind" auch "kindness" und "loli"
    auch "lolita fashion". Fehlalarme sind hier nicht harmlos: eine Sperre,
    die bei erwachsenen Motiven ständig grundlos zuschlägt, wird umgangen
    oder ausgebaut – und schützt dann gar nichts mehr.
    """
    found: list[str] = []
    for needle in needles:
        body = r"\s*".join(re.escape(part) for part in needle.split())
        pattern = r"\b" + body + ("" if prefix else r"\b")
        if re.search(pattern, haystack):
            found.append(needle)
    return found


def _minor_ages(text: str) -> list[str]:
    """Altersangaben unter 18 aus dem Rohtext ziehen."""
    hits: list[str] = []
    for pattern in _AGE_PATTERNS:
        for match in pattern.finditer(text.lower()):
            try:
                age = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if age < ADULT_AGE:
                hits.append(match.group(0).strip())
    return hits


def inspect(prompt: str, negative: str = "") -> tuple[bool, str]:
    """Text prüfen. Rückgabe: (erlaubt, Begründung bei Ablehnung).

    Geprüft wird auch der Negativ-Prompt: dort steht dasselbe Vokabular,
    und eine Sperre, die man mit Umhängen ins Negativfeld umgeht, wäre
    keine.
    """
    raw = f"{prompt} {negative}"
    text = _normalize(raw)

    hard = _contains(text, ALWAYS_BLOCKED)
    if hard:
        return False, (
            "Der Auftrag wurde abgelehnt: Der Prompt enthält Begriffe, die "
            f"Minderjährige bezeichnen ({', '.join(sorted(set(hard))[:4])}). "
            "Sexualisierte Darstellungen Minderjähriger sind strafbar "
            "(§ 184b StGB) – auch computererzeugt. Diese Sperre lässt sich "
            "nicht abschalten."
        )

    minor_hits = (
        _contains(text, MINOR_TERMS)
        + _contains(text, MINOR_PREFIXES, prefix=True)
        + _minor_ages(raw)
    )
    if not minor_hits:
        return True, ""
    sexual_hits = _contains(text, SEXUAL_TERMS) + _contains(
        text, SEXUAL_PREFIXES, prefix=True
    )
    if not sexual_hits:
        # Kinder in harmlosen Zusammenhängen bleiben erlaubt.
        return True, ""

    return False, (
        "Der Auftrag wurde abgelehnt: Der Prompt verbindet Begriffe für "
        f"Minderjährige ({', '.join(sorted(set(minor_hits))[:3])}) mit "
        f"sexuellen Begriffen ({', '.join(sorted(set(sexual_hits))[:3])}). "
        "Solche Darstellungen sind strafbar (§ 184b StGB), auch wenn sie "
        "vollständig computererzeugt sind. Diese Sperre lässt sich nicht "
        "abschalten. Für Erwachsenendarstellungen die Altersbegriffe "
        "entfernen und eindeutig erwachsene Motive beschreiben."
    )


def enforce(prompt: str, negative: str = "") -> None:
    """Wie ``inspect``, wirft aber. Vor jedem Modell-Laden aufrufen."""
    allowed, reason = inspect(prompt, negative)
    if not allowed:
        raise BlockedContent(reason)


def with_protective_negative(negative: str, active: bool = True) -> str:
    """Schutzbegriffe an den Negativ-Prompt hängen (ohne Doppelungen)."""
    if not active:
        return negative
    existing = {part.strip().lower() for part in negative.split(",") if part.strip()}
    additions = [
        term.strip() for term in PROTECTIVE_NEGATIVE.split(",")
        if term.strip().lower() not in existing
    ]
    if not additions:
        return negative
    return ", ".join(([negative.strip()] if negative.strip() else []) + additions)


def describe() -> str:
    """Kurzer Zustandsbericht für Diagnose und Oberfläche."""
    return (
        "Inhaltssperre aktiv: sexualisierte Darstellungen Minderjähriger "
        "werden vor dem Laden des Modells abgelehnt (Prompt und "
        "Negativ-Prompt). Nicht abschaltbar.\n"
        f"Begriffe: {len(ALWAYS_BLOCKED)} gesperrt, "
        f"{len(MINOR_TERMS) + len(MINOR_PREFIXES)} Altersbegriffe, "
        f"{len(SEXUAL_TERMS) + len(SEXUAL_PREFIXES)} Sexualbegriffe, dazu "
        "Altersangaben unter 18."
    )
