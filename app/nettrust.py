"""TLS-Vertrauensanker.

Zwei Fälle, die im Feld beide vorkommen:

  * TLS-prüfende Virenscanner und Firmen-Proxys ersetzen Zertifikate durch
    eigene, die nur im Windows-Zertifikatspeicher liegen. ``certifi`` kennt
    sie nicht -> jeder Download bricht ab. Lösung: ``truststore``, das den
    Systemspeicher benutzt.
  * Manche Systeme haben einen kaputten oder leeren Systemspeicher. Dann
    hilft das mitgelieferte ``certifi`` als Rückfallebene.

Reihenfolge: truststore -> certifi -> nichts (mit Klartext-Meldung).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)

_installed = False
_result: TrustResult | None = None


@dataclass(frozen=True)
class TrustResult:
    mechanism: str  # truststore | certifi | none
    detail: str
    ca_bundle: str = ""

    def label(self) -> str:
        return {
            "truststore": "Windows-Zertifikatspeicher (truststore)",
            "certifi": "mitgeliefertes certifi-Bündel",
            "none": "Python-Vorgabe",
        }.get(self.mechanism, self.mechanism)


def _try_truststore() -> TrustResult | None:
    try:
        import truststore
    except ImportError:
        return None
    try:
        truststore.inject_into_ssl()
    except Exception as exc:
        log.debug("truststore.inject_into_ssl fehlgeschlagen: %s", exc)
        return None
    return TrustResult("truststore", "Systemzertifikate werden verwendet.")


def _try_certifi() -> TrustResult | None:
    try:
        import certifi
    except ImportError:
        return None
    try:
        bundle = certifi.where()
    except Exception as exc:
        log.debug("certifi.where fehlgeschlagen: %s", exc)
        return None
    if not os.path.isfile(bundle):
        return None
    # requests/urllib3/huggingface_hub lesen diese Variablen.
    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        os.environ.setdefault(key, bundle)
    return TrustResult("certifi", "Mitgeliefertes Zertifikatsbündel wird verwendet.", bundle)


def install(prefer_system: bool = True) -> TrustResult:
    """Vertrauensanker einrichten. Idempotent, wirft nie."""
    global _installed, _result
    if _installed and _result is not None:
        return _result

    result: TrustResult | None = None
    if prefer_system:
        result = _try_truststore()
    if result is None:
        result = _try_certifi()
    if result is None:
        result = TrustResult(
            "none",
            "Weder truststore noch certifi vorhanden. Downloads können hinter "
            "TLS-prüfenden Virenscannern oder Firmen-Proxys fehlschlagen.",
        )
        log.warning(result.detail)

    _installed = True
    _result = result
    log.debug("TLS-Vertrauensanker: %s – %s", result.mechanism, result.detail)
    return result


def status() -> TrustResult:
    """Aktueller Zustand ohne erneutes Einrichten."""
    return _result or TrustResult("none", "Noch nicht eingerichtet.")
