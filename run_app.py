"""Dünner Starter für PyInstaller.

Absichtlich klein und ohne schwere Imports: PyInstaller nimmt dieses Modul
als Einstiegspunkt, und der DLL-Suchpfad muss gesetzt sein, BEVOR
irgendetwas torch, diffusers oder onnxruntime anzieht.
"""

from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    # Ohne freeze_support() starten gebündelte Kindprozesse das Programm neu
    # (Windows spawn) – das ergibt endlos viele Fenster.
    multiprocessing.freeze_support()

    from app import accel

    accel.prepare_gpu_dll_path()

    from app.__main__ import main as app_main

    return app_main()


if __name__ == "__main__":
    sys.exit(main())
