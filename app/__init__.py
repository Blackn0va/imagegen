"""StreamForge Studio – lokale Generativ-Anwendung (Bild, Video, Stimme).

Dieses Paket enthält bewusst KEINE schweren Imports auf Modulebene.
Grund: torch/diffusers/onnxruntime dürfen erst geladen werden, nachdem
``app.accel.prepare_gpu_dll_path()`` den DLL-Suchpfad gesetzt hat.
"""

from __future__ import annotations

# Version wird auch vom Build-Skript und der GUI-Titelzeile gelesen.
__version__ = "0.1.0"
__app_name__ = "StreamForge"
__app_display_name__ = "StreamForge Studio"

__all__ = ["__version__", "__app_name__", "__app_display_name__"]
