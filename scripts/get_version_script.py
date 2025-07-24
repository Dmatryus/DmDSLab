#!/usr/bin/env python3
"""
Простой скрипт для получения версии без импорта зависимостей.
"""

import re
import sys
from pathlib import Path


def get_version():
    """Извлечь версию из dmdslab/__init__.py."""
    init_file = Path(__file__).parent.parent / "dmdslab" / "__init__.py"
    
    if not init_file.exists():
        raise FileNotFoundError(f"File not found: {init_file}")
    
    content = init_file.read_text(encoding='utf-8')
    
    # Поиск строки __version__ = "..."
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Could not find __version__ in __init__.py")
    
    return match.group(1)


if __name__ == "__main__":
    try:
        version = get_version()
        print(version)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)