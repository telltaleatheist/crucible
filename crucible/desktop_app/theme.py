from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

THEME_ENV = "CRUCIBLE_APP_THEME"

DEFAULTS_SECONDS = 2


@dataclass(frozen=True)
class Palette:
    name: str
    bg: str
    sidebar: str
    text: str
    muted: str
    line: str
    card: str
    selected: str
    primary: str
    primary_text: str
    secondary: str
    secondary_text: str
    ember: str
    track: str
    ok: str
    warn: str
    bad: str
    notice_bg: str
    entry: str


LIGHT = Palette(
    name="light", bg="#ffffff", sidebar="#f6f6f6", text="#111111", muted="#6e6e73",
    line="#ececec", card="#fafafa", selected="#e9e9eb", primary="#111111", primary_text="#ffffff",
    secondary="#efeff1", secondary_text="#111111", ember="#d66a30", track="#ececec",
    ok="#1f8a3b", warn="#b25e00", bad="#c62828", notice_bg="#fff4f0", entry="#ffffff",
)

DARK = Palette(
    name="dark", bg="#1b1b1c", sidebar="#141415", text="#f2f2f2", muted="#9b9ba1",
    line="#2c2c2e", card="#222224", selected="#2c2c2e", primary="#f2f2f2", primary_text="#111111",
    secondary="#2c2c2e", secondary_text="#f2f2f2", ember="#e07a42", track="#2f2f31",
    ok="#4cc26a", warn="#e0a045", bad="#ff6b6b", notice_bg="#33201a", entry="#262628",
)

TONES = ("ok", "warn", "bad")


def tone_colour(palette: Palette, tone: str) -> str:
    return getattr(palette, tone) if tone in TONES else palette.muted


def windows_dark() -> bool:
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        return value == 0
    except OSError:
        return False


def mac_dark() -> bool:
    try:
        said = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"],
                              capture_output=True, text=True, timeout=DEFAULTS_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return False
    return said.stdout.strip().lower() == "dark"


def os_is_dark(platform: str = sys.platform) -> bool:
    if platform == "win32":
        return windows_dark()
    if platform == "darwin":
        return mac_dark()
    return False


def current(platform: str = sys.platform, env: dict | None = None) -> Palette:
    wanted = (env if env is not None else os.environ).get(THEME_ENV, "").lower()
    if wanted in ("light", "dark"):
        return DARK if wanted == "dark" else LIGHT
    return DARK if os_is_dark(platform) else LIGHT
