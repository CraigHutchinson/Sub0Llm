"""System power state: timings on battery (or in a battery saver) are not comparable with a baseline recorded on mains.

CPU clocks, boost and the power plan all differ, so the gate defers instead of measuring (see gate.py), and a record or
baseline carries the state it was measured in. `unknown` (a desktop, a CI runner, a failed query) counts as `ac` for every
decision but is reported as unknown.

`SUB0PERF_FAKE_POWER=ac|battery|unknown|saver` replaces the query. It is for tests only (`saver` is mains with the saver on);
outside tests it is a warning, see `environment_notes`.
"""

import ctypes
import dataclasses
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

AC, BATTERY, UNKNOWN = "ac", "battery", "unknown"
FAKE_VAR = "SUB0PERF_FAKE_POWER"
OVERRIDE_VAR = "SUB0MATH_PERF_ON_BATTERY"   # measure anyway
STRICT_VAR = "SUB0MATH_PERF_POWER_STRICT"   # fail the push instead of deferring
OVERRIDE_TEXT = (f"{OVERRIDE_VAR}=1: measuring on battery; timings are not comparable with the mains baseline "
                 f"and cannot become one")
OVERRIDE_TEXT_THROTTLED = (f"{OVERRIDE_VAR}=1: measuring on a throttled machine; timings are not comparable with the baseline "
                           f"and cannot become one")
# The override and strict switches cover a throttled machine too (slower than its baseline's calibration, see speed.py); the
# name is kept because it is published.
THROTTLE_CAUSES = ("likely causes: a vendor power mode, a charging power cap or thermal limits; check the power mode, "
                   "let the machine cool, or measure under another power setting")
EFFICIENCY_MODE = "Best power efficiency"
MODE_FIX = "set Balanced or Best performance (Settings > System > Power & battery > Power mode)"
OVERRIDE_HINT = f"To measure anyway: set {OVERRIDE_VAR}=1. Results are flagged and can't become a baseline."
STRICT_HINT = f"To defer instead of failing: unset {STRICT_VAR} and set strict_power to false in .git/sub0perf/config.json."

_SCHEMES = {"381b4222-f694-41f0-9685-ff5bb260df2e": "Balanced", "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "High performance",
            "a1841308-3541-4fab-bc81-f71556f20b4a": "Power saver", "e9a42b02-d5df-448d-aa00-03f14749eb61": "Ultimate performance"}

# Windows power-mode overlays (the slider in Settings > Power & battery). The GUIDs are the overlay keys under
# HKLM\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes, whose FriendlyName resources read 961cc777 "Better Battery-life
# Overlay", 3af9b8d9 "High Performance Overlay" and ded574b5 "Max Performance Overlay" (the Windows 11 slider: Best power efficiency,
# Better performance, Best performance). No overlay value, or the all-zero GUID, is Balanced. Read only; nothing is ever changed.
_OVERLAYS = {"961cc777-2547-4f9d-8174-7d86181b8a7a": EFFICIENCY_MODE, "3af9b8d9-7c97-431d-ad78-34a8bfea439f": "Better performance",
             "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance", "00000000-0000-0000-0000-000000000000": "Balanced"}


@dataclass(frozen=True)
class PowerState:
    state: str = UNKNOWN        # ac, battery or unknown
    saver: bool = False         # battery saver / low-power mode, where the OS says so
    scheme: str = ""            # the active power plan, where cheap to get (reported, not judged)
    source: str = ""            # where the answer came from (windows, linux, macos, fake, none)
    speed_ratio: float = 0.0    # calibration now / the baseline's, where the speed check ran (0 = it did not)
    throttled: bool = False     # slower than the baseline's calibration by more than the limit (speed.py)
    sustained: bool = False     # the slow reading was the sustained one (after a warm load), not the burst
    mode_ac: str = ""           # Windows power-mode overlay on mains / on battery ("Balanced", "Best power efficiency"); reported, not judged
    mode_dc: str = ""

    @property
    def untrusted(self):
        """True when timings made now cannot be compared with a mains baseline."""
        return self.state == BATTERY or self.saver or self.throttled

    @property
    def only_throttled(self):
        return self.throttled and self.state != BATTERY and not self.saver

    @property
    def mode(self):
        """The power-mode overlay in effect for the current power state ('' = not known)."""
        return self.mode_dc if self.state == BATTERY else self.mode_ac

    def reason(self):
        if self.only_throttled:
            kind = "sustained calibration" if self.sustained else "calibration"
            return f"machine throttled ({kind} {self.speed_ratio:.2f}x slower than the baseline's)"
        return "battery saver is on" if self.saver and self.state != BATTERY else "on battery"

    def as_dict(self):
        return {"state": self.state, "saver": self.saver, **({"scheme": self.scheme} if self.scheme else {}),
                **({"speed_ratio": round(self.speed_ratio, 3)} if self.speed_ratio else {}),
                **({"throttled": True} if self.throttled else {}), **({"sustained": True} if self.sustained else {}),
                **({"mode": {"ac": self.mode_ac, "dc": self.mode_dc}} if self.mode_ac or self.mode_dc else {})}


def with_speed(state, ratio, limit, sustained=False):
    """`state` with the speed check's result: the ratio recorded, and throttled when it is over `limit`; `sustained` says the slowest
    reading was the one taken after the warm load. A ratio of None = not checked."""
    return dataclasses.replace(state, speed_ratio=ratio, throttled=ratio > limit, sustained=sustained) if ratio else state


def key(info):
    """What two records must share to be compared: (state with unknown as ac, saver, throttled). An absent field (an old baseline) is ac."""
    info = info or {}
    state = info.get("state", AC)
    return (BATTERY if state == BATTERY else AC), bool(info.get("saver", False)), bool(info.get("throttled", False))


def label(info):
    """A short text for a record's power dict: 'ac', 'battery', 'ac, battery saver', 'unknown'."""
    info = info or {}
    return info.get("state", AC) + (", battery saver" if info.get("saver") else "") + (", throttled" if info.get("throttled") else "")


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte), ("BatteryLifePercent", ctypes.c_ubyte),
                ("SystemStatusFlag", ctypes.c_ubyte), ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]


def _windows_scheme():
    """The active power plan's name, or '' (one cheap call into powrprof; no powercfg process)."""
    try:
        class Guid(ctypes.Structure):
            _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint16), ("c", ctypes.c_uint16), ("d", ctypes.c_ubyte * 8)]
        found = ctypes.POINTER(Guid)()
        if ctypes.windll.powrprof.PowerGetActiveScheme(None, ctypes.byref(found)) != 0:
            return ""
        g = found.contents
        text = (f"{g.a:08x}-{g.b:04x}-{g.c:04x}-{g.d[0]:02x}{g.d[1]:02x}-" + "".join(f"{x:02x}" for x in g.d[2:]))
        ctypes.windll.kernel32.LocalFree(found)
        return _SCHEMES.get(text, text)
    except (OSError, AttributeError, ValueError):
        return ""


def _windows_modes():
    """(mains, battery) power-mode overlay names from the registry, '' where unreadable. One cheap read, no process."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes") as key:
            names = []
            for value in ("ActiveOverlayAcPowerScheme", "ActiveOverlayDcPowerScheme"):
                try:
                    guid = str(winreg.QueryValueEx(key, value)[0]).strip("{}").lower()
                except FileNotFoundError:
                    guid = "00000000-0000-0000-0000-000000000000"  # no overlay set: Balanced
                names.append(_OVERLAYS.get(guid, guid))
            return tuple(names)
    except (OSError, ImportError, ValueError):
        return "", ""


def _windows():
    status = _SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return PowerState(source="windows")
    state = {0: BATTERY, 1: AC}.get(status.ACLineStatus, UNKNOWN)
    ac, dc = _windows_modes()
    return PowerState(state, status.SystemStatusFlag == 1, _windows_scheme(), "windows", mode_ac=ac, mode_dc=dc)


def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _linux(root="/sys/class/power_supply", profile="/sys/firmware/acpi/platform_profile"):
    """Mains online means ac; a battery with no mains online means battery; a machine with no battery is ac."""
    if not Path(root).is_dir():
        return PowerState(source="linux")
    kinds = {p: _read(p / "type") for p in sorted(Path(root).glob("*"))}
    batteries = [p for p, k in kinds.items() if k == "Battery"]
    mains = [p for p, k in kinds.items() if k in ("Mains", "USB", "USB_C", "USB_PD")]
    saver = _read(profile) == "low-power"
    if any(_read(p / "online") == "1" for p in mains) or not batteries:
        return PowerState(AC, saver, source="linux")
    return PowerState(BATTERY, saver, source="linux")


def _macos():
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
    state = BATTERY if "'Battery Power'" in batt else AC if "'AC Power'" in batt else UNKNOWN
    full = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=5).stdout
    saver = any(line.split()[:2] == ["lowpowermode", "1"] for line in full.splitlines())
    return PowerState(state, saver, source="macos")


def _fake(value):
    value = value.strip().lower()
    if value == "saver":
        return PowerState(AC, True, source="fake")
    return PowerState(value if value in (AC, BATTERY, UNKNOWN) else UNKNOWN, source="fake")


def power_state(environ=None):
    """The machine's power state now. Never raises: a failed query is `unknown`."""
    environ = os.environ if environ is None else environ
    if environ.get(FAKE_VAR):
        return _fake(environ[FAKE_VAR])
    try:
        if sys.platform == "win32":
            return _windows()
        if sys.platform == "darwin":
            return _macos()
        return _linux()
    except (OSError, subprocess.SubprocessError, AttributeError, ValueError):
        return PowerState(source="none")


def truthy(value):
    return str(value or "").strip().lower() not in ("", "0", "false", "no", "off")


@dataclass(frozen=True)
class Policy:
    override: bool = False  # SUB0MATH_PERF_ON_BATTERY: measure anyway
    strict: bool = False    # fail instead of deferring


def policy(environ=None, config=None):
    """The gate's power policy: the environment, with `strict_power` from the repository config as the default for strict."""
    environ = os.environ if environ is None else environ
    strict = environ.get(STRICT_VAR)
    return Policy(truthy(environ.get(OVERRIDE_VAR)),
                  truthy(strict) if strict not in (None, "") else bool((config or {}).get("strict_power")))


def override_text(state):
    """OVERRIDE_TEXT, worded for a throttled machine when that is the only thing overridden (a PowerState or a record's power dict)."""
    if isinstance(state, dict):
        state = PowerState(state.get("state", AC), bool(state.get("saver")), throttled=bool(state.get("throttled")))
    return OVERRIDE_TEXT_THROTTLED if state.only_throttled else OVERRIDE_TEXT


def environment_notes(state, pol, environ=None):
    """Lines that make the power switches impossible to miss: an override in effect, one set but ignored, strict mode, a fake."""
    environ = os.environ if environ is None else environ
    notes = []
    if pol.override:
        notes.append(f"sub0perf: {override_text(state)}" if state.untrusted
                     else f"sub0perf: {OVERRIDE_VAR}=1 is set but ignored (on mains); unset it so it is not left on by mistake")
    if pol.strict:
        notes.append(f"sub0perf: strict power mode ({STRICT_VAR} or strict_power): a push on battery is refused, not deferred")
    if environ.get(FAKE_VAR):
        notes.append(f"sub0perf: WARNING: {FAKE_VAR}={environ[FAKE_VAR]} is set: the power state is being faked "
                     f"(for tests only; measuring as {state.state}{', battery saver' if state.saver else ''})")
    return notes


def describe(state):
    """One line: 'ac (windows; Balanced)'."""
    modes = f"power mode {state.mode_ac} (mains) / {state.mode_dc} (battery)" if state.mode_ac or state.mode_dc else ""
    extra = "; ".join(x for x in (state.source, state.scheme, modes) if x)
    return (f"{state.state}{', battery saver' if state.saver else ''}{', throttled' if state.throttled else ''}"
            + (f" ({extra})" if extra else "")
            + (f", {'sustained ' if state.sustained else ''}calibration {state.speed_ratio:.2f}x the baseline's" if state.speed_ratio else ""))


def deferral_text(state, strict):
    """The message when the gate does not measure: why, what happens next, and how to override (or turn strict off)."""
    causes = f" ({THROTTLE_CAUSES})" if state.only_throttled else ""
    if state.only_throttled and state.mode == EFFICIENCY_MODE:  # the likeliest cause, named with its fix
        causes = f" (Windows power mode: {state.mode}; {MODE_FIX}. Other causes: a charging power cap or thermal limits)"
    if strict:
        return (f"sub0perf: {state.reason()}{causes}: performance not measured and the push is refused (strict power mode). "
                f"{OVERRIDE_HINT} {STRICT_HINT}")
    return (f"sub0perf: {state.reason()}{causes}: performance not measured (deferred; the next push on mains measures against "
            f"the unchanged baseline). {OVERRIDE_HINT}")
