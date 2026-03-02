import argparse
from types import SimpleNamespace

from apexcoach.cli import _apply_cli_overrides, _parse_region
from apexcoach.config import ApexCoachConfig


def test_parse_region_ok() -> None:
    assert _parse_region("10,20,300,400") == (10, 20, 300, 400)


def test_parse_region_invalid() -> None:
    try:
        _parse_region("10,20,300")
        assert False, "expected ArgumentTypeError"
    except argparse.ArgumentTypeError:
        pass


def test_realtime_defaults_show_window() -> None:
    cfg = ApexCoachConfig()
    args = SimpleNamespace(
        realtime=True,
        show_window=False,
        disable_overlay=False,
        video=None,
        telemetry=None,
        output_video=None,
        log=None,
        monitor=None,
        region=None,
        duration=None,
    )
    _apply_cli_overrides(cfg, args)
    assert cfg.overlay.show_window is True
