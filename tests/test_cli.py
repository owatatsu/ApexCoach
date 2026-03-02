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
        llm_enable=False,
        llm_provider=None,
        llm_model=None,
        llm_base_url=None,
        llm_review_output=None,
    )
    _apply_cli_overrides(cfg, args)
    assert cfg.overlay.show_window is True


def test_llm_cli_overrides() -> None:
    cfg = ApexCoachConfig()
    args = SimpleNamespace(
        realtime=False,
        show_window=False,
        disable_overlay=False,
        video=None,
        telemetry=None,
        output_video=None,
        log=None,
        monitor=None,
        region=None,
        duration=None,
        llm_enable=True,
        llm_provider="lmstudio",
        llm_model="qwen2.5-14b-instruct-q4_k_m.gguf",
        llm_base_url="http://127.0.0.1:1234",
        llm_review_output="logs/custom_review.md",
    )
    _apply_cli_overrides(cfg, args)
    assert cfg.llm.enabled is True
    assert cfg.llm.provider == "lmstudio"
    assert cfg.llm.model == "qwen2.5-14b-instruct-q4_k_m.gguf"
    assert cfg.llm.base_url == "http://127.0.0.1:1234"
    assert cfg.llm.offline_review_output == "logs/custom_review.md"
