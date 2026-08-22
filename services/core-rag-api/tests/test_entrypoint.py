import importlib

from rag_core import __main__


def test_console_entrypoint_targets_an_installed_package_module(monkeypatch) -> None:
    captured = {}

    def run(target, **kwargs):
        captured.update(target=target, **kwargs)

    monkeypatch.setattr(__main__.uvicorn, "run", run)
    __main__.main()
    assert captured["target"] == "rag_core.app:app"
    module_name, attribute = captured["target"].split(":", 1)
    assert getattr(importlib.import_module(module_name), attribute) is not None
