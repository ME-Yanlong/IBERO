"""源码在同一 Python 进程改变时，回放身份必须随之改变。"""

from ibero.core import reproducibility


def test_simulation_source_hash_reads_current_file_contents(tmp_path, monkeypatch):
    source = tmp_path / "ibero" / "core" / "reproducibility.py"
    source.parent.mkdir(parents=True)
    source.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setattr(reproducibility, "__file__", str(source))

    before = reproducibility.simulation_source_hash()
    source.write_text("version = 2\n", encoding="utf-8")
    after = reproducibility.simulation_source_hash()

    assert before != after
