from pathlib import Path

from tools.gen_interfaces import OUT, build


def test_interface_doc_is_up_to_date():
    """schema.py 를 고쳤으면 python tools/gen_interfaces.py 로 다이어그램을 다시 만든다."""
    assert Path(OUT).read_text(encoding="utf-8") == build()
