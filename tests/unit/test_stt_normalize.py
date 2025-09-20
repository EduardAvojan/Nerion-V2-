from app.chat.voice_io import _normalize_vocab


def test_stt_normalize_self_approved_to_self_improve():
    assert _normalize_vocab("please self approved the plan")
    out = _normalize_vocab("details on self approved")
    assert "self improve" in out.lower()

