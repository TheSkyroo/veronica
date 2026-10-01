from veronica.brain.sentences import SentenceSplitter


def test_splits_on_terminal_punctuation():
    s = SentenceSplitter()
    assert s.feed("Hello there. How are") == ["Hello there."]
    assert s.feed(" you? Fine!") == ["How are you?", "Fine!"]


def test_holds_incomplete_tail():
    s = SentenceSplitter()
    assert s.feed("It is 3 p") == []
    assert s.feed(".m. now") == []          # "3 p.m." not split: no space after period
    assert s.flush() == ["It is 3 p.m. now"]


def test_flush_empty():
    s = SentenceSplitter()
    assert s.flush() == []


def test_strips_markdown_noise():
    s = SentenceSplitter()
    assert s.feed("**Bold** and `code`. ") == ["Bold and code."]


def test_does_not_split_decimal():
    s = SentenceSplitter()
    assert s.feed("Pi is 3.14 roughly. Ok.") == ["Pi is 3.14 roughly.", "Ok."]


def test_devanagari_danda_and_next_char():
    s = SentenceSplitter()
    out = s.feed("कल तीन बजे मीटिंग है। उसके बाद lunch है। ")
    assert out == ["कल तीन बजे मीटिंग है।", "उसके बाद lunch है।"]
    s = SentenceSplitter()
    assert s.feed("Kal 3 baje meeting hai. 4 baje free ho. ") == ["Kal 3 baje meeting hai.", "4 baje free ho."]
    s = SentenceSplitter()
    assert s.feed("ठीक है") == [] and s.flush() == ["ठीक है"]


def test_has_devanagari_lives_in_sentences_and_is_reexported():
    from veronica.brain.sentences import has_devanagari
    assert has_devanagari("कल मीटिंग है") and has_devanagari("kal मीटिंग")
    assert not has_devanagari("kal meeting hai") and not has_devanagari("")
    # tts re-exports it for compatibility; the orchestrator must not need
    # kokoro_onnx just for a script check.
    from veronica.speech import tts
    assert tts.has_devanagari is has_devanagari
    import ast
    import pathlib
    src = pathlib.Path("veronica/orchestrator.py").read_text()
    imports = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)]
    assert not any(n.module == "veronica.speech.tts" for n in imports)
