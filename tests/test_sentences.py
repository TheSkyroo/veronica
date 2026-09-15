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
