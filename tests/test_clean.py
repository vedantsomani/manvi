from clean import clean_text


def test_removes_stage_directions_and_urls():
    raw = "Thank you. (Applause.) Visit https://example.com now [laughter] please."
    assert clean_text(raw) == "Thank you. Visit now please."


def test_normalises_quotes_whitespace_and_italics():
    raw = "“It’s _not true_,”  he\n\n said."
    assert clean_text(raw) == "\"It's not true,\" he said."


def test_removes_dates_and_html():
    assert clean_text("<p>On March 12, 2019 we met.</p>") == "On we met."
