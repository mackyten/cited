from cited.chunking import Chunk
from cited.config import ABSTAIN_SENTENCE
from cited.generate import extract_citations, format_prompt, is_abstention, validate_citations


def test_extract_handles_adjacent_and_lists() -> None:
    assert extract_citations("A [2][1] and [3, 4]; also [1].") == [2, 1, 3, 4]
    assert extract_citations("no cites, array[0] and [x]") == [0]  # numeric brackets are cites


def test_validate_flags_invented_numbers() -> None:
    c = validate_citations("Fact [1] and fact [7].", n_sources=3)
    assert c.valid == [1] and c.invented == [7] and not c.ok


def test_validate_needs_at_least_one_valid() -> None:
    assert not validate_citations("No citation here.", 3).ok
    assert validate_citations("Fine [2].", 3).ok
    assert validate_citations("Zero is invalid [0].", 3).invented == [0]


def test_abstention_detection_is_punctuation_and_case_insensitive() -> None:
    assert is_abstention(ABSTAIN_SENTENCE)
    assert is_abstention("i dont know based on the documentation")  # apostrophe dropped
    assert is_abstention("I don\u2019t know based on the documentation.")  # curly apostrophe
    assert not is_abstention("The docs say X [1].")


def test_prompt_numbers_sources_from_one() -> None:
    cs = [Chunk("a", "x.md", ("T", "S"), None, "alpha text", None),
          Chunk("b", "y.md", ("U",), None, "beta text", None)]
    p = format_prompt("Why?", cs)
    assert "[1] (x.md > T > S)\nalpha text" in p and "[2] (y.md > U)\nbeta text" in p
    assert p.rstrip().endswith("Answer:") and "Question: Why?" in p
