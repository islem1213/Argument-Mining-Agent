from argmine.segmenter import segment, sentences


def test_offsets_index_the_source(sample):
    for seg in segment(sample):
        assert sample[seg.start:seg.end] == seg.text


def test_markers_are_kept_and_stripped_from_spans(sample):
    segs = segment(sample)
    by_marker = {s.marker: s for s in segs if s.marker}
    assert by_marker["so"].text == "later starts improve learning."
    assert by_marker["however"].text.startswith("critics")
    assert by_marker["but"].sentence_initial is False
    assert by_marker["however"].sentence_initial is True


def test_abbreviations_and_decimals_do_not_split_sentences():
    text = "Costs rose, e.g. fuel went up 3.5 percent. Dr. Smith disagrees."
    assert len(sentences(text)) == 2


def test_short_fragments_are_not_split():
    assert len(segment("Yes but it works.")) == 1
