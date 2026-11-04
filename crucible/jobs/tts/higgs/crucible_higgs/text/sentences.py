"""THE SENTENCE SPLITTER the Higgs split ladder cuts on (`engine/higgs/truncation.py`
`split_halves`): narrator's `text/paragraph_packer.split_sentences` and the three
tables it reads, and nothing else of the text layer.

Moved from narrator@dc05e7e7 (`text/paragraph_packer.py` `_hard_sentence_pattern`
and `split_sentences`, `text/lang.py` `punctuation_split_hard`,
`abbreviations_mapping['eng']` and `SENTENCE_ABBREVIATIONS`), BODIES UNCHANGED, so a
split half is cut exactly where it was cut when narrator owned the ladder. The rest
of narrator's text layer is book PREP and stays in BookForge; this module imports
no data file, which is why it is not the packer module itself (the packer loads
`caps_acronyms.json` at import, and a render once died on that file two days in).
"""
from __future__ import annotations

import regex as re

punctuation_split_hard = [
    # Western
    '.', '!', '?', '…', '‽', '—',
    # Arabic-Persian
    '؟',
    # CJK
    '。', '！', '？',
    # Indic
    '।', '॥',
    # Ethiopic
    '።', '፧',
    # Tibetan
    '།',
    # Khmer
    '។', '៕',
]
punctuation_split_hard_set = set(punctuation_split_hard)


abbreviations_mapping = {
    'eng': {
        'Mr.': 'Mister',
        'Mrs.': 'Missus',
        'Dr.': 'Doctor',
        'St.': 'Saint',
        'Jr.': 'Junior',
        'Sr.': 'Senior',
        'Prof.': 'Professor',
        'Capt.': 'Captain',
        'Ave.': 'Avenue',
        'Blvd.': 'Boulevard',
        'Rd.': 'Road',
        'Mt.': 'Mount',
        'etc.': 'et cetera',
        'vs.': 'versus',
        'e.g.': 'for example',
        'i.e.': 'that is',
        'et al.': 'and others',
    },
}


#: NARRATOR'S OWN sentence-break guard - NOT part of the e2a copy above.
#:
#: `abbreviations_mapping` is carried whole and byte-checkable against
#: ebook2audiobook@9daab0ba, so nothing may be added to it. This set exists
#: BESIDE it for the one thing narrator needs and e2a's table does not carry:
#: more abbreviations whose dot must not end a subtitle cue. It is read for its
#: MEMBERS ONLY - there are no expansions here, and there is nothing to expand,
#: because narrator renders the book's text as printed. What a member changes is
#: where a VTT cue breaks; it changes nothing that is spoken, because the
#: splitter is text-preserving and the pieces rejoin to the book's own words.
#: (Same reach as the e2a keys it sits beside: PASS 1 is also what an OVER-BUDGET
#: paragraph is cut on, so a guarded dot can move a chunk edge by a sentence
#: there - the same trade the 'Mr. Darcy' guard has always made.)
#:
#: Owen, 2026-09-09: "'No. 1' becomes two sentences, the first ending on 'no.'
#: and the second starting at '1' ... another example: 'Col. 2:1' - a bibleverse.
#: it reads 'colossians chapter 2 verse 1.' but the VTT reads 'Col.' ' 2:1.'"
#:
#: 'No.' IS DELIBERATELY ABSENT, and must stay absent. It is also an ordinary
#: sentence-final English word, so guarding it would weld `"No." Then he left.`
#: into one cue - a real regression in fiction, traded for a case the pattern
#: builders' digit rule already fixes precisely. Any token that doubles as an
#: ordinary sentence-final word belongs out here, not in.
#:
#: CASE MATTERS, and that is what makes the risky-looking members safe. The
#: guard the pattern builders emit is `(?<!\bStem)\.` in a pattern compiled with
#: re.DOTALL and nothing else - no re.IGNORECASE - so 'Ch.' guards 'Ch.' and not
#: 'much.', 'Sec.' not 'sec.', 'Fr.' not 'fr.'. What survives is the narrow
#: collision of a sentence ending in that exact CAPITALIZED word ('...to Sam.',
#: '...the Sun.'): those cues merge with the next. Cosmetic, rare, and the price
#: of catching the citations. 'Ezra' is left out for the same reason 'No.' is -
#: it is not abbreviated (it carries no dot), so guarding it could only suppress
#: a genuine break, and 'Ezra 7:10' never had a dot to break at.
SENTENCE_ABBREVIATIONS = frozenset({
    # Bible books, the case Owen hit ('Col. 2:1').
    'Gen.', 'Ex.', 'Exod.', 'Lev.', 'Num.', 'Deut.', 'Josh.', 'Judg.',
    'Sam.', 'Kgs.', 'Chron.', 'Neh.', 'Esth.', 'Ps.', 'Pss.', 'Prov.',
    'Eccl.', 'Isa.', 'Jer.', 'Lam.', 'Ezek.', 'Dan.', 'Hos.', 'Obad.',
    'Mic.', 'Nah.', 'Hab.', 'Zeph.', 'Hag.', 'Zech.', 'Mal.', 'Matt.',
    'Mk.', 'Lk.', 'Jn.', 'Rom.', 'Cor.', 'Gal.', 'Eph.', 'Phil.', 'Col.',
    'Thess.', 'Tim.', 'Tit.', 'Philem.', 'Heb.', 'Jas.', 'Pet.', 'Rev.',
    # Publishing apparatus and titles.
    'Vol.', 'Vols.', 'Ch.', 'Chap.', 'Fig.', 'Figs.', 'Sec.', 'Ed.',
    'Eds.', 'Trans.', 'cf.', 'viz.', 'approx.', 'Inc.', 'Ltd.', 'Co.',
    'Corp.', 'Univ.', 'Dept.', 'Gov.', 'Sen.', 'Rep.', 'Pres.', 'Sgt.',
    'Lt.', 'Maj.', 'Adm.', 'Fr.', 'Hon.', 'Msgr.', 'Esq.',
    # Months and days.
    'Jan.', 'Feb.', 'Mar.', 'Apr.', 'Jun.', 'Jul.', 'Aug.', 'Sep.',
    'Sept.', 'Oct.', 'Nov.', 'Dec.', 'Mon.', 'Tue.', 'Tues.', 'Wed.',
    'Thu.', 'Thur.', 'Thurs.', 'Fri.', 'Sat.', 'Sun.',
})


def _hard_sentence_pattern():
    """PASS 1's pattern, minus the escaped-token terminator.

    `text/packer.py` builds this inside `get_sentences` and may not be edited to
    share it, so this is a deliberate second copy. It reads the SAME tables -
    `abbreviations_mapping['eng']`, `SENTENCE_ABBREVIATIONS` and
    `punctuation_split_hard_set` from `text/lang.py` - so the two cannot drift on
    their inputs, and `tests/test_text_paragraph_packer.py` asserts they agree on
    behaviour.

    The `tok_class` terminator is dropped because a block's text carries no
    escaped SML: markers are added by the packer AFTER splitting, on a chunk's
    leading edge.
    """
    stems = set()
    for k in list(abbreviations_mapping.get('eng', {})) + list(SENTENCE_ABBREVIATIONS):
        stem = (k[:-1] if k.endswith('.') else k).split('.')[-1].strip()
        if len(stem) >= 2:
            stems.add(stem)
    guards = ''.join(f'(?<!\\b{re.escape(s)})' for s in sorted(stems))
    # THE DIGIT RULE (Owen, 2026-09-09). A dot followed by whitespace and then a
    # DIGIT never ends a sentence: 'Col. 2:1' and 'No. 1' were breaking into two
    # cues mid-citation, and no abbreviation list can be complete enough to catch
    # every 'No.'-shaped token - a real sentence essentially never starts with a
    # bare digit in this corpus, so the shape is the better test. It lives in the
    # pattern, not in a post-pass, so `split_sentences` stays text-preserving and
    # `packer.get_sentences` gets it from the same construction.
    #
    # KNOWN COST: '...in 1994. 1995 was worse.' now shares one cue. Rarer than
    # the citation case, and cosmetic either way - the audio is already rendered
    # by the time a cue list is built.
    # THE ELLIPSIS RULE (Owen, 2026-09-21). A dot that is part of an ellipsis
    # run - '...', the typeset '. . .', or the one-character '…' - is not a full
    # stop. Before this rule every dot followed by whitespace ended a sentence,
    # so a book's "'. . . a tic.'" arrived in the transcript as four cues:
    # "'." / "." / "." / "a tic.'" - one per dot, each aligned to nothing.
    #
    # An ellipsis DOES end a sentence when a sentence follows it: whitespace and
    # then a capital (with at most one opening quote or bracket in front), as in
    # "He did not answer... Then he did." Followed by a lowercase word it is a
    # pause inside the sentence and the cue runs on. The single-dot rule below
    # excludes a dot that has another dot beside it (one optional space away) on
    # either side, so the run is only ever judged as a whole by this branch.
    closing_run = r'["\'’”»)\]]*'
    # 'I' is the one capital that is not a sentence signal: "Well . . . I
    # suppose so." is one sentence. Known cost: "He stopped... I left." shares
    # a cue - the words are right, the cut is one boundary late.
    ellipsis = (rf'(?:…|\.(?:\s?\.){{2,}}){closing_run}'
                rf'(?=\s+["“‘\'(\[]?(?!I\b)[A-Z])')
    not_in_a_run = r'(?<!\.)(?<!\.\s)'   # not the 2nd or 3rd dot of a run
    not_starting_a_run = r'(?!\s?\.)'     # not the 1st or 2nd dot of a run
    guarded_dot = (rf'(?<!\b[A-Za-z]){guards}{not_in_a_run}\.{not_starting_a_run}'
                   rf'(?!\s+\d){closing_run}(?=\s|$)')
    others = [f'{re.escape(p)}{closing_run}(?=\\s|$)'
              for p in punctuation_split_hard_set if p not in ('.', '…')]
    return re.compile('|'.join([ellipsis, guarded_dot] + others), re.DOTALL)


_HARD_SENTENCE_RE = _hard_sentence_pattern()


def split_sentences(text: str) -> list:
    """One paragraph -> its sentences, at PASS 1's boundaries.

    Never splits inside an abbreviation ('Mr. Darcy' is one sentence) and always
    splits AFTER a closing quote ('"Are you sure?" she said.' ends at the quote,
    not at the '?'). Text-preserving: the pieces rejoin to the input's own words.
    """
    pieces, last = [], 0
    for m in _HARD_SENTENCE_RE.finditer(text):
        piece = text[last:m.end()].strip()
        if piece:
            pieces.append(piece)
        last = m.end()
    tail = text[last:].strip()
    if tail:
        pieces.append(tail)
    return pieces or ([text.strip()] if text.strip() else [])
