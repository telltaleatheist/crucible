"""THE tags no engine ever SPEAKS, as ONE pattern.

Moved from narrator@dc05e7e7 `engine/orpheus/text.py` (ported there from
ebook2audiobook@9daab0ba `lib/conf_models.py`), PATTERN UNCHANGED. Higgs's
`clean_text` (`engine/higgs/prompt.py`) strips with it; the Orpheus module it
used to live in did not come along.

Timing tags ([break]/[pause]) are realized as silence before this point and
[heading]/[item] are pure markup, so by the time text reaches a model none of
them is speech. The optional '/' matches a closing form; none of these tags is
paired today, but stripping '[/x]' is never wrong for a tag that is never spoken.
"""
import re

SML_UNSPOKEN_PATTERN = re.compile(
    r'\[/?(?:break|pause|heading|item|music|sfx|silence)(?::[^\]]+)?\]',
    re.IGNORECASE
)
