"""WS4 simple-constraint review scorer v1. No trimming or case normalization."""
import re
VERSION = "ws4-simple-constraints-1"
def score(text, check):
    if not isinstance(text,str): raise TypeError("Response must be text")
    kind=check["kind"]
    if kind=="start_word":
        word=check["value"]
        passed=text.startswith(word) and (len(text)==len(word) or not (text[len(word)].isalnum() or text[len(word)]=="_"))
    elif kind=="keyword":
        passed=re.search(r"(?<!\w)"+re.escape(check["value"])+r"(?!\w)",text) is not None
    elif kind=="lowercase": passed=text.islower()
    elif kind=="word_range":
        count=len(text.split());passed=check["min"]<=count<=check["max"]
    elif kind=="end_phrase": passed=text.endswith(check["value"])
    else: raise ValueError("Unknown check: "+kind)
    strict=bool(text) and bool(passed)
    return {"strict_pass":strict,"constraint_fraction":float(strict)}
