import re

# -------------------------
# Number patterns (≤ 2 digits per segment)
# -------------------------
ROMAN_UPPER = r'M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})'
ROMAN_LOWER = r'm{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})'

NUMERIC = r'[1-9][0-9]?'                     # 1–99
NUMERIC_MULTI = r'[1-9][0-9]?(?:[.\-][1-9][0-9]?)+'

ALPHA_UPPER = r'[A-Z]'
ALPHA_LOWER = r'[a-z]'

# e.g. 1A, 12b
NUMERIC_SUFFIX = r'[1-9][0-9]?[A-Za-z]'

NUM = rf'(?:{NUMERIC_MULTI}|{NUMERIC_SUFFIX}|{NUMERIC}|{ALPHA_UPPER}|{ALPHA_LOWER}|{ROMAN_UPPER}|{ROMAN_LOWER})'

BULLET = r'[・\-–—*+•‣⁃○∙◦⦾⦿\u2022\u2023\u25E6\u2043\u2219\u25CB\u25CF\uE000-\uF8FF]'

# -------------------------
# Prefix keywords
# -------------------------
PREFIXES = [
    "item", "section", "article", "ariticle",
    "note", "part", "chapter"
]
PREFIX_PATTERN = r'(?:' + '|'.join(PREFIXES) + r')'

# -------------------------
# Patterns
# -------------------------
PATTERNS = [
    # Bullets (including -, –, —)
    rf'^[ \t]*{BULLET}+[ \t]+',

    # "Item 1.2", "Section 2.1", "Note II"
    rf'^[ \t]*{PREFIX_PATTERN}[ \t]+{NUM}[ \t]*[.:]?[ \t]*',

    # "(1.2)", "(a)"
    rf'^[ \t]*\([ \t]*{NUM}[ \t]*\)[ \t]*',

    # "1.2.", "1.2)", "A.", "I)"
    rf'^[ \t]*{NUM}[ \t]*[\.\)][ \t]*',

    # "1.2 " (no punctuation)
    rf'^[ \t]*{NUM}[ \t]+',

    # "§ 1.2"
    rf'^[ \t]*§[ \t]*{NUM}[ \t]*',
]

COMPILED = [re.compile(p, re.IGNORECASE) for p in PATTERNS]


def strip_leading_numbering(text: str) -> str:
    """
    Remove leading numbering / prefixes with ≤2-digit constraint per level.
    Iteratively removes stacked prefixes.
    """
    text = text.strip()

    while True:
        for pattern in COMPILED:
            new_text = pattern.sub('', text, count=1)
            if new_text != text:
                text = new_text.strip()
                break
        else:
            break

    text = text.capitalize()
    
    return text





# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    examples = [
        # contract
        "1.  I hereby acknowledge that I have received a security indoctrination concerning ...",
        "9. (a) With respect to SSI and SBU,",
        "5.  (a) For PCII - (1) Upon the completion of my engagement as an employee, consult...",
        "5.3. is rightfully acquired from others who did not obtain it under obligation of confidentiality;",
        "b) the terms of this Agreement;",
        "A.  The Parties intend to enter into discussions relating to",
        # finance
        "PART I",
        "Item 1. Busines s.",
        "Item 1A. Risk Factors",
        "Item 4. Mine Safety Disclosures.",
        "2018 Restructuring Actions:",
        "NOTE 16. Commitments and Contingencies",
        "ARTICLE 4",
        "Article I. Award of DSUs and dividend equivalents; SETTLEMENT",
        "NOTE 2 — EARNINGS PER SHARE",
        "NOTE 2 - EARNINGS PER SHARE"
        # qasper
        "4 Learning from Matching Features",
        "4.2 CNN Topology",
        "3. CONFIDENTIALITY OBLIGATIONS"
        # "1.2 Risk Factors",
        # "1.2.3 Risk Factors",
        # "(1.2) Risk Factors",
        # "1.2) Risk Factors",
        # "1.2. Risk Factors",
        # "Item 1.2 Risk Factors",
        # "SECTION 2.1.3 Overview",
        # "Part II",
        # "II.",
        # "1.2.3.4 Deep Section",
        # "Section 1.2.3 Item 4A. Risk Factors",  # stacked prefixes
    ]

    for e in examples:
        print(f"{e!r} -> {strip_leading_numbering(e)!r}")