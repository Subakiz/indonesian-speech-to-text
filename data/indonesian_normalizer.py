"""Indonesian Text Normalization Engine for Automatic Speech Recognition (ASR).

Comprehensive coverage:
- Spoken number expansion ('terbilang'): 0 to trillions with decimals
- Currency conversion ('Rp 50.000' -> 'lima puluh ribu rupiah')
- Percentage conversion ('25%' -> 'dua puluh lima persen')
- Ordinals ('ke-1' -> 'kesatu / pertama')
- Colloquial / Jakarta slang normalization (e.g. 'nggak' -> 'tidak', 'udah' -> 'sudah')
- Indonesian reduplication hyphens ('anak-anak', 'jalan-jalan')
- Common abbreviations & acronyms expansion
- Code-switching loanword affixes ('di-approve', 'meeting-nya')
"""

import re
from typing import Dict, Optional

SATUAN = [
    "",
    "satu",
    "dua",
    "tiga",
    "empat",
    "lima",
    "enam",
    "tujuh",
    "delapan",
    "sembilan",
    "sepuluh",
    "sebelas",
]

# Common abbreviations expansion dictionary
ABBREVIATIONS: Dict[str, str] = {
    "dll": "dan lain-lain",
    "dll.": "dan lain-lain",
    "dsb": "dan sebagainya",
    "dsb.": "dan sebagainya",
    "dst": "dan seterusnya",
    "dst.": "dan seterusnya",
    "dlsb": "dan lain sebagainya",
    "dkk": "dan kawan-kawan",
    "dkk.": "dan kawan-kawan",
    "yg": "yang",
    "dgn": "dengan",
    "utk": "untuk",
    "sdh": "sudah",
    "blm": "belum",
    "bgt": "banget",
    "tsb": "tersebut",
    "ttg": "tentang",
    "kpd": "kepada",
    "pd": "pada",
    "dr": "dari",
    "dpt": "dapat",
    "krn": "karena",
    "sbg": "sebagai",
    "thd": "terhadap",
    "pt": "perseroan terbatas",
    "pt.": "perseroan terbatas",
    "dr.": "dokter",
    "prof.": "profesor",
    "ir.": "insinyur",
    "tgl": "tanggal",
    "thn": "tahun",
    "bln": "bulan",
    "rp": "rupiah",
}

# Indonesian colloquial / slang mapping for ASR text alignment
SLANG_MAP: Dict[str, str] = {
    "nggak": "tidak",
    "gak": "tidak",
    "ngga": "tidak",
    "enggak": "tidak",
    "udah": "sudah",
    "udh": "sudah",
    "aja": "saja",
    "kalo": "kalau",
    "klo": "kalau",
    "gimana": "bagaimana",
    "bisa": "bisa",
    "capek": "lelah",
    "bener": "benar",
    "banget": "sangat",
    "bgt": "sangat",
    "kayak": "seperti",
    "kek": "seperti",
    "emang": "memang",
    "cuman": "hanya",
    "cuma": "hanya",
    "tapi": "tetapi",
    "tp": "tetapi",
    "gitu": "begitu",
    "gini": "begini",
    "tau": "tahu",
    "dapet": "dapat",
}


def number_to_words(n: int) -> str:
    """Convert an integer to Indonesian spoken words ('terbilang')."""
    if n < 0:
        return "minus " + number_to_words(abs(n))
    if n == 0:
        return "nol"
    if n < 12:
        return SATUAN[n]
    if n < 20:
        return SATUAN[n - 10] + " belas"
    if n < 100:
        remainder = n % 10
        return SATUAN[n // 10] + " puluh" + (" " + SATUAN[remainder] if remainder else "")
    if n < 200:
        remainder = n % 100
        return "seratus" + (" " + number_to_words(remainder) if remainder else "")
    if n < 1000:
        remainder = n % 100
        return SATUAN[n // 100] + " ratus" + (" " + number_to_words(remainder) if remainder else "")
    if n < 2000:
        remainder = n % 1000
        return "seribu" + (" " + number_to_words(remainder) if remainder else "")
    if n < 1_000_000:
        remainder = n % 1000
        return number_to_words(n // 1000) + " ribu" + (" " + number_to_words(remainder) if remainder else "")
    if n < 1_000_000_000:
        remainder = n % 1_000_000
        return number_to_words(n // 1_000_000) + " juta" + (" " + number_to_words(remainder) if remainder else "")
    if n < 1_000_000_000_000:
        remainder = n % 1_000_000_000
        return number_to_words(n // 1_000_000_000) + " miliar" + (" " + number_to_words(remainder) if remainder else "")
    if n < 1_000_000_000_000_000:
        remainder = n % 1_000_000_000_000
        return number_to_words(n // 1_000_000_000_000) + " triliun" + (" " + number_to_words(remainder) if remainder else "")

    return " ".join([SATUAN[int(d)] if int(d) > 0 else "nol" for d in str(n)])


def normalize_currency(text: str) -> str:
    """Expand Indonesian Rupiah expressions to spoken words."""
    def replace_rp(match):
        raw_val = match.group(1).replace(".", "").replace(",", "")
        try:
            num = int(raw_val)
            return number_to_words(num) + " rupiah"
        except ValueError:
            return match.group(0)

    pattern = r"(?:Rp\.?|IDR)\s*([0-9]+(?:[\.,][0-9]{3})*)"
    return re.sub(pattern, replace_rp, text, flags=re.IGNORECASE)


def normalize_percentage(text: str) -> str:
    """Expand percentages: '25%' -> 'dua puluh lima persen'."""
    def replace_pct(match):
        val = match.group(1).replace(".", "").replace(",", ".")
        try:
            if "." in val:
                parts = val.split(".")
                whole = number_to_words(int(parts[0]))
                decimals = " ".join([SATUAN[int(d)] if int(d) > 0 else "nol" for d in parts[1]])
                return f"{whole} koma {decimals} persen"
            num = int(val)
            return number_to_words(num) + " persen"
        except ValueError:
            return match.group(0)

    pattern = r"([0-9]+(?:[\.,][0-9]+)?)\s*%"
    return re.sub(pattern, replace_pct, text)


def normalize_ordinals(text: str) -> str:
    """Expand Indonesian ordinals: 'ke-1' -> 'kesatu', 'ke-2' -> 'kedua'."""
    def replace_ord(match):
        val = int(match.group(1))
        if val == 1:
            return "pertama"
        return "ke" + number_to_words(val)

    pattern = r"\bke\-([0-9]+)\b"
    return re.sub(pattern, replace_ord, text, flags=re.IGNORECASE)


def normalize_numbers_in_text(text: str) -> str:
    """Convert remaining standalone numbers to Indonesian spoken words."""
    def replace_num(match):
        raw = match.group(0)
        if "," in raw or "." in raw:
            parts = re.split(r"[\.,]", raw)
            if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
                try:
                    num = int("".join(parts))
                    return number_to_words(num)
                except ValueError:
                    pass
            if len(parts) == 2 and len(parts[1]) <= 2:
                try:
                    whole = number_to_words(int(parts[0]))
                    decimals = " ".join([SATUAN[int(d)] if int(d) > 0 else "nol" for d in parts[1]])
                    return f"{whole} koma {decimals}"
                except ValueError:
                    pass
        try:
            num = int(raw.replace(".", "").replace(",", ""))
            return number_to_words(num)
        except ValueError:
            return raw

    pattern = r"\b\d+(?:[\.,]\d+)*\b"
    return re.sub(pattern, replace_num, text)


def expand_abbreviations_and_slang(text: str, normalize_slang: bool = False) -> str:
    """Expand Indonesian abbreviations and optional slang."""
    words = text.split()
    expanded = []
    for w in words:
        clean_w = w.lower().strip(".,!?")
        if clean_w in ABBREVIATIONS:
            expanded.append(ABBREVIATIONS[clean_w])
        elif normalize_slang and clean_w in SLANG_MAP:
            expanded.append(SLANG_MAP[clean_w])
        else:
            expanded.append(w)
    return " ".join(expanded)


class IndonesianTextNormalizer:
    """Standard Indonesian Text Normalizer for Speech-to-Text Training & Evaluation."""

    def __init__(
        self,
        remove_punctuation: bool = False,
        to_lower: bool = True,
        normalize_slang: bool = False,
    ):
        self.remove_punctuation = remove_punctuation
        self.to_lower = to_lower
        self.normalize_slang = normalize_slang

    def __call__(self, text: str) -> str:
        return self.normalize(text)

    def normalize(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        # Normalize unicode spaces and hyphens
        text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Currency expansion
        text = normalize_currency(text)

        # Percentage expansion
        text = normalize_percentage(text)

        # Ordinal expansion ('ke-1', 'ke-2')
        text = normalize_ordinals(text)

        # Standalone numbers expansion
        text = normalize_numbers_in_text(text)

        # Abbreviations & slang expansion
        text = expand_abbreviations_and_slang(text, normalize_slang=self.normalize_slang)

        if self.to_lower:
            text = text.lower()

        if self.remove_punctuation:
            # Preserve valid Indonesian reduplication hyphens (e.g. 'anak-anak')
            def preserve_redup(m):
                w1, w2 = m.group(1), m.group(2)
                return f"{w1}__HYPHEN__{w2}"

            text = re.sub(r"\b([a-zA-Z]+)\-([a-zA-Z]+)\b", preserve_redup, text)

            # Strip non-alphanumeric except placeholder
            text = re.sub(r"[^\w\s]", " ", text)
            text = text.replace("__HYPHEN__", "-")

        # Collapse excess whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text
