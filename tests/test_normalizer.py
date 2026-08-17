"""Unit tests for Indonesian Text Normalizer."""

from data.indonesian_normalizer import (
    number_to_words,
    normalize_currency,
    normalize_percentage,
    IndonesianTextNormalizer,
)


def test_number_to_words():
    assert number_to_words(0) == "nol"
    assert number_to_words(1) == "satu"
    assert number_to_words(11) == "sebelas"
    assert number_to_words(17) == "tujuh belas"
    assert number_to_words(25) == "dua puluh lima"
    assert number_to_words(100) == "seratus"
    assert number_to_words(105) == "seratus lima"
    assert number_to_words(1000) == "seribu"
    assert number_to_words(1945) == "seribu sembilan ratus empat puluh lima"
    assert number_to_words(50000) == "lima puluh ribu"
    assert number_to_words(1500000) == "satu juta lima ratus ribu"


def test_currency_normalization():
    assert "lima puluh ribu rupiah" in normalize_currency("Rp 50.000")
    assert "tujuh puluh lima ribu rupiah" in normalize_currency("IDR 75.000")


def test_percentage_normalization():
    assert "dua puluh lima persen" in normalize_percentage("25%")


def test_slang_and_abbreviations():
    norm = IndonesianTextNormalizer(remove_punctuation=True, to_lower=True, normalize_slang=True)
    text = "Gue gak tau kalo rapat udah dimulai bgt"
    normalized = norm(text)
    assert "tidak" in normalized
    assert "kalau" in normalized
    assert "sudah" in normalized
    assert "banget" in normalized


def test_full_indonesian_normalizer():
    norm = IndonesianTextNormalizer(remove_punctuation=True, to_lower=True)
    text = "Harga tiket Rp 50.000 utk anak-anak pd tgl 17 Agustus."
    normalized = norm(text)
    
    assert "lima puluh ribu rupiah" in normalized
    assert "untuk" in normalized
    assert "anak-anak" in normalized
    assert "pada" in normalized
    assert "tujuh belas" in normalized


if __name__ == "__main__":
    test_number_to_words()
    test_currency_normalization()
    test_percentage_normalization()
    test_slang_and_abbreviations()
    test_full_indonesian_normalizer()
    print("✓ All Indonesian normalizer tests passed!")
