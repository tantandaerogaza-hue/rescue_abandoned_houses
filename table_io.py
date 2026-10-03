"""Strict text decoding with BOM detection and CSV/TSV delimiter recognition."""
import codecs
import csv
import io

import pandas as pd


def read_text_table(content):
    # UTF-32 LE shares its first two bytes with UTF-16 LE: test it first.
    if content.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        encodings = ['utf-32']
    elif content.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encodings = ['utf-16']
    else:
        encodings = ['utf-8-sig', 'cp949']
    for encoding in encodings:
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError('지원하는 문자 인코딩으로 읽을 수 없다. UTF-8 CSV 또는 XLSX로 저장한다.')
    if not text.strip():
        raise ValueError('파일에 열 이름과 데이터가 없다.')
    try:
        delimiter = csv.Sniffer().sniff(text[:65536], delimiters=',\t;|').delimiter
    except csv.Error:
        # A header-only template or irregular record lengths can defeat Sniffer.
        header = text.splitlines()[0]
        candidates = [',', '\t', ';', '|']
        delimiter = max(candidates, key=header.count)
    return pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str)
