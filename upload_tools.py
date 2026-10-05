"""Filename/column hints and bounded, in-memory ZIP reads."""
import io
import zipfile
from pathlib import PurePosixPath
import unicodedata
import pandas as pd

KEYWORDS={
 'houses':['빈집','vacant','house'], 'attraction':['관광','attraction','tourism'],
 'parking':['주차','parking'],'subway':['지하철','도시철도','역사정보','subway'],
 'bus':['버스','bus'],'cctv':['cctv','방범'], 'police':['경찰','지구대','파출소','police'],
 'police_station':['경찰서','경찰관서','police_station'], 'police_local':['지구대','파출소','치안센터','police_local'],
 'fire':['소방','119','fire'],'store':['편의점','store'],'mart':['마트','mart'],'pharmacy':['약국','pharmacy'],
 'amenities':['편의시설','생활편의','amenities'],'industry':['산업단지','산단','industry'],
 'daycare':['어린이집','유치원','daycare','kindergarten'],'school':['초등','school'],
 'university':['대학교','대학','university'],'library':['도서관','library'],'sports':['운동','체육','sports'],
 'visitors':['방문','방문자','유동','visitor'],'sales':['매출','소비','sales']}

def first_match(values,keywords):
    for value in values:
        text=unicodedata.normalize('NFC',str(value)).casefold()
        if any(word.casefold() in text for word in keywords): return value
    return None

def zip_tables(content):
    pool={}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        files=[i for i in archive.infolist() if not i.is_dir()]
        if len(files)>1000: raise ValueError('ZIP 파일 항목은 1,000개 이하로 준비한다.')
        total=0
        for info in files:
            name=info.filename
            if not info.flag_bits & 0x800:
                try: name=name.encode('cp437').decode('cp949')
                except (UnicodeError,LookupError): pass
            name=unicodedata.normalize('NFC',name.replace('\\','/'))
            path=PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts: raise ValueError('ZIP 내부 경로가 올바르지 않다.')
            if '__MACOSX' in path.parts or path.name.startswith(('~$','.')): continue
            if path.suffix.lower() not in ['.csv','.xlsx']: continue
            if info.flag_bits & 1: raise ValueError('암호화되지 않은 ZIP을 사용한다.')
            if info.file_size>100*1024*1024: raise ValueError('개별 파일은 압축 해제 기준 100MB 이하여야 한다.')
            total+=info.file_size
            if total>300*1024*1024: raise ValueError('데이터 전체는 압축 해제 기준 300MB 이하여야 한다.')
            if name in pool: raise ValueError('ZIP 내부에 같은 이름의 데이터 파일이 중복된다.')
            pool[name]=archive.read(info)
    if not pool: raise ValueError('ZIP에 CSV 또는 XLSX가 없다.')
    return pool

def deduplicate_facilities(frame):
    # Reused IDs at different coordinates are legitimate, e.g. stops in two directions.
    keys=['latitude','longitude']
    if 'facility_id' in frame: keys=['facility_id']+keys
    if 'type' in frame: keys+=['type']
    return frame.drop_duplicates(subset=keys).reset_index(drop=True)

def merge_police(first,second):
    # Cross-file IDs often use independent numbering; geometry defines the site here.
    joined=pd.concat([first,second],ignore_index=True)
    return joined.drop_duplicates(['latitude','longitude']).reset_index(drop=True)

def split_amenities(frame,mapping):
    unknown=set(frame['type'].dropna())-set(mapping)
    if frame['type'].isna().any() or unknown: raise ValueError('모든 편의시설 분류 값을 연결한다.')
    kinds=frame['type'].map(mapping)
    if not kinds.isin(['store','mart','pharmacy']).all(): raise ValueError('편의점·마트·약국 중 하나로 분류한다.')
    return {kind:deduplicate_facilities(frame.loc[kinds==kind].drop(columns=['type'])) for kind in ['store','mart','pharmacy']}
