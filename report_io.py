import io
import pandas as pd
import numpy as np
from table_io import read_text_table

def read_file(data, name):
    result = pd.read_excel(io.BytesIO(data), dtype=str) if name.lower().endswith('.xlsx') else read_text_table(data)
    result.columns = result.columns.astype(str).str.strip()
    return result

def normalize(values, code=False):
    s = values.astype('string').str.strip().str.replace(r'\s+', ' ', regex=True).replace('', pd.NA)
    return s.str.replace(r'\.0$', '', regex=True) if code else s

def validate_coordinates(frame, house=False):
    frame = frame.copy()
    for c in ['latitude','longitude']:
        frame[c] = pd.to_numeric(frame[c], errors='coerce')
    if not (frame.latitude.between(-90,90)&frame.longitude.between(-180,180)).all():
        raise ValueError('위도·경도가 비어 있거나 숫자 범위를 벗어난 행이 있다.')
    if house:
        frame['house_id'] = normalize(frame.house_id)
        if frame.empty or frame.house_id.isna().any() or frame.house_id.duplicated().any():
            raise ValueError('빈집 ID는 비어 있거나 중복될 수 없다.')
    return frame

def aggregate_stat(source, month_column, key_column, value_column, code=False, industry_column=None, industries=None):
    d = pd.DataFrame({'month':normalize(source[month_column],True),
                      'admin_key':normalize(source[key_column],code)})
    if pd.to_datetime(d.month,format='%Y%m',errors='coerce').isna().any() or d.empty:
        raise ValueError('기준년월은 YYYYMM 형식이어야 하며 자료가 비어 있으면 안 된다.')
    latest = d.month.max()
    mask = d.month.eq(latest)
    d = d.loc[mask].copy()
    d['value'] = pd.to_numeric(source.loc[mask,value_column].astype(str).str.replace(',','',regex=False),errors='coerce')
    if d.admin_key.isna().any():
        # Unnamed rows cannot join by name; report their exclusion to the caller.
        unnamed = int(d.admin_key.isna().sum())
        d = d.loc[d.admin_key.notna()].copy()
    else:
        unnamed = 0
    if not np.isfinite(d.value).all() or d.value.lt(0).any():
        raise ValueError('최근 월의 값에 결측·문자·음수가 있다. 임의로 0으로 바꾸지 않는다.')
    if industry_column:
        d['industry'] = normalize(source.loc[d.index,industry_column])
        if d.industry.isna().any():
            raise ValueError('업종이 비어 있다.')
        if d.duplicated(['admin_key','industry']).any():
            raise ValueError('같은 행정동·업종이 최근 월에 중복된다.')
        industries = industries or sorted(d.industry.unique())
        d = d.loc[d.industry.isin(industries)]
        result = d.groupby('admin_key').agg(value=('value','sum'),n=('industry','nunique'))
        result.loc[result.n != len(industries),'value'] = np.nan
        result = result['value']
    else:
        if d.admin_key.duplicated().any():
            raise ValueError('최근 월의 행정동별 값은 한 행이어야 한다. 세부 항목은 먼저 같은 기준으로 집계한다.')
        result = d.set_index('admin_key')['value']
    return result, latest, unnamed
