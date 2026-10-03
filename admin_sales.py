"""Aggregate administrative-dong sales without inventing point coordinates."""
import numpy as np
import pandas as pd
import re

def attach_address_dongs(houses, summary):
    """Only exact unique administrative names; never infer numbered dong from legal dong."""
    result = houses.copy()
    if 'admin_name' not in result:
        result['admin_name'] = pd.Series(pd.NA, index=result.index, dtype='string')
    result['address_match_status'] = '주소 없음'
    result['address_dong_candidates'] = ''
    for idx, row in result.iterrows():
        if pd.notna(row.get('admin_code')) and str(row.get('admin_code')).strip():
            result.loc[idx, 'address_match_status'] = '직접 입력 코드 우선'
            continue
        if pd.notna(row.get('admin_name')) and str(row.get('admin_name')).strip():
            result.loc[idx, 'address_match_status'] = '직접 입력 행정동 우선'
            continue
        address = row.get('address')
        if pd.isna(address) or not str(address).strip():
            continue
        address = str(address)
        tokens = set(re.findall(r'(?<![가-힣A-Za-z0-9])([가-힣][가-힣0-9·.]*동)(?![가-힣A-Za-z0-9])', address))
        districts = set(re.findall(r'(?<![가-힣A-Za-z0-9])([가-힣]+[구군])(?![가-힣A-Za-z0-9])', address))
        result.loc[idx, 'address_dong_candidates'] = ', '.join(sorted(tokens))
        candidates = summary.loc[summary.admin_name.str.split().str[-1].isin(tokens)]
        if districts:
            candidates = candidates.loc[candidates.admin_name.map(lambda name: bool(set(name.split()) & districts))]
        if len(candidates) == 1:
            result.loc[idx, 'admin_name'] = candidates.iloc[0].admin_name
            result.loc[idx, 'address_match_status'] = '주소에서 행정동 정확히 일치'
        else:
            result.loc[idx, 'address_match_status'] = '동명 없음' if not tokens else '행정동 불일치 또는 중복 — 확인 필요'
    return result

def clean_code(values):
    return values.astype('string').str.strip().str.replace(r'\.0$', '', regex=True).replace('', pd.NA)

def clean_name(values):
    return values.astype('string').str.strip().str.replace(r'\s+', ' ', regex=True).replace('', pd.NA)

def prepare_sales(source, latest_only=False):
    aliases = {'기준년월': 'month', '행정동코드': 'admin_code', '행정동명': 'admin_name',
               '업종대분류': 'industry', '평균이용금액': 'amount'}
    data = source.rename(columns=aliases).copy()
    required = ['month', 'admin_code', 'admin_name', 'industry', 'amount']
    absent = [c for c in required if c not in data]
    if absent:
        raise ValueError('필요한 열: 기준년월, 행정동코드, 행정동명, 업종대분류, 평균이용금액')
    data = data[required].copy()
    data['admin_code'] = clean_code(data.admin_code)
    data['admin_name'] = clean_name(data.admin_name)
    data['month'] = clean_code(data.month)
    if pd.to_datetime(data.month, format='%Y%m', errors='coerce').isna().any():
        raise ValueError('기준년월은 YYYYMM 형식이어야 한다.')
    if latest_only and not data.empty:
        data = data.loc[data.month == data.month.max()].copy()
    data['admin_name_missing'] = data.admin_name.isna()
    data['admin_name'] = data.admin_name.fillna('행정동명 미제공 (' + data.admin_code + ')')
    data['industry'] = clean_name(data.industry)
    data['amount'] = pd.to_numeric(data.amount.astype('string').str.replace(',', '', regex=False), errors='coerce')
    if data.empty or data[required].isna().any().any() or not np.isfinite(data.amount).all() or (data.amount < 0).any():
        raise ValueError('빈 자료 또는 누락·잘못된 행정동/업종/금액이 있다. 금액은 0 이상이어야 한다.')
    if pd.to_datetime(data.month, format='%Y%m', errors='coerce').isna().any():
        raise ValueError('기준년월은 YYYYMM 형식이어야 한다.')
    if data.duplicated(['month', 'admin_code', 'industry']).any():
        raise ValueError('같은 월·행정동코드·업종이 중복된다. 중복 집계를 방지하려면 원자료를 확인한다.')
    return data

def aggregate_sales(data, months, industries):
    if not months or not industries:
        raise ValueError('월과 업종을 각각 하나 이상 선택한다.')
    selected = data.loc[data.month.isin(months) & data.industry.isin(industries)].copy()
    if selected.empty:
        raise ValueError('선택한 월·업종에 해당하는 자료가 없다.')
    monthly = selected.groupby(['admin_code', 'month'], as_index=False).agg(
        monthly_amount=('amount', 'sum'), industry_count=('industry', 'nunique'))
    summary = monthly.groupby('admin_code', as_index=False).agg(
        sales_amount=('monthly_amount', 'mean'), month_count=('month', 'nunique'),
        min_industry_count=('industry_count', 'min'))
    # Use the latest selected name for each code, never average differing coverage.
    names = selected.sort_values('month').drop_duplicates('admin_code', keep='last')[['admin_code', 'admin_name']]
    summary = summary.merge(names, on='admin_code', validate='one_to_one')
    summary['sales_complete'] = (summary.month_count == len(set(months))) & (summary.min_industry_count == len(set(industries)))
    summary.loc[~summary.sales_complete, 'sales_amount'] = np.nan
    valid = summary.sales_amount.dropna()
    summary['sales_score'] = np.nan
    if len(valid):
        low, high = float(valid.min()), float(valid.max())
        summary.loc[summary.sales_complete, 'sales_score'] = (
            50.0 if high == low else 100 * (summary.loc[summary.sales_complete, 'sales_amount'] - low) / (high - low))
    else:
        low = high = None
    metadata = {'months': list(months), 'industries': list(industries),
                'aggregation': 'sum_industries_within_month_then_equal_mean_of_months',
                'normalization': 'min_max_all_complete_dongs_in_selected_uploaded_data',
                'min_amount': low, 'max_amount': high, 'equal_amount_score': 50,
                'excluded_incomplete_dongs': int((~summary.sales_complete).sum()),
                'unnamed_dongs': int(selected.loc[selected.admin_name_missing, 'admin_code'].nunique()) if 'admin_name_missing' in selected else 0}
    return summary, metadata

def attach_sales(houses, summary):
    result = houses.copy()
    result['sales_admin_name'] = pd.NA
    result['sales_amount'] = np.nan
    result['sales_score'] = np.nan
    result['sales_match_status'] = '행정동 정보 없음'
    by_code = summary.set_index('admin_code')
    name_counts = summary.groupby('admin_name').size()
    by_name = summary.loc[summary.admin_name.map(name_counts) == 1].set_index('admin_name')
    codes = clean_code(result['admin_code']) if 'admin_code' in result else pd.Series(pd.NA, index=result.index)
    names = clean_name(result['admin_name']) if 'admin_name' in result else pd.Series(pd.NA, index=result.index)
    for idx in result.index:
        code, name = codes.loc[idx], names.loc[idx]
        row = None
        if pd.notna(code):
            if code in by_code.index:
                row = by_code.loc[code]
            else:
                result.loc[idx, 'sales_match_status'] = '행정동코드 미일치'
        elif pd.notna(name):
            if name in by_name.index:
                row = by_name.loc[name]
            else:
                result.loc[idx, 'sales_match_status'] = '행정동명 미일치 또는 중복'
        if row is not None:
            result.loc[idx, 'sales_admin_name'] = row.get('admin_name', name)
            result.loc[idx, 'sales_amount'] = row.sales_amount
            result.loc[idx, 'sales_score'] = row.sales_score
            result.loc[idx, 'sales_match_status'] = '연결 완료' if pd.notna(row.sales_score) else '선택 월·업종 자료 불완전'
    return result

def attach_crosswalk(houses, mapping):
    mapping = mapping.copy()
    if 'house_id' not in mapping or not {'admin_code', 'admin_name'} & set(mapping):
        raise ValueError('연결표에 house_id와 admin_code 또는 admin_name이 필요하다.')
    mapping['house_id'] = clean_name(mapping.house_id)
    if mapping.house_id.isna().any() or mapping.house_id.duplicated().any():
        raise ValueError('연결표의 빈집 ID가 비어 있거나 중복된다.')
    result = houses.copy()
    mapping = mapping.set_index('house_id')
    for field, clean in [('admin_code', clean_code), ('admin_name', clean_name)]:
        if field in mapping:
            extra = clean(result.house_id.map(mapping[field]))
            existing = clean(result[field]) if field in result else pd.Series(pd.NA, index=result.index)
            conflict = existing.notna() & extra.notna() & existing.ne(extra)
            if conflict.any():
                raise ValueError(f'빈집 파일과 연결표의 {field}가 서로 다르다. 원자료를 확인한다.')
            result[field] = existing.fillna(extra)
    return result
