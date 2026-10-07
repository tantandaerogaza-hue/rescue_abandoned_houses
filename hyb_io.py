"""Prepared sales density and institution-level dormitory input validation."""
import numpy as np
import pandas as pd
from report_io import normalize
from hyb_model import nonnegative

def density_series(source,admin_column,value_column,code=False,month_column=None):
    d=pd.DataFrame({'admin_key':normalize(source[admin_column],code),
                    'sales_den':nonnegative(source[value_column],allow_missing=True)})
    latest=None
    if month_column:
        months=normalize(source[month_column],True)
        if pd.to_datetime(months,format='%Y%m',errors='coerce').isna().any():
            raise ValueError('기준년월은 YYYYMM 형식이어야 한다.')
        latest=months.max(); d=d.loc[months.eq(latest)].copy()
    if d.empty: raise ValueError('매출밀도 자료가 비어 있다.')
    unnamed=int(d.admin_key.isna().sum()); d=d.loc[d.admin_key.notna()]
    if d.admin_key.duplicated().any():
        raise ValueError('행정동별 관광업종 매출밀도는 한 행이어야 한다. 원매출액이나 업종별 행을 매출밀도로 사용하지 않는다.')
    return d.set_index('admin_key')['sales_den'],latest,unnamed

def dorm_inventory(frame):
    """One university/campus per row; missing unmet data stay missing."""
    frame=frame.copy()
    frame['unmet']=nonnegative(frame['unmet'],allow_missing=True)
    if 'facility_id' in frame and frame.facility_id.duplicated().any():
        raise ValueError('기숙사 자료는 대학/캠퍼스 ID별 한 행이어야 한다. 출입구 중복으로 미충족 인원이 중복 합산되지 않도록 한다.')
    # Same site with conflicting values must not silently select an arbitrary record.
    site=['latitude','longitude']
    if frame.duplicated(site,keep=False).any():
        duplicate=frame.loc[frame.duplicated(site,keep=False)]
        if duplicate.groupby(site,dropna=False)['unmet'].nunique(dropna=False).gt(1).any():
            raise ValueError('동일 대학 위치에 서로 다른 기숙사 값이 있다. 대학/캠퍼스별 자료를 확인한다.')
    return frame.drop_duplicates(site).reset_index(drop=True)


def calculate_dorm_unmet(applicants, capacity):
    """Derived exclusively from the two variables in the confirmed formula."""
    applicants=nonnegative(applicants,allow_missing=True)
    capacity=nonnegative(capacity,allow_missing=True)
    return (applicants-capacity).clip(lower=0)
