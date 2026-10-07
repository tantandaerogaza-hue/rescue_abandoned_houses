"""Explain exclusions without altering administrative keys or scoring policy."""
import numpy as np
import pandas as pd
from report_model import FACTORS, LABELS, cohort_values

def admin_connections(houses, visitors, sales):
    rows=houses[['house_id','admin_key']].copy()
    rows.columns=['빈집 ID','빈집 행정동']
    for label,series in [('방문인원',visitors),('매출액',sales)]:
        values=houses.admin_key.map(series)
        present=houses.admin_key.isin(series.index)
        rows[label+' 연결']=np.where(~present,'행정동 불일치',np.where(values.isna(),'해당 동 집계값 누락','연결됨'))
        rows[label+' 값']=values.to_numpy()
    return rows

def cohort_diagnostics(raw):
    numeric,valid_values=cohort_values(raw)
    required=list(numeric.columns)
    valid=valid_values.all(axis=1)
    labels={'sales_den':'행정동 관광업종 매출밀도','attraction_popularity':'1.5km 내 appear_cnt 합계','dorm_unmet':'3km 내 대학별 미충족 인원 합계'}
    for key,value in LABELS.items():
        labels[key+'_distance']=value+' 최근접 거리'
        labels[key+'_count']=value+(' 1.5km 내 개수' if key=='attraction' else ' 750m 내 개수')
    counts=pd.DataFrame([{'지표':labels.get(c,c),'누락·잘못된 값':int((~valid_values[c]).sum()),
                         '음수':int(numeric[c].lt(0).sum()),'유효 빈집 수':int(valid_values[c].sum())} for c in required])
    rows=raw.loc[~valid,['house_id','admin_key']].copy()
    rows['제외 이유']=[' / '.join(labels.get(c,c) for c in required if not valid_values.loc[i,c]) for i in rows.index]
    return valid,counts,rows
