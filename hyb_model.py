"""HYB/p90 fixed references. Factors 0..1; final scores 0..100."""
import numpy as np
import pandas as pd
from model import distances

MODEL_VERSION='hyb_p90_v5'
LABELS={'attraction':'관광지','parking':'공영주차장','subway':'지하철역 출입구',
 'bus':'버스정류장','cctv':'방범 CCTV 설치지점','police':'경찰시설','fire':'소방시설',
 'store':'편의점','mart':'마트','pharmacy':'약국','industry':'산업시설',
 'daycare':'어린이집,유치원','school':'초등학교','university':'대학교','dorm':'대학별 기숙사 자료'}
STEP_DISTANCE=['subway','industry','university']
BINARY=['parking','police','fire','store','mart','pharmacy','daycare','school']
DISTANCE=STEP_DISTANCE+BINARY
RADII={k:.75 for k in LABELS}; RADII.update(attraction=1.5,dorm=3.)
P90={'bus_count':51.,'attraction_count':9.,'attraction_popularity':51.,'cctv_count':101.,
     'sales_den':103284830.17,'dorm_unmet':661.}
FACTORS={'R':['attraction_count','attraction_popularity'],'Q':['sales_den'],
 'T':['parking_distance','subway_distance','bus_count'],
 'S':['cctv_count','police_distance','fire_distance'],
 'F':['store_distance','mart_distance','pharmacy_distance'],
 'J':['industry_distance'],'E':['daycare_distance','school_distance'],
 'U':['university_distance'],'D':['dorm_unmet']}
TYPES={'A':['R','Q','T','S','F'],'B':['J','E','T','S','F'],'C':['U','D','T','S','F']}
FACTOR_NAMES={'R':'HYB 관광','Q':'관광업종 매출밀도','T':'교통','S':'안전','F':'생활편의',
 'J':'산업접근','E':'교육·보육','U':'대학접근','D':'기숙사 부족'}
DEFAULT_WEIGHTS={'A':[.5,.1,.1,.2,.1],'B':[.5,.1,.1,.1,.2],'C':[.3,.2,.2,.1,.2]}
RELEVANT={'A':['attraction','parking','subway','bus','cctv','police','fire','store','mart','pharmacy'],
 'B':['industry','daycare','school','parking','subway','bus','cctv','police','fire','store','mart','pharmacy'],
 'C':['university','dorm','parking','subway','bus','cctv','police','fire','store','mart','pharmacy']}

def nonnegative(values,allow_missing=False):
    original=values.astype('string').str.strip().str.replace(',','',regex=False)
    result=pd.to_numeric(original,errors='coerce')
    # Blanks may remain missing, but nonempty text is never silently accepted.
    invalid_text=result.isna()&original.notna()&original.ne('')
    valid=result.notna()
    if invalid_text.any() or (not allow_missing and not valid.all()) or (valid&((result<0)|~np.isfinite(result))).any():
        raise ValueError('값은 유한한 0 이상 숫자여야 한다. 결측을 0으로 바꾸지 않는다.')
    return result.astype(float)

def distance_score(values):
    """<=.75:1; <=1.5:.75; <=2.25:.5; <=3:.25; >3:0 (km)."""
    x=np.asarray(values,dtype=float)
    score=np.select([x<=.75+1e-9,x<=1.5+1e-9,x<=2.25+1e-9,x<=3.+1e-9],
                    [1.,.75,.5,.25],default=0.)
    return np.where(np.isnan(x)|(x<0),np.nan,score)

def spatial_raw(houses,facilities,radii=None,taxonomy=None):
    out=houses.copy()
    for kind in LABELS:
        if kind not in facilities: continue
        frame=facilities[kind]
        if kind=='attraction': popularity=nonnegative(frame['appear_cnt'])
        if kind=='dorm': unmet=nonnegative(frame['unmet'],allow_missing=True)
        nearest=[]; counts=[]; sums=[]
        for _,house in houses.iterrows():
            d=distances(house.latitude,house.longitude,frame)
            nearest.append(float(d.min()) if len(d) else np.inf)
            inside=d<=RADII[kind]+1e-9
            counts.append(int(inside.sum()))
            if kind=='attraction': sums.append(float(popularity.loc[inside].sum()))
            if kind=='dorm':
                local=unmet.loc[inside]
                sums.append(np.nan if local.isna().any() else float(local.sum()))
        if kind in DISTANCE: out[kind+'_distance']=nearest
        if kind in ['attraction','bus','cctv']: out[kind+'_count']=counts
        if kind=='attraction': out['attraction_popularity']=sums
        if kind=='dorm': out['dorm_unmet']=sums
    return out

def cohort_values(raw):
    required=sorted({v for values in FACTORS.values() for v in values})
    numeric=raw.reindex(columns=required).apply(pd.to_numeric,errors='coerce')
    validity=np.isfinite(numeric)&numeric.ge(0)
    for kind in DISTANCE:
        # +inf explicitly means a verified empty facility inventory.
        validity[kind+'_distance']=numeric[kind+'_distance'].notna()&numeric[kind+'_distance'].ge(0)
    return numeric,validity

def evaluate(raw,weights=None,log_variables=()):
    if log_variables: raise ValueError('HYB·p90 모델은 로그변환을 사용하지 않는다.')
    weights=DEFAULT_WEIGHTS if weights is None else weights
    numeric,validity=cohort_values(raw)
    absent=[v for v in numeric if v not in raw]
    if absent: raise ValueError('필수 원자료가 부족하다: '+', '.join(absent))
    valid=validity.all(axis=1)
    excluded=raw.loc[~valid].copy(); out=raw.loc[valid].copy()
    if out.empty: raise ValueError('필수 자료가 모두 연결된 공통 분석 대상이 없다. 매출밀도·기숙사 자료 연결을 확인한다.')
    audit=[]
    for variable in numeric:
        values=numeric.loc[valid,variable].astype(float)
        if variable in P90:
            score=(values/P90[variable]).clip(0,1); method='p90 상한 비율'; reference=P90[variable]
        elif variable.removesuffix('_distance') in STEP_DISTANCE:
            score=distance_score(values); method='750m 거리 구간'; reference='750/1500/2250/3000m'
        else:
            score=(values<=.75+1e-9).astype(float); method='750m 유무'; reference=750
        out['score_'+variable]=score
        audit.append({'variable':variable,'method':method,'reference':str(reference),'score_range':'0~1','log1p':False})
    for factor,variables in FACTORS.items(): out[factor]=out[['score_'+v for v in variables]].mean(axis=1)
    for category,factors in TYPES.items():
        w=np.asarray(weights[category],dtype=float)
        if len(w)!=5 or not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1):
            raise ValueError('유형별 가중치는 5개, 0 이상, 합계 1이어야 한다.')
        out[category+'_score']=100*sum(out[f]*weight for f,weight in zip(factors,w))
        out[category+'_baseline']=100*sum(out[f]*weight for f,weight in zip(factors,DEFAULT_WEIGHTS[category]))
        out[category+'_rank']=out[category+'_score'].rank(method='min',ascending=False).astype(int)
        out[category+'_percentile']=(out[category+'_score'].rank(method='average')-.5)/len(out)*100
        out[category+'_baseline_rank']=out[category+'_baseline'].rank(method='min',ascending=False).astype(int)
        out[category+'_rank_change']=out[category+'_baseline_rank']-out[category+'_rank']
    return out,excluded,pd.DataFrame(audit),[]

def evidence(house,category,facilities,radii=None):
    pieces=[]
    for kind in RELEVANT[category]:
        frame=facilities[kind].copy()
        if frame.empty: continue
        frame['distance_km']=distances(house.latitude,house.longitude,frame)
        if kind in STEP_DISTANCE: frame=frame.nsmallest(1,'distance_km')
        else: frame=frame.loc[frame.distance_km<=RADII[kind]+1e-9]
        frame['kind']=kind; frame['facility_type']=LABELS[kind]; pieces.append(frame)
    return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()
