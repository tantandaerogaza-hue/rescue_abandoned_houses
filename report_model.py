"""Report pp.2–9. Common cohort, factor-level equal weights and competition ranks."""
import numpy as np
import pandas as pd
from model import distances

LABELS = {'attraction':'관광지', 'parking':'공영주차장', 'subway':'지하철역 출입구',
 'bus':'버스정류장', 'cctv':'방범 CCTV 설치지점', 'police':'경찰시설', 'fire':'소방시설',
 'store':'편의점', 'mart':'마트', 'pharmacy':'약국', 'industry':'산업단지 출입구',
 'daycare':'어린이집', 'school':'초등학교', 'university':'대학교 출입구',
 'library':'도서관', 'sports':'운동시설'}
DISTANCE = ['parking','subway','police','fire','industry','daycare','school','university','library','sports']
FACTORS = {'R':['attraction_count','diversity'], 'Q':['visitors','sales'],
 'T':['parking_distance','subway_distance','bus_count'],
 'S':['cctv_count','police_distance','fire_distance'], 'F':['amenity'],
 'J':['industry_distance'], 'E':['daycare_distance','school_distance'],
 'U':['university_distance'], 'Y':['library_distance','sports_distance']}
TYPES = {'A':['R','Q','T','S','F'], 'B':['J','E','T','S','F'], 'C':['U','Y','T','S','F']}
FACTOR_NAMES = {'R':'관광자원 접근성','Q':'관광수요·소비활성도','T':'접근성',
 'S':'안전환경','F':'편의시설','J':'산업단지 접근성','E':'교육',
 'U':'대학교 접근성','Y':'청년특화편의시설'}
RELEVANT = {'A':['attraction','parking','subway','bus','cctv','police','fire','store','mart','pharmacy'],
 'B':['industry','daycare','school','parking','subway','bus','cctv','police','fire','store','mart','pharmacy'],
 'C':['university','library','sports','parking','subway','bus','cctv','police','fire','store','mart','pharmacy']}

def spatial_raw(houses, facilities, radii=None, taxonomy=None):
    radii = {kind: .75 for kind in LABELS}
    out = houses.copy()
    for kind, frame in facilities.items():
        if kind == 'attraction':
            if frame['type'].isna().any() or frame['type'].astype(str).str.strip().eq('').any() or frame['type'].nunique() > 7:
                raise ValueError('관광지 유형은 누락 없이 전체 7종 이내의 동일한 분류체계로 입력한다.')
        nearest, counts, diversity = [], [], []
        radius = .75 if kind in ['store','mart','pharmacy'] else radii.get(kind, 1.)
        for _, house in houses.iterrows():
            d = distances(house.latitude, house.longitude, frame)
            nearest.append(float(d.min()) if len(d) else np.nan)
            inside = d <= radius + 1e-9
            counts.append(int(inside.sum()))
            if kind == 'attraction':
                diversity.append(int(frame.loc[inside, 'type'].nunique()))
        if kind in DISTANCE:
            out[f'{kind}_distance'] = nearest
        if kind in ['attraction','bus','cctv','store','mart','pharmacy']:
            out[f'{kind}_count'] = counts
        if kind == 'attraction':
            out['type_count'] = diversity
            out['diversity'] = np.array(diversity) / 7 * 100
    if all(f'{k}_count' in out for k in ['store','mart','pharmacy']):
        out['amenity'] = sum((out[f'{k}_count'] > 0).astype(int) for k in ['store','mart','pharmacy']) / 3 * 100
    return out

def evaluate(raw, weights, log_variables=()):
    required = sorted(set(v for variables in FACTORS.values() for v in variables))
    absent = [v for v in required if v not in raw]
    if absent:
        raise ValueError('필수 원자료가 부족하다: ' + ', '.join(absent))
    numeric = raw[required].apply(pd.to_numeric, errors='coerce')
    valid = np.isfinite(numeric).all(axis=1) & numeric.ge(0).all(axis=1)
    excluded = raw.loc[~valid].copy()
    out = raw.loc[valid].copy()
    if out.empty:
        raise ValueError('세 유형의 필수 자료가 모두 연결된 공통 분석 대상이 없다.')
    audit, scores = [], {}
    for variable in required:
        values = numeric.loc[valid, variable].astype(float)
        fixed = variable in ['amenity','diversity']
        log = variable in log_variables and not fixed and not variable.endswith('_distance')
        transformed = np.log1p(values) if log else values
        lo, hi = float(transformed.min()), float(transformed.max())
        constant = not fixed and hi == lo
        if fixed:
            scores[variable] = values
        elif not constant:
            scores[variable] = 100 * ((hi - transformed) if variable.endswith('_distance') else (transformed - lo)) / (hi - lo)
        out[f'score_{variable}'] = np.nan if constant else scores[variable]
        audit.append({'variable':variable,'min':lo,'max':hi,'log1p':log,'excluded_constant':constant,'fixed_range':fixed})
    disabled = []
    for factor, variables in FACTORS.items():
        active = [v for v in variables if v in scores]
        if not active:
            disabled.append(factor)
            out[factor] = np.nan
        else:
            out[factor] = sum(scores[v] for v in active) / len(active)
    for category, factors in TYPES.items():
        w = np.asarray(weights[category], dtype=float)
        if len(w) != 5 or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(),1):
            raise ValueError('유형별 가중치는 5개, 0 이상, 합계 1이어야 한다.')
        # No undocumented redistribution across top-level factors.
        active = [(f, weight) for f, weight in zip(factors,w) if weight > 0]
        if any(f in disabled for f,_ in active):
            out[f'{category}_score'] = np.nan
            out[f'{category}_baseline'] = np.nan
            continue
        out[f'{category}_score'] = sum(out[f]*weight for f,weight in active)
        out[f'{category}_baseline'] = np.nan if any(f in disabled for f in factors) else sum(out[f] for f in factors)/5
    common = out[[f'{c}_score' for c in TYPES]].notna().all(axis=1).all()
    for category in TYPES:
        score = out[f'{category}_score']
        if not common:
            out[f'{category}_rank'] = np.nan
            out[f'{category}_percentile'] = np.nan
        else:
            out[f'{category}_rank'] = score.rank(method='min',ascending=False).astype(int)
            out[f'{category}_percentile'] = (score.rank(method='average',ascending=True)-.5)/len(out)*100
        baseline = out[f'{category}_baseline']
        out[f'{category}_baseline_rank'] = baseline.rank(method='min',ascending=False)
        out[f'{category}_rank_change'] = out[f'{category}_baseline_rank']-out[f'{category}_rank']
    return out, excluded, pd.DataFrame(audit), disabled

def evidence(house, category, facilities, radii):
    pieces = []
    for kind in RELEVANT[category]:
        frame = facilities[kind].copy()
        if frame.empty:
            continue
        frame['distance_km'] = distances(house.latitude,house.longitude,frame)
        if kind in DISTANCE:
            frame = frame.nsmallest(1,'distance_km')
        else:
            radius = .75 if kind in ['store','mart','pharmacy'] else radii[kind]
            frame = frame.loc[frame.distance_km <= radius + 1e-9]
        frame['kind'] = kind
        frame['facility_type'] = LABELS[kind]
        pieces.append(frame)
    return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()
