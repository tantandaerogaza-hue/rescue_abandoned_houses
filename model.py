"""Spatial evidence and explicitly provisional scoring; WGS84 decimal degrees."""
import numpy as np
import pandas as pd
try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

EARTH_KM = 6371.0088
LABELS = {
    'university': '대학교', 'workplace': '직장·산단', 'attraction': '관광자원',
    'subway': '지하철역', 'bus': '버스정류장', 'cctv': 'CCTV',
    'convenience': '생활편의시설', 'demand': '관광수요·소비 관측점',
}
NAMES = {'A': '관광체류형', 'B': '근로자 정주임대형', 'C': '대학생 정주임대형'}
ANCHOR = {'A': 'attraction', 'B': 'workplace', 'C': 'university'}
REQUIRED = {
    'A': ['attraction', 'subway', 'bus', 'cctv', 'demand'],
    'B': ['workplace', 'subway', 'bus', 'cctv', 'convenience'],
    'C': ['university', 'subway', 'bus', 'cctv', 'convenience'],
}

def xyz(frame):
    lat = np.radians(frame['latitude'].to_numpy(dtype=float))
    lon = np.radians(frame['longitude'].to_numpy(dtype=float))
    return np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])

def distances(lat, lon, frame):
    origin = pd.DataFrame({'latitude': [lat], 'longitude': [lon]})
    chord = np.linalg.norm(xyz(frame) - xyz(origin)[0], axis=1)
    return 2 * EARTH_KM * np.arcsin(np.clip(chord / 2, 0, 1))

def spatial_metrics(houses, datasets, radius):
    """Nearest distances and exact spherical radius counts via spatial index."""
    result = houses.copy()
    targets = xyz(houses)
    chord_radius = 2 * np.sin(radius / (2 * EARTH_KM))
    for kind, frame in datasets.items():
        if frame.empty:
            result[f'{kind}_nearest_km'] = np.nan
            result[f'{kind}_count'] = 0
            if kind == 'demand':
                result['demand_sum'] = 0.0
            continue
        coordinates = xyz(frame)
        if cKDTree is None:
            nearest, counts, sums = [], [], []
            for target in targets:
                separation = np.linalg.norm(coordinates - target, axis=1)
                inside = separation <= chord_radius
                nearest.append(separation.min())
                counts.append(int(inside.sum()))
                if kind == 'demand':
                    sums.append(float(frame.loc[inside, 'value'].sum()))
            result[f'{kind}_nearest_km'] = 2 * EARTH_KM * np.arcsin(np.clip(np.array(nearest) / 2, 0, 1))
            result[f'{kind}_count'] = counts
            if kind == 'demand':
                result['demand_sum'] = sums
            continue
        tree = cKDTree(coordinates)
        nearest, _ = tree.query(targets, k=1)
        result[f'{kind}_nearest_km'] = 2 * EARTH_KM * np.arcsin(np.clip(nearest / 2, 0, 1))
        if kind == 'demand':
            neighbors = tree.query_ball_point(targets, chord_radius)
            values = frame['value'].to_numpy(dtype=float)
            result['demand_count'] = [len(ids) for ids in neighbors]
            result['demand_sum'] = [float(values[ids].sum()) for ids in neighbors]
        else:
            result[f'{kind}_count'] = tree.query_ball_point(targets, chord_radius, return_length=True)
    return result

def score_metrics(raw, datasets, thresholds, weights):
    result = raw.copy()
    missing = {}
    available = []
    for category, required in REQUIRED.items():
        missing[category] = [LABELS[k] for k in required if k not in datasets
                             and not (k == 'demand' and 'sales_score' in raw)]
        anchor = ANCHOR[category]
        if anchor in datasets and datasets[anchor].empty:
            missing[category].append(f'{LABELS[anchor]}: 좌표가 한 개 이상 필요')
        if missing[category]:
            continue
        available.append(category)
        values = [
            100 * (1 - result[f'{anchor}_nearest_km'] / thresholds[f'{category}_distance']).clip(0, 1),
            100 * ((result['subway_count'] + result['bus_count']) / thresholds['transport']).clip(0, 1),
            100 * (result['cctv_count'] / thresholds['cctv']).clip(0, 1),
            (result['sales_score'] if 'sales_score' in result else
             100 * (result['demand_sum'] / thresholds['demand']).clip(0, 1)) if category == 'A'
            else 100 * (result['convenience_count'] / thresholds['convenience']).clip(0, 1),
        ]
        for index, value in enumerate(values, 1):
            result[f'{category}_indicator_{index}'] = value
        result[f'{category}_score'] = sum(value * w for value, w in zip(values, weights[category]))
    if available:
        columns = [f'{c}_score' for c in available]
        result['best_score'] = result[columns].max(axis=1)
        result['category'] = pd.Series(pd.NA, index=result.index, dtype='string')
        valid = result[columns].notna().any(axis=1)
        result.loc[valid, 'category'] = result.loc[valid, columns].idxmax(axis=1).str[0]
        result['tied_categories'] = result.apply(
            lambda row: '/'.join(c for c in available if np.isclose(row[f'{c}_score'], row['best_score'], atol=1e-9, rtol=0)), axis=1)
    return result, available, missing

def rank_houses(scored, available, recommended_only=True):
    """Ordinal ranks; exact ties broken by house_id for a maximum of five markers."""
    pieces = []
    for category in available:
        subset = scored.loc[scored['category'] == category].copy() if recommended_only else scored.copy()
        subset = subset.loc[subset[f'{category}_score'].notna()].copy()
        if subset.empty:
            continue
        subset = subset.sort_values([f'{category}_score', 'house_id'], ascending=[False, True], kind='stable')
        subset['rank'] = np.arange(1, len(subset) + 1)
        subset['rank_category'] = category
        subset['rank_score'] = subset[f'{category}_score']
        subset['badge'] = category + subset['rank'].astype(str)
        subset['label'] = subset['rank_score'].map(lambda value: f'{category}({value:.1f}점)')
        subset['detail'] = subset.apply(lambda row: ' / '.join(
            (f'{c}: {row[f"{c}_score"]:.2f}점' if pd.notna(row[f'{c}_score']) else f'{c}: 미산정')
            for c in available if f'{c}_score' in row.index), axis=1)
        pieces.append(subset)
    if not pieces:
        return pd.DataFrame()
    result = pd.concat(pieces, ignore_index=True)
    first = ['house_id', 'label', 'best_score', 'detail']
    return result[first + [column for column in result.columns if column not in first]]

def evidence_for(house, category, datasets, display_radius, evaluation_radius):
    pieces = []
    anchor = ANCHOR[category]
    for kind in REQUIRED[category]:
        if kind not in datasets or datasets[kind].empty:
            continue
        frame = datasets[kind].copy()
        frame['distance_km'] = distances(house['latitude'], house['longitude'], frame)
        if kind == anchor:
            frame = frame.nsmallest(1, 'distance_km').copy()
            frame['used_in_score'] = True
        else:
            frame = frame.loc[frame['distance_km'] <= display_radius].copy()
            frame['used_in_score'] = frame['distance_km'] <= evaluation_radius
        frame['kind'] = kind
        frame['시설유형'] = LABELS[kind]
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
