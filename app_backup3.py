import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from table_io import read_text_table

from model import (LABELS, NAMES, ANCHOR, REQUIRED, spatial_metrics, score_metrics,
                   rank_houses, evidence_for)

st.set_page_config(page_title='빈집 활용 우선순위', layout='wide')
st.title('빈집 활용 우선순위 · 주변 시설 탐색')
st.caption('WGS84 위경도 · 직선거리 기반 · 원자료에서 지표를 재계산하는 연구용 시제품')
COLORS = {'A': [231, 101, 50], 'B': [42, 112, 212], 'C': [22, 143, 92]}
FACILITY_COLORS = {
    'university': [138, 61, 191], 'workplace': [138, 61, 191], 'attraction': [138, 61, 191],
    'subway': [0, 126, 220], 'bus': [58, 173, 210], 'cctv': [238, 164, 18],
    'convenience': [224, 70, 132], 'demand': [224, 70, 132],
}

@st.cache_data(show_spinner=False)
def read_table(content, extension):
    if extension == '.xlsx':
        return pd.read_excel(io.BytesIO(content), dtype=str)
    return read_text_table(content)

def upload_table(label, key, house=False, demand=False):
    file = st.file_uploader(label, type=['xlsx', 'csv'], key=f'upload_{key}')
    if file is None:
        return None
    try:
        source = read_table(file.getvalue(), Path(file.name).suffix.lower())
        source.columns = source.columns.str.strip()
        if not len(source.columns) or source.columns.duplicated().any():
            raise ValueError('열 이름이 없거나 중복되어 있다.')
        options = list(source.columns)
        aliases = {'latitude': ['latitude', '위도', '역위도', 'lat'], 'longitude': ['longitude', '경도', '역경도', 'lon'],
                   'house_id': ['house_id', 'ID', 'id', '빈집ID'],
                   'name': ['name', '명칭', '시설명', '대학교명', '정류장명', '역사명'],
                   'value': ['value', '값', '소비액', '방문자수']}
        mapped = {}
        fields = ['house_id', 'latitude', 'longitude'] if house else ['latitude', 'longitude', 'name']
        if demand:
            fields.append('value')
        with st.expander(f'{label} — 열 연결', expanded=False):
            for field in fields:
                choices = ['(행 번호 사용)'] + options if field == 'name' else options
                default = next((a for a in aliases[field] if a in options), choices[0])
                mapped[field] = st.selectbox(field, choices, index=choices.index(default), key=f'map_{key}_{field}')
        used = [value for value in mapped.values() if value != '(행 번호 사용)']
        if len(used) != len(set(used)):
            raise ValueError('서로 다른 항목을 같은 열에 연결했다. 열 연결을 확인한다.')
        frame = pd.DataFrame(index=source.index)
        for field, column in mapped.items():
            frame[field] = [f'{label} {i+1}' for i in range(len(source))] if column == '(행 번호 사용)' else source[column]
        for field in ['latitude', 'longitude'] + (['value'] if demand else []):
            frame[field] = pd.to_numeric(frame[field], errors='coerce')
        valid = frame['latitude'].between(-90, 90) & frame['longitude'].between(-180, 180)
        if demand:
            valid &= frame['value'].ge(0) & np.isfinite(frame['value'])
        if not valid.all():
            raise ValueError(f'좌표 또는 value가 잘못된 행 {int((~valid).sum())}개. 누락 없이 수정 후 다시 올린다.')
        id_column = 'house_id' if house else 'name'
        frame[id_column] = frame[id_column].astype('string').str.strip()
        if frame[id_column].isna().any() or frame[id_column].eq('').any():
            raise ValueError(f'{id_column}에 빈칸이 있다.')
        if house and (frame.empty or frame['house_id'].duplicated().any()):
            raise ValueError('빈집 목록이 비어 있거나 ID가 중복되어 있다.')
        if not house:
            duplicate = frame.duplicated().sum()
            if duplicate:
                frame = frame.drop_duplicates().reset_index(drop=True)
                st.caption(f'완전히 동일한 시설 행 {duplicate}개 제거')
        sources[key] = {'file': file.name, 'sha256': hashlib.sha256(file.getvalue()).hexdigest(),
                        'columns': mapped, 'rows': len(frame)}
        st.caption(f'{label}: {len(frame):,}개 연결됨')
        return frame.reset_index(drop=True)
    except Exception as error:
        st.error(f'{label}: {error}')
        return None

@st.cache_data(show_spinner='주변 시설의 거리와 개수를 계산하는 중…', max_entries=6)
def cached_metrics(houses, datasets, radius):
    return spatial_metrics(houses, datasets, radius)

sources = {}
with st.sidebar:
    st.header('1. 원자료 업로드')
    demo = st.checkbox('합성 예제 데이터로 기능 확인', value=False)
    if demo:
        root = Path(__file__).parent / 'examples'
        houses = pd.read_csv(root / 'houses.csv', dtype={'house_id': str})
        datasets = {kind: pd.read_csv(root / f'{kind}.csv') for kind in LABELS}
        sources = {'mode': 'synthetic_example_not_real_vacancies'}
    else:
        houses = upload_table('빈집', 'houses', house=True)
        datasets = {}
        with st.container(border=True):
            st.caption('C 유형에는 대학교·지하철역·버스정류장·CCTV·생활편의시설이 필요하다.')
            for kind, label in LABELS.items():
                data = upload_table(label, kind, demand=(kind == 'demand'))
                if data is not None:
                    datasets[kind] = data

    st.header('2. 반경')
    evaluation_radius = st.number_input('점수 계산 반경 (km)', 0.1, 50.0, 5.0, 0.1)
    follow = st.checkbox('시설 표시 반경을 평가 반경과 맞추기', value=True)
    display_radius = evaluation_radius if follow else st.number_input('시설 표시 반경 (km)', 0.1, 50.0, 5.0, 0.1)
    st.caption('가장 가까운 대학·산단·관광자원은 표시 반경 밖이어도 표시한다.')

    st.header('3. 임시 점수화 기준')
    thresholds = {}
    with st.expander('거리·개수 기준 조정'):
        st.caption('검증된 정책 기준이 아닌 기능 확인용 초기값이다. 연구 기준 확정 후 변경한다.')
        for category in NAMES:
            thresholds[f'{category}_distance'] = st.number_input(
                f'{LABELS[ANCHOR[category]]} 접근성 0점 거리 (km)', 0.1, 100.0, 10.0, 0.1)
        for key, label, default in [('transport', '버스+지하철 합산 100점 개수', 100.0),
                                     ('cctv', 'CCTV 100점 개수', 100.0),
                                     ('convenience', '편의시설 100점 개수', 30.0),
                                     ('demand', '관광수요 value 합계 100점 기준', 1000.0)]:
            thresholds[key] = st.number_input(label, min_value=1.0, value=default, step=1.0)

    st.header('4. 가중치')
    if st.button('동일가중치로 초기화'):
        for c in NAMES:
            for j in range(4):
                st.session_state[f'w_{c}_{j}'] = 25
    weights = {}
    for c, name in NAMES.items():
        with st.expander(f'{c} · {name}'):
            labels = [f'{LABELS[ANCHOR[c]]} 접근성', '대중교통', 'CCTV 분포', '관광수요' if c == 'A' else '생활편의시설']
            values = [st.slider(label, 0, 100, 25, key=f'w_{c}_{j}') for j, label in enumerate(labels)]
            if not sum(values):
                st.error('가중치를 모두 0으로 설정할 수 없다.')
                st.stop()
            weights[c] = [v / sum(values) for v in values]
            st.caption('적용 가중치: ' + ' / '.join(f'{v:.3f}' for v in weights[c]))

if demo:
    st.warning('합성 예제 모드다. 실제 빈집이나 실제 시설 위치·평가 결과가 아니다.')
if houses is None:
    st.info('왼쪽에서 빈집 파일을 올린다. 기존 house_id / latitude / longitude 형식을 사용할 수 있다.')
    st.stop()
st.info('점수는 임시 기준이다. CCTV 개수는 안전환경 전체를 대변하지 않는다. 거리 계산은 도로·경사·이동시간을 반영하지 않는다.')
with st.expander('데이터 범위와 점수 해석'):
    st.write('행정구역 경계 밖 시설도 포함해야 경계 부근 빈집의 시설 개수가 과소계산되지 않는다. 최근접 시설은 업로드된 자료 안에서 가장 가까운 시설이다.')
    st.write('파일 미업로드는 0개와 다르다. 실제로 시설이 없음을 확인한 경우에만 열 이름만 있는 빈 파일을 올린다. 대학·산단·관광자원은 최근접 거리를 구하려면 최소 1개가 필요하다.')
    st.write('관광수요 자료는 비교 가능한 같은 단위·기간의 비음수 value를 사용한다. 행정구역 총량을 임의의 중심점에 넣으면 원형 반경 합산이 왜곡될 수 있다.')

raw = cached_metrics(houses, datasets, evaluation_radius)
scored, available, missing = score_metrics(raw, datasets, thresholds, weights)
for c in NAMES:
    if missing[c]:
        st.caption(f'{c} 계산 불가: ' + ', '.join(missing[c]))
if available and len(available) < 3:
    st.warning('현재 ' + ', '.join(available) + '만 계산 가능하다. 추천 유형은 계산 가능한 유형끼리만 비교한 결과다.')

mode = st.radio('순위 비교 방식', ['추천된 유형 안에서 순위', '전체 빈집을 한 유형으로 평가'], horizontal=True)
if mode == '추천된 유형 안에서 순위':
    category_filter = st.selectbox('지도에 표시할 유형', ['전체'] + available)
    ranks = rank_houses(scored, available, True)
else:
    category_filter = st.selectbox('평가 유형', available or ['계산 불가'])
    ranks = rank_houses(scored, [category_filter] if category_filter in available else [], False)
if not ranks.empty and category_filter != '전체':
    ranks = ranks.loc[ranks['rank_category'] == category_filter].copy()
top = ranks.loc[ranks['rank'] <= 5].copy() if not ranks.empty else ranks.copy()
show_others = st.checkbox('나머지 빈집도 작은 회색 점으로 표시', value=False)
st.caption('상위 5개를 유형별 표시한다. 동점은 빈집 ID 오름차순으로 구분한다. 유형 최고점 동점은 A→B→C 순으로 대표 유형을 표시하며 tied_categories에 동점을 기록한다.')

settings = {'evaluation_radius_km': evaluation_radius, 'display_radius_km': display_radius,
            'thresholds': thresholds, 'weights': weights, 'sources': sources, 'mode': mode,
            'category_filter': category_filter, 'score_model': 'provisional_linear_distance_capped_counts_v1'}
signature_settings = {k: v for k, v in settings.items() if k != 'display_radius_km'}
signature = hashlib.sha256(json.dumps(signature_settings, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
if st.session_state.get('signature') != signature:
    st.session_state.signature = signature
    st.session_state.selected_house = None
    st.session_state.epoch = st.session_state.get('epoch', 0) + 1
st.session_state.setdefault('selected_house', None)
st.session_state.setdefault('epoch', 0)

def point_layer(layer_id, data, radius=6, color='color', pickable=True):
    return pdk.Layer('ScatterplotLayer', id=layer_id, data=data, get_position='[longitude, latitude]',
                     get_fill_color=color, get_radius=radius, radius_units='pixels', pickable=pickable)

def text_layer(layer_id, data):
    return pdk.Layer('TextLayer', id=layer_id, data=data, get_position='[longitude, latitude]',
                     get_text='badge', get_size=14, get_color=[255,255,255],
                     get_text_anchor='middle', get_alignment_baseline='center', character_set='auto', pickable=True)

def tooltip_house(row):
    values = '\n'.join(f'{c}: {row[f"{c}_score"]:.2f}점' for c in available)
    return f'{row["house_id"]} · {row["badge"]}\n{values}\n클릭하면 주변 시설을 표시한다.'

def deck(layers, lat, lon, zoom):
    return pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=float(lat), longitude=float(lon), zoom=zoom),
                    map_provider='carto', map_style='light', tooltip={'text': '{tooltip}'})

selection = st.session_state.selected_house
if selection is None:
    layers = []
    if show_others or top.empty:
        background = houses.copy()
        background['tooltip'] = background['house_id'] + ' · 일반 빈집'
        layers.append(point_layer('background', background, 3, [140, 140, 140, 110], False))
    if not top.empty:
        top['color'] = top['rank_category'].map(COLORS)
        top['tooltip'] = top.apply(tooltip_house, axis=1)
        layers += [point_layer('top_houses', top, 19), text_layer('top_labels', top)]
    event = st.pydeck_chart(deck(layers, houses.latitude.median(), houses.longitude.median(), 12),
                           key=f'overview_{st.session_state.epoch}', on_select='rerun', selection_mode='single-object', height=620)
    for layer_id in ['top_houses', 'top_labels']:
        objects = event.selection.objects.get(layer_id, [])
        if objects:
            obj = objects[0]
            st.session_state.selected_house = {'house_id': obj['house_id'], 'category': obj['rank_category'], 'badge': obj['badge']}
            st.rerun()
    if not top.empty:
        candidates = top['badge'] + ' · ' + top['house_id']
        choice = st.selectbox('겹친 점은 목록에서 선택', candidates.tolist())
        if st.button('선택한 빈집의 주변 시설 보기'):
            row = top.iloc[candidates.tolist().index(choice)]
            st.session_state.selected_house = {'house_id': row.house_id, 'category': row.rank_category, 'badge': row.badge}
            st.rerun()
else:
    if st.button('← 빈집 순위 지도로 돌아가기'):
        st.session_state.selected_house = None
        st.session_state.epoch += 1
        st.rerun()
    house = scored.loc[scored.house_id == selection['house_id']].iloc[0]
    category = selection['category']
    st.subheader(f'{selection["badge"]} · {house.house_id} · {NAMES[category]}')
    st.write(f'해당 유형 점수: {house[f"{category}_score"]:.2f}점 / 평가 반경 {evaluation_radius:g} km / 표시 반경 {display_radius:g} km')
    evidence = evidence_for(house, category, datasets, display_radius, evaluation_radius)
    layers = []
    # Geographic circle: angular destination on a sphere, not Web-Mercator metres.
    bearings = np.linspace(0, 2*np.pi, 181)
    lat, lon = np.radians([house.latitude, house.longitude])
    angle = display_radius / 6371.0088
    circle_lat = np.arcsin(np.sin(lat)*np.cos(angle) + np.cos(lat)*np.sin(angle)*np.cos(bearings))
    circle_lon = lon + np.arctan2(np.sin(bearings)*np.sin(angle)*np.cos(lat), np.cos(angle)-np.sin(lat)*np.sin(circle_lat))
    path = np.column_stack([np.degrees(circle_lon), np.degrees(circle_lat)]).tolist()
    layers.append(pdk.Layer('PathLayer', id='radius', data=[{'path': path}], get_path='path',
                            get_color=[70, 90, 120, 180], get_width=2, width_units='pixels'))
    if not evidence.empty:
        for kind in evidence['kind'].unique():
            frame = evidence.loc[evidence.kind == kind].copy()
            frame['color'] = [FACILITY_COLORS[kind]] * len(frame)
            frame['tooltip'] = frame.apply(lambda row: f'{row["시설유형"]}: {row["name"]}\n직선거리 {row.distance_km:.3f} km\n' +
                ('평가에 포함' if row.used_in_score else '표시 반경에만 포함 · 평가에는 미포함'), axis=1)
            layers.append(point_layer(f'facility_{kind}', frame, 6))
        anchor_points = evidence.loc[evidence.kind == ANCHOR[category]]
        if not anchor_points.empty:
            destination = anchor_points.iloc[0]
            layers.append(pdk.Layer('LineLayer', id='nearest_link', data=[{'source': [house.longitude, house.latitude],
                'target': [destination.longitude, destination.latitude]}], get_source_position='source',
                get_target_position='target', get_color=[138,61,191], get_width=2))
    selected_frame = pd.DataFrame([{'longitude': house.longitude, 'latitude': house.latitude, 'badge': selection['badge'],
        'color': COLORS[category], 'tooltip': f'{house.house_id}\n{category}: {house[f"{category}_score"]:.2f}점'}])
    layers += [point_layer('selected_house', selected_frame, 21), text_layer('selected_label', selected_frame)]
    farthest = float(evidence.distance_km.max()) if not evidence.empty else 0
    extent = max(display_radius, farthest, 0.2)
    zoom = float(np.clip(np.log2(18000 / extent), 2, 15))
    st.pydeck_chart(deck(layers, house.latitude, house.longitude, zoom), key='detail_map', height=620)
    st.caption('보라: 대학·산단·관광자원 / 파랑: 지하철 / 하늘색: 버스 / 노랑: CCTV / 분홍: 편의시설·관광수요')
    if display_radius < evaluation_radius:
        st.warning('표시 반경이 더 작아서 평가에 사용한 일부 시설이 지도에서 생략된다. 전체 근거를 보려면 반경을 맞춘다.')
    labels = [f'{LABELS[ANCHOR[category]]} 접근성', '대중교통', 'CCTV 분포', '관광수요' if category == 'A' else '편의시설']
    raw_values = [f'{house[f"{ANCHOR[category]}_nearest_km"]:.3f} km',
                  f'{int(house.subway_count)}개 역 + {int(house.bus_count)}개 정류장',
                  f'{int(house.cctv_count)}개', str(house.demand_sum) if category == 'A' else f'{int(house.convenience_count)}개']
    st.dataframe(pd.DataFrame({'지표': labels, '원자료에서 계산한 값': raw_values,
        '지표 점수': [house[f'{category}_indicator_{j}'] for j in range(1,5)],
        '가중치': weights[category], '가중 점수': [house[f'{category}_indicator_{j+1}']*weights[category][j] for j in range(4)]}), hide_index=True)
    if not evidence.empty:
        st.write(f'표시한 시설: {len(evidence):,}개 — 최근접 기준시설 1개 포함')
        st.dataframe(evidence.drop(columns=['kind']).sort_values('distance_km'), hide_index=True)
        st.download_button('현재 표시 시설 CSV', evidence.to_csv(index=False).encode('utf-8-sig'), 'facility_evidence.csv', 'text/csv')

st.subheader('순위와 계산 결과')
if not ranks.empty:
    first = ['house_id', 'label', 'best_score', 'detail', 'badge', 'rank_category', 'rank', 'rank_score', 'category', 'tied_categories']
    st.dataframe(ranks[first + [c for c in ranks.columns if c not in first]], hide_index=True)
    st.download_button('순위·원지표·점수 CSV', ranks.to_csv(index=False).encode('utf-8-sig'), 'vacant_house_ranking.csv', 'text/csv')
else:
    pending = houses.copy()
    pending['label'] = '미산정'
    pending['best_score'] = np.nan
    pending['detail'] = '유형별 필수 원자료를 연결하면 계산된다.'
    first = ['house_id', 'label', 'best_score', 'detail']
    st.dataframe(pending[first + [c for c in pending.columns if c not in first]], hide_index=True)
st.download_button('적용 기준·가중치·자료 기록 JSON', json.dumps(settings, ensure_ascii=False, indent=2), 'analysis_settings.json', 'application/json')
