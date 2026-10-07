import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from pathlib import Path
from upload_tools import KEYWORDS, first_match, zip_tables, deduplicate_facilities, merge_police, split_amenities
from visuals import icon_layers, legend_html
from report_io import read_file, normalize, validate_coordinates
from report_model import LABELS, DISTANCE, TYPES, FACTOR_NAMES, spatial_raw, evaluate, evidence, DEFAULT_WEIGHTS, P90, RADII, MODEL_VERSION, nonnegative
from hyb_io import density_series, dorm_inventory, calculate_dorm_unmet

st.set_page_config(page_title='구해줘 빈집 | 입지적합도',layout='wide')
st.markdown('''<style>
.stApp {background:#042A5D;color:#FAFAFA}
[data-testid="stSidebar"] {background:#103B6E}
[data-testid="stHeader"] {background:#042A5D}
h1,h2,h3 {color:#FAFAFA !important}
[data-testid="stSidebar"] h2 {color:#01A8AF !important}
.stButton button {border-color:#01A8AF}
.stButton button:hover {border-color:#FA6C41;color:#FA6C41}
</style>''',unsafe_allow_html=True)
st.sidebar.image(str(Path(__file__).parent/'assets'/'logo.png'),use_container_width=True)
st.title('구해줘 빈집')
st.subheader('부산 빈집 유형별 입지적합도')
st.caption('HYB 관광 · 부산 격자 p90 기준 · 시설 750m / 관광 1.5km / 기숙사 3km')
sources={}
pool={}
with st.sidebar:
    st.header('데이터 업로드 방식')
    upload_mode=st.radio('업로드 방식',['ZIP으로 한번에','각각 업로드'])
    if upload_mode=='ZIP으로 한번에':
        bundle=st.file_uploader('데이터 ZIP',type=['zip'],key='bundle')
        if bundle:
            try: pool=zip_tables(bundle.getvalue())
            except (ValueError, OSError, zipfile.BadZipFile) as error: st.error(str(error))
        extras=st.file_uploader('공통 목록에 데이터 추가',type=['csv','xlsx'],accept_multiple_files=True,key='extras')
        for index,extra in enumerate(extras): pool[f'추가/{index+1}_{extra.name}']=extra.getvalue()
        st.caption(f'선택 가능한 CSV·XLSX {len(pool)}개. 자동 선택 결과는 각 입력칸에서 변경할 수 있다.')
@st.cache_data(show_spinner=False)
def read_cached(data,name): return read_file(data,name)
def load(label,key):
    data=None; name=None
    if upload_mode=='ZIP으로 한번에':
        options=['선택 필요']+list(pool)
        default=first_match(list(pool),KEYWORDS.get(key,[key])) or '선택 필요'
        pool_key=hashlib.sha256(('\n'.join(pool)).encode()).hexdigest()[:10]
        choice=st.selectbox(label+' 데이터 파일',options,index=options.index(default),key=f'file_choice_{key}_{pool_key}')
        replacement=st.file_uploader(label+' 파일 추가·교체 (선택)',type=['csv','xlsx'],key='override_'+key)
        if replacement is not None: data=replacement.getvalue(); name=replacement.name
        elif choice!='선택 필요': data=pool[choice]; name=choice
    else:
        file=st.file_uploader(label,type=['csv','xlsx'],key=key)
        if file is not None: data=file.getvalue(); name=file.name
    if data is None: return None
    sources[key]={'file':name,'sha256':hashlib.sha256(data).hexdigest()}
    try:
        frame=read_cached(data,name)
        frame.attrs['upload_hash']=sources[key]['sha256']
        return frame
    except Exception as error:
        st.error(f'{label} 파일 읽기 실패: {error}')
        return None
def column(source,label,key,aliases,optional=False):
    choices=['선택 필요']+list(source.columns)
    keywords = (['위도','latitude','lat'] if label=='위도' else
                ['경도','longitude','lng','lon'] if label=='경도' else
                ['이름','명','name','nm','nam'] if label.startswith('시설명') else None)
    default=(first_match(list(source.columns),keywords) if keywords else next((a for a in aliases if a in source),None)) or '선택 필요'
    source_id=source.attrs.get('upload_hash','')[:12]
    value=st.selectbox(label,choices,index=choices.index(default),key=key+'_'+source_id)
    if value=='선택 필요':
        if optional: return None
        st.info(label+' 열을 선택한다.'); st.stop()
    sources.setdefault('columns',{})[key]=value
    return value
def facility_upload(kind):
    ui_label={'police_station':'경찰서','police_local':'지구대,파출소','amenities':'편의시설'}.get(kind,LABELS.get(kind,kind))
    source=load(ui_label,kind)
    if source is None: return None
    lat=column(source,'위도',kind+'lat',['latitude','위도','역위도'])
    lon=column(source,'경도',kind+'lon',['longitude','경도','역경도'])
    name=column(source,'시설명 (선택)',kind+'name',['name','시설명','명칭','역사명'],True)
    identifier=column(source,'시설 ID (선택)',kind+'id',['facility_id','id','ID'],True)
    if lat==lon: raise ValueError('위도와 경도는 서로 다른 열을 선택한다.')
    frame=pd.DataFrame({'latitude':source[lat],'longitude':source[lon]})
    frame['name']=source[name].fillna('시설명 없음') if name else [f'{ui_label} {i+1}' for i in range(len(frame))]
    if kind == 'amenities':
        type_col=column(source,'관광지 유형 (필수)' if kind=='attraction' else '시설 종류 (필수)',kind+'type',['type','유형','시설종류','시설유형','업종','관광지유형','분류'])
        frame['type']=normalize(source[type_col])
    if kind=='attraction':
        pop_col=column(source,'관광지 등장횟수 appear_cnt (필수)',kind+'pop',['appear_cnt','등장횟수','출현횟수'])
        frame['appear_cnt']=nonnegative(source[pop_col])
    if kind=='dorm' or (kind=='university' and dorm_mode=='대학교 파일에 포함'):
        app_col=column(source,'기숙사 지원자 수 (필수)',kind+'app',['dorm_app','지원자 수','지원자수','기숙사지원자수'])
        cap_col=column(source,'수용가능인원 (필수)',kind+'cap',['dorm_cap','수용가능인원','수용가능 인원','기숙사수용인원'])
        if app_col==cap_col: raise ValueError('지원자 수와 수용가능인원은 서로 다른 열을 선택한다.')
        frame['dorm_app']=nonnegative(source[app_col],allow_missing=True)
        frame['dorm_cap']=nonnegative(source[cap_col],allow_missing=True)
        frame['unmet']=calculate_dorm_unmet(frame['dorm_app'],frame['dorm_cap'])
        st.caption('미충족 인원 = max(지원자 수 − 수용가능인원, 0). 기존 unmet 열은 읽지 않는다.')
    if identifier:
        frame['facility_id']=normalize(source[identifier])
        if frame.facility_id.isna().any(): raise ValueError('시설 ID에 빈칸이 있다.')
    before=len(frame)
    frame=validate_coordinates(frame)
    if 'unmet' in frame: frame=dorm_inventory(frame)
    else: frame=deduplicate_facilities(frame)
    if frame.empty and not st.checkbox('조사 범위 내 실제 시설 0개인 완비된 자료임을 확인했다',key='empty_'+kind):
        st.info('빈 파일은 누락 자료인지 실제 0개인지 확인한다.'); return None
    if kind=='dorm': frame=dorm_inventory(frame)
    if 'unmet' in frame and frame.unmet.isna().any(): st.warning(f'기숙사 미연결 대학 {frame.unmet.isna().sum()}개: 해당 대학 3km 내 빈집의 D는 미산정한다.')
    st.caption(f'{len(frame)}개 연결 / 중복 {before-len(frame)}개 제거')
    return frame
def stat_upload(code):
    source=load('관광업종 매출밀도','sales')
    if source is None: return None,None
    admin=column(source,'행정동 열','sales_admin',['admin_code','행정동코드'] if code else ['행정동명','admin_name','행정동'])
    value=column(source,'관광업종 매출밀도 sales_den (필수)','sales_den',['sales_den','관광업종매출밀도','매출밀도'])
    month=column(source,'기준년월 (있으면 선택)','sales_month',['month','기준년월'],True)
    st.caption('매출액 자체가 아닌, PDF p90과 같은 산정 방식·단위의 매출밀도를 연결한다. 업종별 원자료는 먼저 집계한다.')
    series,latest,unnamed=density_series(source,admin,value,code,month)
    sources['sales_settings']={'month':latest,'unnamed_rows_excluded':unnamed,'value_column':value}
    st.caption(f'기준월 {latest or "파일에 없음"} / {len(series)}개 행정동 / 연결키 없는 행 {unnamed}개 제외')
    return series,latest

with st.sidebar:
    st.header('1. 빈집 자료')
    source=load('빈집 파일','houses')
    if source is None: st.info('ID·위도·경도·행정동이 포함된 파일을 올린다.'); st.stop()
    identifier=column(source,'빈집 ID','h_id',['house_id','id','ID'])
    lat=column(source,'위도','h_lat',['latitude','위도'])
    lon=column(source,'경도','h_lon',['longitude','경도'])
    code=st.radio('행정동 연결 방식',['행정동코드','행정동명'])=='행정동코드'
    admin=column(source,'행정동 (필수)','h_admin',['행정동코드','admin_code'] if code else ['행정동명','admin_name','행정동'])
    address=column(source,'주소 (선택)','h_address',['address','주소'],True)
    if len({identifier,lat,lon,admin})!=4: st.error('필수 네 항목은 서로 다른 열을 선택한다.'); st.stop()
    houses=pd.DataFrame({'house_id':source[identifier],'latitude':source[lat],'longitude':source[lon],'admin_key':normalize(source[admin],code)})
    if address: houses['address']=source[address]
    try: houses=validate_coordinates(houses,True)
    except ValueError as error: st.error(str(error)); st.stop()
    st.caption('행정동명은 구·군을 포함해 통계 자료와 동일하게 입력한다. 주소에서 추정하지 않는다.')
    st.header('2. 시설 원자료')
    st.markdown('**A 관광체류형:** HYB 관광·관광업종 매출밀도·교통·안전·생활편의\n\n**B 근로자 정주임대형:** 산업접근·교육/보육·교통·안전·생활편의\n\n**C 대학생 정주임대형:** 대학접근·기숙사 부족·교통·안전·생활편의')
    facilities={}
    dorm_mode=st.radio('기숙사 자료 입력',['대학교 파일에 포함','별도 좌표 파일'])
    st.caption('대학/캠퍼스별 지원자 수·수용가능인원을 연결한다. 두 값 중 하나라도 누락되면 미산정한다.')
    police_mode=st.radio('경찰시설 입력',['통합 파일','경찰서 / 지구대·파출소 별도'])
    amenity_mode=st.radio('편의시설 입력',['편의점 / 마트 / 약국 별도','편의시설 통합 파일'])
    kinds=[k for k in LABELS if not (k=='dorm' and dorm_mode=='대학교 파일에 포함') and not (k=='police' and police_mode!='통합 파일')
           and not (k in ['store','mart','pharmacy'] and amenity_mode=='편의시설 통합 파일')]
    for kind in kinds:
        with st.expander(LABELS[kind]):
            try:
                frame=facility_upload(kind)
                if frame is not None:
                    facilities[kind]=frame
                    if kind=='university' and dorm_mode=='대학교 파일에 포함': facilities['dorm']=dorm_inventory(frame)
            except (ValueError,KeyError) as error: st.error(str(error))
    if police_mode!='통합 파일':
        police_parts=[]
        for kind in ['police_station','police_local']:
            with st.expander('경찰서' if kind=='police_station' else '지구대,파출소'):
                try: police_parts.append(facility_upload(kind))
                except (ValueError,KeyError) as error: st.error(str(error)); police_parts.append(None)
        if all(frame is not None for frame in police_parts):
            combined=merge_police(*police_parts)
            if len(combined): facilities['police']=combined
            else: st.error('경찰시설은 합쳐서 최소 1개가 필요하다.')
    if amenity_mode=='편의시설 통합 파일':
        with st.expander('편의시설 통합',expanded=True):
            try:
                combined=facility_upload('amenities')
                if combined is not None:
                    mapping={}; complete=True
                    choices=['선택 필요','편의점','마트','약국']
                    for index,value in enumerate(combined['type'].dropna().unique()):
                        guess=next((label for label in choices[1:] if label in str(value)),'선택 필요')
                        selected=st.selectbox(f'{value} → 시설 종류',choices,index=choices.index(guess),key=f'amenity_map_{index}_{value}')
                        if selected=='선택 필요': complete=False
                        else: mapping[value]={'편의점':'store','마트':'mart','약국':'pharmacy'}[selected]
                    coverage=st.checkbox('통합 자료가 편의점·마트·약국 세 종류를 모두 조사한 자료임을 확인했다')
                    if complete and coverage:
                        facilities.update(split_amenities(combined,mapping))
                        sources['amenity_type_mapping']=mapping
            except (ValueError,KeyError) as error: st.error(str(error))
    st.header('3. 관광업종 매출밀도')
    try:
        with st.expander('행정동별 매출밀도',expanded=True): sales,sales_month=stat_upload(code)
    except (ValueError,KeyError) as error: st.error(str(error)); st.stop()
    confirmed=st.checkbox('매출밀도 단위와 기준기간·기숙사 자료의 조사범위를 확인했다')
    st.header('4. 기본 가중치와 조정')
    st.caption('PDF p90 고정 기준을 적용한다. 로그변환과 표본 Min–Max 정규화는 사용하지 않는다.')
    with st.expander('점수화 기준'):
        st.dataframe(pd.DataFrame({'원지표':list(P90),'p90 기준값':list(P90.values())}),hide_index=True)
        st.caption('거리: ≤750m 1 / ≤1.5km 0.75 / ≤2.25km 0.5 / ≤3km 0.25 / >3km 0. 시설 유무는 750m 기준.')
    for c in TYPES:
        for f,v in zip(TYPES[c],DEFAULT_WEIGHTS[c]):
            key=f'hyb_w_{c}_{f}'
            if key not in st.session_state: st.session_state[key]=int(round(v*100))
    if st.button('확정 가중치로 초기화'):
        for c in TYPES:
            for f,v in zip(TYPES[c],DEFAULT_WEIGHTS[c]): st.session_state[f'hyb_w_{c}_{f}']=int(round(v*100))
    weights={}
    for c,factors in TYPES.items():
        with st.expander(c+' 가중치',expanded=True):
            values=[st.slider(FACTOR_NAMES[f],0,100,key=f'hyb_w_{c}_{f}') for f,v in zip(factors,DEFAULT_WEIGHTS[c])]
            if sum(values)==0: st.error('가중치 합계는 0보다 커야 한다.'); st.stop()
            weights[c]=[v/sum(values) for v in values]
            st.caption('현재 적용: '+' / '.join(f'{f} {w:.3f}' for f,w in zip(factors,weights[c])))
            st.caption('기본값: '+' / '.join(f'{f} {w:.1f}' for f,w in zip(factors,DEFAULT_WEIGHTS[c])))

missing=[LABELS[k] for k in LABELS if k not in facilities]
if sales is None: missing.append('관광업종 매출밀도 sales_den')
if missing: st.info('공통 분석에 필요한 자료: '+', '.join(missing)); st.stop()
if not confirmed: st.info('자료의 단위·조사범위를 확인한 뒤 왼쪽 확인란을 선택한다.'); st.stop()
connection=pd.DataFrame({'빈집 ID':houses.house_id,'행정동':houses.admin_key,'매출밀도':houses.admin_key.map(sales)})
with st.expander('행정동 매출밀도 연결 확인',expanded=bool(connection['매출밀도'].isna().any())):
    st.write(f"매출밀도 연결 {connection['매출밀도'].notna().sum():,}/{len(houses):,}개")
    st.dataframe(connection,hide_index=True)
    st.write('통계 행정동 예시:',', '.join(str(v) for v in sales.index[:10]))
    if connection['매출밀도'].isna().any(): st.download_button('행정동 연결 진단 CSV',connection.to_csv(index=False).encode('utf-8-sig'),'admin_connections.csv')

@st.cache_data(show_spinner='HYB 공간지표 계산 중',max_entries=4)
def raw_calculation(houses,facilities): return spatial_raw(houses,facilities)
try:
    raw=raw_calculation(houses,facilities)
    raw['sales_den']=raw.admin_key.map(sales)
    from diagnostics import cohort_diagnostics
    valid,diagnostic_counts,diagnostic_rows=cohort_diagnostics(raw)
    if not valid.any():
        st.error('공통 분석 대상이 0개다. 매출밀도 연결과 3km 내 대학의 기숙사 미연결값을 확인한다.')
        st.dataframe(diagnostic_counts,hide_index=True); st.dataframe(diagnostic_rows,hide_index=True)
        st.download_button('빈집별 제외 이유 CSV',diagnostic_rows.to_csv(index=False).encode('utf-8-sig'),'exclusion_reasons.csv'); st.stop()
    result,excluded,audit,disabled=evaluate(raw,weights)
except ValueError as error: st.error(str(error)); st.stop()
st.write(f"공통 분석 {len(result):,}개 / 전체 {len(houses):,}개 / 기준월 {sales_month or '파일에 없음'}")
with st.expander('점수화 기준·제외 자료'):
    st.dataframe(audit,hide_index=True)
    st.caption('모든 원지표는 고정 기준으로 0~1점화한다. 상수여도 제외·재가중하지 않는다. 최종 점수는 100배로 표시한다.')
    if len(excluded):
        st.dataframe(diagnostic_rows,hide_index=True)
        st.download_button('미연결 빈집 CSV',excluded.to_csv(index=False).encode('utf-8-sig'),'excluded.csv')
category=st.radio('독립 평가 유형',['A','B','C'],horizontal=True)
settings={'model':MODEL_VERSION,'interface':'v5','upload_mode':upload_mode,'police_mode':police_mode,
 'amenity_mode':amenity_mode,'dorm_mode':dorm_mode,'radii_km':RADII,'p90':P90,'month':sales_month,
 'weights':weights,'default_weights':DEFAULT_WEIGHTS,'logs':[],'sources':sources,'common_N':len(result),
 'rank':'competition','percentile':'midrank','factor_range':[0,1],'final_score_range':[0,100],
 'dorm_formula':'sum_within_3km(max(dorm_app-dorm_cap,0))/661 capped at 1',
 'distance_bins':'<=750m:1; <=1500m:.75; <=2250m:.5; <=3000m:.25; >3000m:0'}
signature=hashlib.sha256(json.dumps([settings,category],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
if st.session_state.get('signature')!=signature:
    st.session_state.signature=signature; st.session_state.selected=None
    st.session_state.epoch=st.session_state.get('epoch',0)+1
colors={'A':[231,101,50],'B':[42,112,212],'C':[22,143,92]}
def points(id,data,radius,color):
    return pdk.Layer('ScatterplotLayer',id=id,data=data,get_position='[longitude,latitude]',get_fill_color=color,
        get_radius=radius,radius_units="'pixels'",radius_min_pixels=radius,radius_max_pixels=radius,pickable=True)
def texts(data):
    return pdk.Layer('TextLayer',id='labels',data=data,get_position='[longitude,latitude]',get_text='badge',get_size=14,
        get_color=[255,255,255],get_text_anchor="'middle'",get_alignment_baseline="'center'",character_set="'auto'",pickable=True)
def chart(layers,lat,lon,zoom):
    return pdk.Deck(layers=layers,initial_view_state=pdk.ViewState(latitude=float(lat),longitude=float(lon),zoom=zoom),
                    map_provider='carto',map_style='light',tooltip={'text':'{tooltip}'})
top=result.loc[result[f'{category}_rank']<=5].copy()
top['badge']=category+':'+top[f'{category}_rank'].astype(int).astype(str)+'위'
top['tooltip']=top.apply(lambda r:f'{r.house_id} · {r.badge}\n{r[f"{category}_score"]:.2f}점',axis=1) if len(top) else ''
if st.session_state.get('selected') is None:
    st.caption('각 유형 5위 이내 표시. 공동순위가 있으면 5개보다 많을 수 있다.')
    layers=[]
    if st.checkbox('나머지 빈집도 표시') or top.empty:
        background=houses.copy(); background['tooltip']=background.house_id
        layers.append(points('background',background,3,[150,150,150,100]))
    if len(top): layers += [points('top',top,30,colors[category]),texts(top)]
    event=st.pydeck_chart(chart(layers,houses.latitude.median(),houses.longitude.median(),12),
        on_select='rerun',selection_mode='single-object',key=f'map_{st.session_state.epoch}',height=600)
    for layer in ['top','labels']:
        chosen=event.selection.objects.get(layer,[])
        if chosen: st.session_state.selected=chosen[0]['house_id']; st.rerun()
    if len(top):
        chosen=st.selectbox('겹친 점은 ID로 선택',top.house_id.tolist())
        if st.button('주변 근거시설 보기'): st.session_state.selected=chosen; st.rerun()
else:
    if st.button('← 순위 지도로 돌아가기'):
        st.session_state.selected=None; st.session_state.epoch+=1; st.rerun()
    row=result.loc[result.house_id==st.session_state.selected].iloc[0]
    st.subheader(f'{row.house_id} · {category}:{int(row[f"{category}_rank"])}위')
    facility=evidence(row,category,facilities,{k:.75 for k in LABELS})
    layers=[]
    if len(facility):
        facility['tooltip']=facility.apply(lambda r:f'{r.facility_type}: {r["name"]}\n{r.distance_km:.3f}km',axis=1)
        layers.extend(icon_layers(facility))
    selected=pd.DataFrame([{'latitude':row.latitude,'longitude':row.longitude,
        'badge':category+':'+str(int(row[f'{category}_rank']))+'위','tooltip':str(row.house_id)}])
    layers += [points('selected',selected,32,colors[category]),texts(selected)]
    extent=max(.75,float(facility.distance_km.max()) if len(facility) else .75)
    st.pydeck_chart(chart(layers,row.latitude,row.longitude,float(np.clip(np.log2(16000/extent),2,15))),height=600)
    st.markdown(legend_html(),unsafe_allow_html=True)
    st.caption('관광지 1.5km · 기숙사 3km · 유무/개수 시설 750m · 거리 단계 시설은 최근접 시설을 표시한다.')
    factors=TYPES[category]
    st.dataframe(pd.DataFrame({'요인':[FACTOR_NAMES[f] for f in factors],'점수 (0~1)':[row[f] for f in factors],
        '가중치':weights[category],'기여점수 (100점 기준)':[100*row[f]*w for f,w in zip(factors,weights[category])]}),hide_index=True)
    if category=='A': st.write(f'행정동 {row.admin_key} / 관광업종 매출밀도 {row.sales_den:,.2f} / 기준월 {sales_month or "파일에 없음"}')
    if category=='C': st.write(f'3km 내 기숙사 미충족 인원 합계 {row.dorm_unmet:,.0f}명')
    st.dataframe(facility,hide_index=True)
st.subheader('유형별 점수·공동순위·백분위')
columns=['house_id','admin_key']+[f'{c}_{field}' for c in TYPES for field in ['score','rank','percentile']]
st.dataframe(result[columns].sort_values(f'{category}_rank'),hide_index=True)
with st.expander('확정 기본 가중치 대비 순위 변화'):
    st.dataframe(result[['house_id']+[f'{c}_{f}' for c in TYPES for f in ['baseline','baseline_rank','rank_change']]],hide_index=True)
st.download_button('전체 계산 결과 CSV',result.to_csv(index=False).encode('utf-8-sig'),'report_scores.csv')
st.download_button('분석 설정 JSON',json.dumps(settings,ensure_ascii=False,indent=2),'analysis_settings.json')
