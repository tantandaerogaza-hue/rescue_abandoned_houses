import hashlib
from pathlib import Path
import streamlit as st
from admin_sales import prepare_sales, aggregate_sales, attach_crosswalk, attach_address_dongs

def upload_admin_sales(read_table, sources, houses):
    summary, metadata = None, {}
    st.markdown('**행정동별 소비활성도**')
    st.caption('소비 자료에 위경도는 필요 없다. 빈집 address에서 동명을 추출해 정확히 일치하는 행정동에 연결한다. 코드·행정동명을 직접 입력하면 우선 적용한다.')
    crosswalk = st.file_uploader('빈집 ID–행정동 연결표 (빈집 파일에 행정동이 없을 때)', type=['xlsx', 'csv'], key='admin_crosswalk')
    if crosswalk is not None and houses is not None:
        try:
            mapping = read_table(crosswalk.getvalue(), Path(crosswalk.name).suffix.lower())
            mapping.columns = mapping.columns.str.strip()
            aliases = {'행정동코드': 'admin_code', '행정동명': 'admin_name', 'id': 'house_id', 'ID': 'house_id', '빈집ID': 'house_id'}
            mapping = mapping.rename(columns=aliases)
            if mapping.columns.duplicated().any():
                raise ValueError('연결표의 열 이름이 중복된다.')
            houses = attach_crosswalk(houses, mapping)
            sources['admin_crosswalk'] = {'file': crosswalk.name, 'sha256': hashlib.sha256(crosswalk.getvalue()).hexdigest()}
        except Exception as error:
            st.error(f'행정동 연결표: {error}')
            return houses, None, {'error': str(error)}
    file = st.file_uploader('행정동별 평균이용금액 파일', type=['xlsx', 'csv'], key='admin_sales_file')
    if file is None:
        return houses, summary, metadata
    try:
        source = read_table(file.getvalue(), Path(file.name).suffix.lower())
        source.columns = source.columns.str.strip()
        data = prepare_sales(source, latest_only=True)
        months = sorted(data.month.unique().tolist(), reverse=True)
        chosen_months = months[:1]
        st.caption(f'최근 월만 반영: {months[0][:4]}년 {months[0][4:]}월')
        industries = sorted(data.loc[data.month.isin(chosen_months), 'industry'].unique().tolist())
        chosen_industries = st.multiselect('소비 자료 평가 업종', industries, default=industries)
        st.caption('최근 월의 선택 업종 평균이용금액을 합산한다. 평균이용건수로 나누지 않는다.')
        summary, metadata = aggregate_sales(data, chosen_months, chosen_industries)
        metadata['latest_month_only'] = True
        if metadata['unnamed_dongs']:
            st.caption(f'행정동명이 비어 있는 {metadata["unnamed_dongs"]}개 행정동은 코드로만 연결 가능하다.')
        if houses is not None:
            houses = attach_address_dongs(houses, summary)
        st.caption('선택 자료의 행정동 최솟값=0점, 최댓값=100점. 금액이 모두 같으면 50점. 선택 월·업종이 빠진 행정동은 미산정.')
        sources['admin_sales'] = {'file': file.name, 'sha256': hashlib.sha256(file.getvalue()).hexdigest(), 'rows': len(source), 'latest_month_rows': len(data)}
        st.caption(f'최근 월 소비 자료: {len(data):,}행 / 행정동 {len(summary):,}개')
    except Exception as error:
        st.error(f'행정동 소비 자료: {error}')
    return houses, summary, metadata
