import hashlib

import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="부산 빈집 활용 지도", layout="wide")
st.title("부산 빈집 활용 지도")

# 1. 파일 업로드
uploaded = st.file_uploader("빈집 엑셀 파일을 올린다", type=["xlsx"])

if uploaded is None:
    st.info("house_id, latitude, longitude 열이 있는 엑셀 파일이 필요하다.")
    st.stop()

try:
    df = pd.read_excel(uploaded, dtype={"house_id": str})
except Exception as error:
    st.error(f"파일을 읽을 수 없다: {error}")
    st.stop()

required = ["house_id", "latitude", "longitude"]

if not set(required).issubset(df.columns):
    st.error("필수 열 이름을 확인한다: house_id, latitude, longitude")
    st.stop()

if df.empty:
    st.error("파일에 빈집 데이터가 없다.")
    st.stop()

df["house_id"] = df["house_id"].astype("string").str.strip()

if df["house_id"].isna().any() or df["house_id"].eq("").any():
    st.error("빈집 ID에 빈칸이 있다.")
    st.stop()

if df["house_id"].duplicated().any():
    st.error("중복된 빈집 ID가 있다. 빈집마다 다른 ID를 사용한다.")
    st.stop()

for column in ["latitude", "longitude"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

valid = (
    df["latitude"].between(-90, 90)
    & df["longitude"].between(-180, 180)
)

if not valid.all():
    st.error("숫자가 아니거나 범위를 벗어난 좌표가 있다. 아래 행을 수정한다.")
    st.dataframe(df.loc[~valid])
    st.stop()

# 2. 유형별 지표
criteria = {
    "A": ["관광자원 접근성", "대중교통 편의성", "안전환경", "관광수요·소비활성도"],
    "B": ["직장·산단 접근성", "대중교통 편의성", "안전환경", "생활편의시설"],
    "C": ["대학 접근성", "대중교통 편의성", "안전환경", "생활편의시설"],
}

names = {
    "A": "관광체류형",
    "B": "근로자 정주임대형",
    "C": "대학생 정주임대형",
}

demo = st.sidebar.checkbox("시험용 가상 점수 사용", value=False)

# ID와 지표가 같으면 항상 같은 가상 점수를 만든다.
# 실제 위치나 주변 환경을 평가한 점수가 아니다.
def fake_score(house_id, criterion):
    key = f"{house_id}|{criterion}".encode("utf-8")
    number = int(hashlib.sha256(key).hexdigest()[:8], 16)
    return 40 + number % 61

if demo:
    st.warning("시험 모드: 모든 지표 점수와 추천은 가상 결과다.")

    if st.sidebar.button("슬라이더 초기화"):
        for category in criteria:
            for index in range(4):
                st.session_state[f"{category}_{index}"] = 25

    weights = {}
    st.sidebar.caption("슬라이더는 상대 비중이다. 합계가 1이 되도록 자동 환산한다.")

    for category, items in criteria.items():
        with st.sidebar.expander(
            f"{category}: {names[category]}", expanded=(category == "A")
        ):
            raw = [
                st.slider(
                    item, 0, 100, 25, key=f"{category}_{index}"
                )
                for index, item in enumerate(items)
            ]

            total = sum(raw)

            if total == 0:
                st.error("하나 이상의 비중을 0보다 크게 설정한다.")
                st.stop()

            weights[category] = [value / total for value in raw]

            for item, weight in zip(items, weights[category]):
                st.caption(f"{item}: 적용 가중치 {weight:.3f}")

        score = pd.Series(0.0, index=df.index)

        for item, weight in zip(items, weights[category]):
            column = f"시험_{item}"
            df[column] = df["house_id"].map(
                lambda house_id: fake_score(house_id, item)
            )
            score += df[column] * weight

        df[f"{category}_score"] = score

    score_columns = ["A_score", "B_score", "C_score"]
    df["category"] = df[score_columns].idxmax(axis=1).str[0]
    df["best_score"] = df[score_columns].max(axis=1)
    df["label"] = df.apply(
        lambda row: f"{row['category']}({row['best_score']:.1f}점)", axis=1
    )
    df["detail"] = df.apply(
        lambda row: (
            f"시험용 가상 점수\n"
            f"A: {row['A_score']:.1f}점\n"
            f"B: {row['B_score']:.1f}점\n"
            f"C: {row['C_score']:.1f}점"
        ),
        axis=1,
    )
    df["data_status"] = "시험용 가상 점수"
else:
    df["category"] = "미산정"
    df["label"] = df["house_id"] + " · 미산정"
    df["detail"] = "실제 지표 점수 연결 전"
    df["data_status"] = "점수 미산정"

# 3. 지도 색상과 라벨
colors = {
    "A": [230, 100, 60],
    "B": [45, 125, 220],
    "C": [45, 165, 100],
    "미산정": [120, 120, 120],
}
df["color"] = df["category"].map(colors)

show_labels = st.checkbox("지도 위 텍스트 표시", value=True)

layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius=20,
        radius_min_pixels=5,
        radius_max_pixels=12,
        pickable=True,
    )
]

if show_labels:
    layers.append(
        pdk.Layer(
            "TextLayer",
            data=df,
            get_position="[longitude, latitude]",
            get_text="label",
            get_size=13,
            get_color=[30, 30, 30],
            get_pixel_offset=[0, -18],
            character_set="auto",
            pickable=True,
        )
    )

view = pdk.ViewState(
    latitude=float(df["latitude"].median()),
    longitude=float(df["longitude"].median()),
    zoom=11,
)

st.caption("A: 관광체류형 · B: 근로자 정주임대형 · C: 대학생 정주임대형")
st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_provider="carto",
        map_style="light",
        tooltip={"text": "ID: {house_id}\n{label}\n{detail}"},
    )
)

# 4. 상세 정보와 다운로드
st.subheader(f"빈집 목록: {len(df):,}개")
selected = st.selectbox("상세 정보를 확인할 빈집", df["house_id"].tolist())
st.dataframe(df.loc[df["house_id"] == selected].drop(columns=["color"]))

export = df.drop(columns=["color"])
st.dataframe(export)

st.download_button(
    "현재 결과 CSV 다운로드",
    data=export.to_csv(index=False).encode("utf-8-sig"),
    file_name="vacant_house_demo.csv" if demo else "vacant_house_locations.csv",
    mime="text/csv",
)