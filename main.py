import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="2023–2024 인구·의료 비교", page_icon="📊", layout="wide")

# --- UI ---
st.markdown(
    """
    <style>
      :root { --bg:#fbfbff; --card:rgba(255,255,255,0.75); --stroke:rgba(49, 51, 63, 0.14); }
      .stApp { background: var(--bg); }
      .block-container { padding-top: 2rem; padding-bottom: 2rem; }
      .card { border:1px solid var(--stroke); border-radius:18px; padding:14px 16px; background:var(--card);
              box-shadow:0 8px 26px rgba(18, 18, 28, 0.06); }
      .card-title { font-size:0.9rem; opacity:0.78; margin-bottom:6px; }
      .card-value { font-size:1.55rem; font-weight:750; line-height:1.15; }
      .card-sub { font-size:0.8rem; opacity:0.7; margin-top:6px; }
      .section-title { font-size:1.05rem; font-weight:750; margin:0.2rem 0 0.6rem; }
      .hint { font-size:0.92rem; opacity:0.78; }
      .pill { display:inline-block; padding:3px 10px; border-radius:999px; border:1px solid var(--stroke);
              font-size:.78rem; opacity:.85; background:rgba(255,255,255,0.6); margin-right:6px; }
      .small { font-size:.85rem; opacity:.78; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# 파일 설정
# -------------------------
FILES = {
    "wide": "compare_2023_2024_wide.csv",                  # optional (재계산함)
    "long": "compare_2023_2024_long.csv",                  # required
    "pop_raw": "202301_202512_주민등록인구기타현황(인구증감)_월간.csv",  # required (복구 핵심)
}

MODE = st.sidebar.radio("데이터 불러오기", ["폴더에서 읽기(기본)", "파일 업로드"])

def read_csv_safely(fp):
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(fp, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(fp)

@st.cache_data
def load_local():
    wide = None
    try:
        wide = read_csv_safely(FILES["wide"])
    except Exception:
        wide = None
    long = read_csv_safely(FILES["long"])
    pop_raw = read_csv_safely(FILES["pop_raw"])
    return wide, long, pop_raw

def load_upload():
    f_long = st.sidebar.file_uploader(f"업로드: {FILES['long']}", type=["csv"], key="long")
    f_pop  = st.sidebar.file_uploader(f"업로드: {FILES['pop_raw']}", type=["csv"], key="pop_raw")
    f_wide = st.sidebar.file_uploader(f"(선택) 업로드: {FILES['wide']}", type=["csv"], key="wide")

    if (f_long is None) or (f_pop is None):
        st.sidebar.info("업로드 모드에서는 long CSV + 원본 인구 CSV를 꼭 올려야 해요.")
        return None

    wide = read_csv_safely(f_wide) if f_wide is not None else None
    return wide, read_csv_safely(f_long), read_csv_safely(f_pop)

if MODE.startswith("폴더"):
    try:
        wide, long, pop_raw = load_local()
    except Exception as e:
        st.error("폴더에서 파일을 못 찾았어요. 업로드 모드로 바꾸거나 파일명을 확인해 주세요.")
        st.exception(e)
        st.stop()
else:
    loaded = load_upload()
    if loaded is None:
        st.stop()
    wide, long, pop_raw = loaded

# -------------------------
# Helper
# -------------------------
def card(title, value, sub=""):
    st.markdown(
        f"""
        <div class="card">
          <div class="card-title">{title}</div>
          <div class="card-value">{value}</div>
          <div class="card-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def uniq_cols(cols):
    return list(dict.fromkeys(cols))

def add_reg_line(df, x, y):
    d = df[[x, y]].dropna()
    if len(d) < 2:
        return None
    xv = d[x].astype(float).values
    yv = d[y].astype(float).values
    a, b = np.polyfit(xv, yv, 1)
    r = float(np.corrcoef(xv, yv)[0, 1])
    xs = np.array([float(xv.min()), float(xv.max())])
    line = go.Scatter(x=xs, y=a * xs + b, mode="lines", name=f"회귀선 (r={r:.2f})")
    return a, b, r, line

def top_split(df, metric, n=5):
    d = df[["sido", metric]].dropna().copy()
    inc = d.sort_values(metric, ascending=False).head(n)
    dec = d.sort_values(metric, ascending=True).head(n)
    return inc, dec

def build_report_html(title, subtitle, figs, tables):
    parts = []
    parts.append(f"<h1 style='font-family:system-ui; margin:0 0 6px;'>{title}</h1>")
    parts.append(f"<p style='font-family:system-ui; margin:0 0 18px; opacity:.8;'>{subtitle}</p>")
    for t, fig in figs:
        parts.append(f"<h2 style='font-family:system-ui; margin:18px 0 8px;'>{t}</h2>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    for t, df in tables:
        parts.append(f"<h2 style='font-family:system-ui; margin:18px 0 8px;'>{t}</h2>")
        parts.append(df.to_html(index=False))
    return "<html><head><meta charset='utf-8'></head><body style='margin:24px;'>" + "\n".join(parts) + "</body></html>"

# -------------------------
# ✅ (업그레이드 핵심) 원본 인구 파일에서 pop_month 직접 생성 + 강원/전북 coalesce
# -------------------------
pop_raw = pop_raw.copy()
pop_raw.columns = pop_raw.columns.str.strip()
region_col = "행정구역"

# 행정구역명(코드) 분리
name_code = pop_raw[region_col].astype(str).str.extract(r"^\s*(.*?)\s*\((\d+)\)\s*$")
pop_raw["region_name"] = name_code[0].fillna(pop_raw[region_col].astype(str)).str.strip()
pop_raw["region_code"] = name_code[1].astype(str)

def trailing_zeros(s):
    s = str(s)
    return len(s) - len(s.rstrip("0"))

pop_raw["tz"] = pop_raw["region_code"].apply(trailing_zeros)

# 시도 레벨만(전국+시도)
pop_sido = pop_raw[pop_raw["tz"] >= 8].copy()
pop_sido["sido"] = pop_sido["region_name"]

def to_num(v):
    return pd.to_numeric(str(v).replace(",", ""), errors="coerce")

# 시도별 월별 tidy 생성(2023, 2024)
rows = []
for _, r in pop_sido.iterrows():
    sido = r["sido"]
    if sido == "전국":
        continue
    for y in [2023, 2024]:
        for m in range(1, 13):
            mm = f"{y}년{m:02d}월"
            endp = to_num(r.get(f"{mm}_당월인구수_계"))
            chg  = to_num(r.get(f"{mm}_인구증감_계"))
            rows.append({"sido": sido, "year": y, "month": mm, "pop_end": endp, "pop_change": chg})

pop_month = pd.DataFrame(rows)

# ✅ 강원/전북 coalesce: "값 있는 쪽 우선" (원본에 강원도 + 강원특별 둘 다 있을 수 있음)
def coalesce_two_sidos_monthly(df, sido_a, sido_b, keep):
    base = ["year", "month"]
    A = df[df["sido"] == sido_a][base + ["pop_end", "pop_change"]].copy()
    B = df[df["sido"] == sido_b][base + ["pop_end", "pop_change"]].copy()

    if len(A) == 0 and len(B) == 0:
        return df
    if len(A) == 0:
        out = df.copy()
        out.loc[out["sido"] == sido_b, "sido"] = keep
        return out
    if len(B) == 0:
        out = df.copy()
        out.loc[out["sido"] == sido_a, "sido"] = keep
        return out

    M = A.merge(B, on=base, how="outer", suffixes=("_a", "_b"))
    M["pop_change"] = M["pop_change_a"].where(M["pop_change_a"].notna(), M["pop_change_b"])
    M["pop_end"] = M["pop_end_a"].where(M["pop_end_a"].notna(), M["pop_end_b"])
    M["sido"] = keep
    M = M[["sido"] + base + ["pop_end", "pop_change"]]

    out = df[~df["sido"].isin([sido_a, sido_b])].copy()
    out = pd.concat([out, M], ignore_index=True)
    return out

# 강원: 강원도 + 강원특별자치도 → 강원특별자치도
pop_month = coalesce_two_sidos_monthly(pop_month, "강원특별자치도", "강원도", "강원특별자치도")
# 전북: 전북특별자치도 + 전라북도 → 전북특별자치도
pop_month = coalesce_two_sidos_monthly(pop_month, "전북특별자치도", "전라북도", "전북특별자치도")

# 혹시 남은 중복 정리(월별 합/평균)
pop_month = (
    pop_month.groupby(["sido", "year", "month"], as_index=False)
    .agg(pop_change=("pop_change", "sum"), pop_end=("pop_end", "mean"))
)

# -------------------------
# 의료(연도요약) + 인구(연도요약) 결합해서 wide 재생성
# -------------------------
long = long.copy()
long.columns = long.columns.str.strip()

# long에 필요한 컬럼 체크
need = ["sido", "year", "patients_year", "claims_year", "amount_year"]
for c in need:
    if c not in long.columns:
        st.error(f"compare_2023_2024_long.csv에 '{c}' 컬럼이 필요해요.")
        st.stop()

# 연도별 인구 요약
pop_year = (
    pop_month.groupby(["sido", "year"], as_index=False)
    .agg(pop_change_year=("pop_change", "sum"), pop_avg_year=("pop_end", "mean"))
)

# 연도별 의료 요약(혹시 long에 행이 많으면 합)
med_year = (
    long.groupby(["sido", "year"], as_index=False)
    .agg(
        patients_year=("patients_year", "sum"),
        claims_year=("claims_year", "sum"),
        amount_year=("amount_year", "sum"),
    )
)

merged_long = pop_year.merge(med_year, on=["sido", "year"], how="inner")
merged_long["patients_per_1k"] = merged_long["patients_year"] / merged_long["pop_avg_year"] * 1000
merged_long["amount_per_capita"] = merged_long["amount_year"] / merged_long["pop_avg_year"]

wide = merged_long.pivot(index="sido", columns="year", values=[
    "pop_change_year", "pop_avg_year",
    "patients_year", "claims_year", "amount_year",
    "patients_per_1k", "amount_per_capita"
]).reset_index()
wide.columns = ["sido"] + [f"{a}_{b}" for a, b in wide.columns[1:]]

wide["delta_pop_change"] = wide["pop_change_year_2024"] - wide["pop_change_year_2023"]
wide["delta_patients_per_1k"] = wide["patients_per_1k_2024"] - wide["patients_per_1k_2023"]
wide["delta_amount_per_capita"] = wide["amount_per_capita_2024"] - wide["amount_per_capita_2023"]
wide["delta_amount_total"] = wide["amount_year_2024"] - wide["amount_year_2023"]

# -------------------------
# UI 내용
# -------------------------
THEMES = {
    "delta_pop_change": ("인구증감 변화", "2024 - 2023 (명)"),
    "delta_patients_per_1k": ("인구 1천명당 환자수 변화", "2024 - 2023"),
    "delta_amount_per_capita": ("1인당 의료비 변화", "2024 - 2023 (원)"),
    "delta_amount_total": ("총 의료비 변화", "2024 - 2023 (원)"),
}

st.markdown("## 2023–2024 시도별 인구증감과 의료이용 비교")
st.markdown('<span class="pill">연도 비교</span><span class="pill">시도 단위</span><span class="pill">관계 분석</span>', unsafe_allow_html=True)
st.markdown('<div class="hint">강원/전북 명칭 변경 구간은 원본 인구 자료에서 자동 보정되어 반영됩니다.</div>', unsafe_allow_html=True)

# 즐겨찾기
if "favorites" not in st.session_state:
    st.session_state["favorites"] = []
all_sidos = sorted(wide["sido"].dropna().unique().tolist())
st.sidebar.markdown("### ⭐ 관심 지역")
st.session_state["favorites"] = st.sidebar.multiselect("관심 지역 선택", options=all_sidos, default=st.session_state["favorites"])

# KPI
c1, c2, c3, c4 = st.columns(4)
with c1: card("시도 수", f"{wide['sido'].nunique():,}", "분석 대상 지역 수")
with c2: card("인구증감 변화(평균)", f"{wide['delta_pop_change'].mean():,.0f}", "2024 - 2023 평균")
with c3: card("1인당 의료비 변화(중앙값)", f"{wide['delta_amount_per_capita'].median():,.0f}", "2024 - 2023 중앙값")
with c4: card("환자수/1천명 변화(중앙값)", f"{wide['delta_patients_per_1k'].median():,.1f}", "2024 - 2023 중앙값")

st.markdown("---")

tab_home, tab_rel, tab_quad, tab_map, tab_detail = st.tabs(["🏠 메인", "📈 관계", "🧭 4분면 분석", "🗺️ 지도", "📅 시도 상세"])

# 메인
with tab_home:
    st.markdown('<div class="section-title">테마별 상위 지역 (증가 / 감소)</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">기준: 변화량(2024 − 2023)</div>', unsafe_allow_html=True)

    for key, (tname, unit) in THEMES.items():
        st.markdown(f"**{tname}** <span class='small'>({unit})</span>", unsafe_allow_html=True)
        inc, dec = top_split(wide, key, n=5)
        a, b = st.columns(2)
        with a:
            fig_inc = px.bar(inc.sort_values(key, ascending=True), x=key, y="sido", orientation="h", title="증가 상위 5", template="plotly_white")
            fig_inc.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=10), xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_inc, use_container_width=True)
        with b:
            fig_dec = px.bar(dec.sort_values(key, ascending=True), x=key, y="sido", orientation="h", title="감소 상위 5", template="plotly_white")
            fig_dec.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=10), xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_dec, use_container_width=True)
        st.markdown("---")

    st.markdown('<div class="section-title">관심 지역(표)</div>', unsafe_allow_html=True)
    if len(st.session_state["favorites"]) == 0:
        st.info("왼쪽 사이드바에서 관심 지역을 선택하면 여기에서 요약을 볼 수 있어요.")
    else:
        fav = wide[wide["sido"].isin(st.session_state["favorites"])].copy()
        st.dataframe(fav[["sido","delta_pop_change","delta_patients_per_1k","delta_amount_per_capita","delta_amount_total"]], use_container_width=True)

    st.markdown('<div class="section-title">전체 데이터(표)</div>', unsafe_allow_html=True)
    st.dataframe(wide.sort_values("delta_amount_per_capita", ascending=False), use_container_width=True)

# 관계
with tab_rel:
    st.markdown('<div class="section-title">변화량 간 관계 (산점도 + 회귀선)</div>', unsafe_allow_html=True)
    x_key = st.selectbox("X 축", ["delta_pop_change", "delta_patients_per_1k"], index=0)
    y_key = st.selectbox("Y 축", ["delta_amount_per_capita", "delta_amount_total"], index=0)

    fig = px.scatter(wide, x=x_key, y=y_key, hover_name="sido", title=f"{THEMES[x_key][0]} ↔ {THEMES[y_key][0]}", template="plotly_white")
    reg = add_reg_line(wide, x_key, y_key)
    if reg is not None:
        a, b, r, line = reg
        fig.add_trace(line)
        st.caption(f"상관계수 r = {r:.2f} (단순선형 기준)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">내보내기</div>', unsafe_allow_html=True)
    rep_table = wide[["sido", x_key, y_key]].sort_values(y_key, ascending=False).head(10).copy()
    html = build_report_html(
        title="2023–2024 비교 리포트",
        subtitle=f"{THEMES[x_key][0]} ↔ {THEMES[y_key][0]} (Δ=2024−2023)",
        figs=[("관계(산점도)", fig)],
        tables=[("상위 10개 지역(표)", rep_table)],
    )
    st.download_button("리포트(HTML) 다운로드", data=html.encode("utf-8"), file_name="report_2023_2024.html", mime="text/html", use_container_width=True)

# 4분면
with tab_quad:
    st.markdown('<div class="section-title">4분면 분석으로 관심 지역 찾기</div>', unsafe_allow_html=True)
    x_key = st.selectbox("X(분할)", ["delta_pop_change", "delta_patients_per_1k"], index=0, key="qx")
    y_key = st.selectbox("Y(분할)", ["delta_amount_per_capita", "delta_amount_total"], index=0, key="qy")
    basis = st.radio("분할 기준", ["중앙값(median)", "평균(mean)"], index=0, horizontal=True)

    df = wide.copy()
    x_cut = df[x_key].median() if basis.startswith("중앙") else df[x_key].mean()
    y_cut = df[y_key].median() if basis.startswith("중앙") else df[y_key].mean()

    df["구역"] = np.select(
        [
            (df[x_key] >= x_cut) & (df[y_key] >= y_cut),
            (df[x_key] <  x_cut) & (df[y_key] >= y_cut),
            (df[x_key] <  x_cut) & (df[y_key] <  y_cut),
            (df[x_key] >= x_cut) & (df[y_key] <  y_cut),
        ],
        ["1사분면(X↑·Y↑)", "2사분면(X↓·Y↑)", "3사분면(X↓·Y↓)", "4사분면(X↑·Y↓)"],
        default="",
    )

    fig = px.scatter(df, x=x_key, y=y_key, color="구역", hover_name="sido", title="4분면 분류", template="plotly_white")
    fig.add_vline(x=float(x_cut))
    fig.add_hline(y=float(y_cut))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">2사분면(X↓·Y↑) 지역</div>', unsafe_allow_html=True)
    cols = uniq_cols(["sido", x_key, y_key, "delta_amount_per_capita", "delta_patients_per_1k", "delta_pop_change"])
    q2 = df[df["구역"].str.startswith("2사분면")][cols].copy()
    st.dataframe(q2.sort_values(y_key, ascending=False), use_container_width=True)

# 지도
with tab_map:
    st.markdown('<div class="section-title">지역 분포 보기</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">지도는 시도 대표 좌표(대략)에 표시됩니다.</div>', unsafe_allow_html=True)

    coords = {
        "서울특별시": (37.5665, 126.9780),
        "부산광역시": (35.1796, 129.0756),
        "대구광역시": (35.8714, 128.6014),
        "인천광역시": (37.4563, 126.7052),
        "광주광역시": (35.1595, 126.8526),
        "대전광역시": (36.3504, 127.3845),
        "울산광역시": (35.5384, 129.3114),
        "세종특별자치시": (36.4801, 127.2890),
        "경기도": (37.4138, 127.5183),
        "강원특별자치도": (37.8228, 128.1555),
        "충청북도": (36.6357, 127.4917),
        "충청남도": (36.5184, 126.8000),
        "전북특별자치도": (35.7175, 127.1530),
        "전라남도": (34.8161, 126.4629),
        "경상북도": (36.4919, 128.8889),
        "경상남도": (35.4606, 128.2132),
        "제주특별자치도": (33.4996, 126.5312),
    }

    df = wide.copy()
    df["lat"] = df["sido"].map(lambda x: coords.get(x, (np.nan, np.nan))[0])
    df["lon"] = df["sido"].map(lambda x: coords.get(x, (np.nan, np.nan))[1])
    df = df.dropna(subset=["lat", "lon"]).copy()

    color_metric = st.selectbox("색상 지표", list(THEMES.keys()), index=2)
    size_metric = st.selectbox("크기 지표", ["pop_avg_year_2024", "amount_year_2024", "patients_year_2024"], index=0)

    fig = px.scatter_mapbox(
        df, lat="lat", lon="lon", color=color_metric, size=size_metric,
        hover_name="sido",
        hover_data={
            "delta_pop_change": ":,.0f",
            "delta_patients_per_1k": ":,.1f",
            "delta_amount_per_capita": ":,.0f",
            "delta_amount_total": ":,.0f",
        },
        zoom=5, center={"lat": 36.3, "lon": 127.8}, height=640
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=40,b=0), title=f"지도: {THEMES[color_metric][0]}")
    st.plotly_chart(fig, use_container_width=True)

# 상세(월별 인구증감)
with tab_detail:
    st.markdown('<div class="section-title">시도별 월별 인구증감(2023/2024)</div>', unsafe_allow_html=True)
    sido_list = sorted(pop_month["sido"].unique().tolist())
    selected = st.selectbox("시도 선택", sido_list, index=0)

    d = pop_month[pop_month["sido"] == selected].copy()
    d["month_num"] = d["month"].str.extract(r"(\d{4})년(\d{2})월").apply(lambda x: int(x[0]) * 100 + int(x[1]), axis=1)
    d = d.sort_values("month_num")

    fig = px.line(d, x="month", y="pop_change", color="year", markers=True, title=f"{selected} 월별 인구증감", template="plotly_white")
    fig.update_layout(xaxis_title="", yaxis_title="인구증감(명)", height=420)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("※ 환자수/명세서건수는 시도 단위 합계라 ‘고유 인원’과 다를 수 있습니다. 비교·탐색 목적의 지표로 활용하세요.")
