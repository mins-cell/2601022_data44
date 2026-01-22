import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="2023–2024 인구·의료 비교", page_icon="📊", layout="wide")

# =========================
# UI 스타일(기존 느낌 유지, 문구는 일반화)
# =========================
st.markdown(
    """
    <style>
      :root {
        --bg: #fbfbff;
        --card: rgba(255,255,255,0.75);
        --stroke: rgba(49, 51, 63, 0.14);
      }
      .stApp { background: var(--bg); }
      .block-container { padding-top: 2rem; padding-bottom: 2rem; }
      .card {
        border: 1px solid var(--stroke);
        border-radius: 18px;
        padding: 14px 16px;
        background: var(--card);
        box-shadow: 0 8px 26px rgba(18, 18, 28, 0.06);
      }
      .card-title { font-size: 0.9rem; opacity: 0.78; margin-bottom: 6px; }
      .card-value { font-size: 1.55rem; font-weight: 750; line-height: 1.15; }
      .card-sub { font-size: 0.8rem; opacity: 0.7; margin-top: 6px; }
      .section-title { font-size: 1.05rem; font-weight: 750; margin: 0.2rem 0 0.6rem; }
      .hint { font-size: 0.92rem; opacity: 0.78; }
      .pill {
        display:inline-block; padding: 3px 10px; border-radius: 999px;
        border: 1px solid var(--stroke); font-size:.78rem; opacity:.85;
        background: rgba(255,255,255,0.6); margin-right: 6px;
      }
      .small { font-size: .85rem; opacity: .78; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# 파일명(레포에 그대로 올려두면 자동 인식)
# =========================
DEFAULT_FILES = {
    "인구(월간)": "202301_202512_주민등록인구기타현황(인구증감)_월간.csv",
    "의료(2023)": "건강보험심사평가원_시도별 의료행위 통계 2023.csv",
    "의료(2024)": "건강보험심사평가원_의료행위별 시도별 건강보험 진료 통계_20241231.csv",
}

MODE = st.sidebar.radio("데이터 불러오기", ["폴더에서 읽기(기본)", "파일 업로드"])

def read_csv_safely(file_or_path):
    # file_uploader 객체 또는 path 모두 처리
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(file_or_path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(file_or_path)

@st.cache_data
def load_local():
    pop = read_csv_safely(DEFAULT_FILES["인구(월간)"])
    h23 = read_csv_safely(DEFAULT_FILES["의료(2023)"])
    h24 = read_csv_safely(DEFAULT_FILES["의료(2024)"])
    return pop, h23, h24

def load_upload():
    f_pop = st.sidebar.file_uploader("업로드: 인구(월간) CSV", type=["csv"], key="pop")
    f_h23 = st.sidebar.file_uploader("업로드: 의료(2023) CSV", type=["csv"], key="h23")
    f_h24 = st.sidebar.file_uploader("업로드: 의료(2024) CSV", type=["csv"], key="h24")
    if (f_pop is None) or (f_h23 is None) or (f_h24 is None):
        st.sidebar.info("업로드 모드에서는 CSV 3개를 모두 올려야 해요.")
        return None
    return read_csv_safely(f_pop), read_csv_safely(f_h23), read_csv_safely(f_h24)

if MODE.startswith("폴더"):
    try:
        pop_raw, h23_raw, h24_raw = load_local()
    except Exception as e:
        st.error("폴더(레포)에서 파일을 못 찾았어요. 파일 업로드 모드로 바꾸거나 파일명을 확인해 주세요.")
        st.exception(e)
        st.stop()
else:
    loaded = load_upload()
    if loaded is None:
        st.stop()
    pop_raw, h23_raw, h24_raw = loaded

# =========================
# 공통 유틸
# =========================
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
    ys = a * xs + b
    line = go.Scatter(x=xs, y=ys, mode="lines", name=f"회귀선 (r={r:.2f})")
    return a, b, r, line

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

# =========================
# 1) 인구 데이터 전처리 + "강원/전북 coalesce" 적용
# =========================
pop = pop_raw.copy()
pop.columns = pop.columns.str.strip()
region_col = "행정구역"

# (행정구역명 + 코드) 추출
name_code = pop[region_col].astype(str).str.extract(r"^\s*(.*?)\s*\((\d+)\)\s*$")
pop["region_name"] = name_code[0].fillna(pop[region_col].astype(str)).str.strip()
pop["region_code"] = name_code[1].astype(str)

def trailing_zeros(s):
    s = str(s)
    return len(s) - len(s.rstrip("0"))

pop["tz"] = pop["region_code"].apply(trailing_zeros)

# 시도 레벨(전국+시도)만
pop_sido = pop[pop["tz"] >= 8].copy()
pop_sido["sido"] = pop_sido["region_name"]

# 숫자 변환(콤마 제거)
def to_num(v):
    return pd.to_numeric(str(v).replace(",", ""), errors="coerce")

# ✅ 강원/전북 coalesce: "값이 있는 쪽을 우선"으로 월별 컬럼을 합쳐서 1행으로 만들기
def coalesce_rows(df, left_name, right_name, keep_name, cols):
    a = df[df["sido"] == left_name]
    b = df[df["sido"] == right_name]
    if len(a) == 0 and len(b) == 0:
        return df
    if len(a) == 0:
        # right만 있으면 keep_name으로 이름만 바꿔 반환
        df2 = df.copy()
        df2.loc[df2["sido"] == right_name, "sido"] = keep_name
        return df2
    if len(b) == 0:
        df2 = df.copy()
        df2.loc[df2["sido"] == left_name, "sido"] = keep_name
        return df2

    a = a.iloc[0]
    b = b.iloc[0]

    merged = {c: None for c in df.columns}
    merged["sido"] = keep_name
    merged["region_name"] = keep_name
    merged["region_code"] = a.get("region_code", b.get("region_code", ""))
    merged["tz"] = a.get("tz", b.get("tz", 8))

    for c in cols:
        av = to_num(a.get(c))
        bv = to_num(b.get(c))
        merged[c] = av if pd.notna(av) else bv

    # cols 외 나머지는 a를 우선(없으면 b)
    for c in df.columns:
        if c in cols or c in ["sido", "region_name", "region_code", "tz"]:
            continue
        av = a.get(c)
        bv = b.get(c)
        merged[c] = av if pd.notna(av) else bv

    df2 = df[~df["sido"].isin([left_name, right_name])].copy()
    return pd.concat([df2, pd.DataFrame([merged])], ignore_index=True)

# coalesce 대상 월별 컬럼(2023, 2024의 당월인구수_계 / 인구증감_계)
month_cols = []
for y in [2023, 2024]:
    for m in range(1, 13):
        mm = f"{y}년{m:02d}월"
        month_cols += [f"{mm}_당월인구수_계", f"{mm}_인구증감_계"]

# 적용: 강원/전북
pop_sido = coalesce_rows(pop_sido, "강원도", "강원특별자치도", "강원특별자치도", month_cols)
pop_sido = coalesce_rows(pop_sido, "전라북도", "전북특별자치도", "전북특별자치도", month_cols)

# (연간 요약) 2023/2024
def year_pop_summary(row, year):
    chg_sum = 0.0
    end_vals = []
    for m in range(1, 13):
        mm = f"{year}년{m:02d}월"
        chg = to_num(row.get(f"{mm}_인구증감_계"))
        endp = to_num(row.get(f"{mm}_당월인구수_계"))
        if pd.notna(chg):
            chg_sum += float(chg)
        if pd.notna(endp):
            end_vals.append(float(endp))
    return chg_sum, (float(np.mean(end_vals)) if end_vals else np.nan)

pop_year_rows = []
for _, r in pop_sido.iterrows():
    if r["sido"] == "전국":
        continue
    for year in [2023, 2024]:
        chg, avgp = year_pop_summary(r, year)
        pop_year_rows.append({
            "sido": r["sido"],
            "year": year,
            "pop_change_year": chg,
            "pop_avg_year": avgp
        })
pop_year = pd.DataFrame(pop_year_rows)

# (월별 tidy) 2023/2024
pop_month_rows = []
for _, r in pop_sido.iterrows():
    if r["sido"] == "전국":
        continue
    for y in [2023, 2024]:
        for m in range(1, 13):
            mm = f"{y}년{m:02d}월"
            endp = to_num(r.get(f"{mm}_당월인구수_계"))
            chg = to_num(r.get(f"{mm}_인구증감_계"))
            pop_month_rows.append({
                "sido": r["sido"],
                "year": y,
                "month": mm,
                "pop_end": endp,
                "pop_change": chg
            })
pop_month = pd.DataFrame(pop_month_rows)

# =========================
# 2) 의료 데이터 전처리(2023, 2024)
# =========================
h23 = h23_raw.copy()
h24 = h24_raw.copy()
for df in (h23, h24):
    df.columns = df.columns.str.strip()
    for c in ["환자수", "명세서건수", "의료행위총사용량", "의료행위청구금액"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

# 시도 약칭 → 정식명 매핑(의료 파일)
map_sido = {
    "서울": "서울특별시",
    "부산": "부산광역시",
    "대구": "대구광역시",
    "인천": "인천광역시",
    "광주": "광주광역시",
    "대전": "대전광역시",
    "울산": "울산광역시",
    "세종": "세종특별자치시",
    "경기": "경기도",
    "강원": "강원특별자치도",
    "충북": "충청북도",
    "충남": "충청남도",
    "전북": "전북특별자치도",
    "전남": "전라남도",
    "경북": "경상북도",
    "경남": "경상남도",
    "제주": "제주특별자치도",
}

def hira_year_summary(df, year):
    g = (
        df.groupby("시도", as_index=False)
          .agg(
              patients_year=("환자수", "sum"),
              claims_year=("명세서건수", "sum"),
              amount_year=("의료행위청구금액", "sum"),
          )
    )
    g["sido"] = g["시도"].map(map_sido)
    g["year"] = year
    return g.drop(columns=["시도"]).dropna(subset=["sido"])

hira_2023 = hira_year_summary(h23, 2023)
hira_2024 = hira_year_summary(h24, 2024)
hira_year = pd.concat([hira_2023, hira_2024], ignore_index=True)

# =========================
# 3) 결합 + 비교(Δ=2024−2023)
# =========================
merged_long = pop_year.merge(hira_year, on=["sido", "year"], how="inner")

merged_long["patients_per_1k"] = merged_long["patients_year"] / merged_long["pop_avg_year"] * 1000
merged_long["amount_per_capita"] = merged_long["amount_year"] / merged_long["pop_avg_year"]

wide = merged_long.pivot(index="sido", columns="year", values=[
    "pop_change_year", "pop_avg_year",
    "patients_year", "claims_year", "amount_year",
    "patients_per_1k", "amount_per_capita"
]).reset_index()

wide.columns = ["sido"] + [f"{a}_{b}" for a, b in wide.columns[1:]]
wide = wide.loc[:, ~wide.columns.duplicated()].copy()

# 변화량(2024-2023)
wide["delta_pop_change"] = wide["pop_change_year_2024"] - wide["pop_change_year_2023"]
wide["delta_patients_per_1k"] = wide["patients_per_1k_2024"] - wide["patients_per_1k_2023"]
wide["delta_amount_per_capita"] = wide["amount_per_capita_2024"] - wide["amount_per_capita_2023"]
wide["delta_amount_total"] = wide["amount_year_2024"] - wide["amount_year_2023"]

# =========================
# 앱 UI: 한글 용어/표/4분면
# =========================
THEMES = {
    "delta_pop_change": ("인구증감 변화", "2024 - 2023 (명)"),
    "delta_patients_per_1k": ("인구 1천명당 환자수 변화", "2024 - 2023"),
    "delta_amount_per_capita": ("1인당 의료비 변화", "2024 - 2023 (원)"),
    "delta_amount_total": ("총 의료비 변화", "2024 - 2023 (원)"),
}

# 헤더
st.markdown("## 2023–2024 시도별 인구증감과 의료이용 비교")
st.markdown(
    '<span class="pill">연도 비교</span><span class="pill">시도 단위</span><span class="pill">관계 분석</span>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="hint">메인 화면에서 테마별 상위 지역을 증가/감소로 나눠 확인하고, '
    '탭에서 관계, 4분면 분석, 지도, 시도별 월별 추이를 자세히 볼 수 있어요.</div>',
    unsafe_allow_html=True
)

# 즐겨찾기(관심지역)
if "favorites" not in st.session_state:
    st.session_state["favorites"] = []

all_sidos = sorted(wide["sido"].dropna().unique().tolist())
st.sidebar.markdown("### ⭐ 관심 지역")
fav_pick = st.sidebar.multiselect("관심 지역 선택", options=all_sidos, default=st.session_state["favorites"])
st.session_state["favorites"] = fav_pick

with st.sidebar.expander("빠른 추가/삭제", expanded=False):
    quick = st.selectbox("시도 선택", all_sidos, index=0, key="quick_sido")
    c_add, c_rm = st.columns(2)
    with c_add:
        if st.button("추가", use_container_width=True):
            if quick not in st.session_state["favorites"]:
                st.session_state["favorites"] = st.session_state["favorites"] + [quick]
                st.rerun()
    with c_rm:
        if st.button("삭제", use_container_width=True):
            st.session_state["favorites"] = [x for x in st.session_state["favorites"] if x != quick]
            st.rerun()

# KPI 카드
c1, c2, c3, c4 = st.columns(4)
with c1: card("시도 수", f"{wide['sido'].nunique():,}", "분석 대상 지역 수")
with c2: card("인구증감 변화(평균)", f"{wide['delta_pop_change'].mean():,.0f}", "2024 - 2023 평균")
with c3: card("1인당 의료비 변화(중앙값)", f"{wide['delta_amount_per_capita'].median():,.0f}", "2024 - 2023 중앙값")
with c4: card("환자수/1천명 변화(중앙값)", f"{wide['delta_patients_per_1k'].median():,.1f}", "2024 - 2023 중앙값")

st.markdown("---")

tab_home, tab_rel, tab_quad, tab_map, tab_detail = st.tabs(
    ["🏠 메인", "📈 관계", "🧭 4분면 분석", "🗺️ 지도", "📅 시도 상세"]
)

def top_split(df, metric, n=5):
    d = df[["sido", metric]].dropna().copy()
    inc = d.sort_values(metric, ascending=False).head(n)
    dec = d.sort_values(metric, ascending=True).head(n)
    return inc, dec

# ---------------- 메인: 테마별 증가/감소 상위 5 ----------------
with tab_home:
    st.markdown('<div class="section-title">테마별 상위 지역 (증가 / 감소)</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">기준: 변화량(2024 − 2023). 증가=큰 값, 감소=작은 값</div>', unsafe_allow_html=True)

    for key, (tname, unit) in THEMES.items():
        st.markdown(f"**{tname}** <span class='small'>({unit})</span>", unsafe_allow_html=True)
        inc, dec = top_split(wide, key, n=5)

        colA, colB = st.columns(2)
        with colA:
            fig_inc = px.bar(
                inc.sort_values(key, ascending=True),
                x=key, y="sido", orientation="h",
                title="증가 상위 5",
                template="plotly_white",
            )
            fig_inc.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=10), xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_inc, use_container_width=True)

        with colB:
            fig_dec = px.bar(
                dec.sort_values(key, ascending=True),
                x=key, y="sido", orientation="h",
                title="감소 상위 5",
                template="plotly_white",
            )
            fig_dec.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=10), xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_dec, use_container_width=True)

        st.markdown("---")

    st.markdown('<div class="section-title">관심 지역 요약(표)</div>', unsafe_allow_html=True)
    if len(st.session_state["favorites"]) == 0:
        st.info("왼쪽 사이드바에서 관심 지역을 선택하면 여기에서 요약을 볼 수 있어요.")
    else:
        fav = wide[wide["sido"].isin(st.session_state["favorites"])].copy()
        show_cols = ["sido", "delta_pop_change", "delta_patients_per_1k", "delta_amount_per_capita", "delta_amount_total"]
        st.dataframe(fav[show_cols].sort_values("delta_amount_per_capita", ascending=False), use_container_width=True)

    st.markdown('<div class="section-title">전체 데이터(표)</div>', unsafe_allow_html=True)
    st.dataframe(wide.sort_values("delta_amount_per_capita", ascending=False), use_container_width=True)

# ---------------- 관계: 산점도 + 회귀선 + 내보내기 ----------------
with tab_rel:
    st.markdown('<div class="section-title">변화량 간 관계(산점도 + 회귀선)</div>', unsafe_allow_html=True)

    x_key = st.selectbox("X 축", ["delta_pop_change", "delta_patients_per_1k"], index=0)
    y_key = st.selectbox("Y 축", ["delta_amount_per_capita", "delta_amount_total"], index=0)

    x_label = THEMES.get(x_key, (x_key, ""))[0]
    y_label = THEMES.get(y_key, (y_key, ""))[0]

    df = wide.copy()
    fig = px.scatter(df, x=x_key, y=y_key, hover_name="sido",
                     title=f"{x_label} ↔ {y_label}", template="plotly_white")
    reg = add_reg_line(df, x_key, y_key)
    if reg is not None:
        a, b, r, line = reg
        fig.add_trace(line)
        st.caption(f"상관계수 r = {r:.2f} (단순선형 기준)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">내보내기</div>', unsafe_allow_html=True)
    rep_table = wide[["sido", x_key, y_key]].sort_values(y_key, ascending=False).head(10).copy()
    html = build_report_html(
        title="2023–2024 비교 리포트",
        subtitle=f"선택한 관계: {x_label} ↔ {y_label} (변화량=2024−2023)",
        figs=[("관계(산점도)", fig)],
        tables=[("상위 10개 지역(표)", rep_table)],
    )
    st.download_button(
        "리포트(HTML) 다운로드",
        data=html.encode("utf-8"),
        file_name="report_2023_2024.html",
        mime="text/html",
        use_container_width=True,
    )

# ---------------- 4분면 분석: 한글 라벨 + 내보내기 ----------------
with tab_quad:
    st.markdown('<div class="section-title">4분면 분석으로 관심 지역 찾기</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">기준선: 선택한 분할 기준(중앙값/평균)</div>', unsafe_allow_html=True)

    x_key = st.selectbox("X(분할)", ["delta_pop_change", "delta_patients_per_1k"], index=0, key="q_x")
    y_key = st.selectbox("Y(분할)", ["delta_amount_per_capita", "delta_amount_total"], index=0, key="q_y")
    basis = st.radio("분할 기준", ["중앙값(median)", "평균(mean)"], index=0, horizontal=True)

    df = wide.copy()
    x_cut = df[x_key].median() if basis.startswith("중앙") else df[x_key].mean()
    y_cut = df[y_key].median() if basis.startswith("중앙") else df[y_key].mean()

    # ✅ 한글 4분면 이름(해석용)
    # 1사분면: X↑ Y↑ / 2사분면: X↓ Y↑ / 3사분면: X↓ Y↓ / 4사분면: X↑ Y↓
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

    fig = px.scatter(df, x=x_key, y=y_key, color="구역", hover_name="sido",
                     title="4분면 분류", template="plotly_white")
    fig.add_vline(x=float(x_cut))
    fig.add_hline(y=float(y_cut))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">2사분면(X↓·Y↑) 지역(표)</div>', unsafe_allow_html=True)
    cols = uniq_cols(["sido", x_key, y_key, "delta_amount_per_capita", "delta_patients_per_1k", "delta_pop_change"])
    q2 = df[df["구역"].str.startswith("2사분면")][cols].copy()
    st.dataframe(q2.sort_values(y_key, ascending=False), use_container_width=True)

    st.markdown('<div class="section-title">내보내기</div>', unsafe_allow_html=True)
    q2_top = q2.sort_values(y_key, ascending=False).head(10).copy()
    html = build_report_html(
        title="2023–2024 4분면 요약",
        subtitle=f"분할: {THEMES.get(x_key,(x_key,''))[0]} × {THEMES.get(y_key,(y_key,''))[0]} (변화량=2024−2023)",
        figs=[("4분면 분류(산점도)", fig)],
        tables=[("2사분면 상위 10개 지역(표)", q2_top)],
    )
    st.download_button(
        "4분면 요약(HTML) 다운로드",
        data=html.encode("utf-8"),
        file_name="four_quadrant_summary_2023_2024.html",
        mime="text/html",
        use_container_width=True,
    )

# ---------------- 지도: 내보내기 ----------------
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
        df,
        lat="lat",
        lon="lon",
        color=color_metric,
        size=size_metric,
        hover_name="sido",
        hover_data={
            "delta_pop_change": ":,.0f",
            "delta_patients_per_1k": ":,.1f",
            "delta_amount_per_capita": ":,.0f",
            "delta_amount_total": ":,.0f",
        },
        zoom=5,
        center={"lat": 36.3, "lon": 127.8},
        height=640,
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0),
                      title=f"지도: {THEMES[color_metric][0]}")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">내보내기</div>', unsafe_allow_html=True)
    tbl = df[["sido", color_metric, size_metric]].sort_values(color_metric, ascending=False).head(10).copy()
    html = build_report_html(
        title="2023–2024 지도 요약",
        subtitle=f"색상: {THEMES[color_metric][0]} / 크기: {size_metric}",
        figs=[("지도", fig)],
        tables=[("상위 10개 지역(표)", tbl)],
    )
    st.download_button(
        "지도 요약(HTML) 다운로드",
        data=html.encode("utf-8"),
        file_name="map_summary_2023_2024.html",
        mime="text/html",
        use_container_width=True,
    )

# ---------------- 시도 상세: 월별 추이(2023/2024) + 내보내기 ----------------
with tab_detail:
    st.markdown('<div class="section-title">시도별 월별 인구증감(2023/2024)</div>', unsafe_allow_html=True)
    sido_list = sorted(pop_month["sido"].unique().tolist())
    selected = st.selectbox("시도 선택", sido_list, index=0)

    d = pop_month[pop_month["sido"] == selected].copy()
    d["month_num"] = d["month"].str.extract(r"(\d{4})년(\d{2})월").apply(lambda x: int(x[0]) * 100 + int(x[1]), axis=1)
    d = d.sort_values("month_num")

    fig = px.line(d, x="month", y="pop_change", color="year", markers=True,
                  title=f"{selected} 월별 인구증감", template="plotly_white")
    fig.update_layout(xaxis_title="", yaxis_title="인구증감(명)", height=420)
    st.plotly_chart(fig, use_container_width=True)

    row = wide[wide["sido"] == selected].iloc[0]
    c1, c2, c3 = st.columns(3)
    with c1: card("인구증감 변화", f"{row['delta_pop_change']:,.0f}", "2024 - 2023")
    with c2: card("환자수/1천명 변화", f"{row['delta_patients_per_1k']:,.1f}", "2024 - 2023")
    with c3: card("1인당 의료비 변화", f"{row['delta_amount_per_capita']:,.0f}", "2024 - 2023")

    st.markdown('<div class="section-title">내보내기</div>', unsafe_allow_html=True)
    detail_tbl = wide[wide["sido"] == selected].copy()
    html = build_report_html(
        title="시도 상세 요약",
        subtitle=f"{selected} (2023–2024 변화량)",
        figs=[("월별 인구증감(2023/2024)", fig)],
        tables=[("요약(표)", detail_tbl)],
    )
    st.download_button(
        "시도 상세(HTML) 다운로드",
        data=html.encode("utf-8"),
        file_name=f"detail_{selected}_2023_2024.html",
        mime="text/html",
        use_container_width=True,
    )

st.markdown("---")
st.caption(
    "※ 환자수/명세서건수는 의료행위별 통계를 시도 단위로 합산한 값이라 '고유 인원'과 다를 수 있습니다. "
    "비교·탐색 목적의 지표로 활용하세요."
)

