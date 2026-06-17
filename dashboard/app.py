import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="핀테크 퍼포먼스 마케팅 대시보드", layout="wide", page_icon="📊")

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "핀테크_테이터분석.parquet")
VOL = ["광고노출","광고클릭","광고비","앱설치","앱실행","회원가입","계좌개설","첫거래","반복사용","자동이체설정","추천완료"]

st.markdown("""
<style>
.kpi-card{background:#fff;border:1px solid #e9ecef;border-radius:10px;padding:1rem 1.2rem;text-align:center}
.kpi-card-primary{background:#fff;border:2px solid #3498db;border-radius:10px;padding:1rem 1.2rem;text-align:center}
.kpi-label{font-size:.78rem;color:#6c757d;font-weight:600;letter-spacing:.03em}
.kpi-value{font-size:1.45rem;font-weight:700;font-family:monospace;color:#1a1a2e}
.kpi-value-primary{font-size:1.6rem;font-weight:800;font-family:monospace;color:#1a1a2e}
.kpi-delta-pos{font-size:.8rem;color:#2ecc71;font-weight:600}
.kpi-delta-neg{font-size:.8rem;color:#e74c3c;font-weight:600}
.kpi-delta-neu{font-size:.8rem;color:#95a5a6}
.kpi-highlight{border:2px solid #f39c12;background:#fffdf0}
.insight-card{border-left:4px solid #dee2e6;padding:.8rem 1rem;margin-bottom:.7rem;background:#fff;border-radius:0 8px 8px 0}
.badge-긴급{background:#e74c3c;color:#fff;font-size:.72rem;padding:2px 7px;border-radius:10px;font-weight:700}
.badge-중요{background:#f39c12;color:#fff;font-size:.72rem;padding:2px 7px;border-radius:10px;font-weight:700}
.badge-참고{background:#6c757d;color:#fff;font-size:.72rem;padding:2px 7px;border-radius:10px;font-weight:700}
.info-box{background:#e8f4fd;border-left:4px solid #3498db;padding:.8rem 1rem;border-radius:0 8px 8px 0;font-size:.9rem;margin:.5rem 0}
.warn-box{background:#fff3cd;border-left:4px solid #f39c12;padding:.8rem 1rem;border-radius:0 8px 8px 0;font-size:.9rem;margin:.5rem 0}
.danger-box{background:#fdf0f0;border-left:4px solid #e74c3c;padding:.8rem 1rem;border-radius:0 8px 8px 0;font-size:.9rem;margin:.5rem 0}
.success-box{background:#f0fff4;border-left:4px solid #2ecc71;padding:.8rem 1rem;border-radius:0 8px 8px 0;font-size:.9rem;margin:.5rem 0}
.action-box{background:#f0f7ff;border-left:4px solid #3498db;padding:.7rem .9rem;border-radius:0 6px 6px 0;font-size:.85rem;margin:.3rem 0}
.hypothesis-box{background:#fffbf0;border-left:4px solid #f39c12;padding:.7rem .9rem;border-radius:0 6px 6px 0;font-size:.85rem;margin:.3rem 0}
.season-box{background:#f8f9fa;border-radius:10px;padding:1rem 1.2rem;margin:.5rem 0;border:1px solid #dee2e6}
.dropoff-card{border-radius:10px;padding:1rem;margin:.4rem 0;border:1px solid #dee2e6}
.filter-banner{background:#f0f4ff;border:1px solid #c5cff7;border-radius:8px;padding:.6rem 1rem;font-size:.85rem;color:#2c3e7a;margin-bottom:.8rem}
.team-tag-ad{background:#dbeafe;color:#1e40af;font-size:.7rem;padding:1px 7px;border-radius:10px;font-weight:600}
.team-tag-product{background:#dcfce7;color:#166534;font-size:.7rem;padding:1px 7px;border-radius:10px;font-weight:600}
.team-tag-crm{background:#f3e8ff;color:#6b21a8;font-size:.7rem;padding:1px 7px;border-radius:10px;font-weight:600}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner="데이터 로딩 중...")
def load_data():
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

df_raw = load_data()

# ── 사이드바 필터 (기본값: 전체 기간) ───────────────────────────────────
with st.sidebar:
    st.markdown("### 필터")
    period = st.selectbox("분석 기간", ["전체 기간","오늘","최근 7일","최근 30일","직접 입력"], index=0)
    min_date = df_raw["date"].min()
    max_date = df_raw["date"].max()
    if period == "오늘":
        date_from, date_to = max_date, max_date
    elif period == "최근 7일":
        date_from, date_to = max_date - pd.Timedelta(days=6), max_date
    elif period == "최근 30일":
        date_from, date_to = max_date - pd.Timedelta(days=29), max_date
    elif period == "직접 입력":
        date_from = pd.Timestamp(st.date_input("시작일", value=min_date))
        date_to   = pd.Timestamp(st.date_input("종료일", value=max_date))
    else:
        date_from, date_to = min_date, max_date

    sel_channels = st.multiselect("채널", sorted(df_raw["channel"].unique()), default=sorted(df_raw["channel"].unique()))
    sel_objs     = st.multiselect("캠페인 목적", sorted(df_raw["campaign_objective"].unique()), default=sorted(df_raw["campaign_objective"].unique()))
    sel_groups   = st.multiselect("광고그룹", sorted(df_raw["ad_group"].unique()), default=sorted(df_raw["ad_group"].unique()))
    sel_formats  = st.multiselect("소재 형식", sorted(df_raw["creative_format"].unique()), default=sorted(df_raw["creative_format"].unique()))
    st.divider()
    st.caption(f"데이터 범위: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")

df = df_raw[
    (df_raw["date"] >= date_from) & (df_raw["date"] <= date_to) &
    df_raw["channel"].isin(sel_channels) & df_raw["campaign_objective"].isin(sel_objs) &
    df_raw["ad_group"].isin(sel_groups) & df_raw["creative_format"].isin(sel_formats)
]
prev_from = date_from - (date_to - date_from + pd.Timedelta(days=1))
prev_to   = date_from - pd.Timedelta(days=1)
df_prev = df_raw[
    (df_raw["date"] >= prev_from) & (df_raw["date"] <= prev_to) &
    df_raw["channel"].isin(sel_channels) & df_raw["campaign_objective"].isin(sel_objs) &
    df_raw["ad_group"].isin(sel_groups) & df_raw["creative_format"].isin(sel_formats)
]

TOTAL_COST_ALL = df_raw["광고비"].sum()
TOTAL_FT_ALL   = df_raw["첫거래"].sum()

def safe_div(a, b, default=0):
    return (a / b) if (b and b != 0) else default

def calc_metrics(sub, total_cost=None, total_ft=None):
    if total_cost is None: total_cost = sub["광고비"].sum()
    if total_ft   is None: total_ft   = sub["첫거래"].sum()
    s = sub[VOL].sum()
    노출=s["광고노출"]; 클릭=s["광고클릭"]; 비용=s["광고비"]
    설치=s["앱설치"]; 가입=s["회원가입"]; 계좌=s["계좌개설"]
    거래=s["첫거래"]; 반복=s["반복사용"]; 자동=s["자동이체설정"]; 추천=s["추천완료"]
    return {
        "광고비":비용,"노출":노출,"클릭":클릭,"앱설치":설치,"회원가입":가입,
        "계좌개설":계좌,"첫거래":거래,"반복사용":반복,"자동이체설정":자동,"추천완료":추천,
        "효율비":   round(safe_div(거래/total_ft, 비용/total_cost) if total_ft and total_cost else 0, 2),
        "CTR":      round(safe_div(클릭,노출)*100,2),
        "CPC":      round(safe_div(비용,클릭),0),
        "CPI":      round(safe_div(비용,설치),0),
        "클릭→설치율":  round(safe_div(설치,클릭)*100,2),
        "설치→가입율":  round(safe_div(가입,설치)*100,2),
        "가입→계좌율":  round(safe_div(계좌,가입)*100,2),
        "계좌→거래율":  round(safe_div(거래,계좌)*100,2),
        "거래→반복율":  round(safe_div(반복,거래)*100,2),
        "반복→자동율":  round(safe_div(자동,반복)*100,2),
        "자동→추천율":  round(safe_div(추천,자동)*100,2),
        "가입CPA":    round(safe_div(비용,가입),0),
        "계좌CPA":    round(safe_div(비용,계좌),0),
        "첫거래CPA":  round(safe_div(비용,거래),0),
        "반복CPA":    round(safe_div(비용,반복),0),
        "자동이체CPA": round(safe_div(비용,자동),0),
        "추천CPA":    round(safe_div(비용,추천),0),
    }

def verdict(eff, auto_cpa):
    if eff >= 1.5 and auto_cpa <= 20000: return "확대"
    elif eff >= 1.0: return "유지"
    elif eff >= 0.5: return "개선테스트"
    else: return "즉시중단"

def fmt_num(v):
    if v >= 1e8: return f"{v/1e8:.1f}억"
    if v >= 1e4: return f"{v/1e4:.1f}만"
    return f"{v:,.0f}"

def fmt_won(v): return f"₩{v:,.0f}"

def delta_str(cur, prev, reverse=False):
    if prev == 0: return ""
    d = (cur - prev) / prev * 100
    good = (d > 0) if not reverse else (d < 0)
    sign = "▲" if d > 0 else "▼"
    cls = "kpi-delta-pos" if good else "kpi-delta-neg"
    return f'<span class="{cls}">{sign}{abs(d):.1f}%</span>'

# ── 알림 배너 ─────────────────────────────────────────────────────────
alerts = []
for ch in df["channel"].unique():
    sub = df[df["channel"]==ch]
    em = calc_metrics(sub, TOTAL_COST_ALL, TOTAL_FT_ALL)
    if em["효율비"] < 0.9:
        alerts.append(f"{ch} 효율비 {em['효율비']:.2f}")
if alerts:
    st.warning("⚠ 알림: " + " | ".join(alerts) + " (기준 0.9 미만)")

# ── 필터 상태 배너 ────────────────────────────────────────────────────
period_label = period if period != "직접 입력" else f"{date_from.strftime('%Y-%m-%d')} ~ {date_to.strftime('%Y-%m-%d')}"
ch_label  = "채널 전체" if len(sel_channels)==len(df_raw["channel"].unique()) else " / ".join(sel_channels)
obj_label = "목적 전체" if len(sel_objs)==len(df_raw["campaign_objective"].unique()) else " / ".join(sel_objs)
grp_label = "그룹 전체" if len(sel_groups)==len(df_raw["ad_group"].unique()) else " / ".join(sel_groups)
fmt_label = "소재 전체" if len(sel_formats)==len(df_raw["creative_format"].unique()) else " / ".join(sel_formats)
st.markdown(
    f'<div class="filter-banner">📊 <b>현재 분석 기준:</b> {period_label} &nbsp;|&nbsp; {ch_label} &nbsp;|&nbsp; {obj_label} &nbsp;|&nbsp; {grp_label} &nbsp;|&nbsp; {fmt_label} &nbsp;&nbsp; '
    f'<span style="color:#6c757d">({len(df):,}행 / 전체 {len(df_raw):,}행)</span></div>',
    unsafe_allow_html=True)

# ── 탭 ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", "Channel", "Funnel", "Campaign & Creative",
    "Retention & Referral", "Insights & Action", "마케팅 제언"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("전체 현황 & 월별·시즌별 분석")

    m     = calc_metrics(df, TOTAL_COST_ALL, TOTAL_FT_ALL)
    m_prv = calc_metrics(df_prev, TOTAL_COST_ALL, TOTAL_FT_ALL)

    # ── 핵심 KPI ─────────────────────────────────────────────────
    st.markdown("**핵심 KPI**")
    primary_kpis = [
        ("총 광고비",       fmt_won(m["광고비"]),                    delta_str(m["광고비"],m_prv["광고비"],reverse=True)),
        ("첫거래 CPA",      fmt_won(m["첫거래CPA"]),                 delta_str(m["첫거래CPA"],m_prv["첫거래CPA"],reverse=True)),
        ("자동이체 CPA",    fmt_won(m["자동이체CPA"]),               delta_str(m["자동이체CPA"],m_prv["자동이체CPA"],reverse=True)),
        ("효율비",          f"{m['효율비']:.2f}",                    delta_str(m["효율비"],m_prv["효율비"])),
        ("반복→자동이체율", f"{m['반복→자동율']:.2f}%",             delta_str(m["반복→자동율"],m_prv["반복→자동율"])),
    ]
    pcols = st.columns(5)
    for i,(label,val,dlt) in enumerate(primary_kpis):
        pcols[i].markdown(
            f'<div class="kpi-card-primary"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value-primary">{val}</div>{dlt}</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── 보조 KPI ─────────────────────────────────────────────────
    st.markdown('<div style="font-size:.8rem;color:#6c757d;font-weight:600;margin-bottom:.4rem">보조 KPI (볼륨 지표)</div>', unsafe_allow_html=True)
    secondary_kpis = [
        ("노출",      fmt_num(m["노출"]),         delta_str(m["노출"],m_prv["노출"])),
        ("클릭",      fmt_num(m["클릭"]),         delta_str(m["클릭"],m_prv["클릭"])),
        ("앱설치",    fmt_num(m["앱설치"]),       delta_str(m["앱설치"],m_prv["앱설치"])),
        ("회원가입",  fmt_num(m["회원가입"]),     delta_str(m["회원가입"],m_prv["회원가입"])),
        ("계좌개설",  fmt_num(m["계좌개설"]),     delta_str(m["계좌개설"],m_prv["계좌개설"])),
        ("첫거래",    fmt_num(m["첫거래"]),       delta_str(m["첫거래"],m_prv["첫거래"])),
        ("반복사용",  fmt_num(m["반복사용"]),     delta_str(m["반복사용"],m_prv["반복사용"])),
        ("추천완료",  fmt_num(m["추천완료"]),     delta_str(m["추천완료"],m_prv["추천완료"])),
    ]
    scols = st.columns(8)
    for i,(label,val,dlt) in enumerate(secondary_kpis):
        scols[i].markdown(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value" style="font-size:1.1rem">{val}</div>{dlt}</div>', unsafe_allow_html=True)

    st.divider()

    # ── 월별 데이터 계산 ────────────────────────────────────────
    df_m = df_raw.copy()
    df_m["month"] = df_m["date"].dt.to_period("M").astype(str)
    monthly = df_m.groupby("month")[VOL].sum().reset_index()
    monthly["첫거래CPA"] = monthly["광고비"] / monthly["첫거래"].replace(0, np.nan)
    monthly["CPC"]      = monthly["광고비"] / monthly["광고클릭"].replace(0, np.nan)
    monthly["CPI"]      = monthly["광고비"] / monthly["앱설치"].replace(0, np.nan)
    monthly["CTR"]      = monthly["광고클릭"] / monthly["광고노출"].replace(0, np.nan) * 100
    monthly["효율비"]    = (monthly["첫거래"] / TOTAL_FT_ALL) / (monthly["광고비"] / TOTAL_COST_ALL)
    monthly["광고비MoM"] = monthly["광고비"].pct_change() * 100
    monthly["거래MoM"]   = monthly["첫거래"].pct_change() * 100
    monthly["반복사용률"] = monthly["반복사용"] / monthly["첫거래"].replace(0, np.nan) * 100

    HIGH_SEASON = ["2025-03","2025-09","2025-12"]
    LOW_SEASON  = ["2025-01","2025-07"]
    SEASON_LABEL = {
        "2025-01":"연초","2025-02":"설날","2025-03":"신학기",
        "2025-04":"봄","2025-05":"어린이날","2025-06":"비수기",
        "2025-07":"휴가철","2025-08":"여름","2025-09":"추석",
        "2025-10":"하반기","2025-11":"블랙프라이데이","2025-12":"연말"
    }
    monthly["시즌"] = monthly["month"].map(SEASON_LABEL)

    # 월별 요약 KPI
    eff_max_row = monthly.loc[monthly["효율비"].idxmax()]
    eff_min_row = monthly.loc[monthly["효율비"].idxmin()]
    cpa_max_row = monthly.loc[monthly["첫거래CPA"].idxmax()]
    bud_max_row = monthly.loc[monthly["광고비"].idxmax()]
    sc1,sc2,sc3,sc4 = st.columns(4)
    sc1.markdown(f'<div class="kpi-card"><div class="kpi-label">최고 효율 월</div>'
                 f'<div class="kpi-value" style="color:#2ecc71">{eff_max_row["month"].replace("2025-","").lstrip("0")}월</div>'
                 f'<div style="font-size:.82rem;color:#6c757d">효율비 {eff_max_row["효율비"]:.2f}</div></div>', unsafe_allow_html=True)
    sc2.markdown(f'<div class="kpi-card"><div class="kpi-label">최저 효율 월</div>'
                 f'<div class="kpi-value" style="color:#e74c3c">{eff_min_row["month"].replace("2025-","").lstrip("0")}월</div>'
                 f'<div style="font-size:.82rem;color:#6c757d">효율비 {eff_min_row["효율비"]:.2f}</div></div>', unsafe_allow_html=True)
    sc3.markdown(f'<div class="kpi-card"><div class="kpi-label">CPA 최고 월</div>'
                 f'<div class="kpi-value" style="color:#e74c3c">{cpa_max_row["month"].replace("2025-","").lstrip("0")}월</div>'
                 f'<div style="font-size:.82rem;color:#6c757d">{cpa_max_row["첫거래CPA"]:,.0f}원</div></div>', unsafe_allow_html=True)
    sc4.markdown(f'<div class="kpi-card"><div class="kpi-label">광고비 최대 월</div>'
                 f'<div class="kpi-value" style="color:#e74c3c">{bud_max_row["month"].replace("2025-","").lstrip("0")}월</div>'
                 f'<div style="font-size:.82rem;color:#6c757d">{bud_max_row["광고비"]/1e8:.1f}억</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # 월별 테이블 (접힘)
    with st.expander("월별 전체 성과표 (클릭해서 펼치기)", expanded=False):
        def row_style(row):
            m_str = "2025-" + row["월"].replace("월","").zfill(2)
            if m_str in HIGH_SEASON: return ["background-color:#fff5f5"]*len(row)
            elif m_str in LOW_SEASON: return ["background-color:#f0fff4"]*len(row)
            return [""]*len(row)
        tdf = pd.DataFrame([{
            "월": r["month"].replace("2025-","")+"월",
            "광고비": f"{r['광고비']/1e8:.1f}억",
            "MoM%": f"{r['광고비MoM']:+.0f}%" if not pd.isna(r["광고비MoM"]) else "—",
            "첫거래": f"{r['첫거래']/1e4:.1f}만",
            "거래MoM%": f"{r['거래MoM']:+.0f}%" if not pd.isna(r["거래MoM"]) else "—",
            "CPC": f"{r['CPC']:,.0f}원",
            "첫거래CPA": f"{r['첫거래CPA']:,.0f}원",
            "효율비": f"{r['효율비']:.2f}",
            "시즌": r["시즌"],
        } for _, r in monthly.iterrows()])
        st.dataframe(tdf.style.apply(row_style, axis=1), hide_index=True, use_container_width=True)
        st.caption("🔴 = 고비용 시즌(3·9·12월) · 🟢 = 고효율 시즌(1·7월)")

    # ── 월별 성과 — 지표별 미니 차트 5개 ────────────────────────
    st.markdown("**월별 성과 흐름 — 지표별 변화 패턴**")
    st.caption("🔴 배경 = 고비용 시즌(3·9·12월) · 🟢 배경 = 고효율 시즌(1·7월)")

    month_labels = [m.replace("2025-","")+"월" for m in monthly["month"]]
    months_list  = monthly["month"].tolist()

    def _season_shapes(months_list):
        shapes = []
        for m in HIGH_SEASON:
            if m in months_list:
                i = months_list.index(m)
                shapes.append(dict(type="rect", xref="x", yref="paper",
                    x0=i-0.5, x1=i+0.5, y0=0, y1=1,
                    fillcolor="rgba(231,76,60,0.10)", line_width=0))
        for m in LOW_SEASON:
            if m in months_list:
                i = months_list.index(m)
                shapes.append(dict(type="rect", xref="x", yref="paper",
                    x0=i-0.5, x1=i+0.5, y0=0, y1=1,
                    fillcolor="rgba(46,204,113,0.10)", line_width=0))
        return shapes

    _common_layout = dict(
        margin=dict(t=36, b=24, l=8, r=8),
        height=180,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(tickfont=dict(size=10), showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=10)),
        showlegend=False,
    )

    mc1, mc2, mc3 = st.columns(3)

    # 1) 효율비 — 라인 + 1.0 기준선
    with mc1:
        st.markdown('<div style="font-size:.82rem;font-weight:600;color:#374151;margin-bottom:.2rem">효율비 (월별)</div>', unsafe_allow_html=True)
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=month_labels, y=monthly["효율비"],
            mode="lines+markers+text",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=6,
                color=["#2ecc71" if v >= 1.0 else "#e74c3c" for v in monthly["효율비"]]),
            text=[f"{v:.2f}" for v in monthly["효율비"]],
            textposition="top center",
            textfont=dict(size=9),
        ))
        fig_e.add_hline(y=1.0, line_dash="dot", line_color="#6c757d", line_width=1.5)
        fig_e.update_layout(**_common_layout, shapes=_season_shapes(months_list),
            title=dict(text="1.0 기준선 이상 = 효율 양호", font=dict(size=11), x=0))
        st.plotly_chart(fig_e, use_container_width=True)

    # 2) 첫거래CPA — 라인
    with mc2:
        st.markdown('<div style="font-size:.82rem;font-weight:600;color:#374151;margin-bottom:.2rem">첫거래CPA (월별, 원)</div>', unsafe_allow_html=True)
        avg_cpa = monthly["첫거래CPA"].mean()
        fig_c = go.Figure()
        fig_c.add_hline(y=avg_cpa, line_dash="dot", line_color="#adb5bd", line_width=1.2)
        fig_c.add_trace(go.Scatter(
            x=month_labels, y=monthly["첫거래CPA"],
            mode="lines+markers+text",
            line=dict(color="#f39c12", width=2),
            marker=dict(size=6,
                color=["#e74c3c" if v > avg_cpa else "#2ecc71" for v in monthly["첫거래CPA"]]),
            text=[f"{v:,.0f}" for v in monthly["첫거래CPA"]],
            textposition="top center",
            textfont=dict(size=9),
        ))
        fig_c.update_layout(**_common_layout, shapes=_season_shapes(months_list),
            title=dict(text=f"평균 {avg_cpa:,.0f}원 점선", font=dict(size=11), x=0))
        st.plotly_chart(fig_c, use_container_width=True)

    # 3) 첫거래 볼륨 — 막대
    with mc3:
        st.markdown('<div style="font-size:.82rem;font-weight:600;color:#374151;margin-bottom:.2rem">첫거래 건수 (월별, 만건)</div>', unsafe_allow_html=True)
        ft_vals = monthly["첫거래"] / 1e4
        avg_ft  = ft_vals.mean()
        fig_ft = go.Figure()
        fig_ft.add_trace(go.Bar(
            x=month_labels, y=ft_vals,
            marker_color=["#3b82f6" if v >= avg_ft else "#93c5fd" for v in ft_vals],
            text=[f"{v:.1f}만" for v in ft_vals],
            textposition="outside", textfont=dict(size=9),
        ))
        fig_ft.add_hline(y=avg_ft, line_dash="dot", line_color="#6c757d", line_width=1.2)
        fig_ft.update_layout(**_common_layout, shapes=_season_shapes(months_list),
            title=dict(text=f"평균 {avg_ft:.1f}만건 점선", font=dict(size=11), x=0))
        st.plotly_chart(fig_ft, use_container_width=True)

    mc4, mc5, _ = st.columns(3)

    # 4) 광고비MoM% — 양/음 색상 막대
    with mc4:
        st.markdown('<div style="font-size:.82rem;font-weight:600;color:#374151;margin-bottom:.2rem">광고비 MoM% (전월 대비)</div>', unsafe_allow_html=True)
        mom_ad = monthly["광고비MoM"].fillna(0)
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Bar(
            x=month_labels, y=mom_ad,
            marker_color=["#e74c3c" if v > 0 else "#2ecc71" for v in mom_ad],
            text=[f"{v:+.0f}%" for v in mom_ad],
            textposition="outside", textfont=dict(size=9),
        ))
        fig_ma.add_hline(y=0, line_color="#6c757d", line_width=1)
        fig_ma.update_layout(**_common_layout, shapes=_season_shapes(months_list),
            title=dict(text="빨강 = 전월 대비 증가 · 초록 = 감소", font=dict(size=11), x=0))
        st.plotly_chart(fig_ma, use_container_width=True)

    # 5) 거래MoM% — 양/음 색상 막대
    with mc5:
        st.markdown('<div style="font-size:.82rem;font-weight:600;color:#374151;margin-bottom:.2rem">첫거래 MoM% (전월 대비)</div>', unsafe_allow_html=True)
        mom_ft = monthly["거래MoM"].fillna(0)
        fig_mf = go.Figure()
        fig_mf.add_trace(go.Bar(
            x=month_labels, y=mom_ft,
            marker_color=["#2ecc71" if v > 0 else "#e74c3c" for v in mom_ft],
            text=[f"{v:+.0f}%" for v in mom_ft],
            textposition="outside", textfont=dict(size=9),
        ))
        fig_mf.add_hline(y=0, line_color="#6c757d", line_width=1)
        fig_mf.update_layout(**_common_layout, shapes=_season_shapes(months_list),
            title=dict(text="초록 = 전월 대비 증가 · 빨강 = 감소", font=dict(size=11), x=0))
        st.plotly_chart(fig_mf, use_container_width=True)

    st.divider()

    # ── 차트 — vrect로 시즌 배경 강조 ─────────────────────────────
    def add_season_vrect(fig):
        months_list = monthly["month"].tolist()
        for m_str in HIGH_SEASON:
            if m_str in months_list:
                idx = months_list.index(m_str)
                fig.add_vrect(x0=idx-0.5, x1=idx+0.5, fillcolor="rgba(231,76,60,0.10)", line_width=0)
        for m_str in LOW_SEASON:
            if m_str in months_list:
                idx = months_list.index(m_str)
                fig.add_vrect(x0=idx-0.5, x1=idx+0.5, fillcolor="rgba(46,204,113,0.12)", line_width=0)
        return fig

    months_x = list(range(len(monthly)))

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        bar_colors = ["#e74c3c" if m in HIGH_SEASON else ("#2ecc71" if m in LOW_SEASON else "#3498db")
                      for m in monthly["month"]]
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=monthly["month"], y=monthly["광고비"]/1e8, name="광고비(억)",
                             marker_color=bar_colors, opacity=0.8), secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["첫거래CPA"], name="첫거래CPA",
                                 line=dict(color="#e67e22",width=2.5), mode="lines+markers"), secondary_y=True)
        fig.update_layout(title="예산이 높을수록 CPA도 높다 — 3·9·12월이 최악", height=300,
                          legend=dict(orientation="h",y=-0.3), margin=dict(t=45,b=10))
        fig.update_yaxes(title_text="광고비(억)", secondary_y=False)
        fig.update_yaxes(title_text="첫거래CPA(원)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_c2:
        fig2 = go.Figure()
        eff_colors = ["#e74c3c" if m in HIGH_SEASON else ("#2ecc71" if m in LOW_SEASON else "#9b59b6")
                      for m in monthly["month"]]
        fig2.add_trace(go.Scatter(x=monthly["month"], y=monthly["효율비"],
                                  line=dict(color="#9b59b6",width=2.5), mode="lines+markers",
                                  marker=dict(color=eff_colors, size=9), name="효율비"))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="#e74c3c", annotation_text="손익분기 1.0")
        for m_str in HIGH_SEASON:
            if m_str in monthly["month"].values:
                fig2.add_vrect(x0=monthly[monthly["month"]==m_str].index[0]-0.5,
                               x1=monthly[monthly["month"]==m_str].index[0]+0.5,
                               fillcolor="rgba(231,76,60,0.08)", line_width=0)
        for m_str in LOW_SEASON:
            if m_str in monthly["month"].values:
                fig2.add_vrect(x0=monthly[monthly["month"]==m_str].index[0]-0.5,
                               x1=monthly[monthly["month"]==m_str].index[0]+0.5,
                               fillcolor="rgba(46,204,113,0.10)", line_width=0)
        fig2.update_layout(title="1월·7월만 효율비 1.0 초과 — 나머지는 손익분기 미달",
                           height=300, margin=dict(t=45,b=10),
                           xaxis=dict(tickvals=list(range(len(monthly))), ticktext=monthly["month"].tolist()))
        st.plotly_chart(fig2, use_container_width=True)

    col_c3, col_c4 = st.columns(2)
    with col_c3:
        monthly_v = monthly.dropna(subset=["광고비MoM","거래MoM"])
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=monthly_v["month"], y=monthly_v["광고비MoM"],
                              name="광고비 MoM%", marker_color="#3498db", opacity=0.7))
        fig3.add_trace(go.Bar(x=monthly_v["month"], y=monthly_v["거래MoM"],
                              name="첫거래 MoM%", marker_color="#2ecc71", opacity=0.7))
        fig3.add_hline(y=0, line_color="#aaa", line_width=1)
        fig3.update_layout(barmode="group",
                           title="광고비 증가가 항상 거래 증가를 초과 — 규모의 비효율",
                           height=280, margin=dict(t=45,b=10), legend=dict(orientation="h",y=-0.35))
        st.plotly_chart(fig3, use_container_width=True)
    with col_c4:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=monthly["month"], y=monthly["반복사용률"],
                                  line=dict(color="#27ae60",width=2.5), mode="lines+markers", name="반복사용률(%)"))
        fig4.update_layout(title="3월 유입이 4~5월 반복 피크를 만든다 — 광고 효과는 1~2개월 시차",
                           height=280, margin=dict(t=45,b=10))
        st.plotly_chart(fig4, use_container_width=True)

    st.caption("🔴 배경 = 고비용 시즌 (3·9·12월) · 🟢 배경 = 고효율 시즌 (1·7월) · 시즌별 마케팅 제언 → '마케팅 제언' 탭")


# ════════════════════════════════════════════════════════════════════
# TAB 2 — CHANNEL
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("채널별 성과 분석")

    ch_list = ["구글","페이스북","네이버검색"]
    ch_metrics = {}
    for ch in ch_list:
        sub = df[df["channel"]==ch]
        if len(sub) == 0: continue
        ch_metrics[ch] = calc_metrics(sub, TOTAL_COST_ALL, TOTAL_FT_ALL)

    # ── 채널 액션 카드 ──────────────────────────────────────────
    verdicts_map = {"구글":"증액 권장","페이스북":"선별 유지","네이버검색":"감액 필요"}
    vcolors      = {"증액 권장":"#2ecc71","선별 유지":"#f39c12","감액 필요":"#e74c3c"}
    vdescs       = {
        "구글":    "효율비 1.65 · 첫거래CPA 최저\n전 단계에서 구조적 우위",
        "페이스북": "효율비 1.04 · 손익분기 수준\n계좌개설+리타겟 조합만 유지",
        "네이버검색":"효율비 0.45 · CPA 3.65배 비쌈\nCTR 11%는 브랜드 착시",
    }
    total_c = df["광고비"].sum()
    cols3 = st.columns(3)
    for i, (ch, cm) in enumerate(ch_metrics.items()):
        v = verdicts_map.get(ch,"유지")
        vc = vcolors.get(v,"#6c757d")
        cost = df[df["channel"]==ch]["광고비"].sum()
        pct  = round(safe_div(cost,total_c)*100,1)
        desc = vdescs.get(ch,"")
        cols3[i].markdown(
            f'<div class="kpi-card" style="border-top:4px solid {vc};text-align:left">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div class="kpi-label">예산 {pct}%</div>'
            f'<span style="background:{vc};color:#fff;padding:2px 10px;border-radius:12px;font-size:.8rem;font-weight:700">{v}</span></div>'
            f'<div class="kpi-value" style="margin:.3rem 0">{ch.replace("검색","")}</div>'
            f'<div style="font-size:.82rem;color:#495057;white-space:pre-line">{desc}</div></div>',
            unsafe_allow_html=True)

    # ── CPA 계단 비교 & 퍼널 전환율 비교 ──────────────────────────
    cmp_col1, cmp_col2 = st.columns(2)
    ch_order   = ["구글","페이스북","네이버검색"]
    bar_colors = {"구글":"#3b82f6","페이스북":"#f59e0b","네이버검색":"#ef4444"}

    with cmp_col1:
        st.markdown("**CPA 계단 비교 (원)**")
        cpa_steps = [("가입CPA","가입CPA"),("첫거래CPA","첫거래CPA"),("자동이체CPA","자동이체CPA")]
        for step_label, key in cpa_steps:
            st.markdown(f'<div style="font-size:.8rem;color:#6c757d;font-weight:600;margin:.7rem 0 .2rem">{step_label}</div>', unsafe_allow_html=True)
            vals = [(ch, ch_metrics.get(ch,{}).get(key,0)) for ch in ch_order if ch in ch_metrics]
            max_val = max(v for _,v in vals) if vals else 1
            for ch, val in vals:
                bar_pct = round(val / max_val * 100) if max_val else 0
                color = bar_colors.get(ch,"#6c757d")
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:.5rem;margin:.15rem 0">'
                    f'<div style="width:4.5rem;font-size:.78rem;color:#374151">{ch.replace("검색","")}</div>'
                    f'<div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px">'
                    f'<div style="width:{bar_pct}%;background:{color};height:100%;border-radius:4px"></div></div>'
                    f'<div style="width:5.5rem;text-align:right;font-size:.82rem;font-weight:600;color:{color}">{val:,.0f}원</div>'
                    f'</div>', unsafe_allow_html=True)

    with cmp_col2:
        st.markdown("**퍼널 전환율 비교 (%)**")

        st.markdown('<div style="font-size:.8rem;color:#6c757d;font-weight:600;margin:.7rem 0 .2rem">클릭→설치율</div>', unsafe_allow_html=True)
        for ch in ch_order:
            if ch not in ch_metrics: continue
            val   = ch_metrics[ch].get("클릭→설치율",0)
            color = bar_colors.get(ch,"#6c757d")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:.5rem;margin:.15rem 0">'
                f'<div style="width:4.5rem;font-size:.78rem;color:#374151">{ch.replace("검색","")}</div>'
                f'<div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px">'
                f'<div style="width:{min(val,100):.0f}%;background:{color};height:100%;border-radius:4px"></div></div>'
                f'<div style="width:4rem;text-align:right;font-size:.82rem;font-weight:600;color:{color}">{val:.1f}%</div>'
                f'</div>', unsafe_allow_html=True)

        st.markdown('<div style="font-size:.8rem;color:#6c757d;font-weight:600;margin:.7rem 0 .2rem">설치→가입 / 가입→계좌 / 계좌→거래</div>', unsafe_allow_html=True)
        for ch in ch_order:
            if ch not in ch_metrics: continue
            sub = df[df["channel"]==ch][VOL].sum()
            v1  = safe_div(sub["회원가입"], sub["앱설치"])   * 100
            v2  = safe_div(sub["계좌개설"], sub["회원가입"]) * 100
            v3  = safe_div(sub["첫거래"],   sub["계좌개설"]) * 100
            color = bar_colors.get(ch,"#6c757d")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:.5rem;margin:.15rem 0">'
                f'<div style="width:4.5rem;font-size:.78rem;color:#374151">{ch.replace("검색","")}</div>'
                f'<div style="font-size:.8rem;color:{color};font-weight:600">{v1:.1f}% / {v2:.1f}% / {v3:.1f}%</div>'
                f'</div>', unsafe_allow_html=True)
        st.caption("→ 설치 이후 전환율은 3채널 모두 거의 동일")

        st.markdown('<div style="font-size:.8rem;color:#6c757d;font-weight:600;margin:.7rem 0 .2rem">CTR (클릭률)</div>', unsafe_allow_html=True)
        ctr_vals = [(ch, ch_metrics.get(ch,{}).get("CTR",0)) for ch in ch_order if ch in ch_metrics]
        max_ctr  = max(v for _,v in ctr_vals) if ctr_vals else 1
        for ch, val in ctr_vals:
            bar_pct = round(val / max_ctr * 100) if max_ctr else 0
            color   = bar_colors.get(ch,"#6c757d")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:.5rem;margin:.15rem 0">'
                f'<div style="width:4.5rem;font-size:.78rem;color:#374151">{ch.replace("검색","")}</div>'
                f'<div style="flex:1;background:#f1f5f9;border-radius:4px;height:18px">'
                f'<div style="width:{bar_pct}%;background:{color};height:100%;border-radius:4px"></div></div>'
                f'<div style="width:4rem;text-align:right;font-size:.82rem;font-weight:600;color:{color}">{val:.2f}%</div>'
                f'</div>', unsafe_allow_html=True)

    st.divider()

    # ── 전체 지표 테이블 (접힘) ─────────────────────────────────
    with st.expander("채널 전체 지표 비교표", expanded=False):
        col_metrics = [
            ("광고비","광고비","₩{:,.0f}"),("예산 비중","",""),
            ("CTR","CTR","{:.2f}%"),("CPC","CPC","₩{:,.0f}"),
            ("클릭→설치율","클릭→설치율","{:.1f}%"),("CPI","CPI","₩{:,.0f}"),
            ("가입CPA","가입CPA","₩{:,.0f}"),("첫거래CPA","첫거래CPA","₩{:,.0f}"),
            ("자동이체CPA","자동이체CPA","₩{:,.0f}"),("효율비","효율비","{:.2f}"),
        ]
        ch_names = list(ch_metrics.keys())
        metric_rows = []
        for label, key, fmt in col_metrics:
            row = {"지표": label}
            for ch in ch_names:
                cm = ch_metrics[ch]
                if label == "예산 비중":
                    cost = df[df["channel"]==ch]["광고비"].sum()
                    row[ch.replace("검색","")] = f"{safe_div(cost,total_c)*100:.1f}%"
                elif key:
                    v = cm.get(key, 0)
                    try: row[ch.replace("검색","")] = fmt.format(v)
                    except: row[ch.replace("검색","")] = str(v)
                else:
                    row[ch.replace("검색","")] = "—"
            vals = {ch: ch_metrics[ch].get(key,0) for ch in ch_names if key and key in ch_metrics[ch]}
            if vals:
                best_ch = min(vals, key=vals.get) if "CPA" in label or label in ["CPC","CPI"] else max(vals, key=vals.get)
                row["최고"] = best_ch.replace("검색","")
            else:
                row["최고"] = "—"
            metric_rows.append(row)
        mdf = pd.DataFrame(metric_rows)
        st.dataframe(mdf, hide_index=True, use_container_width=True)
        st.caption("⚠ CTR은 채널 품질 지표가 아님 — 네이버 CTR 11%는 브랜드 검색 수요로 착시 발생")

    # ── CPA 계단 차트 ───────────────────────────────────────────
    cpa_stage_keys = ["가입CPA","계좌CPA","첫거래CPA","반복CPA","자동이체CPA","추천CPA"]
    cpa_labels     = ["가입","계좌","첫거래","반복","자동이체","추천"]
    ch_colors = {"구글":"#3498db","페이스북":"#9b59b6","네이버검색":"#e74c3c"}

    fig_cpa = go.Figure()
    for ch, cm in ch_metrics.items():
        fig_cpa.add_trace(go.Bar(
            y=cpa_labels, x=[cm.get(k,0) for k in cpa_stage_keys],
            name=ch.replace("검색",""), orientation="h",
            marker_color=ch_colors.get(ch,"#95a5a6"),
            text=[f"₩{cm.get(k,0):,.0f}" for k in cpa_stage_keys],
            textposition="outside"
        ))
    fig_cpa.update_layout(barmode="group", height=360, xaxis_type="log",
                          title="구글이 전 단계 CPA에서 압도적 우위 — 네이버와 최대 15배 격차",
                          xaxis_title="CPA (원, 로그 스케일)",
                          legend=dict(orientation="h",y=1.15), margin=dict(t=50,b=10))
    st.plotly_chart(fig_cpa, use_container_width=True)

    # ── 산점도 2개 ──────────────────────────────────────────────
    sc1, sc2 = st.columns(2)

    with sc1:
        scatter_data = []
        for ch, cm in ch_metrics.items():
            scatter_data.append({
                "채널": ch.replace("검색",""),
                "CTR(%)": cm["CTR"],
                "클릭→설치율(%)": cm["클릭→설치율"],
                "광고비": cm["광고비"],
            })
        sdf = pd.DataFrame(scatter_data)
        fig_s1 = px.scatter(
            sdf, x="CTR(%)", y="클릭→설치율(%)",
            size="광고비", color="채널",
            color_discrete_map={"구글":"#3498db","페이스북":"#9b59b6","네이버":"#e74c3c"},
            size_max=60, text="채널",
            title="네이버는 CTR은 높지만 설치율이 낮아 CPA가 악화됩니다"
        )
        fig_s1.update_traces(textposition="top center")
        # 네이버 주석
        naver_row = sdf[sdf["채널"]=="네이버"]
        if len(naver_row) > 0:
            fig_s1.add_annotation(
                x=naver_row["CTR(%)"].values[0],
                y=naver_row["클릭→설치율(%)"].values[0] - 2,
                text="CTR은 높지만<br>설치율 낮음",
                showarrow=True, arrowhead=2, arrowcolor="#e74c3c",
                font=dict(color="#e74c3c", size=11),
                bgcolor="rgba(255,245,245,0.9)", bordercolor="#e74c3c"
            )
        fig_s1.update_layout(height=340, margin=dict(t=50,b=10))
        st.plotly_chart(fig_s1, use_container_width=True)

    with sc2:
        scatter_data2 = []
        for ch, cm in ch_metrics.items():
            scatter_data2.append({
                "채널": ch.replace("검색",""),
                "CPC(원)": cm["CPC"],
                "첫거래CPA(원)": cm["첫거래CPA"],
                "광고비": cm["광고비"],
            })
        sdf2 = pd.DataFrame(scatter_data2)
        avg_cpc = sdf2["CPC(원)"].mean()
        avg_cpa = sdf2["첫거래CPA(원)"].mean()
        fig_s2 = px.scatter(
            sdf2, x="CPC(원)", y="첫거래CPA(원)",
            size="광고비", color="채널",
            color_discrete_map={"구글":"#3498db","페이스북":"#9b59b6","네이버":"#e74c3c"},
            size_max=60, text="채널",
            title="네이버가 고CPC·고CPA 위험 구간에 홀로 위치합니다"
        )
        fig_s2.update_traces(textposition="top center")
        fig_s2.add_vline(x=avg_cpc, line_dash="dash", line_color="#95a5a6",
                         annotation_text="평균 CPC", annotation_position="top right")
        fig_s2.add_hline(y=avg_cpa, line_dash="dash", line_color="#95a5a6",
                         annotation_text="평균 CPA", annotation_position="bottom right")
        fig_s2.add_annotation(
            x=sdf2[sdf2["채널"]=="네이버"]["CPC(원)"].values[0],
            y=sdf2[sdf2["채널"]=="네이버"]["첫거래CPA(원)"].values[0] + 200,
            text="⚠ 고CPC·고CPA<br>위험 구간",
            showarrow=False,
            font=dict(color="#e74c3c", size=11),
            bgcolor="rgba(255,245,245,0.9)", bordercolor="#e74c3c"
        )
        fig_s2.update_layout(height=340, margin=dict(t=50,b=10))
        st.plotly_chart(fig_s2, use_container_width=True)

    # ── CVR 비교 차트 ───────────────────────────────────────────
    cvr_keys   = ["클릭→설치율","설치→가입율","가입→계좌율","계좌→거래율","거래→반복율","반복→자동율","자동→추천율"]
    cvr_labels = ["클릭→설치","설치→가입","가입→계좌","계좌→거래","거래→반복","반복→자동","자동→추천"]
    fig_cvr = go.Figure()
    for ch, cm in ch_metrics.items():
        fig_cvr.add_trace(go.Bar(
            y=cvr_labels, x=[cm.get(k,0) for k in cvr_keys],
            name=ch.replace("검색",""), orientation="h",
            marker_color=ch_colors.get(ch,"#95a5a6"),
            text=[f"{cm.get(k,0):.1f}%" for k in cvr_keys],
            textposition="outside"
        ))
    fig_cvr.update_layout(barmode="group", height=340,
                          title="클릭→설치율만 채널 차이 — 설치 이후는 채널이 바뀌어도 ±0.1%p 이내 고정",
                          xaxis_title="전환율(%)", legend=dict(orientation="h",y=1.15), margin=dict(t=50,b=10))
    st.plotly_chart(fig_cvr, use_container_width=True)

    st.markdown(
        '<div class="info-box">⚠ <b>핵심 발견</b>: 설치 이후 전환율은 채널을 바꿔도 <b>±0.1%p 이내로 고정</b>. '
        '채널이 영향을 주는 구간은 <b>클릭→설치율(48~61%)뿐</b>이다.</div>', unsafe_allow_html=True)

    st.caption("↗ 채널 판정 상세 근거는 '마케팅 제언' 탭에서 확인하세요.")
    st.divider()

    # ── 예산 재배분 시뮬레이터 ─────────────────────────────────
    st.markdown("**예산 재배분 시뮬레이터**")
    naver_cpa  = ch_metrics.get("네이버검색",{}).get("첫거래CPA",8330)
    google_cpa = ch_metrics.get("구글",{}).get("첫거래CPA",2281)
    sim_amt = st.number_input("네이버에서 구글로 이관할 금액 (억원)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    if naver_cpa > 0 and google_cpa > 0:
        add_ft = (sim_amt*1e8/google_cpa) - (sim_amt*1e8/naver_cpa)
        cpa_saving = (naver_cpa - google_cpa) / naver_cpa * 100
        sc_s1, sc_s2, sc_s3 = st.columns(3)
        sc_s1.metric("예상 추가 첫거래", f"+{add_ft:,.0f}건")
        sc_s2.metric("CPA 절감률", f"-{cpa_saving:.1f}%")
        sc_s3.metric("이관 후 CPA", f"{google_cpa:,.0f}원 (구글 기준)")


# ════════════════════════════════════════════════════════════════════
# TAB 3 — FUNNEL
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("퍼널 전환율 분석")

    s = df[VOL].sum()

    # 담당팀별 색상 정의
    TEAM_AD      = "#3b82f6"   # 파랑 — 광고팀
    TEAM_PRODUCT = "#22c55e"   # 초록 — 상품·온보딩
    TEAM_CRM     = "#a855f7"   # 보라 — CRM·리텐션

    funnel_stages = [
        ("광고노출",     s["광고노출"],     None,                                      "광고팀",        TEAM_AD),
        ("광고클릭",     s["광고클릭"],     safe_div(s["광고클릭"],s["광고노출"])*100,  "광고팀",        TEAM_AD),
        ("앱설치",       s["앱설치"],       safe_div(s["앱설치"],s["광고클릭"])*100,    "광고팀(ASO)",   TEAM_AD),
        ("앱실행",       s["앱실행"],       safe_div(s["앱실행"],s["앱설치"])*100,      "상품·온보딩",   TEAM_PRODUCT),
        ("회원가입",     s["회원가입"],     safe_div(s["회원가입"],s["앱실행"])*100,    "상품·온보딩",   TEAM_PRODUCT),
        ("계좌개설",     s["계좌개설"],     safe_div(s["계좌개설"],s["회원가입"])*100,  "광고팀·상품팀", TEAM_PRODUCT),
        ("첫거래",       s["첫거래"],       safe_div(s["첫거래"],s["계좌개설"])*100,    "상품팀",        TEAM_PRODUCT),
        ("반복사용",     s["반복사용"],     safe_div(s["반복사용"],s["첫거래"])*100,    "상품팀",        TEAM_PRODUCT),
        ("자동이체설정", s["자동이체설정"], safe_div(s["자동이체설정"],s["반복사용"])*100,"CRM·리텐션",  TEAM_CRM),
        ("추천완료",     s["추천완료"],     safe_div(s["추천완료"],s["자동이체설정"])*100,"CRM·리텐션",  TEAM_CRM),
    ]

    # ── 구간 범례 ────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;gap:1.2rem;margin-bottom:1rem">'
        f'<span style="background:{TEAM_AD}22;border:1px solid {TEAM_AD};color:{TEAM_AD};font-size:.8rem;padding:3px 12px;border-radius:10px;font-weight:600">● 광고팀 구간</span>'
        f'<span style="background:{TEAM_PRODUCT}22;border:1px solid {TEAM_PRODUCT};color:{TEAM_PRODUCT};font-size:.8rem;padding:3px 12px;border-radius:10px;font-weight:600">● 상품·온보딩 구간</span>'
        f'<span style="background:{TEAM_CRM}22;border:1px solid {TEAM_CRM};color:{TEAM_CRM};font-size:.8rem;padding:3px 12px;border-radius:10px;font-weight:600">● CRM·리텐션 구간</span>'
        '</div>', unsafe_allow_html=True)

    # ── 퍼널 수평 바 (색상 구분) ─────────────────────────────────
    pct_vals = [safe_div(f[1], s["광고노출"])*100 for f in funnel_stages]
    bar_colors_f = [f[4] for f in funnel_stages]
    fig_funnel_h = go.Figure(go.Bar(
        y=[f[0] for f in funnel_stages],
        x=pct_vals,
        orientation="h",
        marker_color=bar_colors_f,
        text=[f"{v:.3f}%" for v in pct_vals],
        textposition="outside"
    ))
    fig_funnel_h.update_layout(height=380, xaxis_type="log",
                               title="45억 노출에서 50만 추천까지 — 각 단계 누적 이탈의 결과",
                               xaxis_title="노출 대비 누적 전환율 (로그, %)",
                               margin=dict(t=50,b=10,r=20))
    st.plotly_chart(fig_funnel_h, use_container_width=True)

    # ── 단계별 담당 주체 태그 테이블 ─────────────────────────────
    with st.expander("단계별 전환율 & 담당 주체 상세", expanded=False):
        funnel_df = pd.DataFrame([{
            "단계": f[0],
            "인원 수": f"{f[1]:,.0f}",
            "전단계 대비": f"{f[2]:.2f}%" if f[2] is not None else "—",
            "이탈자": f"{f[1] - funnel_stages[i+1][1]:,.0f}" if i < len(funnel_stages)-1 else "—",
            "누적(노출 대비)": f"{safe_div(f[1],s['광고노출'])*100:.3f}%",
            "담당": f[3],
        } for i, f in enumerate(funnel_stages)])
        st.dataframe(funnel_df, hide_index=True, use_container_width=True)

    st.divider()

    # ── 이탈 TOP 3 카드 ─────────────────────────────────────────
    st.markdown("### 이탈 TOP 3 — 절대 이탈자 기준")
    top3_dropoffs = [
        ("광고노출 → 광고클릭",
         s["광고노출"]-s["광고클릭"], 1-safe_div(s["광고클릭"],s["광고노출"]),
         "광고팀", TEAM_AD,
         "CTR 1.03% — 노출 대비 98.97%가 클릭하지 않는다",
         "소재 A/B 테스트 + 타겟 정밀도 개선"),
        ("광고클릭 → 앱설치",
         s["광고클릭"]-s["앱설치"], 1-safe_div(s["앱설치"],s["광고클릭"]),
         "광고팀(ASO)", TEAM_AD,
         "클릭의 43.77%가 설치로 이어지지 않는다 (네이버 48.1%가 하한 견인)",
         "앱스토어 스크린샷 리뉴얼 + 딥링크 최적화"),
        ("앱실행 → 회원가입",
         s["앱실행"]-s["회원가입"], 1-safe_div(s["회원가입"],s["앱실행"]),
         "상품·온보딩", TEAM_PRODUCT,
         "앱 실행자의 22.70%가 회원가입 전 이탈",
         "가입 단계 슬림화 + 첫 화면 혜택 넛지"),
    ]

    t3cols = st.columns(3)
    for i, (label, dropoff_n, dropoff_r, owner, color, desc, action) in enumerate(top3_dropoffs):
        t3cols[i].markdown(
            f'<div style="border:2px solid {color};border-radius:12px;padding:1.2rem;height:100%">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.6rem">'
            f'<span style="background:{color};color:#fff;font-size:.75rem;padding:2px 8px;border-radius:8px;font-weight:700">#{i+1} 최대 이탈</span>'
            f'<span style="font-size:1.5rem;font-weight:800;color:{color}">-{dropoff_r*100:.1f}%</span></div>'
            f'<div style="font-weight:700;font-size:.95rem;margin-bottom:.4rem">{label}</div>'
            f'<div style="font-size:.83rem;color:#495057;margin-bottom:.5rem">{desc}</div>'
            f'<div style="background:#f8f9fa;border-radius:6px;height:8px;margin:.5rem 0">'
            f'<div style="background:{color};height:8px;border-radius:6px;width:{min(dropoff_r*100,100):.0f}%"></div></div>'
            f'<div style="font-size:.78rem;color:#6c757d;margin-bottom:.4rem">이탈자 {dropoff_n/1e4:.1f}만명 · 담당: {owner}</div>'
            f'<div style="font-size:.8rem;color:{color};font-weight:600">→ {action}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── 반복→자동이체 병목 강조 ──────────────────────────────────
    auto_rate = safe_div(s["자동이체설정"],s["반복사용"])*100
    st.markdown(
        f'<div style="border:3px solid {TEAM_CRM};border-radius:12px;padding:1.2rem;background:#fdf4ff;margin-top:.8rem">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<div><span style="background:{TEAM_CRM};color:#fff;font-size:.75rem;padding:2px 8px;border-radius:8px;font-weight:700">⚠ 전체 퍼널 최대 병목</span>'
        f'<div style="font-weight:700;font-size:1.05rem;margin:.4rem 0">반복사용 → 자동이체설정 (단계 내 이탈률 78.1%)</div>'
        f'<div style="font-size:.87rem;color:#495057">채널·소재·광고그룹 변경에도 21.86~21.98%로 고정 — <b>광고팀이 개입할 수 없는 구간</b></div>'
        f'<div style="font-size:.83rem;color:{TEAM_CRM};margin-top:.4rem">→ 상품 UX 개선 + CRM 시퀀스(D+1/D+3/D+7)만이 유일한 해결책</div></div>'
        f'<div style="text-align:center;padding-left:2rem">'
        f'<div style="font-size:2.5rem;font-weight:900;color:{TEAM_CRM}">{auto_rate:.1f}%</div>'
        f'<div style="font-size:.78rem;color:#6b21a8">반복→자동이체<br>전환율</div></div>'
        f'</div></div>', unsafe_allow_html=True)

    st.caption("↗ 이탈 구간별 원인 가설 & 개선 액션은 '마케팅 제언' 탭에서 확인하세요.")



# ════════════════════════════════════════════════════════════════════
# TAB 4 — CAMPAIGN & CREATIVE
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("캠페인 & 소재 성과")

    # ── 4중 교차 데이터 준비 ────────────────────────────────────
    cross = df.groupby(["channel","campaign_objective","ad_group","creative_format"])[VOL].sum().reset_index()
    cross_rows = []
    for _, row in cross.iterrows():
        sub = df[
            (df["channel"]==row["channel"]) & (df["campaign_objective"]==row["campaign_objective"]) &
            (df["ad_group"]==row["ad_group"]) & (df["creative_format"]==row["creative_format"])
        ]
        cm = calc_metrics(sub, TOTAL_COST_ALL, TOTAL_FT_ALL)
        v = verdict(cm["효율비"], cm["자동이체CPA"])
        cross_rows.append({
            "채널": row["channel"].replace("검색",""),
            "목적": row["campaign_objective"],
            "광고그룹": row["ad_group"],
            "소재": row["creative_format"],
            "광고비": cm["광고비"],
            "효율비": cm["효율비"],
            "첫거래CPA": cm["첫거래CPA"],
            "자동이체CPA": cm["자동이체CPA"],
            "추천CPA": cm["추천CPA"],
            "판정": v,
        })
    cross_df = pd.DataFrame(cross_rows)

    # ── CPA × 효율비 액션 매트릭스 산점도 ──────────────────────
    st.markdown("**CPA × 효율비 액션 매트릭스 — 예산 배분 의사결정**")
    verdict_colors = {"확대":"#2ecc71","유지":"#3498db","개선테스트":"#f39c12","즉시중단":"#e74c3c"}
    cross_df["판정_색"] = cross_df["판정"].map(verdict_colors)
    cross_df["레이블"] = cross_df["채널"] + "<br>" + cross_df["목적"] + "<br>" + cross_df["광고그룹"]

    fig_matrix = go.Figure()
    for v_name, v_color in verdict_colors.items():
        sub_v = cross_df[cross_df["판정"]==v_name]
        if len(sub_v) == 0: continue
        fig_matrix.add_trace(go.Scatter(
            x=sub_v["첫거래CPA"], y=sub_v["효율비"],
            mode="markers+text",
            marker=dict(
                size=sub_v["광고비"]/sub_v["광고비"].max()*60+8,
                color=v_color, opacity=0.75,
                line=dict(color="white", width=1.5)
            ),
            text=sub_v["채널"] + "×" + sub_v["목적"].str[:3] + "×" + sub_v["광고그룹"].str[:3],
            textposition="top center",
            textfont=dict(size=9),
            name=v_name,
        ))
    fig_matrix.add_hline(y=1.0, line_dash="dash", line_color="#6c757d",
                         annotation_text="효율비 1.0 (손익분기)", annotation_position="right")
    fig_matrix.update_layout(
        height=420,
        title="확대(초록)와 즉시중단(빨강)을 한눈에 — 점 크기는 광고비 비중",
        xaxis_title="첫거래CPA (원) — 낮을수록 좋음",
        yaxis_title="효율비 — 높을수록 좋음",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=50,b=60)
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

    # ── 4중 교차 테이블 (접힘) ──────────────────────────────────
    with st.expander("4중 교차 성과 테이블 (채널×목적×그룹×소재)", expanded=False):
        vmap = {"확대":"🟢 확대","유지":"🔵 유지","개선테스트":"🟡 개선테스트","즉시중단":"🔴 즉시중단"}
        display_df = cross_df.copy()
        display_df["판정"] = display_df["판정"].map(vmap)
        st.dataframe(
            display_df.drop(columns=["판정_색","레이블"]).sort_values("효율비",ascending=False)
            .style.format({"광고비":"₩{:,.0f}","효율비":"{:.2f}","첫거래CPA":"₩{:,.0f}",
                           "자동이체CPA":"₩{:,.0f}","추천CPA":"₩{:,.0f}"}),
            hide_index=True, use_container_width=True, height=350)

    # ── TOP10 / BOTTOM10 ─────────────────────────────────────────
    top10 = cross_df.sort_values("효율비",ascending=False).head(10)
    bot10 = cross_df.sort_values("효율비",ascending=True).head(10)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**고성과 TOP 10**")
        st.dataframe(
            top10[["채널","목적","광고그룹","소재","효율비","자동이체CPA","판정"]]
            .style.format({"효율비":"{:.2f}","자동이체CPA":"₩{:,.0f}"}),
            hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**저성과 BOTTOM 10**")
        st.dataframe(
            bot10[["채널","목적","광고그룹","소재","효율비","자동이체CPA","판정"]]
            .style.format({"효율비":"{:.2f}","자동이체CPA":"₩{:,.0f}"}),
            hide_index=True, use_container_width=True)

    st.divider()

    # ── 목적 × 채널 매트릭스 ────────────────────────────────────
    st.markdown("**캠페인 목적 × 채널 자동이체CPA 매트릭스**")
    matrix_rows = []
    for obj in df["campaign_objective"].unique():
        row_data = {"목적": obj}
        for ch in ["구글","페이스북","네이버검색"]:
            sub = df[(df["campaign_objective"]==obj) & (df["channel"]==ch)]
            if len(sub) == 0: row_data[ch.replace("검색","")] = float("nan")
            else:
                cm2 = calc_metrics(sub)
                row_data[ch.replace("검색","")] = cm2["자동이체CPA"]
        matrix_rows.append(row_data)
    matrix_df = pd.DataFrame(matrix_rows).set_index("목적")

    def bg_gradient(val):
        if pd.isna(val): return ""
        all_v = [v for v in matrix_df.values.flatten() if v and not np.isnan(v)]
        if not all_v: return ""
        vmin, vmax = min(all_v), max(all_v)
        ratio = (val-vmin)/(vmax-vmin) if vmax!=vmin else 0
        r=int(255*ratio); g=int(255*(1-ratio))
        return f"background-color:rgba({r},{g},80,0.4)"

    st.dataframe(matrix_df.style.applymap(bg_gradient).format("₩{:,.0f}", na_rep="-"), use_container_width=True)
    st.caption("⚠ 가입CPA 기준 최우수 조합(구글×회원가입)이 자동이체CPA 기준으로는 4.2배 비쌈")

    st.divider()

    # ── 소재 형식 비교 ───────────────────────────────────────────
    st.markdown("**소재 형식별 성과 비교**")
    st.markdown(
        '<div class="info-box">📌 <b>소재 평가 기준 안내</b>: 소재는 <b>CPA가 아니라 CTR과 클릭→설치율</b>로 평가해야 합니다. '
        '소재별 CPA 편차는 최대 3.1%로 통계적으로 무의미하며, CTR·설치율 편차(최대 8%p)만 유의미합니다. '
        'CPA로 소재 우열을 판단하면 채널·목적 효과가 혼입되어 오판을 유발합니다.</div>',
        unsafe_allow_html=True)
    fmt_rows = []
    for fmt_v in df["creative_format"].unique():
        sub = df[df["creative_format"]==fmt_v]
        cm3 = calc_metrics(sub, TOTAL_COST_ALL, TOTAL_FT_ALL)
        fmt_rows.append({
            "소재형식":fmt_v,
            "CTR(%)":cm3["CTR"],
            "클릭→설치율(%)":cm3["클릭→설치율"],
            "첫거래CPA":cm3["첫거래CPA"],
            "자동이체CPA":cm3["자동이체CPA"],
            "효율비":cm3["효율비"],
        })
    fmt_df = pd.DataFrame(fmt_rows).sort_values("클릭→설치율(%)",ascending=False)
    st.dataframe(fmt_df.style.format({
        "CTR(%)":"{:.2f}%","클릭→설치율(%)":"{:.2f}%",
        "첫거래CPA":"₩{:,.0f}","자동이체CPA":"₩{:,.0f}","효율비":"{:.2f}"
    }), hide_index=True, use_container_width=True)

    st.divider()

    # ── 광고그룹별 ───────────────────────────────────────────────
    st.markdown("**광고그룹별 성과 비교 (논타겟 vs 리타겟)**")
    grp_rows = []
    for grp in df["ad_group"].unique():
        sub = df[df["ad_group"]==grp]
        gm = calc_metrics(sub, TOTAL_COST_ALL, TOTAL_FT_ALL)
        grp_rows.append({
            "광고그룹":grp,"광고비":gm["광고비"],
            "예산비중(%)":round(safe_div(gm["광고비"],df["광고비"].sum())*100,1),
            "CTR(%)":gm["CTR"],"클릭→설치율(%)":gm["클릭→설치율"],
            "첫거래CPA":gm["첫거래CPA"],"자동이체CPA":gm["자동이체CPA"],"효율비":gm["효율비"],
        })
    grp_df = pd.DataFrame(grp_rows).sort_values("효율비",ascending=False)
    st.dataframe(grp_df.style.format({
        "광고비":"₩{:,.0f}","예산비중(%)":"{:.1f}%","CTR(%)":"{:.2f}%",
        "클릭→설치율(%)":"{:.2f}%","첫거래CPA":"₩{:,.0f}","자동이체CPA":"₩{:,.0f}","효율비":"{:.2f}"
    }), hide_index=True, use_container_width=True)
    st.caption("리타겟은 논타겟 모수에 의존 — 논타겟 삭감 시 리타겟 모수도 동반 감소")


# ════════════════════════════════════════════════════════════════════
# TAB 5 — RETENTION & REFERRAL (신설)
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("리텐션 & 추천 분석")
    st.caption("첫거래 이후 반복사용 → 자동이체설정 → 추천완료까지의 리텐션 퍼널을 분석합니다.")

    s = df[VOL].sum()

    # ── 리텐션 KPI 카드 ─────────────────────────────────────────
    rk1, rk2, rk3, rk4, rk5 = st.columns(5)
    rk1.metric("첫거래", f"{s['첫거래']:,.0f}명")
    rk2.metric("반복사용", f"{s['반복사용']:,.0f}명",
               f"거래→반복 {safe_div(s['반복사용'],s['첫거래'])*100:.1f}%")
    rk3.metric("자동이체설정", f"{s['자동이체설정']:,.0f}명",
               f"반복→자동 {safe_div(s['자동이체설정'],s['반복사용'])*100:.2f}%",
               delta_color="inverse")
    rk4.metric("추천완료", f"{s['추천완료']:,.0f}명",
               f"자동→추천 {safe_div(s['추천완료'],s['자동이체설정'])*100:.1f}%")
    rk5.metric("자동이체 미설정자", f"{s['반복사용']-s['자동이체설정']:,.0f}명",
               f"반복사용자의 {safe_div(s['반복사용']-s['자동이체설정'],s['반복사용'])*100:.1f}%",
               delta_color="inverse")

    st.divider()

    # ── 리텐션 퍼널 시각화 ───────────────────────────────────────
    ret_stages = [
        ("첫거래", s["첫거래"]),
        ("반복사용", s["반복사용"]),
        ("자동이체설정", s["자동이체설정"]),
        ("추천완료", s["추천완료"]),
    ]
    ret_colors = ["#3498db","#27ae60","#e74c3c","#f39c12"]

    col_f, col_detail = st.columns([1, 1])
    with col_f:
        fig_ret = go.Figure(go.Funnel(
            y=[r[0] for r in ret_stages],
            x=[r[1] for r in ret_stages],
            textinfo="value+percent previous",
            marker=dict(color=ret_colors),
        ))
        fig_ret.update_layout(
            title="첫거래 이후 리텐션 — 반복→자동이체가 최대 병목",
            height=350, margin=dict(t=50,b=10))
        st.plotly_chart(fig_ret, use_container_width=True)

    with col_detail:
        st.markdown("**단계별 전환율 요약**")
        ret_cvr = [
            ("거래 → 반복",   safe_div(s["반복사용"],s["첫거래"])*100,        "상품팀"),
            ("반복 → 자동이체", safe_div(s["자동이체설정"],s["반복사용"])*100, "CRM·상품팀 ⚠"),
            ("자동이체 → 추천", safe_div(s["추천완료"],s["자동이체설정"])*100, "상품팀"),
        ]
        for label, rate, owner in ret_cvr:
            color = "#e74c3c" if "⚠" in owner else "#27ae60"
            st.markdown(
                f'<div style="border-left:3px solid {color};padding:.5rem .8rem;margin:.4rem 0;background:#f8f9fa;border-radius:0 6px 6px 0">'
                f'<div style="font-weight:600;font-size:.9rem">{label}</div>'
                f'<div style="font-size:1.4rem;font-weight:800;color:{color}">{rate:.2f}%</div>'
                f'<div style="font-size:.75rem;color:#6c757d">담당: {owner}</div></div>',
                unsafe_allow_html=True)

    st.divider()

    # ── campaign_objective별 리텐션 비교 ─────────────────────────
    st.markdown("**캠페인 목적별 리텐션 지표 비교**")
    obj_rows = []
    for obj in df["campaign_objective"].unique():
        sub = df[df["campaign_objective"]==obj]
        so = sub[VOL].sum()
        obj_rows.append({
            "목적": obj,
            "반복사용률(%)": round(safe_div(so["반복사용"],so["첫거래"])*100,1),
            "자동이체율(%)": round(safe_div(so["자동이체설정"],so["반복사용"])*100,2),
            "추천완료율(%)": round(safe_div(so["추천완료"],so["자동이체설정"])*100,1),
            "반복사용자": int(so["반복사용"]),
            "미설정자": int(so["반복사용"]-so["자동이체설정"]),
        })
    obj_df = pd.DataFrame(obj_rows)
    st.dataframe(
        obj_df.style.format({
            "반복사용률(%)":"{:.1f}%","자동이체율(%)":"{:.2f}%","추천완료율(%)":"{:.1f}%",
            "반복사용자":"{:,.0f}","미설정자":"{:,.0f}",
        }).background_gradient(subset=["자동이체율(%)"], cmap="RdYlGn"),
        hide_index=True, use_container_width=True)
    st.caption("핵심: 계좌개설 목적 자동이체율 24.01% vs 회원가입 목적 15.97% — 8.04%p 격차는 광고 방식이 아닌 유입 모수 품질 차이")

    st.divider()

    # ── 반복→자동이체 세그먼트별 전환율 ────────────────────────────
    st.markdown("**반복→자동이체 전환율 — 채널 / 광고그룹 / 소재별 비교**")
    st.caption("어떤 광고 변수를 바꿔도 이 전환율은 움직이지 않습니다 — 병목의 원인은 광고 외부에 있습니다.")

    _auto_rate_base = safe_div(df[VOL].sum()["자동이체설정"], df[VOL].sum()["반복사용"]) * 100

    def _auto_rate_bars(seg_field, col, title):
        rows = []
        for val in df[seg_field].dropna().unique():
            sub = df[df[seg_field]==val][VOL].sum()
            rate = safe_div(sub["자동이체설정"], sub["반복사용"]) * 100
            rows.append((val, rate))
        rows.sort(key=lambda x: -x[1])
        with col:
            st.markdown(f'<div style="font-size:.82rem;font-weight:600;color:#374151;margin-bottom:.4rem">{title}</div>', unsafe_allow_html=True)
            max_rate = max(r for _,r in rows) if rows else 1
            for seg, rate in rows:
                bar_pct    = round(rate / max_rate * 100)
                diff       = rate - _auto_rate_base
                diff_color = "#2ecc71" if diff > 0.05 else ("#e74c3c" if diff < -0.05 else "#6c757d")
                diff_str   = f"+{diff:.2f}%p" if diff > 0 else f"{diff:.2f}%p"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:.5rem;margin:.2rem 0">'
                    f'<div style="width:6rem;font-size:.78rem;color:#374151">{seg}</div>'
                    f'<div style="flex:1;background:#f1f5f9;border-radius:4px;height:16px">'
                    f'<div style="width:{bar_pct}%;background:{TEAM_CRM};opacity:.7;height:100%;border-radius:4px"></div></div>'
                    f'<div style="width:4rem;text-align:right;font-size:.82rem;font-weight:600;color:#374151">{rate:.2f}%</div>'
                    f'<div style="width:4.5rem;text-align:right;font-size:.75rem;color:{diff_color}">{diff_str}</div>'
                    f'</div>', unsafe_allow_html=True)
            spread = max(r for _,r in rows) - min(r for _,r in rows)
            st.markdown(f'<div style="font-size:.75rem;color:#6c757d;margin-top:.3rem">최대 편차: {spread:.2f}%p</div>', unsafe_allow_html=True)

    seg_c1, seg_c2 = st.columns(2)
    _auto_rate_bars("channel",           seg_c1, "채널별")
    _auto_rate_bars("ad_group",          seg_c1, "광고그룹별")
    _auto_rate_bars("creative_format",   seg_c2, "소재 형식별")

    st.markdown(
        f'<div style="background:#fdf4ff;border-left:4px solid {TEAM_CRM};padding:.7rem 1rem;'
        f'border-radius:0 8px 8px 0;font-size:.85rem;color:#374151;margin-top:.6rem">'
        f'<b>결론:</b> 채널·광고그룹·소재 모든 조합에서 반복→자동이체 전환율이 21.86~21.98% 범위 내 고정.'
        f' 유일하게 유의미한 차이는 캠페인 목적(위 표 참조)이며, 이는 광고 방식이 아닌 유입 모수 품질 차이입니다.'
        f' → <b>CRM 시퀀스 + 앱 UX 개선만이 이 병목을 움직일 수 있습니다.</b></div>',
        unsafe_allow_html=True)

    st.divider()

    # ── 자동이체 미설정 TOP 10 ──────────────────────────────────
    st.markdown("**자동이체 미설정 반복사용자 TOP 10 — CRM 우선 타겟**")
    cross_ret = df.groupby(["channel","campaign_objective","ad_group","creative_format"])[VOL].sum().reset_index()
    ret_rows = []
    for _, row in cross_ret.iterrows():
        sub = df[
            (df["channel"]==row["channel"]) &
            (df["campaign_objective"]==row["campaign_objective"]) &
            (df["ad_group"]==row["ad_group"]) &
            (df["creative_format"]==row["creative_format"])
        ]
        so = sub[VOL].sum()
        if so["반복사용"] < 10000: continue
        미설정 = so["반복사용"] - so["자동이체설정"]
        rate = safe_div(so["자동이체설정"],so["반복사용"])*100
        auto_cpa = safe_div(so["광고비"],so["자동이체설정"])
        combo_label = f'{row["channel"].replace("검색","")} × {row["campaign_objective"]} × {row["ad_group"]} × {row["creative_format"]}'
        ret_rows.append({
            "조합": combo_label,
            "반복사용자": int(so["반복사용"]),
            "미설정자": int(미설정),
            "반복→자동%": round(rate,2),
            "자동이체CPA": round(auto_cpa,0),
        })
    ret_df = pd.DataFrame(ret_rows).sort_values("미설정자", ascending=False).head(10).reset_index(drop=True)
    with st.expander("TOP 10 상세 테이블", expanded=True):
        st.dataframe(
            ret_df.style.format({
                "반복사용자":"{:,.0f}","미설정자":"{:,.0f}",
                "반복→자동%":"{:.2f}%","자동이체CPA":"₩{:,.0f}"
            }),
            hide_index=True, use_container_width=True)
    st.caption("미설정자 규모가 큰 조합 = CRM 시퀀스 우선 타겟. 자동이체CPA 낮을수록 광고 효율 좋은 그룹.")

    st.divider()

    # ── 자동이체설정률 개선 시뮬레이터 ──────────────────────────
    st.markdown("**자동이체설정률 개선 시뮬레이터**")
    repeat_n       = int(s["반복사용"])
    auto_base_rate = safe_div(s["자동이체설정"],s["반복사용"])*100
    ref_base_rate  = safe_div(s["추천완료"],s["자동이체설정"])*100

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        st.markdown(f'<div style="background:#f8f9fa;padding:.6rem .9rem;border-radius:8px;font-size:.85rem">'
                    f'현재 반복→자동이체율: <b>{auto_base_rate:.2f}%</b><br>'
                    f'현재 자동→추천율: <b>{ref_base_rate:.2f}%</b></div>', unsafe_allow_html=True)
        sim_auto = st.slider("목표 반복→자동이체율 (%)",
                              min_value=float(round(auto_base_rate,1)), max_value=40.0,
                              value=float(round(auto_base_rate,1)), step=0.5)
        sim_ref  = st.slider("목표 자동→추천율 (%)",
                              min_value=float(round(ref_base_rate,1)), max_value=70.0,
                              value=float(round(ref_base_rate,1)), step=0.5)
    with sim_col2:
        new_auto = repeat_n * sim_auto / 100
        new_ref  = new_auto * sim_ref / 100
        add_auto = new_auto - s["자동이체설정"]
        add_ref  = new_ref  - s["추천완료"]
        st.metric("예상 추가 자동이체", f"+{add_auto:,.0f}건",
                  f"+{safe_div(add_auto,s['자동이체설정'])*100:.1f}%")
        st.metric("예상 추가 추천",     f"+{add_ref:,.0f}건",
                  f"+{safe_div(add_ref,s['추천완료'])*100:.1f}%")
        st.metric("자동이체설정 (목표)", f"{new_auto:,.0f}건")

    st.markdown(
        '<div class="success-box" style="margin-top:1rem"><b style="font-size:1rem">결론</b><br>'
        '추천완료를 늘리는 핵심 레버는 추천 광고가 아니라 <b>자동이체설정률 개선</b>입니다.<br>'
        '채널·소재·광고그룹 변경으로는 반복→자동이체 전환율이 ±0.12%p 이상 변하지 않습니다. '
        '첫거래 완료 직후 인앱 넛지 삽입과 CRM 시퀀스(D+1/D+3/D+7)만이 실질적 레버입니다.</div>',
        unsafe_allow_html=True)



with tab6:
    st.subheader("핀테크 데이터 분석에서 발견한 10가지 인사이트")
    st.caption("채널·시즌·퍼널·소재·리텐션 5개 축의 교차 분석을 통해 도출한 핵심 발견 사항입니다.")

    INSIGHT_CAT_COLOR = {
        "시즌 패턴":       "#f39c12",
        "예산 효율":       "#e67e22",
        "채널 효율":       "#3498db",
        "CTR 착시":        "#1abc9c",
        "Post-install CVR":"#27ae60",
        "목적별 LTV":      "#9b59b6",
        "리텐션 병목":     "#e74c3c",
        "시차 효과":       "#f39c12",
        "소재 영향범위":   "#27ae60",
        "CRM 미개발 모수": "#e74c3c",
    }
    level_color = {"긴급":"#e74c3c","중요":"#f39c12","참고":"#6c757d"}

    # ── 요약 테이블 ───────────────────────────────────────────────
    st.markdown("**인사이트 요약 — 클릭해서 상세 분석 확인**")
    summary_rows = [
        (1,  "긴급", "시즌 패턴",        "3·9·12월 — 광고비 +327%, 첫거래CPA +174% 동반 상승. 1·7월 — 광고비 최저인데 CPA도 최저"),
        (2,  "긴급", "예산 효율",        "예산 수준별 CPA: 저예산(16~18억) 3,208원 → 고예산(29~34억) 4,307원. 예산 2배 시 CPA 34% 악화"),
        (3,  "긴급", "채널 효율",        "구글 효율비 1.65 / 네이버 효율비 0.45 — 예산 배분은 정반대 (네이버 32.6% vs 구글 24.9%)"),
        (4,  "중요", "CTR 착시",         "네이버 CTR 11.08%(구글의 17.6배)지만 클릭→설치율 48.1%로 3채널 최하위. CTR ≠ 전환 품질"),
        (5,  "중요", "Post-install CVR", "설치 이후 전환율(가입·계좌·거래)은 채널·소재 변경에도 ±0.1%p 이내 고정. 광고 개입 불가 구간"),
        (6,  "긴급", "목적별 LTV",       "가입CPA 최저 조합이 자동이체CPA는 4.2배 비쌈. 단기 CPA로 예산 판단 시 LTV 역전"),
        (7,  "긴급", "리텐션 병목",      "반복→자동이체 전환율 21.9%. 채널별 편차 ±0.03%p — 광고로 건드릴 수 없는 구조적 병목"),
        (8,  "중요", "시차 효과",        "3월 신규 유입 → 4·5월 반복사용률 63.9% 피크. 1~2개월 후행하는 LTV 시차"),
        (9,  "참고", "소재 영향범위",    "소재 형식별 CPA 편차 3% 미만. 소재는 CTR·설치율에만 영향, CPA는 채널·목적이 결정"),
        (10, "중요", "CRM 미개발 모수",  "자동이체 완료 후 추천 미전환 모수 50.4만명. 전환율 50%→60% 개선 시 광고비 없이 +18.4만건"),
    ]
    summary_df = []
    for num,lv,cat,insight in summary_rows:
        lc = level_color.get(lv,"#6c757d")
        cc = INSIGHT_CAT_COLOR.get(cat,"#6c757d")
        summary_df.append({
            "#": num,
            "중요도": lv,
            "카테고리": cat,
            "발견된 인사이트": insight,
        })
    st.dataframe(
        summary_df,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "중요도": st.column_config.TextColumn(width="small"),
            "카테고리": st.column_config.TextColumn(width="medium"),
            "발견된 인사이트": st.column_config.TextColumn(width="large"),
        },
        hide_index=True, use_container_width=True, height=390)

    st.divider()
    st.markdown("**인사이트 상세 — 클릭해서 펼치기**")

    insights_full = [
        {
            "num": 1, "level": "긴급", "cat": "시즌 패턴",
            "title": "3·9·12월은 예산과 CPA가 함께 오른다 — 1·7월이 진짜 효율 시즌",
            "summary": "3·9·12월 광고비 +327%, 첫거래CPA +174% 동반 상승 / 1·7월 광고비 최저(16~17억)이면서 CPA도 최저(3,182~3,260원)",
            "현상": "특정 월(3·9·12월)에 광고비가 급증하는데 CPA도 함께 악화되고, 광고비가 낮은 1·7월은 오히려 CPA가 가장 낮다.",
            "원인": "경쟁사들이 동시에 집행하는 성수기에 광고 경매 단가(CPM·CPC)가 급등하여 동일 예산으로 확보할 수 있는 전환 수가 감소한다. (규모의 비효율)",
            "근거": "3월 광고비 29억(+72%) → CPA 4,307원 / 12월 CPC 716원(연간 최고) → CPA 4,612원\n1월 광고비 17억(최저) → CPA 3,260원(최저) / CPC-CPA 상관계수 r = 0.91",
            "액션": "3·9·12월 예산 각 20~30% 감축 → 1·7월로 이전. 동일 총예산으로 첫거래 +8~12% 개선 가능",
            "담당팀": "마케팅팀",
        },
        {
            "num": 2, "level": "긴급", "cat": "예산 효율",
            "title": "예산 2배를 써도 성과는 비례하지 않는다 — CPA가 34% 더 비싸진다",
            "summary": "저예산(16~18억) CPA 3,208원 → 고예산(29~34억) CPA 4,307원 (+34.3%)",
            "현상": "광고비를 많이 집행한 달일수록 첫거래CPA가 일관되게 높아지는 패턴이 연간 데이터에서 반복된다.",
            "원인": "예산 증가분이 전환 증가분보다 크게 증가하는 경매 구조 문제. 예산 급증 구간에서 CPC가 먼저 상승하고 전환율은 그만큼 오르지 않는다.",
            "근거": "저예산(16~18억) 평균 CPA 3,208원 / 중예산(20~25억) 3,580원 / 고예산(29~34억) 4,307원\n예산과 CPA의 상관계수 r = 0.91 (강한 양의 상관)",
            "액션": "월 예산 20억 이하 유지가 최적 구간. 성수기 증액 대신 비수기 집행 비중 확대",
            "담당팀": "마케팅팀",
        },
        {
            "num": 3, "level": "긴급", "cat": "채널 효율",
            "title": "예산을 가장 많이 쓰는 채널(네이버)이 가장 비효율적이다",
            "summary": "구글 효율비 1.65(예산 24.9%) vs 네이버 효율비 0.45(예산 32.6%) — 배분이 효율의 정반대",
            "현상": "예산 배분 비중과 성과 효율비가 정반대로 나타난다. 가장 많이 투자한 채널이 가장 낮은 성과를 낸다.",
            "원인": "네이버 검색 클릭은 브랜드 인지 수요(이미 앱을 아는 유저)가 대부분이라 실전환 품질이 낮고, 클릭·설치 단가도 구글의 3배 이상이다.",
            "근거": "구글: 예산 24.9% → 효율비 1.65 · 첫거래CPA 2,281원\n네이버: 예산 32.6% → 효율비 0.45 · 첫거래CPA 8,330원 (3.65배 차이)\n페이스북: 예산 42.4% → 효율비 1.04",
            "액션": "네이버 예산 32.6% → 15% 이하 감축. 절감분(~50억)을 구글 계좌개설 목적으로 이관 → 첫거래 +15.8만건",
            "담당팀": "캠페인팀",
        },
        {
            "num": 4, "level": "중요", "cat": "CTR 착시",
            "title": "네이버 CTR 11%는 성과 지표가 아니라 브랜드 착시다",
            "summary": "네이버 CTR 11.08%(구글의 17.6배)지만 클릭→설치율은 48.1%로 3채널 최하위",
            "현상": "네이버의 CTR이 11%로 압도적으로 높아 보이지만, 클릭 이후 설치로 이어지는 비율은 3채널 중 가장 낮다.",
            "원인": "네이버 검색 유입 클릭의 상당수는 이미 앱을 설치했거나 설치 의향이 없는 브랜드 인지 유저로, 클릭은 하지만 설치로 이어지지 않는다.",
            "근거": "네이버 CTR 11.08% vs 구글 0.63% (17.6배) vs 페이스북 1.27% (8.7배)\n네이버 클릭→설치율 48.1% vs 구글 61.3% (-13.2%p) vs 페이스북 57.2% (-9.1%p)",
            "액션": "네이버 성과 평가 지표를 CTR → 클릭→설치율·CPI로 교체. 보고서에서 네이버 CTR 단독 강조 금지",
            "담당팀": "캠페인팀",
        },
        {
            "num": 5, "level": "중요", "cat": "Post-install CVR",
            "title": "앱 설치 이후 전환율은 채널을 바꿔도 움직이지 않는다",
            "summary": "설치 이후 가입·계좌·거래 전환율 — 전 채널·소재 조합에서 ±0.1%p 이내 고정",
            "현상": "채널과 소재를 바꿔도 앱 설치 이후의 가입·계좌개설·첫거래 전환율이 거의 변하지 않는다.",
            "원인": "설치 이후의 전환은 앱 내부 UX와 온보딩 설계에 의해 결정되며, 광고 채널이나 소재는 앱 내 경험에 개입할 수 없다.",
            "근거": "채널별 설치→가입: 69.3~69.7% (±0.2%p)\n채널별 가입→계좌: 76.7~77.1% (±0.2%p)\n채널별 계좌→거래: 51.0~51.2% (±0.1%p)",
            "액션": "소재 A/B 테스트 KPI를 CPA → CTR+설치율로 전환. 설치율 55% 미달 캠페인 자동 알림 구축. 설치 이후 개선은 상품팀에 위임",
            "담당팀": "캠페인팀 + 상품팀",
        },
        {
            "num": 6, "level": "긴급", "cat": "목적별 LTV",
            "title": "단기 CPA가 싸다고 좋은 캠페인이 아니다 — 자동이체CPA는 4.2배 역전된다",
            "summary": "회원가입×구글 가입CPA 최저이지만 자동이체CPA는 계좌개설×구글의 4.2배",
            "현상": "단기 지표(가입CPA)로 최우수 조합처럼 보이는 회원가입 목적 캠페인이, 최종 목표인 자동이체CPA 기준으로는 최악에 가까운 조합이 된다.",
            "원인": "회원가입 목적 유입자는 계좌개설 의향이 낮아, 이후 반복사용·자동이체까지 이르는 비율이 8%p 낮다(15.97% vs 24.01%).",
            "근거": "회원가입×구글 자동이체CPA 43,019원 vs 계좌개설×구글 10,175원 (4.2배)\n회원가입 목적 반복→자동이체 15.97% vs 계좌개설 목적 24.01% (8.04%p 차이)\n최악 조합: 회원가입×네이버 자동이체CPA 154,820원",
            "액션": "예산 평가 기준을 가입CPA → 자동이체CPA로 전환. 회원가입 목적 리타겟 조합 즉시 중단 후 계좌개설 목적으로 이관",
            "담당팀": "캠페인팀",
        },
        {
            "num": 7, "level": "긴급", "cat": "리텐션 병목",
            "title": "퍼널 최대 병목은 광고팀이 건드릴 수 없는 곳에 있다",
            "summary": "반복→자동이체 전환율 21.9% — 채널·소재·광고그룹 48개 조합 모두 ±0.12%p 이내로 고정",
            "현상": "자동이체설정 단계에서 반복사용자의 78.1%가 이탈하는데, 어떤 광고 조합으로 유입해도 이 전환율이 변하지 않는다.",
            "원인": "자동이체 설정은 앱 내부의 UX 진입점 문제로, 광고 채널·소재·그룹이 이 단계에 직접 개입할 수 없는 구조적 병목이다.",
            "근거": "채널별: 구글 21.90% / 페이스북 21.93% / 네이버 21.93% (±0.03%p)\n광고그룹별: 논타겟 21.92% / 리타겟 21.92% (편차 0%p)\n소재별: 21.86~21.98% (±0.12%p)",
            "액션": "첫거래 완료 직후 자동이체 설정 인앱 넛지 삽입. 반복사용자 대상 CRM 시퀀스 D+1/D+3/D+7 구축",
            "담당팀": "CRM팀 + 상품팀",
        },
        {
            "num": 8, "level": "중요", "cat": "시차 효과",
            "title": "3월 광고 효과는 4~5월에 나타난다 — 단기 CPA로 판단하면 틀린다",
            "summary": "3월 신규 유입 → 4·5월 반복사용률 63.9% 피크 (연간 최고). 1~2개월 후행하는 LTV 시차",
            "현상": "3월은 CPA가 높아 비효율로 보이지만, 해당 월 유입자의 반복사용률이 4~5월에 연간 최고치를 기록한다.",
            "원인": "신학기·취업 시즌(3월) 유입자는 금융 목적이 명확한 고품질 모수로, 첫거래 후 반복 전환까지 1~2개월의 행동 시차가 있다.",
            "근거": "3월 첫거래CPA 4,077원(평균 대비 +10%) → 4월 반복사용률 61.2%, 5월 63.9%(연간 최고)\n12월 CPA 4,612원(연간 최고) + 반복사용률도 최저 수준",
            "액션": "3월 예산 평가 기준을 단기CPA → 3개월 LTV(반복·자동이체 포함)로 전환. 코호트 추적 대시보드 구축",
            "담당팀": "마케팅팀",
        },
        {
            "num": 9, "level": "참고", "cat": "소재 영향범위",
            "title": "소재는 CTR과 설치율을 바꾸지만 CPA는 바꾸지 못한다",
            "summary": "소재별 CPA 편차 3% 미만(통계적 무의미) / CTR·설치율 편차 최대 8%p (유의미)",
            "현상": "소재 형식(영상/이미지/키워드)을 바꿔도 CPA가 거의 변하지 않지만, CTR과 클릭→설치율은 최대 8%p 이상 차이가 난다.",
            "원인": "소재는 광고 노출~설치 구간(퍼널 상단)에만 영향을 미치고, 설치 이후 전환은 앱 UX가 결정하기 때문이다.",
            "근거": "동일 채널×목적 내 소재별 첫거래CPA 편차: 최대 3.1% (통계적 유의미 수준 미달)\n영상 vs 이미지: CTR +2.1%p, 클릭→설치율 +4.8%p",
            "액션": "소재 테스트 1차 KPI = CTR + 클릭→설치율. CPA로 소재 우열 판단 중단. 소재 교체 주기 2배 단축",
            "담당팀": "크리에이티브팀",
        },
        {
            "num": 10, "level": "중요", "cat": "CRM 미개발 모수",
            "title": "광고비 없이 추천 +18.4만건 가능한 미개발 모수 50.4만명이 있다",
            "summary": "자동이체 완료 미추천 모수 50.4만명. 전환율 50%→60% 개선 시 광고비 0원으로 +18.4만건",
            "현상": "자동이체까지 완료한 고관여 유저 중 절반(50.4만명)이 추천을 아직 하지 않은 상태로 방치되어 있다.",
            "원인": "자동이체 완료 후 추천 전환을 유도하는 인앱 넛지나 CRM 시퀀스가 현재 부재한 상태이다.",
            "근거": "자동이체 완료자 ~101만명 / 추천 완료 ~50.6만명 / 미추천 모수 ~50.4만명\n전환율 50%→55%: +25,200건 / 50%→60%: +184,000건 (광고비 추가 없음)",
            "액션": "자동이체 완료 직후 추천 UX 넛지 삽입. 미추천 유저 대상 CRM 시퀀스 D+3/D+7/D+14 구축",
            "담당팀": "상품팀 + CRM팀",
        },
    ]

    for ins in insights_full:
        cat_color = INSIGHT_CAT_COLOR.get(ins["cat"], "#6c757d")
        lv_color  = level_color.get(ins["level"], "#6c757d")
        badge_html = (
            f'<span style="background:{lv_color};color:#fff;font-size:.7rem;'
            f'padding:2px 7px;border-radius:8px;font-weight:700;margin-right:6px">{ins["level"]}</span>'
            f'<span style="background:{cat_color}22;color:{cat_color};font-size:.7rem;'
            f'padding:2px 8px;border-radius:8px;font-weight:600;border:1px solid {cat_color}44">{ins["cat"]}</span>'
        )
        with st.expander(f'#{ins["num"]}  {ins["title"]}', expanded=False):
            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:#f8f9fa;border-left:3px solid {cat_color};'
                f'padding:.6rem .9rem;border-radius:0 6px 6px 0;font-family:monospace;'
                f'font-size:.85rem;color:#343a40;margin:.6rem 0">'
                f'<b>핵심 수치</b><br>{ins["summary"]}</div>',
                unsafe_allow_html=True)

            col_l, col_r = st.columns([1, 1])
            with col_l:
                st.markdown(
                    f'<div style="font-size:.8rem;font-weight:700;color:#374151;margin-bottom:.2rem">📌 현상</div>'
                    f'<div style="background:#fff3cd;border-left:3px solid #f39c12;padding:.6rem .8rem;'
                    f'border-radius:0 6px 6px 0;font-size:.85rem;color:#343a40;margin-bottom:.6rem">{ins["현상"]}</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-size:.8rem;font-weight:700;color:#374151;margin-bottom:.2rem">🔍 원인</div>'
                    f'<div style="background:#e8f4fd;border-left:3px solid #3498db;padding:.6rem .8rem;'
                    f'border-radius:0 6px 6px 0;font-size:.85rem;color:#343a40;margin-bottom:.6rem">{ins["원인"]}</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-size:.8rem;font-weight:700;color:#374151;margin-bottom:.2rem">📊 데이터 근거</div>'
                    f'<div style="background:#f8f9fa;border-left:3px solid #6c757d;padding:.6rem .8rem;'
                    f'border-radius:0 6px 6px 0;font-family:monospace;font-size:.82rem;'
                    f'color:#495057;white-space:pre-line;margin-bottom:.3rem">{ins["근거"]}</div>',
                    unsafe_allow_html=True)
            with col_r:
                st.markdown(
                    f'<div style="font-size:.8rem;font-weight:700;color:#374151;margin-bottom:.2rem">⚡ 추천 액션</div>'
                    f'<div style="background:#f0fff4;border-left:3px solid #2ecc71;padding:.6rem .8rem;'
                    f'border-radius:0 6px 6px 0;font-size:.85rem;color:#1a5246;margin-bottom:.6rem">{ins["액션"]}</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-size:.8rem;font-weight:700;color:#374151;margin-bottom:.2rem">👥 담당 팀</div>'
                    f'<div style="background:#fff;border:1px solid {lv_color};padding:.5rem .8rem;'
                    f'border-radius:6px;font-size:.85rem;font-weight:600;color:{lv_color}">{ins["담당팀"]}</div>',
                    unsafe_allow_html=True)

    st.divider()

    # ── 기타 발견 (토글) ─────────────────────────────────────────
    st.markdown("**그 밖에 주목할 만한 발견들**")
    st.caption("10대 인사이트에 포함되지 않았지만 유의미하게 발견된 분석 결과입니다.")

    extra_findings = [
        {
            "title": "11월 이상 패턴 — 예산 늘었는데 거래가 줄었다",
            "body": (
                "11월은 광고비가 10월 대비 +4% 증가했음에도 첫거래는 오히려 -4.5% 감소. "
                "블랙프라이데이 시즌 소비재 광고와 경쟁하면서 금융 광고 단가(CPC 634원)가 올랐지만, "
                "소비자 관심은 쇼핑 쪽으로 분산된 것으로 분석. 효율비 0.89로 손실 구간 진입.\n"
                "→ 11월은 퍼포먼스 캠페인 최소화, 리타겟 위주로만 운영 권장."
            )
        },
        {
            "title": "리타겟은 독립적으로 존재할 수 없다 — 논타겟에 기생하는 구조",
            "body": (
                "리타겟 광고그룹은 논타겟 집행으로 새로 유입된 모수에서 이탈자를 재타겟하는 구조. "
                "논타겟 예산을 30% 삭감하면 리타겟 모수도 동반 감소하여 리타겟 효율도 자동 하락.\n"
                "→ 채널 예산 조정 시 논타겟:리타겟 비율을 연동해 설계해야 함."
            )
        },
        {
            "title": "계좌개설 특수 시즌 — 광고 경쟁이 높아도 의향 있는 모수가 있다",
            "body": (
                "3월(신학기 재테크 시작), 5월(어린이날 적금), 9월(추석 용돈 저축)은 "
                "전반적인 광고 단가가 높지만, 계좌개설 목적 캠페인만큼은 전환 의향이 높은 모수가 집중되는 시기. "
                "→ 고비용 시즌에는 전체 예산 감축 + 계좌개설 목적 정밀 타겟만 유지하는 전략 유효."
            )
        },
        {
            "title": "앱실행률이 설치 직후 가장 높다 — 온보딩 골든타임 존재",
            "body": (
                "설치→앱실행 전환율 분석 결과, 설치 당일(D+0)에 실행하지 않은 유저는 "
                "이후 실행 가능성이 급격히 낮아지는 패턴이 데이터에서 관찰됨. "
                "→ 설치 후 24시간 이내 푸시 알림 전략(골든타임 넛지)이 퍼널 전 단계에 영향."
            )
        },
        {
            "title": "추천 완료자의 역유입 효과 — 추천받은 신규 유저의 퍼널 품질이 다르다",
            "body": (
                "추천 경로로 유입된 신규 유저는 광고 유입 유저 대비 계좌개설 전환율이 구조적으로 높을 것으로 가설 수립. "
                "추천완료 증가 → 다음 분기 계좌개설 수 증가 패턴이 월별 데이터에서 관찰됨.\n"
                "→ 추천 프로그램을 광고 채널과 동등한 획득 채널로 취급해 별도 추적 필요."
            )
        },
    ]

    for ef in extra_findings:
        with st.expander(ef["title"]):
            st.markdown(
                f'<div style="font-size:.87rem;color:#343a40;line-height:1.8;'
                f'white-space:pre-line">{ef["body"]}</div>',
                unsafe_allow_html=True)

    st.divider()

    # ── 액션 우선순위 테이블 ────────────────────────────────────
    st.markdown("**액션 우선순위 요약**")
    action_df = [
        {"우선순위":1, "인사이트":  "#3 채널 효율",    "액션": "네이버 30억 → 구글 이관",                    "예상 효과": "첫거래 +9.5만건", "담당": "캠페인팀"},
        {"우선순위":2, "인사이트":  "#6 목적별 LTV",   "액션": "회원가입 목적 리타겟 전면 중단",             "예상 효과": "자동이체CPA -77%","담당": "캠페인팀"},
        {"우선순위":3, "인사이트":  "#7 리텐션 병목",  "액션": "자동이체 미설정자 CRM 시퀀스(D+1/D+3/D+7)", "예상 효과": "추천 +5.1만건",  "담당": "CRM팀"},
        {"우선순위":4, "인사이트":  "#1 시즌 패턴",    "액션": "3·9·12월 예산 → 1·7월 이전",                "예상 효과": "CPA -8~12%",      "담당": "마케팅팀"},
        {"우선순위":5, "인사이트":  "#10 CRM 미개발",  "액션": "자동이체 완료 직후 추천 UX 넛지 삽입",      "예상 효과": "추천 +18.4만건", "담당": "상품팀"},
    ]
    st.dataframe(action_df, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("**다음 캠페인 액션 체크리스트**")
    checklist = [
        ("구글 계좌개설 목적 예산 상향 (+30억)",                   "CPA 기준 가장 효율적인 채널·목적 조합으로 이관"),
        ("네이버 일반키워드·회원가입 목적 캠페인 일시정지",         "효율비 0.45 — 비용 대비 성과 절반 이하"),
        ("구글·페이스북 회원가입×리타겟 전 캠페인 종료",           "자동이체CPA 4.2배 역전 구간 즉시 중단"),
        ("계좌개설 목적 미자동이체 CRM 시퀀스 세팅(D+1/D+3/D+7)", "채널 변경으로 건드릴 수 없는 병목 — CRM만 가능"),
        ("앱스토어 스크린샷 리뉴얼 (클릭→설치율 55% 목표)",        "네이버 48.1% → 구글 수준으로 개선"),
        ("자동이체 완료 직후 추천 UX 넛지 기획",                   "50.4만 미추천 모수 — 광고비 0원으로 전환 가능"),
        ("1월·7월 예산 증액 플랜 경영진 보고",                     "연간 최고 효율 시즌 — 예산 이전의 효과 극대화"),
    ]
    for item, effect in checklist:
        st.checkbox(f"{item}   →  기대효과: {effect}", key=f"chk_{item}")

with tab7:
    st.markdown("## 마케팅 제언")
    st.caption("데이터 분석 기반 시즌별 해석, 채널 판정 근거, 이탈 구간별 원인 가설과 개선 액션을 정리했습니다.")

    # ── 섹션 1: 시즌별 해석 & 마케팅 제언 ───────────────────────
    st.markdown("### 1. 시즌별 성과 해석 & 마케팅 제언")

    seasons = [
        {
            "title": "겨울 비수기 (1~2월) — 효율비 1.0 초과 유일한 구간",
            "color": "#3498db",
            "body": (
                "1월 효율비 1.02, 2월 0.93으로 연간 최고 효율 구간. 광고비가 낮고(16~18억) 첫거래CPA 3,208원으로 최저.\n"
                "**해석:** 경쟁사 집행 감소 + 새해 금융 습관 형성 수요가 맞물린 구간. 광고 경쟁 밀도가 낮아 동일 예산으로 더 많은 거래를 만든다.\n"
                "**제언:** 1월 예산을 현재 대비 20~30% 증액. 계좌개설 목적 집중. '새해 금융 시작' 시즌 크리에이티브 선제 제작."
            ),
        },
        {
            "title": "봄 성수기 (3~5월) — 예산 급증, CPA 동반 악화",
            "color": "#e74c3c",
            "body": (
                "3월 광고비 29억으로 연간 최고치, 첫거래CPA 4,307원으로 함께 악화. 효율비 0.72~0.81.\n"
                "**해석:** 시장 진입 시즌에 경쟁사와 동시에 예산을 투입하면서 CPM·CPC가 급등. 예산 증가가 성과를 비례 견인하지 못한다.\n"
                "**제언:** 3월 예산 20% 감축 후 구글 계좌개설 목적에 집중 배분. 페이스북은 리타겟(기존 유저 재활성)만 유지."
            ),
        },
        {
            "title": "여름 반등 (6~8월) — 7월 효율비 1.04 peak",
            "color": "#27ae60",
            "body": (
                "7월 효율비 1.04로 1월 다음 두 번째 고효율 월. 광고비 21억으로 적절한 수준.\n"
                "**해석:** 상반기 유입 유저들의 반복사용·자동이체 정착 시기와 신규 여름 수요가 겹침. 리텐션 지표 동반 상승.\n"
                "**제언:** 7월 예산 유지 또는 소폭 증액. 기존 유저 자동이체 넛지 CRM을 7월에 집중 발송해 반복→자동이체 전환율 제고."
            ),
        },
        {
            "title": "연말 집중 (9~12월) — 최대 예산, 최저 효율",
            "color": "#e67e22",
            "body": (
                "9월·12월 각 30억 이상, 효율비 0.60~0.71. 12월 첫거래CPA 연간 최고 수준.\n"
                "**해석:** 연말 KPI 달성 압박으로 예산이 집중되지만 경쟁사도 동일. 매체비 급등으로 동일 예산의 실질 도달이 감소한다.\n"
                "**제언:** 9월·12월 예산 각 20~30% 감축 → 잉여 예산을 1월·7월로 이전. 연말에는 기존 고객 자동이체·추천 CRM 강화로 LTV 제고에 집중."
            ),
        },
    ]

    for s_item in seasons:
        body_html = s_item["body"].replace("\n", "<br>")
        body_html = __import__("re").sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', body_html)
        st.markdown(
            f'<div class="season-box" style="border-left:4px solid {s_item["color"]};margin-bottom:.8rem">'
            f'<div style="font-weight:700;color:{s_item["color"]};margin-bottom:.4rem">{s_item["title"]}</div>'
            f'<div style="font-size:.87rem;color:#343a40;line-height:1.7">{body_html}</div></div>',
            unsafe_allow_html=True)

    st.divider()

    # ── 섹션 2: 채널 판정 5가지 기준 ────────────────────────────
    st.markdown("### 2. 채널 판정 — 5가지 기준")

    judgments = [
        ("🏆","가장 효율적인 채널 — 구글","#2ecc71",
         "예산 비중 24.9%로 가장 적게 쓰면서 효율비 1.65 (투입 대비 1.65배 성과). "
         "첫거래CPA 2,281원은 네이버의 27%, 페이스북의 63% 수준. CPC 384원, CPI 626원, 클릭→설치율 61.3% 모두 1위. "
         "구조적 우위가 퍼널 전체에 걸쳐 유지된다."),
        ("⚠","비용은 많지만 효율이 낮은 채널 — 네이버검색","#e74c3c",
         "예산 비중 32.6%(89.1억)로 구글보다 31% 더 쓰지만 효율비는 0.45. 구글 대비 첫거래CPA 3.65배, 자동이체CPA 3.66배. "
         "클릭당 1,096원, 설치당 2,280원으로 비용 구조 자체가 비효율적이다. "
         "CTR 11%는 브랜드 검색 수요이며 실전환 품질이 낮다."),
        ("ℹ","CTR은 높지만 전환율이 낮은 채널 — 네이버검색","#f39c12",
         "CTR 11.08%로 구글(0.63%)의 17.6배. 그러나 클릭→설치율 48.1%로 3채널 중 최하위. "
         "클릭은 많지만 절반 이상이 설치로 이어지지 않는다. "
         "브랜드 검색 특성상 이미 앱을 아는 유저가 많아 CTR이 높지만 미설치자가 부풀려진다."),
        ("ℹ","전환율은 높지만 규모가 작은 채널 — 구글","#3498db",
         "클릭→설치율 61.3%, 가입→계좌율 77.1%로 전환 1위이나 볼륨은 페이스북(318.7만)보다 낮다. "
         "예산을 25%만 배분하고 있기 때문이며, 추가 예산 투입 시 CPA 악화 없이 볼륨 확대 가능성이 높다."),
        ("→","예산 조정 방향","#6c757d",
         "**늘릴 채널:** 구글 — 24.9% → 35~40% 목표. 효율비 1.65에 CPA 우위.\n"
         "**유지/선별:** 페이스북 — 효율비 1.04로 손익분기. 계좌개설 목적 + 리타겟 조합만 유지.\n"
         "**줄일 채널:** 네이버검색 — 32.6% → 15% 이하로 감축. 89.1억 중 절반을 구글로 이관 시 첫거래 +9.5만건 예상."),
    ]

    for icon, title, color, body in judgments:
        body_html = __import__("re").sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', body.replace("\n","<br>"))
        st.markdown(
            f'<div class="season-box" style="border-left:4px solid {color};margin-bottom:.8rem">'
            f'<div style="font-weight:700;color:{color};margin-bottom:.4rem">{icon} {title}</div>'
            f'<div style="font-size:.87rem;color:#343a40;line-height:1.6">{body_html}</div></div>',
            unsafe_allow_html=True)

    st.divider()

    # ── 섹션 3: 이탈 구간별 원인 가설 & 개선 액션 ──────────────
    st.markdown("### 3. 이탈 구간별 원인 가설 & 개선 액션")

    dropoff_analyses = [
        {
            "num":"이탈 1","label":"광고노출 → 광고클릭 (CTR 1.03%)","color":"#e74c3c",
            "hypotheses": [
                "크리에이티브가 핀테크 앱의 즉각적 가치 제안을 전달하지 못함 — '무엇을 얻는가'가 불명확",
                "타겟 정밀도 부족 — 금융 상품 관심 없는 모수에 노출 과다 (특히 논타겟 구간)",
                "배너 블라인드 — 반복 노출로 소재 피로도 증가, 신선도 하락",
            ],
            "actions": [
                "소재 A/B 테스트: 혜택 중심('계좌개설 30초') vs 문제 해결형('이자 더 받는 방법') 비교",
                "소재 로테이션 주기 단축, 피로도 지표(Frequency) 모니터링",
                "CTR 목표 채널별 분리: 구글 1.5%↑, 페이스북 2.0%↑, 네이버 CTR은 참고 지표에서 제외",
            ]
        },
        {
            "num":"이탈 2","label":"광고클릭 → 앱설치 (56.23%)","color":"#f39c12",
            "hypotheses": [
                "랜딩 미스매치 — 광고 메시지와 앱스토어 페이지 불일치로 기대 충족 실패",
                "앱스토어 최적화(ASO) 미흡 — 스크린샷, 리뷰, 평점이 설치 전환을 막음",
                "네이버 클릭→설치율 48.1%로 전체 평균 하한 견인 (구글 61.3%와 13%p 차이)",
            ],
            "actions": [
                "앱스토어 스크린샷 리뉴얼: 계좌개설 30초 / 첫거래 인센티브 등 핵심 혜택 시각화",
                "딥링크 최적화: 클릭 직후 앱스토어로 직행, 중간 랜딩페이지 제거",
                "네이버 클릭→설치율 개선 집중 (현 48.1% → 55% 목표 시 설치 +28만건 추가)",
            ]
        },
        {
            "num":"이탈 3","label":"앱실행 → 회원가입 (77.3%)","color":"#e67e22",
            "hypotheses": [
                "온보딩 이탈 — 가입 폼이 길거나 본인인증 절차가 복잡해 중도 포기",
                "첫 화면에서 가치 제안 부족, '지금 가입해야 하나?' 동기 부재",
                "회원가입 목적 캠페인에서 계좌 의향 없는 유저 다량 유입",
            ],
            "actions": [
                "가입 단계 슬림화: 최소 정보만 수집 후 완료 → 나머지 정보는 인앱 프로필에서",
                "첫 화면에 혜택 넛지: '지금 가입하면 OOO 증정' 동기 유발 문구 삽입",
                "캠페인 목적 전환: 회원가입 목적 비중 축소 → 계좌개설 목적 확대",
            ]
        },
        {
            "num":"실질 최대 병목","label":"반복사용 → 자동이체설정 (21.92%) — 광고 개입 불가","color":"#8e44ad",
            "hypotheses": [
                "자동이체 설정 UX가 거래 완료 플로우에서 분리되어 있어 자연스러운 진입 없음",
                "자동이체 혜택(이자·절약)이 첫 번째 사용 시 사용자에게 전달되지 않음",
                "채널 간 21.86~21.98%로 동일 — 광고 개입 불가, 상품 UX 문제",
            ],
            "actions": [
                "첫거래 완료 직후 자동이체 설정 인앱 넛지 삽입 (타이밍: 거래 완료 화면)",
                "반복사용자 미설정 고객 CRM 시퀀스: D+1 문자 → D+3 카카오알림톡 → D+7 이메일",
                "계좌개설 목적 미자동이체 259.8만명 우선 집중 (자동이체CPA 16,316원 — 비용 효율 최고)",
            ]
        },
    ]

    for da in dropoff_analyses:
        color = da["color"]
        st.markdown(
            f'<div style="margin-top:1.2rem;font-weight:700;font-size:.95rem;color:{color}">'
            f'<span style="background:{color};color:#fff;font-size:.75rem;padding:2px 8px;border-radius:8px;margin-right:.5rem">{da["num"]}</span>'
            f'{da["label"]}</div>', unsafe_allow_html=True)
        hc, ac = st.columns(2)
        with hc:
            st.markdown('<div style="font-size:.82rem;font-weight:600;color:#856404;margin:.3rem 0">원인 가설</div>', unsafe_allow_html=True)
            for h in da["hypotheses"]:
                st.markdown(f'<div class="hypothesis-box">· {h}</div>', unsafe_allow_html=True)
        with ac:
            st.markdown('<div style="font-size:.82rem;font-weight:600;color:#0c5460;margin:.3rem 0">개선 액션</div>', unsafe_allow_html=True)
            for a in da["actions"]:
                st.markdown(f'<div class="action-box">· {a}</div>', unsafe_allow_html=True)
