"""Design System và CSS Styling hiện đại cho Streamlit Dashboard.

Cung cấp giao diện cao cấp (Glassmorphism, Vibrant Accents, Modern Typography,
Pill Badges, Metric Cards, và Micro-animations) tương thích cả Light và Dark mode.
"""

from __future__ import annotations

import streamlit as st

CSS_STYLES = """
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --primary-color: #4f46e5;
    --primary-light: #6366f1;
    --primary-dark: #3730a3;
    --accent-color: #06b6d4;
    --success-color: #10b981;
    --success-bg: rgba(16, 185, 129, 0.12);
    --warning-color: #f59e0b;
    --warning-bg: rgba(245, 158, 11, 0.12);
    --danger-color: #ef4444;
    --danger-bg: rgba(239, 68, 68, 0.12);
    --info-color: #3b82f6;
    --info-bg: rgba(59, 130, 246, 0.12);
    --card-bg: rgba(255, 255, 255, 0.85);
    --card-border: rgba(226, 232, 240, 0.8);
    --text-main: #0f172a;
    --text-muted: #64748b;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 18px;
    --shadow-soft: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.03);
}

/* Áp dụng font chữ toàn diện */
html, body, [class*="css"], .stMarkdown, .stButton, .stSelectbox, .stTextInput {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Ẩn header mặc định của Streamlit nhưng giữ thanh công cụ */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Main Container padding */
.main .block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3rem !important;
    max-width: 1350px !important;
}

/* Hero Banner Card */
.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    color: #ffffff !important;
    border-radius: var(--radius-lg);
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 30px -10px rgba(49, 46, 129, 0.35);
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.15);
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 380px;
    height: 380px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.3) 0%, rgba(6, 182, 212, 0) 70%);
    border-radius: 50%;
    pointer-events: none;
}

.hero-title {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em;
    margin-bottom: 0.5rem !important;
    color: #ffffff !important;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.hero-subtitle {
    font-size: 1.05rem !important;
    color: #e0e7ff !important;
    font-weight: 400;
    max-width: 800px;
    line-height: 1.6;
    margin: 0 !important;
}

/* KPI Metric Cards */
.kpi-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-md);
    padding: 1.4rem 1.6rem;
    box-shadow: var(--shadow-soft);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
}

.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 30px -5px rgba(0, 0, 0, 0.09);
    border-color: rgba(99, 102, 241, 0.4);
}

.kpi-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
}

.kpi-label {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
}

.kpi-icon {
    font-size: 1.4rem;
    padding: 0.5rem;
    border-radius: var(--radius-sm);
    display: inline-flex;
    align-items: center;
    justify-content: center;
}

.kpi-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--text-main);
    letter-spacing: -0.03em;
    line-height: 1.2;
}

.kpi-subtext {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* Biến thể màu cho KPI Card */
.kpi-primary .kpi-icon { background: rgba(79, 70, 229, 0.12); color: var(--primary-color); }
.kpi-success .kpi-icon { background: var(--success-bg); color: var(--success-color); }
.kpi-warning .kpi-icon { background: var(--warning-bg); color: var(--warning-color); }
.kpi-danger .kpi-icon { background: var(--danger-bg); color: var(--danger-color); }
.kpi-info .kpi-icon { background: var(--info-bg); color: var(--info-color); }

/* Badge Pill */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    line-height: 1;
}

.badge-success {
    background-color: #d1fae5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}

.badge-warning {
    background-color: #fef3c7;
    color: #92400e;
    border: 1px solid #fde68a;
}

.badge-danger {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #fecaca;
}

.badge-info {
    background-color: #e0f2fe;
    color: #075985;
    border: 1px solid #bae6fd;
}

.badge-neutral {
    background-color: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
}

/* Thẻ định danh sinh viên (ID Recognition Card) */
.student-id-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.9));
    border: 2px solid #10b981;
    border-radius: var(--radius-lg);
    padding: 1.8rem;
    box-shadow: 0 12px 32px rgba(16, 185, 129, 0.15);
    margin: 1.2rem 0;
    animation: fadeIn 0.4s ease-out;
}

.student-id-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 1rem;
    margin-bottom: 1.2rem;
}

.student-avatar-box {
    width: 68px;
    height: 68px;
    border-radius: 50%;
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    font-weight: 700;
    box-shadow: 0 6px 15px rgba(16, 185, 129, 0.35);
}

.student-meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.student-meta-item {
    background: rgba(241, 245, 249, 0.7);
    padding: 0.75rem 1rem;
    border-radius: var(--radius-sm);
    border: 1px solid #e2e8f0;
}

.student-meta-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #64748b;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.student-meta-val {
    font-size: 1rem;
    font-weight: 700;
    color: #1e293b;
    margin-top: 0.2rem;
}

/* Glassmorphism Section Container */
.glass-panel {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 1.8rem;
    margin-bottom: 1.8rem;
    box-shadow: var(--shadow-soft);
}

.section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.2rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--card-border);
}

.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-main);
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0;
}

/* Pulse Animation for Live Status */
@keyframes pulseDot {
    0% { transform: scale(0.95); opacity: 0.8; }
    50% { transform: scale(1.3); opacity: 1; }
    100% { transform: scale(0.95); opacity: 0.8; }
}

.live-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background-color: #10b981;
    display: inline-block;
    animation: pulseDot 2s infinite ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Nâng cấp style nút bấm */
div.stButton > button {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
}

div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.09) !important;
}

/* Tabs styling */
div[data-testid="stTabs"] button[role="tab"] {
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1.4rem !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
}

/* Custom Dataframe styling */
div[data-testid="stDataFrame"] {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--card-border) !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
    border-right: 1px solid #e2e8f0 !important;
}

.sidebar-brand {
    padding: 1.2rem 0.8rem 1.6rem 0.8rem;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1.4rem;
}

.sidebar-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: #1e1b4b;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0;
}

.sidebar-subtitle {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 0.35rem;
    line-height: 1.4;
}

.sidebar-status-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: var(--radius-sm);
    padding: 0.8rem 1rem;
    margin-top: 1.2rem;
    font-size: 0.82rem;
}
</style>
"""


def apply_custom_styles() -> None:
    """Nạp toàn bộ quy tắc CSS tùy biến vào Streamlit app."""
    st.markdown(CSS_STYLES, unsafe_allow_html=True)


def render_hero_banner(title: str, subtitle: str, badge_text: str = "Hệ thống v1.2") -> None:
    """Hiển thị banner tiêu đề nổi bật với hiệu ứng thị giác hiện đại."""
    html = f"""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div class="hero-title">
                    <span>🎓</span> {title}
                </div>
                <p class="hero-subtitle">{subtitle}</p>
            </div>
            <div class="badge-pill badge-info" style="background: rgba(255,255,255,0.2); color: white; border: 1px solid rgba(255,255,255,0.3);">
                <span class="live-dot"></span> {badge_text}
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_card(
    label: str,
    value: str | int,
    icon: str = "📊",
    subtext: str = "",
    variant: str = "primary",
) -> None:
    """Hiển thị thẻ KPI số liệu phong cách hiện đại."""
    html = f"""
    <div class="kpi-card kpi-{variant}">
        <div class="kpi-top">
            <span class="kpi-label">{label}</span>
            <span class="kpi-icon">{icon}</span>
        </div>
        <div class="kpi-value">{value}</div>
        {f'<div class="kpi-subtext">{subtext}</div>' if subtext else ""}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_student_recognition_card(
    student_code: str,
    full_name: str,
    class_name: str,
    status: str,
    distance: float,
    margin: float,
    check_in_time: str,
) -> None:
    """Hiển thị thẻ chứng nhận điểm danh sinh viên thành công."""
    is_present = status.lower() in {"present", "có mặt"}
    status_label = "CÓ MẶT" if is_present else "ĐI TRỄ"
    badge_cls = "badge-success" if is_present else "badge-warning"
    avatar_char = full_name.split()[-1][0].upper() if full_name else "S"

    html = f"""
    <div class="student-id-card">
        <div class="student-id-header">
            <div style="display: flex; align-items: center; gap: 1.2rem;">
                <div class="student-avatar-box">{avatar_char}</div>
                <div>
                    <h3 style="margin: 0; font-size: 1.35rem; font-weight: 800; color: #0f172a;">{full_name}</h3>
                    <div style="font-size: 0.95rem; color: #64748b; font-weight: 500;">MSSV: <strong style="color: #1e293b;">{student_code}</strong> · Lớp: {class_name}</div>
                </div>
            </div>
            <div class="badge-pill {badge_cls}" style="font-size: 0.95rem; padding: 0.5rem 1.2rem;">
                ✓ {status_label}
            </div>
        </div>
        <div class="student-meta-grid">
            <div class="student-meta-item">
                <div class="student-meta-label">Khoảng cách Euclidean L2</div>
                <div class="student-meta-val" style="color: #059669;">{distance:.4f} (≤ 0.50)</div>
            </div>
            <div class="student-meta-item">
                <div class="student-meta-label">Độ phân biệt (Margin)</div>
                <div class="student-meta-val" style="color: #2563eb;">Δ = {margin:.4f} (≥ 0.05)</div>
            </div>
            <div class="student-meta-item">
                <div class="student-meta-label">Thời gian ghi nhận</div>
                <div class="student-meta-val">{check_in_time}</div>
            </div>
            <div class="student-meta-item">
                <div class="student-meta-label">Thuật toán xác thực</div>
                <div class="student-meta-val" style="color: #7c3aed;">Open-Set Top-2 Mean</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
