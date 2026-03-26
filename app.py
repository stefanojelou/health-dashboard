import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

st.set_page_config(
    page_title="Salud Cuentas Dashboard",
    page_icon="📊",
    layout="wide"
)

# Password protection
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets.get("password", "jelou2024"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; min-height: 60vh;">
            <div style="text-align: center;">
                <h1 style="color: #00d4ff;">🔐 Salud Cuentas Dashboard</h1>
                <p style="color: #8892b0;">Enter password to access the dashboard</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input(
                "Password", type="password", on_change=password_entered, key="password"
            )
        return False
    
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; min-height: 60vh;">
            <div style="text-align: center;">
                <h1 style="color: #00d4ff;">🔐 Salud Cuentas Dashboard</h1>
                <p style="color: #8892b0;">Enter password to access the dashboard</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input(
                "Password", type="password", on_change=password_entered, key="password"
            )
            st.error("😕 Password incorrect")
        return False
    
    else:
        # Password correct
        return True

if not check_password():
    st.stop()

# Custom CSS for a modern dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #00d4ff !important;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1e1e3f, #2a2a4a);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 600;
        color: #00d4ff;
    }
    
    .metric-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.9rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .mom-positive {
        color: #00ff88;
        font-size: 1rem;
    }
    
    .mom-negative {
        color: #ff6b6b;
        font-size: 1rem;
    }
    
    .stSelectbox label, .stMultiSelect label {
        color: #ccd6f6 !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv('data/merged_weekly_data.csv')
    df['week_end_sunday'] = pd.to_datetime(df['week_end_sunday'])
    connect_plan_cols = [c for c in df.columns if c.startswith('connect_plan_') and c != 'connect_plan_list']
    kyc_metric_cols = [c for c in df.columns if c.startswith('kyc_')]
    hil_gov_cols = [c for c in df.columns if c.startswith('hil_') or c.startswith('gov_')]
    marketplace_cols = [c for c in df.columns if c.startswith('marketplace_app_')]

    numeric_cols = [
        'execution_count', 'dau_count', 'unique_users_count',
        'MARKETING', 'UTILITY', 'AUTHENTICATION', 'connect_licenses',
        'marketplace_install_total',
        'signups', 'self_service', 'enterprise', 'other_plans'
    ] + connect_plan_cols + kyc_metric_cols + hil_gov_cols + marketplace_cols
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if 'companyName' in df.columns:
        df['companyName'] = df['companyName'].fillna('Unknown')

    df = df.sort_values(['companyId', 'week_end_sunday'])

    company_metric_cols = [
        'execution_count', 'dau_count', 'unique_users_count',
        'MARKETING', 'UTILITY', 'AUTHENTICATION', 'connect_licenses',
        'marketplace_install_total',
    ]
    company_metric_cols += connect_plan_cols + kyc_metric_cols + hil_gov_cols + marketplace_cols
    company_metric_cols = [c for c in company_metric_cols if c in df.columns]

    return df, company_metric_cols


def format_plan_column_name(col_name):
    plan_text = re.sub(r'^connect_plan_', '', col_name)
    return plan_text.replace('_', ' ').strip().title()


def combine_plan_lists(series):
    plans = set()
    for value in series.dropna():
        for item in str(value).split(','):
            cleaned = item.strip()
            if cleaned:
                plans.add(cleaned)
    return ', '.join(sorted(plans))


def format_kyc_stage_label(col_name):
    stage_name = re.sub(r'^kyc_', '', col_name)
    stage_name = re.sub(r'_total$', '', stage_name)
    return stage_name.replace('_', ' ').strip().title()


def get_kyc_stage_total_cols(columns):
    preferred = ['kyc_document_check_total', 'kyc_liveness_total', 'kyc_facematch_total']
    present = [c for c in preferred if c in columns]
    if present:
        return present
    return sorted([
        c for c in columns
        if c.startswith(('kyc_document_check_', 'kyc_liveness_', 'kyc_facematch_'))
        and c.endswith('_total')
        and c != 'kyc_total'
    ])


def get_kyc_status_cols(columns):
    return sorted([
        c for c in columns
        if c.startswith(('kyc_document_check_', 'kyc_liveness_', 'kyc_facematch_'))
        and not c.endswith('_total')
        and c not in {'kyc_total'}
        and not c.endswith('_wow')
    ])


def get_marketplace_app_cols(columns):
    return sorted([c for c in columns if c.startswith('marketplace_app_') and not c.endswith('_wow')])


def format_marketplace_app_label(col_name):
    app_text = re.sub(r'^marketplace_app_', '', col_name)
    tokens = []
    token_map = {
        'api': 'API',
        'crm': 'CRM',
        'kyc': 'KYC',
        'mcp': 'MCP',
        'pro': 'PRO',
        'sap': 'SAP',
    }
    for token in app_text.split('_'):
        lower = token.lower()
        if lower in token_map:
            tokens.append(token_map[lower])
        elif lower == 's':
            tokens.append('S')
        elif lower == '4hana':
            tokens.append('4HANA')
        else:
            tokens.append(token.title())
    label = ' '.join(tokens)
    replacements = {
        'Hubspot': 'HubSpot',
        'Woocommerce': 'WooCommerce',
        'Wordpress.Com': 'WordPress.com',
        'Mercadopago': 'Mercado Pago',
        'Elevenlabs': 'ElevenLabs',
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def build_widget_key(prefix, value):
    cleaned = re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')
    return f"{prefix}_{cleaned}"


def render_checkbox_filter(options, key_prefix, title, columns=3):
    if not options:
        return []

    selected = []
    with st.expander(title, expanded=False):
        checkbox_cols = st.columns(columns)
        for i, option in enumerate(options):
            widget_key = build_widget_key(key_prefix, option)
            if widget_key not in st.session_state:
                st.session_state[widget_key] = True
            checked = checkbox_cols[i % columns].checkbox(option, key=widget_key)
            if checked:
                selected.append(option)
    return selected


def build_period_data(df, company_metric_cols, frequency):
    signup_cols = ['signups', 'self_service', 'enterprise', 'other_plans']
    available_signup_cols = [c for c in signup_cols if c in df.columns]
    has_plan_list = 'connect_plan_list' in df.columns

    if frequency == "Weekly":
        period_col = 'week_end_sunday'
        period_label = "Week (Sunday end)"

        df_period = df.copy()
        totals = df_period.groupby(period_col, as_index=False)[company_metric_cols].sum()

        if available_signup_cols:
            signup_weekly = (
                df_period[[period_col] + available_signup_cols]
                .drop_duplicates(subset=[period_col])
                .sort_values(period_col)
            )
            totals = totals.merge(signup_weekly, on=period_col, how='left')

        if has_plan_list:
            plan_weekly = (
                df_period[['companyId', period_col, 'connect_plan_list']]
                .groupby(['companyId', period_col], as_index=False)['connect_plan_list']
                .agg(combine_plan_lists)
            )
            df_period = df_period.drop(columns=['connect_plan_list'], errors='ignore').merge(
                plan_weekly, on=['companyId', period_col], how='left'
            )

        return df_period, totals, period_col, period_label

    # Monthly aggregation from weekly input
    period_col = 'month_start'
    period_label = "Month"

    df_month = df.copy()
    df_month[period_col] = df_month['week_end_sunday'].dt.to_period('M').dt.to_timestamp()

    agg_map = {metric: 'sum' for metric in company_metric_cols}
    if 'companyName' in df_month.columns:
        agg_map['companyName'] = 'first'
    if has_plan_list:
        agg_map['connect_plan_list'] = combine_plan_lists

    df_period = (
        df_month
        .groupby(['companyId', period_col], as_index=False)
        .agg(agg_map)
    )

    # Keep a consistent companyName fallback after aggregation
    if 'companyName' in df_period.columns:
        df_period['companyName'] = df_period['companyName'].fillna('Unknown')

    totals = df_period.groupby(period_col, as_index=False)[company_metric_cols].sum()

    if available_signup_cols:
        signup_monthly = (
            df_month[['week_end_sunday'] + available_signup_cols]
            .drop_duplicates(subset=['week_end_sunday'])
            .assign(month_start=lambda x: x['week_end_sunday'].dt.to_period('M').dt.to_timestamp())
            .groupby('month_start', as_index=False)[available_signup_cols]
            .sum()
        )
        totals = totals.merge(signup_monthly, on='month_start', how='left')

    return df_period, totals, period_col, period_label


def add_period_change_columns(df, metrics, period_col):
    """Add period-over-period change columns for each metric."""
    df_with_change = df.copy()
    df_with_change = df_with_change.sort_values(['companyId', period_col])
    for metric in metrics:
        if metric in df_with_change.columns:
            df_with_change[f'{metric}_wow'] = df_with_change.groupby('companyId')[metric].pct_change() * 100
    return df_with_change


# Load data
df_raw, company_metric_cols = load_data()

# Sidebar
st.sidebar.markdown("## 🎛️ Filters")
frequency = st.sidebar.selectbox("Frequency", ["Weekly", "Monthly"], index=0)

df, weekly_totals, period_col, period_label = build_period_data(df_raw, company_metric_cols, frequency)
df_with_wow = add_period_change_columns(df, company_metric_cols, period_col)

change_label = "WoW" if frequency == "Weekly" else "MoM"
period_short = "Week" if frequency == "Weekly" else "Month"

# Period filter
periods = sorted(weekly_totals[period_col].dropna().unique())
if not periods:
    st.error(
        f"No {frequency.lower()} data available. "
        "Run `python update_dashboard_data.py` to regenerate `data/merged_weekly_data.csv`."
    )
    st.stop()
selected_period = st.sidebar.selectbox(f"Select {period_label}", periods, index=len(periods)-1)
selected_period = pd.Timestamp(selected_period)
selected_period_label = selected_period.strftime('%Y-%m-%d') if frequency == "Weekly" else selected_period.strftime('%Y-%m')

# View selector
view = st.sidebar.radio("View", ["📈 Overview", "🔍 Company Lookup", "📊 Distributions", f"📉 {change_label} Analysis"])

# Main content
st.title("📊 Salud Cuentas Dashboard")

# Variable descriptions for sidebar
with st.sidebar.expander("📖 Glosario de Variables"):
    st.markdown("""
    **execution_count**  
    Número de ejecuciones semanales de workflows (Brain). Indica uso de automatizaciones.
    
    **MARKETING**  
    HSMs semanales de campañas de marketing (incluye MARKETING + MARKETING_LITE).
    
    **UTILITY**  
    HSMs semanales transaccionales/utility.

    **AUTHENTICATION**  
    HSMs semanales de autenticación.
    
    **dau_count**  
    Daily Active Users agregados semanalmente (suma de usuarios activos por día dentro de la semana).
    
    **unique_users_count**  
    Usuarios únicos que interactuaron en la semana.
    
    **connect_licenses**  
    Licencias Connect activas (ACTIVE/TRIALING) por semana.

    **marketplace_install_total / marketplace_app_***  
    Instalaciones de integraciones del marketplace por semana, total y desglose por app.

    **kyc_*_total / kyc_total**  
    Volumen de pasos KYC por etapa (`document_check`, `liveness`, `facematch`) y total de pasos KYC por periodo.

    **signups / other_plans**  
    Signups semanales y Nuevos SMBs (`other_plans`).

    **_wow**  
    Cambio porcentual vs periodo anterior ({change_label}).
    """)

if view == "📈 Overview":
    st.header(f"Overview - {period_short}: {selected_period_label}")
    
    st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); border-left: 4px solid #00d4ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <strong>📋 ¿Qué muestra esta vista?</strong><br>
        Resumen ejecutivo por periodo con métricas agregadas de todo el portafolio.<br><br>
        <strong>🎯 Métricas clave:</strong> Workflow Executions, DAU, Connect Licenses, WhatsApp por tipo, Sign Ups y Nuevos SMBs.<br><br>
        <strong>💡 Qué aprendes:</strong> Salud general del periodo y tendencia de crecimiento/decrecimiento.
    </div>
    """, unsafe_allow_html=True)

    week_row = weekly_totals[weekly_totals[period_col] == selected_period]
    if week_row.empty:
        st.warning(f"No data for selected {period_short.lower()}.")
        st.stop()
    week_row = week_row.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Workflow Executions", f"{week_row.get('execution_count', 0):,.0f}")
    with col2:
        st.metric(f"DAU ({frequency.lower()})", f"{week_row.get('dau_count', 0):,.0f}")
    with col3:
        st.metric("Connect Licenses", f"{week_row.get('connect_licenses', 0):,.0f}")
    with col4:
        active_companies = df[df[period_col] == selected_period]['companyId'].nunique()
        st.metric("Active Companies", f"{active_companies:,}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("WhatsApp Marketing", f"{week_row.get('MARKETING', 0):,.0f}")
    with col2:
        st.metric("WhatsApp Utility", f"{week_row.get('UTILITY', 0):,.0f}")
    with col3:
        st.metric("WhatsApp Authentication", f"{week_row.get('AUTHENTICATION', 0):,.0f}")
    with col4:
        st.metric("Sign Ups / Nuevos SMBs", f"{week_row.get('signups', 0):,.0f} / {week_row.get('other_plans', 0):,.0f}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(weekly_totals, x=period_col, y='execution_count',
                      title=f'Workflow Executions Over Time ({frequency})',
                      markers=True)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff'
        )
        fig.update_traces(line_color='#00d4ff', marker_color='#00ff88')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        wpp_cols = [c for c in ['MARKETING', 'UTILITY', 'AUTHENTICATION'] if c in weekly_totals.columns]
        colors = ['#00d4ff', '#ff6b6b', '#ffd93d']
        for i, col in enumerate(wpp_cols):
            fig.add_trace(go.Bar(
                x=weekly_totals[period_col],
                y=weekly_totals[col],
                name=col,
                marker_color=colors[i % len(colors)]
            ))
        fig.update_layout(
            title=f'WhatsApp Billable Count by Type ({frequency})',
            barmode='stack',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff'
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(weekly_totals, x=period_col, y='dau_count',
                      title=f'Daily Active Users Over Time ({frequency})',
                      markers=True)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff'
        )
        fig.update_traces(line_color='#ff6b6b', marker_color='#ffd93d')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(weekly_totals, x=period_col, y='connect_licenses',
                      title=f'Connect Licenses Over Time ({frequency})',
                      markers=True)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff'
        )
        fig.update_traces(line_color='#00ff88', marker_color='#00d4ff')
        st.plotly_chart(fig, use_container_width=True)

    marketplace_app_cols = [
        c for c in get_marketplace_app_cols(weekly_totals.columns)
        if weekly_totals[c].sum() > 0
    ]
    if marketplace_app_cols:
        st.subheader(f"🧩 Marketplace Installations by App ({frequency})")
        marketplace_app_map = {format_marketplace_app_label(col): col for col in marketplace_app_cols}
        selected_marketplace_apps = render_checkbox_filter(
            list(marketplace_app_map.keys()),
            key_prefix=f"overview_marketplace_app_filter_{frequency}",
            title="Apps to show"
        )
        selected_marketplace_cols = [marketplace_app_map[label] for label in selected_marketplace_apps]

        if selected_marketplace_cols:
            fig = go.Figure()
            marketplace_colors = ['#00d4ff', '#00ff88', '#ffd93d', '#ff6b6b', '#9b59b6', '#1abc9c', '#f39c12', '#e74c3c']
            for i, col in enumerate(selected_marketplace_cols):
                fig.add_trace(go.Bar(
                    x=weekly_totals[period_col],
                    y=weekly_totals[col],
                    name=format_marketplace_app_label(col),
                    marker_color=marketplace_colors[i % len(marketplace_colors)]
                ))
            fig.update_layout(
                title=f'Marketplace Installations by App ({frequency})',
                barmode='stack',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff',
                xaxis_title=period_short,
                yaxis_title='Installations'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one marketplace app to display the chart.")

    plan_cols = [c for c in weekly_totals.columns if c.startswith('connect_plan_') and c != 'connect_plan_list']
    if plan_cols:
        st.subheader(f"📦 Active Connect Plans by {period_short}")
        fig = go.Figure()
        plan_colors = ['#00d4ff', '#00ff88', '#ffd93d', '#ff6b6b', '#9b59b6', '#1abc9c']
        for i, col in enumerate(plan_cols):
            fig.add_trace(go.Bar(
                x=weekly_totals[period_col],
                y=weekly_totals[col],
                name=format_plan_column_name(col),
                marker_color=plan_colors[i % len(plan_colors)]
            ))
        fig.update_layout(
            title=f'Connect Active Plans Mix ({frequency})',
            barmode='stack',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff',
            xaxis_title=period_short,
            yaxis_title='Active licenses'
        )
        st.plotly_chart(fig, use_container_width=True)

    kyc_stage_cols = get_kyc_stage_total_cols(weekly_totals.columns)
    if kyc_stage_cols:
        st.subheader(f"🛂 KYC Steps by Stage ({frequency})")
        fig = go.Figure()
        kyc_colors = ['#00d4ff', '#00ff88', '#ffd93d', '#ff6b6b', '#9b59b6']
        for i, col in enumerate(kyc_stage_cols):
            fig.add_trace(go.Bar(
                x=weekly_totals[period_col],
                y=weekly_totals[col],
                name=format_kyc_stage_label(col),
                marker_color=kyc_colors[i % len(kyc_colors)]
            ))
        fig.update_layout(
            title=f'KYC Steps Volume by Stage ({frequency})',
            barmode='stack',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff',
            xaxis_title=period_short,
            yaxis_title='KYC steps'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"🏢 Top Companies This {period_short}")
    df_week = df[df[period_col] == selected_period]
    display_cols = [
        'companyId', 'companyName', 'execution_count',
        'MARKETING', 'UTILITY', 'AUTHENTICATION',
        'dau_count', 'connect_licenses', 'marketplace_install_total', 'connect_plan_list'
    ]
    display_cols = [c for c in display_cols if c in df_week.columns]
    top_companies = df_week.nlargest(10, 'execution_count')[display_cols]
    st.dataframe(top_companies, use_container_width=True, hide_index=True)

elif view == "🔍 Company Lookup":
    st.header("🔍 Company Lookup")
    
    # Page description
    st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); border-left: 4px solid #00d4ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <strong>📋 ¿Qué muestra esta vista?</strong><br>
        Deep-dive por periodo en una cuenta específica con toda su historia.<br><br>
        <strong>🎯 Métricas clave:</strong> Valores actuales de cada métrica, cambio {change_label} (%) y gráficos de evolución temporal.<br><br>
        <strong>💡 Qué aprendes:</strong> ¿Esta cuenta está creciendo o decayendo? ¿Qué features usa más (workflows vs campañas)? Preparación para llamadas de CS o renovaciones.
    </div>
    """, unsafe_allow_html=True)
    
    # Company selector
    companies = df[df['companyName'].notna()].drop_duplicates('companyId')[['companyId', 'companyName']].sort_values('companyName')
    company_options = {f"{row['companyName']} ({row['companyId']})": row['companyId'] for _, row in companies.iterrows()}
    
    selected_company_label = st.selectbox("Select Company", list(company_options.keys()))
    selected_company_id = company_options[selected_company_label]
    
    # Filter data for selected company
    df_company = df_with_wow[df_with_wow['companyId'] == selected_company_id].sort_values(period_col)
    
    if not df_company.empty:
        st.subheader(f"📋 Data for {selected_company_label}")
        
        latest = df_company.iloc[-1].copy()
        selected_company_period = df_company[df_company[period_col] == selected_period]
        selected_company_period = selected_company_period.iloc[0] if not selected_company_period.empty else latest
        
        metric_cols = [
            'execution_count', 'MARKETING', 'UTILITY', 'AUTHENTICATION',
            'dau_count', 'unique_users_count', 'connect_licenses', 'marketplace_install_total'
        ]
        metric_cols = [c for c in metric_cols if c in df_company.columns]
        
        cols = st.columns(len(metric_cols))
        
        for i, metric in enumerate(metric_cols):
            with cols[i]:
                val = latest[metric]
                wow_col = f'{metric}_wow'
                wow = latest[wow_col] if wow_col in latest.index else None
                delta = f"{change_label}: {wow:+.1f}%" if pd.notna(wow) else None
                label = metric.replace('_', ' ').title()
                st.metric(label, f"{val:,.0f}" if pd.notna(val) else "N/A", delta=delta)
        
        st.divider()

        plan_cols_company = [c for c in df_company.columns if c.startswith('connect_plan_') and c != 'connect_plan_list']
        if plan_cols_company:
            st.subheader(f"📦 Active Connect Plans ({period_short}: {selected_period_label})")
            active_plan_names = [
                format_plan_column_name(c)
                for c in plan_cols_company
                if selected_company_period.get(c, 0) > 0
            ]
            if active_plan_names:
                st.markdown("**Plans activos:** " + " | ".join(active_plan_names))
            else:
                st.info("No active Connect plans in selected period.")
        
        # Company time series
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(df_company, x=period_col, y='execution_count',
                        title=f'Workflow Executions by {period_short}')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff'
            )
            fig.update_traces(marker_color='#00d4ff')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            colors = ['#00d4ff', '#ff6b6b', '#ffd93d']
            for i, col in enumerate(['MARKETING', 'UTILITY', 'AUTHENTICATION']):
                if col in df_company.columns:
                    fig.add_trace(go.Bar(
                        x=df_company[period_col],
                        y=df_company[col],
                        name=col,
                        marker_color=colors[i % len(colors)]
                    ))
            fig.update_layout(
                title=f'WhatsApp Billable Count by Type ({frequency})',
                barmode='stack',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(df_company, x=period_col, y='dau_count',
                        title=f'DAU by {period_short}')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff'
            )
            fig.update_traces(marker_color='#ff6b6b')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(df_company, x=period_col, y='connect_licenses',
                        title=f'Connect Licenses by {period_short}')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff'
            )
            fig.update_traces(marker_color='#00ff88')
            st.plotly_chart(fig, use_container_width=True)

        marketplace_app_cols_company = [
            c for c in get_marketplace_app_cols(df_company.columns)
            if df_company[c].sum() > 0
        ]
        if marketplace_app_cols_company:
            st.subheader(f"🧩 Marketplace Installations ({period_short}: {selected_period_label})")
            marketplace_app_map_company = {
                format_marketplace_app_label(col): col for col in marketplace_app_cols_company
            }
            selected_company_marketplace_apps = render_checkbox_filter(
                list(marketplace_app_map_company.keys()),
                key_prefix=f"company_marketplace_app_filter_{selected_company_id}_{frequency}",
                title="Apps to show"
            )
            selected_company_marketplace_cols = [
                marketplace_app_map_company[label] for label in selected_company_marketplace_apps
            ]
            active_marketplace_apps = [
                format_marketplace_app_label(c)
                for c in marketplace_app_cols_company
                if selected_company_period.get(c, 0) > 0
            ]
            if active_marketplace_apps:
                st.markdown("**Apps con instalaciones en el periodo seleccionado:** " + " | ".join(active_marketplace_apps))
            else:
                st.info("No marketplace installations in selected period.")

            if selected_company_marketplace_cols:
                fig = go.Figure()
                marketplace_colors = ['#00d4ff', '#00ff88', '#ffd93d', '#ff6b6b', '#9b59b6', '#1abc9c', '#f39c12', '#e74c3c']
                for i, col in enumerate(selected_company_marketplace_cols):
                    fig.add_trace(go.Bar(
                        x=df_company[period_col],
                        y=df_company[col],
                        name=format_marketplace_app_label(col),
                        marker_color=marketplace_colors[i % len(marketplace_colors)]
                    ))
                fig.update_layout(
                    title=f'Marketplace Installations - {selected_company_label}',
                    barmode='stack',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#ccd6f6',
                    title_font_color='#00d4ff',
                    xaxis_title=period_short,
                    yaxis_title='Installations'
                )
                st.plotly_chart(fig, use_container_width=True)

                company_installation_rows = [
                    {
                        'App': format_marketplace_app_label(col),
                        'Installations': int(selected_company_period.get(col, 0))
                    }
                    for col in selected_company_marketplace_cols
                    if selected_company_period.get(col, 0) > 0
                ]
                if company_installation_rows:
                    st.dataframe(
                        pd.DataFrame(company_installation_rows).sort_values('Installations', ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.info("Select at least one marketplace app to display the company chart.")

        if plan_cols_company:
            st.subheader(f"📦 Connect Plan Mix by {period_short}")
            fig = go.Figure()
            plan_colors = ['#00d4ff', '#00ff88', '#ffd93d', '#ff6b6b', '#9b59b6', '#1abc9c']
            for i, col in enumerate(plan_cols_company):
                fig.add_trace(go.Bar(
                    x=df_company[period_col],
                    y=df_company[col],
                    name=format_plan_column_name(col),
                    marker_color=plan_colors[i % len(plan_colors)]
                ))
            fig.update_layout(
                title=f'Connect Active Plans - {selected_company_label}',
                barmode='stack',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff',
                xaxis_title=period_short,
                yaxis_title='Active licenses'
            )
            st.plotly_chart(fig, use_container_width=True)

        kyc_stage_cols_company = get_kyc_stage_total_cols(df_company.columns)
        if kyc_stage_cols_company:
            st.subheader(f"🛂 KYC by Stage ({period_short}: {selected_period_label})")

            kyc_metric_cards = list(kyc_stage_cols_company)
            if 'kyc_total' in df_company.columns:
                kyc_metric_cards.append('kyc_total')

            kyc_card_cols = st.columns(len(kyc_metric_cards))
            for i, metric in enumerate(kyc_metric_cards):
                with kyc_card_cols[i]:
                    value = selected_company_period.get(metric, 0)
                    wow_col = f'{metric}_wow'
                    wow = selected_company_period.get(wow_col) if wow_col in selected_company_period.index else None
                    delta = f"{change_label}: {wow:+.1f}%" if pd.notna(wow) else None
                    label = "KYC Total" if metric == 'kyc_total' else f"KYC {format_kyc_stage_label(metric)}"
                    st.metric(label, f"{value:,.0f}" if pd.notna(value) else "N/A", delta=delta)

            fig = go.Figure()
            kyc_colors = ['#00d4ff', '#00ff88', '#ffd93d', '#ff6b6b', '#9b59b6']
            for i, col in enumerate(kyc_stage_cols_company):
                fig.add_trace(go.Bar(
                    x=df_company[period_col],
                    y=df_company[col],
                    name=format_kyc_stage_label(col),
                    marker_color=kyc_colors[i % len(kyc_colors)]
                ))
            fig.update_layout(
                title=f'KYC Stage Volume - {selected_company_label}',
                barmode='stack',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff',
                xaxis_title=period_short,
                yaxis_title='KYC steps'
            )
            st.plotly_chart(fig, use_container_width=True)

            kyc_status_cols = get_kyc_status_cols(df_company.columns)
            if kyc_status_cols:
                status_rows = []
                for col in kyc_status_cols:
                    value = selected_company_period.get(col, 0)
                    if pd.notna(value) and value > 0:
                        status_rows.append({
                            'KYC Stage + Status': col.replace('kyc_', '').replace('_', ' ').title(),
                            'Count': int(value)
                        })
                if status_rows:
                    st.markdown("**KYC status detail (selected period)**")
                    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

        hil_gov_base_cols = [
            'kyc_sessions_total',
            'gov_validated',
            'gov_not_validated',
            'hil_doccheck_total',
            'hil_doccheck_approved',
            'hil_doccheck_disapproved',
            'hil_liveness_total',
            'hil_liveness_approved',
            'hil_liveness_disapproved',
            'hil_facematch_total',
            'hil_facematch_approved',
            'hil_facematch_disapproved',
        ]
        if all(col in df_company.columns for col in hil_gov_base_cols):
            st.subheader(f"🏛️ Gobierno + Human in the Loop ({period_short}: {selected_period_label})")

            hil_source_cols = [period_col] + hil_gov_base_cols
            if 'kyc_total' in df_company.columns:
                hil_source_cols.append('kyc_total')
            df_hil = df_company[hil_source_cols].copy().sort_values(period_col).reset_index(drop=True)
            for col in hil_gov_base_cols:
                df_hil[col] = pd.to_numeric(df_hil[col], errors='coerce').fillna(0)

            df_hil['gov_total_queries'] = df_hil['gov_validated'] + df_hil['gov_not_validated']
            df_hil['gov_validation_rate_pct'] = (
                df_hil['gov_validated'] / df_hil['gov_total_queries'].replace(0, pd.NA) * 100
            ).fillna(0)

            df_hil['hil_total_steps'] = (
                df_hil['hil_doccheck_total'] + df_hil['hil_liveness_total'] + df_hil['hil_facematch_total']
            )
            hil_denominator = df_hil['kyc_total'] if 'kyc_total' in df_hil.columns else df_hil['hil_total_steps']
            df_hil['hil_derivation_pct'] = (
                df_hil['hil_total_steps'] / hil_denominator.replace(0, pd.NA) * 100
            ).fillna(0)

            for step in ['doccheck', 'liveness', 'facematch']:
                total_col = f'hil_{step}_total'
                approved_col = f'hil_{step}_approved'
                disapproved_col = f'hil_{step}_disapproved'
                df_hil[f'hil_{step}_approval_rate_pct'] = (
                    df_hil[approved_col] / df_hil[total_col].replace(0, pd.NA) * 100
                ).fillna(0)
                df_hil[f'hil_{step}_disapproval_rate_pct'] = (
                    df_hil[disapproved_col] / df_hil[total_col].replace(0, pd.NA) * 100
                ).fillna(0)

            selected_rows = df_hil[df_hil[period_col] == selected_period]
            selected_idx = int(selected_rows.index[-1]) if not selected_rows.empty else len(df_hil) - 1
            current_hil = df_hil.iloc[selected_idx]
            prev_hil = df_hil.iloc[selected_idx - 1] if selected_idx > 0 else None

            def _delta_text(curr, prev, is_pct=False):
                if prev is None or pd.isna(prev):
                    return None
                delta = curr - prev
                return f"{delta:+.1f} pp" if is_pct else f"{delta:+,.0f}"

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(
                    "Consultas entidad (val/no val)",
                    f"{current_hil['gov_validated']:,.0f} / {current_hil['gov_not_validated']:,.0f}",
                )
            with c2:
                st.metric(
                    "% Validación entidad",
                    f"{current_hil['gov_validation_rate_pct']:.1f}%",
                    _delta_text(
                        current_hil['gov_validation_rate_pct'],
                        prev_hil['gov_validation_rate_pct'] if prev_hil is not None else None,
                        is_pct=True,
                    ),
                )
            with c3:
                st.metric(
                    "% Derivación HIL",
                    f"{current_hil['hil_derivation_pct']:.1f}%",
                    _delta_text(
                        current_hil['hil_derivation_pct'],
                        prev_hil['hil_derivation_pct'] if prev_hil is not None else None,
                        is_pct=True,
                    ),
                )
            with c4:
                st.metric(
                    "HIL total pasos",
                    f"{current_hil['hil_total_steps']:,.0f}",
                    _delta_text(
                        current_hil['hil_total_steps'],
                        prev_hil['hil_total_steps'] if prev_hil is not None else None,
                    ),
                )

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_hil[period_col], y=df_hil['gov_validated'],
                    name='Gov validated', marker_color='#00ff88'
                ))
                fig.add_trace(go.Bar(
                    x=df_hil[period_col], y=df_hil['gov_not_validated'],
                    name='Gov not validated', marker_color='#ff6b6b'
                ))
                fig.update_layout(
                    title=f'Consultas Entidad - {selected_company_label}',
                    barmode='stack',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#ccd6f6',
                    title_font_color='#00d4ff',
                    xaxis_title=period_short,
                    yaxis_title='Count'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_hil[period_col], y=df_hil['hil_doccheck_approval_rate_pct'],
                    mode='lines+markers', name='DocCheck approval %', line=dict(color='#00d4ff')
                ))
                fig.add_trace(go.Scatter(
                    x=df_hil[period_col], y=df_hil['hil_liveness_approval_rate_pct'],
                    mode='lines+markers', name='Liveness approval %', line=dict(color='#ffd93d')
                ))
                fig.add_trace(go.Scatter(
                    x=df_hil[period_col], y=df_hil['hil_facematch_approval_rate_pct'],
                    mode='lines+markers', name='Facematch approval %', line=dict(color='#9b59b6')
                ))
                fig.update_layout(
                    title=f'HIL Approval Rate by Step - {selected_company_label}',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#ccd6f6',
                    title_font_color='#00d4ff',
                    xaxis_title=period_short,
                    yaxis_title='Percent (%)'
                )
                st.plotly_chart(fig, use_container_width=True)

            step_rows = []
            for step_key, step_label in [
                ('doccheck', 'Document Check'),
                ('liveness', 'Liveness'),
                ('facematch', 'Facematch'),
            ]:
                total = current_hil[f'hil_{step_key}_total']
                approved = current_hil[f'hil_{step_key}_approved']
                disapproved = current_hil[f'hil_{step_key}_disapproved']
                step_rows.append({
                    'Step': step_label,
                    'HIL Total': int(total),
                    'Approved': int(approved),
                    'Disapproved': int(disapproved),
                    'Approval %': round(float(current_hil[f'hil_{step_key}_approval_rate_pct']), 2),
                    'Disapproval %': round(float(current_hil[f'hil_{step_key}_disapproval_rate_pct']), 2),
                })
            st.markdown("**HIL breakdown por etapa (selected period)**")
            st.dataframe(pd.DataFrame(step_rows), use_container_width=True, hide_index=True)
        
        st.subheader(f"📊 Full History with {change_label} Changes")
        display_cols = [period_col, 'execution_count', 'execution_count_wow']
        for col in ['MARKETING', 'UTILITY', 'AUTHENTICATION']:
            display_cols.extend([col, f'{col}_wow'])
        display_cols.extend([
            'dau_count', 'dau_count_wow',
            'unique_users_count', 'unique_users_count_wow',
            'connect_licenses', 'connect_licenses_wow',
            'connect_plan_list'
        ])
        kyc_history_cols = get_kyc_stage_total_cols(df_company.columns)
        if 'kyc_total' in df_company.columns:
            kyc_history_cols.append('kyc_total')
        for col in kyc_history_cols:
            display_cols.extend([col, f'{col}_wow'])
        for col in [
            'kyc_sessions_total',
            'gov_validated',
            'gov_not_validated',
            'hil_doccheck_total',
            'hil_doccheck_approved',
            'hil_doccheck_disapproved',
            'hil_liveness_total',
            'hil_liveness_approved',
            'hil_liveness_disapproved',
            'hil_facematch_total',
            'hil_facematch_approved',
            'hil_facematch_disapproved',
        ]:
            if col in df_company.columns:
                display_cols.extend([col, f'{col}_wow'])
        display_cols = [c for c in display_cols if c in df_company.columns]
        st.dataframe(df_company[display_cols].round(2), use_container_width=True, hide_index=True)
    else:
        st.warning("No data found for this company")

elif view == "📊 Distributions":
    st.header(f"📊 Distributions - {selected_period_label}")
    
    st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); border-left: 4px solid #00d4ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <strong>📋 ¿Qué muestra esta vista?</strong><br>
        Cómo se distribuyen las métricas entre todas las compañías.<br><br>
        <strong>🎯 Visualizaciones:</strong> Histograma (frecuencia), Box plot (outliers, mediana, cuartiles), Top 15 compañías.<br><br>
        <strong>💡 Qué aprendes:</strong> ¿Hay concentración? (pocas cuentas generan la mayoría del uso), ¿Quiénes son los power users?, ¿Cuál es el "típico" cliente? (mediana vs promedio).
    </div>
    """, unsafe_allow_html=True)
    
    df_week = df[df[period_col] == selected_period]
    
    available_metrics = [
        'execution_count', 'MARKETING', 'UTILITY',
        'AUTHENTICATION', 'dau_count', 'unique_users_count', 'connect_licenses'
    ]
    available_metrics = [m for m in available_metrics if m in df_week.columns]
    metric = st.selectbox("Select Metric", available_metrics)
    
    # Define thresholds for each metric to cap the distribution
    metric_thresholds = {
        'execution_count': 100000,
        'MARKETING': 100000,
        'UTILITY': 50000,
        'AUTHENTICATION': 50000,
        'dau_count': 10000,
        'unique_users_count': 5000,
        'connect_licenses': 200
    }
    threshold = metric_thresholds.get(metric, 100000)
    
    outliers_count = (df_week[metric] > threshold).sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        df_for_hist = df_week[df_week[metric].notna()].copy()
        df_normal = df_for_hist[df_for_hist[metric] <= threshold]
        
        fig = px.histogram(df_normal, x=metric, nbins=50,
                          title=f'Distribution of {metric} (0 to {threshold:,})')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff',
            xaxis=dict(range=[0, threshold])
        )
        fig.update_traces(marker_color='#00d4ff')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show outliers info
        if outliers_count > 0:
            st.info(f"📊 **{outliers_count} companies** have {metric} > {threshold:,} (not shown in histogram)")
    
    with col2:
        fig = px.box(df_week, y=metric,
                    title=f'Box Plot of {metric} (full range)')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff'
        )
        fig.update_traces(marker_color='#00ff88', line_color='#00d4ff')
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("📈 Statistics")
    stats = df_week[metric].describe()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mean", f"{stats['mean']:,.1f}" if pd.notna(stats['mean']) else "N/A")
    col2.metric("Median", f"{stats['50%']:,.1f}" if pd.notna(stats['50%']) else "N/A")
    col3.metric("Std Dev", f"{stats['std']:,.1f}" if pd.notna(stats['std']) else "N/A")
    col4.metric("Min", f"{stats['min']:,.0f}" if pd.notna(stats['min']) else "N/A")
    col5.metric("Max", f"{stats['max']:,.0f}" if pd.notna(stats['max']) else "N/A")
    
    # Top companies for this metric - with grouped outliers bar
    st.subheader(f"🏆 Top Companies by {metric}")
    
    # Get top 10 companies above threshold as "Power Users"
    top_outliers = df_week[df_week[metric] > threshold].nlargest(10, metric)
    top_normal = df_week[df_week[metric] <= threshold].nlargest(15, metric)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Power users (above threshold)
        st.markdown(f"**🚀 Power Users (>{threshold:,})**")
        if not top_outliers.empty:
            fig = px.bar(top_outliers, x='companyName', y=metric, 
                         title=f'Top {len(top_outliers)} Power Users')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff',
                xaxis_tickangle=-45
            )
            fig.update_traces(marker_color='#ff6b6b')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No companies with {metric} > {threshold:,}")
    
    with col2:
        # Normal range top companies
        st.markdown(f"**📊 Top Companies (≤{threshold:,})**")
        if not top_normal.empty:
            fig = px.bar(top_normal, x='companyName', y=metric, 
                         title=f'Top {len(top_normal)} in Normal Range')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff',
                xaxis_tickangle=-45
            )
            fig.update_traces(marker_color='#00d4ff')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No companies in normal range")

elif view == f"📉 {change_label} Analysis":
    st.header(f"📉 {change_label} Analysis")
    
    st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); border-left: 4px solid #00d4ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <strong>📋 ¿Qué muestra esta vista?</strong><br>
        Cambios entre periodos a nivel portafolio y por cuenta.<br><br>
        <strong>🎯 Visualizaciones:</strong> Evolución temporal del total, % cambio {change_label} (positivo/negativo), Top Gainers y Top Decliners.<br><br>
        <strong>💡 Qué aprendes:</strong> ¿Qué cuentas están en riesgo de churn? (decliners consistentes), ¿Qué cuentas están despegando? (gainers para upsell), Estacionalidad o patrones temporales.
    </div>
    """, unsafe_allow_html=True)
    
    available_metrics = [
        'execution_count', 'dau_count', 'unique_users_count',
        'MARKETING', 'UTILITY', 'AUTHENTICATION', 'connect_licenses',
        'signups', 'other_plans'
    ]
    available_metrics = [m for m in available_metrics if m in weekly_totals.columns]
    metric = st.selectbox("Select Metric", available_metrics)
    
    overall_wow = weekly_totals[[period_col, metric]].copy().sort_values(period_col)
    overall_wow['wow_change'] = overall_wow[metric].pct_change() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(overall_wow, x=period_col, y=metric,
                    title=f'Total {metric} by {period_short}')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff'
        )
        fig.update_traces(marker_color='#00d4ff')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(overall_wow, x=period_col, y='wow_change',
                    title=f'{change_label} Change (%) for {metric}',
                    color='wow_change',
                    color_continuous_scale=['#ff6b6b', '#ffd93d', '#00ff88'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader(f"📋 {change_label} Summary Table")
    st.dataframe(overall_wow.round(2), use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.subheader(f"🔥 Biggest {change_label} Changes in {selected_period_label}")
    
    df_week_wow = df_with_wow[df_with_wow[period_col] == selected_period].copy()
    wow_col = f'{metric}_wow'
    
    if wow_col in df_week_wow.columns:
        st.markdown(f"**🎯 Volume vs {change_label} Change (find big companies at risk)**")
        df_scatter = df_week_wow[df_week_wow[metric].notna() & df_week_wow[wow_col].notna()].copy()
        
        if not df_scatter.empty:
            df_scatter['status'] = df_scatter[wow_col].apply(lambda x: '📉 Declining' if x < 0 else '📈 Growing')
            
            fig = px.scatter(
                df_scatter,
                x=metric,
                y=wow_col,
                color='status',
                color_discrete_map={'📉 Declining': '#ff6b6b', '📈 Growing': '#00ff88'},
                hover_data=['companyName', 'companyId'],
                title=f'{metric} vs {change_label} Change - Identify Big Accounts at Risk',
                labels={metric: f'Total {metric}', wow_col: f'{change_label} Change (%)'}
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="#ffd93d", opacity=0.5)
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff'
            )
            fig.update_traces(marker=dict(size=10, opacity=0.7))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"**⚠️ Big Accounts Declining (High Volume + Negative {change_label})**")
            median_metric = df_scatter[metric].median()
            big_decliners = df_scatter[(df_scatter[metric] > median_metric) & (df_scatter[wow_col] < 0)].sort_values(wow_col)
            
            if not big_decliners.empty:
                st.dataframe(
                    big_decliners[['companyId', 'companyName', metric, wow_col]].head(15).round(2),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("No big accounts declining this week! 🎉")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Top Gainers**")
            top_gainers = df_week_wow.nlargest(10, wow_col)[['companyId', 'companyName', metric, wow_col]]
            st.dataframe(top_gainers.round(2), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**📉 Top Decliners**")
            top_decliners = df_week_wow.nsmallest(10, wow_col)[['companyId', 'companyName', metric, wow_col]]
            st.dataframe(top_decliners.round(2), use_container_width=True, hide_index=True)
    else:
        st.info(f"{change_label} data not available for this metric")

# Footer
st.sidebar.divider()
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit")
