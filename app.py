import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

    numeric_cols = [
        'execution_count', 'dau_count', 'unique_users_count',
        'MARKETING', 'UTILITY', 'AUTHENTICATION', 'connect_licenses',
        'signups', 'self_service', 'enterprise', 'other_plans'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if 'companyName' in df.columns:
        df['companyName'] = df['companyName'].fillna('Unknown')

    df = df.sort_values(['companyId', 'week_end_sunday'])

    company_metric_cols = [
        'execution_count', 'dau_count', 'unique_users_count',
        'MARKETING', 'UTILITY', 'AUTHENTICATION', 'connect_licenses'
    ]
    company_metric_cols = [c for c in company_metric_cols if c in df.columns]

    weekly_totals = df.groupby('week_end_sunday', as_index=False)[company_metric_cols].sum()
    if {'week_end_sunday', 'signups', 'self_service', 'enterprise', 'other_plans'}.issubset(df.columns):
        signup_weekly = (
            df[['week_end_sunday', 'signups', 'self_service', 'enterprise', 'other_plans']]
            .drop_duplicates(subset=['week_end_sunday'])
            .sort_values('week_end_sunday')
        )
        weekly_totals = weekly_totals.merge(signup_weekly, on='week_end_sunday', how='left')

    return df, weekly_totals, company_metric_cols


def add_wow_columns(df, metrics):
    """Add WoW change columns for each metric."""
    df_with_wow = df.copy()
    for metric in metrics:
        if metric in df_with_wow.columns:
            df_with_wow[f'{metric}_wow'] = df_with_wow.groupby('companyId')[metric].pct_change() * 100
    return df_with_wow


# Load data
df, weekly_totals, company_metric_cols = load_data()
df_with_wow = add_wow_columns(df, company_metric_cols)

# Sidebar
st.sidebar.markdown("## 🎛️ Filters")

# Week filter
weeks = sorted(weekly_totals['week_end_sunday'].dropna().unique())
if not weeks:
    st.error("No weekly data available. Refresh data exports and notebook output.")
    st.stop()
selected_week = st.sidebar.selectbox("Select Week (Sunday end)", weeks, index=len(weeks)-1)
selected_week = pd.Timestamp(selected_week)
selected_week_label = selected_week.strftime('%Y-%m-%d')

# View selector
view = st.sidebar.radio("View", ["📈 Overview", "🔍 Company Lookup", "📊 Distributions", "📉 WoW Analysis"])

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

    **signups / other_plans**  
    Signups semanales y Nuevos SMBs (`other_plans`).

    **_wow**  
    Week-over-Week change (%) - cambio porcentual vs semana anterior.
    """)

if view == "📈 Overview":
    st.header(f"Overview - Week ending {selected_week_label}")
    
    st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); border-left: 4px solid #00d4ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <strong>📋 ¿Qué muestra esta vista?</strong><br>
        Resumen ejecutivo semanal con métricas agregadas de todo el portafolio.<br><br>
        <strong>🎯 Métricas clave:</strong> Workflow Executions, DAU, Connect Licenses, WhatsApp por tipo, Sign Ups y Nuevos SMBs.<br><br>
        <strong>💡 Qué aprendes:</strong> Salud general semanal y tendencia de crecimiento/decrecimiento.
    </div>
    """, unsafe_allow_html=True)

    week_row = weekly_totals[weekly_totals['week_end_sunday'] == selected_week]
    if week_row.empty:
        st.warning("No data for selected week.")
        st.stop()
    week_row = week_row.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Workflow Executions", f"{week_row.get('execution_count', 0):,.0f}")
    with col2:
        st.metric("DAU (weekly)", f"{week_row.get('dau_count', 0):,.0f}")
    with col3:
        st.metric("Connect Licenses", f"{week_row.get('connect_licenses', 0):,.0f}")
    with col4:
        active_companies = df[df['week_end_sunday'] == selected_week]['companyId'].nunique()
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
        fig = px.line(weekly_totals, x='week_end_sunday', y='execution_count',
                      title='Workflow Executions Over Time (Weekly)',
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
                x=weekly_totals['week_end_sunday'],
                y=weekly_totals[col],
                name=col,
                marker_color=colors[i % len(colors)]
            ))
        fig.update_layout(
            title='WhatsApp Billable Count by Type (Weekly)',
            barmode='stack',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff'
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(weekly_totals, x='week_end_sunday', y='dau_count',
                      title='Daily Active Users Over Time (Weekly)',
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
        fig = px.line(weekly_totals, x='week_end_sunday', y='connect_licenses',
                      title='Connect Licenses Over Time (Weekly)',
                      markers=True)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff'
        )
        fig.update_traces(line_color='#00ff88', marker_color='#00d4ff')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏢 Top Companies This Week")
    df_week = df[df['week_end_sunday'] == selected_week]
    display_cols = [
        'companyId', 'companyName', 'execution_count',
        'MARKETING', 'UTILITY', 'AUTHENTICATION',
        'dau_count', 'connect_licenses'
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
        Deep-dive semanal en una cuenta específica con toda su historia.<br><br>
        <strong>🎯 Métricas clave:</strong> Valores actuales de cada métrica, cambio WoW (%) y gráficos de evolución semanal.<br><br>
        <strong>💡 Qué aprendes:</strong> ¿Esta cuenta está creciendo o decayendo? ¿Qué features usa más (workflows vs campañas)? Preparación para llamadas de CS o renovaciones.
    </div>
    """, unsafe_allow_html=True)
    
    # Company selector
    companies = df[df['companyName'].notna()].drop_duplicates('companyId')[['companyId', 'companyName']].sort_values('companyName')
    company_options = {f"{row['companyName']} ({row['companyId']})": row['companyId'] for _, row in companies.iterrows()}
    
    selected_company_label = st.selectbox("Select Company", list(company_options.keys()))
    selected_company_id = company_options[selected_company_label]
    
    # Filter data for selected company
    df_company = df_with_wow[df_with_wow['companyId'] == selected_company_id].sort_values('week_end_sunday')
    
    if not df_company.empty:
        st.subheader(f"📋 Data for {selected_company_label}")
        
        latest = df_company.iloc[-1].copy()
        
        metric_cols = [
            'execution_count', 'MARKETING', 'UTILITY', 'AUTHENTICATION',
            'dau_count', 'unique_users_count', 'connect_licenses'
        ]
        metric_cols = [c for c in metric_cols if c in df_company.columns]
        
        cols = st.columns(len(metric_cols))
        
        for i, metric in enumerate(metric_cols):
            with cols[i]:
                val = latest[metric]
                wow_col = f'{metric}_wow'
                wow = latest[wow_col] if wow_col in latest.index else None
                delta = f"{wow:+.1f}%" if pd.notna(wow) else None
                label = metric.replace('_', ' ').title()
                st.metric(label, f"{val:,.0f}" if pd.notna(val) else "N/A", delta=delta)
        
        st.divider()
        
        # Company time series
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(df_company, x='week_end_sunday', y='execution_count',
                        title='Workflow Executions by Week')
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
                        x=df_company['week_end_sunday'],
                        y=df_company[col],
                        name=col,
                        marker_color=colors[i % len(colors)]
                    ))
            fig.update_layout(
                title='WhatsApp Billable Count by Type (Weekly)',
                barmode='stack',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(df_company, x='week_end_sunday', y='dau_count',
                        title='DAU by Week')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff'
            )
            fig.update_traces(marker_color='#ff6b6b')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(df_company, x='week_end_sunday', y='connect_licenses',
                        title='Connect Licenses by Week')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff'
            )
            fig.update_traces(marker_color='#00ff88')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Full History with WoW Changes")
        display_cols = ['week_end_sunday', 'execution_count', 'execution_count_wow']
        for col in ['MARKETING', 'UTILITY', 'AUTHENTICATION']:
            display_cols.extend([col, f'{col}_wow'])
        display_cols.extend([
            'dau_count', 'dau_count_wow',
            'unique_users_count', 'unique_users_count_wow',
            'connect_licenses', 'connect_licenses_wow'
        ])
        display_cols = [c for c in display_cols if c in df_company.columns]
        st.dataframe(df_company[display_cols].round(2), use_container_width=True, hide_index=True)
    else:
        st.warning("No data found for this company")

elif view == "📊 Distributions":
    st.header(f"📊 Distributions - {selected_week_label}")
    
    st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); border-left: 4px solid #00d4ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <strong>📋 ¿Qué muestra esta vista?</strong><br>
        Cómo se distribuyen las métricas entre todas las compañías.<br><br>
        <strong>🎯 Visualizaciones:</strong> Histograma (frecuencia), Box plot (outliers, mediana, cuartiles), Top 15 compañías.<br><br>
        <strong>💡 Qué aprendes:</strong> ¿Hay concentración? (pocas cuentas generan la mayoría del uso), ¿Quiénes son los power users?, ¿Cuál es el "típico" cliente? (mediana vs promedio).
    </div>
    """, unsafe_allow_html=True)
    
    df_week = df[df['week_end_sunday'] == selected_week]
    
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

elif view == "📉 WoW Analysis":
    st.header("📉 Week-over-Week Analysis")
    
    st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); border-left: 4px solid #00d4ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <strong>📋 ¿Qué muestra esta vista?</strong><br>
        Cambios semana a semana a nivel portafolio y por cuenta.<br><br>
        <strong>🎯 Visualizaciones:</strong> Evolución semanal del total, % cambio WoW (positivo/negativo), Top Gainers y Top Decliners.<br><br>
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
    
    overall_wow = weekly_totals[['week_end_sunday', metric]].copy().sort_values('week_end_sunday')
    overall_wow['wow_change'] = overall_wow[metric].pct_change() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(overall_wow, x='week_end_sunday', y=metric,
                    title=f'Total {metric} by Week')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff'
        )
        fig.update_traces(marker_color='#00d4ff')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(overall_wow, x='week_end_sunday', y='wow_change',
                    title=f'WoW Change (%) for {metric}',
                    color='wow_change',
                    color_continuous_scale=['#ff6b6b', '#ffd93d', '#00ff88'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 WoW Summary Table")
    st.dataframe(overall_wow.round(2), use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.subheader(f"🔥 Biggest WoW Changes in {selected_week_label}")
    
    df_week_wow = df_with_wow[df_with_wow['week_end_sunday'] == selected_week].copy()
    wow_col = f'{metric}_wow'
    
    if wow_col in df_week_wow.columns:
        st.markdown("**🎯 Volume vs WoW Change (find big companies at risk)**")
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
                title=f'{metric} vs WoW Change - Identify Big Accounts at Risk',
                labels={metric: f'Total {metric}', wow_col: 'WoW Change (%)'}
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
            
            st.markdown("**⚠️ Big Accounts Declining (High Volume + Negative WoW)**")
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
        st.info("WoW data not available for this metric")

# Footer
st.sidebar.divider()
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit")
