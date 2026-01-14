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
    # Load individual files and merge
    df_billing = pd.read_csv('data/session_billing_events.csv')
    df_dau = pd.read_csv('data/logsM.ai_daily_active_users.csv')
    df_workflow = pd.read_csv('data/builder.workflow_executions_logs.csv')
    
    # Combine MARKETING and MARKETING_LITE into a single MARKETING category
    df_billing['conversationType'] = df_billing['conversationType'].replace({
        'MARKETING_LITE': 'MARKETING',
    })
    
    # Pivot billing data to have conversation types as separate columns
    df_billing_pivot = df_billing.pivot_table(
        index=['companyId', 'month'],
        columns='conversationType',
        values='billable_count',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    df_billing_pivot.columns.name = None
    
    # Get the conversation type columns dynamically
    conv_type_cols = [col for col in df_billing_pivot.columns if col not in ['companyId', 'month']]
    
    # Merge all
    df = df_workflow.merge(df_billing_pivot, on=['companyId', 'month'], how='outer')
    df = df.merge(df_dau, on=['companyId', 'month'], how='outer')
    
    # Sort by month
    df = df.sort_values(['companyId', 'month'])
    
    return df, df_billing, conv_type_cols


def add_mom_columns(df, conv_type_cols):
    """Add MoM change columns for each metric"""
    metrics = ['execution_count', 'dau_count', 'unique_users_count'] + conv_type_cols
    df_with_mom = df.copy()
    
    for metric in metrics:
        if metric in df_with_mom.columns:
            df_with_mom[f'{metric}_mom'] = df_with_mom.groupby('companyId')[metric].pct_change() * 100
    
    return df_with_mom


# Load data
df, df_billing_raw, conv_type_cols = load_data()
df_with_mom = add_mom_columns(df, conv_type_cols)

# Sidebar
st.sidebar.markdown("## 🎛️ Filters")

# Month filter
months = sorted(df['month'].dropna().unique())
selected_month = st.sidebar.selectbox("Select Month", months, index=len(months)-1 if months else 0)

# View selector
view = st.sidebar.radio("View", ["📈 Overview", "🔍 Company Lookup", "📊 Distributions", "📉 MoM Analysis"])

# Main content
st.title("📊 Salud Cuentas Dashboard")

# Variable descriptions for sidebar
with st.sidebar.expander("📖 Glosario de Variables"):
    st.markdown("""
    **execution_count**  
    Número de ejecuciones de workflows (Brain). Indica uso de automatizaciones.
    
    **MARKETING**  
    HSMs de campañas de marketing (incluye MARKETING y MARKETING_LITE - plantillas promocionales de WhatsApp).
    
    **UTILITY**  
    HSMs transaccionales/utility (confirmaciones, notificaciones, alertas).
    
    **dau_count**  
    Daily Active Users - suma de usuarios activos por día en el mes.
    
    **unique_users_count**  
    Usuarios únicos que interactuaron en el mes.
    
    **_mom**  
    Month-over-Month change (%) - cambio porcentual vs mes anterior.
    """)

if view == "📈 Overview":
    st.header(f"Overview - {selected_month}")
    
    # Page description
    st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); border-left: 4px solid #00d4ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <strong>📋 ¿Qué muestra esta vista?</strong><br>
        Resumen ejecutivo del mes seleccionado con métricas agregadas de todo el portafolio.<br><br>
        <strong>🎯 Métricas clave:</strong> Total Workflow Executions (uso de Brain), Marketing Lite & Utility HSMs (campañas), Total DAUs (usuarios activos), Active Companies (cuentas con actividad).<br><br>
        <strong>💡 Qué aprendes:</strong> Salud general del producto en un vistazo, tendencias temporales (¿estamos creciendo o decreciendo?), comparación entre tipos de conversación (¿dominan campañas de marketing o utility?).
    </div>
    """, unsafe_allow_html=True)
    
    # Filter data for selected month
    df_month = df[df['month'] == selected_month]
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_executions = df_month['execution_count'].sum()
        st.metric("Total Executions", f"{total_executions:,.0f}" if pd.notna(total_executions) else "N/A")
    
    with col2:
        total_marketing = df_month['MARKETING'].sum() if 'MARKETING' in df_month.columns else 0
        st.metric("Marketing", f"{total_marketing:,.0f}" if pd.notna(total_marketing) else "N/A")
    
    with col3:
        total_utility = df_month['UTILITY'].sum() if 'UTILITY' in df_month.columns else 0
        st.metric("Utility", f"{total_utility:,.0f}" if pd.notna(total_utility) else "N/A")
    
    with col4:
        total_dau = df_month['dau_count'].sum()
        st.metric("Total DAU", f"{total_dau:,.0f}" if pd.notna(total_dau) else "N/A")
    
    with col5:
        active_companies = df_month['companyId'].nunique()
        st.metric("Active Companies", f"{active_companies:,}")
    
    st.divider()
    
    # Time series charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Executions over time
        exec_by_month = df.groupby('month')['execution_count'].sum().reset_index()
        fig = px.line(exec_by_month, x='month', y='execution_count', 
                      title='Workflow Executions Over Time',
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
        # Billing by conversation type over time - stacked bar chart
        billing_by_month = df.groupby('month')[conv_type_cols].sum().reset_index()
        fig = go.Figure()
        colors = ['#00d4ff', '#ff6b6b', '#ffd93d', '#00ff88']
        for i, col in enumerate(conv_type_cols):
            fig.add_trace(go.Bar(
                x=billing_by_month['month'],
                y=billing_by_month[col],
                name=col,
                marker_color=colors[i % len(colors)]
            ))
        fig.update_layout(
            title='Billable Count by Conversation Type',
            barmode='stack',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # Calculate shared y-axis range for DAU and Unique Users
    dau_by_month = df.groupby('month')['dau_count'].sum().reset_index()
    users_by_month = df.groupby('month')['unique_users_count'].sum().reset_index()
    max_y_users = max(dau_by_month['dau_count'].max(), users_by_month['unique_users_count'].max()) * 1.1
    
    with col1:
        # DAU over time
        fig = px.line(dau_by_month, x='month', y='dau_count',
                      title='Daily Active Users Over Time',
                      markers=True)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff',
            yaxis=dict(range=[0, max_y_users])
        )
        fig.update_traces(line_color='#ff6b6b', marker_color='#ffd93d')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Unique users over time
        fig = px.line(users_by_month, x='month', y='unique_users_count',
                      title='Unique Users Over Time',
                      markers=True)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff',
            yaxis=dict(range=[0, max_y_users])
        )
        fig.update_traces(line_color='#00ff88', marker_color='#00d4ff')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top companies table
    st.subheader("🏢 Top Companies This Month")
    display_cols = ['companyId', 'companyName', 'execution_count'] + conv_type_cols + ['dau_count']
    display_cols = [c for c in display_cols if c in df_month.columns]
    top_companies = df_month.nlargest(10, 'execution_count')[display_cols]
    st.dataframe(top_companies, use_container_width=True, hide_index=True)

elif view == "🔍 Company Lookup":
    st.header("🔍 Company Lookup")
    
    # Page description
    st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); border-left: 4px solid #00d4ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <strong>📋 ¿Qué muestra esta vista?</strong><br>
        Deep-dive en una cuenta específica con toda su historia.<br><br>
        <strong>🎯 Métricas clave:</strong> Valores actuales de cada métrica, cambio MoM (%) para detectar tendencias, gráficos de evolución mensual.<br><br>
        <strong>💡 Qué aprendes:</strong> ¿Esta cuenta está creciendo o decayendo? ¿Qué features usa más (workflows vs campañas)? Preparación para llamadas de CS o renovaciones.
    </div>
    """, unsafe_allow_html=True)
    
    # Company selector
    companies = df[df['companyName'].notna()].drop_duplicates('companyId')[['companyId', 'companyName']].sort_values('companyName')
    company_options = {f"{row['companyName']} ({row['companyId']})": row['companyId'] for _, row in companies.iterrows()}
    
    selected_company_label = st.selectbox("Select Company", list(company_options.keys()))
    selected_company_id = company_options[selected_company_label]
    
    # Filter data for selected company
    df_company = df_with_mom[df_with_mom['companyId'] == selected_company_id].sort_values('month')
    
    if not df_company.empty:
        st.subheader(f"📋 Data for {selected_company_label}")
        
        # Latest month metrics
        latest = df_company.iloc[-1]
        
        # Dynamic columns based on available metrics
        metric_cols = ['execution_count'] + conv_type_cols + ['dau_count', 'unique_users_count']
        metric_cols = [c for c in metric_cols if c in df_company.columns]
        
        cols = st.columns(len(metric_cols))
        
        for i, metric in enumerate(metric_cols):
            with cols[i]:
                val = latest[metric]
                mom_col = f'{metric}_mom'
                mom = latest[mom_col] if mom_col in latest.index else None
                delta = f"{mom:+.1f}%" if pd.notna(mom) else None
                label = metric.replace('_', ' ').title()
                st.metric(label, f"{val:,.0f}" if pd.notna(val) else "N/A", delta=delta)
        
        st.divider()
        
        # Company time series
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(df_company, x='month', y='execution_count',
                        title='Workflow Executions by Month')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff'
            )
            fig.update_traces(marker_color='#00d4ff')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Stacked bar for conversation types
            fig = go.Figure()
            colors = ['#00d4ff', '#ff6b6b', '#ffd93d']
            for i, col in enumerate(conv_type_cols):
                if col in df_company.columns:
                    fig.add_trace(go.Bar(
                        x=df_company['month'],
                        y=df_company[col],
                        name=col,
                        marker_color=colors[i % len(colors)]
                    ))
            fig.update_layout(
                title='Billable Count by Conversation Type',
                barmode='stack',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(df_company, x='month', y='dau_count',
                        title='DAU by Month')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff'
            )
            fig.update_traces(marker_color='#ff6b6b')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(df_company, x='month', y='unique_users_count',
                        title='Unique Users by Month')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff'
            )
            fig.update_traces(marker_color='#00ff88')
            st.plotly_chart(fig, use_container_width=True)
        
        # Full data table with MoM
        st.subheader("📊 Full History with MoM Changes")
        display_cols = ['month', 'execution_count', 'execution_count_mom']
        for col in conv_type_cols:
            display_cols.extend([col, f'{col}_mom'])
        display_cols.extend(['dau_count', 'dau_count_mom', 'unique_users_count', 'unique_users_count_mom'])
        display_cols = [c for c in display_cols if c in df_company.columns]
        st.dataframe(df_company[display_cols].round(2), use_container_width=True, hide_index=True)
    else:
        st.warning("No data found for this company")

elif view == "📊 Distributions":
    st.header(f"📊 Distributions - {selected_month}")
    
    # Page description
    st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); border-left: 4px solid #00d4ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <strong>📋 ¿Qué muestra esta vista?</strong><br>
        Cómo se distribuyen las métricas entre todas las compañías.<br><br>
        <strong>🎯 Visualizaciones:</strong> Histograma (frecuencia), Box plot (outliers, mediana, cuartiles), Top 15 compañías.<br><br>
        <strong>💡 Qué aprendes:</strong> ¿Hay concentración? (pocas cuentas generan la mayoría del uso), ¿Quiénes son los power users?, ¿Cuál es el "típico" cliente? (mediana vs promedio).
    </div>
    """, unsafe_allow_html=True)
    
    df_month = df[df['month'] == selected_month]
    
    # Metric selector
    available_metrics = ['execution_count'] + conv_type_cols + ['dau_count', 'unique_users_count']
    available_metrics = [m for m in available_metrics if m in df_month.columns]
    metric = st.selectbox("Select Metric", available_metrics)
    
    # Define thresholds for each metric to cap the distribution
    metric_thresholds = {
        'execution_count': 100000,
        'MARKETING': 100000,
        'UTILITY': 50000,
        'dau_count': 10000,
        'unique_users_count': 5000
    }
    threshold = metric_thresholds.get(metric, 100000)
    
    # Create capped data for histogram
    df_month_capped = df_month.copy()
    df_month_capped[f'{metric}_capped'] = df_month_capped[metric].apply(
        lambda x: threshold if pd.notna(x) and x > threshold else x
    )
    
    # Count outliers
    outliers_count = (df_month[metric] > threshold).sum()
    normal_count = (df_month[metric] <= threshold).sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram with capped values
        # Create bins from 0 to threshold, plus one bin for outliers
        df_for_hist = df_month[df_month[metric].notna()].copy()
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
        # Box plot (keep original for full picture)
        fig = px.box(df_month, y=metric,
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
    stats = df_month[metric].describe()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mean", f"{stats['mean']:,.1f}" if pd.notna(stats['mean']) else "N/A")
    col2.metric("Median", f"{stats['50%']:,.1f}" if pd.notna(stats['50%']) else "N/A")
    col3.metric("Std Dev", f"{stats['std']:,.1f}" if pd.notna(stats['std']) else "N/A")
    col4.metric("Min", f"{stats['min']:,.0f}" if pd.notna(stats['min']) else "N/A")
    col5.metric("Max", f"{stats['max']:,.0f}" if pd.notna(stats['max']) else "N/A")
    
    # Top companies for this metric - with grouped outliers bar
    st.subheader(f"🏆 Top Companies by {metric}")
    
    # Get top 10 companies above threshold as "Power Users"
    top_outliers = df_month[df_month[metric] > threshold].nlargest(10, metric)
    top_normal = df_month[df_month[metric] <= threshold].nlargest(15, metric)
    
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

elif view == "📉 MoM Analysis":
    st.header("📉 Month-over-Month Analysis")
    
    # Page description
    st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); border-left: 4px solid #00d4ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <strong>📋 ¿Qué muestra esta vista?</strong><br>
        Cambios mes a mes a nivel portafolio y por cuenta.<br><br>
        <strong>🎯 Visualizaciones:</strong> Evolución mensual del total, % cambio MoM (positivo/negativo), Top Gainers y Top Decliners.<br><br>
        <strong>💡 Qué aprendes:</strong> ¿Qué cuentas están en riesgo de churn? (decliners consistentes), ¿Qué cuentas están despegando? (gainers para upsell), Estacionalidad o patrones temporales.
    </div>
    """, unsafe_allow_html=True)
    
    # Metric selector
    available_metrics = ['execution_count'] + conv_type_cols + ['dau_count', 'unique_users_count']
    available_metrics = [m for m in available_metrics if m in df.columns]
    metric = st.selectbox("Select Metric", available_metrics)
    
    # Calculate overall MoM
    overall_mom = df.groupby('month')[metric].sum().reset_index()
    overall_mom = overall_mom.sort_values('month')
    overall_mom['mom_change'] = overall_mom[metric].pct_change() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Total metric over time
        fig = px.bar(overall_mom, x='month', y=metric,
                    title=f'Total {metric} by Month')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff'
        )
        fig.update_traces(marker_color='#00d4ff')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # MoM change
        fig = px.bar(overall_mom, x='month', y='mom_change',
                    title=f'MoM Change (%) for {metric}',
                    color='mom_change',
                    color_continuous_scale=['#ff6b6b', '#ffd93d', '#00ff88'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            title_font_color='#00d4ff'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # MoM table
    st.subheader("📋 MoM Summary Table")
    st.dataframe(overall_mom.round(2), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Companies with biggest MoM changes
    st.subheader(f"🔥 Biggest MoM Changes in {selected_month}")
    
    df_month_mom = df_with_mom[df_with_mom['month'] == selected_month].copy()
    mom_col = f'{metric}_mom'
    
    if mom_col in df_month_mom.columns:
        # Scatter plot: Total metric vs MoM change (to find big companies declining)
        st.markdown("**🎯 Volume vs MoM Change (find big companies at risk)**")
        df_scatter = df_month_mom[df_month_mom[metric].notna() & df_month_mom[mom_col].notna()].copy()
        
        if not df_scatter.empty:
            # Color by MoM: red for negative, green for positive
            df_scatter['status'] = df_scatter[mom_col].apply(lambda x: '📉 Declining' if x < 0 else '📈 Growing')
            
            fig = px.scatter(
                df_scatter,
                x=metric,
                y=mom_col,
                color='status',
                color_discrete_map={'📉 Declining': '#ff6b6b', '📈 Growing': '#00ff88'},
                hover_data=['companyName', 'companyId'],
                title=f'{metric} vs MoM Change - Identify Big Accounts at Risk',
                labels={metric: f'Total {metric}', mom_col: 'MoM Change (%)'}
            )
            
            # Add quadrant lines
            fig.add_hline(y=0, line_dash="dash", line_color="#ffd93d", opacity=0.5)
            
            # Highlight danger zone: high volume + negative MoM (top-left quadrant)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                title_font_color='#00d4ff'
            )
            fig.update_traces(marker=dict(size=10, opacity=0.7))
            st.plotly_chart(fig, use_container_width=True)
            
            # Big decliners: companies with high volume AND negative MoM
            st.markdown("**⚠️ Big Accounts Declining (High Volume + Negative MoM)**")
            median_metric = df_scatter[metric].median()
            big_decliners = df_scatter[(df_scatter[metric] > median_metric) & (df_scatter[mom_col] < 0)].sort_values(mom_col)
            
            if not big_decliners.empty:
                st.dataframe(
                    big_decliners[['companyId', 'companyName', metric, mom_col]].head(15).round(2),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("No big accounts declining this month! 🎉")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Top Gainers**")
            top_gainers = df_month_mom.nlargest(10, mom_col)[['companyId', 'companyName', metric, mom_col]]
            st.dataframe(top_gainers.round(2), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**📉 Top Decliners**")
            top_decliners = df_month_mom.nsmallest(10, mom_col)[['companyId', 'companyName', metric, mom_col]]
            st.dataframe(top_decliners.round(2), use_container_width=True, hide_index=True)
    else:
        st.info("MoM data not available for this metric")

# Footer
st.sidebar.divider()
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit")
