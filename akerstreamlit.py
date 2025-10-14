import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page config
st.set_page_config(
    page_title="Rent Roll Analytics",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv('unified_rent_roll.csv')
    
    # Convert date columns
    date_cols = ['snapshot_date', 'move_in_date', 'move_out_date', 'lease_start', 'lease_end', 'notice_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Convert boolean
    df['lease_start_backfilled'] = df['lease_start_backfilled'].map({'True': True, 'False': False})
    
    return df

# Calculate KPIs
def calculate_portfolio_kpis(df, snapshot_date):
    """Calculate portfolio-level KPIs for a specific snapshot date"""
    df_snapshot = df[df['snapshot_date'] == snapshot_date].copy()
    
    # Filter to current leases only (exclude future pipeline for core metrics)
    current_leases = df_snapshot[df_snapshot['lease_state'] == 'current'].copy()
    
    # Total units (unique units in current state)
    total_units = current_leases['unit_number_canonical'].nunique()
    
    # Occupied units
    occupied_units = current_leases[current_leases['status'] == 'occupied']['unit_number_canonical'].nunique()
    
    # Physical occupancy
    physical_occ = (occupied_units / total_units * 100) if total_units > 0 else 0
    
    # Economic occupancy
    potential_rent = current_leases['market_rent'].sum()
    actual_rent = current_leases['actual_rent'].sum()
    economic_occ = (actual_rent / potential_rent * 100) if potential_rent > 0 else 0
    
    # Vacant units
    vacant_ready = len(current_leases[current_leases['status'] == 'vacant'])
    
    # Pre-leased units (future residents)
    future_leases = df_snapshot[df_snapshot['lease_state'] == 'future']
    pre_leased = len(future_leases)
    
    # Effective rent (actual vs market)
    effective_rent_pct = (actual_rent / potential_rent * 100) if potential_rent > 0 else 0
    
    # Revenue per available unit
    revenue_per_unit = actual_rent / total_units if total_units > 0 else 0
    
    # Concessions
    total_concessions = current_leases['concessions'].sum()
    concessions_pct = (total_concessions / actual_rent * 100) if actual_rent > 0 else 0
    
    # Delinquency
    total_balance = current_leases['balance'].sum()
    total_charges = current_leases['total_charges'].sum()
    delinquency_rate = (total_balance / total_charges * 100) if total_charges > 0 else 0
    
    # Average rent
    avg_market_rent = current_leases['market_rent'].mean()
    avg_actual_rent = current_leases['actual_rent'].mean()
    
    return {
        'total_units': total_units,
        'occupied_units': occupied_units,
        'physical_occupancy': physical_occ,
        'economic_occupancy': economic_occ,
        'vacant_ready': vacant_ready,
        'pre_leased': pre_leased,
        'effective_rent_pct': effective_rent_pct,
        'revenue_per_unit': revenue_per_unit,
        'concessions_pct': concessions_pct,
        'delinquency_rate': delinquency_rate,
        'avg_market_rent': avg_market_rent,
        'avg_actual_rent': avg_actual_rent,
        'total_revenue': actual_rent
    }

def calculate_property_kpis(df, property_name, snapshot_date):
    """Calculate property-specific KPIs"""
    df_filtered = df[(df['property_name'] == property_name) & (df['snapshot_date'] == snapshot_date)].copy()
    current_leases = df_filtered[df_filtered['lease_state'] == 'current'].copy()
    
    total_units = current_leases['unit_number_canonical'].nunique()
    occupied_units = current_leases[current_leases['status'] == 'occupied']['unit_number_canonical'].nunique()
    
    physical_occ = (occupied_units / total_units * 100) if total_units > 0 else 0
    
    potential_rent = current_leases['market_rent'].sum()
    actual_rent = current_leases['actual_rent'].sum()
    economic_occ = (actual_rent / potential_rent * 100) if potential_rent > 0 else 0
    
    return {
        'total_units': total_units,
        'occupied_units': occupied_units,
        'physical_occupancy': physical_occ,
        'economic_occupancy': economic_occ,
        'total_revenue': actual_rent,
        'avg_rent': current_leases['actual_rent'].mean()
    }

def calculate_turnover_metrics(df, snapshot_date):
    """Calculate turnover metrics"""
    df_snapshot = df[df['snapshot_date'] == snapshot_date].copy()
    current_leases = df_snapshot[df_snapshot['lease_state'] == 'current'].copy()
    
    # Move-outs (units with notice or move-out date)
    move_outs = len(current_leases[current_leases['status'].isin(['occupied-ntv', 'occupied-ntvl'])])
    
    # Leases expiring in next 3 months
    three_months_out = snapshot_date + timedelta(days=90)
    expiring_soon = len(current_leases[
        (current_leases['lease_end'] >= snapshot_date) & 
        (current_leases['lease_end'] <= three_months_out)
    ])
    
    return {
        'move_outs': move_outs,
        'expiring_soon': expiring_soon
    }

def analyze_unit_history(df):
    """Analyze unit-level changes across months to identify renewals, turnover, etc."""
    
    # Filter to properties with resh_id data
    df_with_resh = df[df['resh_id'].notna()].copy()
    properties_with_resh = df_with_resh['property_name'].unique()
    
    # Filter to current leases only for cleaner analysis
    df_current = df_with_resh[df_with_resh['lease_state'] == 'current'].copy()
    
    # Sort by unit and date
    df_current = df_current.sort_values(['property_name', 'unit_number_canonical', 'snapshot_date'])
    
    results = []
    
    # Group by property and unit
    for (prop, unit), group in df_current.groupby(['property_name', 'unit_number_canonical']):
        group = group.sort_values('snapshot_date')
        
        for i in range(len(group) - 1):
            current_row = group.iloc[i]
            next_row = group.iloc[i + 1]
            
            current_date = current_row['snapshot_date']
            next_date = next_row['snapshot_date']
            
            # Check if consecutive months (within 45 days)
            days_diff = (next_date - current_date).days
            if days_diff > 45:
                continue
            
            current_resh = current_row['resh_id']
            next_resh = next_row['resh_id']
            current_status = current_row['status']
            next_status = next_row['status']
            
            event_type = None
            rent_change = None
            
            # Identify event type
            if current_status == 'occupied' and next_status == 'occupied':
                if current_resh == next_resh:
                    event_type = 'retention'
                    rent_change = next_row['actual_rent'] - current_row['actual_rent']
                else:
                    event_type = 'turnover'
                    rent_change = next_row['actual_rent'] - current_row['actual_rent']
            
            elif current_status == 'occupied' and next_status == 'vacant':
                event_type = 'move_out'
            
            elif current_status == 'vacant' and next_status == 'occupied':
                event_type = 'new_lease'
            
            elif current_status == 'vacant' and next_status == 'vacant':
                event_type = 'vacant_continued'
            
            if event_type:
                results.append({
                    'property': prop,
                    'unit': unit,
                    'from_date': current_date,
                    'to_date': next_date,
                    'event_type': event_type,
                    'from_resh_id': current_resh,
                    'to_resh_id': next_resh,
                    'from_rent': current_row['actual_rent'],
                    'to_rent': next_row['actual_rent'],
                    'rent_change': rent_change,
                    'from_status': current_status,
                    'to_status': next_status
                })
    
    return pd.DataFrame(results), properties_with_resh

def calculate_vacancy_duration(df):
    """Calculate how long units stay vacant before being leased"""
    
    df_with_resh = df[df['resh_id'].notna()].copy()
    df_current = df_with_resh[df_with_resh['lease_state'] == 'current'].copy()
    df_current = df_current.sort_values(['property_name', 'unit_number_canonical', 'snapshot_date'])
    
    vacancy_periods = []
    
    for (prop, unit), group in df_current.groupby(['property_name', 'unit_number_canonical']):
        group = group.sort_values('snapshot_date')
        
        vacancy_start = None
        
        for _, row in group.iterrows():
            if row['status'] == 'vacant':
                if vacancy_start is None:
                    vacancy_start = row['snapshot_date']
            
            elif row['status'] == 'occupied' and vacancy_start is not None:
                # Unit was leased
                vacancy_days = (row['snapshot_date'] - vacancy_start).days
                vacancy_periods.append({
                    'property': prop,
                    'unit': unit,
                    'vacancy_start': vacancy_start,
                    'lease_date': row['snapshot_date'],
                    'vacancy_days': vacancy_days,
                    'new_rent': row['actual_rent']
                })
                vacancy_start = None
    
    return pd.DataFrame(vacancy_periods)

# Load data
df = load_data()

# Get available dates and properties
available_dates = sorted(df['snapshot_date'].unique())
latest_date = available_dates[-1]
properties = sorted(df['property_name'].unique())

# Helper function for chat
def generate_response(prompt, df, latest_date, properties, available_dates):
    """Generate responses to user queries"""
    prompt_lower = prompt.lower()
    
    # Renewal/turnover queries
    if any(word in prompt_lower for word in ['renewal', 'turnover', 'retention', 'lease up']):
        history_df, properties_with_resh = analyze_unit_history(df)
        
        if history_df.empty:
            return "**Renewal/Turnover Analysis Not Available**\n\nThis analysis requires resh_id data, which is not available for all properties in the dataset."
        
        # Get last 3 months of data
        recent_months = sorted(history_df['to_date'].unique())[-3:]
        recent_data = history_df[history_df['to_date'].isin(recent_months)]
        
        retentions = len(recent_data[recent_data['event_type'] == 'retention'])
        turnovers = len(recent_data[recent_data['event_type'] == 'turnover'])
        occupied_transitions = retentions + turnovers
        renewal_rate = (retentions / occupied_transitions * 100) if occupied_transitions > 0 else 0
        
        # Calculate rent changes
        renewal_rents = recent_data[
            (recent_data['event_type'] == 'retention') & 
            (recent_data['rent_change'].notna())
        ]
        avg_renewal_increase = renewal_rents['rent_change'].mean() if not renewal_rents.empty else 0
        
        return f"""**Renewal & Turnover Analysis (Last 3 Months)**

📊 **Performance**
- Renewal Rate: {renewal_rate:.1f}%
- Retentions: {retentions}
- Turnovers: {turnovers}

💰 **Rent Impact**
- Avg Renewal Rent Increase: ${avg_renewal_increase:.0f}

📝 **Note:** Analysis based on properties with resh_id tracking: {', '.join(sorted(properties_with_resh))}

View the Time-Series Analysis page for detailed charts and property-level breakdowns!
"""
    
    # Portfolio summary
    elif any(word in prompt_lower for word in ['portfolio', 'overall', 'total', 'all properties']):
        kpis = calculate_portfolio_kpis(df, latest_date)
        return f"""**Portfolio Summary ({latest_date.strftime('%B %Y')})**

📊 **Occupancy**
- Physical Occupancy: {kpis['physical_occupancy']:.1f}%
- Economic Occupancy: {kpis['economic_occupancy']:.1f}%
- Total Units: {kpis['total_units']:,}
- Occupied: {kpis['occupied_units']:,}
- Vacant Ready: {kpis['vacant_ready']:,}

💰 **Revenue**
- Total Revenue: ${kpis['total_revenue']:,.0f}
- Revenue per Unit: ${kpis['revenue_per_unit']:,.0f}
- Average Market Rent: ${kpis['avg_market_rent']:,.0f}
- Average Actual Rent: ${kpis['avg_actual_rent']:,.0f}

📈 **Performance**
- Delinquency Rate: {kpis['delinquency_rate']:.1f}%
- Pre-Leased Units: {kpis['pre_leased']:,}
"""
    
    # Property comparison
    elif 'compare' in prompt_lower or 'best' in prompt_lower or 'worst' in prompt_lower:
        property_data = []
        for prop in properties:
            kpis = calculate_property_kpis(df, prop, latest_date)
            property_data.append({
                'Property': prop,
                'Occupancy': kpis['physical_occupancy'],
                'Revenue': kpis['total_revenue'],
                'Units': kpis['total_units']
            })
        
        prop_df = pd.DataFrame(property_data).sort_values('Occupancy', ascending=False)
        
        response = f"**Property Comparison ({latest_date.strftime('%B %Y')})**\n\n"
        for _, row in prop_df.iterrows():
            response += f"- **{row['Property']}**: {row['Occupancy']:.1f}% occupied, ${row['Revenue']:,.0f} revenue, {row['Units']} units\n"
        
        return response
    
    # Trends
    elif any(word in prompt_lower for word in ['trend', 'change', 'growth', 'month']):
        trend_data = []
        for date in available_dates[-3:]:  # Last 3 months
            kpis = calculate_portfolio_kpis(df, date)
            trend_data.append({
                'Month': date.strftime('%B %Y'),
                'Occupancy': kpis['physical_occupancy'],
                'Revenue': kpis['total_revenue']
            })
        
        response = "**Recent Trends (Last 3 Months)**\n\n"
        for data in trend_data:
            response += f"- **{data['Month']}**: {data['Occupancy']:.1f}% occupied, ${data['Revenue']:,.0f} revenue\n"
        
        return response
    
    else:
        return """I can help you with:
- **Portfolio summaries**: "Tell me about the portfolio"
- **Property comparisons**: "Which property performs best?"
- **Trends**: "Show me recent trends"
- **Renewal & Turnover**: "What's our renewal rate?"
- **Specific properties**: "Tell me about Ellsworth"

Try asking one of these questions!"""

# Sidebar navigation
st.sidebar.title("🏢 Rent Roll Analytics")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Portfolio Overview", "🏘️ Property Drill-Down", "📈 Time-Series Analysis", 
     "💬 Chat Interface", "⚠️ Anomaly Monitor"]
)

# ============================================================================
# PAGE 1: PORTFOLIO OVERVIEW
# ============================================================================
if page == "📊 Portfolio Overview":
    st.title("📊 Portfolio Overview")
    
    # Date selector
    selected_date = st.sidebar.selectbox(
        "Select Snapshot Date",
        available_dates,
        index=len(available_dates)-1,
        format_func=lambda x: x.strftime('%B %Y')
    )
    
    # Calculate current and previous month KPIs
    current_kpis = calculate_portfolio_kpis(df, selected_date)
    
    # Get previous month for comparison
    prev_date_idx = available_dates.index(selected_date) - 1
    if prev_date_idx >= 0:
        prev_date = available_dates[prev_date_idx]
        prev_kpis = calculate_portfolio_kpis(df, prev_date)
    else:
        prev_kpis = None
    
    # Display key metrics in columns
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta = current_kpis['physical_occupancy'] - prev_kpis['physical_occupancy'] if prev_kpis else None
        st.metric(
            "Physical Occupancy",
            f"{current_kpis['physical_occupancy']:.1f}%",
            f"{delta:+.1f}%" if delta else None
        )
    
    with col2:
        delta = current_kpis['economic_occupancy'] - prev_kpis['economic_occupancy'] if prev_kpis else None
        st.metric(
            "Economic Occupancy",
            f"{current_kpis['economic_occupancy']:.1f}%",
            f"{delta:+.1f}%" if delta else None
        )
    
    with col3:
        delta = current_kpis['total_revenue'] - prev_kpis['total_revenue'] if prev_kpis else None
        st.metric(
            "Total Revenue",
            f"${current_kpis['total_revenue']:,.0f}",
            f"${delta:+,.0f}" if delta else None
        )
    
    with col4:
        delta = current_kpis['delinquency_rate'] - prev_kpis['delinquency_rate'] if prev_kpis else None
        st.metric(
            "Delinquency Rate",
            f"{current_kpis['delinquency_rate']:.1f}%",
            f"{delta:+.1f}%" if delta else None,
            delta_color="inverse"
        )
    
    # Second row of metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Total Units", f"{current_kpis['total_units']:,}")
    
    with col6:
        st.metric("Occupied Units", f"{current_kpis['occupied_units']:,}")
    
    with col7:
        st.metric("Vacant Ready", f"{current_kpis['vacant_ready']:,}")
    
    with col8:
        st.metric("Pre-Leased", f"{current_kpis['pre_leased']:,}")
    
    # Charts section
    st.subheader("Portfolio Trends")
    
    # Calculate trends across all months
    trend_data = []
    for date in available_dates:
        kpis = calculate_portfolio_kpis(df, date)
        trend_data.append({
            'Date': date,
            'Physical Occupancy': kpis['physical_occupancy'],
            'Economic Occupancy': kpis['economic_occupancy'],
            'Avg Market Rent': kpis['avg_market_rent'],
            'Avg Actual Rent': kpis['avg_actual_rent'],
            'Total Revenue': kpis['total_revenue'],
            'Delinquency Rate': kpis['delinquency_rate']
        })
    
    trend_df = pd.DataFrame(trend_data)
    
    # Occupancy trend
    col1, col2 = st.columns(2)
    
    with col1:
        fig_occ = go.Figure()
        fig_occ.add_trace(go.Scatter(
            x=trend_df['Date'],
            y=trend_df['Physical Occupancy'],
            name='Physical Occupancy',
            mode='lines+markers',
            line=dict(color='#1f77b4', width=3)
        ))
        fig_occ.add_trace(go.Scatter(
            x=trend_df['Date'],
            y=trend_df['Economic Occupancy'],
            name='Economic Occupancy',
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=3)
        ))
        fig_occ.update_layout(
            title="Occupancy Trends",
            yaxis_title="Occupancy %",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_occ, use_container_width=True)
    
    with col2:
        fig_rent = go.Figure()
        fig_rent.add_trace(go.Scatter(
            x=trend_df['Date'],
            y=trend_df['Avg Market Rent'],
            name='Market Rent',
            mode='lines+markers',
            line=dict(color='#2ca02c', width=3)
        ))
        fig_rent.add_trace(go.Scatter(
            x=trend_df['Date'],
            y=trend_df['Avg Actual Rent'],
            name='Actual Rent',
            mode='lines+markers',
            line=dict(color='#d62728', width=3)
        ))
        fig_rent.update_layout(
            title="Average Rent Trends",
            yaxis_title="Rent ($)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_rent, use_container_width=True)
    
    # Revenue and delinquency
    col3, col4 = st.columns(2)
    
    with col3:
        fig_rev = px.bar(
            trend_df,
            x='Date',
            y='Total Revenue',
            title="Total Revenue by Month"
        )
        fig_rev.update_layout(height=400)
        st.plotly_chart(fig_rev, use_container_width=True)
    
    with col4:
        fig_delinq = px.line(
            trend_df,
            x='Date',
            y='Delinquency Rate',
            title="Delinquency Rate Trend",
            markers=True
        )
        fig_delinq.update_layout(height=400)
        st.plotly_chart(fig_delinq, use_container_width=True)
    
    # Property comparison
    st.subheader("Property Performance Comparison")
    
    property_data = []
    for prop in properties:
        prop_kpis = calculate_property_kpis(df, prop, selected_date)
        property_data.append({
            'Property': prop,
            'Units': prop_kpis['total_units'],
            'Occupancy %': prop_kpis['physical_occupancy'],
            'Revenue': prop_kpis['total_revenue'],
            'Avg Rent': prop_kpis['avg_rent']
        })
    
    prop_df = pd.DataFrame(property_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_prop_occ = px.bar(
            prop_df.sort_values('Occupancy %', ascending=False),
            x='Property',
            y='Occupancy %',
            title="Physical Occupancy by Property",
            color='Occupancy %',
            color_continuous_scale='RdYlGn'
        )
        fig_prop_occ.update_layout(height=400)
        st.plotly_chart(fig_prop_occ, use_container_width=True)
    
    with col2:
        fig_prop_rev = px.bar(
            prop_df.sort_values('Revenue', ascending=False),
            x='Property',
            y='Revenue',
            title="Revenue by Property"
        )
        fig_prop_rev.update_layout(height=400)
        st.plotly_chart(fig_prop_rev, use_container_width=True)

# ============================================================================
# PAGE 2: PROPERTY DRILL-DOWN
# ============================================================================
elif page == "🏘️ Property Drill-Down":
    st.title("🏘️ Property Drill-Down")
    
    # Property and date selectors
    col1, col2 = st.columns(2)
    with col1:
        selected_property = st.selectbox("Select Property", properties)
    with col2:
        selected_date = st.selectbox(
            "Select Snapshot Date",
            available_dates,
            index=len(available_dates)-1,
            format_func=lambda x: x.strftime('%B %Y')
        )
    
    # Filter data
    df_filtered = df[
        (df['property_name'] == selected_property) & 
        (df['snapshot_date'] == selected_date)
    ].copy()
    
    current_leases = df_filtered[df_filtered['lease_state'] == 'current'].copy()
    
    # Calculate KPIs
    prop_kpis = calculate_property_kpis(df, selected_property, selected_date)
    turnover = calculate_turnover_metrics(df_filtered, selected_date)
    
    # Display KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Units", prop_kpis['total_units'])
    with col2:
        st.metric("Occupancy", f"{prop_kpis['physical_occupancy']:.1f}%")
    with col3:
        st.metric("Revenue", f"${prop_kpis['total_revenue']:,.0f}")
    with col4:
        st.metric("Avg Rent", f"${prop_kpis['avg_rent']:,.0f}")
    with col5:
        st.metric("Expiring Soon", turnover['expiring_soon'])
    
    # Unit details
    st.subheader("Unit Details")
    
    # Create display dataframe
    display_df = current_leases[[
        'unit_number', 'unit_type', 'sq_ft', 'status', 'resident_name',
        'lease_start', 'lease_end', 'market_rent', 'actual_rent', 'balance'
    ]].copy()
    
    # Format columns
    display_df['lease_start'] = display_df['lease_start'].dt.strftime('%Y-%m-%d')
    display_df['lease_end'] = display_df['lease_end'].dt.strftime('%Y-%m-%d')
    display_df['market_rent'] = display_df['market_rent'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    display_df['actual_rent'] = display_df['actual_rent'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    display_df['balance'] = display_df['balance'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Charts
    st.subheader("Property Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Status distribution
        status_counts = current_leases['status'].value_counts()
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Unit Status Distribution"
        )
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        # Rent distribution
        fig_rent = px.histogram(
            current_leases,
            x='actual_rent',
            nbins=30,
            title="Rent Distribution"
        )
        fig_rent.update_layout(xaxis_title="Actual Rent ($)", yaxis_title="Count")
        st.plotly_chart(fig_rent, use_container_width=True)
    
    # Property trend
    st.subheader("Historical Trends")
    
    prop_trend = []
    for date in available_dates:
        kpis = calculate_property_kpis(df, selected_property, date)
        prop_trend.append({
            'Date': date,
            'Occupancy': kpis['physical_occupancy'],
            'Revenue': kpis['total_revenue'],
            'Avg Rent': kpis['avg_rent']
        })
    
    prop_trend_df = pd.DataFrame(prop_trend)
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=prop_trend_df['Date'],
        y=prop_trend_df['Occupancy'],
        name='Occupancy %',
        yaxis='y',
        mode='lines+markers'
    ))
    fig_trend.add_trace(go.Scatter(
        x=prop_trend_df['Date'],
        y=prop_trend_df['Revenue'],
        name='Revenue',
        yaxis='y2',
        mode='lines+markers'
    ))
    
    fig_trend.update_layout(
        title=f"{selected_property} - Occupancy & Revenue Trends",
        yaxis=dict(title="Occupancy %", side='left'),
        yaxis2=dict(title="Revenue ($)", overlaying='y', side='right'),
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)

# ============================================================================
# PAGE 3: TIME-SERIES ANALYSIS
# ============================================================================
elif page == "📈 Time-Series Analysis":
    st.title("📈 Time-Series Analysis")
    st.info("**Note:** This analysis uses `resh_id` to track residents across months. Only properties with resh_id data are included.")
    
    # Run the analysis
    with st.spinner("Analyzing unit history across all months..."):
        history_df, properties_with_resh = analyze_unit_history(df)
        vacancy_df = calculate_vacancy_duration(df)
    
    if history_df.empty:
        st.warning("No resh_id data available for time-series analysis.")
    else:
        # Show which properties are included
        st.sidebar.markdown("**Properties with tracking:**")
        for prop in sorted(properties_with_resh):
            st.sidebar.markdown(f"✓ {prop}")
        
        # Date range selector
        available_months = sorted(history_df['to_date'].unique())
        if len(available_months) > 3:
            default_start = available_months[-4]
        else:
            default_start = available_months[0]
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.selectbox(
                "From Date",
                available_months,
                index=available_months.index(default_start),
                format_func=lambda x: x.strftime('%B %Y')
            )
        with col2:
            end_date = st.selectbox(
                "To Date",
                [d for d in available_months if d >= start_date],
                index=len([d for d in available_months if d >= start_date]) - 1,
                format_func=lambda x: x.strftime('%B %Y')
            )
        
        # Filter data
        history_filtered = history_df[
            (history_df['to_date'] >= start_date) & 
            (history_df['to_date'] <= end_date)
        ].copy()
        
        # Calculate key metrics
        total_events = len(history_filtered)
        retentions = len(history_filtered[history_filtered['event_type'] == 'retention'])
        turnovers = len(history_filtered[history_filtered['event_type'] == 'turnover'])
        new_leases = len(history_filtered[history_filtered['event_type'] == 'new_lease'])
        move_outs = len(history_filtered[history_filtered['event_type'] == 'move_out'])
        
        # Renewal rate
        occupied_transitions = retentions + turnovers
        renewal_rate = (retentions / occupied_transitions * 100) if occupied_transitions > 0 else 0
        
        # Display KPIs
        st.subheader("Key Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Renewal Rate", f"{renewal_rate:.1f}%")
        with col2:
            st.metric("Retentions", retentions)
        with col3:
            st.metric("Turnovers", turnovers)
        with col4:
            st.metric("New Leases", new_leases)
        with col5:
            st.metric("Move-Outs", move_outs)
        
        # Vacancy duration analysis
        if not vacancy_df.empty:
            vacancy_filtered = vacancy_df[
                (vacancy_df['lease_date'] >= start_date) & 
                (vacancy_df['lease_date'] <= end_date)
            ]
            
            if not vacancy_filtered.empty:
                avg_vacancy_days = vacancy_filtered['vacancy_days'].mean()
                
                st.subheader("Vacancy Duration")
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.metric("Avg Days to Lease", f"{avg_vacancy_days:.0f}")
                    st.metric("Total Leased Units", len(vacancy_filtered))
                
                with col2:
                    fig_vacancy = px.histogram(
                        vacancy_filtered,
                        x='vacancy_days',
                        nbins=20,
                        title="Distribution of Vacancy Duration",
                        labels={'vacancy_days': 'Days Vacant', 'count': 'Number of Units'}
                    )
                    fig_vacancy.add_vline(
                        x=avg_vacancy_days,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Avg: {avg_vacancy_days:.0f} days"
                    )
                    st.plotly_chart(fig_vacancy, use_container_width=True)
        
        # Rent change analysis: New Leases vs Renewals
        st.subheader("Rent Analysis: New Leases vs Renewals")
        
        # Calculate rent changes
        renewal_rents = history_filtered[
            (history_filtered['event_type'] == 'retention') & 
            (history_filtered['rent_change'].notna())
        ]
        
        turnover_rents = history_filtered[
            (history_filtered['event_type'] == 'turnover') & 
            (history_filtered['to_rent'].notna()) & 
            (history_filtered['from_rent'].notna())
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not renewal_rents.empty:
                avg_renewal_increase = renewal_rents['rent_change'].mean()
                avg_renewal_pct = (renewal_rents['rent_change'] / renewal_rents['from_rent'] * 100).mean()
                
                st.metric(
                    "Avg Renewal Rent Increase",
                    f"${avg_renewal_increase:.0f}",
                    f"{avg_renewal_pct:.1f}%"
                )
                
                fig_renewal = px.histogram(
                    renewal_rents,
                    x='rent_change',
                    nbins=30,
                    title="Renewal Rent Changes"
                )
                fig_renewal.update_layout(xaxis_title="Rent Change ($)", yaxis_title="Count")
                st.plotly_chart(fig_renewal, use_container_width=True)
            else:
                st.info("No renewal data available for selected period")
        
        with col2:
            if not turnover_rents.empty:
                avg_turnover_rent_new = turnover_rents['to_rent'].mean()
                avg_turnover_rent_old = turnover_rents['from_rent'].mean()
                trade_out = avg_turnover_rent_new - avg_turnover_rent_old
                trade_out_pct = (trade_out / avg_turnover_rent_old * 100) if avg_turnover_rent_old > 0 else 0
                
                st.metric(
                    "Avg Turnover Rent Change (Trade-Out)",
                    f"${trade_out:.0f}",
                    f"{trade_out_pct:.1f}%"
                )
                
                fig_turnover = px.histogram(
                    turnover_rents,
                    x='rent_change',
                    nbins=30,
                    title="Turnover Rent Changes"
                )
                fig_turnover.update_layout(xaxis_title="Rent Change ($)", yaxis_title="Count")
                st.plotly_chart(fig_turnover, use_container_width=True)
            else:
                st.info("No turnover data available for selected period")
        
        # Monthly trends
        st.subheader("Monthly Trends")
        
        # Aggregate by month
        monthly_events = history_filtered.groupby(['to_date', 'event_type']).size().reset_index(name='count')
        
        fig_events = px.line(
            monthly_events,
            x='to_date',
            y='count',
            color='event_type',
            title="Event Types Over Time",
            markers=True
        )
        fig_events.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Events",
            hovermode='x unified'
        )
        st.plotly_chart(fig_events, use_container_width=True)
        
        # Calculate rolling renewal rate
        monthly_stats = []
        for date in sorted(history_filtered['to_date'].unique()):
            month_data = history_filtered[history_filtered['to_date'] == date]
            month_retentions = len(month_data[month_data['event_type'] == 'retention'])
            month_turnovers = len(month_data[month_data['event_type'] == 'turnover'])
            month_occupied = month_retentions + month_turnovers
            month_renewal_rate = (month_retentions / month_occupied * 100) if month_occupied > 0 else None
            
            monthly_stats.append({
                'date': date,
                'renewal_rate': month_renewal_rate,
                'retentions': month_retentions,
                'turnovers': month_turnovers
            })
        
        monthly_stats_df = pd.DataFrame(monthly_stats)
        
        fig_renewal_trend = go.Figure()
        fig_renewal_trend.add_trace(go.Scatter(
            x=monthly_stats_df['date'],
            y=monthly_stats_df['renewal_rate'],
            mode='lines+markers',
            name='Renewal Rate',
            line=dict(color='#2ca02c', width=3)
        ))
        fig_renewal_trend.update_layout(
            title="Renewal Rate Trend",
            xaxis_title="Month",
            yaxis_title="Renewal Rate (%)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_renewal_trend, use_container_width=True)
        
        # Property comparison
        st.subheader("Property Comparison")
        
        property_stats = []
        for prop in sorted(history_filtered['property'].unique()):
            prop_data = history_filtered[history_filtered['property'] == prop]
            prop_retentions = len(prop_data[prop_data['event_type'] == 'retention'])
            prop_turnovers = len(prop_data[prop_data['event_type'] == 'turnover'])
            prop_occupied = prop_retentions + prop_turnovers
            prop_renewal_rate = (prop_retentions / prop_occupied * 100) if prop_occupied > 0 else 0
            
            property_stats.append({
                'Property': prop,
                'Renewal Rate': prop_renewal_rate,
                'Retentions': prop_retentions,
                'Turnovers': prop_turnovers,
                'New Leases': len(prop_data[prop_data['event_type'] == 'new_lease'])
            })
        
        prop_stats_df = pd.DataFrame(property_stats)
        
        fig_prop_renewal = px.bar(
            prop_stats_df.sort_values('Renewal Rate', ascending=False),
            x='Property',
            y='Renewal Rate',
            title="Renewal Rate by Property",
            color='Renewal Rate',
            color_continuous_scale='RdYlGn'
        )
        fig_prop_renewal.update_layout(height=400)
        st.plotly_chart(fig_prop_renewal, use_container_width=True)
        
        # Detailed event log
        with st.expander("📋 View Detailed Event Log"):
            display_events = history_filtered[[
                'property', 'unit', 'to_date', 'event_type', 
                'from_rent', 'to_rent', 'rent_change'
            ]].copy()
            
            display_events['to_date'] = display_events['to_date'].dt.strftime('%Y-%m-%d')
            display_events = display_events.sort_values(['to_date', 'property', 'unit'], ascending=[False, True, True])
            
            st.dataframe(display_events, use_container_width=True, height=400)

# ============================================================================
# PAGE 4: CHAT INTERFACE
# ============================================================================
elif page == "💬 Chat Interface":
    st.title("💬 Chat with Your Data")
    
    st.info("""
    **Available Insights:**
    - Ask about occupancy rates, revenue, or trends
    - Compare properties
    - Analyze lease expirations
    - Review unit types and pricing
    - Ask about renewal rates and turnover (from time-series analysis)
    """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your rent roll data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            response = generate_response(prompt, df, latest_date, properties, available_dates)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# ============================================================================
# PAGE 5: ANOMALY MONITOR
# ============================================================================
elif page == "⚠️ Anomaly Monitor":
    st.title("⚠️ Anomaly Monitor")
    st.subheader(f"Data Quality Checks - {latest_date.strftime('%B %Y')}")
    
    # Filter to latest month
    df_latest = df[df['snapshot_date'] == latest_date].copy()
    current_latest = df_latest[df_latest['lease_state'] == 'current'].copy()
    
    # Anomaly checks
    anomalies = []
    
    # 1. Missing critical fields
    missing_resident = len(current_latest[(current_latest['status'] == 'occupied') & (current_latest['resident_name'].isna())])
    if missing_resident > 0:
        anomalies.append({
            'severity': 'High',
            'category': 'Missing Data',
            'description': f'{missing_resident} occupied units missing resident name',
            'count': missing_resident
        })
    
    # 2. Zero rent on occupied units
    zero_rent = len(current_latest[(current_latest['status'] == 'occupied') & (current_latest['actual_rent'] == 0)])
    if zero_rent > 0:
        anomalies.append({
            'severity': 'High',
            'category': 'Revenue Issue',
            'description': f'{zero_rent} occupied units with zero rent',
            'count': zero_rent
        })
    
    # 3. High delinquency
    high_delinq = current_latest[current_latest['balance'] > current_latest['actual_rent'] * 2]
    if len(high_delinq) > 0:
        anomalies.append({
            'severity': 'Medium',
            'category': 'Delinquency',
            'description': f'{len(high_delinq)} units with balance > 2x rent',
            'count': len(high_delinq)
        })
    
    # 4. Expired leases
    expired = current_latest[current_latest['lease_end'] < latest_date]
    if len(expired) > 0:
        anomalies.append({
            'severity': 'Medium',
            'category': 'Lease Issue',
            'description': f'{len(expired)} units with expired leases',
            'count': len(expired)
        })
    
    # 5. Actual rent > market rent significantly
    overmarket = current_latest[current_latest['actual_rent'] > current_latest['market_rent'] * 1.1]
    if len(overmarket) > 0:
        anomalies.append({
            'severity': 'Low',
            'category': 'Pricing',
            'description': f'{len(overmarket)} units with actual rent >10% above market',
            'count': len(overmarket)
        })
    
    # 6. Backfilled lease starts
    backfilled = len(current_latest[current_latest['lease_start_backfilled'] == True])
    if backfilled > 0:
        anomalies.append({
            'severity': 'Info',
            'category': 'Data Quality',
            'description': f'{backfilled} units with backfilled lease start dates',
            'count': backfilled
        })
    
    # Display anomalies
    if anomalies:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            high_severity = len([a for a in anomalies if a['severity'] == 'High'])
            st.metric("High Severity", high_severity, delta_color="inverse")
        with col2:
            medium_severity = len([a for a in anomalies if a['severity'] == 'Medium'])
            st.metric("Medium Severity", medium_severity)
        with col3:
            total_anomalies = sum([a['count'] for a in anomalies])
            st.metric("Total Issues", total_anomalies)
        
        # Anomaly table
        st.subheader("Detected Anomalies")
        anomaly_df = pd.DataFrame(anomalies)
        
        # Color code by severity
        def color_severity(val):
            if val == 'High':
                return 'background-color: #ffcccc'
            elif val == 'Medium':
                return 'background-color: #fff4cc'
            elif val == 'Low':
                return 'background-color: #e6f3ff'
            return ''
        
        styled_df = anomaly_df.style.applymap(color_severity, subset=['severity'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Detailed views
        st.subheader("Anomaly Details")
        
        selected_anomaly = st.selectbox(
            "Select anomaly to investigate",
            [a['description'] for a in anomalies]
        )
        
        # Show affected units based on selection
        if 'missing resident name' in selected_anomaly:
            affected = current_latest[(current_latest['status'] == 'occupied') & (current_latest['resident_name'].isna())]
        elif 'zero rent' in selected_anomaly:
            affected = current_latest[(current_latest['status'] == 'occupied') & (current_latest['actual_rent'] == 0)]
        elif 'balance > 2x rent' in selected_anomaly:
            affected = current_latest[current_latest['balance'] > current_latest['actual_rent'] * 2]
        elif 'expired leases' in selected_anomaly:
            affected = current_latest[current_latest['lease_end'] < latest_date]
        elif 'above market' in selected_anomaly:
            affected = current_latest[current_latest['actual_rent'] > current_latest['market_rent'] * 1.1]
        elif 'backfilled' in selected_anomaly:
            affected = current_latest[current_latest['lease_start_backfilled'] == True]
        else:
            affected = pd.DataFrame()
        
        if not affected.empty:
            display_cols = ['property_name', 'unit_number', 'status', 'resident_name', 
                          'actual_rent', 'market_rent', 'balance', 'lease_end']
            st.dataframe(affected[display_cols].head(50), use_container_width=True)
    
    else:
        st.success("✅ No anomalies detected! Data quality looks good.")
    
    # Data completeness metrics
    st.subheader("Data Completeness")
    
    completeness = {
        'Field': [],
        'Completeness %': [],
        'Missing': []
    }
    
    key_fields = ['resident_name', 'actual_rent', 'market_rent', 'lease_start', 
                  'lease_end', 'move_in_date', 'unit_type']
    
    for field in key_fields:
        if field in current_latest.columns:
            complete = current_latest[field].notna().sum()
            total = len(current_latest)
            pct = (complete / total * 100) if total > 0 else 0
            completeness['Field'].append(field)
            completeness['Completeness %'].append(pct)
            completeness['Missing'].append(total - complete)
    
    comp_df = pd.DataFrame(completeness)
    
    fig_comp = px.bar(
        comp_df,
        x='Field',
        y='Completeness %',
        title="Field Completeness",
        color='Completeness %',
        color_continuous_scale='RdYlGn',
        range_color=[0, 100]
    )
    fig_comp.update_layout(height=400)
    st.plotly_chart(fig_comp, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Data Summary**")
st.sidebar.markdown(f"Properties: {len(properties)}")
st.sidebar.markdown(f"Date Range: {available_dates[0].strftime('%b %Y')} - {available_dates[-1].strftime('%b %Y')}")
st.sidebar.markdown(f"Total Records: {len(df):,}")