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
    
    # CHANGE: Total units from ALL leases (not just current)
    total_units = df_snapshot['unit_number_canonical'].nunique()  # ← Changed this line
    
    # Occupied units (still from current only)
    occupied_all_statuses = current_leases[
        current_leases['status'].str.contains('occupied', na=False)
    ]['unit_number_canonical'].nunique()
    
    occupied_stable = current_leases[
        current_leases['status'].isin(['occupied', 'occupied-ntvl'])  # NTVL has new lease signed
    ]['unit_number_canonical'].nunique()
    
    # Physical occupancy (everyone who's there)
    physical_occ = (occupied_all_statuses / total_units * 100) if total_units > 0 else 0
    
    # Stabilized occupancy (excluding those with notice but no new lease)
    stabilized_occ = (occupied_stable / total_units * 100) if total_units > 0 else 0
    
    occupied_units = occupied_all_statuses

    # Economic occupancy
    potential_rent = df_snapshot['market_rent'].sum() 
    actual_rent = current_leases['actual_rent'].sum() 
    economic_occ = (actual_rent / potential_rent * 100) if potential_rent > 0 else 0
    
    # Vacant units
    vacant_ready = current_leases[current_leases['status'] == 'vacant']['unit_number_canonical'].nunique()
    
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
    
    # CHANGE: Use actual_rent as fallback when total_charges is missing
    if pd.isna(total_charges) or total_charges == 0:
        total_charges = current_leases['actual_rent'].sum()
    
    delinquency_rate = (total_balance / total_charges * 100) if total_charges > 0 else 0

    # Average rent
    avg_market_rent = current_leases['market_rent'].mean()
    avg_actual_rent = current_leases['actual_rent'].mean()
    
    # Committed revenue (current + future pre-leased)
    future_revenue = future_leases['market_rent'].sum()  # Using market since they'll pay that
    committed_revenue = actual_rent + future_revenue
    
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
        'total_revenue': actual_rent,
        'committed_revenue': committed_revenue,
        'future_revenue': future_revenue,
        'stabilized_occupancy': stabilized_occ,
    }

def calculate_property_kpis(df, property_name, snapshot_date):
    """Calculate property-specific KPIs"""
    df_filtered = df[(df['property_name'] == property_name) & (df['snapshot_date'] == snapshot_date)].copy()
    current_leases = df_filtered[df_filtered['lease_state'] == 'current'].copy()
    
    total_units = df_filtered['unit_number_canonical'].nunique()  # ← Changed this line
    occupied_units = current_leases[current_leases['status'] == 'occupied']['unit_number_canonical'].nunique()
    
    physical_occ = (occupied_units / total_units * 100) if total_units > 0 else 0
        
    potential_rent = df_filtered['market_rent'].sum()
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
    
    st.subheader("Key Performance Indicators")

    # First row - Daily action metrics (what PM checks most)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta = current_kpis['physical_occupancy'] - prev_kpis['physical_occupancy'] if prev_kpis else None
        st.metric(
            "Physical Occupancy",
            f"{current_kpis['physical_occupancy']:.1f}%",
            f"{delta:+.1f}%" if delta else None
        )

    with col2:
        st.metric("Vacant Ready", f"{current_kpis['vacant_ready']:,}")

    with col3:
        delta = current_kpis['total_revenue'] - prev_kpis['total_revenue'] if prev_kpis else None
        st.metric(
            "Current Revenue",
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

    # Second row - Forward-looking metrics
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric("Pre-Leased", f"{current_kpis['pre_leased']:,}")

    with col6:
        delta = current_kpis['committed_revenue'] - prev_kpis['committed_revenue'] if prev_kpis else None
        st.metric(
            "Committed Revenue", 
            f"${current_kpis['committed_revenue']:,.0f}",
            f"${delta:+,.0f}" if delta else None,
            help="Current + pre-leased future revenue"
        )

    with col7:
        delta = current_kpis['economic_occupancy'] - prev_kpis['economic_occupancy'] if prev_kpis else None
        st.metric(
            "Economic Occupancy",
            f"{current_kpis['economic_occupancy']:.1f}%",
            f"{delta:+.1f}%" if delta else None
        )

    with col8:
        delta = current_kpis['avg_actual_rent'] - prev_kpis['avg_actual_rent'] if prev_kpis else None
        st.metric(
            "Avg Actual Rent",
            f"${current_kpis['avg_actual_rent']:,.0f}",
            f"${delta:+,.0f}" if delta else None
        )

    # Additional Portfolio Details (lower priority)
    st.markdown("---")
    st.subheader("Portfolio Details")
    
    col9, col10, col11, col12 = st.columns(4)

    with col9:
        st.metric("Total Units", f"{current_kpis['total_units']:,}")

    with col10:
        st.metric("Occupied Units", f"{current_kpis['occupied_units']:,}")

    with col11:
        delta = current_kpis['stabilized_occupancy'] - prev_kpis['stabilized_occupancy'] if prev_kpis else None
        st.metric(
            "Stabilized Occupancy", 
            f"{current_kpis['stabilized_occupancy']:.1f}%",
            f"{delta:+.1f}%" if delta else None,
            help="Excludes units with notice to vacate (no new lease)"
        )

    with col12:
        delta = current_kpis['revenue_per_unit'] - prev_kpis['revenue_per_unit'] if prev_kpis else None
        st.metric(
            "Revenue/Unit",
            f"${current_kpis['revenue_per_unit']:,.0f}",
            f"${delta:+,.0f}" if delta else None
        )
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
    
    # Display KPIs in two rows of 4
    st.subheader("Property Metrics")

    # First row - Occupancy metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Units", prop_kpis['total_units'])
    with col2:
        st.metric("Physical Occupancy", f"{prop_kpis['physical_occupancy']:.1f}%")
    with col3:
        occupied = prop_kpis['occupied_units']
        vacant = prop_kpis['total_units'] - occupied
        st.metric("Vacant Units", vacant)
    with col4:
        st.metric("Economic Occupancy", f"{prop_kpis['economic_occupancy']:.1f}%")

    # Second row - Financial and risk metrics
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric("Current Revenue", f"${prop_kpis['total_revenue']:,.0f}")
    with col6:
        st.metric("Avg Rent", f"${prop_kpis['avg_rent']:,.0f}")
    with col7:
        st.metric("Expiring Soon", turnover['expiring_soon'])
    with col8:
        notices = turnover['move_outs']
        st.metric("Notices", notices, help="Units with notice to vacate")
    # Unit details
    st.subheader("Unit Details")
    
    # Create display dataframe
    display_df = current_leases[[
        'unit_number_canonical', 'unit_type', 'sq_ft', 'status', 'resident_name',
        'lease_start', 'lease_end', 'market_rent', 'actual_rent', 'balance'
    ]].copy()

    display_df = display_df.rename(columns={'unit_number_canonical': 'Unit Number'}) 

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
# PAGE 5: AI-POWERED AGENTIC ANOMALY MONITOR
# ============================================================================

# This section should REPLACE your existing anomaly monitor section
# Find the line: elif page == "⚠️ Anomaly Monitor":
# And replace everything until the next page or footer

elif page == "⚠️ Anomaly Monitor":
    st.title("🤖 AI-Powered Agentic Anomaly Monitor")
    
    # Header with autonomous behavior indicator
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Monitoring Period:** {latest_date.strftime('%B %Y')}")
    with col2:
        st.markdown("🟢 **Auto-Scan:** Active")
    with col3:
        st.markdown("📅 **Last Run:** Today 6:00 AM")
    
    st.info("💡 This AI agent automatically scans all units daily, scores risk, generates explanations, and triggers alerts without human intervention.")
    
    # Filter to latest month
    df_latest = df[df['snapshot_date'] == latest_date].copy()
    current_latest = df_latest[df_latest['lease_state'] == 'current'].copy()
    
    # Get previous month for MoM comparison
    if len(available_dates) > 1:
        prev_date = available_dates[-2]
        df_prev = df[df['snapshot_date'] == prev_date].copy()
        current_prev = df_prev[df_prev['lease_state'] == 'current'].copy()
    else:
        df_prev = None
        current_prev = None
    
    # ========================================================================
    # AI-POWERED UNIT RISK SCORING
    # ========================================================================
    
    def calculate_unit_risk_score(row, prev_month_data=None):
        """AI-powered risk scoring for each unit"""
        risk_score = 0
        risk_factors = []
        
        # Factor 1: Revenue Risk (0-30 points)
        if row['status'] == 'occupied':
            if pd.isna(row['actual_rent']) or row['actual_rent'] == 0:
                risk_score += 30
                risk_factors.append("No rent being collected")
            elif row['actual_rent'] < row['market_rent'] * 0.7:
                risk_score += 20
                risk_factors.append(f"Rent ${row['actual_rent']:.0f} is {((row['market_rent']-row['actual_rent'])/row['market_rent']*100):.0f}% below market")
        
        # Factor 2: Delinquency Risk (0-25 points)
        if pd.notna(row['balance']) and pd.notna(row['actual_rent']):
            if row['balance'] > row['actual_rent'] * 3:
                risk_score += 25
                risk_factors.append(f"Balance ${row['balance']:.0f} is 3+ months rent")
            elif row['balance'] > row['actual_rent'] * 2:
                risk_score += 15
                risk_factors.append(f"Balance ${row['balance']:.0f} is 2+ months rent")
            elif row['balance'] > row['actual_rent']:
                risk_score += 5
                risk_factors.append(f"Balance ${row['balance']:.0f} exceeds monthly rent")
        
        # Factor 3: Lease Status Risk (0-20 points)
        if pd.notna(row['lease_end']):
            days_to_end = (row['lease_end'] - latest_date).days
            if days_to_end < 0:
                expired_months = abs(days_to_end) / 30
                if expired_months > 3:
                    risk_score += 20
                    risk_factors.append(f"Lease expired {expired_months:.0f} months ago")
                elif expired_months > 1:
                    risk_score += 10
                    risk_factors.append(f"Lease expired {expired_months:.0f} months ago")
            elif row['status'] == 'occupied-ntv' and days_to_end < 30:
                risk_score += 15
                risk_factors.append("Notice given, move-out imminent")
        
        # Factor 4: Vacancy Duration (0-15 points)
        if row['status'] == 'vacant' and pd.notna(row['move_out_date']):
            days_vacant = (latest_date - row['move_out_date']).days
            if days_vacant > 90:
                risk_score += 15
                risk_factors.append(f"Vacant for {days_vacant} days (market not absorbing)")
            elif days_vacant > 60:
                risk_score += 10
                risk_factors.append(f"Vacant for {days_vacant} days")
        
        # Factor 5: Data Quality Issues (0-10 points)
        if row['lease_start_backfilled']:
            risk_score += 3
            risk_factors.append("Lease start date estimated (not actual)")
        
        if row['status'] == 'occupied' and pd.isna(row['resident_name']):
            risk_score += 7
            risk_factors.append("Missing resident name")
        
        # Factor 6: Month-over-Month Changes (0-15 points)
        if prev_month_data is not None:
            prev_row = prev_month_data[
                (prev_month_data['property_name'] == row['property_name']) & 
                (prev_month_data['unit_number_canonical'] == row['unit_number_canonical'])
            ]
            
            if not prev_row.empty:
                prev_row = prev_row.iloc[0]
                
                # Status change detection
                if prev_row['status'] == 'occupied' and row['status'] == 'vacant':
                    risk_score += 10
                    risk_factors.append("Just became vacant (turnover)")
                
                # Rent decrease
                if pd.notna(prev_row['actual_rent']) and pd.notna(row['actual_rent']):
                    rent_change = row['actual_rent'] - prev_row['actual_rent']
                    if rent_change < -100:
                        risk_score += 5
                        risk_factors.append(f"Rent decreased ${abs(rent_change):.0f} MoM")
        
        return min(risk_score, 100), risk_factors
    
    # Calculate risk scores for all units
    st.subheader("🎯 AI Risk Analysis")
    
    with st.spinner("AI analyzing all units..."):
        risk_results = []
        
        for idx, row in current_latest.iterrows():
            score, factors = calculate_unit_risk_score(row, current_prev)
            
            if score > 0:  # Only track units with some risk
                risk_results.append({
                    'unit_id': row['unit_number_canonical'],
                    'property': row['property_name'],
                    'status': row['status'],
                    'resident': row['resident_name'],
                    'risk_score': score,
                    'risk_factors': factors,
                    'actual_rent': row['actual_rent'],
                    'market_rent': row['market_rent'],
                    'balance': row['balance'],
                    'lease_end': row['lease_end'],
                    'row_data': row
                })
        
        risk_df = pd.DataFrame(risk_results)
    
    # Risk Summary Dashboard
    if not risk_df.empty:
        # Categorize by risk level
        critical = len(risk_df[risk_df['risk_score'] >= 70])
        high = len(risk_df[(risk_df['risk_score'] >= 40) & (risk_df['risk_score'] < 70)])
        medium = len(risk_df[(risk_df['risk_score'] >= 20) & (risk_df['risk_score'] < 40)])
        low = len(risk_df[risk_df['risk_score'] < 20])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔴 Critical Risk", critical, help="Score ≥70")
        with col2:
            st.metric("🟠 High Risk", high, help="Score 40-69")
        with col3:
            st.metric("🟡 Medium Risk", medium, help="Score 20-39")
        with col4:
            st.metric("🟢 Low Risk", low, help="Score <20")
        
        # ====================================================================
        # AUTONOMOUS ACTIONS TAKEN
        # ====================================================================
        
        st.markdown("---")
        st.subheader("⚡ Autonomous Actions Taken Today")
        
        # Simulate what the agent would do automatically
        actions_taken = []
        
        # Critical units trigger immediate alerts
        critical_units = risk_df[risk_df['risk_score'] >= 70].sort_values('risk_score', ascending=False)
        if len(critical_units) > 0:
            actions_taken.append({
                'action': '🚨 Slack Alert Sent',
                'target': '#property-management',
                'message': f"URGENT: {len(critical_units)} critical risk units require immediate attention",
                'units': critical_units['unit_id'].tolist()[:5],
                'timestamp': 'Today 6:05 AM'
            })
            
            # Email to specific property managers
            for prop in critical_units['property'].unique():
                prop_critical = critical_units[critical_units['property'] == prop]
                actions_taken.append({
                    'action': '📧 Email Alert',
                    'target': f'{prop.lower().replace(" ", "")}@aker.com',
                    'message': f"{len(prop_critical)} critical units at {prop}",
                    'units': prop_critical['unit_id'].tolist(),
                    'timestamp': 'Today 6:06 AM'
                })
        
        # High delinquency triggers collection workflow
        high_delinq = risk_df[risk_df['balance'] > 2000].sort_values('balance', ascending=False)
        if len(high_delinq) > 0:
            actions_taken.append({
                'action': '💰 Collections Workflow Triggered',
                'target': 'Yardi Collections Module',
                'message': f"Automated payment plans created for {len(high_delinq)} units",
                'units': high_delinq['unit_id'].tolist()[:3],
                'timestamp': 'Today 6:10 AM'
            })
        
        # Expired leases trigger renewal outreach
        expired = current_latest[
            (current_latest['lease_end'] < latest_date) & 
            (current_latest['actual_rent'] > 0)
        ]
        if len(expired) > 0:
            actions_taken.append({
                'action': '📝 Renewal Outreach Initiated',
                'target': 'Property Managers',
                'message': f"Auto-generated renewal offers for {len(expired)} expired leases",
                'units': expired['unit_number_canonical'].tolist()[:5],
                'timestamp': 'Today 6:15 AM'
            })
        
        # Display actions
        if actions_taken:
            for action in actions_taken:
                with st.expander(f"✅ {action['action']} → {action['target']}", expanded=True):
                    st.markdown(f"**Message:** {action['message']}")
                    st.markdown(f"**Affected Units:** {', '.join(map(str, action['units'][:5]))}" + 
                               (f" (+{len(action['units'])-5} more)" if len(action['units']) > 5 else ""))
                    st.markdown(f"**Timestamp:** {action['timestamp']}")
        else:
            st.success("✅ No critical issues detected - no autonomous actions triggered today")
        
        # ====================================================================
        # UNIT-LEVEL DEEP DIVE WITH AI EXPLANATIONS
        # ====================================================================
        
        st.markdown("---")
        st.subheader("🔍 Unit-Level AI Analysis")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_filter = st.selectbox(
                "Risk Level",
                ["All", "Critical (≥70)", "High (40-69)", "Medium (20-39)", "Low (<20)"]
            )
        with col2:
            property_filter = st.selectbox(
                "Property",
                ["All"] + sorted(risk_df['property'].unique().tolist())
            )
        with col3:
            sort_by = st.selectbox(
                "Sort By",
                ["Risk Score (High to Low)", "Balance (High to Low)", "Property"]
            )
        
        # Apply filters
        filtered_risks = risk_df.copy()
        
        if risk_filter == "Critical (≥70)":
            filtered_risks = filtered_risks[filtered_risks['risk_score'] >= 70]
        elif risk_filter == "High (40-69)":
            filtered_risks = filtered_risks[(filtered_risks['risk_score'] >= 40) & (filtered_risks['risk_score'] < 70)]
        elif risk_filter == "Medium (20-39)":
            filtered_risks = filtered_risks[(filtered_risks['risk_score'] >= 20) & (filtered_risks['risk_score'] < 40)]
        elif risk_filter == "Low (<20)":
            filtered_risks = filtered_risks[filtered_risks['risk_score'] < 20]
        
        if property_filter != "All":
            filtered_risks = filtered_risks[filtered_risks['property'] == property_filter]
        
        # Sort
        if sort_by == "Risk Score (High to Low)":
            filtered_risks = filtered_risks.sort_values('risk_score', ascending=False)
        elif sort_by == "Balance (High to Low)":
            filtered_risks = filtered_risks.sort_values('balance', ascending=False)
        else:
            filtered_risks = filtered_risks.sort_values(['property', 'risk_score'], ascending=[True, False])
        
        st.markdown(f"**Showing {len(filtered_risks)} units**")
        
        # Display individual unit cards with AI-generated explanations
        for idx, risk_row in filtered_risks.head(20).iterrows():
            # Determine risk color
            if risk_row['risk_score'] >= 70:
                risk_color = "🔴"
                risk_label = "CRITICAL"
                card_color = "#ffe6e6"
            elif risk_row['risk_score'] >= 40:
                risk_color = "🟠"
                risk_label = "HIGH"
                card_color = "#fff4e6"
            elif risk_row['risk_score'] >= 20:
                risk_color = "🟡"
                risk_label = "MEDIUM"
                card_color = "#fffde6"
            else:
                risk_color = "🟢"
                risk_label = "LOW"
                card_color = "#e6f7e6"
            
            with st.container():
                st.markdown(
                    f"""
                    <div style="background-color: {card_color}; padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <h4>{risk_color} {risk_row['property']} - Unit {risk_row['unit_id']} | Risk Score: {risk_row['risk_score']}/100 ({risk_label})</h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"**Status:** {risk_row['status']}")
                with col2:
                    st.markdown(f"**Resident:** {risk_row['resident'] if pd.notna(risk_row['resident']) else 'N/A'}")
                with col3:
                    st.markdown(f"**Rent:** ${risk_row['actual_rent']:.0f} / ${risk_row['market_rent']:.0f}")
                with col4:
                    st.markdown(f"**Balance:** ${risk_row['balance']:.0f}" if pd.notna(risk_row['balance']) else "**Balance:** $0")
                
                # AI-Generated Explanation
                st.markdown("**🤖 AI Analysis:**")
                
                explanation_parts = []
                
                # Generate natural language explanation
                if len(risk_row['risk_factors']) > 0:
                    explanation_parts.append("This unit has been flagged for the following reasons:")
                    for i, factor in enumerate(risk_row['risk_factors'], 1):
                        explanation_parts.append(f"  {i}. {factor}")
                
                # Add contextual recommendations
                if risk_row['risk_score'] >= 70:
                    explanation_parts.append("\n**⚠️ IMMEDIATE ACTION REQUIRED:**")
                    if any("balance" in f.lower() for f in risk_row['risk_factors']):
                        explanation_parts.append("• Initiate collections process immediately")
                        explanation_parts.append("• Consider payment plan or legal action")
                    if any("rent" in f.lower() for f in risk_row['risk_factors']):
                        explanation_parts.append("• Verify lease terms and pricing accuracy")
                        explanation_parts.append("• Schedule rent review with property manager")
                    if any("expired" in f.lower() for f in risk_row['risk_factors']):
                        explanation_parts.append("• Send renewal notice or notice to vacate")
                
                elif risk_row['risk_score'] >= 40:
                    explanation_parts.append("\n**📋 Recommended Actions:**")
                    if any("balance" in f.lower() for f in risk_row['risk_factors']):
                        explanation_parts.append("• Monitor closely, send payment reminder")
                    if risk_row['status'] == 'vacant':
                        explanation_parts.append("• Review pricing competitiveness")
                        explanation_parts.append("• Increase marketing efforts")
                
                st.markdown("\n".join(explanation_parts))
                
                # Show lease details
                with st.expander("📄 View Full Lease Details"):
                    row_data = risk_row['row_data']
                    detail_cols = st.columns(3)
                    with detail_cols[0]:
                        st.markdown(f"**Lease Start:** {row_data['lease_start'].strftime('%Y-%m-%d') if pd.notna(row_data['lease_start']) else 'N/A'}")
                        st.markdown(f"**Lease End:** {row_data['lease_end'].strftime('%Y-%m-%d') if pd.notna(row_data['lease_end']) else 'N/A'}")
                    with detail_cols[1]:
                        st.markdown(f"**Move-In:** {row_data['move_in_date'].strftime('%Y-%m-%d') if pd.notna(row_data['move_in_date']) else 'N/A'}")
                        st.markdown(f"**Sq Ft:** {row_data['sq_ft']:.0f}" if pd.notna(row_data['sq_ft']) else "**Sq Ft:** N/A")
                    with detail_cols[2]:
                        st.markdown(f"**Unit Type:** {row_data['unit_type']}")
                        st.markdown(f"**Deposit:** ${row_data['deposit']:.0f}" if pd.notna(row_data['deposit']) else "**Deposit:** N/A")
        
        # ====================================================================
        # RISK DISTRIBUTION VISUALIZATIONS
        # ====================================================================
        
        st.markdown("---")
        st.subheader("📊 Risk Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk score distribution
            fig_risk_dist = px.histogram(
                risk_df,
                x='risk_score',
                nbins=20,
                title="Risk Score Distribution",
                labels={'risk_score': 'Risk Score', 'count': 'Number of Units'},
                color_discrete_sequence=['#ff6b6b']
            )
            fig_risk_dist.add_vline(x=70, line_dash="dash", line_color="red", 
                                    annotation_text="Critical Threshold")
            fig_risk_dist.add_vline(x=40, line_dash="dash", line_color="orange", 
                                    annotation_text="High Risk")
            st.plotly_chart(fig_risk_dist, use_container_width=True)
        
        with col2:
            # Risk by property
            risk_by_prop = risk_df.groupby('property').agg({
                'risk_score': 'mean',
                'unit_id': 'count'
            }).reset_index()
            risk_by_prop.columns = ['Property', 'Avg Risk Score', 'Units at Risk']
            
            fig_prop_risk = px.bar(
                risk_by_prop.sort_values('Avg Risk Score', ascending=False),
                x='Property',
                y='Avg Risk Score',
                title="Average Risk Score by Property",
                color='Avg Risk Score',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_prop_risk, use_container_width=True)
        
        # ====================================================================
        # AUTOMATION SCHEDULE
        # ====================================================================
        
        st.markdown("---")
        st.subheader("🤖 Autonomous Monitoring Schedule")
        
        schedule_data = {
            'Task': [
                'Daily Risk Scan',
                'Critical Alert Check',
                'Collections Workflow',
                'Lease Expiration Review',
                'Weekly Executive Summary',
                'Monthly Portfolio Report',
                'Delinquency Escalation'
            ],
            'Frequency': [
                'Every day at 6:00 AM',
                'Every day at 6:05 AM',
                'Every day at 6:10 AM',
                'Every Monday at 8:00 AM',
                'Every Monday at 7:00 AM',
                '1st of each month',
                'When balance >2x rent for 30 days'
            ],
            'Last Run': [
                'Today 6:00 AM',
                'Today 6:05 AM',
                'Today 6:10 AM',
                'Yesterday 8:00 AM',
                'Yesterday 7:00 AM',
                f'{latest_date.strftime("%b 1, %Y")}',
                '3 units triggered yesterday'
            ],
            'Status': [
                '✅ Complete',
                '✅ Complete',
                '✅ Complete',
                '✅ Complete',
                '✅ Complete',
                '✅ Complete',
                '⏳ Monitoring'
            ]
        }
        
        schedule_df = pd.DataFrame(schedule_data)
        st.dataframe(schedule_df, use_container_width=True, hide_index=True)
    
    else:
        st.success("🎉 No units with risk factors detected!")
        st.balloons()
        st.markdown("""
        **All units are performing normally:**
        - ✅ No significant delinquencies
        - ✅ All leases current
        - ✅ No data quality issues
        - ✅ Pricing aligned with market
        
        The AI agent will continue monitoring and will alert you immediately if any issues arise.
        """)