"""
Production-Ready Sales & Quotes Dashboard
A scalable Streamlit application that handles CSV data analysis without breaking under load.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import re
import json
import os
# Removed matplotlib and seaborn imports - using Plotly for all charts
warnings.filterwarnings('ignore')


# Configure page - MUST be first Streamlit command
st.set_page_config(
    page_title="Sales & Quotes Dashboard",
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_data_path():
    """Find the data directory across different platforms"""
    # Primary path - the exact location specified by user
    primary_path = Path.home() / "Dropbox" / "python" / "streamlit_dashboard" / "src" / "data"
    
    possible_paths = [
        primary_path,  # User's specified path
        Path.cwd() / "data",  # Relative data directory
        Path("C:\\Users\\manzalone\\Dropbox\\python\\streamlit_dashboard\\src\\data"),  # Absolute path
        Path.cwd()  # Fallback to current directory
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Default to the primary path even if it doesn't exist (will show proper error)
    return primary_path

DATA_DIR = get_data_path()
SALES_CSV_PATH = str(DATA_DIR / "sales.csv")
QUOTES_CSV_PATH = str(DATA_DIR / "quotes.csv")

# Custom CSS - optimized for dark theme and quota attainment
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1f77b4;
        text-align: center;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
    /* Remove hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Quota attainment styling - optimized for dark theme */
    .quota-table {
        font-family: 'Arial', sans-serif;
        color: white;
    }
    .positive-delta {
        color: #40e540;
        font-weight: bold;
    }
    .negative-delta {
        color: #ff6b6b;
        font-weight: bold;
    }
    .neutral-delta {
        color: #ffd93d;
        font-weight: bold;
    }
    .trend-up {
        color: #40e540;
    }
    .trend-down {
        color: #ff6b6b;
    }
    .header-row {
        background-color: #4a5568;
        font-weight: bold;
        color: white;
    }
    .quarter-total-row {
        background-color: #2d5a37;
        font-weight: bold;
        color: white;
    }
    .final-total-row {
        background-color: #5a2d2d;
        font-weight: bold;
        color: white;
    }
    /* Improve visibility for dark theme */
    .stDataFrame {
        background-color: #1e1e1e;
    }
</style>
""", unsafe_allow_html=True)

class DataManager:
    """Handles data loading and caching with production-ready error handling"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_sales_data(uploaded_file=None) -> Optional[pd.DataFrame]:
        """Load and cache sales data"""
        try:
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
            else:
                # Return None if no file is uploaded - application starts blank
                return None
            # Clean and standardize data
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
            df['Year'] = df['Year'].astype(int)
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            
            # Handle Motivating Factor column - rename and clean
            if 'Motivating Factor' in df.columns:
                # Rename column to use underscore for consistency
                df['Motivating_Factor'] = df['Motivating Factor']
                # Drop the original column with space
                df = df.drop('Motivating Factor', axis=1)
                # Clean up the renamed column - remove empty values
                df['Motivating_Factor'] = df['Motivating_Factor'].fillna('Unknown')
                df['Motivating_Factor'] = df['Motivating_Factor'].replace('', 'Unknown')
            
            # Clean coordinate data - handle #N/A and other invalid values
            if 'lat' in df.columns:
                df['lat'] = df['lat'].replace(['#N/A', '#DIV/0!', '#VALUE!', 'N/A'], pd.NA)
                df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            
            if 'lng' in df.columns:
                df['lng'] = df['lng'].replace(['#N/A', '#DIV/0!', '#VALUE!', 'N/A'], pd.NA)
                df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
            
            # Remove rows with zero sales or empty account names
            df = df[(df['Sales'] > 0) & (df['Account'].notna()) & (df['Account'] != '')]
            return df
        except Exception as e:
            st.error(f"Error loading sales data: {str(e)}")
            return None
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_quotes_data(uploaded_file=None) -> Optional[pd.DataFrame]:
        """Load and cache quotes data"""
        try:
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
            else:
                # Return None if no file is uploaded - application starts blank
                return None
            # Clean and standardize data
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Quote_Total'] = pd.to_numeric(df['Quote_Total'], errors='coerce')
            df['Year'] = df['Year'].astype(int)
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            
            # Clean coordinate data - handle #N/A and other invalid values
            if 'lat' in df.columns:
                df['lat'] = df['lat'].replace(['#N/A', '#DIV/0!', '#VALUE!', 'N/A'], pd.NA)
                df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            
            if 'lng' in df.columns:
                df['lng'] = df['lng'].replace(['#N/A', '#DIV/0!', '#VALUE!', 'N/A'], pd.NA)
                df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
            
            # Remove rows with zero quotes or empty account names
            df = df[(df['Quote_Total'] > 0) & (df['Account'].notna()) & (df['Account'] != '')]
            return df
        except Exception as e:
            st.error(f"Error loading quotes data: {str(e)}")
            return None

class FilterManager:
    """Handles sidebar filters with session state management"""
    
    @staticmethod
    def create_sidebar_filters(df: pd.DataFrame, prefix: str) -> Dict:
        """Create sidebar filters for the given dataframe"""
        
        st.sidebar.markdown(f"### 🎛️ {prefix} Filters")
        
        # Year filter with Select All button
        available_years = sorted(df['Year'].dropna().unique())
        
        col1, col2, col3 = st.sidebar.columns([2, 1, 1])
        with col1:
            st.markdown("📅 **Years**")
        with col2:
            if st.button("All", key=f"{prefix}_years_all", help="Select all years"):
                st.session_state[f"{prefix}_years"] = available_years
        with col3:
            if st.button("Clear", key=f"{prefix}_years_clear", help="Clear all years"):
                st.session_state[f"{prefix}_years"] = []
        
        selected_years = st.sidebar.multiselect(
            "Select Years",
            options=available_years,
            default=available_years,  # Select all years by default
            key=f"{prefix}_years",
            label_visibility="collapsed"
        )
        
        # Market filter with Select All button
        available_markets = sorted(df['Market'].dropna().unique())
        
        col1, col2, col3 = st.sidebar.columns([2, 1, 1])
        with col1:
            st.markdown("🏢 **Markets**")
        with col2:
            if st.button("All", key=f"{prefix}_markets_all", help="Select all markets"):
                st.session_state[f"{prefix}_markets"] = available_markets
        with col3:
            if st.button("Clear", key=f"{prefix}_markets_clear", help="Clear all markets"):
                st.session_state[f"{prefix}_markets"] = []
        
        selected_markets = st.sidebar.multiselect(
            "Select Markets",
            options=available_markets,
            default=available_markets,  # Select all markets by default
            key=f"{prefix}_markets",
            label_visibility="collapsed"
        )
        
        # Application filter with Select All button
        available_applications = sorted(df['Application'].dropna().unique())
        
        col1, col2, col3 = st.sidebar.columns([2, 1, 1])
        with col1:
            st.markdown("🔧 **Applications**")
        with col2:
            if st.button("All", key=f"{prefix}_applications_all", help="Select all applications"):
                st.session_state[f"{prefix}_applications"] = available_applications
        with col3:
            if st.button("Clear", key=f"{prefix}_applications_clear", help="Clear all applications"):
                st.session_state[f"{prefix}_applications"] = []
        
        selected_applications = st.sidebar.multiselect(
            "Select Applications", 
            options=available_applications,
            default=available_applications,  # Select all applications by default
            key=f"{prefix}_applications",
            label_visibility="collapsed"
        )
        
        # Motivating Factor filter (only for sales data) with Select All button
        selected_motivating_factors = []
        if 'Motivating_Factor' in df.columns:
            available_motivating_factors = sorted(df['Motivating_Factor'].dropna().unique())
            
            col1, col2, col3 = st.sidebar.columns([2, 1, 1])
            with col1:
                st.markdown("🎯 **Motivating Factors**")
            with col2:
                if st.button("All", key=f"{prefix}_motivating_factors_all", help="Select all motivating factors"):
                    st.session_state[f"{prefix}_motivating_factors"] = available_motivating_factors
            with col3:
                if st.button("Clear", key=f"{prefix}_motivating_factors_clear", help="Clear all motivating factors"):
                    st.session_state[f"{prefix}_motivating_factors"] = []
            
            selected_motivating_factors = st.sidebar.multiselect(
                "Select Motivating Factors",
                options=available_motivating_factors,
                default=available_motivating_factors,  # Select all factors by default
                key=f"{prefix}_motivating_factors",
                label_visibility="collapsed"
            )
        
        # Customer filter (Yes/No) with Select All button
        customer_options = ['Yes', 'No']
        
        col1, col2, col3 = st.sidebar.columns([2, 1, 1])
        with col1:
            st.markdown("👤 **Customer Types**")
        with col2:
            if st.button("All", key=f"{prefix}_customers_all", help="Select all customer types"):
                st.session_state[f"{prefix}_customers"] = customer_options
        with col3:
            if st.button("Clear", key=f"{prefix}_customers_clear", help="Clear all customer types"):
                st.session_state[f"{prefix}_customers"] = []
        
        selected_customers = st.sidebar.multiselect(
            "Select Customer Types",
            options=customer_options,
            default=customer_options,  # Select all customer types by default
            key=f"{prefix}_customers",
            label_visibility="collapsed"
        )
        
        # Channel filter with Select All button
        available_channels = sorted(df['Channel'].dropna().unique())
        
        col1, col2, col3 = st.sidebar.columns([2, 1, 1])
        with col1:
            st.markdown("📡 **Channels**")
        with col2:
            if st.button("All", key=f"{prefix}_channels_all", help="Select all channels"):
                st.session_state[f"{prefix}_channels"] = available_channels
        with col3:
            if st.button("Clear", key=f"{prefix}_channels_clear", help="Clear all channels"):
                st.session_state[f"{prefix}_channels"] = []
        
        selected_channels = st.sidebar.multiselect(
            "Select Channels",
            options=available_channels,
            default=available_channels,  # Select all channels by default
            key=f"{prefix}_channels",
            label_visibility="collapsed"
        )
        
        # State filter (if available) with Select All button
        if 'State' in df.columns:
            available_states = sorted(df['State'].dropna().unique())
            
            col1, col2, col3 = st.sidebar.columns([2, 1, 1])
            with col1:
                st.markdown("🗺️ **States**")
            with col2:
                if st.button("All", key=f"{prefix}_states_all", help="Select all states"):
                    st.session_state[f"{prefix}_states"] = available_states
            with col3:
                if st.button("Clear", key=f"{prefix}_states_clear", help="Clear all states"):
                    st.session_state[f"{prefix}_states"] = []
            
            selected_states = st.sidebar.multiselect(
                "Select States",
                options=available_states,
                default=available_states,  # Select all states by default
                key=f"{prefix}_states",
                label_visibility="collapsed"
            )
        else:
            selected_states = []
        
        st.sidebar.markdown("---")
        
        return {
            'years': selected_years,
            'markets': selected_markets,
            'applications': selected_applications,
            'customers': selected_customers,
            'channels': selected_channels,
            'states': selected_states,
            'motivating_factors': selected_motivating_factors
        }
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to dataframe"""
        filtered_df = df.copy()
        
        if filters['years']:
            filtered_df = filtered_df[filtered_df['Year'].isin(filters['years'])]
        
        if filters['markets']:
            filtered_df = filtered_df[filtered_df['Market'].isin(filters['markets'])]
            
        if filters['applications']:
            filtered_df = filtered_df[filtered_df['Application'].isin(filters['applications'])]
            
        if filters['customers']:
            filtered_df = filtered_df[filtered_df['Customer'].isin(filters['customers'])]
            
        if filters['channels']:
            filtered_df = filtered_df[filtered_df['Channel'].isin(filters['channels'])]
        
        if filters['states'] and 'State' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['State'].isin(filters['states'])]
        
        if filters['motivating_factors'] and 'Motivating_Factor' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Motivating_Factor'].isin(filters['motivating_factors'])]
        
        return filtered_df

class ChartManager:
    """Handles chart creation with consistent styling"""
    
    @staticmethod
    def diagnose_geographic_data(df: pd.DataFrame) -> Dict:
        """Diagnose geographic data availability and quality"""
        diagnosis = {
            'has_coordinates': False,
            'coordinate_columns': [],
            'total_records': len(df),
            'records_with_state': 0,
            'records_with_coordinates': 0,
            'coordinate_quality': 'unknown',
            'sample_coordinates': [],
            'issues': []
        }
        
        # Check for coordinate columns
        coord_patterns = ['lat', 'lng', 'latitude', 'longitude', 'lon']
        for col in df.columns:
            if col.lower() in coord_patterns:
                diagnosis['coordinate_columns'].append(col)
        
        diagnosis['has_coordinates'] = len(diagnosis['coordinate_columns']) >= 2
        
        # Check state data
        if 'State' in df.columns:
            diagnosis['records_with_state'] = len(df[df['State'].notna()])
        
        # If we have coordinates, check their quality
        if diagnosis['has_coordinates']:
            lat_col = lng_col = None
            for col in diagnosis['coordinate_columns']:
                if col.lower() in ['lat', 'latitude'] and lat_col is None:
                    lat_col = col
                elif col.lower() in ['lng', 'longitude', 'lon'] and lng_col is None:
                    lng_col = col
            
            if lat_col and lng_col:
                # Convert to numeric and check validity
                lat_numeric = pd.to_numeric(df[lat_col], errors='coerce')
                lng_numeric = pd.to_numeric(df[lng_col], errors='coerce')
                
                valid_coords = (
                    (lat_numeric.notna()) & (lng_numeric.notna()) &
                    (lat_numeric >= 24) & (lat_numeric <= 50) &
                    (lng_numeric >= -125) & (lng_numeric <= -66)
                )
                
                diagnosis['records_with_coordinates'] = valid_coords.sum()
                
                if diagnosis['records_with_coordinates'] > 0:
                    diagnosis['coordinate_quality'] = 'good'
                    # Get sample coordinates
                    sample_df = df[valid_coords].head(5)
                    diagnosis['sample_coordinates'] = [
                        (row[lat_col], row[lng_col], row.get('Account', 'Unknown'))
                        for _, row in sample_df.iterrows()
                    ]
                else:
                    diagnosis['coordinate_quality'] = 'poor'
                    diagnosis['issues'].append('No valid US coordinates found')
        else:
            diagnosis['issues'].append('No coordinate columns found')
        
        return diagnosis
    
    @staticmethod
    def create_sales_charts(df: pd.DataFrame, filters: Dict) -> Dict:
        """Create sales-specific charts"""
        charts = {}
        
        if not df.empty:
            # Sales by Application
            app_sales = df.groupby('Application')['Sales'].sum().reset_index()
            app_sales = app_sales.sort_values('Sales', ascending=False)
            
            # Pie chart for percentage distribution
            charts['sales_by_application_pie'] = px.pie(
                app_sales,
                values='Sales',
                names='Application',
                title='Sales Distribution by Application (%)'
            )
            
            # Bar chart for absolute values and growth
            charts['sales_by_application_bar'] = px.bar(
                app_sales,
                x='Application',
                y='Sales',
                title='Sales by Application ($)',
                color='Sales',
                color_continuous_scale='Blues'
            )
            charts['sales_by_application_bar'].update_layout(
                xaxis_title="Application",
                yaxis_title="Total Sales ($)",
                xaxis={'tickangle': 45}
            )
            
            # Sales by Year with color-coded trend line
            year_sales = df.groupby('Year')['Sales'].sum().reset_index()
            
            # Create the line chart using plotly graph objects for more control
            charts['sales_by_year'] = go.Figure()
            
            # Add the main sales line
            charts['sales_by_year'].add_trace(go.Scatter(
                x=year_sales['Year'],
                y=year_sales['Sales'],
                mode='lines+markers',
                name='Sales',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8, color='#1f77b4')
            ))
            
            # Calculate and add trend line with color coding
            if len(year_sales) > 1:
                # Calculate linear trend
                x_vals = range(len(year_sales))
                trend_coeffs = np.polyfit(x_vals, year_sales['Sales'], 1)
                trend_line = np.polyval(trend_coeffs, x_vals)
                
                # Determine trend color based on slope
                slope = trend_coeffs[0]
                if slope > 100000:  # Positive trend (threshold for "significant" increase)
                    trend_color = 'green'
                elif slope < -100000:  # Negative trend (threshold for "significant" decrease)
                    trend_color = 'red'
                else:  # Flat trend
                    trend_color = 'gold'
                
                # Add trend line
                charts['sales_by_year'].add_trace(go.Scatter(
                    x=year_sales['Year'],
                    y=trend_line,
                    mode='lines',
                    name='Trend',
                    line=dict(color=trend_color, width=2, dash='dash'),
                    showlegend=False  # Hide from legend as requested
                ))
            
            charts['sales_by_year'].update_layout(
                title='Sales Trend by Year with Trend Analysis',
                xaxis_title="Year",
                yaxis_title="Total Sales ($)"
            )
            
            # Sales Performance vs Goals by Year
            # Define goals for each year (you can modify these values)
            goals = {
                2019: 2250000,
                2020: 2250000,
                2021: 2250000,
                2022: 2250000,
                2023: 2250000,
                2024: 3000000,
                2025: 3666666
            }
            
            # Create performance data
            performance_data = year_sales.copy()
            performance_data['Goal'] = performance_data['Year'].map(goals)
            performance_data['Delta'] = performance_data['Sales'] - performance_data['Goal']
            performance_data['%_to_Goal'] = (performance_data['Sales'] / performance_data['Goal'] * 100).round(2)
            
            # Create the performance chart using plotly graph objects
            charts['sales_performance_vs_goals'] = go.Figure()
            
            # Add Total Sales bars
            charts['sales_performance_vs_goals'].add_trace(go.Bar(
                x=performance_data['Year'],
                y=performance_data['Sales'],
                name='Total Sales',
                marker_color='#1f77b4',
                yaxis='y'
            ))
            
            # Add Goal bars
            charts['sales_performance_vs_goals'].add_trace(go.Bar(
                x=performance_data['Year'],
                y=performance_data['Goal'],
                name='Goal',
                marker_color='#d62728',
                yaxis='y'
            ))
            
            # Add Delta bars (positive and negative)
            positive_delta = performance_data[performance_data['Delta'] >= 0].copy()
            negative_delta = performance_data[performance_data['Delta'] < 0].copy()
            
            if not positive_delta.empty:
                charts['sales_performance_vs_goals'].add_trace(go.Bar(
                    x=positive_delta['Year'],
                    y=positive_delta['Delta'],
                    name='Delta (Positive)',
                    marker_color='#ff7f0e',
                    yaxis='y'
                ))
            
            if not negative_delta.empty:
                charts['sales_performance_vs_goals'].add_trace(go.Bar(
                    x=negative_delta['Year'],
                    y=negative_delta['Delta'],
                    name='Delta (Negative)',
                    marker_color='#ff7f0e',
                    yaxis='y'
                ))
            
            charts['sales_performance_vs_goals'].update_layout(
                title='Total Sales Performance by Year',
                xaxis_title="Year",
                yaxis_title="Amount ($)",
                barmode='group',
                yaxis=dict(
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=2
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Store performance data for table display
            charts['performance_data'] = performance_data
            
            # Create Application Sales Heatmap
            heatmap_fig = ChartManager.create_application_sales_heatmap(df)
            if heatmap_fig is not None:
                charts['application_sales_heatmap'] = heatmap_fig
            
            # Sales by Market
            market_sales = df.groupby('Market')['Sales'].sum().reset_index()
            market_sales = market_sales.sort_values('Sales', ascending=False)
            
            # Pie chart for percentage distribution
            charts['sales_by_market_pie'] = px.pie(
                market_sales,
                values='Sales',
                names='Market',
                title='Sales Distribution by Market (%)'
            )
            
            # Bar chart for absolute values
            charts['sales_by_market_bar'] = px.bar(
                market_sales,
                x='Market',
                y='Sales',
                title='Sales by Market ($)',
                color='Sales',
                color_continuous_scale='Blues'
            )
            charts['sales_by_market_bar'].update_layout(
                xaxis_title="Market",
                yaxis_title="Total Sales ($)",
                xaxis={'tickangle': 45}
            )
            
            # Create Market Sales Heatmap
            market_heatmap_fig = ChartManager.create_market_sales_heatmap(df)
            if market_heatmap_fig is not None:
                charts['market_sales_heatmap'] = market_heatmap_fig
            
            # Sales by Channel
            channel_sales = df.groupby('Channel')['Sales'].sum().reset_index()
            charts['sales_by_channel'] = px.bar(
                channel_sales,
                x='Channel',
                y='Sales',
                title='Sales by Channel',
                color='Sales',
                color_continuous_scale='Greens'
            )
            
            # AV Integrator Accounts horizontal bar chart
            av_integrator_data = df[df['Channel'] == 'AV Integrator']
            if not av_integrator_data.empty:
                av_accounts = av_integrator_data.groupby('Account')['Sales'].sum().reset_index()
                av_accounts = av_accounts.sort_values('Sales', ascending=True)  # ascending=True for horizontal bar
                
                charts['av_integrator_accounts'] = px.bar(
                    av_accounts,
                    x='Sales',
                    y='Account',
                    title='AV Integrator Accounts by Sales',
                    orientation='h',
                    color='Sales',
                    color_continuous_scale='Blues',
                    text='Sales'
                )
                charts['av_integrator_accounts'].update_traces(
                    texttemplate='$%{text:,.0f}',
                    textposition='outside'
                )
                charts['av_integrator_accounts'].update_layout(
                    xaxis_title="Total Sales ($)",
                    yaxis_title="Account",
                    height=max(400, len(av_accounts) * 25),  # Dynamic height based on number of accounts
                    margin=dict(l=200)  # Add left margin for account names
                )
            
            # AV Integrator vs Direct Sales line chart by year
            channel_year_data = df[df['Channel'].isin(['AV Integrator', 'Direct'])]
            if not channel_year_data.empty:
                channel_year_sales = channel_year_data.groupby(['Year', 'Channel'])['Sales'].sum().reset_index()
                
                charts['av_vs_direct_by_year'] = go.Figure()
                
                # Add line for AV Integrator
                av_data = channel_year_sales[channel_year_sales['Channel'] == 'AV Integrator']
                if not av_data.empty:
                    charts['av_vs_direct_by_year'].add_trace(go.Scatter(
                        x=av_data['Year'],
                        y=av_data['Sales'],
                        mode='lines+markers',
                        name='AV Integrator',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=8, color='#1f77b4'),
                        hovertemplate='<b>AV Integrator</b><br>' +
                                    'Year: %{x}<br>' +
                                    'Sales: $%{y:,.0f}<extra></extra>'
                    ))
                
                # Add line for Direct Sales
                direct_data = channel_year_sales[channel_year_sales['Channel'] == 'Direct']
                if not direct_data.empty:
                    charts['av_vs_direct_by_year'].add_trace(go.Scatter(
                        x=direct_data['Year'],
                        y=direct_data['Sales'],
                        mode='lines+markers',
                        name='Direct',
                        line=dict(color='#ff7f0e', width=3),
                        marker=dict(size=8, color='#ff7f0e'),
                        hovertemplate='<b>Direct</b><br>' +
                                    'Year: %{x}<br>' +
                                    'Sales: $%{y:,.0f}<extra></extra>'
                    ))
                
                charts['av_vs_direct_by_year'].update_layout(
                    title='AV Integrator vs Direct Sales by Year',
                    xaxis_title='Year',
                    yaxis_title='Total Sales ($)',
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5,
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="rgba(0,0,0,0.2)",
                        borderwidth=1,
                        font=dict(color='black', size=12)
                    ),
                    xaxis=dict(
                        tickmode='linear',
                        dtick=1  # Show every year
                    ),
                    yaxis=dict(
                        tickformat='$,.0f'  # Format y-axis as currency
                    )
                )
            
            # Sales by Customer Type
            customer_sales = df.groupby('Customer')['Sales'].sum().reset_index()
            
            # Pie chart for percentage distribution
            charts['sales_by_customer_pie'] = px.pie(
                customer_sales,
                values='Sales',
                names='Customer',
                title='Sales Distribution by Customer Type (%)'
            )
            
            # Bar chart for absolute values
            charts['sales_by_customer_bar'] = px.bar(
                customer_sales,
                x='Customer',
                y='Sales',
                title='Sales by Customer Type ($)',
                color='Sales',
                color_continuous_scale='Greens'
            )
            charts['sales_by_customer_bar'].update_layout(
                xaxis_title="Customer Type",
                yaxis_title="Total Sales ($)"
            )
            
            # Top 10 Customers horizontal bar chart (by Account)
            top_customers = df.groupby('Account')['Sales'].sum().reset_index()
            top_customers = top_customers.sort_values('Sales', ascending=True).tail(10)  # ascending=True for horizontal bar
            
            charts['top_customers_horizontal'] = px.bar(
                top_customers,
                x='Sales',
                y='Account',
                title='Top 10 Customers by Total Sales',
                orientation='h',
                color='Sales',
                color_continuous_scale='Viridis',
                text='Sales'
            )
            charts['top_customers_horizontal'].update_traces(
                texttemplate='$%{text:,.0f}',
                textposition='outside'
            )
            charts['top_customers_horizontal'].update_layout(
                xaxis_title="Total Sales ($)",
                yaxis_title="Customer Account",
                height=400,
                margin=dict(l=200)  # Add left margin for customer names
            )
            
            # Top 10 Customers pie chart (by Account)
            top_customers_pie = df.groupby('Account')['Sales'].sum().reset_index()
            top_customers_pie = top_customers_pie.sort_values('Sales', ascending=False).head(10)
            
            charts['top_customers_pie'] = px.pie(
                top_customers_pie,
                values='Sales',
                names='Account',
                title='Top 10 Customers by Total Sales (%)',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            charts['top_customers_pie'].update_traces(
                textposition='inside', 
                textinfo='percent+label'
            )
            
            # Sales by State (if available)
            if 'State' in df.columns:
                state_sales = df.groupby('State')['Sales'].sum().reset_index()
                
                # Pie chart for percentage distribution
                charts['sales_by_state_pie'] = px.pie(
                    state_sales,
                    values='Sales',
                    names='State',
                    title='Sales Distribution by State (%)'
                )
                
                # Bar chart for absolute values
                charts['sales_by_state_bar'] = px.bar(
                    state_sales,
                    x='State',
                    y='Sales',
                    title='Sales by State ($)',
                    color='Sales',
                    color_continuous_scale='Purples'
                )
                charts['sales_by_state_bar'].update_layout(
                    xaxis_title="State",
                    yaxis_title="Total Sales ($)",
                    xaxis={'tickangle': 45}
                )
                
                # Geographic map for Northeast and Mid-Atlantic states
                northeast_states = {
                    'Maine': 'ME', 'New Hampshire': 'NH', 'Vermont': 'VT', 'Massachusetts': 'MA',
                    'Rhode Island': 'RI', 'Connecticut': 'CT', 'New York': 'NY', 'New Jersey': 'NJ',
                    'Pennsylvania': 'PA', 'Virginia': 'VA', 'West Virginia': 'WV', 'Delaware': 'DE', 
                    'Maryland': 'MD', 'Washington DC': 'DC', 'District of Columbia': 'DC', 'DC': 'DC'
                }
                
                # Create state code mapping for the data
                state_sales_map = state_sales.copy()
                # Handle both full names and abbreviations in the data
                state_sales_map['State_Code'] = state_sales_map['State'].map(lambda x: 
                    northeast_states.get(x, x) if x in northeast_states else x
                )
                
                # Filter for Northeast states only (check both full names and abbreviations)
                northeast_data = state_sales_map[
                    (state_sales_map['State'].isin(northeast_states.keys())) | 
                    (state_sales_map['State'].isin(northeast_states.values()))
                ]
                
                if not northeast_data.empty:
                    # Ensure we have proper state codes for the map
                    northeast_data['State_Code'] = northeast_data['State'].map(lambda x: 
                        northeast_states.get(x, x) if len(x) > 2 else x
                    )
                    
                    charts['northeast_map'] = px.choropleth(
                        northeast_data,
                        locations='State_Code',
                        color='Sales',
                        hover_name='State',
                        hover_data={'Sales': ':$,.0f', 'State_Code': False},
                        color_continuous_scale='Blues',
                        title='Sales by Northeast & Mid-Atlantic States with Customer Markers',
                        locationmode='USA-states'
                    )
                    
                    # Add customer markers if coordinate data is available
                    total_northeast_customers = len(df[
                        (df['State'].isin(northeast_states.keys())) | 
                        (df['State'].isin(northeast_states.values()))
                    ])
                    
                    # Add customer markers if coordinate columns exist
                    if 'lat' in df.columns and 'lng' in df.columns:
                        # Filter for Northeast and Mid-Atlantic customers
                        northeast_customers = df[
                            (df['State'].isin(northeast_states.keys())) | 
                            (df['State'].isin(northeast_states.values()))
                        ].copy()
                        
                        if not northeast_customers.empty:
                            # Use the already-cleaned coordinates from load_sales_data()
                            # Convert coordinates to numeric (coordinates should already be cleaned)
                            northeast_customers['lat_num'] = pd.to_numeric(northeast_customers['lat'], errors='coerce')
                            northeast_customers['lng_num'] = pd.to_numeric(northeast_customers['lng'], errors='coerce')
                            
                            # Filter for valid coordinates within reasonable US bounds
                            valid_coords_customers = northeast_customers[
                                (northeast_customers['lat_num'].notna()) & 
                                (northeast_customers['lng_num'].notna()) &
                                (northeast_customers['lat_num'] >= 20) & 
                                (northeast_customers['lat_num'] <= 50) &
                                (northeast_customers['lng_num'] >= -130) & 
                                (northeast_customers['lng_num'] <= -60)
                            ].copy()
                            
                            if not valid_coords_customers.empty:
                                # Group by Account and sum sales, keeping coordinates
                                customer_markers = valid_coords_customers.groupby('Account').agg({
                                    'Sales': 'sum',
                                    'lat_num': 'first',
                                    'lng_num': 'first',
                                    'State': 'first'
                                }).reset_index()
                                
                                # Categorize by sales volume and assign colors with better visibility
                                def get_marker_info(sales):
                                    if sales >= 250000:
                                        return 'High (≥$250K)', 'darkgreen', 12
                                    elif sales >= 100000:
                                        return 'Medium ($100K-$249K)', 'blue', 10
                                    else:
                                        return 'Low (<$100K)', 'red', 8
                                
                                customer_markers[['category', 'color', 'size']] = customer_markers['Sales'].apply(
                                    lambda x: pd.Series(get_marker_info(x))
                                )
                                
                                # Add markers by category
                                for category in customer_markers['category'].unique():
                                    category_data = customer_markers[customer_markers['category'] == category]
                                    if not category_data.empty:
                                        # Use dynamic sizing based on sales volume
                                        marker_size = category_data['size'].iloc[0]
                                        
                                        charts['northeast_map'].add_trace(go.Scattergeo(
                                            lat=category_data['lat_num'],
                                            lon=category_data['lng_num'],
                                            text=category_data.apply(
                                                lambda x: f"<b>{x['Account']}</b><br>Sales: ${x['Sales']:,.0f}<br>State: {x['State']}",
                                                axis=1
                                            ),
                                            mode='markers',
                                            marker=dict(
                                                size=marker_size,
                                                color=category_data['color'].iloc[0],
                                                line=dict(width=1, color='white'),
                                                opacity=0.9
                                            ),
                                            name=f'{category} ({len(category_data)} customers)',
                                            hovertemplate='%{text}<extra></extra>'
                                        ))
                                
                                # Update title with detailed mapping info
                                mapped_customers = len(customer_markers)
                                total_markers_added = sum(len(customer_markers[customer_markers['category'] == cat]) for cat in customer_markers['category'].unique())
                                
                                # Count by category for better debugging
                                high_count = len(customer_markers[customer_markers['category'] == 'High (≥$250K)'])
                                medium_count = len(customer_markers[customer_markers['category'] == 'Medium ($100K-$249K)'])
                                low_count = len(customer_markers[customer_markers['category'] == 'Low (<$100K)'])
                                
                                charts['northeast_map'].update_layout(
                                    title=f'Sales by Northeast & Mid-Atlantic States with Customer Markers<br><sub>{mapped_customers} customers mapped from {total_northeast_customers} total in region | High: {high_count}, Medium: {medium_count}, Low: {low_count}</sub>'
                                )
                            else:
                                # No valid coordinates found
                                charts['northeast_map'].update_layout(
                                    title=f'Sales by Northeast & Mid-Atlantic States<br><sub>No customers with valid coordinates found from {total_northeast_customers} total in region</sub>'
                                )
                        else:
                            # No customers in region
                            charts['northeast_map'].update_layout(
                                title='Sales by Northeast & Mid-Atlantic States<br><sub>No customers found in Northeast/Mid-Atlantic region</sub>'
                            )
                    else:
                        # No coordinate data
                        charts['northeast_map'].update_layout(
                            title='Sales by Northeast & Mid-Atlantic States<br><sub>Customer markers unavailable - no coordinate data in dataset</sub>'
                        )
                    
                    # Focus on Northeast and Mid-Atlantic region
                    charts['northeast_map'].update_layout(
                        geo=dict(
                            scope='usa',
                            projection_scale=1,
                            center=dict(lat=40.5, lon=-76),  # Center on expanded region
                            lonaxis=dict(range=[-84, -64]),   # Expanded longitude range to include all coordinates
                            lataxis=dict(range=[35, 48])      # Expanded latitude range to include all coordinates
                        ),
                        height=800,  # Larger map for better visibility
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.2,  # Move legend down a bit more
                            xanchor="center",
                            x=0.5,
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="rgba(0,0,0,0.3)",
                            borderwidth=1
                        ),
                        # Add annotations for debugging if no markers are shown
                        annotations=[
                            dict(
                                text="💡 Tip: Customer markers require valid latitude/longitude coordinates in the data",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.5, y=-0.35,
                                xanchor='center', yanchor='bottom',
                                font=dict(size=10, color="gray")
                            )
                        ]
                    )
            
            # Sales by Motivating Factor (if available)
            if 'Motivating_Factor' in df.columns:
                factor_sales = df.groupby('Motivating_Factor')['Sales'].sum().reset_index()
                
                # Pie chart for percentage distribution
                charts['sales_by_factor_pie'] = px.pie(
                    factor_sales,
                    values='Sales',
                    names='Motivating_Factor',
                    title='Sales Distribution by Motivating Factor (%)'
                )
                
                # Bar chart for absolute values
                charts['sales_by_factor_bar'] = px.bar(
                    factor_sales,
                    x='Motivating_Factor',
                    y='Sales',
                    title='Sales by Motivating Factor ($)',
                    color='Sales',
                    color_continuous_scale='Reds'
                )
                charts['sales_by_factor_bar'].update_layout(
                    xaxis_title="Motivating Factor",
                    yaxis_title="Total Sales ($)",
                    xaxis={'tickangle': 45}
                )
                
                # Top 25 Accounts by Motivating Factor horizontal bar chart
                top_accounts_factor = df.groupby(['Account', 'Motivating_Factor'])['Sales'].sum().reset_index()
                # Get the top 25 accounts by total sales
                account_totals = df.groupby('Account')['Sales'].sum().reset_index()
                top_25_accounts = account_totals.sort_values('Sales', ascending=False).head(25)['Account'].tolist()
                
                # Filter for top 25 accounts
                top_accounts_factor = top_accounts_factor[top_accounts_factor['Account'].isin(top_25_accounts)]
                
                # Sort accounts by total sales (ascending for horizontal bar display)
                account_order = account_totals[account_totals['Account'].isin(top_25_accounts)]
                account_order = account_order.sort_values('Sales', ascending=True)['Account'].tolist()
                
                if not top_accounts_factor.empty:
                    # Define color mapping for motivating factors
                    factor_colors = {
                        'VALT expansion': '#1f77b4',
                        'New system': '#ff7f0e', 
                        'Replace legacy system': '#2ca02c',
                        'Hardware refresh': '#d62728',
                        'SSA': '#9467bd',
                        'Unknown': '#8c564b'
                    }
                    
                    charts['top_accounts_motivating_factor'] = go.Figure()
                    
                    # Add stacked bars for each motivating factor
                    for factor in sorted(top_accounts_factor['Motivating_Factor'].unique()):
                        factor_data = top_accounts_factor[top_accounts_factor['Motivating_Factor'] == factor]
                        if not factor_data.empty:
                            charts['top_accounts_motivating_factor'].add_trace(go.Bar(
                                y=factor_data['Account'],
                                x=factor_data['Sales'],
                                name=factor,
                                orientation='h',
                                marker_color=factor_colors.get(factor, '#17becf'),
                                text=factor_data['Sales'].apply(lambda x: f'${x:,.0f}' if x > 50000 else ''),
                                textposition='inside',
                                textfont=dict(color='white', size=10),
                                hovertemplate='<b>%{y}</b><br>' +
                                            'Motivating Factor: ' + factor + '<br>' +
                                            'Sales: $%{x:,.0f}<extra></extra>'
                            ))
                    
                    charts['top_accounts_motivating_factor'].update_layout(
                        title='Top 25 Accounts by Sales with Stacked Motivating Factors',
                        xaxis_title='Total Sales ($)',
                        yaxis_title='Account',
                        barmode='stack',  # Enable stacking
                        height=800,  # Increased height for 25 accounts
                        margin=dict(l=200, r=100, b=120),  # Increased bottom margin for legend
                        yaxis=dict(
                            categoryorder='array',
                            categoryarray=account_order  # Ensure proper sorting
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.18,
                            xanchor="center",
                            x=0.5,
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="rgba(0,0,0,0.3)",
                            borderwidth=1,
                            font=dict(color='black', size=12)  # Black text for legend
                        ),
                        showlegend=True
                    )
            
            # Quarterly vs Monthly Sales
            quarterly_sales = df.groupby(['Year', 'Quarter'])['Sales'].sum().reset_index()
            quarterly_sales['Period'] = quarterly_sales['Year'].astype(str) + ' Q' + quarterly_sales['Quarter'].astype(str)
            
            monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
            monthly_sales['Period'] = monthly_sales['Year'].astype(str) + ' M' + monthly_sales['Month'].astype(str)
            
            charts['quarterly_vs_monthly'] = go.Figure()
            
            # Add quarterly data
            charts['quarterly_vs_monthly'].add_trace(go.Bar(
                x=quarterly_sales['Period'],
                y=quarterly_sales['Sales'],
                name='Quarterly Sales',
                marker_color='#1f77b4'
            ))
            
            # Add monthly data
            charts['quarterly_vs_monthly'].add_trace(go.Scatter(
                x=monthly_sales['Period'],
                y=monthly_sales['Sales'],
                name='Monthly Sales',
                mode='lines+markers',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=6)
            ))
            
            # Add trend line for quarterly sales
            if len(quarterly_sales) > 1:
                quarterly_trend = np.polyfit(range(len(quarterly_sales)), quarterly_sales['Sales'], 1)
                quarterly_trend_line = np.polyval(quarterly_trend, range(len(quarterly_sales)))
                charts['quarterly_vs_monthly'].add_trace(go.Scatter(
                    x=quarterly_sales['Period'],
                    y=quarterly_trend_line,
                    name='Quarterly Trend',
                    mode='lines',
                    line=dict(color='#1f77b4', width=3, dash='dash'),
                    showlegend=True
                ))
            
            # Add trend line for monthly sales
            if len(monthly_sales) > 1:
                monthly_trend = np.polyfit(range(len(monthly_sales)), monthly_sales['Sales'], 1)
                monthly_trend_line = np.polyval(monthly_trend, range(len(monthly_sales)))
                charts['quarterly_vs_monthly'].add_trace(go.Scatter(
                    x=monthly_sales['Period'],
                    y=monthly_trend_line,
                    name='Monthly Trend',
                    mode='lines',
                    line=dict(color='#ff7f0e', width=3, dash='dash'),
                    showlegend=True
                ))
            
            charts['quarterly_vs_monthly'].update_layout(
                title='Sales by Quarter vs Sales by Month (with Trend Lines)',
                xaxis_title="Time Period",
                yaxis_title="Total Sales ($)",
                barmode='overlay',
                xaxis={'tickangle': 45}
            )
        
        return charts
    
    @staticmethod
    def create_application_sales_heatmap(df: pd.DataFrame) -> go.Figure:
        """Create a heatmap showing Application sales by Year using Plotly"""
        try:
            # Group by Year and Application, sum sales
            heatmap_data = df.groupby(['Year', 'Application'])['Sales'].sum().reset_index()
            
            # Pivot the data to have years as columns and applications as rows
            heatmap_pivot = heatmap_data.pivot(index='Application', columns='Year', values='Sales')
            
            # Fill NaN values with 0
            heatmap_pivot = heatmap_pivot.fillna(0)
            
            # Sort applications by total sales (descending) for better visual hierarchy
            app_totals = heatmap_pivot.sum(axis=1).sort_values(ascending=False)
            heatmap_pivot = heatmap_pivot.loc[app_totals.index]
            
            # Create text annotations for the heatmap
            text_annotations = heatmap_pivot.map(lambda x: f'${x:,.0f}' if x > 0 else '')
            
            # Create Plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns.astype(str),  # Years as strings
                y=heatmap_pivot.index,  # Applications
                text=text_annotations.values,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(
                    title="Sales Volume ($)",
                    tickmode="linear",
                    tick0=0,
                    dtick=500000
                ),
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>' +
                             'Year: %{x}<br>' +
                             'Sales: $%{z:,.0f}<br>' +
                             '<extra></extra>'
            ))
            
            # Update layout for better appearance
            fig.update_layout(
                title={
                    'text': '🔥 Sales Heatmap: Applications by Year',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': 'white'}
                },
                xaxis_title='Year',
                yaxis_title='Application',
                font=dict(size=12),
                height=max(400, len(heatmap_pivot) * 40),  # Dynamic height based on applications
                margin=dict(l=150, r=100, t=80, b=50),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    tickangle=0,
                    title_font=dict(size=14, color='white'),
                    tickfont=dict(size=11, color='white')
                ),
                yaxis=dict(
                    title_font=dict(size=14, color='white'),
                    tickfont=dict(size=11, color='white')
                )
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")
            return None
    
    @staticmethod
    def create_market_sales_heatmap(df: pd.DataFrame) -> go.Figure:
        """Create a heatmap showing Market sales by Year using Plotly"""
        try:
            # Group by Year and Market, sum sales
            heatmap_data = df.groupby(['Year', 'Market'])['Sales'].sum().reset_index()
            
            # Pivot the data to have years as columns and markets as rows
            heatmap_pivot = heatmap_data.pivot(index='Market', columns='Year', values='Sales')
            
            # Fill NaN values with 0
            heatmap_pivot = heatmap_pivot.fillna(0)
            
            # Sort markets by total sales (descending) for better visual hierarchy
            market_totals = heatmap_pivot.sum(axis=1).sort_values(ascending=False)
            heatmap_pivot = heatmap_pivot.loc[market_totals.index]
            
            # Create text annotations for the heatmap
            text_annotations = heatmap_pivot.map(lambda x: f'${x:,.0f}' if x > 0 else '')
            
            # Create Plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns.astype(str),  # Years as strings
                y=heatmap_pivot.index,  # Markets
                text=text_annotations.values,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorscale='Blues',  # Using Blues colorscale to differentiate from Applications
                showscale=True,
                colorbar=dict(
                    title="Sales Volume ($)",
                    tickmode="linear",
                    tick0=0,
                    dtick=500000
                ),
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>' +
                             'Year: %{x}<br>' +
                             'Sales: $%{z:,.0f}<br>' +
                             '<extra></extra>'
            ))
            
            # Update layout for better appearance
            fig.update_layout(
                title={
                    'text': '🌐 Sales Heatmap: Markets by Year',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': 'white'}
                },
                xaxis_title='Year',
                yaxis_title='Market',
                font=dict(size=12),
                height=max(400, len(heatmap_pivot) * 40),  # Dynamic height based on markets
                margin=dict(l=150, r=100, t=80, b=50),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    tickangle=0,
                    title_font=dict(size=14, color='white'),
                    tickfont=dict(size=11, color='white')
                ),
                yaxis=dict(
                    title_font=dict(size=14, color='white'),
                    tickfont=dict(size=11, color='white')
                )
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating market heatmap: {str(e)}")
            return None
    
    @staticmethod
    def create_quotes_application_heatmap(df: pd.DataFrame) -> go.Figure:
        """Create a heatmap showing Quote Applications by Year using Plotly"""
        try:
            # Group by Year and Application, sum quote totals
            heatmap_data = df.groupby(['Year', 'Application'])['Quote_Total'].sum().reset_index()
            
            # Pivot the data to have years as columns and applications as rows
            heatmap_pivot = heatmap_data.pivot(index='Application', columns='Year', values='Quote_Total')
            
            # Fill NaN values with 0
            heatmap_pivot = heatmap_pivot.fillna(0)
            
            # Sort applications by total quotes (descending) for better visual hierarchy
            app_totals = heatmap_pivot.sum(axis=1).sort_values(ascending=False)
            heatmap_pivot = heatmap_pivot.loc[app_totals.index]
            
            # Create text annotations for the heatmap
            text_annotations = heatmap_pivot.map(lambda x: f'${x:,.0f}' if x > 0 else '')
            
            # Create Plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns.astype(str),  # Years as strings
                y=heatmap_pivot.index,  # Applications
                text=text_annotations.values,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorscale='Oranges',  # Using Oranges colorscale for quotes
                showscale=True,
                colorbar=dict(
                    title="Quote Volume ($)",
                    tickmode="linear",
                    tick0=0,
                    dtick=200000
                ),
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>' +
                             'Year: %{x}<br>' +
                             'Quotes: $%{z:,.0f}<br>' +
                             '<extra></extra>'
            ))
            
            # Update layout for better appearance
            fig.update_layout(
                title={
                    'text': '💼 Quotes Heatmap: Applications by Year',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': 'white'}
                },
                xaxis_title='Year',
                yaxis_title='Application',
                font=dict(size=12),
                height=max(400, len(heatmap_pivot) * 40),  # Dynamic height based on applications
                margin=dict(l=150, r=100, t=80, b=50),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    tickangle=0,
                    title_font=dict(size=14, color='white'),
                    tickfont=dict(size=11, color='white')
                ),
                yaxis=dict(
                    title_font=dict(size=14, color='white'),
                    tickfont=dict(size=11, color='white')
                )
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating quotes application heatmap: {str(e)}")
            return None
    
    @staticmethod
    def create_quotes_market_heatmap(df: pd.DataFrame) -> go.Figure:
        """Create a heatmap showing Quote Markets by Year using Plotly"""
        try:
            # Group by Year and Market, sum quote totals
            heatmap_data = df.groupby(['Year', 'Market'])['Quote_Total'].sum().reset_index()
            
            # Pivot the data to have years as columns and markets as rows
            heatmap_pivot = heatmap_data.pivot(index='Market', columns='Year', values='Quote_Total')
            
            # Fill NaN values with 0
            heatmap_pivot = heatmap_pivot.fillna(0)
            
            # Sort markets by total quotes (descending) for better visual hierarchy
            market_totals = heatmap_pivot.sum(axis=1).sort_values(ascending=False)
            heatmap_pivot = heatmap_pivot.loc[market_totals.index]
            
            # Create text annotations for the heatmap
            text_annotations = heatmap_pivot.map(lambda x: f'${x:,.0f}' if x > 0 else '')
            
            # Create Plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns.astype(str),  # Years as strings
                y=heatmap_pivot.index,  # Markets
                text=text_annotations.values,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorscale='Purples',  # Using Purples colorscale for quote markets
                showscale=True,
                colorbar=dict(
                    title="Quote Volume ($)",
                    tickmode="linear",
                    tick0=0,
                    dtick=200000
                ),
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>' +
                             'Year: %{x}<br>' +
                             'Quotes: $%{z:,.0f}<br>' +
                             '<extra></extra>'
            ))
            
            # Update layout for better appearance
            fig.update_layout(
                title={
                    'text': '🎯 Quotes Heatmap: Markets by Year',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': 'white'}
                },
                xaxis_title='Year',
                yaxis_title='Market',
                font=dict(size=12),
                height=max(400, len(heatmap_pivot) * 40),  # Dynamic height based on markets
                margin=dict(l=150, r=100, t=80, b=50),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    tickangle=0,
                    title_font=dict(size=14, color='white'),
                    tickfont=dict(size=11, color='white')
                ),
                yaxis=dict(
                    title_font=dict(size=14, color='white'),
                    tickfont=dict(size=11, color='white')
                )
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating quotes market heatmap: {str(e)}")
            return None
    
    @staticmethod 
    def create_quotes_charts(df: pd.DataFrame, filters: Dict) -> Dict:
        """Create quotes-specific charts"""
        charts = {}
        
        if not df.empty:
            # Quotes by Application
            app_quotes = df.groupby('Application')['Quote_Total'].sum().reset_index()
            app_quotes = app_quotes.sort_values('Quote_Total', ascending=False)
            
            # Pie chart for percentage distribution
            charts['quotes_by_application_pie'] = px.pie(
                app_quotes,
                values='Quote_Total',
                names='Application',
                title='Quote Distribution by Application (%)'
            )
            
            # Bar chart for absolute values and growth
            charts['quotes_by_application_bar'] = px.bar(
                app_quotes,
                x='Application', 
                y='Quote_Total',
                title='Quotes by Application ($)',
                color='Quote_Total',
                color_continuous_scale='Oranges'
            )
            charts['quotes_by_application_bar'].update_layout(
                xaxis_title="Application",
                yaxis_title="Total Quote Amount ($)",
                xaxis={'tickangle': 45}
            )
            
            # Create Quotes Application Heatmap
            quotes_app_heatmap = ChartManager.create_quotes_application_heatmap(df)
            if quotes_app_heatmap is not None:
                charts['quotes_application_heatmap'] = quotes_app_heatmap
            
            # Quotes by Year with color-coded trend line
            year_quotes = df.groupby('Year')['Quote_Total'].sum().reset_index()
            
            # Create the line chart using plotly graph objects for more control
            charts['quotes_by_year'] = go.Figure()
            
            # Add the main quotes line
            charts['quotes_by_year'].add_trace(go.Scatter(
                x=year_quotes['Year'],
                y=year_quotes['Quote_Total'],
                mode='lines+markers',
                name='Quotes',
                line=dict(color='#ff7f0e', width=3),  # Orange color for quotes
                marker=dict(size=8, color='#ff7f0e')
            ))
            
            # Calculate and add trend line with color coding
            if len(year_quotes) > 1:
                # Calculate linear trend
                x_vals = range(len(year_quotes))
                trend_coeffs = np.polyfit(x_vals, year_quotes['Quote_Total'], 1)
                trend_line = np.polyval(trend_coeffs, x_vals)
                
                # Determine trend color based on slope (same thresholds as sales)
                slope = trend_coeffs[0]
                if slope > 100000:  # Positive trend (threshold for "significant" increase)
                    trend_color = 'green'
                elif slope < -100000:  # Negative trend (threshold for "significant" decrease)
                    trend_color = 'red'
                else:  # Flat trend
                    trend_color = 'gold'
                
                # Add trend line
                charts['quotes_by_year'].add_trace(go.Scatter(
                    x=year_quotes['Year'],
                    y=trend_line,
                    mode='lines',
                    name='Trend',
                    line=dict(color=trend_color, width=2, dash='dash'),
                    showlegend=False  # Hide from legend as requested
                ))
            
            charts['quotes_by_year'].update_layout(
                title='Quote Trends by Year with Trend Analysis',
                xaxis_title="Year",
                yaxis_title="Total Quote Amount ($)"
            )
            
            # Quotes by Status
            if 'Status' in df.columns:
                status_quotes = df.groupby('Status')['Quote_Total'].sum().reset_index()
                charts['quotes_by_status'] = px.pie(
                    status_quotes,
                    values='Quote_Total',
                    names='Status',
                    title='Quote Distribution by Status'
                )
            
            # Quotes by Market
            market_quotes = df.groupby('Market')['Quote_Total'].sum().reset_index()
            market_quotes = market_quotes.sort_values('Quote_Total', ascending=False)
            
            # Pie chart for percentage distribution
            charts['quotes_by_market_pie'] = px.pie(
                market_quotes,
                values='Quote_Total',
                names='Market',
                title='Quote Distribution by Market (%)'
            )
            
            # Bar chart for absolute values
            charts['quotes_by_market_bar'] = px.bar(
                market_quotes,
                x='Market',
                y='Quote_Total',
                title='Quotes by Market ($)',
                color='Quote_Total',
                color_continuous_scale='Purples'
            )
            charts['quotes_by_market_bar'].update_layout(
                xaxis_title="Market",
                yaxis_title="Total Quote Amount ($)",
                xaxis={'tickangle': 45}
            )
            
            # Create Quotes Market Heatmap
            quotes_market_heatmap = ChartManager.create_quotes_market_heatmap(df)
            if quotes_market_heatmap is not None:
                charts['quotes_market_heatmap'] = quotes_market_heatmap
            
            # Quotes by Customer Type
            customer_quotes = df.groupby('Customer')['Quote_Total'].sum().reset_index()
            
            # Pie chart for percentage distribution
            charts['quotes_by_customer_pie'] = px.pie(
                customer_quotes,
                values='Quote_Total',
                names='Customer',
                title='Quote Distribution by Customer Type (%)'
            )
            
            # Bar chart for absolute values
            charts['quotes_by_customer_bar'] = px.bar(
                customer_quotes,
                x='Customer',
                y='Quote_Total',
                title='Quotes by Customer Type ($)',
                color='Quote_Total',
                color_continuous_scale='Oranges'
            )
            charts['quotes_by_customer_bar'].update_layout(
                xaxis_title="Customer Type",
                yaxis_title="Total Quote Amount ($)"
            )
            
            # Top 25 Customers horizontal bar chart with status-based colors (by Account)
            if 'Status' in df.columns:
                # Group by Account, Status, Year, and Application to include application information
                top_customers_detail = df.groupby(['Account', 'Status', 'Year', 'Application'])['Quote_Total'].sum().reset_index()
                
                # Get total quote amount per customer (sorted highest to lowest)
                customer_totals = df.groupby('Account')['Quote_Total'].sum().reset_index()
                top_25_customers = customer_totals.sort_values('Quote_Total', ascending=False).head(25)
                top_25_accounts = top_25_customers['Account'].tolist()
                
                # Filter detail data for top 25 customers
                top_customers_detail = top_customers_detail[
                    top_customers_detail['Account'].isin(top_25_accounts)
                ]
                
                # Create the horizontal bar chart using plotly graph objects
                charts['top_customers_quotes_horizontal'] = go.Figure()
                
                # Define color mapping for status
                status_colors = {
                    'Closed': 'green',
                    'Lost': 'red', 
                    'Open': 'gold',
                    'Pending': 'gold',  # Treat pending as open
                    'Won': 'green',     # Alternative for closed
                    'Awarded': 'green'  # Alternative for closed
                }
                
                # Sort customers by total value (highest to lowest) for proper y-axis ordering
                customer_order = top_25_customers.sort_values('Quote_Total', ascending=True)['Account'].tolist()
                
                # Add bars for each status
                for status in sorted(top_customers_detail['Status'].unique()):
                    status_data = top_customers_detail[top_customers_detail['Status'] == status]
                    if not status_data.empty:
                        color = status_colors.get(status, 'gray')  # Default to gray for unknown status
                        
                        # Create text labels with year, application, and amount
                        text_labels = status_data.apply(
                            lambda row: f"{row['Year']}<br>{row['Application']}<br>${row['Quote_Total']:,.0f}", 
                            axis=1
                        )
                        
                        # Create hover text with detailed information including application
                        hover_text = status_data.apply(
                            lambda row: f"<b>{row['Account']}</b><br>Application: {row['Application']}<br>Status: {status}<br>Year: {row['Year']}<br>Amount: ${row['Quote_Total']:,.0f}",
                            axis=1
                        )
                        
                        charts['top_customers_quotes_horizontal'].add_trace(go.Bar(
                            y=status_data['Account'],
                            x=status_data['Quote_Total'],
                            name=status,
                            orientation='h',
                            marker_color=color,
                            text=text_labels,
                            textposition='inside',
                            textfont=dict(size=8),  # Smaller font to fit application labels
                            hovertemplate='%{hovertext}<extra></extra>',
                            hovertext=hover_text
                        ))
                
                # Update layout with proper sorting and legend positioning
                charts['top_customers_quotes_horizontal'].update_layout(
                    title='Top 25 Customers by Quote Value (Status-Based Colors with Years & Applications)',
                    xaxis_title='Total Quote Amount ($)',
                    yaxis_title='Customer Account',
                    barmode='stack',
                    height=800,  # Increased height for 25 customers
                    margin=dict(l=200, b=100),  # Add bottom margin for legend
                    yaxis=dict(
                        categoryorder='array',
                        categoryarray=customer_order  # Ensure proper sorting
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.15,  # Move further down to avoid x-axis labels
                        xanchor="center",
                        x=0.5,
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="rgba(0,0,0,0.2)",
                        borderwidth=1
                    )
                )
            else:
                # Fallback: simple horizontal bar if no Status column
                top_customers = df.groupby('Account')['Quote_Total'].sum().reset_index()
                top_customers = top_customers.sort_values('Quote_Total', ascending=True).tail(10)
                
                charts['top_customers_quotes_horizontal'] = px.bar(
                    top_customers,
                    x='Quote_Total',
                    y='Account',
                    title='Top 10 Customers by Total Quote Value',
                    orientation='h',
                    color='Quote_Total',
                    color_continuous_scale='Viridis',
                    text='Quote_Total'
                )
                charts['top_customers_quotes_horizontal'].update_traces(
                    texttemplate='$%{text:,.0f}',
                    textposition='outside'
                )
                charts['top_customers_quotes_horizontal'].update_layout(
                    xaxis_title='Total Quote Amount ($)',
                    yaxis_title='Customer Account',
                    height=400,
                    margin=dict(l=200)
                )
            
            # Quotes by Channel
            channel_quotes = df.groupby('Channel')['Quote_Total'].sum().reset_index()
            
            # Pie chart for percentage distribution
            charts['quotes_by_channel_pie'] = px.pie(
                channel_quotes,
                values='Quote_Total',
                names='Channel',
                title='Quote Distribution by Channel (%)'
            )
            
            # Bar chart for absolute values
            charts['quotes_by_channel_bar'] = px.bar(
                channel_quotes,
                x='Channel',
                y='Quote_Total',
                title='Quotes by Channel ($)',
                color='Quote_Total',
                color_continuous_scale='Blues'
            )
            charts['quotes_by_channel_bar'].update_layout(
                xaxis_title="Channel",
                yaxis_title="Total Quote Amount ($)",
                xaxis={'tickangle': 45}
            )
            
            # Quotes by State (if available)
            if 'State' in df.columns:
                state_quotes = df.groupby('State')['Quote_Total'].sum().reset_index()
                
                # Pie chart for percentage distribution
                charts['quotes_by_state_pie'] = px.pie(
                    state_quotes,
                    values='Quote_Total',
                    names='State',
                    title='Quote Distribution by State (%)'
                )
                
                # Bar chart for absolute values
                charts['quotes_by_state_bar'] = px.bar(
                    state_quotes,
                    x='State',
                    y='Quote_Total',
                    title='Quotes by State ($)',
                    color='Quote_Total',
                    color_continuous_scale='Greens'
                )
                charts['quotes_by_state_bar'].update_layout(
                    xaxis_title="State",
                    yaxis_title="Total Quote Amount ($)",
                    xaxis={'tickangle': 45}
                )
                
                # Geographic map for Northeast and Mid-Atlantic states
                northeast_states = {
                    'Maine': 'ME', 'New Hampshire': 'NH', 'Vermont': 'VT', 'Massachusetts': 'MA',
                    'Rhode Island': 'RI', 'Connecticut': 'CT', 'New York': 'NY', 'New Jersey': 'NJ',
                    'Pennsylvania': 'PA', 'Virginia': 'VA', 'West Virginia': 'WV', 'Delaware': 'DE', 
                    'Maryland': 'MD', 'Washington DC': 'DC', 'District of Columbia': 'DC', 'DC': 'DC'
                }
                
                # Create state code mapping for the quotes data
                state_quotes_map = state_quotes.copy()
                # Handle both full names and abbreviations in the data
                state_quotes_map['State_Code'] = state_quotes_map['State'].map(lambda x: 
                    northeast_states.get(x, x) if x in northeast_states else x
                )
                
                # Filter for Northeast states only (check both full names and abbreviations)
                northeast_quotes_data = state_quotes_map[
                    (state_quotes_map['State'].isin(northeast_states.keys())) | 
                    (state_quotes_map['State'].isin(northeast_states.values()))
                ]
                
                if not northeast_quotes_data.empty:
                    # Ensure we have proper state codes for the map
                    northeast_quotes_data['State_Code'] = northeast_quotes_data['State'].map(lambda x: 
                        northeast_states.get(x, x) if len(x) > 2 else x
                    )
                    
                    charts['northeast_quotes_map'] = px.choropleth(
                        northeast_quotes_data,
                        locations='State_Code',
                        color='Quote_Total',
                        hover_name='State',
                        hover_data={'Quote_Total': ':$,.0f', 'State_Code': False},
                        color_continuous_scale='Greens',
                        title='Quotes by Northeast & Mid-Atlantic States with Account Markers',
                        locationmode='USA-states'
                    )
                    
                    # Add account markers if coordinate data and status are available
                    total_northeast_quotes = len(df[
                        (df['State'].isin(northeast_states.keys())) | 
                        (df['State'].isin(northeast_states.values()))
                    ])
                    
                    # Check if we have coordinate columns and status column
                    if 'lat' in df.columns and 'lng' in df.columns and 'Status' in df.columns:
                        # Filter for Northeast and Mid-Atlantic quotes with status
                        northeast_quotes_with_coords = df[
                            (df['State'].isin(northeast_states.keys())) | 
                            (df['State'].isin(northeast_states.values()))
                        ].copy()
                        
                        if not northeast_quotes_with_coords.empty:
                            # Clean coordinates (should already be cleaned from load_quotes_data())
                            northeast_quotes_with_coords['lat_num'] = pd.to_numeric(northeast_quotes_with_coords['lat'], errors='coerce')
                            northeast_quotes_with_coords['lng_num'] = pd.to_numeric(northeast_quotes_with_coords['lng'], errors='coerce')
                            
                            # Filter for valid coordinates within reasonable US bounds
                            valid_coords_quotes = northeast_quotes_with_coords[
                                (northeast_quotes_with_coords['lat_num'].notna()) & 
                                (northeast_quotes_with_coords['lng_num'].notna()) &
                                (northeast_quotes_with_coords['lat_num'] >= 20) & 
                                (northeast_quotes_with_coords['lat_num'] <= 50) &
                                (northeast_quotes_with_coords['lng_num'] >= -130) & 
                                (northeast_quotes_with_coords['lng_num'] <= -60)
                            ].copy()
                            
                            if not valid_coords_quotes.empty:
                                # Group by Account and Status, sum quote totals, keep coordinates, market, and city info
                                account_markers = valid_coords_quotes.groupby(['Account', 'Status']).agg({
                                    'Quote_Total': 'sum',
                                    'lat_num': 'first',
                                    'lng_num': 'first',
                                    'State': 'first',
                                    'Market': 'first',  # Include market information
                                    'City': 'first' if 'City' in valid_coords_quotes.columns else None  # Include city information if available
                                }).reset_index()
                                
                                # Define status-based colors and grouping
                                def get_status_info(status):
                                    status_lower = status.lower()
                                    if status_lower in ['closed', 'won', 'awarded']:
                                        return 'Closed/Won', 'green', 10
                                    elif status_lower in ['lost', 'denied', 'rejected']:
                                        return 'Lost', 'red', 10
                                    elif status_lower in ['open', 'pending', 'active', 'in progress']:
                                        return 'Open/Pending', 'gold', 10
                                    else:
                                        return 'Other', 'gray', 8
                                
                                account_markers[['status_category', 'color', 'size']] = account_markers['Status'].apply(
                                    lambda x: pd.Series(get_status_info(x))
                                )
                                
                                # Add markers by status category
                                for status_category in account_markers['status_category'].unique():
                                    category_data = account_markers[account_markers['status_category'] == status_category]
                                    if not category_data.empty:
                                        marker_size = category_data['size'].iloc[0]
                                        marker_color = category_data['color'].iloc[0]
                                        
                                        charts['northeast_quotes_map'].add_trace(go.Scattergeo(
                                            lat=category_data['lat_num'],
                                            lon=category_data['lng_num'],
                                            text=category_data.apply(
                                                lambda x: f"<b>{x['Account']}</b><br>Market: {x['Market']}<br>City: {x.get('City', 'N/A') if x.get('City') and str(x.get('City')).strip() not in ['', 'nan', 'None'] else 'N/A'}, {x['State']}<br>Status: {x['Status']}<br>Quote Total: ${x['Quote_Total']:,.0f}",
                                                axis=1
                                            ),
                                            mode='markers',
                                            marker=dict(
                                                size=marker_size,
                                                color=marker_color,
                                                line=dict(width=2, color='white'),
                                                opacity=0.9
                                            ),
                                            name=f'{status_category} ({len(category_data)} accounts)',
                                            hovertemplate='%{text}<extra></extra>'
                                        ))
                                
                                # Update title with detailed mapping info
                                mapped_accounts = len(account_markers)
                                closed_count = len(account_markers[account_markers['status_category'] == 'Closed/Won'])
                                lost_count = len(account_markers[account_markers['status_category'] == 'Lost'])
                                open_count = len(account_markers[account_markers['status_category'] == 'Open/Pending'])
                                other_count = len(account_markers[account_markers['status_category'] == 'Other'])
                                
                                charts['northeast_quotes_map'].update_layout(
                                    title=f'Quotes by Northeast & Mid-Atlantic States with Account Markers<br><sub>{mapped_accounts} accounts mapped from {total_northeast_quotes} total in region | 🟢 Closed: {closed_count}, 🔴 Lost: {lost_count}, 🟡 Open: {open_count}, ⚪ Other: {other_count}</sub>'
                                )
                            else:
                                # No valid coordinates found
                                charts['northeast_quotes_map'].update_layout(
                                    title=f'Quotes by Northeast & Mid-Atlantic States<br><sub>No accounts with valid coordinates found from {total_northeast_quotes} total in region</sub>'
                                )
                        else:
                            # No accounts in region
                            charts['northeast_quotes_map'].update_layout(
                                title='Quotes by Northeast & Mid-Atlantic States<br><sub>No accounts found in Northeast/Mid-Atlantic region</sub>'
                            )
                    else:
                        # Missing required data
                        missing_cols = []
                        if 'lat' not in df.columns:
                            missing_cols.append('latitude')
                        if 'lng' not in df.columns:
                            missing_cols.append('longitude')
                        if 'Status' not in df.columns:
                            missing_cols.append('status')
                        
                        charts['northeast_quotes_map'].update_layout(
                            title=f'Quotes by Northeast & Mid-Atlantic States<br><sub>Account markers unavailable - missing {" and ".join(missing_cols)} data in dataset</sub>'
                        )
                    
                    # Focus on Northeast and Mid-Atlantic region
                    charts['northeast_quotes_map'].update_layout(
                        geo=dict(
                            scope='usa',
                            projection_scale=1,
                            center=dict(lat=40.0, lon=-76),  # Center on expanded region
                            lonaxis=dict(range=[-82, -66]),   # Longitude range for expanded region
                            lataxis=dict(range=[36, 47])      # Latitude range for expanded region
                        ),
                        height=800,  # Increased height for better visibility
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.15,  # Move legend down to avoid overlap
                            xanchor="center",
                            x=0.5,
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="rgba(0,0,0,0.3)",
                            borderwidth=1
                        ),
                        # Add annotations for legend explanation
                        annotations=[
                            dict(
                                text="💡 Account markers: 🟢 Green (Closed/Won), 🔴 Red (Lost), 🟡 Yellow (Open/Pending)",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.5, y=-0.3,
                                xanchor='center', yanchor='bottom',
                                font=dict(size=11, color="gray")
                            )
                        ]
                    )
            
            # Quarterly vs Monthly Quotes
            quarterly_quotes = df.groupby(['Year', 'Quarter'])['Quote_Total'].sum().reset_index()
            quarterly_quotes['Period'] = quarterly_quotes['Year'].astype(str) + ' Q' + quarterly_quotes['Quarter'].astype(str)
            
            monthly_quotes = df.groupby(['Year', 'Month'])['Quote_Total'].sum().reset_index()
            monthly_quotes['Period'] = monthly_quotes['Year'].astype(str) + ' M' + monthly_quotes['Month'].astype(str)
            
            charts['quarterly_vs_monthly'] = go.Figure()
            
            # Add quarterly data
            charts['quarterly_vs_monthly'].add_trace(go.Bar(
                x=quarterly_quotes['Period'],
                y=quarterly_quotes['Quote_Total'],
                name='Quarterly Quotes',
                marker_color='#9467bd'
            ))
            
            # Add monthly data
            charts['quarterly_vs_monthly'].add_trace(go.Scatter(
                x=monthly_quotes['Period'],
                y=monthly_quotes['Quote_Total'],
                name='Monthly Quotes',
                mode='lines+markers',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=6)
            ))
            
            # Add trend line for quarterly quotes
            if len(quarterly_quotes) > 1:
                quarterly_trend = np.polyfit(range(len(quarterly_quotes)), quarterly_quotes['Quote_Total'], 1)
                quarterly_trend_line = np.polyval(quarterly_trend, range(len(quarterly_quotes)))
                charts['quarterly_vs_monthly'].add_trace(go.Scatter(
                    x=quarterly_quotes['Period'],
                    y=quarterly_trend_line,
                    name='Quarterly Trend',
                    mode='lines',
                    line=dict(color='#9467bd', width=3, dash='dash'),
                    showlegend=True
                ))
            
            # Add trend line for monthly quotes
            if len(monthly_quotes) > 1:
                monthly_trend = np.polyfit(range(len(monthly_quotes)), monthly_quotes['Quote_Total'], 1)
                monthly_trend_line = np.polyval(monthly_trend, range(len(monthly_quotes)))
                charts['quarterly_vs_monthly'].add_trace(go.Scatter(
                    x=monthly_quotes['Period'],
                    y=monthly_trend_line,
                    name='Monthly Trend',
                    mode='lines',
                    line=dict(color='#ff7f0e', width=3, dash='dash'),
                    showlegend=True
                ))
            
            charts['quarterly_vs_monthly'].update_layout(
                title='Quotes by Quarter vs Quotes by Month (with Trend Lines)',
                xaxis_title="Time Period",
                yaxis_title="Total Quote Amount ($)",
                barmode='overlay',
                xaxis={'tickangle': 45}
            )
        
        return charts

class DashboardApp:
    """Main dashboard application with production-ready architecture"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.filter_manager = FilterManager()
        self.chart_manager = ChartManager()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state to prevent recomputation"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.current_page = "Sales Dashboard"
            st.session_state.last_refresh = None
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">📊 Sales & Quotes Analytics Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sales_page(self, sales_uploaded_file=None):
        """Render the sales analysis page"""
        
        # Load sales data
        with st.spinner("Loading sales data..."):
            sales_df = self.data_manager.load_sales_data(sales_uploaded_file)
        
        if sales_df is None:
            if sales_uploaded_file is None:
                st.info("📁 **Welcome to the Sales Dashboard!**")
                st.markdown("""
                To get started, please upload your **Sales CSV file** using the file uploader in the sidebar.
                
                **Expected CSV format:**
                - Date, Account, Sales, Application, Market, Channel, Customer, Year columns (required)
                - Optional: State, Motivating_Factor (or 'Motivating Factor'), lat, lng columns for enhanced analysis
                """)
                st.markdown("---")
                st.markdown("📊 Once you upload your file, you'll see interactive charts and analysis including:")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    - 📊 Sales trends and performance vs goals
                    - 🎨 Application and market analysis
                    - 🏆 Top customer rankings
                    - 🗺️ Geographic distribution maps
                    """)
                with col2:
                    st.markdown("""
                    - 📊 Channel performance (AV Integrator vs Direct)
                    - 🎯 Motivating factor analysis
                    - 📈 Quarterly vs monthly comparisons
                    - 🔍 Advanced filtering and data exploration
                    """)
            else:
                st.error("❌ Error loading the uploaded sales data. Please check the file format and try again.")
            return
        
        # Create sidebar filters
        st.sidebar.markdown("### 📊 Sales Dashboard")
        filters = self.filter_manager.create_sidebar_filters(sales_df, "sales")
        
        # Apply filters
        filtered_df = self.filter_manager.apply_filters(sales_df, filters)
        
        if filtered_df.empty:
            st.warning("⚠️ No data matches the selected filters. Please adjust your selections.")
            return
        
        # Date range display
        if not filtered_df.empty:
            start_year = filtered_df['Date'].min().year
            end_year = filtered_df['Date'].max().year
            st.markdown(f"### 📅 **Sales Data from {start_year} to {end_year}**")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = filtered_df['Sales'].sum()
            st.metric(
                label="💰 Total Sales",
                value=f"${total_sales:,.0f}",
                delta=None
            )
        
        with col2:
            total_accounts = filtered_df['Account'].nunique()
            st.metric(
                label="🏢 Unique Accounts", 
                value=f"{total_accounts:,}",
                delta=None
            )
        
        with col3:
            # Calculate average sale excluding SSA and hardware refresh
            exclude_categories = ['SSA', 'Hardware Refresh']
            filtered_sales = filtered_df[~filtered_df['Application'].isin(exclude_categories)]
            
            if not filtered_sales.empty:
                avg_sale = filtered_sales['Sales'].mean()
                st.metric(
                    label="📈 Average Sale (excl. SSA & Hardware)",
                    value=f"${avg_sale:,.0f}",
                    delta=None
                )
            else:
                st.metric(
                    label="📈 Average Sale (excl. SSA & Hardware)",
                    value="N/A",
                    delta=None
                )
        
        with col4:
            total_records = len(filtered_df)
            st.metric(
                label="📋 Total Records",
                value=f"{total_records:,}",
                delta=None
            )
        
        st.markdown("---")
        
        # Create charts
        charts = self.chart_manager.create_sales_charts(filtered_df, filters)
        
        # Display charts in tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 Trends", "📊 Applications", "🥧 Markets", "🥧 Customers", "🗺️ States", "📡 Channels", "🎯 Motivating Factors"])
        
        with tab1:
            if 'quarterly_vs_monthly' in charts:
                st.plotly_chart(charts['quarterly_vs_monthly'], use_container_width=True)
            
            if 'sales_performance_vs_goals' in charts:
                st.plotly_chart(charts['sales_performance_vs_goals'], use_container_width=True)
                
                # Performance data table
                if 'performance_data' in charts:
                    st.subheader("📊 Sales Performance vs Goals")
                    display_performance = charts['performance_data'].copy()
                    display_performance['Sales'] = display_performance['Sales'].apply(lambda x: f"${x:,.0f}")
                    display_performance['Goal'] = display_performance['Goal'].apply(lambda x: f"${x:,.0f}")
                    display_performance['Delta'] = display_performance['Delta'].apply(lambda x: f"${x:,.0f}")
                    display_performance['%_to_Goal'] = display_performance['%_to_Goal'].apply(lambda x: f"{x}%")
                    
                    # Rename columns for display
                    display_performance.columns = ['Year', 'Total Sales', 'Goal', 'Delta', '% to Goal']
                    st.dataframe(display_performance, use_container_width=True, hide_index=True)
            
            if 'sales_by_year' in charts:
                st.plotly_chart(charts['sales_by_year'], use_container_width=True)
        
        with tab2:
            # Display both pie and bar charts for applications
            col1, col2 = st.columns(2)
            
            with col1:
                if 'sales_by_application_pie' in charts:
                    st.plotly_chart(charts['sales_by_application_pie'], use_container_width=True)
            
            with col2:
                if 'sales_by_application_bar' in charts:
                    st.plotly_chart(charts['sales_by_application_bar'], use_container_width=True)
            
            # Add heatmap section
            st.markdown("### 🔥 Application Sales Heatmap by Year")
            if 'application_sales_heatmap' in charts:
                st.plotly_chart(charts['application_sales_heatmap'], use_container_width=True)
                st.markdown("*Interactive heatmap showing sales intensity across applications and years. Darker colors indicate higher sales volumes.*")
            
            # Top applications table
            st.subheader("🔝 Top Applications by Sales")
            top_apps = filtered_df.groupby('Application')['Sales'].sum().reset_index()
            top_apps = top_apps.sort_values('Sales', ascending=False).head(10)
            top_apps['Sales'] = top_apps['Sales'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(top_apps, use_container_width=True, hide_index=True)
        
        with tab3:
            # Display both pie and bar charts for markets
            col1, col2 = st.columns(2)
            
            with col1:
                if 'sales_by_market_pie' in charts:
                    st.plotly_chart(charts['sales_by_market_pie'], use_container_width=True)
            
            with col2:
                if 'sales_by_market_bar' in charts:
                    st.plotly_chart(charts['sales_by_market_bar'], use_container_width=True)
            
            # Add market heatmap section
            st.markdown("### 🌐 Market Sales Heatmap by Year")
            if 'market_sales_heatmap' in charts:
                st.plotly_chart(charts['market_sales_heatmap'], use_container_width=True)
                st.markdown("*Interactive heatmap showing sales intensity across markets and years. Darker colors indicate higher sales volumes.*")
        
        with tab4:
            # Display both pie and bar charts for customer types
            col1, col2 = st.columns(2)
            
            with col1:
                if 'sales_by_customer_pie' in charts:
                    st.plotly_chart(charts['sales_by_customer_pie'], use_container_width=True)
            
            with col2:
                if 'sales_by_customer_bar' in charts:
                    st.plotly_chart(charts['sales_by_customer_bar'], use_container_width=True)
            
            # Top 10 Individual Customers section
            st.markdown("### 🔝 Top 10 Individual Customers")
            
            # Display top customers charts
            col3, col4 = st.columns(2)
            
            with col3:
                if 'top_customers_horizontal' in charts:
                    st.plotly_chart(charts['top_customers_horizontal'], use_container_width=True)
            
            with col4:
                if 'top_customers_pie' in charts:
                    st.plotly_chart(charts['top_customers_pie'], use_container_width=True)
            
            # Top customers data table
            st.subheader("📋 Top 10 Customers by Sales")
            top_customers_table = filtered_df.groupby('Account')['Sales'].sum().reset_index()
            top_customers_table = top_customers_table.sort_values('Sales', ascending=False).head(10)
            top_customers_table['Sales'] = top_customers_table['Sales'].apply(lambda x: f"${x:,.0f}")
            top_customers_table.columns = ['Customer Account', 'Total Sales']
            top_customers_table.index = range(1, len(top_customers_table) + 1)  # Add ranking
            st.dataframe(top_customers_table, use_container_width=True)
        
        with tab5:
            if 'sales_by_state_pie' in charts and 'sales_by_state_bar' in charts:
                # Display both pie and bar charts for states
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(charts['sales_by_state_pie'], use_container_width=True)
                
                with col2:
                    st.plotly_chart(charts['sales_by_state_bar'], use_container_width=True)
                
                # Display Northeast geographical map if available
                if 'northeast_map' in charts:
                    st.markdown("### 🗺️ Northeast & Mid-Atlantic Sales Map")
                    st.plotly_chart(charts['northeast_map'], use_container_width=True)
                    st.markdown("*Interactive map showing sales distribution across Northeast and Mid-Atlantic states*")
                    
                    # Add geographic data diagnostics in an expander
                    with st.expander("🔍 Geographic Data Diagnostics", expanded=False):
                        geo_diag = self.chart_manager.diagnose_geographic_data(filtered_df)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Records", geo_diag['total_records'])
                            st.metric("Records with State", geo_diag['records_with_state'])
                        
                        with col2:
                            st.metric("Records with Valid Coordinates", geo_diag['records_with_coordinates'])
                            quality_color = "🟢" if geo_diag['coordinate_quality'] == 'good' else "🔴" if geo_diag['coordinate_quality'] == 'poor' else "🟡"
                            st.write(f"**Coordinate Quality:** {quality_color} {geo_diag['coordinate_quality'].title()}")
                        
                        with col3:
                            st.write(f"**Coordinate Columns Found:** {', '.join(geo_diag['coordinate_columns']) if geo_diag['coordinate_columns'] else 'None'}")
                        
                        if geo_diag['sample_coordinates']:
                            st.markdown("**Sample Coordinates:**")
                            sample_df = pd.DataFrame(geo_diag['sample_coordinates'], columns=['Latitude', 'Longitude', 'Account'])
                            st.dataframe(sample_df, use_container_width=True, hide_index=True)
                        
                        if geo_diag['issues']:
                            st.markdown("**Issues Found:**")
                            for issue in geo_diag['issues']:
                                st.write(f"⚠️ {issue}")
                else:
                    st.info("🗺️ No Northeast/Mid-Atlantic states found in the current data for geographical mapping.")
            else:
                st.info("State data not available in the dataset.")
        
        with tab6:
            if 'sales_by_channel' in charts:
                st.plotly_chart(charts['sales_by_channel'], use_container_width=True)
            
            # AV Integrator vs Direct Sales line chart
            if 'av_vs_direct_by_year' in charts:
                st.markdown("### 📈 AV Integrator vs Direct Sales Trends")
                st.plotly_chart(charts['av_vs_direct_by_year'], use_container_width=True)
                
                # Channel comparison summary table
                st.subheader("📋 Channel Performance by Year")
                channel_comparison = filtered_df[filtered_df['Channel'].isin(['AV Integrator', 'Direct'])]
                if not channel_comparison.empty:
                    yearly_comparison = channel_comparison.groupby(['Year', 'Channel'])['Sales'].sum().reset_index()
                    yearly_pivot = yearly_comparison.pivot(index='Year', columns='Channel', values='Sales').fillna(0)
                    
                    # Format as currency and add total column
                    for col in yearly_pivot.columns:
                        yearly_pivot[col] = yearly_pivot[col].apply(lambda x: f"${x:,.0f}")
                    
                    st.dataframe(yearly_pivot, use_container_width=True)
                else:
                    st.info("No AV Integrator or Direct sales data found for comparison.")
            
            # AV Integrator Accounts section
            if 'av_integrator_accounts' in charts:
                st.markdown("### 🔧 AV Integrator Accounts")
                st.plotly_chart(charts['av_integrator_accounts'], use_container_width=True)
                
                # AV Integrator accounts data table
                st.subheader("📋 AV Integrator Accounts by Sales")
                av_integrator_filtered = filtered_df[filtered_df['Channel'] == 'AV Integrator']
                if not av_integrator_filtered.empty:
                    av_table = av_integrator_filtered.groupby('Account')['Sales'].sum().reset_index()
                    av_table = av_table.sort_values('Sales', ascending=False)
                    av_table['Sales'] = av_table['Sales'].apply(lambda x: f"${x:,.0f}")
                    av_table.columns = ['Account', 'Total Sales']
                    av_table.index = range(1, len(av_table) + 1)  # Add ranking
                    st.dataframe(av_table, use_container_width=True)
                else:
                    st.info("No AV Integrator accounts found in the current date range.")
            else:
                st.info("No AV Integrator data available for the selected filters.")
        
        with tab7:
            if 'sales_by_factor_pie' in charts and 'sales_by_factor_bar' in charts:
                # Display both pie and bar charts for motivating factors
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(charts['sales_by_factor_pie'], use_container_width=True)
                
                with col2:
                    st.plotly_chart(charts['sales_by_factor_bar'], use_container_width=True)
                
                # Top 25 Accounts by Motivating Factor section
                if 'top_accounts_motivating_factor' in charts:
                    st.markdown("### 🏆 Top 25 Accounts by Motivating Factor")
                    st.plotly_chart(charts['top_accounts_motivating_factor'], use_container_width=True)
                    
                    # Top accounts by motivating factor data table
                    st.subheader("📋 Top 25 Accounts with Motivating Factors")
                    top_accounts_table = filtered_df.groupby(['Account', 'Motivating_Factor'])['Sales'].sum().reset_index()
                    # Get top 25 accounts by total sales
                    account_totals = filtered_df.groupby('Account')['Sales'].sum().reset_index()
                    top_25_list = account_totals.sort_values('Sales', ascending=False).head(25)['Account'].tolist()
                    
                    top_accounts_table = top_accounts_table[top_accounts_table['Account'].isin(top_25_list)]
                    top_accounts_table = top_accounts_table.sort_values('Sales', ascending=False)
                    top_accounts_table['Sales'] = top_accounts_table['Sales'].apply(lambda x: f"${x:,.0f}")
                    top_accounts_table.columns = ['Account', 'Motivating Factor', 'Total Sales']
                    top_accounts_table.index = range(1, len(top_accounts_table) + 1)  # Add ranking
                    st.dataframe(top_accounts_table, use_container_width=True)
                
                # Top motivating factors table
                st.subheader("🎯 Sales by Motivating Factor")
                factor_breakdown = filtered_df.groupby('Motivating_Factor')['Sales'].sum().reset_index()
                factor_breakdown = factor_breakdown.sort_values('Sales', ascending=False)
                factor_breakdown['Sales'] = factor_breakdown['Sales'].apply(lambda x: f"${x:,.0f}")
                factor_breakdown.columns = ['Motivating Factor', 'Total Sales']
                st.dataframe(factor_breakdown, use_container_width=True, hide_index=True)
            else:
                st.info("Motivating Factor data not available in the dataset.")
        
        
        # Data table section
        with st.expander("📋 Filtered Sales Data", expanded=False):
            display_columns = ['Date', 'Account', 'Sales', 'Application', 'Market', 'Channel', 'Customer', 'Motivating_Factor', 'State']
            # Only include columns that exist in the dataframe
            display_columns = [col for col in display_columns if col in filtered_df.columns]
            display_df = filtered_df[display_columns].copy()
            display_df['Sales'] = display_df['Sales'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    def render_quotes_page(self, quotes_uploaded_file=None):
        """Render the quotes analysis page"""
        
        # Load quotes data
        with st.spinner("Loading quotes data..."):
            quotes_df = self.data_manager.load_quotes_data(quotes_uploaded_file)
        
        if quotes_df is None:
            if quotes_uploaded_file is None:
                st.info("📁 **Welcome to the Quotes Dashboard!**")
                st.markdown("""
                To get started, please upload your **Quotes CSV file** using the file uploader in the sidebar.
                
                **Expected CSV format:**
                - Date, Account, Quote_Total, Application, Market, Channel, Customer, Year columns (required)
                - Optional: Status, State, lat, lng, City columns for enhanced analysis
                """)
                st.markdown("---")
                st.markdown("📊 Once you upload your file, you'll see interactive charts and analysis including:")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    - 📊 Quote trends and yearly analysis
                    - 🎨 Application and market breakdown
                    - 🏆 Top customer quotes with status tracking
                    - 🗺️ Geographic distribution with account markers
                    """)
                with col2:
                    st.markdown("""
                    - 📋 Quote status analysis (Won/Lost/Open)
                    - 📈 Win rate calculations
                    - 📊 Channel and customer type analysis
                    - 🔍 Advanced filtering and data exploration
                    """)
            else:
                st.error("❌ Error loading the uploaded quotes data. Please check the file format and try again.")
            return
        
        # Create sidebar filters
        st.sidebar.markdown("### 💼 Quotes Dashboard")
        filters = self.filter_manager.create_sidebar_filters(quotes_df, "quotes")
        
        # Apply filters
        filtered_df = self.filter_manager.apply_filters(quotes_df, filters)
        
        if filtered_df.empty:
            st.warning("⚠️ No data matches the selected filters. Please adjust your selections.")
            return
        
        # Date range display
        if not filtered_df.empty:
            start_year = filtered_df['Date'].min().year
            end_year = filtered_df['Date'].max().year
            st.markdown(f"### 📅 **Quotes Data from {start_year} to {end_year}**")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_quotes = filtered_df['Quote_Total'].sum()
            st.metric(
                label="💰 Total Quotes",
                value=f"${total_quotes:,.0f}",
                delta=None
            )
        
        with col2:
            total_accounts = filtered_df['Account'].nunique()
            st.metric(
                label="🏢 Unique Accounts",
                value=f"{total_accounts:,}",
                delta=None
            )
        
        with col3:
            avg_quote = filtered_df['Quote_Total'].mean()
            st.metric(
                label="📈 Average Quote",
                value=f"${avg_quote:,.0f}",
                delta=None
            )
        
        with col4:
            if 'Status' in filtered_df.columns:
                closed_quotes = len(filtered_df[filtered_df['Status'] == 'Closed'])
                total_quotes_count = len(filtered_df)
                win_rate = (closed_quotes / total_quotes_count * 100) if total_quotes_count > 0 else 0
                st.metric(
                    label="🎯 Win Rate",
                    value=f"{win_rate:.1f}%",
                    delta=None
                )
        
        st.markdown("---")
        
        # Create charts
        charts = self.chart_manager.create_quotes_charts(filtered_df, filters)
        
        # Display charts in tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 Trends", "📊 Applications", "🏢 Markets", "🥧 Customers", "🗺️ States", "📡 Channels", "📋 Status"])
        
        with tab1:
            if 'quarterly_vs_monthly' in charts:
                st.plotly_chart(charts['quarterly_vs_monthly'], use_container_width=True)
            
            if 'quotes_by_year' in charts:
                st.plotly_chart(charts['quotes_by_year'], use_container_width=True)
        
        with tab2:
            # Display both pie and bar charts for applications
            col1, col2 = st.columns(2)
            
            with col1:
                if 'quotes_by_application_pie' in charts:
                    st.plotly_chart(charts['quotes_by_application_pie'], use_container_width=True)
            
            with col2:
                if 'quotes_by_application_bar' in charts:
                    st.plotly_chart(charts['quotes_by_application_bar'], use_container_width=True)
            
            # Add quotes application heatmap section
            st.markdown("### 💼 Quotes Application Heatmap by Year")
            if 'quotes_application_heatmap' in charts:
                st.plotly_chart(charts['quotes_application_heatmap'], use_container_width=True)
                st.markdown("*Interactive heatmap showing quote intensity across applications and years. Darker colors indicate higher quote volumes.*")
            
            # Top applications table
            st.subheader("🔝 Top Applications by Quote Value")
            top_apps = filtered_df.groupby('Application')['Quote_Total'].sum().reset_index()
            top_apps = top_apps.sort_values('Quote_Total', ascending=False).head(10)
            top_apps['Quote_Total'] = top_apps['Quote_Total'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(top_apps, use_container_width=True, hide_index=True)
        
        with tab3:
            # Display both pie and bar charts for markets
            col1, col2 = st.columns(2)
            
            with col1:
                if 'quotes_by_market_pie' in charts:
                    st.plotly_chart(charts['quotes_by_market_pie'], use_container_width=True)
            
            with col2:
                if 'quotes_by_market_bar' in charts:
                    st.plotly_chart(charts['quotes_by_market_bar'], use_container_width=True)
            
            # Add quotes market heatmap section
            st.markdown("### 🎯 Quotes Market Heatmap by Year")
            if 'quotes_market_heatmap' in charts:
                st.plotly_chart(charts['quotes_market_heatmap'], use_container_width=True)
                st.markdown("*Interactive heatmap showing quote intensity across markets and years. Darker colors indicate higher quote volumes.*")
        
        with tab4:
            # Display both pie and bar charts for customers
            col1, col2 = st.columns(2)
            
            with col1:
                if 'quotes_by_customer_pie' in charts:
                    st.plotly_chart(charts['quotes_by_customer_pie'], use_container_width=True)
            
            with col2:
                if 'quotes_by_customer_bar' in charts:
                    st.plotly_chart(charts['quotes_by_customer_bar'], use_container_width=True)
            
            # Top 25 Individual Customers horizontal bar chart with status colors
            st.markdown("### 🔝 Top 25 Individual Customers by Quote Value")
            
            if 'top_customers_quotes_horizontal' in charts:
                st.plotly_chart(charts['top_customers_quotes_horizontal'], use_container_width=True)
                st.markdown("*Colors represent quote status: 🟢 Green (Closed/Won), 🔴 Red (Lost), 🟡 Yellow (Open/Pending)*")
            
            # Top customers data table
            st.subheader("📋 Top 25 Customers by Quote Value")
            if 'Status' in filtered_df.columns:
                # Create a detailed breakdown by customer and status
                top_customers_table = filtered_df.groupby(['Account', 'Status'])['Quote_Total'].sum().reset_index()
                customer_totals = filtered_df.groupby('Account')['Quote_Total'].sum().reset_index()
                top_25_accounts = customer_totals.sort_values('Quote_Total', ascending=False).head(25)['Account']
                
                # Filter for top 25 and pivot to show status breakdown
                top_customers_detail = top_customers_table[top_customers_table['Account'].isin(top_25_accounts)]
                pivot_table = top_customers_detail.pivot_table(
                    index='Account', 
                    columns='Status', 
                    values='Quote_Total', 
                    fill_value=0, 
                    aggfunc='sum'
                ).reset_index()
                
                # Add total column
                status_cols = [col for col in pivot_table.columns if col != 'Account']
                pivot_table['Total'] = pivot_table[status_cols].sum(axis=1)
                pivot_table = pivot_table.sort_values('Total', ascending=False)
                
                # Format currency columns
                for col in status_cols + ['Total']:
                    if col in pivot_table.columns:
                        pivot_table[col] = pivot_table[col].apply(lambda x: f"${x:,.0f}" if x > 0 else "-")
                
                st.dataframe(pivot_table, use_container_width=True, hide_index=True)
            else:
                # Simple table if no status column
                top_customers_simple = filtered_df.groupby('Account')['Quote_Total'].sum().reset_index()
                top_customers_simple = top_customers_simple.sort_values('Quote_Total', ascending=False).head(25)
                top_customers_simple['Quote_Total'] = top_customers_simple['Quote_Total'].apply(lambda x: f"${x:,.0f}")
                top_customers_simple.columns = ['Customer Account', 'Total Quote Value']
                top_customers_simple.index = range(1, len(top_customers_simple) + 1)
                st.dataframe(top_customers_simple, use_container_width=True)
        
        with tab5:
            if 'quotes_by_state_pie' in charts and 'quotes_by_state_bar' in charts:
                # Display both pie and bar charts for states
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(charts['quotes_by_state_pie'], use_container_width=True)
                
                with col2:
                    st.plotly_chart(charts['quotes_by_state_bar'], use_container_width=True)
                
                # Display Northeast geographical map if available
                if 'northeast_quotes_map' in charts:
                    st.markdown("### 🗺️ Northeast & Mid-Atlantic Quotes Map")
                    st.plotly_chart(charts['northeast_quotes_map'], use_container_width=True)
                    st.markdown("*Interactive map showing quote distribution across Northeast and Mid-Atlantic states*")
                else:
                    st.info("🗺️ No Northeast/Mid-Atlantic states found in the current data for geographical mapping.")
            else:
                st.info("State data not available in the dataset.")
        
        with tab6:
            # Display both pie and bar charts for channels
            col1, col2 = st.columns(2)
            
            with col1:
                if 'quotes_by_channel_pie' in charts:
                    st.plotly_chart(charts['quotes_by_channel_pie'], use_container_width=True)
            
            with col2:
                if 'quotes_by_channel_bar' in charts:
                    st.plotly_chart(charts['quotes_by_channel_bar'], use_container_width=True)
        
        with tab7:
            if 'quotes_by_status' in charts:
                st.plotly_chart(charts['quotes_by_status'], use_container_width=True)
            
            # Status breakdown table
            if 'Status' in filtered_df.columns:
                st.subheader("📊 Quote Status Breakdown")
                status_breakdown = filtered_df.groupby('Status').agg({
                    'Quote_Total': ['count', 'sum', 'mean']
                }).round(0)
                status_breakdown.columns = ['Count', 'Total Amount', 'Average Amount']
                status_breakdown['Total Amount'] = status_breakdown['Total Amount'].apply(lambda x: f"${x:,.0f}")
                status_breakdown['Average Amount'] = status_breakdown['Average Amount'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(status_breakdown, use_container_width=True)
        
        # Data table section
        with st.expander("📋 Filtered Quotes Data", expanded=False):
            display_columns = ['Date', 'Account', 'Quote_Total', 'Application', 'Market', 'Channel', 'Customer', 'Status']
            if 'Status' in filtered_df.columns:
                display_df = filtered_df[display_columns].copy()
            else:
                display_columns.remove('Status')
                display_df = filtered_df[display_columns].copy()
            display_df['Quote_Total'] = display_df['Quote_Total'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    def run(self):
        """Main application entry point"""
        try:
            # Render header
            self.render_header()
            
            # File upload section in sidebar (placed above dashboard selection)
            st.sidebar.markdown("### 📁 Upload Your Data Files")
            st.sidebar.markdown("🚀 **Get started by uploading your CSV files below:**")
            
            # Sales CSV upload
            sales_uploaded_file = st.sidebar.file_uploader(
                "📊 Sales Data (CSV)",
                type=['csv'],
                key="global_sales_uploader",
                help="Upload your sales CSV file to enable Sales Dashboard and Quota Attainment features"
            )
            
            # Show upload status for sales
            if sales_uploaded_file is not None:
                st.sidebar.success(f"✅ Sales data uploaded: {sales_uploaded_file.name}")
            else:
                st.sidebar.info("⏳ Sales Dashboard awaiting data...")
            
            # Quotes CSV upload
            quotes_uploaded_file = st.sidebar.file_uploader(
                "💼 Quotes Data (CSV)",
                type=['csv'],
                key="global_quotes_uploader",
                help="Upload your quotes CSV file to enable Quotes Dashboard features"
            )
            
            # Show upload status for quotes
            if quotes_uploaded_file is not None:
                st.sidebar.success(f"✅ Quotes data uploaded: {quotes_uploaded_file.name}")
            else:
                st.sidebar.info("⏳ Quotes Dashboard awaiting data...")
            
            st.sidebar.markdown("---")
            
            # Page navigation in sidebar
            st.sidebar.title("🎯 Dashboard Selection")
            page = st.sidebar.radio(
                "Select Dashboard:",
                options=["Sales Dashboard", "Quotes Dashboard", "🎯 Quota Attainment"],
                key="page_selector"
            )
            
            st.sidebar.markdown("---")
            
            # Render selected page
            if page == "Sales Dashboard":
                self.render_sales_page(sales_uploaded_file)
            elif page == "Quotes Dashboard":
                self.render_quotes_page(quotes_uploaded_file)
            else:
                self.render_quota_attainment_page(sales_uploaded_file)
            
            # Display timestamp
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            st.exception(e)
    
    def render_quota_attainment_page(self, sales_uploaded_file=None):
        """Render the quota attainment page"""
        st.title("🎯 Quota Attainment Dashboard")
        st.markdown("---")
        
        # Load data
        sales_df = self.data_manager.load_sales_data(sales_uploaded_file)
        
        if sales_df is None:
            if sales_uploaded_file is None:
                st.info("📁 **Welcome to the Quota Attainment Dashboard!**")
                st.markdown("""
                To get started, please upload your **Sales CSV file** using the file uploader in the sidebar.
                
                **This dashboard provides:**
                - 🎯 Quarterly and monthly quota attainment tracking
                - 📊 Performance vs targets with visual indicators
                - 📈 Multi-year comparison charts
                - 📅 Trend analysis with color-coded performance metrics
                
                **Expected CSV format:** Same as Sales Dashboard
                - Date, Account, Sales, Application, Market, Channel, Customer, Year columns (required)
                """)
            else:
                st.error("❌ Error loading the uploaded sales data for quota analysis. Please check the file format.")
            return
        
        # Load targets
        targets = QuotaAttainmentManager.create_targets_data()
        
        # Year selection
        available_years = sorted(sales_df['Year'].unique())
        
        # Multi-year selection for comparison
        st.sidebar.markdown("### 📅 Year Selection")
        selected_years = st.sidebar.multiselect(
            "Select years for comparison",
            available_years,
            default=[available_years[-1]] if available_years else [],
            help="Select multiple years to compare performance"
        )
        
        primary_year = st.sidebar.selectbox(
            "Primary year for detailed view",
            available_years,
            index=len(available_years)-1 if available_years else 0
        )
        
        if not selected_years:
            st.warning("Please select at least one year for analysis.")
            return
        
        # Performance Summary at the top
        st.markdown("### 📊 Performance Summary")
        
        total_sales = sales_df[sales_df['Year'] == primary_year]['Sales'].sum()
        total_target = sum(targets.get((str(primary_year), f'Q{q}'), 0) for q in [1, 2, 3, 4])
        total_delta = total_sales - total_target
        total_percentage = (total_sales / total_target * 100) if total_target > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label=f"Total Sales ({primary_year})",
                value=QuotaAttainmentManager.format_currency(total_sales),
                delta=None
            )
        
        with col2:
            st.metric(
                label=f"Total Target ({primary_year})",
                value=QuotaAttainmentManager.format_currency(total_target),
                delta=None
            )
        
        with col3:
            st.metric(
                label="Delta",
                value=QuotaAttainmentManager.format_currency(total_delta),
                delta=f"{total_delta:,.0f}" if total_delta != 0 else None,
                delta_color="normal"
            )
        
        with col4:
            st.metric(
                label="Attainment %",
                value=QuotaAttainmentManager.format_percentage(total_percentage),
                delta=f"{total_percentage - 100:.1f}%" if total_percentage != 100 else None,
                delta_color="normal"
            )
        
        st.markdown("---")
        
        # Year comparison table (if multiple years selected)
        if len(selected_years) > 1:
            # Create and display multi-year comparison line chart
            comparison_chart = QuotaAttainmentManager.create_multi_year_comparison_chart(sales_df, targets, selected_years)
            if comparison_chart:
                st.plotly_chart(comparison_chart, use_container_width=True)
                st.markdown("---")
            
            comparison_data = QuotaAttainmentManager.create_year_comparison_table(sales_df, targets, selected_years)
            QuotaAttainmentManager.display_comparison_table(comparison_data, selected_years)
            st.markdown("---")
        
        # Create quota attainment tables for primary year
        result = QuotaAttainmentManager.create_quota_attainment_table(sales_df, targets, primary_year)
        if result:
            quarterly_data, monthly_data = result
            
            # Display tables stacked vertically
            QuotaAttainmentManager.display_quota_table(quarterly_data, f"{primary_year} Quarterly Performance", year=primary_year, is_quarterly=True)
            
            st.markdown("---")  # Add separator between tables
            
            QuotaAttainmentManager.display_quota_table(monthly_data, f"{primary_year} Monthly Performance", year=primary_year, is_quarterly=False)
        else:
            st.error(f"No data available for {primary_year}")

class QuotaAttainmentManager:
    """Handles quota attainment dashboard functionality"""
    
    @staticmethod
    def create_targets_data():
        """Create target data - in production this could come from another CSV"""
        targets = {
            # 2024 targets
            ('2024', 'Q1'): 700000,
            ('2024', 'Q2'): 800000, 
            ('2024', 'Q3'): 850000,
            ('2024', 'Q4'): 900000,
            ('2024', '01'): 233333, ('2024', '02'): 233333, ('2024', '03'): 233334,
            ('2024', '04'): 266667, ('2024', '05'): 266666, ('2024', '06'): 266667,
            ('2024', '07'): 283333, ('2024', '08'): 283333, ('2024', '09'): 283334,
            ('2024', '10'): 300000, ('2024', '11'): 300000, ('2024', '12'): 300000,
            
            # 2025 targets
            ('2025', 'Q1'): 733333,
            ('2025', 'Q2'): 1100000, 
            ('2025', 'Q3'): 953333,
            ('2025', 'Q4'): 880000,
            ('2025', '01'): 244444, ('2025', '02'): 244444, ('2025', '03'): 244444,
            ('2025', '04'): 366667, ('2025', '05'): 366667, ('2025', '06'): 366667,
            ('2025', '07'): 317778, ('2025', '08'): 317778, ('2025', '09'): 317778,
            ('2025', '10'): 293333, ('2025', '11'): 293333, ('2025', '12'): 293333,
            
            # 2023 targets
            ('2023', 'Q1'): 600000,
            ('2023', 'Q2'): 650000, 
            ('2023', 'Q3'): 700000,
            ('2023', 'Q4'): 750000,
            ('2023', '01'): 200000, ('2023', '02'): 200000, ('2023', '03'): 200000,
            ('2023', '04'): 216667, ('2023', '05'): 216666, ('2023', '06'): 216667,
            ('2023', '07'): 233333, ('2023', '08'): 233333, ('2023', '09'): 233334,
            ('2023', '10'): 250000, ('2023', '11'): 250000, ('2023', '12'): 250000,
        }
        
        return targets

    @staticmethod
    def format_currency(value):
        """Format number as currency"""
        if pd.isna(value) or value == 0:
            return "$0"
        return f"${value:,.0f}"

    @staticmethod
    def format_percentage(value):
        """Format number as percentage"""
        if pd.isna(value):
            return "0.00%"
        return f"{value:.2f}%"

    @staticmethod
    def get_trend_arrow(current, previous):
        """Get trend arrow based on comparison with improved logic"""
        if pd.isna(current) or pd.isna(previous):
            return "➖"
        
        # Handle cases where previous is 0
        if previous == 0:
            if current > 0:
                return "▲"  # Any positive sales when previous was 0 is an increase
            else:
                return "➖"  # No change if both are 0
        
        # Calculate percentage change for more meaningful trends
        pct_change = ((current - previous) / previous) * 100
        
        if pct_change > 5:  # More than 5% increase
            return "▲"
        elif pct_change < -5:  # More than 5% decrease
            return "▼"
        else:  # Within 5% range
            return "➖"

    @staticmethod
    def create_quota_attainment_table(df, targets, year=2025):
        """Create the quota attainment table similar to the spreadsheet"""
        
        # Filter for the specified year
        df_year = df[df['Year'] == year].copy()
        
        if df_year.empty:
            st.warning(f"No sales data available for {year}")
            return [], []
        
        # Create quarterly summary
        quarterly_data = []
        
        for quarter in [1, 2, 3, 4]:
            q_data = df_year[df_year['Quarter'] == quarter]
            total_sales = q_data['Sales'].sum()
            target_key = (str(year), f'Q{quarter}')
            target = targets.get(target_key, 0)
            delta = total_sales - target
            percentage = (total_sales / target * 100) if target > 0 else 0
            
            # Calculate trend based on performance vs target and sequential comparison
            if quarter == 1:
                # For Q1, show trend based on performance vs target since no previous quarter
                if percentage >= 100:
                    trend = "▲"
                elif percentage >= 90:
                    trend = "➖"
                else:
                    trend = "▼"
            else:
                # For other quarters, compare to previous quarter in same year
                prev_q_data = df_year[df_year['Quarter'] == (quarter - 1)]
                prev_sales = prev_q_data['Sales'].sum() if not prev_q_data.empty else 0
                trend = QuotaAttainmentManager.get_trend_arrow(total_sales, prev_sales)
            
            quarterly_data.append({
                'Period': f'{year}-Q{quarter}',
                'Total_Sales': total_sales,
                'Target': target,
                'Delta': delta,
                'Percentage': percentage,
                'Trend': trend
            })
        
        # Total row
        total_sales = df_year['Sales'].sum()
        total_target = sum(targets.get((str(year), f'Q{q}'), 0) for q in [1, 2, 3, 4])
        total_delta = total_sales - total_target
        total_percentage = (total_sales / total_target * 100) if total_target > 0 else 0
        
        # For total trend, base it on whether we're meeting targets (performance-based)
        if total_percentage >= 100:
            total_trend = "▲"  # Meeting or exceeding targets
        elif total_percentage >= 90:
            total_trend = "➖"  # Close to target (neutral)
        else:
            total_trend = "▼"  # Significantly below target
        
        quarterly_data.append({
            'Period': 'Total',
            'Total_Sales': total_sales,
            'Target': total_target,
            'Delta': total_delta,
            'Percentage': total_percentage,
            'Trend': total_trend
        })
        
        # Create monthly summary
        monthly_data = []
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month_num in range(1, 13):
            m_data = df_year[df_year['Month'] == month_num]
            total_sales = m_data['Sales'].sum()
            target_key = (str(year), f'{month_num:02d}')
            target = targets.get(target_key, 0)
            delta = total_sales - target
            percentage = (total_sales / target * 100) if target > 0 else 0
            
            # Calculate trend for monthly data
            if month_num == 1:
                # For January, show trend based on performance vs target
                if percentage >= 100:
                    trend = "▲"
                elif percentage >= 90:
                    trend = "➖"
                else:
                    trend = "▼"
            else:
                # For other months, compare to previous month in same year
                prev_m_data = df_year[df_year['Month'] == (month_num - 1)]
                prev_sales = prev_m_data['Sales'].sum() if not prev_m_data.empty else 0
                trend = QuotaAttainmentManager.get_trend_arrow(total_sales, prev_sales)
            
            monthly_data.append({
                'Period': f'{month_names[month_num-1]}-{str(year)[-2:]}',
                'Total_Sales': total_sales,
                'Target': target,
                'Delta': delta,
                'Percentage': percentage,
                'Trend': trend
            })
        
        # Monthly total - use same performance-based trend as quarterly
        if total_percentage >= 100:
            monthly_total_trend = "▲"  # Meeting or exceeding targets
        elif total_percentage >= 90:
            monthly_total_trend = "➖"  # Close to target (neutral)
        else:
            monthly_total_trend = "▼"  # Significantly below target
        
        monthly_data.append({
            'Period': 'Totals',
            'Total_Sales': total_sales,
            'Target': total_target,
            'Delta': total_delta,
            'Percentage': total_percentage,
            'Trend': monthly_total_trend
        })
        
        return quarterly_data, monthly_data

    @staticmethod
    def create_quarterly_chart(data, year):
        """Create area chart for quarterly performance"""
        
        # Filter out empty rows and total row for chart
        chart_data = [row for row in data if row['Period'] and 'Q' in row['Period']]
        
        if not chart_data:
            return None
        
        periods = [row['Period'] for row in chart_data]
        sales = [row['Total_Sales'] for row in chart_data]
        targets = [row['Target'] for row in chart_data]
        
        fig = go.Figure()
        
        # Add target area (background)
        fig.add_trace(go.Scatter(
            x=periods,
            y=targets,
            fill='tonexty',
            mode='lines+markers',
            name='Target',
            line=dict(color='rgba(255, 107, 107, 0.8)', width=3),
            fillcolor='rgba(255, 107, 107, 0.2)',
            marker=dict(size=8, color='rgba(255, 107, 107, 0.8)')
        ))
        
        # Add actual sales area (foreground)
        fig.add_trace(go.Scatter(
            x=periods,
            y=sales,
            fill='tonexty',
            mode='lines+markers',
            name='Total Sales',
            line=dict(color='rgba(64, 229, 64, 0.8)', width=3),
            fillcolor='rgba(64, 229, 64, 0.2)',
            marker=dict(size=8, color='rgba(64, 229, 64, 0.8)')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{year} Quarterly Performance - Sales vs Target',
            xaxis_title='Quarter',
            yaxis_title='Amount ($)',
            hovermode='x unified',
            template='plotly_dark',
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(tickformat='$,.0f')  # Format y-axis as currency
        )
        
        # Add hover template
        fig.update_traces(
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Quarter: %{x}<br>' +
                         'Amount: $%{y:,.0f}<br>' +
                         '<extra></extra>'
        )
        
        return fig

    @staticmethod
    def create_monthly_chart(data, year):
        """Create area chart for monthly performance"""
        
        # Filter out empty rows and total row for chart - look for month patterns
        chart_data = [row for row in data if row['Period'] and '-' in row['Period'] and 'Total' not in row['Period']]
        
        if not chart_data:
            return None
        
        periods = [row['Period'] for row in chart_data]
        sales = [row['Total_Sales'] for row in chart_data]
        targets = [row['Target'] for row in chart_data]
        
        fig = go.Figure()
        
        # Add target area (background)
        fig.add_trace(go.Scatter(
            x=periods,
            y=targets,
            fill='tonexty',
            mode='lines+markers',
            name='Target',
            line=dict(color='rgba(255, 107, 107, 0.8)', width=2),
            fillcolor='rgba(255, 107, 107, 0.2)',
            marker=dict(size=6, color='rgba(255, 107, 107, 0.8)')
        ))
        
        # Add actual sales area (foreground)
        fig.add_trace(go.Scatter(
            x=periods,
            y=sales,
            fill='tonexty',
            mode='lines+markers',
            name='Total Sales',
            line=dict(color='rgba(64, 229, 64, 0.8)', width=2),
            fillcolor='rgba(64, 229, 64, 0.2)',
            marker=dict(size=6, color='rgba(64, 229, 64, 0.8)')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{year} Monthly Performance - Sales vs Target',
            xaxis_title='Month',
            yaxis_title='Amount ($)',
            hovermode='x unified',
            template='plotly_dark',
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(tickformat='$,.0f'),  # Format y-axis as currency
            xaxis=dict(tickangle=45)  # Angle month labels for better readability
        )
        
        # Add hover template
        fig.update_traces(
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Month: %{x}<br>' +
                         'Amount: $%{y:,.0f}<br>' +
                         '<extra></extra>'
        )
        
        return fig

    @staticmethod
    def display_quota_table(data, title, year=None, is_quarterly=True):
        """Display quota attainment table with styling and optional chart"""
        
        st.markdown(f"### {title}")
        
        # Add area chart for quarterly or monthly data
        if year:
            if is_quarterly:
                chart = QuotaAttainmentManager.create_quarterly_chart(data, year)
            else:
                chart = QuotaAttainmentManager.create_monthly_chart(data, year)
            
            if chart:
                st.plotly_chart(chart, use_container_width=True)
                st.markdown("---")  # Separator between chart and table
        
        # Convert data to DataFrame for better display
        df = pd.DataFrame(data)
        
        # Format the data for display
        display_df = df.copy()
        display_df['Total Sales'] = display_df['Total_Sales'].apply(QuotaAttainmentManager.format_currency)
        display_df['Target'] = display_df['Target'].apply(QuotaAttainmentManager.format_currency)
        display_df['Delta'] = display_df['Delta'].apply(QuotaAttainmentManager.format_currency)
        display_df['Percentage'] = display_df['Percentage'].apply(QuotaAttainmentManager.format_percentage)
        
        # Select and rename columns for display
        display_df = display_df[['Period', 'Total Sales', 'Target', 'Delta', 'Percentage', 'Trend']]
        
        # Apply conditional formatting function
        def highlight_rows(row):
            styles = [''] * len(row)
            
            # Highlight total rows (darker red for visibility)
            if row['Period'] in ['Total', 'Totals']:
                styles = ['background-color: #8b2635; color: white; font-weight: bold'] * len(row)
            # Highlight quarterly rows (darker green for visibility)
            elif is_quarterly and 'Q' in str(row['Period']):
                styles = ['background-color: #2d5a37; color: white; font-weight: bold'] * len(row)
            # Empty rows (darker gray for visibility)
            elif row['Period'] == '':
                styles = ['background-color: #4a5568; color: white'] * len(row)
            
            return styles
        
        def color_values(val):
            """Color positive/negative values - optimized for dark background"""
            if isinstance(val, str):
                if '$-' in val:  # Negative currency
                    return 'color: #ff6b6b; font-weight: bold'
                elif '$' in val and val != '$0':  # Positive currency
                    if '-' not in val:
                        return 'color: #40e540; font-weight: bold'
                elif '%' in val:  # Percentage
                    try:
                        pct_val = float(val.replace('%', ''))
                        if pct_val >= 100:
                            return 'color: #40e540; font-weight: bold'
                        else:
                            return 'color: #ff6b6b; font-weight: bold'
                    except:
                        pass
                elif val in ['▲']:
                    return 'color: #40e540; font-weight: bold'
                elif val in ['▼']:
                    return 'color: #ff6b6b; font-weight: bold'
            return ''
        
        # Apply styling
        styled_df = display_df.style.apply(highlight_rows, axis=1).applymap(color_values)
        
        # Display the styled dataframe
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Period": st.column_config.TextColumn("Period", width="medium"),
                "Total Sales": st.column_config.TextColumn("Total Sales", width="medium"),
                "Target": st.column_config.TextColumn("Target", width="medium"),
                "Delta": st.column_config.TextColumn("Delta", width="medium"),
                "Percentage": st.column_config.TextColumn("Percentage", width="medium"),
                "Trend": st.column_config.TextColumn("Trend", width="small")
            }
        )

    @staticmethod
    def create_multi_year_comparison_chart(sales_df, targets, selected_years):
        """Create multi-year quarterly comparison line chart showing only total sales"""
        if len(selected_years) <= 1:
            return None
        
        fig = go.Figure()
        
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        
        # Color palette for different years
        colors = [
            '#40e540',  # Green
            '#ff6b6b',  # Red
            '#ffd93d',  # Yellow
            '#6c5ce7',  # Purple
            '#00cec9',  # Teal
            '#fd79a8',  # Pink
            '#fdcb6e',  # Orange
            '#74b9ff'   # Blue
        ]
        
        for i, year in enumerate(selected_years):
            year_data = sales_df[sales_df['Year'] == year]
            
            # Calculate quarterly sales
            quarterly_sales = []
            
            for quarter in [1, 2, 3, 4]:
                q_data = year_data[year_data['Quarter'] == quarter]
                total_sales = q_data['Sales'].sum()
                quarterly_sales.append(total_sales)
            
            # Color index with wrap-around
            color_idx = i % len(colors)
            
            # Add sales line
            fig.add_trace(go.Scatter(
                x=quarters,
                y=quarterly_sales,
                mode='lines+markers',
                name=f'{year} Sales',
                line=dict(color=colors[color_idx], width=3),
                marker=dict(size=8, color=colors[color_idx]),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Quarter: %{x}<br>' +
                             'Sales: $%{y:,.0f}<br>' +
                             '<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title='Multi-Year Quarterly Sales Comparison',
            xaxis_title='Quarter',
            yaxis_title='Total Sales ($)',
            hovermode='x unified',
            template='plotly_dark',
            height=500,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            margin=dict(l=0, r=120, t=50, b=0),
            yaxis=dict(tickformat='$,.0f'),
            showlegend=True
        )
        
        return fig

    @staticmethod
    def create_year_comparison_table(sales_df, targets, selected_years):
        """Create year-over-year comparison table for quarterly performance"""
        comparison_data = []
        
        for quarter in [1, 2, 3, 4]:
            row_data = {'Quarter': f'Q{quarter}'}
            
            for year in selected_years:
                year_data = sales_df[sales_df['Year'] == year]
                q_data = year_data[year_data['Quarter'] == quarter]
                total_sales = q_data['Sales'].sum()
                
                target_key = (str(year), f'Q{quarter}')
                target = targets.get(target_key, 0)
                attainment = (total_sales / target * 100) if target > 0 else 0
                
                row_data[f'{year} Sales'] = total_sales
                row_data[f'{year} Target'] = target
                row_data[f'{year} Attainment'] = attainment
            
            comparison_data.append(row_data)
        
        # Add totals row
        totals_row = {'Quarter': 'Total'}
        for year in selected_years:
            year_data = sales_df[sales_df['Year'] == year]
            total_sales = year_data['Sales'].sum()
            total_target = sum(targets.get((str(year), f'Q{q}'), 0) for q in [1, 2, 3, 4])
            total_attainment = (total_sales / total_target * 100) if total_target > 0 else 0
            
            totals_row[f'{year} Sales'] = total_sales
            totals_row[f'{year} Target'] = total_target
            totals_row[f'{year} Attainment'] = total_attainment
        
        comparison_data.append(totals_row)
        
        return comparison_data

    @staticmethod
    def display_comparison_table(data, selected_years):
        """Display year comparison table"""
        st.markdown("### 📊 Year-over-Year Quarterly Comparison")
        
        df = pd.DataFrame(data)
        
        # Format the display columns
        display_df = df.copy()
        for year in selected_years:
            display_df[f'{year} Sales'] = display_df[f'{year} Sales'].apply(QuotaAttainmentManager.format_currency)
            display_df[f'{year} Target'] = display_df[f'{year} Target'].apply(QuotaAttainmentManager.format_currency)
            display_df[f'{year} Attainment'] = display_df[f'{year} Attainment'].apply(lambda x: QuotaAttainmentManager.format_percentage(x))
        
        def highlight_comparison_rows(row):
            styles = [''] * len(row)
            if row['Quarter'] == 'Total':
                styles = ['background-color: #8b2635; color: white; font-weight: bold'] * len(row)
            return styles
        
        def color_comparison_values(val):
            if isinstance(val, str):
                if '%' in val:
                    try:
                        pct_val = float(val.replace('%', ''))
                        if pct_val >= 100:
                            return 'color: #40e540; font-weight: bold'
                        else:
                            return 'color: #ff6b6b; font-weight: bold'
                    except:
                        pass
            return ''
        
        styled_df = display_df.style.apply(highlight_comparison_rows, axis=1).applymap(color_comparison_values)
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )

def main():
    """Application entry point with error handling"""
    try:
        app = DashboardApp()
        app.run()
    except Exception as e:
        st.error("Critical application error occurred. Please refresh the page.")
        st.exception(e)

if __name__ == "__main__":
    main()