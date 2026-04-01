import os
from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from main import (
    infer_species_from_historical_df,
    extract_key_info,
    habitat_risk_analysis,
    population_trend_model,
    generate_report_and_recommendations,
    visualize_population_trends,
    save_report_to_pdf,
)

st.set_page_config(
    page_title="AI Conservation Report Generator",
    page_icon="🌿",
    layout="wide",
)

# Custom CSS matching the design
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      
      html, body, [data-testid="stAppViewContainer"] {
        background: #f8f8f8;
        font-family: 'Inter', sans-serif;
      }
      
      [data-testid="stHeader"] { 
        background: transparent; 
      }
      
      .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #000;
        margin: 1rem 0 2rem 0;
        padding: 0.5rem;
      }
      
      .leaf-icon {
        color: #5a9f6a;
        font-size: 2.2rem;
        margin-right: 0.5rem;
      }
      
      .stTextInput > div > div > input,
      .stTextArea > div > div > textarea {
        border: 1px solid #ccc !important;
        border-radius: 4px !important;
        padding: 12px !important;
        font-size: 1rem !important;
        background: white !important;
        color: #000 !important;
      }
      
      .stTextInput > label,
      .stTextArea > label,
      .stFileUploader > label {
        display: none !important;
      }
      
      .stButton > button {
        width: 100%;
        background: #4a8f5a !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 14px 24px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        margin-top: 1rem !important;
      }
      
      .stButton > button:hover {
        background: #3d7a4a !important;
      }
      
      [data-testid="stFileUploader"] {
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 1rem;
        background: white;
        margin: 0.5rem 0;
      }
      
      .report-card {
        background: white;
        border-radius: 8px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
      }
      
      .report-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #000;
        margin-bottom: 1.5rem;
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        padding: 12px 20px;
        border-radius: 8px;
        display: inline-block;
      }
      
      .report-row {
        text-align: left;
        margin: 1rem 0;
        font-size: 1.1rem;
      }
      
      .report-label {
        font-weight: 700;
        color: #000;
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-right: 8px;
      }
      
      .report-value {
        color: #333;
      }
      
      .download-btn {
        margin-top: 2rem;
      }
      
      .block-container {
        max-width: 900px;
        padding: 2rem 3rem;
      }
      
      .input-label {
        font-size: 1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 8px;
        margin-top: 15px;
        display: block;
      }
      
      /* Style for tabs */
      .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f0f0f0;
        padding: 10px;
        border-radius: 8px;
      }
      
      .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        border: 2px solid #ddd;
      }
      
      .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%) !important;
        color: #000 !important;
        border: 2px solid #4a8f5a !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown(
    """
    <div class="main-title">
        <span class="leaf-icon">🌿</span>
        AI Conservation Report Generator
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# Input Form
st.markdown('<p class="input-label">📍 Location</p>', unsafe_allow_html=True)
location = st.text_input("", placeholder="e.g., KBR National Park, Hyderabad", key="location", label_visibility="collapsed", value="")

st.markdown('<p class="input-label">🌿 Species Observed</p>', unsafe_allow_html=True)
species = st.text_input("", placeholder="e.g., tiger, elephant, deer", key="species", label_visibility="collapsed", value="")

st.markdown('<p class="input-label">⚠️ Threats Observed</p>', unsafe_allow_html=True)
threats = st.text_area("", placeholder="e.g., Deforestation, Poaching, Habitat Loss", height=100, key="threats", label_visibility="collapsed", value="")

st.markdown('<p class="input-label">📁 Upload Historical Data (CSV)</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=['csv'], label_visibility="collapsed")

# Show uploaded file info
if uploaded_file is not None:
    st.markdown(f"""
    <div style='background: #e8f5e9; 
                border-left: 4px solid #4a8f5a; 
                padding: 10px 15px; 
                border-radius: 4px; 
                margin: 10px 0;'>
        <p style='margin: 0; color: #2e7d32;'>
            ✅ <strong>File uploaded:</strong> {uploaded_file.name}
        </p>
    </div>
    """, unsafe_allow_html=True)

# Show CSV format help with custom styling
st.markdown("""
<div style='background: linear-gradient(120deg, #fff9e6 0%, #ffe6f0 100%); 
            border: 2px solid #ffd700; 
            border-radius: 8px; 
            padding: 15px; 
            margin: 15px 0;'>
    <h4 style='margin: 0 0 10px 0; color: #d97706;'>ℹ️ CSV Format Help</h4>
    <p style='margin: 5px 0; color: #333;'>
        Your CSV file must have:
    </p>
    <ul style='margin: 5px 0; color: #333;'>
        <li>A <strong>Year</strong> column (required)</li>
        <li>One or more species population columns</li>
    </ul>
    <p style='margin: 10px 0 5px 0; color: #333;'><strong>Example format:</strong></p>
    <pre style='background: #f5f5f5; padding: 10px; border-radius: 4px; color: #000;'>Year,Tiger_Population,Deer_Population
2020,45,120
2021,48,125
2022,52,130</pre>
    <p style='margin: 10px 0 0 0; color: #333;'>
        <strong>Sample files available:</strong><br>
        • <code>historical_population_data.csv</code> (Zebra, Wild Dogs, Giraffes)<br>
        • <code>tiger_population_data.csv</code> (Tiger, Deer, Wild Boar)
    </p>
</div>
""", unsafe_allow_html=True)

analyze_button = st.button("🔍 Analyze Biodiversity")

# Process Analysis
if analyze_button:
    if not location or not species or uploaded_file is None:
        st.error("⚠️ Please fill in all fields and upload a CSV file.")
    else:
        with st.spinner("🔄 Analyzing biodiversity data..."):
            try:
                # Parse species input
                species_list = [s.strip() for s in species.split(',')]
                species_sightings = []
                threat_level = "Low"
                
                # Simple parsing - if numbers provided use them, otherwise default
                for sp in species_list:
                    if ':' in sp:
                        parts = sp.split(':')
                        species_name = parts[0].strip()
                        count_str = parts[1].strip()
                        try:
                            count = int(''.join(filter(str.isdigit, count_str)))
                            species_sightings.append(f"**{count}** **{species_name}**")
                        except:
                            species_sightings.append(f"**50** **{species_name}**")
                    else:
                        species_sightings.append(f"**50** **{sp}**")
                
                # Determine threat level
                threats_lower = threats.lower()
                if any(word in threats_lower for word in ['critical', 'severe', 'poaching', 'snare']):
                    threat_level = "High"
                elif any(word in threats_lower for word in ['moderate', 'concern', 'declining', 'deforestation']):
                    threat_level = "Medium"
                
                # Build field notes
                field_notes = f"""FIELD REPORT {datetime.now().strftime('%Y-%m-%d')}
Location: {location}
Observer: Field Researcher

Notes: {' '.join(species_sightings)}. The habitat condition is **Fair**. Threat level is **{threat_level}**.
{threats}
"""
                
                # Load historical data
                historical_df = pd.read_csv(uploaded_file)
                
                if 'Year' not in historical_df.columns:
                    st.error(f"❌ CSV must contain a 'Year' column. Found columns: {', '.join(historical_df.columns.tolist())}")
                    st.info("💡 Please ensure your CSV has a 'Year' column with years and species population columns.")
                else:
                    # Run analysis
                    inferred_species = infer_species_from_historical_df(historical_df)
                    extraction = extract_key_info(field_notes, expected_species=inferred_species)
                    risk_score = habitat_risk_analysis(extraction["habitat_condition"], extraction["threat_level"])
                    trend_df = population_trend_model(historical_df, extraction["species_sightings"])
                    report_text = generate_report_and_recommendations(extraction, trend_df, risk_score)
                    
                    # Store results
                    st.session_state.analysis_result = {
                        "extraction": extraction,
                        "risk_score": risk_score,
                        "trend_df": trend_df,
                        "report_text": report_text,
                        "inferred_species": inferred_species
                    }
                    
                    st.success("✅ Analysis complete!")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Display Results
if st.session_state.analysis_result:
    analysis = st.session_state.analysis_result
    extraction = analysis["extraction"]
    risk_score = analysis["risk_score"]
    trend_df = analysis["trend_df"]
    report_text = analysis["report_text"]
    
    # Biodiversity Score calculation
    biodiversity_score = "High" if risk_score < 50 else ("Medium" if risk_score < 70 else "Low")
    
    # Report Card
    st.markdown(
        f"""
        <div class="report-card">
            <div class="report-title">
                🌍 Biodiversity Analysis Report
            </div>
            <div class="report-row">
                <span class="report-label">Location:</span>
                <span class="report-value">{extraction['location']}</span>
            </div>
            <div class="report-row">
                <span class="report-label">Species Observed:</span>
                <span class="report-value">{species}</span>
            </div>
            <div class="report-row">
                <span class="report-label">Threats:</span>
                <span class="report-value">{threats}</span>
            </div>
            <div class="report-row">
                <span class="report-label">Biodiversity Score:</span>
                <span class="report-value">{biodiversity_score}</span>
            </div>
            <div class="report-row">
                <span class="report-label">Threat Level:</span>
                <span class="report-value">{extraction['threat_level']}</span>
            </div>
            <div class="report-row">
                <span class="report-label">Recommendation:</span>
                <span class="report-value">Protect habitat and monitor species population regularly.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Download PDF Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📄 Download PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                try:
                    os.makedirs("Output_Reports", exist_ok=True)
                    temp_plot = os.path.join("Output_Reports", "temp_plot.png")
                    visualize_population_trends(trend_df, save_path=temp_plot)
                    
                    pdf_name = f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    save_report_to_pdf(report_text, temp_plot, pdf_name)
                    pdf_path = os.path.join("Output_Reports", pdf_name)
                    
                    if os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "📥 Download PDF",
                                data=BytesIO(f.read()),
                                file_name=pdf_name,
                                mime="application/pdf",
                                use_container_width=True
                            )
                    
                    if os.path.exists(temp_plot):
                        os.remove(temp_plot)
                except Exception as e:
                    st.error(f"PDF error: {str(e)}")
    
    # Visualizations Section
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%); 
                padding: 12px 20px; 
                border-radius: 8px; 
                margin-bottom: 20px;'>
        <h2 style='margin: 0; color: #000;'>📊 Population Trends & Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📈 Trend Analysis", "📊 Species Comparison", "🎯 Risk Assessment"])
    
    with viz_tab1:
        st.markdown("""
        <div style='background: linear-gradient(120deg, #e8f5e9 0%, #e3f2fd 100%); 
                    padding: 10px 15px; 
                    border-radius: 6px; 
                    margin-bottom: 15px;'>
            <h3 style='margin: 0; color: #2e7d32;'>📈 Population Trends Over Time</h3>
        </div>
        """, unsafe_allow_html=True)
        
        fig_line = px.line(
            trend_df, 
            x="Year", 
            y="Population", 
            color="Species",
            markers=True,
            title="Historical Population Trends"
        )
        fig_line.update_traces(line=dict(width=3), marker=dict(size=10))
        fig_line.update_layout(
            height=500,
            hovermode="x unified",
            template="plotly_white"
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    with viz_tab2:
        st.markdown("""
        <div style='background: linear-gradient(120deg, #e8f5e9 0%, #e3f2fd 100%); 
                    padding: 10px 15px; 
                    border-radius: 6px; 
                    margin-bottom: 15px;'>
            <h3 style='margin: 0; color: #2e7d32;'>📊 Species Population Comparison</h3>
        </div>
        """, unsafe_allow_html=True)
        
        current_year = trend_df['Year'].max()
        latest_data = trend_df[trend_df['Year'] == current_year]
        
        fig_bar = px.bar(
            latest_data,
            x="Species",
            y="Population",
            color="Species",
            title=f"Current Population Distribution ({current_year})",
            text="Population"
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(
            showlegend=False, 
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with viz_tab3:
        st.markdown("""
        <div style='background: linear-gradient(120deg, #e8f5e9 0%, #e3f2fd 100%); 
                    padding: 10px 15px; 
                    border-radius: 6px; 
                    margin-bottom: 15px;'>
            <h3 style='margin: 0; color: #2e7d32;'>🎯 Habitat Risk Assessment</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Risk Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={'text': "Risk Score (0-100)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#4a8f5a"},
                    'steps': [
                        {'range': [0, 40], 'color': "#e8f5e9"},
                        {'range': [40, 70], 'color': "#fff9c4"},
                        {'range': [70, 100], 'color': "#ffcdd2"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(height=350)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col_b:
            # Risk Metrics
            st.metric("Threat Level", extraction['threat_level'])
            st.metric("Habitat Condition", extraction['habitat_condition'])
            st.metric("Biodiversity Score", biodiversity_score)
            
            risk_status = "CRITICAL" if risk_score > 70 else ("HIGH" if risk_score > 40 else "MODERATE")
            st.metric("Overall Status", risk_status)
    
    # Detailed Report Section
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%); 
                padding: 12px 20px; 
                border-radius: 8px; 
                margin-bottom: 20px;'>
        <h2 style='margin: 0; color: #000;'>📄 Detailed Conservation Report</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Show report directly (not in expander)
    st.text_area("Full Report", value=report_text, height=400)
    
    # Download buttons
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        report_name = f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        st.download_button(
            "📥 Download Report (TXT)",
            data=report_text.encode("utf-8"),
            file_name=report_name,
            mime="text/plain",
            use_container_width=True
        )
    
    with col_dl2:
        if st.button("📄 Generate & Download PDF", use_container_width=True, key="pdf_bottom"):
            with st.spinner("Generating PDF..."):
                try:
                    os.makedirs("Output_Reports", exist_ok=True)
                    temp_plot = os.path.join("Output_Reports", "temp_plot.png")
                    visualize_population_trends(trend_df, save_path=temp_plot)
                    
                    pdf_name = f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    save_report_to_pdf(report_text, temp_plot, pdf_name)
                    pdf_path = os.path.join("Output_Reports", pdf_name)
                    
                    if os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "📥 Download PDF",
                                data=BytesIO(f.read()),
                                file_name=pdf_name,
                                mime="application/pdf",
                                use_container_width=True,
                                key="pdf_download_bottom"
                            )
                    
                    if os.path.exists(temp_plot):
                        os.remove(temp_plot)
                except Exception as e:
                    st.error(f"PDF error: {str(e)}")
