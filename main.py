import pandas as pd
import numpy as np
import os
import random
import textwrap
import re
from typing import List
from io import StringIO

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    letter = None
    SimpleDocTemplate = None
    Paragraph = None
    Spacer = None
    Image = None
    getSampleStyleSheet = None
    ParagraphStyle = None
    inch = None
    TA_CENTER = None
    REPORTLAB_AVAILABLE = False

try:
    import spacy
except ImportError:
    spacy = None

# --- 1. SETUP AND UTILITY FUNCTIONS ---

# --- LLM SETUP ---
LLM_MODEL_NAME = "distilgpt2"
try:
    from transformers import pipeline
    generator = pipeline(
        'text-generation', 
        model=LLM_MODEL_NAME, 
        device=-1, # Use CPU
        do_sample=True,
        max_new_tokens=256,
        temperature=0.7
    )
    LLM_AVAILABLE = True
    print(f"\n[INFO] Successfully loaded open-source LLM: {LLM_MODEL_NAME}")
except Exception as e:
    print(f"\n[ERROR] Could not load the LLM ({LLM_MODEL_NAME}).")
    LLM_AVAILABLE = False
# --- END LLM SETUP ---


# Try to load a simple spaCy model for data extraction.
try:
    # Run: python -m spacy download en_core_web_sm
    if spacy is not None:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    else:
        raise ImportError("spaCy package not installed")
except (OSError, ImportError, AttributeError):
    nlp = None
    SPACY_AVAILABLE = False


def create_example_data_files():
    """Creates a sample field notes file and a historical population data CSV."""
    if not os.path.exists("Output_Reports"):
        os.makedirs("Output_Reports")

    # 1. Sample Field Notes (Text File)
    # Reverting to the most natural phrasing for testing the new regex.
    sample_notes = textwrap.dedent(f"""
    FIELD REPORT 2024-10-03
    Location: Mzuri Savannah, Sector Alpha
    Time: 08:30 - 11:00
    Observer: Dr. K. Njoroge

    Notes: Initial observations confirm a healthy number of **Grevy's Zebra**. Counted **58** near the main watering hole. Activity was moderate. Noted signs of human interference, specifically **2** discarded snare wires near the eastern boundary, suggesting a **high** threat level. Also, saw a small group of **28** critically endangered **Rothschild's Giraffes** passing through. Only observed **10** **African Wild Dogs** in the riverbed. The habitat condition is **Fair**.
    """)
    with open("sample_field_notes.txt", "w") as f:
        f.write(sample_notes.strip())

    # 2. Historical Population Data (CSV)
    years = np.arange(2016, 2026)
    data = {
        'Year': years,
        "Grevy's Zebra_Population": (np.random.normal(50, 5, size=10) + np.arange(10) * 1).astype(int),
        "African Wild Dogs_Population": (np.random.normal(15, 2, size=10) - np.arange(10) * 0.5).astype(int),
        "Rothschild's Giraffes_Population": (np.random.normal(25, 3, size=10) + np.arange(10) * 0.2).astype(int)
    }
    for col in data:
        if 'Population' in col:
            data[col] = np.maximum(data[col], 5)

    df_hist = pd.DataFrame(data)
    df_hist.to_csv("historical_population_data.csv", index=False)

    print("\n--- INFO: Generated Sample Data Files ---")
    print("Created 'sample_field_notes.txt' (Field Data Input)")
    print("Created 'historical_population_data.csv' (Historical Data Input)")
    print("An 'Output_Reports' folder has been created for export.")
    print("-" * 40)

# --- 2. DATA COLLECTION AND NLP PROCESSING (PHASE 1 & 2) ---

def _column_to_species(column_name: str) -> str:
    """Maps a historical CSV column name to a species label."""
    metric_suffixes = [
        "_Population",
        "_Count",
        "_Biomass_mg_m3",
        "_Biomass",
        "_Abundance"
    ]
    name = column_name.strip()
    for suffix in metric_suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    if "_" in name:
        name = name.split("_", 1)[0]
    return name.replace("_", " ").strip()


def infer_species_from_historical_df(historical_df: pd.DataFrame) -> List[str]:
    """Infers species labels from historical dataframe columns."""
    species = []
    for column in historical_df.columns:
        if column == "Year":
            continue
        parsed = _column_to_species(column)
        if parsed and parsed not in species:
            species.append(parsed)
    return species


def _species_aliases(species_name: str) -> List[str]:
    """Returns simple alias variants to handle singular/plural mismatch."""
    aliases = [species_name]
    if species_name.endswith("es") and len(species_name) > 2:
        aliases.append(species_name[:-2])
    if species_name.endswith("s") and len(species_name) > 1:
        aliases.append(species_name[:-1])
    return list(dict.fromkeys([alias.strip() for alias in aliases if alias.strip()]))


def _find_count_near_alias(notes_text: str, alias: str, window: int = 180):
    """
    Finds the nearest numeric value around a species mention.
    This handles cases where count and species are in adjacent sentences.
    """
    escaped_alias = re.escape(alias)
    candidates = []

    for match in re.finditer(rf"\b{escaped_alias}\b", notes_text, re.IGNORECASE):
        start, end = match.span()
        local_start = max(0, start - window)
        local_end = min(len(notes_text), end + window)
        snippet = notes_text[local_start:local_end]
        species_pos = start - local_start

        for num_match in re.finditer(r"\d+(?:\.\d+)?", snippet):
            token = num_match.group(0)
            nstart, nend = num_match.span()
            before = snippet[nstart - 1] if nstart > 0 else ""
            after = snippet[nend] if nend < len(snippet) else ""

            # Ignore numeric fragments likely from date/time formatting.
            if before in (":", "-") or after in (":", "-"):
                continue

            value = float(token)
            if len(token) == 4 and 1900 <= value <= 2100:
                continue

            distance = min(abs(nstart - species_pos), abs(nend - species_pos))
            candidates.append((distance, value))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return int(round(candidates[0][1]))


def extract_key_info(notes_text: str, expected_species: List[str] = None) -> dict:
    """Extracts structured data from unstructured notes using dataset-driven species matching."""
    extracted_data = {
        "species_sightings": [],
        "location": "Unknown",
        "observer": "Unknown",
        "date": "Unknown",
        "threat_level": "Low",
        "habitat_condition": "Unknown"
    }

    notes_lines = notes_text.split("\n")
    normalized_notes = notes_text.replace("**", "")
    normalized_lower = normalized_notes.lower()

    # 1. Core metadata extraction
    for line in notes_lines:
        if "Location:" in line:
            extracted_data["location"] = line.split("Location:", 1)[1].strip()
        elif "Observer:" in line:
            extracted_data["observer"] = line.split("Observer:", 1)[1].strip()
        elif "FIELD REPORT" in line:
            extracted_data["date"] = line.split("FIELD REPORT", 1)[1].strip()

    # 2. Species extraction using expected species from CSV columns
    default_species = ["Grevy's Zebra", "Rothschild's Giraffes", "African Wild Dogs"]
    species_to_scan = expected_species if expected_species else default_species

    for species_name in species_to_scan:
        count_value = None
        for alias in _species_aliases(species_name):
            found = _find_count_near_alias(normalized_notes, alias)
            if found is not None:
                count_value = found
                break

        if count_value is not None:
            extracted_data["species_sightings"].append({
                "species": species_name,
                "count": count_value
            })

    # 3. Threat and habitat extraction
    threat_match = re.search(r"\b(low|medium|high|critical)\s+threat level\b", normalized_lower)
    if threat_match:
        level = threat_match.group(1).capitalize()
        extracted_data["threat_level"] = "High" if level == "Critical" else level
    elif "critically endangered" in normalized_lower or "snare wires" in normalized_lower or "poaching" in normalized_lower:
        extracted_data["threat_level"] = "High"
    elif "healthy number" in normalized_lower:
        extracted_data["threat_level"] = "Medium"

    habitat_match = re.search(
        r"habitat condition is\s*([a-zA-Z]+)",
        normalized_lower,
        re.IGNORECASE
    )
    if habitat_match:
        extracted_data["habitat_condition"] = habitat_match.group(1).capitalize()

    return extracted_data

# --- 3. DATA SCIENCE MODELING (PHASE 4) ---

def habitat_risk_analysis(habitat_condition: str, threat_level: str) -> float:
    """Calculates a numerical habitat risk score (0 to 100)."""
    score = 0
    if habitat_condition.lower() == 'critical': score += 40
    if habitat_condition.lower() == 'poor': score += 30
    elif habitat_condition.lower() == 'fair': score += 15

    if threat_level.lower() == 'high': score += 40
    elif threat_level.lower() == 'medium': score += 20

    score += random.randint(0, 10)
    return min(score, 100)

def population_trend_model(historical_df: pd.DataFrame, current_sightings: list) -> pd.DataFrame:
    """Integrates current sightings with historical data and projects a short-term trend."""
    current_year = pd.to_datetime('today').year
    df_long = pd.melt(historical_df, id_vars=['Year'], var_name='Species_Metric', value_name='Population')
    df_long['Species'] = df_long['Species_Metric'].apply(_column_to_species)
    df_long = df_long[['Year', 'Species', 'Population']]
    df_long['Population'] = pd.to_numeric(df_long['Population'], errors='coerce').fillna(0).astype(int)

    current_data = []
    for s in current_sightings:
        current_data.append({'Year': current_year, 'Species': s['species'], 'Population': int(s['count'])})

    df_combined = pd.concat([df_long, pd.DataFrame(current_data)], ignore_index=True)
    df_combined.drop_duplicates(subset=['Year', 'Species'], keep='last', inplace=True) 

    projected_data = []
    for species in df_combined['Species'].unique():
        species_df = df_combined[df_combined['Species'] == species].sort_values('Year')
        if species_df.shape[0] >= 2:
            years = species_df['Year'].values
            pops = species_df['Population'].values # pops is a NumPy array
            
            if len(np.unique(years)) > 1:
                slope, _ = np.polyfit(years, pops, 1)
                last_pop = pops[-1]
                next_pop = int(max(0, last_pop + slope))
            else:
                last_pop = pops[-1]
                next_pop = last_pop + random.randint(-5, 5)
        else:
            # This branch handles cases where only one historical or only current data exists
            last_pop = species_df['Population'].iloc[-1] if not species_df.empty else 0
            next_pop = last_pop + random.randint(-2, 2)

        projected_data.append({'Year': current_year + 1, 'Species': species, 'Population': next_pop})

    df_final = pd.concat([df_combined, pd.DataFrame(projected_data)], ignore_index=True)
    df_final.sort_values(by=['Species', 'Year'], inplace=True)
    return df_final.astype({'Year': int, 'Population': int})

def visualize_population_trends(population_df: pd.DataFrame, save_path=None):
    """Generates and displays/saves the population trend plot."""
    if plt is None:
        print("[WARNING] Matplotlib is not installed. Skipping visualization step.")
        return False

    plt.figure(figsize=(10, 6))
    for species in population_df['Species'].unique():
        df_species = population_df[population_df['Species'] == species]
        plt.plot(df_species['Year'], df_species['Population'], marker='o', label=species)

    plt.title('Wildlife Species Population Trend Analysis (Historical + Projection)')
    plt.xlabel('Year')
    plt.ylabel('Estimated Population Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Species')
    plt.xticks(population_df['Year'].unique(), rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close() # Close plot to free memory
    else:
        plt.show()
    return True

# --- 4. LLM REPORTING AND RECOMMENDATIONS (PHASE 3 & 5) ---

def generate_llm_text(prompt: str, max_length: int) -> str:
    """Uses the loaded LLM to generate text."""
    if not LLM_AVAILABLE:
        return "LLM Not Available. (Fallback to simple text)"

    output = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=generator.tokenizer.eos_token_id,
        truncation=True
    )
    generated_text = output[0]['generated_text'].replace(prompt, '').strip()
    return generated_text

def generate_report_and_recommendations(extraction_data: dict, trend_df: pd.DataFrame, risk_score: float) -> str:
    """
    Uses the LLM to generate the Conservation Analysis and Protection Recommendations, 
    and assembles the final report markdown string.
    """
    current_year = pd.to_datetime('today').year
    latest_counts = trend_df[trend_df['Year'] == current_year]
    risk_status = "CRITICAL" if risk_score > 70 else ("HIGH" if risk_score > 40 else "MODERATE")

    # 1. Generate Conservation Analysis and Narrative (LLM)
    species_list_for_prompt = ', '.join([f"{row['Species']} ({row['Population']} individuals)" for _, row in latest_counts.iterrows()])
    
    # Use fallback species list if no current counts were successfully extracted
    if not species_list_for_prompt:
        species_list_for_prompt = ', '.join(trend_df['Species'].unique().tolist())
    
    analysis_prompt_base = f"""
    The following data was gathered during a wildlife conservation survey:
    Location: {extraction_data['location']}
    Species Observed: {species_list_for_prompt}
    Habitat Condition: {extraction_data['habitat_condition']}
    Calculated Risk Score (0-100): {risk_score:.1f}. Overall Status: {risk_status}.

    Write a detailed conservation analysis narrative (2-3 paragraphs) that synthesizes these findings, paying special attention to the habitat condition and the calculated risk score.
    """
    analysis_prompt = f"### Conservation Analysis Report\n\n{analysis_prompt_base}\n\nAnalysis:"
    analysis_text = generate_llm_text(analysis_prompt, max_length=150)
    analysis_text = analysis_text.split("### Conservation Analysis Report")[0].split("Analysis:")[0].strip()
    analysis_text = "\n".join(textwrap.wrap(analysis_text, width=80))

    # 2. Generate Recommendations (LLM)
    species_for_rec = latest_counts['Species'].tolist()
    if not species_for_rec:
        species_for_rec = trend_df['Species'].unique().tolist()
        
    recommendation_prompt_base = f"""
    Based on the status ({risk_status} risk score of {risk_score:.1f}) and the presence of snare wires/poaching threat, generate a list of 4-5 specific, prioritized protection recommendations for {extraction_data['location']} focusing on the observed species {', '.join(species_for_rec)}. Start each recommendation with an asterisk (*).
    """
    recommendation_prompt = f"### Protection Recommendations\n\n{recommendation_prompt_base}\n\nRecommendations:"
    recommendation_text = generate_llm_text(recommendation_prompt, max_length=100)
    rec_list = [line.strip() for line in recommendation_text.split('\n') if line.strip().startswith('*')]
    recommendation_text = "\n".join(rec_list)
    if not recommendation_text:
        recommendation_text = f"* Rule-Based Fallback: Deploy rapid intervention teams immediately due to {risk_status} risk score.\n* Rule-Based Fallback: Conduct bi-weekly surveys focusing on {', '.join(species_for_rec)}."
    
    # 3. Assemble the Final Report Template
    species_summary = []
    next_year = current_year + 1
    for _, row in latest_counts.iterrows():
        projection = trend_df[(trend_df['Species'] == row['Species']) & (trend_df['Year'] == next_year)]
        projected_pop = projection['Population'].iloc[0] if not projection.empty else 'N/A'
        species_summary.append(f"  - **{row['Species']}**: Observed: {row['Population']}, Projected {next_year}: {projected_pop}")

    # If no current sightings were extracted, use the latest available historical values.
    if not species_summary:
        latest_year = trend_df['Year'].max()
        latest_historical = trend_df[trend_df['Year'] == latest_year]
        for _, row in latest_historical.iterrows():
            projection = trend_df[(trend_df['Species'] == row['Species']) & (trend_df['Year'] == next_year)]
            projected_pop = projection['Population'].iloc[0] if not projection.empty else 'N/A'
            species_summary.append(f"  - **{row['Species']}**: Observed: {row['Population']}, Projected {next_year}: {projected_pop} (Note: Observed count is from latest historical record due to extraction failure.)")


    # Build the report string
    report_string = textwrap.dedent(f"""
    # **CONSERVATION MONITORING REPORT: LLM GENERATED**

    **Date Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    **Report Date:** {extraction_data['date']}
    **Location:** {extraction_data['location']}
    **Observer:** {extraction_data['observer']}
    ---

    ## 1. DATA SCIENCE & POPULATION METRICS

    ### Habitat Risk Assessment
    * **Calculated Habitat Risk Score (0-100):** **{risk_score:.1f}**
    * **Overall Risk Status:** **{risk_status}**

    ### Species Population Summary
    The data science model integrated current field counts with historical data to project trends.
    {'\n'.join(species_summary)}

    ## 2. LLM GENERATED CONSERVATION ANALYSIS (NLP & GENAI)

    {analysis_text}

    ## 3. LLM GENERATED PROTECTION RECOMMENDATIONS

    {recommendation_text}
    """)
    return report_string

# --- 6. EXPORT FUNCTION (New Feature - FIXED) ---

def save_report_to_pdf(report_text: str, plot_path: str, output_filename: str):
    """
    Saves the report text and the Matplotlib plot to a PDF file.
    FIXED: Avoiding KeyError by only defining custom unique styles.
    """
    if not REPORTLAB_AVAILABLE:
        print("[ERROR] reportlab is not installed. Cannot export PDF.")
        return

    output_path = os.path.join("Output_Reports", output_filename)
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    Story = []

    # FIX: Only define unique custom styles. Modify existing ones by reference.
    styles.add(ParagraphStyle(name='ReportTitle', alignment=TA_CENTER, fontSize=18, fontName='Helvetica-Bold'))
    
    # Modify built-in styles for headers
    styles['Heading2'].fontName = 'Helvetica-Bold'
    styles['Heading3'].fontName = 'Helvetica-Bold'
    styles['Normal'].fontName = 'Helvetica'
    styles['Normal'].fontSize = 10
    
    # Parse the report text line by line
    for line in report_text.split('\n'):
        line = line.strip()
        if not line:
            Story.append(Spacer(1, 0.1 * inch))
        elif line.startswith('# **'):
            Story.append(Paragraph(line.replace('# **', '').replace('**', ''), styles['ReportTitle']))
            Story.append(Spacer(1, 0.2 * inch))
        elif line.startswith('## '):
            Story.append(Paragraph(line.replace('## ', ''), styles['Heading2']))
        elif line.startswith('### '):
            Story.append(Paragraph(line.replace('### ', ''), styles['Heading3']))
        else:
            # Simple text parsing for bold (**text**) and lists (* item)
            # Use <b> tag for bold in ReportLab
            line = line.replace('**', '<b>', 1).replace('**', '</b>', 1) 
            if line.startswith('* '):
                 Story.append(Paragraph('- ' + line[2:], styles['Normal']))
            elif line.startswith('- '):
                 Story.append(Paragraph('- ' + line[2:], styles['Normal']))
            else:
                 Story.append(Paragraph(line, styles['Normal']))
    
    Story.append(Spacer(1, 0.4 * inch))
    
    # Add the visualization
    Story.append(Paragraph("<b>4. Population Trend Visualization</b>", styles['Heading2']))
    Story.append(Spacer(1, 0.2 * inch))
    img = Image(plot_path, width=5.5 * inch, height=3.5 * inch)
    Story.append(img)

    doc.build(Story)
    print(f"\n[SUCCESS] Report saved to PDF: {output_path}")

# --- 7. MAIN EXECUTION LOOP ---

def run_analysis():
    """Handles the core data processing steps."""
    
    # Ensure the user provides a *text* file for field notes
    while True:
        field_notes_path = input("Enter path to Field Notes file (e.g., sample_field_notes.txt): ").strip()
        if field_notes_path.lower().endswith('.txt') or field_notes_path.lower() == 'sample_field_notes.txt':
            break
        print("[ERROR] Please ensure the Field Notes file is the correct .txt file containing the narrative observations.")

    history_data_path = input("Enter path to Historical Population CSV (e.g., historical_population_data.csv): ").strip()

    try:
        with open(field_notes_path, 'r') as f:
            field_notes_text = f.read()
        historical_df = pd.read_csv(history_data_path)
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e.filename}. Please check paths and try again.")
        return
    except pd.errors.EmptyDataError:
        print(f"\n[ERROR] Historical Data CSV is empty.")
        return

    # 2. NLP and Data Extraction
    print("\n" + "="*60)
    print("PHASE 1: NLP Extraction (SpaCy/Regex)")
    inferred_species = infer_species_from_historical_df(historical_df)
    print(f"Inferred species from historical dataset: {', '.join(inferred_species) if inferred_species else 'None'}")

    extraction_data = extract_key_info(field_notes_text, expected_species=inferred_species)
    print(pd.DataFrame([extraction_data]))
    
    if not extraction_data['species_sightings']:
        print("[WARNING] Could not extract any species sightings. Analysis will proceed using historical data as the current count.")
        # FALLBACK: If extraction fails, inject placeholder data based on the last historical entry for calculation purposes
        # The report generation logic will handle the missing current sightings gracefully.
    else:
        print(f"[SUCCESS] Extracted {len(extraction_data['species_sightings'])} species sightings.")

    # 3. Data Science Modeling
    print("\n" + "="*60)
    print("PHASE 2: Data Science Modeling (Population & Risk)")
    risk_score = habitat_risk_analysis(extraction_data['habitat_condition'], extraction_data['threat_level'])
    trend_df = population_trend_model(historical_df, extraction_data['species_sightings'])
    print(f"Calculated Habitat Risk Score: {risk_score:.1f}")
    
    print("\n--- Population Trend Data (with Next Year Projection) ---")
    print(trend_df.pivot_table(index='Year', columns='Species', values='Population', aggfunc='first').fillna(''))

    # 4. LLM Report Generation
    print("\n" + "="*60)
    print(f"PHASE 3: LLM Report Generation (Using {LLM_MODEL_NAME if LLM_AVAILABLE else 'Rule-Based Fallback'})")
    final_report_text = generate_report_and_recommendations(extraction_data, trend_df, risk_score)
    
    print("\n" + "*" * 20 + " GENERATED CONSERVATION REPORT " + "*" * 20)
    print(final_report_text)
    print("*" * 67 + "\n")

    # 5. Visualization and Export
    plot_temp_path = os.path.join("Output_Reports", "temp_plot.png")
    print("\n" + "="*60)
    print("PHASE 4: Trend Visualization (Matplotlib)")
    
    # Save the plot temporarily before asking for export
    plot_generated = visualize_population_trends(trend_df, save_path=plot_temp_path)
    if plot_generated:
        print("Visualization generated and saved temporarily.")
    
    # Ask for export
    export_choice = input("\nDo you want to export the report as PDF? (yes/no): ").strip().lower()
    
    if export_choice == 'yes' or export_choice == 'y':
        if plot_generated and os.path.exists(plot_temp_path):
            file_name = f"Conservation_Report_{extraction_data['date']}_{random.randint(1000,9999)}.pdf"
            save_report_to_pdf(final_report_text, plot_temp_path, file_name)
        else:
            print("[ERROR] Plot was not generated. Skipping PDF export.")
    else:
        print("Report not exported.")
        
    # Clean up the temporary plot file
    if os.path.exists(plot_temp_path):
        os.remove(plot_temp_path)
        
    # Show the plot last (non-blocking if saved)
    if plot_generated:
        visualize_population_trends(trend_df, save_path=None) # Display for user view

def main():
    """Main function to run the entire Conservation Report Generator workflow with loop."""
    
    create_example_data_files()
    
    while True:
        print("\n--- Conservation Report Generator (NLP & Data Science) ---")
        run_analysis()
        
        # Ask user if they want to run again
        run_again = input("\nDo you want to run a new analysis with different data? (yes/no): ").strip().lower()
        if run_again not in ('yes', 'y'):
            print("\nExiting Conservation Report Generator. Goodbye! 👋")
            break

if __name__ == "__main__":
    main()
