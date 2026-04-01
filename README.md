# 🌿 AI Conservation Report Generator

An AI-powered biodiversity analysis and conservation reporting system that helps wildlife researchers, conservationists, and park managers analyze species population data and generate comprehensive conservation reports.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

- **📊 Interactive Data Analysis**: Upload CSV files with historical population data
- **🤖 AI-Powered Insights**: Uses NLP and LLM to generate intelligent conservation recommendations
- **📈 Visual Analytics**: Interactive charts showing population trends, comparisons, and risk assessments
- **📄 Report Generation**: Export professional PDF and TXT reports
- **🎨 Modern UI**: Clean, intuitive interface with highlighted sections and real-time feedback

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/vijayasri-manda/Conservation-Report-Generator.git
cd Conservation-Report-Generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

### Running the Application

**Web Interface (Recommended):**
```bash
streamlit run app.py
```

**Command Line Interface:**
```bash
python main.py
```

The web app will open in your browser at `http://localhost:8501`

## 📖 Usage

### 1. Input Data
- **Location**: Enter the observation location (e.g., "KBR National Park, Hyderabad")
- **Species Observed**: List species (e.g., "tiger, elephant, deer")
- **Threats**: Describe environmental threats (e.g., "Deforestation, Poaching")
- **Upload CSV**: Historical population data with a "Year" column

### 2. CSV Format
Your CSV file must include:
- A `Year` column (required)
- One or more species population columns

**Example:**
```csv
Year,Tiger_Population,Deer_Population
2020,45,120
2021,48,125
2022,52,130
```

### 3. Analyze
Click "🔍 Analyze Biodiversity" to generate:
- Biodiversity analysis report
- Population trend visualizations
- Risk assessment metrics
- AI-generated conservation recommendations

### 4. Export
Download reports in PDF or TXT format

## 📊 Sample Data

The project includes sample datasets:
- `historical_population_data.csv` - Zebra, Wild Dogs, Giraffes
- `tiger_population_data.csv` - Tiger, Deer, Wild Boar
- `sample_field_notes.txt` - Example field observations

## 🛠️ Technology Stack

- **Frontend**: Streamlit, Custom CSS
- **Data Processing**: Pandas, NumPy
- **NLP**: spaCy
- **AI/ML**: Transformers (DistilGPT2)
- **Visualization**: Plotly, Matplotlib
- **PDF Generation**: ReportLab

## 📁 Project Structure

```
Conservation-Report-Generator/
├── app.py                          # Main Streamlit web interface
├── main.py                         # Core analysis logic & CLI
├── streamlit_app.py               # Alternative interface
├── requirements.txt               # Dependencies
├── historical_population_data.csv # Sample data
├── tiger_population_data.csv     # Sample data
├── sample_field_notes.txt        # Sample notes
└── Output_Reports/               # Generated reports
```

## 🎨 Screenshots

### Input Interface
Clean, modern interface with labeled input fields and CSV format help.

### Analysis Results
- Biodiversity Analysis Report card
- Interactive population trend charts
- Risk assessment gauges
- Detailed conservation recommendations

## 🔬 How It Works

1. **Data Collection**: User inputs location, species, threats, and uploads historical data
2. **NLP Extraction**: spaCy extracts key information from field notes
3. **Risk Analysis**: Calculates habitat risk scores (0-100)
4. **Population Modeling**: Projects future trends using historical data
5. **LLM Generation**: DistilGPT2 generates conservation recommendations
6. **Visualization**: Creates interactive charts and reports

## 💡 Use Cases

- Wildlife conservation monitoring
- National park biodiversity tracking
- Research and academic studies
- Stakeholder reporting
- Threat assessment and prioritization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 👥 Author

**Vijayasri Manda**
- GitHub: [@vijayasri-manda](https://github.com/vijayasri-manda)

## 🙏 Acknowledgments

- Built with Streamlit
- Powered by HuggingFace Transformers
- NLP by spaCy

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with 💚 for wildlife conservation**
