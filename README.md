# X-Community-Notes-Analysis
![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange) ![NetworkX](https://img.shields.io/badge/Library-NetworkX-green) ![Pandas](https://img.shields.io/badge/Data-Pandas-150458) ![Status](https://img.shields.io/badge/Status-Completed-success)

**A data science project analyzing 1.2M+ X (Twitter) Community Notes to detect polarization, bot activity, and consensus anomalies.**

## 🛠️ Technologies & Tools
* **Core:** Python 3.9, Pandas, NumPy
* **Machine Learning (NLP):** Scikit-Learn (TF-IDF, Logistic Regression), NLTK/TextBlob
* **Graph Theory:** NetworkX (Graph construction, Centrality metrics), Community Detection (Louvain Algorithm)
* **Visualization:** Matplotlib, Seaborn
* **DevOps:** Git, Jupyter Notebooks, Modular Project Structure

---

## 📖 Motivation
**Who is checking the fact-checkers?**
This project applies **Natural Language Processing (NLP)** and **Network Analysis** to scrutinize the "crowd" behind Community Notes. We investigate whether the consensus algorithms successfully bridge political divides or if they inadvertently create "echo chambers" where only one side validates the truth.

---

## 📊 Key Findings & Technical Insights

### 1. Topic Classification & The "Consensus Gap"
We engineered a **custom 15-class Text Classifier** (TF-IDF + Logistic Regression) to categorize notes into topics like *Gaza Conflict*, *Politics*, and *Health*.
* **Technical Insight:** While the model achieved **81.4% accuracy**, we found a massive disparity in "Helpfulness" rates. "Objective" topics like *Scams* have high consensus, whereas *Geopolitical Conflicts* show agreement rates as low as **13-21%**, highlighting a flaw in the consensus mechanism for polarized data.

![Topic Polarization](plots/topic_classification/topic_discriminating_features.png)
*Figure 1: Helpfulness rates by topic. Green bars represent the percentage of notes rated "Helpful."*

### 2. Validation via Temporal Event Mapping
To validate our unsupervised classification without ground-truth labels, we performed **Time-Series Analysis** correlating note volume with real-world timestamps.
* **Result:** The model correctly identified volume spikes corresponding to the **October 7th War** (Gaza Label) and **US Presidential Debates** (Politics Label) with near-perfect temporal alignment.

![Temporal Analysis](plots/topic_classification/temporal_analysis.png)
*Figure 2: Temporal analysis showing topic volume spikes aligning with major global events.*

### 3. Community Detection (Graph Theory)
We constructed a **Contributor-Note Graph** (Nodes=Users, Edges=Agreements) and applied the **Louvain Modularity Algorithm** to detect communities.
* **Result:** The algorithm partitioned the user base into distinct clusters (visualized below). We found that specific communities specialize heavily (e.g., a "Medical" cluster vs. a "Political" cluster), effectively operating as distinct echo chambers rather than a unified "crowd."

![Community Graph](plots/comuunity_detection/community_dist.jpg)
*Figure 3: Network Graph of 15 contributor communities detected by the Louvain Algorithm. Each color represents a distinct community cluster.*

### 4. The "Power User" Long-Tail Distribution
Statistical analysis of user activity reveals a heavy **Power Law distribution**.
* **Data:** A tiny fraction (<1%) of "hyper-active" users contribute the vast majority of ratings. This raises algorithmic concerns about the outsized influence of a few "super-raters" on the global consensus.

![User Activity Distribution](plots/comuunity_detection/user_activity_distribution.jpg)
*Figure 4: Distribution of user activity, showing that a small percentage of users generate the majority of ratings.*

---

## 💻 Project Structure & Code Quality
The project is structured as a production-ready data pipeline, separating core logic from analysis notebooks.

```text
.
├── code/
│   ├── classification/
│   │   ├── topic_classifier.py          # Main classification module
│   │   └── analyze_classification_results.py  # Results analysis and visualization
│   ├── clustering/
│   │   ├── communities_analysis.py     # Community detection and analysis
│   │   └── louvain_communities.ipynb    # Jupyter notebook for community analysis
│   ├── eda/
│   │   └── eda_analysis.py              # Exploratory data analysis
│   ├── demo_workflow.py                 # Demo script showcasing workflow
│   └── generate_report.py               # Report generator for presentations
├── run_analysis.py                      # Main entry point script
├── plots/
│   ├── topic_classification/          # Classification visualizations
│   ├── comuunity_detection/            # Community analysis visualizations
│   └── eda/                            # EDA visualizations
└── README.md                           # This file


```markdown
## 🚀 Getting Started

### Prerequisites
* Python 3.8 or higher
* Required Python packages (see `requirements.txt` or install instructions below)

### Installation
1. **Clone or download this repository**
2. **Install required dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn networkx langdetect

```

*Optional (for better language detection):*

```bash
pip install textblob

```

### Data Setup

Place your Community Notes data in the following structure:

```text
data/
  notes/
    notes-00000.tsv

```

The TSV file should contain columns including:

* `noteId`: Unique identifier for each note
* `summary`: Text content of the note
* `createdAtMillis`: Timestamp (optional, for temporal analysis)

---

## 🎯 Usage

### Quick Start (Recommended)

The easiest way to run the analysis is using the main entry point script:

```bash
# Run complete pipeline (classification + analysis + report)
python run_analysis.py all --max-notes 10000

# Or run individual components
python run_analysis.py classify --max-notes 10000
python run_analysis.py analyze
python run_analysis.py report

# Run demo workflow
python run_analysis.py demo --quick

```

### Demo Workflow

For a quick demonstration of the complete workflow:

```bash
# Quick demo with 10K notes
python code/demo_workflow.py --quick

# Full demo with all notes
python code/demo_workflow.py --full

# Analysis only (on existing results)
python code/demo_workflow.py --analysis-only

```

### Programmatic Usage

**Topic Classification**

```python
from code.classification.topic_classifier import CustomTopicClassifier

# Initialize classifier
classifier = CustomTopicClassifier(data_path="data", output_dir="results")

# Run complete pipeline
results = classifier.run_complete_pipeline(
    max_notes=10000,      # Limit to 10K notes for testing
    english_only=True,    # Filter to English-only notes
    force_refilter=False  # Use cached English-filtered data if available
)

```

**Classification Results Analysis**

```python
from code.classification.analyze_classification_results import ClassificationAnalyzer

# Initialize analyzer
analyzer = ClassificationAnalyzer(results_dir="custom_topic_results")

# Load latest results and run complete analysis
results = analyzer.run_complete_analysis()

```

**Generate Summary Report**

```python
from code.generate_report import ReportGenerator

# Generate presentation-ready report
generator = ReportGenerator(output_dir="reports")
report_file = generator.generate_report(results_dir="custom_topic_results")

```

**Community Analysis**

```python
from code.clustering.communities_analysis import SpecializedVisualizationsDevide

# Initialize visualizer
visualizer = SpecializedVisualizationsDevide()

# Run all visualizations
results = visualizer.run_all_visualizations()

```

---

## 📊 Output Files

### Classification Outputs

* `classified_notes_*.csv`: Full classification results with topic labels and confidence scores
* `trained_topic_model_*.pkl`: Saved trained model for reuse
* `seed_terms_*.json`: Seed terms used for each topic
* `classification_summary_*.json`: Complete metadata and statistics

### Analysis Outputs

* `classification_analytics/`: Classification analysis charts and visualizations
* `reports/`: Generated summary reports (text and markdown formats)
* `plots/topic_classification/`: Topic distribution, confidence analysis, temporal trends
* `plots/comuunity_detection/`: Community structure, topic leadership matrices

### Generated Reports

The report generator creates presentation-ready summaries:

* `analysis_report_*.txt`: Text format report
* `analysis_report_*.md`: Markdown format report

Reports include:

* Executive summary with key metrics
* Topic distribution analysis
* Confidence statistics
* Methodology overview
* Key insights and findings

```

```
