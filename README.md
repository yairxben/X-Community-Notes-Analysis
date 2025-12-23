# X-Community-Notes-Analysis
![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Status](https://img.shields.io/badge/Status-Active-success) ![License](https://img.shields.io/badge/License-MIT-green)

**Analysis of 1.2M+ X (Twitter) Community Notes using NLP and Network Analysis.**

This project investigates the mechanics of crowdsourced fact-checking, specifically focusing on the impact of trustworthy sources, contributor polarization, and the "Consensus Paradox."

## 📖 Project Overview

### Motivation
> "Who is actually fact-checking the fact-checkers?"

Beyond analyzing the notes themselves, this project scrutinizes the "crowd" behind them to see if contributors bridge divides or fracture into polarized echo chambers. We investigate whether user activity patterns signal healthy diversity or potential manipulation (e.g., bots) by hyperactive minority groups.

Ultimately, we aim to answer a critical question: **Does Community Notes serve as a genuine corrective to misinformation, or does it risk reproducing the very societal divisions it was designed to fix?**

### Key Objectives
* **Data Mining:** Processed a dataset of **1.2 million** notes and ratings.
* **Network Analysis:** Modeled contributor interactions to detect polarization and echo chambers.
* **NLP Analysis:** Used natural language processing to assess sentiment and topic clustering.
* **Bot Detection:** Analyzed temporal activity patterns to identify non-human behavior and hyperactive minority groups.

---

## 📊 Visualizations & Findings

### 1. The Consensus Paradox
We observed that notes requiring broad consensus often fail to gain traction in highly polarized topics. The graph below visualizes the separation between contributor groups.

![Network Graph Placeholder](path/to/your/network_graph_image.png)
*Figure 1: Network graph showing contributor clusters. The scarcity of edges between groups highlights the difficulty of achieving cross-partisan consensus.*

### 2. Activity Distribution
A small minority of "hyper-active" users contribute the vast majority of notes.

![Distribution Plot Placeholder](path/to/your/distribution_image.png)
*Figure 2: Distribution of User Contributions vs. Helpfulness Ratio.*

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Network Analysis:** NetworkX
* **NLP:** [Insert libraries here, e.g., NLTK, spaCy, Transformers]
* **Visualization:** Matplotlib, Seaborn

---

## 💻 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/X-Community-Notes-Analysis.git](https://github.com/yourusername/X-Community-Notes-Analysis.git)
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the analysis:**
    ```bash
    jupyter notebook analysis_main.ipynb
    ```

---

## 📝 Conclusion
Our analysis suggests that while Community Notes effectively flags obvious misinformation, it struggles in subjective, political contexts due to **structural polarization**. The mechanism designed to ensure fairness—requiring diverse agreement—may inadvertently stifle fact-checking on controversial topics.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!
