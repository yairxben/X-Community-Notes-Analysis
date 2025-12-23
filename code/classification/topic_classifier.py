#!/usr/bin/env python3
"""
Custom Topic Classification for Comprehensive Content Analysis
Implements 15-topic classification covering conflicts, geopolitics, and general categories
Topics: Ukraine, Gaza, Syria, Iran, China-Taiwan, China Influence, Other Conflicts, Scams,
        Health/Medical, Climate/Environment, Politics, Technology, Economics, Entertainment, 
        Immigration
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
from datetime import datetime
import warnings
import sys
import psutil
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re
warnings.filterwarnings('ignore')

# Language detection - try multiple libraries
LANGDETECT_AVAILABLE = False
TEXTBLOB_AVAILABLE = False

try:
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    try:
        from textblob import TextBlob
        TEXTBLOB_AVAILABLE = True
    except ImportError:
        pass

def detect_language(text):
    """Detect if text is in English"""
    if not text or len(str(text).strip()) < 10:
        return False  # Too short to reliably detect
    
    try:
        if LANGDETECT_AVAILABLE:
            # Using langdetect (more accurate)
            lang = detect(str(text))
            return lang == 'en'
        elif TEXTBLOB_AVAILABLE:
            # Using TextBlob as fallback
            blob = TextBlob(str(text))
            try:
                lang = blob.detect_language()
                return lang == 'en'
            except:
                return False
        else:
            # Simple heuristic fallback - check for common English words
            english_indicators = ['the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but']
            text_lower = str(text).lower()
            english_count = sum(1 for word in english_indicators if f' {word} ' in f' {text_lower} ')
            return english_count >= 2  # At least 2 common English words
    except (LangDetectException, Exception):
        # Fallback to heuristic if language detection fails
        english_indicators = ['the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but']
        text_lower = str(text).lower()
        english_count = sum(1 for word in english_indicators if f' {word} ' in f' {text_lower} ')
        return english_count >= 2

class CustomTopicClassifier:
    """
    Custom Topic Classification for Comprehensive Content Analysis
    Covers 15 categories: conflicts, geopolitics, health, climate, politics, 
    technology, economics, entertainment, immigration, and scams
    """
    
    def __init__(self, data_path="data"):
        self.data_path = Path(data_path)
        self.trained_pipeline = None
        self.topic_labels = None
        
        # Create output directory
        self.output_dir = Path("custom_topic_results")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        
        # Print language detection status
        if LANGDETECT_AVAILABLE:
            print("✅ Using langdetect for language detection")
        elif TEXTBLOB_AVAILABLE:
            print("✅ Using TextBlob for language detection")
        else:
            print("⚠️  No language detection library available. Install with: pip install langdetect")
        
        # Initialize topic categories
        self.setup_custom_topics()
    
    def setup_custom_topics(self):
        """Initialize our custom topic categories and seed terms"""
        # Get custom seed terms
        self.seed_terms = self.create_custom_seed_terms()
        
        # Create topic label mapping
        self.topic_labels = {
            0: "Unassigned",
            1: "UkraineConflict", 
            2: "GazaConflict",
            3: "SyriaWar",
            4: "Iran",
            5: "ChinaTaiwan", 
            6: "ChinaInfluence",
            7: "OtherConflicts",
            8: "Scams",
            9: "HealthMedical",
            10: "ClimateEnvironment",
            11: "Politics",
            12: "Technology",
            13: "Economics",
            14: "Entertainment",
            15: "Immigration"
        }
        
        print("✅ Custom topic model initialized successfully!")
        print(f"📋 Available topics ({len(self.seed_terms)}):")
        for topic, terms in self.seed_terms.items():
            print(f"  - {topic}: {len(terms)} seed terms")
    
    def create_custom_seed_terms(self):
        """Create completely custom seed terms with new topics"""
        # Custom conflict, geopolitical, and general topics (15 categories total)
        custom_seed_terms = {
            # Ukraine Conflict
            "UkraineConflict": {
                "ukraine", "ukrainian", "russia", "russian", "kiev", "kyiv", 
                "moscow", "zelensky", "putin", "crimea", "donbas", "donetsk", 
                "luhansk", "mariupol", "kharkiv", "lviv", "odesa", "nato", 
                "invasion", "war", "conflict", "sanctions", "wagner", "bakhmut",
                "kherson", "zaporizhzhia", "bucha", "irpin", "azov", "dpr", "lpr"
            },
            
            # Gaza Conflict
            "GazaConflict": {
                "israel", "israeli", "palestine", "palestinian", "gaza", 
                "jerusalem", "hamas", "idf", "west bank", "tel aviv", 
                "netanyahu", "conflict", "ceasefire", "intifada", "settlement", 
                "occupation", "hezbollah", "fatah", "abbas", "ramallah",
                "bethlehem", "rafah", "khan younis", "iron dome", "rockets"
            },
            
            # Syria War
            "SyriaWar": {
                "syria", "syrian", "assad", "damascus", "aleppo", "idlib",
                "kurds", "kurdish", "ypg", "sdf", "turkey", "turkish",
                "erdogan", "civil war", "rebels", "opposition", "isis",
                "refugees", "chemical weapons", "barrel bombs", "ghouta",
                "daraa", "homs", "latakia", "tartus", "deir ez-zor"
            },
            
            # Iran
            "Iran": {
                "iran", "iranian", "tehran", "ayatollah", "khamenei", 
                "rouhani", "nuclear", "sanctions", "irgc", "revolutionary guard",
                "persian", "shiite", "shia", "sunni", "proxy", "hezbollah",
                "houthis", "mahdi army", "quds force", "uranium", "centrifuge",
                "jcpoa", "nuclear deal", "protest", "hijab", "mahsa amini"
            },
            
            # China-Taiwan
            "ChinaTaiwan": {
                "taiwan", "taiwanese", "china", "chinese", "xi jinping",
                "ccp", "communist party", "beijing", "taipei", "strait",
                "reunification", "independence", "one china", "tsai ing-wen",
                "kuomintang", "dpp", "pla", "fighter jets", "south china sea",
                "hong kong", "macau", "uyghur", "xinjiang", "tibet"
            },
            
            # China Influence
            "ChinaInfluence": {
                "belt and road", "bri", "debt trap", "huawei", "tiktok",
                "wechat", "confucius institute", "spy balloon", "trade war",
                "tariffs", "rare earth", "semiconductor", "technology transfer",
                "intellectual property", "uighur", "genocide", "surveillance",
                "social credit", "great firewall", "censorship", "wolf warrior"
            },
            
            # Other Conflicts
            "OtherConflicts": {
                "afghanistan", "taliban", "kabul", "myanmar", "burma",
                "rohingya", "coup", "junta", "yemen", "saudi", "houthis",
                "ethiopia", "tigray", "amhara", "somalia", "al-shabaab",
                "mali", "burkina faso", "sudan", "darfur", "south sudan",
                "libya", "tripoli", "benghazi", "nagorno-karabakh", "armenia",
                "azerbaijan", "kashmir", "pakistan", "border clash"
            },
            
            # Scams
            "Scams": {
                "scam", "undisclosed ad", "terms of service", "help.x.com",
                "x.com/tos", "engagement farm", "spam", "gambling", "apostas",
                "apuestas", "dropship", "drop ship", "promotion", "fake",
                "fraud", "phishing", "cryptocurrency scam", "bitcoin scam",
                "nft scam", "ponzi", "pyramid scheme", "mlm", "affiliate",
                "clickbait", "bot", "fake followers", "manipulation",
                "misinformation", "disinformation", "deepfake"
            },
            
            # Health & Medical
            "HealthMedical": {
                "vaccine", "covid", "coronavirus", "pandemic", "virus",
                "medicine", "health", "doctor", "hospital", "treatment",
                "medical", "pharma", "drug", "illness", "disease",
                "fda", "cdc", "who", "clinical trial", "side effect",
                "immunity", "antibody", "mask", "quarantine", "pfizer",
                "moderna", "astrazeneca", "johnson", "booster", "variant", "clinical", "coronavirus", "corona"
            },
            
            # Climate & Environment  
            "ClimateEnvironment": {
                "climate change", "global warming", "carbon", "emissions",
                "renewable", "solar", "wind", "fossil fuel", "oil",
                "gas", "coal", "pollution", "environment", "green",
                "sustainable", "recycling", "deforestation", "biodiversity",
                "extinction", "conservation", "greenhouse gas", "ipcc",
                "paris agreement", "carbon footprint", "electric vehicle"
            },
            
            # Politics
            "Politics": {
                "election", "vote", "voting", "democrat", "republican",
                "biden", "trump", "congress", "senate", "house", "jd vance", "newsom",
                "politician", "government", "policy", "law", "bill",
                "constitution", "supreme court", "president", "campaign",
                "political", "partisan", "ballot", "primary", "midterm",
                "governor", "mayor", "legislature", "impeachment",
                "pam bondi", "keir starmer", "cabinet", "minister", "nomination",
                "appointment", "confirmation", "secretary", "ambassador",
                "white house", "downing street", "parliament", "mps",
                "conservative", "labour", "liberal", "progressive",
                "administration", "executive", "judicial", "legislative",
                "foreign secretary", "home secretary", "chancellor",
                "prime minister", "vice president", "speaker",
                "political party", "caucus", "committee", "hearing",
                "social media regulation", "tech regulation", "antitrust",
                "section 230", "content moderation", "free speech", "censorship",
                "tiktok ban", "facebook regulation", "twitter policy"
            },
            
            # Technology  
            "Technology": {
                "machine learning", "algorithm",
                "tech", "software", "hardware",
                "programming", "coding", "developer", "app development",
                "google", "apple", "microsoft", "amazon", "tesla",
                "metaverse",
                "openai", "silicon valley", "startup",
                "data science", "cybersecurity", "cloud computing", "5g",
                "virtual reality", "augmented reality", "robotics", "automation",
                "semiconductor", "chip", "processor", "innovation", "patent",
                "tech company", "software engineering", "tech industry"
            },
            
            # Economics & Finance
            "Economics": {
                "economy","cryptocurrency", "bitcoin", "blockchain", "economic", "finance", "financial", "money",
                "dollar", "inflation", "recession", "market", "stock",
                "investment", "bank", "banking", "federal reserve", "fed",
                "interest rate", "gdp", "unemployment", "job", "wage",
                "tax", "budget", "debt", "credit", "loan", "wall street",
                "nasdaq", "dow jones", "s&p 500", "crypto", "trading",
                "earnings", "revenue", "profit", "ipo", "valuation",
                "meta earnings", "apple earnings", "google revenue",
                "tech stocks", "market cap", "dividend", "quarterly results"
            },
            
            # Entertainment
            "Entertainment": {
                "movie", "film", "actor", "actress", "celebrity", "hollywood",
                "netflix", "disney", "music", "singer", "artist", "album",
                "concert", "tv show", "series", "streaming", "oscar",
                "grammy", "entertainment", "sport", "game", "gaming",
                "taylor swift", "beyonce", "marvel", "nfl", "nba", "fifa"
            },
            
            # Immigration
            "Immigration": {
                "immigration", "immigrant", "migrant", "refugee", "asylum",
                "border", "deportation", "visa", "green card", "citizenship",
                "naturalization", "illegal", "legal", "undocumented",
                "sanctuary", "daca", "dreamer", "ice", "cbp", "detention",
                "caravan", "mexico", "southern border", "ellis island"
            }
        }
        
        print(f"📚 Created {len(custom_seed_terms)} custom topic categories:")
        for topic, terms in custom_seed_terms.items():
            print(f"  - {topic}: {len(terms)} seed terms")
            
        return custom_seed_terms
    
    def assign_seed_labels(self, notes_df, progress_tracker=None):
        """Assign initial topic labels based on seed term matching"""
        print("\n🔍 Assigning seed labels based on keyword matching...")
        
        # Initialize arrays
        seed_labels = np.zeros(len(notes_df), dtype=int)  # 0 = Unassigned
        conflicted_notes = np.zeros(len(notes_df), dtype=bool)
        
        # Topic to number mapping
        topic_to_num = {topic: i+1 for i, topic in enumerate(self.seed_terms.keys())}
        
        total_assigned = 0
        total_conflicts = 0
        
        for idx, row in notes_df.iterrows():
            if progress_tracker and idx % 1000 == 0:
                progress_tracker.update(1000)
                
            text = str(row['summary']).lower()
            matches = []
            
            # Check each topic for matches
            for topic, terms in self.seed_terms.items():
                for term in terms:
                    if term.lower() in text:
                        matches.append(topic)
                        break  # Found match for this topic, move to next
            
            # Assign label based on matches with priority rules
            if len(matches) == 1:
                # Single match - assign topic
                seed_labels[idx] = topic_to_num[matches[0]]
                total_assigned += 1
            elif len(matches) > 1:
                # Multiple matches - apply priority rules
                # Rule 1: Politics takes priority over Technology
                if "Politics" in matches and "Technology" in matches:
                    seed_labels[idx] = topic_to_num["Politics"]
                    total_assigned += 1
                # Rule 2: Any conflict topic takes priority over general topics
                elif any(topic in ["UkraineConflict", "GazaConflict", "SyriaWar", "Iran", 
                              "ChinaTaiwan", "ChinaInfluence", "OtherConflicts"] for topic in matches):
                    # Find first conflict topic
                    conflict_topics = ["UkraineConflict", "GazaConflict", "SyriaWar", "Iran", 
                                     "ChinaTaiwan", "ChinaInfluence", "OtherConflicts"]
                    for conflict_topic in conflict_topics:
                        if conflict_topic in matches:
                            seed_labels[idx] = topic_to_num[conflict_topic]
                            break
                    total_assigned += 1
                else:
                    # Default: use first match, mark as conflicted
                    seed_labels[idx] = topic_to_num[matches[0]]
                    conflicted_notes[idx] = True
                    total_assigned += 1
                    total_conflicts += 1
            # else: remains 0 (Unassigned)
        
        print(f"✅ Seed labeling completed:")
        print(f"   📊 Total assigned: {total_assigned:,} ({(total_assigned/len(notes_df))*100:.1f}%)")
        print(f"   ⚠️  Conflicted notes: {total_conflicts:,} ({(total_conflicts/len(notes_df))*100:.1f}%)")
        print(f"   🚫 Unassigned: {len(notes_df) - total_assigned:,} ({((len(notes_df) - total_assigned)/len(notes_df))*100:.1f}%)")
        
        return seed_labels, conflicted_notes
    
    def train_topic_classifier(self, notes_df, seed_labels, max_features=50000):
        """Train TF-IDF + Logistic Regression classifier"""
        print("\n🎓 Training topic classifier...")
        
        # Filter to only labeled data for training
        labeled_mask = seed_labels > 0
        training_texts = notes_df[labeled_mask]['summary'].values
        training_labels = seed_labels[labeled_mask]
        
        print(f"📊 Training data: {len(training_texts):,} labeled notes")
        print(f"📈 Label distribution:")
        unique, counts = np.unique(training_labels, return_counts=True)
        for label, count in zip(unique, counts):
            topic_name = self.topic_labels[label]
            print(f"   {topic_name}: {count:,} notes")
        
        # Create TF-IDF + Logistic Regression pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),  # unigrams and bigrams
                stop_words='english',
                lowercase=True,
                min_df=2,  # minimum document frequency
                max_df=0.95  # maximum document frequency
            )),
            ('classifier', LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                multi_class='ovr',
                random_state=42,
                class_weight='balanced'  # Handle class imbalance
            ))
        ])
        
        # Split training data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            training_texts, training_labels, 
            test_size=0.2, random_state=42, 
            stratify=training_labels
        )
        
        # Train the model
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        
        print(f"✅ Model trained in {training_time:.1f} seconds")
        print(f"📊 Test accuracy: {accuracy:.3f}")
        
        # Show classification report
        print(f"\n📋 Classification Report:")
        target_names = [self.topic_labels[i] for i in sorted(unique)]
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        
        return pipeline, {
            'training_time': training_time,
            'test_accuracy': accuracy,
            'training_size': len(training_texts),
            'feature_count': len(pipeline.named_steps['tfidf'].get_feature_names_out())
        }
    
    def classify_all_notes(self, notes_df, trained_pipeline, progress_tracker=None):
        """Apply trained classifier to all notes"""
        print(f"\n🔍 Classifying all {len(notes_df):,} notes...")
        
        # Get predictions for all notes
        all_texts = notes_df['summary'].fillna('').values
        
        start_time = time.time()
        predictions = trained_pipeline.predict(all_texts)
        prediction_probs = trained_pipeline.predict_proba(all_texts)
        classification_time = time.time() - start_time
        
        # Create results dataframe
        results_df = notes_df.copy()
        results_df['topicLabel'] = predictions
        results_df['topicName'] = [self.topic_labels[label] for label in predictions]
        
        # Add confidence scores (max probability)
        results_df['confidence'] = np.max(prediction_probs, axis=1)
        
        print(f"✅ Classification completed in {classification_time:.1f} seconds")
        
        return results_df, {
            'classification_time': classification_time,
            'total_classified': len(results_df)
        }
    
    def analyze_classification_results(self, classified_notes):
        """Analyze and display classification results"""
        print("\n" + "="*70)
        print("CLASSIFICATION RESULTS ANALYSIS")
        print("="*70)
        
        # Topic distribution
        topic_dist = classified_notes['topicName'].value_counts()
        total_notes = len(classified_notes)
        
        print(f"\n📊 Topic Distribution ({total_notes:,} total notes):")
        print("-" * 50)
        for topic, count in topic_dist.items():
            percentage = (count / total_notes) * 100
            print(f"  {topic:<20}: {count:>8,} ({percentage:>5.1f}%)")
        
        # Sample notes for each topic
        print(f"\n📝 Sample Notes by Topic:")
        print("-" * 50)
        for topic in topic_dist.index[:15]:  # Show all 15 topics
            topic_notes = classified_notes[classified_notes['topicName'] == topic]
            if len(topic_notes) > 0:
                sample_note = topic_notes['summary'].iloc[0]
                preview = sample_note[:80] + "..." if len(str(sample_note)) > 80 else str(sample_note)
                print(f"  {topic}:")
                print(f"    '{preview}'")
        
        # Confidence analysis
        confidence_stats = classified_notes['confidence'].describe()
        print(f"\n📈 Confidence Statistics:")
        print(f"   Mean confidence: {confidence_stats['mean']:.3f}")
        print(f"   Median confidence: {confidence_stats['50%']:.3f}")
        print(f"   Low confidence (<0.5): {len(classified_notes[classified_notes['confidence'] < 0.5]):,}")
        
        return {
            'topic_distribution': topic_dist.to_dict(),
            'total_notes': total_notes,
            'topics_assigned': len(topic_dist) - (1 if 'Unassigned' in topic_dist else 0),
            'confidence_stats': confidence_stats.to_dict()
        }
    
    def save_all_outputs(self, classified_notes, trained_pipeline, metadata, analysis_results):
        """Save all outputs to files"""
        print("\n" + "="*70)
        print("SAVING OUTPUTS")
        print("="*70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save classified notes
        notes_file = self.output_dir / f"classified_notes_{timestamp}.csv"
        print(f"💾 Saving classified notes to: {notes_file}")
        classified_notes.to_csv(notes_file, index=False)
        
        # 2. Save trained model
        model_file = self.output_dir / f"trained_topic_model_{timestamp}.pkl"
        print(f"💾 Saving trained model to: {model_file}")
        with open(model_file, 'wb') as f:
            pickle.dump(trained_pipeline, f)
        
        # 3. Save seed terms
        seed_file = self.output_dir / f"seed_terms_{timestamp}.json"
        print(f"💾 Saving seed terms to: {seed_file}")
        # Convert sets to lists for JSON serialization
        seed_terms_serializable = {k: list(v) for k, v in self.seed_terms.items()}
        with open(seed_file, 'w') as f:
            json.dump(seed_terms_serializable, f, indent=2)
        
        # 4. Save metadata and analysis
        summary_data = {
            'timestamp': timestamp,
            'processing_metadata': metadata,
            'analysis_results': analysis_results,
            'topic_labels': self.topic_labels,
            'file_paths': {
                'classified_notes': str(notes_file),
                'trained_model': str(model_file),
                'seed_terms': str(seed_file)
            }
        }
        
        summary_file = self.output_dir / f"classification_summary_{timestamp}.json"
        print(f"💾 Saving summary to: {summary_file}")
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"\n✅ All outputs saved successfully!")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        
        return summary_data
    
    def load_community_notes_data(self, english_only=True, force_refilter=False):
        """Load the full Community Notes dataset with optional English filtering"""
        print("\n" + "="*70)
        print("LOADING COMMUNITY NOTES DATA")
        print("="*70)
        
        # Check if English-filtered dataset already exists
        english_filtered_file = self.data_path / "notes" / "notes_english_only.tsv"
        
        if english_only and english_filtered_file.exists() and not force_refilter:
            print(f"📂 Found existing English-filtered dataset: {english_filtered_file}")
            print("🔄 Loading pre-filtered English notes...")
            notes_df = pd.read_csv(english_filtered_file, sep='\t', low_memory=False)
            print(f"✅ Loaded {len(notes_df):,} English notes from cache")
            print(f"📊 Dataset shape: {notes_df.shape}")
            print(f"💾 Memory usage: {notes_df.memory_usage().sum() / 1024**2:.1f} MB")
            print(f"✅ Final dataset: {len(notes_df):,} notes ready for classification")
            return notes_df
        
        # Load original notes data
        notes_file = self.data_path / "notes" / "notes-00000.tsv"
        print(f"📂 Loading notes from: {notes_file}")
        
        if not notes_file.exists():
            raise FileNotFoundError(f"Notes file not found: {notes_file}")
        
        # Load with progress tracking
        print("🔄 Reading notes TSV file...")
        notes_df = pd.read_csv(notes_file, sep='\t', low_memory=False)
        
        print(f"✅ Loaded {len(notes_df):,} notes")
        print(f"📊 Dataset shape: {notes_df.shape}")
        print(f"💾 Memory usage: {notes_df.memory_usage().sum() / 1024**2:.1f} MB")
        
        # Check required columns
        required_cols = ['noteId', 'summary']
        missing_cols = [col for col in required_cols if col not in notes_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter out notes without text
        initial_count = len(notes_df)
        notes_df = notes_df[notes_df['summary'].notna() & (notes_df['summary'].str.strip() != '')]
        notes_df = notes_df.reset_index(drop=True)  # Reset index after filtering
        after_text_filter = len(notes_df)
        
        if after_text_filter < initial_count:
            print(f"⚠️  Filtered out {initial_count - after_text_filter:,} notes without text")
        
        # Filter for English only if requested
        if english_only:
            print(f"\n🌍 Filtering for English-only notes...")
            print(f"📊 Processing {len(notes_df):,} notes for language detection...")
            
            # Sample a subset first to estimate filtering impact
            sample_size = min(5000, len(notes_df))
            sample_df = notes_df.sample(n=sample_size, random_state=42)
            
            print(f"🔍 Testing language detection on {sample_size:,} sample notes...")
            english_sample = []
            for idx, text in enumerate(sample_df['summary'].values):
                if idx % 1000 == 0 and idx > 0:
                    print(f"   Processed {idx:,}/{sample_size:,} sample notes...")
                english_sample.append(detect_language(text))
            
            english_rate = np.mean(english_sample)
            print(f"📈 Sample English rate: {english_rate:.1%}")
            estimated_english_count = int(len(notes_df) * english_rate)
            print(f"📊 Estimated English notes: {estimated_english_count:,}")
            
            # Apply to full dataset
            print(f"🔄 Applying English filter to full dataset...")
            english_mask = []
            total_notes = len(notes_df)
            
            for idx, text in enumerate(notes_df['summary'].values):
                if idx % 10000 == 0 and idx > 0:
                    print(f"   Processed {idx:,}/{total_notes:,} notes ({idx/total_notes:.1%})...")
                english_mask.append(detect_language(text))
            
            # Filter to English only
            before_lang_filter = len(notes_df)
            notes_df = notes_df[english_mask].reset_index(drop=True)
            after_lang_filter = len(notes_df)
            
            print(f"✅ Language filtering completed:")
            print(f"   📊 English notes: {after_lang_filter:,} ({after_lang_filter/before_lang_filter:.1%})")
            print(f"   🚫 Non-English filtered: {before_lang_filter - after_lang_filter:,}")
            
            # Save the English-filtered dataset for future use
            print(f"💾 Saving English-filtered dataset to: {english_filtered_file}")
            notes_df.to_csv(english_filtered_file, sep='\t', index=False)
            print(f"✅ English-filtered dataset saved for future use!")
        
        print(f"✅ Final dataset: {len(notes_df):,} notes ready for classification")
        
        return notes_df
    
    def run_complete_pipeline(self, max_notes=None, english_only=True, force_refilter=False):
        """Run the complete custom topic classification pipeline"""
        print("="*70)
        print("CUSTOM GEOPOLITICAL TOPIC CLASSIFICATION PIPELINE")
        print("="*70)
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if english_only:
            print("🌍 English-only filtering: ENABLED")
        else:
            print("🌍 English-only filtering: DISABLED")
        
        pipeline_start = time.time()
        
        try:
            # Load data with English filtering
            notes_df = self.load_community_notes_data(english_only=english_only, force_refilter=force_refilter)
            
            # Limit dataset size if specified
            if max_notes and len(notes_df) > max_notes:
                print(f"⚠️  Limiting to {max_notes:,} notes for processing")
                notes_df = notes_df.sample(n=max_notes, random_state=42).reset_index(drop=True).copy()
            
            # Step 1: Assign seed labels
            seed_labels, conflicted_notes = self.assign_seed_labels(notes_df)
            
            # Step 2: Train classifier
            trained_pipeline, training_metadata = self.train_topic_classifier(notes_df, seed_labels)
            
            # Step 3: Classify all notes
            classified_notes, classification_metadata = self.classify_all_notes(notes_df, trained_pipeline)
            
            # Step 4: Analyze results
            analysis_results = self.analyze_classification_results(classified_notes)
            
            # Step 5: Save everything
            all_metadata = {
                **training_metadata,
                **classification_metadata,
                'seed_labels_assigned': np.sum(seed_labels > 0),
                'conflicted_notes': np.sum(conflicted_notes),
                'total_notes_processed': len(notes_df),
                'english_only_filter': english_only
            }
            
            summary_data = self.save_all_outputs(
                classified_notes, trained_pipeline, all_metadata, analysis_results
            )
            
            total_time = time.time() - pipeline_start
            
            print(f"\n" + "="*70)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"⏱️  Total processing time: {total_time:.1f} seconds")
            print(f"📊 Notes processed: {len(classified_notes):,}")
            print(f"🎯 Topics assigned: {analysis_results['topics_assigned']}")
            print(f"💾 Files saved in: {self.output_dir.absolute()}")
            
            return summary_data
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise
        """Create completely custom seed terms with new topics"""
        # Custom conflict and geopolitical-focused topics
        custom_seed_terms = {
            # Ukraine Conflict
            "UkraineConflict": {
                "ukraine", "ukrainian", "russia", "russian", "kiev", "kyiv", 
                "moscow", "zelensky", "putin", "crimea", "donbas", "donetsk", 
                "luhansk", "mariupol", "kharkiv", "lviv", "odesa", "nato", 
                "invasion", "war", "conflict", "sanctions", "wagner", "bakhmut",
                "kherson", "zaporizhzhia", "bucha", "irpin", "azov", "dpr", "lpr"
            },
            
            # Gaza Conflict
            "GazaConflict": {
                "israel", "israeli", "palestine", "palestinian", "gaza", 
                "jerusalem", "hamas", "idf", "west bank", "tel aviv", 
                "netanyahu", "conflict", "ceasefire", "intifada", "settlement", 
                "occupation", "hezbollah", "fatah", "abbas", "ramallah",
                "bethlehem", "rafah", "khan younis", "iron dome", "rockets"
            },
            
            # Syria War
            "SyriaWar": {
                "syria", "syrian", "assad", "damascus", "aleppo", "idlib",
                "kurds", "kurdish", "ypg", "sdf", "turkey", "turkish",
                "erdogan", "civil war", "rebels", "opposition", "isis",
                "refugees", "chemical weapons", "barrel bombs", "ghouta",
                "daraa", "homs", "latakia", "tartus", "deir ez-zor"
            },
            
            # Iran
            "Iran": {
                "iran", "iranian", "tehran", "ayatollah", "khamenei", 
                "rouhani", "nuclear", "sanctions", "irgc", "revolutionary guard",
                "persian", "shiite", "shia", "sunni", "proxy", "hezbollah",
                "houthis", "mahdi army", "quds force", "uranium", "centrifuge",
                "jcpoa", "nuclear deal", "protest", "hijab", "mahsa amini"
            },
            
            # China-Taiwan
            "ChinaTaiwan": {
                "taiwan", "taiwanese", "china", "chinese", "xi jinping",
                "ccp", "communist party", "beijing", "taipei", "strait",
                "reunification", "independence", "one china", "tsai ing-wen",
                "kuomintang", "dpp", "pla", "fighter jets", "south china sea",
                "hong kong", "macau", "uyghur", "xinjiang", "tibet"
            },
            
            # China Influence
            "ChinaInfluence": {
                "belt and road", "bri", "debt trap", "huawei", "tiktok",
                "wechat", "confucius institute", "spy balloon", "trade war",
                "tariffs", "rare earth", "semiconductor", "technology transfer",
                "intellectual property", "uighur", "genocide", "surveillance",
                "social credit", "great firewall", "censorship", "wolf warrior"
            },
            
            # Other Conflicts
            "OtherConflicts": {
                "afghanistan", "taliban", "kabul", "myanmar", "burma",
                "rohingya", "coup", "junta", "yemen", "saudi", "houthis",
                "ethiopia", "tigray", "amhara", "somalia", "al-shabaab",
                "mali", "burkina faso", "sudan", "darfur", "south sudan",
                "libya", "tripoli", "benghazi", "nagorno-karabakh", "armenia",
                "azerbaijan", "kashmir", "pakistan", "border clash"
            },
            
            # Scams
            "Scams": {
                "scam", "undisclosed ad", "terms of service", "help.x.com",
                "x.com/tos", "engagement farm", "spam", "gambling", "apostas",
                "apuestas", "dropship", "drop ship", "promotion", "fake",
                "fraud", "phishing", "cryptocurrency scam", "bitcoin scam",
                "nft scam", "ponzi", "pyramid scheme", "mlm", "affiliate",
                "clickbait", "bot", "fake followers", "manipulation",
                "misinformation", "disinformation", "deepfake"
            }
        }
        
        print(f"📚 Created {len(custom_seed_terms)} custom topic categories:")
        for topic, terms in custom_seed_terms.items():
            print(f"  - {topic}: {len(terms)} seed terms")
            
        return custom_seed_terms
    
    def show_seed_term_examples(self):
        """Display examples of seed terms for each topic"""
        print("\n" + "="*70)
        print("ENHANCED SEED TERMS PREVIEW")
        print("="*70)
        
        custom_terms = self.create_custom_seed_terms()
        
        for topic, terms in custom_terms.items():
            print(f"\n🏷️  {topic} ({len(terms)} terms):")
            sample_terms = list(terms)[:8]  # Show first 8 terms
            print(f"   {', '.join(sample_terms)}")
            if len(terms) > 8:
                print(f"   ... and {len(terms) - 8} more")
    
    def load_data_and_preview_topics(self, max_preview=1000):
        """Load data and preview how many notes would match each topic"""
        print("\n" + "="*70)
        print("TOPIC MATCHING PREVIEW")
        print("="*70)
        
        # Load a sample of data
        notes_file = self.data_path / "notes" / "notes-00000.tsv"
        print(f"📂 Loading sample from: {notes_file}")
        
        # Read a smaller sample for preview
        notes_df = pd.read_csv(notes_file, sep='\t', nrows=max_preview, low_memory=False)
        notes_df = notes_df[notes_df['summary'].notna()].copy()
        
        print(f"📊 Analyzing {len(notes_df):,} sample notes")
        
        # Get custom seed terms
        custom_terms = self.create_custom_seed_terms()
        
        # Count matches for each topic
        topic_matches = {}
        
        for topic, terms in custom_terms.items():
            match_count = 0
            sample_matches = []
            
            for idx, row in notes_df.iterrows():
                text = str(row['summary']).lower()
                
                # Check if any seed term matches
                for term in terms:
                    if term.lower() in text:
                        match_count += 1
                        if len(sample_matches) < 3:  # Keep 3 samples
                            preview = text[:60] + "..." if len(text) > 60 else text
                            sample_matches.append(f"'{preview}'")
                        break  # Don't double count same note
            
            topic_matches[topic] = {
                'count': match_count,
                'percentage': (match_count / len(notes_df)) * 100,
                'samples': sample_matches
            }
        
        # Display results
        print(f"\n📊 Topic Matching Results (from {len(notes_df):,} sample notes):")
        print("-" * 70)
        
        # Sort by match count
        sorted_topics = sorted(topic_matches.items(), key=lambda x: x[1]['count'], reverse=True)
        
        for topic, data in sorted_topics:
            print(f"\n🏷️  {topic}:")
            print(f"   Matches: {data['count']:,} ({data['percentage']:.1f}%)")
            if data['samples']:
                print(f"   Examples:")
                for sample in data['samples']:
                    print(f"     {sample}")
        
        total_matched = sum(data['count'] for data in topic_matches.values())
        unmatched = len(notes_df) - total_matched
        
        print(f"\n📈 Summary:")
        print(f"   Total notes analyzed: {len(notes_df):,}")
        print(f"   Notes with topic matches: {total_matched:,}")
        print(f"   Notes unmatched: {unmatched:,} ({(unmatched/len(notes_df))*100:.1f}%)")
        
        return topic_matches
    
def main():
    """Main function to demonstrate comprehensive topic classification"""
    print("="*70)
    print("COMPREHENSIVE 15-TOPIC CLASSIFICATION")
    print("="*70)
    print("Covering: Conflicts, Geopolitics, Health, Climate, Politics,")
    print("Technology, Economics, Entertainment, Immigration, Scams")
    
    # Initialize custom classifier
    classifier = CustomTopicClassifier(data_path="data")
    
    # Show what custom topics look like
    classifier.show_seed_term_examples()
    
    # Preview topic matching on sample data
    topic_matches = classifier.load_data_and_preview_topics(max_preview=5000)
        
    print(f"\n💡 Ready to run full classification pipeline:")
    print(f"   1. Load Community Notes data")
    print(f"   2. Assign seed labels based on 15 topic categories")  
    print(f"   3. Train TF-IDF + Logistic Regression model")
    print(f"   4. Classify all notes into 15 comprehensive topics")
    
    # Ask user if they want to run test
    print(f"\n🧪 Test Options:")
    print(f"   - Run test with 10K notes: classifier.run_test_pipeline()")
    print(f"   - Run full pipeline: classifier.run_complete_pipeline()")

def run_test():
    """Run a test with 10K notes with English filtering"""
    print("="*70)
    print("TESTING 15-TOPIC CLASSIFICATION ON 10K ENGLISH NOTES")
    print("="*70)
    
    # Initialize classifier
    classifier = CustomTopicClassifier(data_path="data")
    
    # Run test pipeline with 10K notes and English filtering
    results = classifier.run_complete_pipeline(max_notes=10000, english_only=True)
    
    return results

def run_test_50k():
    """Run a test with 50K notes with English filtering"""
    print("="*70)
    print("TESTING 15-TOPIC CLASSIFICATION ON 50K ENGLISH NOTES")
    print("="*70)
    
    # Initialize classifier
    classifier = CustomTopicClassifier(data_path="data")
    
    # Run test pipeline with 50K notes and English filtering
    results = classifier.run_complete_pipeline(max_notes=50000, english_only=True)
    
    return results

if __name__ == "__main__":
    main()
