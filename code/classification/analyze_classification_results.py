#!/usr/bin/env python3
"""
Classification Results Analytics and Visualization
Analyzes the 15-topic classification results from the full dataset
Creates comprehensive graphs and statistics
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')

# Define distinct colors for better visualization
DISTINCT_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', 
    '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA',
    '#F1948A', '#85929E', '#D7BDE2', '#A9DFBF', '#F9E79F', '#AED6F1',
    '#FADBD8', '#D5DBDB', '#E8DAEF', '#D1F2EB', '#FCF3CF', '#DEEBF7'
]

class ClassificationAnalyzer:
    """Analyze and visualize classification results"""
    
    def __init__(self, results_dir="custom_topic_results"):
        self.results_dir = Path(results_dir)
        self.classified_notes = None
        self.summary_data = None
        
        # Create analytics output directory
        self.output_dir = Path("classification_analytics")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"📁 Analytics output directory: {self.output_dir.absolute()}")
    
    def load_latest_results(self):
        """Load the most recent classification results"""
        print("🔍 Loading latest classification results...")
        
        # Find the most recent results file
        csv_files = list(self.results_dir.glob("classified_notes_*.csv"))
        json_files = list(self.results_dir.glob("classification_summary_*.json"))
        
        if not csv_files:
            raise FileNotFoundError(f"No classification results found in {self.results_dir}")
        
        # Get the most recent files
        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime) if json_files else None
        
        print(f"📂 Loading classified notes: {latest_csv.name}")
        self.classified_notes = pd.read_csv(latest_csv)
        
        if latest_json:
            print(f"📂 Loading summary data: {latest_json.name}")
            with open(latest_json, 'r') as f:
                self.summary_data = json.load(f)
        
        print(f"✅ Loaded {len(self.classified_notes):,} classified notes")
        return self.classified_notes, self.summary_data
    
    def create_topic_distribution_chart(self):
        """Create a comprehensive topic distribution chart"""
        print("📊 Creating topic distribution chart...")
        
        # Calculate topic distribution
        topic_dist = self.classified_notes['topicName'].value_counts()
        total_notes = len(self.classified_notes)
        
        # 1. Horizontal bar chart
        plt.figure(figsize=(14, 10))
        colors = DISTINCT_COLORS[:len(topic_dist)]
        bars = plt.barh(range(len(topic_dist)), topic_dist.values, color=colors)
        plt.yticks(range(len(topic_dist)), topic_dist.index, fontsize=11)
        plt.xlabel('Number of Notes', fontsize=12)
        plt.title('Community Notes Topic Distribution\n(1.2M English Notes)', fontsize=16, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, topic_dist.values)):
            percentage = (value / total_notes) * 100
            plt.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2, 
                    f'{value:,}\n({percentage:.1f}%)', 
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the horizontal bar chart
        output_file = self.output_dir / "topic_distribution_bar.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_file}")
        plt.show()
        plt.close()
        
        # 2. Pie chart (separate figure)
        plt.figure(figsize=(12, 10))
        
        # Group smaller categories for better visualization
        threshold = 0.02  # 2%
        large_topics = topic_dist[topic_dist/total_notes >= threshold]
        small_topics = topic_dist[topic_dist/total_notes < threshold]
        
        if len(small_topics) > 0:
            pie_data = large_topics.copy()
            pie_data['Others'] = small_topics.sum()
        else:
            pie_data = large_topics
        
        # Use distinct colors for pie chart
        pie_colors = DISTINCT_COLORS[:len(pie_data)]
        wedges, texts, autotexts = plt.pie(pie_data.values, labels=pie_data.index, 
                                          autopct='%1.1f%%', startangle=90,
                                          colors=pie_colors, textprops={'fontsize': 10})
        plt.title('Topic Distribution (Percentage)', fontsize=16, fontweight='bold', pad=20)
        
        # Improve readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        plt.tight_layout()
        
        # Save the pie chart
        output_file = self.output_dir / "topic_distribution_pie.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_file}")
        plt.show()
        plt.close()
        
        return topic_dist
    
    def create_confidence_analysis(self):
        """Analyze and visualize model confidence scores"""
        print("📈 Creating confidence analysis...")
        
        # 1. Overall confidence distribution
        plt.figure(figsize=(12, 8))
        plt.hist(self.classified_notes['confidence'], bins=50, alpha=0.7, color='#4ECDC4', edgecolor='black')
        plt.axvline(self.classified_notes['confidence'].mean(), color='#FF6B6B', linestyle='--', linewidth=2,
                   label=f'Mean: {self.classified_notes["confidence"].mean():.3f}')
        plt.axvline(self.classified_notes['confidence'].median(), color='#45B7D1', linestyle='--', linewidth=2,
                   label=f'Median: {self.classified_notes["confidence"].median():.3f}')
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Number of Notes', fontsize=12)
        plt.title('Distribution of Classification Confidence Scores', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save confidence distribution
        output_file = self.output_dir / "confidence_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_file}")
        plt.show()
        plt.close()
        
        # 2. Confidence by topic (box plot)
        plt.figure(figsize=(14, 10))
        topic_order = self.classified_notes['topicName'].value_counts().index[:12]  # Top 12 topics
        subset_data = self.classified_notes[self.classified_notes['topicName'].isin(topic_order)]
        
        # Create custom color palette for box plot
        box_colors = DISTINCT_COLORS[:len(topic_order)]
        sns.boxplot(data=subset_data, y='topicName', x='confidence', order=topic_order, palette=box_colors)
        plt.title('Confidence Distribution by Topic (Top 12)', fontsize=16, fontweight='bold')
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Topic', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save confidence by topic
        output_file = self.output_dir / "confidence_by_topic.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_file}")
        plt.show()
        plt.close()
        
        # 3. Low confidence notes by topic
        plt.figure(figsize=(14, 8))
        low_confidence = self.classified_notes[self.classified_notes['confidence'] < 0.5]
        low_conf_by_topic = low_confidence['topicName'].value_counts()
        
        low_conf_colors = DISTINCT_COLORS[:len(low_conf_by_topic)]
        bars = plt.bar(range(len(low_conf_by_topic)), low_conf_by_topic.values, color=low_conf_colors)
        plt.xticks(range(len(low_conf_by_topic)), low_conf_by_topic.index, rotation=45, ha='right')
        plt.ylabel('Number of Low Confidence Notes', fontsize=12)
        plt.title('Low Confidence Notes by Topic (< 0.5)', fontsize=16, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, low_conf_by_topic.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                    f'{value:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save low confidence analysis
        output_file = self.output_dir / "low_confidence_by_topic.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_file}")
        plt.show()
        plt.close()
        
        # 4. Confidence vs Topic Size correlation
        plt.figure(figsize=(12, 8))
        topic_counts = self.classified_notes['topicName'].value_counts()
        topic_conf_means = self.classified_notes.groupby('topicName')['confidence'].mean()
        
        # Merge the data
        topic_analysis = pd.DataFrame({
            'count': topic_counts,
            'mean_confidence': topic_conf_means
        }).dropna()
        
        # Use different colors for different topic categories
        conflict_topics = ['UkraineConflict', 'GazaConflict', 'SyriaWar', 'Iran', 'ChinaTaiwan', 'ChinaInfluence', 'OtherConflicts']
        colors = ['#FF6B6B' if topic in conflict_topics else '#4ECDC4' for topic in topic_analysis.index]
        
        scatter = plt.scatter(topic_analysis['count'], topic_analysis['mean_confidence'], 
                   s=150, alpha=0.7, c=colors, edgecolors='black', linewidth=1)
        
        # Add topic labels
        for topic, row in topic_analysis.iterrows():
            plt.annotate(topic, (row['count'], row['mean_confidence']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
        
        plt.xlabel('Number of Notes in Topic', fontsize=12)
        plt.ylabel('Mean Confidence Score', fontsize=12)
        plt.title('Topic Size vs Mean Confidence', fontsize=16, fontweight='bold')
        plt.grid(alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#FF6B6B', label='Geopolitical/Conflict Topics'),
                          Patch(facecolor='#4ECDC4', label='General Topics')]
        plt.legend(handles=legend_elements, fontsize=12)
        
        plt.tight_layout()
        
        # Save correlation analysis
        output_file = self.output_dir / "topic_size_vs_confidence.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_file}")
        plt.show()
        plt.close()
        
        return low_conf_by_topic, topic_analysis
    
    def create_geopolitical_focus_chart(self):
        """Create specific analysis for geopolitical/conflict topics"""
        print("🌍 Creating geopolitical topics analysis...")
        
        # Define geopolitical categories
        conflict_topics = [
            'UkraineConflict', 'GazaConflict', 'SyriaWar', 'Iran', 
            'ChinaTaiwan', 'ChinaInfluence', 'OtherConflicts'
        ]
        
        general_topics = [
            'Politics', 'HealthMedical', 'Scams', 'Entertainment', 
            'Immigration', 'Economics', 'Technology', 'ClimateEnvironment'
        ]
        
        # Calculate distributions
        conflict_data = self.classified_notes[
            self.classified_notes['topicName'].isin(conflict_topics)
        ]['topicName'].value_counts()
        
        general_data = self.classified_notes[
            self.classified_notes['topicName'].isin(general_topics)
        ]['topicName'].value_counts()
        
        # 1. Conflict topics chart
        plt.figure(figsize=(14, 8))
        conflict_colors = ['#FF6B6B', '#E74C3C', '#C0392B', '#A93226', '#922B21', '#7B2C3B', '#641E16']
        bars1 = plt.bar(range(len(conflict_data)), conflict_data.values, color=conflict_colors[:len(conflict_data)])
        plt.xticks(range(len(conflict_data)), conflict_data.index, rotation=45, ha='right')
        plt.ylabel('Number of Notes', fontsize=12)
        plt.title('Geopolitical & Conflict Topics\n(Total: {:,} notes)'.format(conflict_data.sum()), 
                 fontsize=16, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, conflict_data.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                    f'{value:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save conflict topics chart
        output_file = self.output_dir / "geopolitical_topics.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_file}")
        plt.show()
        plt.close()
        
        # 2. General topics chart
        plt.figure(figsize=(14, 8))
        general_colors = ['#4ECDC4', '#45B7D1', '#5DADE2', '#85C1E9', '#AED6F1', '#D6EAF8', '#EBF5FB', '#F8F9FA']
        bars2 = plt.bar(range(len(general_data)), general_data.values, color=general_colors[:len(general_data)])
        plt.xticks(range(len(general_data)), general_data.index, rotation=45, ha='right')
        plt.ylabel('Number of Notes', fontsize=12)
        plt.title('General Topics\n(Total: {:,} notes)'.format(general_data.sum()), 
                 fontsize=16, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, general_data.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                    f'{value:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save general topics chart
        output_file = self.output_dir / "general_topics.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_file}")
        plt.show()
        plt.close()
        
        # 3. Comparison pie chart
        plt.figure(figsize=(10, 8))
        comparison_data = pd.Series([
            conflict_data.sum(),
            general_data.sum()
        ], index=['Geopolitical/Conflicts', 'General Topics'])
        
        colors3 = ['#FF6B6B', '#4ECDC4']
        wedges, texts, autotexts = plt.pie(comparison_data.values, 
                                          labels=comparison_data.index,
                                          autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*len(self.classified_notes)):,})',
                                          colors=colors3, startangle=90, textprops={'fontsize': 12})
        plt.title('Geopolitical vs General Topics', fontsize=16, fontweight='bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        plt.tight_layout()
        
        # Save comparison chart
        output_file = self.output_dir / "geopolitical_vs_general.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_file}")
        plt.show()
        plt.close()
        
        return conflict_data, general_data
    
    def create_time_series_analysis(self):
        """Analyze topics over time if timestamp data is available"""
        print("📅 Creating temporal analysis...")
        
        # Check if we have timestamp data
        if 'createdAtMillis' not in self.classified_notes.columns:
            print("⚠️  No timestamp data available for temporal analysis")
            return None
        
        # Convert timestamp to datetime
        self.classified_notes['created_date'] = pd.to_datetime(
            self.classified_notes['createdAtMillis'], unit='ms'
        )
        
        # Filter data to June 2023 onwards for relevance
        cutoff_date = pd.Timestamp('2023-06-01')
        recent_notes = self.classified_notes[self.classified_notes['created_date'] >= cutoff_date].copy()
        
        print(f"📊 Filtering to data from June 2023 onwards...")
        print(f"   Original notes: {len(self.classified_notes):,}")
        print(f"   Filtered notes: {len(recent_notes):,} (from {cutoff_date.strftime('%B %Y')})")
        
        if len(recent_notes) == 0:
            print("⚠️  No data available from January 2023 onwards")
            return None
        
        # Group by month and topic
        monthly_data = recent_notes.groupby([
            recent_notes['created_date'].dt.to_period('M'),
            'topicName'
        ]).size().unstack(fill_value=0)
        
        # Define specific topics to show
        selected_topics = ['Politics','UkraineConflict', 'Immigration', 'GazaConflict']
        
        plt.figure(figsize=(16, 10))
        
        # Use distinct colors for each topic
        topic_colors = {
            'Politics': '#FF6B6B',
            'Scams': '#4ECDC4', 
            'UkraineConflict': '#45B7D1',
            'Immigration': '#96CEB4',
            'GazaConflict': '#FFEAA7'
        }
        
        # Plot topic lines
        for topic in selected_topics:
            if topic in monthly_data.columns:
                plt.plot(monthly_data.index.to_timestamp(), monthly_data[topic], 
                        marker='o', linewidth=3, label=topic, color=topic_colors[topic], markersize=6)
        
        # Define event dates and labels with numbers
        events = [
            ('2023-10-07', 'October 7 War'),
            ('2024-05-10', 'Russian Offensive on Kharkiv'), 
            ('2024-06-27', 'First U.S. Presidential Debate'),
            ('2024-09-10', 'Trump–Harris Debate'),
            ('2024-10-01', 'Hunter Biden trial gains traction'),
            ('2024-11-05', 'U.S. Presidential Election Day'),
            ('2025-01-20', 'Inauguration Day')
        ]
        
        # Add vertical lines for events (grey/black colors)
        event_colors = ['#2C3E50', '#34495E', '#5D6D7E', '#85929E', '#AEB6BF', '#D5D8DC', '#BDC3C7']
        event_handles = []
        
        for i, (date_str, event_name) in enumerate(events):
            event_date = pd.Timestamp(date_str)
            if event_date >= cutoff_date and event_date <= pd.Timestamp.now():
                line = plt.axvline(x=event_date, color=event_colors[i], 
                                 linestyle='--', linewidth=2, alpha=0.8)
                event_handles.append((line, f'{i+1}. {event_name}'))
                
                # Add number annotation next to the vertical line
                # Get the y-axis limits to position the text
                y_max = plt.ylim()[1]
                plt.text(event_date, y_max * 0.95, str(i+1), 
                        fontsize=12, fontweight='bold', ha='center', va='top',
                        bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', 
                                edgecolor=event_colors[i], linewidth=2))
        
        plt.xlabel('Date', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Notes', fontsize=14, fontweight='bold')
        plt.title('Topic Trends Over Time (January 2023 - Present)', fontsize=18, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        
        # Create separate legends
        # Topics legend
        topic_legend = plt.legend(handles=[plt.Line2D([0], [0], color=topic_colors[topic], linewidth=3, 
                                                     marker='o', markersize=6, label=topic) 
                                          for topic in selected_topics if topic in monthly_data.columns],
                                 title='Topics', loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)
        topic_legend.get_title().set_fontsize(14)
        topic_legend.get_title().set_fontweight('bold')
        plt.gca().add_artist(topic_legend)
        
        # Events legend
        if event_handles:
            event_legend = plt.legend(handles=[line for line, _ in event_handles],
                                    labels=[label for _, label in event_handles],
                                    title='Events', loc='upper left', bbox_to_anchor=(1.02, 0.6), 
                                    fontsize=12)
            event_legend.get_title().set_fontsize(14)
            event_legend.get_title().set_fontweight('bold')
        
        # Set x-axis to show months more clearly
        import matplotlib.dates as mdates
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / "temporal_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_file}")
        
        plt.show()
        plt.close()
        
        return monthly_data
    
    def create_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        print("📊 Generating summary statistics...")
        
        stats = {}
        
        # Basic statistics
        stats['total_notes'] = len(self.classified_notes)
        stats['unique_topics'] = self.classified_notes['topicName'].nunique()
        
        # Topic distribution
        topic_dist = self.classified_notes['topicName'].value_counts()
        stats['largest_topic'] = topic_dist.index[0]
        stats['largest_topic_count'] = topic_dist.iloc[0]
        stats['largest_topic_percentage'] = (topic_dist.iloc[0] / len(self.classified_notes)) * 100
        
        stats['smallest_topic'] = topic_dist.index[-1]
        stats['smallest_topic_count'] = topic_dist.iloc[-1]
        stats['smallest_topic_percentage'] = (topic_dist.iloc[-1] / len(self.classified_notes)) * 100
        
        # Confidence statistics
        stats['mean_confidence'] = self.classified_notes['confidence'].mean()
        stats['median_confidence'] = self.classified_notes['confidence'].median()
        stats['low_confidence_count'] = len(self.classified_notes[self.classified_notes['confidence'] < 0.5])
        stats['low_confidence_percentage'] = (stats['low_confidence_count'] / len(self.classified_notes)) * 100
        
        # Geopolitical vs General
        conflict_topics = [
            'UkraineConflict', 'GazaConflict', 'SyriaWar', 'Iran', 
            'ChinaTaiwan', 'ChinaInfluence', 'OtherConflicts'
        ]
        geopolitical_count = len(self.classified_notes[
            self.classified_notes['topicName'].isin(conflict_topics)
        ])
        stats['geopolitical_count'] = geopolitical_count
        stats['geopolitical_percentage'] = (geopolitical_count / len(self.classified_notes)) * 100
        
        # Print summary
        print("\n" + "="*70)
        print("CLASSIFICATION SUMMARY STATISTICS")
        print("="*70)
        
        print(f"📊 Dataset Overview:")
        print(f"   Total notes classified: {stats['total_notes']:,}")
        print(f"   Unique topics: {stats['unique_topics']}")
        
        print(f"\n🏆 Topic Distribution:")
        print(f"   Largest topic: {stats['largest_topic']} ({stats['largest_topic_count']:,} notes, {stats['largest_topic_percentage']:.1f}%)")
        print(f"   Smallest topic: {stats['smallest_topic']} ({stats['smallest_topic_count']:,} notes, {stats['smallest_topic_percentage']:.1f}%)")
        
        print(f"\n📈 Confidence Analysis:")
        print(f"   Mean confidence: {stats['mean_confidence']:.3f}")
        print(f"   Median confidence: {stats['median_confidence']:.3f}")
        print(f"   Low confidence notes (<0.5): {stats['low_confidence_count']:,} ({stats['low_confidence_percentage']:.1f}%)")
        
        print(f"\n🌍 Content Analysis:")
        print(f"   Geopolitical/Conflict topics: {stats['geopolitical_count']:,} ({stats['geopolitical_percentage']:.1f}%)")
        print(f"   General topics: {stats['total_notes'] - stats['geopolitical_count']:,} ({100 - stats['geopolitical_percentage']:.1f}%)")
        
        # Save statistics (convert numpy types to Python types for JSON)
        stats_serializable = {}
        for key, value in stats.items():
            if hasattr(value, 'item'):  # numpy scalar
                stats_serializable[key] = value.item()
            else:
                stats_serializable[key] = value
        
        stats_file = self.output_dir / "summary_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats_serializable, f, indent=2)
        print(f"\n💾 Statistics saved to: {stats_file}")
        
        return stats
    
    def run_complete_analysis(self):
        """Run all analytics and create all visualizations"""
        print("="*70)
        print("COMPREHENSIVE CLASSIFICATION ANALYTICS")
        print("="*70)
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        self.load_latest_results()
        
        # # Generate all analyses
        # print("\n🎯 Running comprehensive analytics...")
        
        # # 1. Topic distribution
        # topic_dist = self.create_topic_distribution_chart()
        
        # # 2. Confidence analysis
        # low_conf, topic_analysis = self.create_confidence_analysis()
        
        # # 3. Geopolitical analysis
        # conflict_data, general_data = self.create_geopolitical_focus_chart()
        
        # 4. Temporal analysis (if available)
        temporal_data = self.create_time_series_analysis()
        
        # # 5. Summary statistics
        # stats = self.create_summary_statistics()
        
        # print(f"\n✅ Analytics completed!")
        # print(f"📁 All files saved in: {self.output_dir.absolute()}")
        
        # return {
        #     'topic_distribution': topic_dist,
        #     'confidence_analysis': (low_conf, topic_analysis),
        #     'geopolitical_analysis': (conflict_data, general_data),
        #     'temporal_analysis': temporal_data,
        #     'summary_statistics': stats
        # }

def main():
    """Main function to run analytics"""
    analyzer = ClassificationAnalyzer()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    main()
