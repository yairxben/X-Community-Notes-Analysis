#!/usr/bin/env python3
"""
Main Entry Point for X Community Notes Analysis

This script provides a unified interface to run the complete analysis pipeline
or individual components.

Usage:
    python run_analysis.py [command] [options]
    
Commands:
    classify    - Run topic classification
    analyze     - Analyze classification results
    report      - Generate summary report
    demo        - Run demo workflow
    all         - Run complete pipeline (classify + analyze + report)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from code.classification.topic_classifier import CustomTopicClassifier, run_test
from code.classification.analyze_classification_results import ClassificationAnalyzer
from code.generate_report import ReportGenerator
from code.demo_workflow import demo_classification, demo_analysis, print_summary


def run_classification(args):
    """Run topic classification."""
    print("\n" + "="*70)
    print("RUNNING TOPIC CLASSIFICATION")
    print("="*70)
    
    classifier = CustomTopicClassifier(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    results = classifier.run_complete_pipeline(
        max_notes=args.max_notes if args.max_notes else None,
        english_only=args.english_only,
        force_refilter=args.force_refilter
    )
    
    print("\nClassification complete!")
    return results


def run_analysis(args):
    """Run results analysis."""
    print("\n" + "="*70)
    print("RUNNING RESULTS ANALYSIS")
    print("="*70)
    
    analyzer = ClassificationAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.analysis_output
    )
    
    results = analyzer.run_complete_analysis()
    
    print("\nAnalysis complete!")
    return results


def run_report(args):
    """Generate summary report."""
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)
    
    generator = ReportGenerator(output_dir=args.report_output)
    report_file = generator.generate_report(results_dir=args.results_dir)
    
    print("\nReport generated!")
    return report_file


def run_all(args):
    """Run complete pipeline."""
    print("\n" + "="*70)
    print("RUNNING COMPLETE ANALYSIS PIPELINE")
    print("="*70)
    
    # Step 1: Classification
    classification_results = run_classification(args)
    
    # Step 2: Analysis
    results_dir = classification_results.get('file_paths', {}).get('classified_notes', args.output_dir)
    if isinstance(results_dir, str) and results_dir.endswith('.csv'):
        results_dir = Path(results_dir).parent
    
    analysis_args = argparse.Namespace(
        results_dir=str(results_dir),
        analysis_output=args.analysis_output
    )
    analysis_results = run_analysis(analysis_args)
    
    # Step 3: Report
    report_args = argparse.Namespace(
        results_dir=str(results_dir),
        report_output=args.report_output
    )
    report_file = run_report(report_args)
    
    print("\n" + "="*70)
    print("COMPLETE PIPELINE FINISHED")
    print("="*70)
    print("\nAll outputs saved:")
    print(f"  • Classification: {args.output_dir}")
    print(f"  • Analysis: {args.analysis_output}")
    print(f"  • Report: {report_file}")
    
    return {
        'classification': classification_results,
        'analysis': analysis_results,
        'report': str(report_file)
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="X Community Notes Analysis - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py classify --max-notes 10000
  python run_analysis.py analyze --results-dir custom_topic_results
  python run_analysis.py report
  python run_analysis.py all --max-notes 10000
  python run_analysis.py demo --quick
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Classification command
    classify_parser = subparsers.add_parser('classify', help='Run topic classification')
    classify_parser.add_argument('--data-path', default='data', help='Path to data directory')
    classify_parser.add_argument('--output-dir', default='custom_topic_results', help='Output directory')
    classify_parser.add_argument('--max-notes', type=int, help='Maximum notes to process')
    classify_parser.add_argument('--english-only', action='store_true', default=True, help='Filter to English only')
    classify_parser.add_argument('--force-refilter', action='store_true', help='Force re-filtering')
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze classification results')
    analyze_parser.add_argument('--results-dir', default='custom_topic_results', help='Results directory')
    analyze_parser.add_argument('--analysis-output', default='classification_analytics', help='Analysis output directory')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate summary report')
    report_parser.add_argument('--results-dir', default='custom_topic_results', help='Results directory')
    report_parser.add_argument('--report-output', default='reports', help='Report output directory')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo workflow')
    demo_parser.add_argument('--quick', action='store_true', help='Quick demo (10K notes)')
    demo_parser.add_argument('--full', action='store_true', help='Full demo (all notes)')
    demo_parser.add_argument('--analysis-only', action='store_true', help='Skip classification, only run analysis on existing results')
    
    # All command (combines classify, analyze, report)
    all_parser = subparsers.add_parser('all', help='Run complete pipeline')
    all_parser.add_argument('--data-path', default='data', help='Path to data directory')
    all_parser.add_argument('--output-dir', default='custom_topic_results', help='Classification output directory')
    all_parser.add_argument('--analysis-output', default='classification_analytics', help='Analysis output directory')
    all_parser.add_argument('--report-output', default='reports', help='Report output directory')
    all_parser.add_argument('--max-notes', type=int, help='Maximum notes to process')
    all_parser.add_argument('--english-only', action='store_true', default=True, help='Filter to English only')
    all_parser.add_argument('--force-refilter', action='store_true', help='Force re-filtering')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'classify':
            run_classification(args)
        elif args.command == 'analyze':
            run_analysis(args)
        elif args.command == 'report':
            run_report(args)
        elif args.command == 'demo':
            from code.demo_workflow import main as demo_main
            # Pass the args from the demo subparser to demo_main
            demo_main(args)
        elif args.command == 'all':
            run_all(args)
        else:
            parser.print_help()
            return 1
        
        print("\nCommand completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\nWARNING: Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nERROR: Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

