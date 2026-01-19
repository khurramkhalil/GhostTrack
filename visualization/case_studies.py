"""
Case study generation for detailed analysis.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

from tracking import HypothesisTracker
from detection import DivergenceMetrics, HallucinationDetector
from data import HallucinationExample


class CaseStudyGenerator:
    """Generate detailed case studies for analysis."""

    def __init__(
        self,
        detector: HallucinationDetector,
        num_layers: int = 12
    ):
        """
        Initialize case study generator.

        Args:
            detector: Trained HallucinationDetector.
            num_layers: Number of layers.
        """
        self.detector = detector
        self.num_layers = num_layers

    def generate_case_study(
        self,
        example: HallucinationExample,
        tracker_factual: HypothesisTracker,
        tracker_hallucinated: HypothesisTracker,
        output_dir: str = './case_studies'
    ) -> Dict:
        """
        Generate detailed case study comparing factual and hallucinated answers.

        Args:
            example: TruthfulQA example.
            tracker_factual: Tracker for factual answer.
            tracker_hallucinated: Tracker for hallucinated answer.
            output_dir: Output directory.

        Returns:
            Case study dictionary.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Compute metrics for both
        # Compute metrics for both
        metrics_computer = DivergenceMetrics()
        metrics_factual = metrics_computer.compute_all_metrics(
            tracker_factual, self.num_layers
        )
        metrics_hallucinated = metrics_computer.compute_all_metrics(
            tracker_hallucinated, self.num_layers
        )

        # Get predictions
        pred_factual = self.detector.predict_proba([tracker_factual])[0, 1]
        pred_hallucinated = self.detector.predict_proba([tracker_hallucinated])[0, 1]

        # Compare metrics
        metric_comparison = self._compare_metrics(
            metrics_factual, metrics_hallucinated
        )

        # Analyze tracks
        track_analysis_factual = self._analyze_tracks(tracker_factual)
        track_analysis_hallucinated = self._analyze_tracks(tracker_hallucinated)

        # Build case study
        case_study = {
            'question': example.prompt,
            'factual_answer': example.factual_answer,
            'hallucinated_answer': example.hallucinated_answer,
            'predictions': {
                'factual': float(pred_factual),
                'hallucinated': float(pred_hallucinated),
                'correct_classification': bool(pred_hallucinated > pred_factual)
            },
            'metrics': {
                'factual': {k: float(v) for k, v in metrics_factual.items()},
                'hallucinated': {k: float(v) for k, v in metrics_hallucinated.items()},
                'comparison': metric_comparison
            },
            'track_analysis': {
                'factual': track_analysis_factual,
                'hallucinated': track_analysis_hallucinated
            },
            'insights': self._generate_insights(
                metric_comparison,
                track_analysis_factual,
                track_analysis_hallucinated
            )
        }

        # Save case study
        study_id = abs(hash(example.prompt)) % 10000
        json_path = output_path / f'case_study_{study_id}.json'
        with open(json_path, 'w') as f:
            json.dump(case_study, f, indent=2)

        # Generate markdown report
        report = self._generate_markdown_report(case_study)
        md_path = output_path / f'case_study_{study_id}.md'
        with open(md_path, 'w') as f:
            f.write(report)

        print(f"Case study saved to {md_path}")

        return case_study

    def _compare_metrics(
        self,
        metrics_factual: Dict[str, float],
        metrics_hallucinated: Dict[str, float]
    ) -> Dict:
        """Compare metrics between factual and hallucinated."""
        comparison = {}

        for key in metrics_factual.keys():
            factual_val = metrics_factual[key]
            halluc_val = metrics_hallucinated[key]

            diff = halluc_val - factual_val
            pct_change = (diff / factual_val * 100) if factual_val != 0 else 0

            comparison[key] = {
                'factual': float(factual_val),
                'hallucinated': float(halluc_val),
                'difference': float(diff),
                'percent_change': float(pct_change),
                'higher_in_hallucination': bool(halluc_val > factual_val)
            }

        return comparison

    def _analyze_tracks(self, tracker: HypothesisTracker) -> Dict:
        """Analyze tracks in tracker."""
        all_tracks = tracker.tracks

        if not all_tracks:
            return {
                'total_tracks': 0,
                'avg_lifespan': 0,
                'max_activation': 0,
                'competition_score': 0
            }

        lifespans = [len(t.trajectory) for t in all_tracks]
        max_activations = [
            max([act for _, act, _ in t.trajectory])
            for t in all_tracks
        ]

        # Compute competition: variance in activations
        all_activations = []
        for track in all_tracks:
            all_activations.extend([act for _, act, _ in track.trajectory])

        competition = np.var(all_activations) if all_activations else 0

        return {
            'total_tracks': len(all_tracks),
            'avg_lifespan': float(np.mean(lifespans)),
            'std_lifespan': float(np.std(lifespans)),
            'max_activation': float(np.max(max_activations)),
            'mean_activation': float(np.mean(all_activations)) if all_activations else 0,
            'competition_score': float(competition),
            'num_active': len([t for t in all_tracks if t.death_layer is None]),
            'num_dead': len([t for t in all_tracks if t.death_layer is not None])
        }

    def _generate_insights(
        self,
        metric_comparison: Dict,
        track_analysis_factual: Dict,
        track_analysis_hallucinated: Dict
    ) -> List[str]:
        """Generate insights from comparison."""
        insights = []

        # Check entropy
        entropy_comp = metric_comparison.get('entropy_mean', {})
        if entropy_comp.get('higher_in_hallucination', False):
            pct = entropy_comp.get('percent_change', 0)
            insights.append(
                f"Entropy is {pct:.1f}% higher in hallucination, "
                "indicating more uncertainty in hypothesis selection."
            )

        # Check churn
        churn_comp = metric_comparison.get('churn_rate', {})
        if churn_comp.get('higher_in_hallucination', False):
            pct = churn_comp.get('percent_change', 0)
            insights.append(
                f"Churn rate is {pct:.1f}% higher in hallucination, "
                "showing more unstable hypothesis formation."
            )

        # Check competition
        comp_comp = metric_comparison.get('competition_mean', {})
        if comp_comp.get('higher_in_hallucination', False):
            insights.append(
                "Higher competition between hypotheses in hallucination, "
                "suggesting the model is considering multiple alternatives."
            )

        # Check track lifespans
        lifespan_fact = track_analysis_factual.get('avg_lifespan', 0)
        lifespan_hall = track_analysis_hallucinated.get('avg_lifespan', 0)
        if lifespan_hall < lifespan_fact:
            insights.append(
                f"Tracks have shorter lifespan in hallucination "
                f"({lifespan_hall:.2f} vs {lifespan_fact:.2f} layers), "
                "indicating less stable representations."
            )

        # Check number of tracks
        num_tracks_fact = track_analysis_factual.get('total_tracks', 0)
        num_tracks_hall = track_analysis_hallucinated.get('total_tracks', 0)
        if num_tracks_hall > num_tracks_fact:
            pct = ((num_tracks_hall - num_tracks_fact) / num_tracks_fact * 100)
            insights.append(
                f"Hallucination has {pct:.1f}% more tracks, "
                "showing higher hypothesis density."
            )

        if not insights:
            insights.append("Metrics are similar between factual and hallucinated answers.")

        return insights

    def _generate_markdown_report(self, case_study: Dict) -> str:
        """Generate markdown report."""
        report = f"""# Case Study: Hallucination Detection

## Question
{case_study['question']}

## Answers

### Factual Answer
{case_study['factual_answer']}

**Prediction Score**: {case_study['predictions']['factual']:.3f}

### Hallucinated Answer
{case_study['hallucinated_answer']}

**Prediction Score**: {case_study['predictions']['hallucinated']:.3f}

## Classification Result
{'✅ **CORRECT**' if case_study['predictions']['correct_classification'] else '❌ **INCORRECT**'}: Hallucination score ({case_study['predictions']['hallucinated']:.3f}) {'>' if case_study['predictions']['correct_classification'] else '<'} Factual score ({case_study['predictions']['factual']:.3f})

---

## Track Analysis

### Factual Answer
- **Total Tracks**: {case_study['track_analysis']['factual']['total_tracks']}
- **Avg Lifespan**: {case_study['track_analysis']['factual']['avg_lifespan']:.2f} layers
- **Max Activation**: {case_study['track_analysis']['factual']['max_activation']:.3f}
- **Competition Score**: {case_study['track_analysis']['factual']['competition_score']:.3f}

### Hallucinated Answer
- **Total Tracks**: {case_study['track_analysis']['hallucinated']['total_tracks']}
- **Avg Lifespan**: {case_study['track_analysis']['hallucinated']['avg_lifespan']:.2f} layers
- **Max Activation**: {case_study['track_analysis']['hallucinated']['max_activation']:.3f}
- **Competition Score**: {case_study['track_analysis']['hallucinated']['competition_score']:.3f}

---

## Key Metrics Comparison

| Metric | Factual | Hallucinated | Difference | % Change |
|--------|---------|--------------|------------|----------|
"""
        # Add top metrics
        top_metrics = [
            'entropy_mean', 'churn_rate', 'competition_mean',
            'stability_mean', 'dominance_top1'
        ]

        for metric in top_metrics:
            if metric in case_study['metrics']['comparison']:
                comp = case_study['metrics']['comparison'][metric]
                report += f"| {metric} | {comp['factual']:.3f} | {comp['hallucinated']:.3f} | "
                report += f"{comp['difference']:+.3f} | {comp['percent_change']:+.1f}% |\n"

        report += f"""
---

## Insights

"""
        for i, insight in enumerate(case_study['insights'], 1):
            report += f"{i}. {insight}\n\n"

        report += """
---

*Generated by GhostTrack Case Study Generator*
"""
        return report

    def generate_batch_studies(
        self,
        examples: List[HallucinationExample],
        trackers_factual: List[HypothesisTracker],
        trackers_hallucinated: List[HypothesisTracker],
        output_dir: str = './case_studies',
        num_studies: int = 10
    ) -> List[Dict]:
        """
        Generate multiple case studies.

        Args:
            examples: List of examples.
            trackers_factual: List of factual trackers.
            trackers_hallucinated: List of hallucinated trackers.
            output_dir: Output directory.
            num_studies: Number of studies to generate.

        Returns:
            List of case study dictionaries.
        """
        case_studies = []

        for i in range(min(num_studies, len(examples))):
            print(f"\nGenerating case study {i+1}/{num_studies}...")
            study = self.generate_case_study(
                examples[i],
                trackers_factual[i],
                trackers_hallucinated[i],
                output_dir
            )
            case_studies.append(study)

        # Generate summary
        summary_path = Path(output_dir) / 'summary.md'
        self._generate_summary(case_studies, summary_path)

        return case_studies

    def _generate_summary(self, case_studies: List[Dict], output_path: Path):
        """Generate summary of all case studies."""
        correct = sum(1 for s in case_studies if s['predictions']['correct_classification'])
        total = len(case_studies)

        summary = f"""# Case Studies Summary

**Total Studies**: {total}
**Correct Classifications**: {correct}/{total} ({correct/total*100:.1f}%)

## Study Results

| Study | Question Preview | Correct | Factual Score | Halluc Score |
|-------|-----------------|---------|---------------|--------------|
"""
        for i, study in enumerate(case_studies, 1):
            question_preview = study['question'][:50] + '...'
            correct_mark = '✅' if study['predictions']['correct_classification'] else '❌'
            summary += f"| {i} | {question_preview} | {correct_mark} | "
            summary += f"{study['predictions']['factual']:.3f} | "
            summary += f"{study['predictions']['hallucinated']:.3f} |\n"

        with open(output_path, 'w') as f:
            f.write(summary)

        print(f"\nSummary saved to {output_path}")
