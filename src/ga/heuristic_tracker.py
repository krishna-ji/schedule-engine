"""
Heuristic Application Tracker

Tracks which heuristics are applied in each generation, their effects,
and generates detailed reports and visualizations.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class HeuristicApplication:
    """Record of a single heuristic application."""
    
    generation: int
    heuristic_name: str
    category: str
    fitness_before: Tuple[float, float]  # (hard_violations, soft_penalty)
    fitness_after: Tuple[float, float]
    improvement: float  # Positive = better
    execution_time: float  # seconds
    success: bool  # Whether it improved fitness
    individual_id: int  # Index in population


@dataclass
class GenerationStats:
    """Aggregated stats for a generation."""
    
    generation: int
    applications: List[HeuristicApplication] = field(default_factory=list)
    total_improvements: float = 0.0
    successful_applications: int = 0
    failed_applications: int = 0
    best_heuristic: Optional[str] = None
    best_improvement: float = 0.0
    execution_time: float = 0.0


class HeuristicTracker:
    """
    Track heuristic applications across generations.
    
    Features:
    - Per-generation statistics
    - Per-heuristic performance metrics
    - Temporal analysis (which heuristic works when)
    - JSON export for analysis
    - Visualization plots
    """
    
    def __init__(self):
        self.applications: List[HeuristicApplication] = []
        self.generation_stats: Dict[int, GenerationStats] = {}
        
        # Per-heuristic cumulative stats
        self.heuristic_stats = defaultdict(lambda: {
            'total_applications': 0,
            'successful_applications': 0,
            'total_improvement': 0.0,
            'average_improvement': 0.0,
            'best_improvement': 0.0,
            'worst_improvement': 0.0,
            'total_time': 0.0,
            'average_time': 0.0,
            'success_rate': 0.0,
            'generations_applied': set(),
        })
        
        # Round-robin state
        self.current_heuristic_index = 0
        self.heuristic_order: List[str] = []
    
    def set_heuristic_order(self, heuristic_names: List[str]) -> None:
        """Set the order of heuristics for round-robin rotation."""
        self.heuristic_order = heuristic_names
        self.current_heuristic_index = 0
    
    def get_next_heuristic(self) -> str:
        """Get next heuristic in round-robin order."""
        if not self.heuristic_order:
            raise ValueError("Heuristic order not set. Call set_heuristic_order() first.")
        
        heuristic = self.heuristic_order[self.current_heuristic_index]
        self.current_heuristic_index = (self.current_heuristic_index + 1) % len(self.heuristic_order)
        return heuristic
    
    def record_application(
        self,
        generation: int,
        heuristic_name: str,
        category: str,
        fitness_before: Tuple[float, float],
        fitness_after: Tuple[float, float],
        execution_time: float,
        individual_id: int = 0,
    ) -> None:
        """
        Record a heuristic application.
        
        Args:
            generation: Current generation number
            heuristic_name: Name of heuristic applied
            category: Category (construction/perturbation/improvement/diversity/meta)
            fitness_before: Fitness before application (hard, soft)
            fitness_after: Fitness after application (hard, soft)
            execution_time: Time taken in seconds
            individual_id: Index of individual in population
        """
        # Calculate improvement (negative = better for minimization)
        # We want positive values for improvements
        hard_improvement = fitness_before[0] - fitness_after[0]
        soft_improvement = fitness_before[1] - fitness_after[1]
        
        # Combined improvement (weight hard violations more)
        improvement = hard_improvement + (soft_improvement * 0.01)
        success = improvement > 0
        
        app = HeuristicApplication(
            generation=generation,
            heuristic_name=heuristic_name,
            category=category,
            fitness_before=fitness_before,
            fitness_after=fitness_after,
            improvement=improvement,
            execution_time=execution_time,
            success=success,
            individual_id=individual_id,
        )
        
        self.applications.append(app)
        
        # Update generation stats
        if generation not in self.generation_stats:
            self.generation_stats[generation] = GenerationStats(generation=generation)
        
        gen_stats = self.generation_stats[generation]
        gen_stats.applications.append(app)
        gen_stats.execution_time += execution_time
        
        if success:
            gen_stats.successful_applications += 1
            gen_stats.total_improvements += improvement
            
            if improvement > gen_stats.best_improvement:
                gen_stats.best_improvement = improvement
                gen_stats.best_heuristic = heuristic_name
        else:
            gen_stats.failed_applications += 1
        
        # Update per-heuristic stats
        stats = self.heuristic_stats[heuristic_name]
        stats['total_applications'] += 1
        stats['generations_applied'].add(generation)
        stats['total_time'] += execution_time
        
        if success:
            stats['successful_applications'] += 1
            stats['total_improvement'] += improvement
            stats['best_improvement'] = max(stats['best_improvement'], improvement)
        else:
            stats['worst_improvement'] = min(stats['worst_improvement'], improvement)
        
        # Update averages
        stats['average_improvement'] = (
            stats['total_improvement'] / stats['total_applications']
        )
        stats['average_time'] = stats['total_time'] / stats['total_applications']
        stats['success_rate'] = (
            stats['successful_applications'] / stats['total_applications'] * 100
        )
    
    def get_summary(self) -> Dict:
        """Get overall summary statistics."""
        if not self.applications:
            return {
                'total_applications': 0,
                'message': 'No heuristic applications recorded'
            }
        
        total_apps = len(self.applications)
        successful = sum(1 for app in self.applications if app.success)
        total_improvement = sum(app.improvement for app in self.applications if app.success)
        total_time = sum(app.execution_time for app in self.applications)
        
        # Find best heuristic overall
        best_heuristic = max(
            self.heuristic_stats.items(),
            key=lambda x: x[1]['total_improvement']
        )
        
        return {
            'total_applications': total_apps,
            'successful_applications': successful,
            'failed_applications': total_apps - successful,
            'success_rate_percent': (successful / total_apps * 100) if total_apps > 0 else 0,
            'total_improvement': total_improvement,
            'average_improvement': total_improvement / successful if successful > 0 else 0,
            'total_time_seconds': total_time,
            'average_time_seconds': total_time / total_apps if total_apps > 0 else 0,
            'best_heuristic': best_heuristic[0],
            'best_heuristic_improvement': best_heuristic[1]['total_improvement'],
            'generations_tracked': len(self.generation_stats),
            'unique_heuristics': len(self.heuristic_stats),
        }
    
    def export_json(self, output_dir: Path) -> None:
        """Export tracking data to JSON files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export summary
        summary_path = output_dir / "heuristic_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        
        # Export per-heuristic stats
        heuristic_stats_path = output_dir / "heuristic_stats.json"
        stats_export = {}
        for name, stats in self.heuristic_stats.items():
            stats_export[name] = {
                k: v for k, v in stats.items()
                if k != 'generations_applied'
            }
            stats_export[name]['generations_applied'] = sorted(list(stats['generations_applied']))
        
        with open(heuristic_stats_path, 'w') as f:
            json.dump(stats_export, f, indent=2)
        
        # Export generation timeline
        timeline_path = output_dir / "generation_timeline.json"
        timeline = []
        for gen in sorted(self.generation_stats.keys()):
            gen_stats = self.generation_stats[gen]
            timeline.append({
                'generation': gen,
                'total_applications': len(gen_stats.applications),
                'successful_applications': gen_stats.successful_applications,
                'failed_applications': gen_stats.failed_applications,
                'total_improvements': gen_stats.total_improvements,
                'best_heuristic': gen_stats.best_heuristic,
                'best_improvement': gen_stats.best_improvement,
                'execution_time': gen_stats.execution_time,
                'applications': [
                    {
                        'heuristic': app.heuristic_name,
                        'category': app.category,
                        'improvement': float(app.improvement),  # Convert to native Python float
                        'success': bool(app.success),  # Convert to native Python bool
                        'time': float(app.execution_time),
                    }
                    for app in gen_stats.applications
                ]
            })
        
        with open(timeline_path, 'w') as f:
            json.dump(timeline, f, indent=2)
        
        print(f"      [!ok] heuristic_summary.json")
        print(f"      [!ok] heuristic_stats.json")
        print(f"      [!ok] generation_timeline.json")
    
    def generate_plots(self, output_dir: Path) -> None:
        """Generate visualization plots."""
        if not self.applications:
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Heuristic Performance Comparison (Bar Chart)
        self._plot_heuristic_performance(output_dir)
        
        # 2. Temporal Application Pattern (Timeline)
        self._plot_temporal_pattern(output_dir)
        
        # 3. Success Rate by Heuristic (Pie/Bar)
        self._plot_success_rates(output_dir)
        
        # 4. Improvement Distribution (Histogram)
        self._plot_improvement_distribution(output_dir)
        
        # 5. Category Performance (Grouped Bar)
        self._plot_category_performance(output_dir)
        
        print(f"      [!ok] heuristic_performance.png")
        print(f"      [!ok] temporal_pattern.png")
        print(f"      [!ok] success_rates.png")
        print(f"      [!ok] improvement_distribution.png")
        print(f"      [!ok] category_performance.png")
    
    def _plot_heuristic_performance(self, output_dir: Path) -> None:
        """Plot total improvement per heuristic."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Sort heuristics by total improvement
        sorted_heuristics = sorted(
            self.heuristic_stats.items(),
            key=lambda x: x[1]['total_improvement'],
            reverse=True
        )
        
        names = [h[0] for h in sorted_heuristics]
        improvements = [h[1]['total_improvement'] for h in sorted_heuristics]
        applications = [h[1]['total_applications'] for h in sorted_heuristics]
        
        # Plot 1: Total Improvement
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        ax1.barh(names, improvements, color=colors, alpha=0.7)
        ax1.set_xlabel('Total Improvement (Higher = Better)', fontsize=12)
        ax1.set_title('Heuristic Performance: Total Improvement', fontsize=14, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Application Count
        ax2.barh(names, applications, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Number of Applications', fontsize=12)
        ax2.set_title('Heuristic Usage Frequency', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "heuristic_performance.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_temporal_pattern(self, output_dir: Path) -> None:
        """Plot which heuristics were applied in each generation."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create matrix: generations × heuristics
        all_heuristics = sorted(self.heuristic_stats.keys())
        all_generations = sorted(self.generation_stats.keys())
        
        # Build matrix (1 = applied, 0 = not applied)
        matrix = np.zeros((len(all_generations), len(all_heuristics)))
        
        for gen_idx, gen in enumerate(all_generations):
            gen_stats = self.generation_stats[gen]
            for app in gen_stats.applications:
                heur_idx = all_heuristics.index(app.heuristic_name)
                # Color by improvement (green = positive, red = negative)
                matrix[gen_idx, heur_idx] = app.improvement
        
        # Plot heatmap
        im = ax.imshow(matrix.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Heuristic', fontsize=12)
        ax.set_title('Heuristic Application Timeline (Color = Improvement)', fontsize=14, fontweight='bold')
        
        # Set ticks
        ax.set_xticks(range(0, len(all_generations), max(1, len(all_generations) // 10)))
        ax.set_xticklabels([all_generations[i] for i in range(0, len(all_generations), max(1, len(all_generations) // 10))])
        ax.set_yticks(range(len(all_heuristics)))
        ax.set_yticklabels(all_heuristics, fontsize=8)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Improvement', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(output_dir / "temporal_pattern.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_success_rates(self, output_dir: Path) -> None:
        """Plot success rate per heuristic."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by success rate
        sorted_heuristics = sorted(
            self.heuristic_stats.items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )
        
        names = [h[0] for h in sorted_heuristics]
        success_rates = [h[1]['success_rate'] for h in sorted_heuristics]
        
        colors = ['green' if rate >= 50 else 'orange' if rate >= 25 else 'red' for rate in success_rates]
        
        ax.barh(names, success_rates, color=colors, alpha=0.7)
        ax.set_xlabel('Success Rate (%)', fontsize=12)
        ax.set_title('Heuristic Success Rates', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.axvline(x=50, color='black', linestyle='--', linewidth=0.8, label='50% threshold')
        ax.grid(axis='x', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "success_rates.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_distribution(self, output_dir: Path) -> None:
        """Plot distribution of improvements."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # All improvements
        all_improvements = [app.improvement for app in self.applications]
        axes[0].hist(all_improvements, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Improvement', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of All Applications', fontsize=12, fontweight='bold')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='No improvement')
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        
        # Only successful improvements
        successful_improvements = [app.improvement for app in self.applications if app.success]
        if successful_improvements:
            axes[1].hist(successful_improvements, bins=30, color='green', alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('Improvement', fontsize=12)
            axes[1].set_ylabel('Frequency', fontsize=12)
            axes[1].set_title('Distribution of Successful Applications', fontsize=12, fontweight='bold')
            axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "improvement_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_category_performance(self, output_dir: Path) -> None:
        """Plot performance grouped by category."""
        # Aggregate by category
        category_stats = defaultdict(lambda: {
            'total_improvement': 0.0,
            'applications': 0,
            'success_rate': 0.0,
            'successful': 0,
        })
        
        for name, stats in self.heuristic_stats.items():
            # Infer category from heuristic name or metadata
            category = self._infer_category(name)
            category_stats[category]['total_improvement'] += stats['total_improvement']
            category_stats[category]['applications'] += stats['total_applications']
            category_stats[category]['successful'] += stats['successful_applications']
        
        # Calculate success rates
        for cat, stats in category_stats.items():
            if stats['applications'] > 0:
                stats['success_rate'] = stats['successful'] / stats['applications'] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        categories = list(category_stats.keys())
        improvements = [category_stats[cat]['total_improvement'] for cat in categories]
        success_rates = [category_stats[cat]['success_rate'] for cat in categories]
        
        # Plot 1: Total Improvement by Category
        ax1.bar(categories, improvements, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Total Improvement', fontsize=12)
        ax1.set_title('Performance by Category', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Success Rate by Category
        ax2.bar(categories, success_rates, color='green', alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Success Rate (%)', fontsize=12)
        ax2.set_title('Success Rate by Category', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / "category_performance.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _infer_category(self, heuristic_name: str) -> str:
        """Infer category from heuristic name."""
        name_lower = heuristic_name.lower()
        
        if any(x in name_lower for x in ['degree', 'constrained', 'deadline']):
            return 'construction'
        elif any(x in name_lower for x in ['swap', 'shift', 'shuffle', 'reassign', 'perturbation']):
            return 'perturbation'
        elif any(x in name_lower for x in ['kempe', 'ejection', 'depth']):
            return 'improvement'
        elif any(x in name_lower for x in ['diversity', 'crowding', 'niche', 'distance']):
            return 'diversity'
        elif any(x in name_lower for x in ['neighborhood', 'iterated', 'adaptive', 'guided']):
            return 'meta'
        elif any(x in name_lower for x in ['repair', 'igls', 'lns']):
            return 'repair'
        else:
            return 'other'
