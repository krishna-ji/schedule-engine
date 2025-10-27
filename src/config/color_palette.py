"""
Color Palette Reference for Schedule Engine Plots
This file documents all color codes used in the plotting modules.
"""

# Main trend colors
COLORS = {
    "hard_constraints": "#E74C3C",  # Vibrant red
    "soft_constraints": "#27AE60",  # Emerald green
    "diversity": "#F39C12",  # Vibrant orange
}

# Extended palette for detailed hard constraint plots
HARD_CONSTRAINT_PALETTE = [
    "#E74C3C",  # Vibrant red
    "#C0392B",  # Dark red
    "#E67E22",  # Carrot orange
    "#D35400",  # Pumpkin
    "#F39C12",  # Orange
    "#F1C40F",  # Sunflower
    "#8E44AD",  # Amethyst purple
    "#9B59B6",  # Wisteria purple
    "#2C3E50",  # Midnight blue
    "#34495E",  # Wet asphalt
]

# Extended palette for detailed soft constraint plots
SOFT_CONSTRAINT_PALETTE = [
    "#27AE60",  # Nephritis green
    "#229954",  # Medium green
    "#1E8449",  # Dark green
    "#16A085",  # Green sea
    "#138D75",  # Light sea green
    "#0B5345",  # Deep green
    "#1ABC9C",  # Turquoise
    "#17A589",  # Medium turquoise
    "#48C9B0",  # Light turquoise
    "#45B39D",  # Medium aqua
]

# Statistical overlay colors (used in individual constraint plots)
STATISTICS_COLORS = {
    "final": "#E74C3C",  # Red for final value
    "maximum": "#E67E22",  # Orange for maximum
    "average": "gray",  # Gray for average
}

# Pareto front colors
PARETO_COLORS = {
    "population": "lightblue",
    "pareto_front": "#E74C3C",
    "density_map": "Blues",
    "frequency_map": "viridis",
}

if __name__ == "__main__":
    print("Schedule Engine Color Palette")
    print("=" * 50)
    print("\nMain Colors:")
    for name, color in COLORS.items():
        print(f"  {name:20s}: {color}")

    print("\nHard Constraint Palette (10 colors):")
    for i, color in enumerate(HARD_CONSTRAINT_PALETTE, 1):
        print(f"  Color {i:2d}: {color}")

    print("\nSoft Constraint Palette (10 colors):")
    for i, color in enumerate(SOFT_CONSTRAINT_PALETTE, 1):
        print(f"  Color {i:2d}: {color}")
