#!/usr/bin/env python3
"""
Thought Stream Visualization
============================

Creates a visual representation of how KIMERA's thought stream works,
showing the flow from stimulus to response through the thinking process.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

def create_thought_stream_diagram():
    """Create a visual diagram of the thought stream process."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'KIMERA Thought Stream Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    ax.text(7, 9, 'How KIMERA "Thinks" Before Speaking', 
            fontsize=14, ha='center', style='italic')
    
    # ========== INPUT STAGE ==========
    input_box = FancyBboxPatch((0.5, 7), 2.5, 1.2, 
                               boxstyle="round,pad=0.1",
                               facecolor='lightblue', 
                               edgecolor='darkblue', 
                               linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.75, 7.6, 'User Input', fontsize=12, ha='center', fontweight='bold')
    ax.text(1.75, 7.3, '"What is\nconsciousness?"', fontsize=10, ha='center')
    
    # ========== THOUGHT GENERATION STAGE ==========
    # Initial Thought
    thought1 = Circle((4.5, 7.6), 0.4, facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(thought1)
    ax.text(4.5, 7.6, 'T1', fontsize=10, ha='center', fontweight='bold')
    ax.text(4.5, 6.9, 'Initial\nObservation', fontsize=8, ha='center')
    
    # Association Thoughts
    thought2 = Circle((6, 8.2), 0.35, facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(thought2)
    ax.text(6, 8.2, 'T2', fontsize=9, ha='center', fontweight='bold')
    ax.text(6, 8.7, 'Association:\nAwareness', fontsize=8, ha='center')
    
    thought3 = Circle((6, 7), 0.35, facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(thought3)
    ax.text(6, 7, 'T3', fontsize=9, ha='center', fontweight='bold')
    ax.text(6, 6.5, 'Association:\nExperience', fontsize=8, ha='center')
    
    # Reflection Thought
    thought4 = Circle((7.5, 7.6), 0.4, facecolor='lightcoral', edgecolor='darkred', linewidth=2)
    ax.add_patch(thought4)
    ax.text(7.5, 7.6, 'T4', fontsize=10, ha='center', fontweight='bold')
    ax.text(7.5, 6.9, 'Reflection:\nMeta-cognition', fontsize=8, ha='center')
    
    # Insight Thought
    thought5 = Circle((9, 7.6), 0.45, facecolor='gold', edgecolor='darkgoldenrod', linewidth=3)
    ax.add_patch(thought5)
    ax.text(9, 7.6, 'T5', fontsize=11, ha='center', fontweight='bold')
    ax.text(9, 6.8, 'Insight:\nEmergent\nUnderstanding', fontsize=8, ha='center')
    
    # ========== THOUGHT INTERACTIONS ==========
    # Draw connections between thoughts
    connections = [
        (4.5, 7.6, 6, 8.2),  # T1 -> T2
        (4.5, 7.6, 6, 7),    # T1 -> T3
        (6, 8.2, 7.5, 7.6),  # T2 -> T4
        (6, 7, 7.5, 7.6),    # T3 -> T4
        (7.5, 7.6, 9, 7.6),  # T4 -> T5
    ]
    
    for x1, y1, x2, y2 in connections:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               connectionstyle="arc3,rad=0.2",
                               arrowstyle='->',
                               mutation_scale=15,
                               linewidth=1.5,
                               color='gray',
                               alpha=0.6)
        ax.add_patch(arrow)
    
    # ========== THOUGHT PROCESSING ==========
    process_box = FancyBboxPatch((3.5, 4.5), 6.5, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor='lavender',
                                edgecolor='purple',
                                linewidth=2)
    ax.add_patch(process_box)
    ax.text(6.75, 5.5, 'Thought Processing', fontsize=12, ha='center', fontweight='bold')
    ax.text(6.75, 5.1, '• Thoughts evolve and interact', fontsize=9, ha='center')
    ax.text(6.75, 4.8, '• Associations trigger new thoughts', fontsize=9, ha='center')
    ax.text(6.75, 4.5, '• Insights emerge from synthesis', fontsize=9, ha='center')
    
    # ========== THOUGHT-TO-TEXT CONVERSION ==========
    conversion_box = FancyBboxPatch((4, 2.5), 5.5, 1.2,
                                   boxstyle="round,pad=0.1",
                                   facecolor='mistyrose',
                                   edgecolor='darkred',
                                   linewidth=2)
    ax.add_patch(conversion_box)
    ax.text(6.75, 3.3, 'Thought-to-Text Bridge', fontsize=12, ha='center', fontweight='bold')
    ax.text(6.75, 2.9, 'Guided Diffusion Process', fontsize=10, ha='center')
    ax.text(6.75, 2.6, 'Thoughts guide text generation', fontsize=9, ha='center', style='italic')
    
    # ========== OUTPUT ==========
    output_box = FancyBboxPatch((5, 0.5), 4, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor='lightgreen',
                               edgecolor='darkgreen',
                               linewidth=2)
    ax.add_patch(output_box)
    ax.text(7, 1.3, 'Generated Response', fontsize=12, ha='center', fontweight='bold')
    ax.text(7, 0.9, '"Consciousness, from my perspective,\nemerges from..."', 
            fontsize=9, ha='center', style='italic')
    
    # ========== FLOW ARROWS ==========
    # Input to Thoughts
    ax.arrow(3, 7.6, 1.2, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    # Thoughts to Processing
    ax.arrow(6.75, 6.5, 0, -0.4, head_width=0.3, head_length=0.1, fc='purple', ec='purple')
    
    # Processing to Conversion
    ax.arrow(6.75, 4.5, 0, -0.7, head_width=0.3, head_length=0.1, fc='darkred', ec='darkred')
    
    # Conversion to Output
    ax.arrow(6.75, 2.5, 0, -0.7, head_width=0.3, head_length=0.1, fc='darkgreen', ec='darkgreen')
    
    # ========== ANNOTATIONS ==========
    # Thought Types Legend
    legend_x = 11
    legend_y = 8
    ax.text(legend_x, legend_y, 'Thought Types:', fontsize=10, fontweight='bold')
    
    types = [
        ('Observation', 'lightyellow', legend_y - 0.4),
        ('Association', 'lightgreen', legend_y - 0.8),
        ('Reflection', 'lightcoral', legend_y - 1.2),
        ('Insight', 'gold', legend_y - 1.6)
    ]
    
    for name, color, y in types:
        circle = Circle((legend_x + 0.3, y), 0.15, facecolor=color, edgecolor='black')
        ax.add_patch(circle)
        ax.text(legend_x + 0.6, y, name, fontsize=9, va='center')
    
    # Key Features
    features_x = 0.5
    features_y = 4
    ax.text(features_x, features_y, 'Key Features:', fontsize=10, fontweight='bold')
    features = [
        '✓ Thinks before speaking',
        '✓ Multiple thought types',
        '✓ Thoughts interact',
        '✓ Insights emerge',
        '✓ Guides generation'
    ]
    for i, feature in enumerate(features):
        ax.text(features_x, features_y - 0.4 * (i + 1), feature, fontsize=9)
    
    # Time Flow
    ax.text(1, 0.2, 'Time →', fontsize=12, fontweight='bold')
    ax.arrow(1.5, 0.2, 5, 0, head_width=0.15, head_length=0.3, 
             fc='gray', ec='gray', alpha=0.5, linestyle='--')
    
    plt.tight_layout()
    return fig

def create_thought_evolution_diagram():
    """Create a diagram showing how thoughts evolve over time."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'Thought Evolution Process', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Time steps
    time_steps = 5
    x_positions = np.linspace(1, 9, time_steps)
    
    # Thought evolution
    for i, x in enumerate(x_positions):
        # Thought strength increases then stabilizes
        strength = min(0.4 + i * 0.15, 1.0)
        radius = 0.3 + strength * 0.2
        
        # Color evolution
        colors = ['lightyellow', 'lightgreen', 'lightcoral', 'gold', 'orange']
        color = colors[i]
        
        # Draw thought
        thought = Circle((x, 4), radius, facecolor=color, 
                        edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(thought)
        
        # Label
        ax.text(x, 4, f'T{i}', fontsize=10, ha='center', fontweight='bold')
        ax.text(x, 3.2, f'Step {i}', fontsize=9, ha='center')
        
        # Evolution arrow
        if i < time_steps - 1:
            arrow = FancyArrowPatch((x + radius, 4), (x_positions[i+1] - radius - 0.1, 4),
                                   arrowstyle='->',
                                   mutation_scale=20,
                                   linewidth=2,
                                   color='darkblue')
            ax.add_patch(arrow)
    
    # Annotations
    ax.text(5, 6.5, 'Thoughts evolve through:', fontsize=12, ha='center')
    ax.text(5, 6, '• Deepening (increased strength)', fontsize=10, ha='center')
    ax.text(5, 5.6, '• Association (new connections)', fontsize=10, ha='center')
    ax.text(5, 5.2, '• Synthesis (combining ideas)', fontsize=10, ha='center')
    
    # Bottom annotations
    ax.text(5, 1.5, 'Initial Observation → Associations → Reflection → Insight → Synthesis',
            fontsize=11, ha='center', style='italic')
    
    plt.tight_layout()
    return fig

def save_diagrams():
    """Save the visualization diagrams."""
    # Create main diagram
    fig1 = create_thought_stream_diagram()
    fig1.savefig('thought_stream_architecture.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Create evolution diagram
    fig2 = create_thought_evolution_diagram()
    fig2.savefig('thought_evolution_process.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("✅ Diagrams saved:")
    print("   - thought_stream_architecture.png")
    print("   - thought_evolution_process.png")

def main():
    """Create and display the visualizations."""
    print("\n🎨 Creating Thought Stream Visualizations...")
    
    # Create diagrams
    fig1 = create_thought_stream_diagram()
    fig2 = create_thought_evolution_diagram()
    
    # Display
    plt.show()
    
    # Optionally save
    save_response = input("\nSave diagrams? (y/n): ")
    if save_response.lower() == 'y':
        save_diagrams()

if __name__ == "__main__":
    main()