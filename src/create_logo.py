"""
Script to create custom visual asset assets/project_logo.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_logo():
    os.makedirs("assets", exist_ok=True)
    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    fig.patch.set_facecolor('#1E4620')
    ax.set_facecolor('#1E4620')
    
    # Draw leaf icon
    leaf = patches.Ellipse((0.5, 0.5), 0.6, 0.8, angle=30, color='#4CAF50', ec='#2E7D32', lw=2)
    ax.add_patch(leaf)
    
    # Draw leaf vein
    ax.plot([0.3, 0.7], [0.2, 0.8], color='#FFFFFF', lw=3)
    ax.plot([0.5, 0.65], [0.5, 0.45], color='#FFFFFF', lw=2)
    ax.plot([0.4, 0.55], [0.35, 0.3], color='#FFFFFF', lw=2)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("assets/project_logo.png", facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print("Project logo created at 'assets/project_logo.png'")

if __name__ == "__main__":
    create_logo()
