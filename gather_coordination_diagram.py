#!/usr/bin/env python3

"""
Diagram showing how gather operation coordinates distributed data access in cupynumeric.
This illustrates what happens when you access A[i] on a distributed deferred array.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, Arrow
import matplotlib.patches as mpatches

def create_gather_diagram():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    # Colors
    gpu0_color = '#FF6B6B'  # Red
    gpu1_color = '#4ECDC4'  # Teal
    comm_color = '#FFE66D'  # Yellow
    result_color = '#95E1D3' # Light green
    
    ############################################################################
    # Step 1: Original Distributed Array
    ############################################################################
    ax1.set_title("Step 1: Original Distributed Array A (size=10)", fontsize=14, fontweight='bold')
    
    # GPU 0 holds elements 0-4
    gpu0_rect = FancyBboxPatch((0, 0.3), 5, 0.4, boxstyle="round,pad=0.05", 
                               facecolor=gpu0_color, edgecolor='black', linewidth=2)
    ax1.add_patch(gpu0_rect)
    ax1.text(2.5, 0.5, 'GPU 0\nElements [0,1,2,3,4]', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # GPU 1 holds elements 5-9  
    gpu1_rect = FancyBboxPatch((5, 0.3), 5, 0.4, boxstyle="round,pad=0.05",
                               facecolor=gpu1_color, edgecolor='black', linewidth=2)
    ax1.add_patch(gpu1_rect)
    ax1.text(7.5, 0.5, 'GPU 1\nElements [5,6,7,8,9]', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Array indices
    for i in range(10):
        ax1.text(i+0.5, 0.1, str(i), ha='center', va='center', fontsize=10)
        ax1.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
    
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Array Index')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ############################################################################
    # Step 2: Access A[[8,2,6,1]] - Gather Operation
    ############################################################################  
    ax2.set_title("Step 2: Access A[[8,2,6,1]] - Distributed Gather Coordination", fontsize=14, fontweight='bold')
    
    # Show the request indices
    indices = [8, 2, 6, 1]
    ax2.text(5, 0.9, f'Request: A[{indices}]', ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    
    # GPU 0 data
    gpu0_rect2 = FancyBboxPatch((0, 0.6), 5, 0.2, boxstyle="round,pad=0.02",
                                facecolor=gpu0_color, edgecolor='black', linewidth=1)
    ax2.add_patch(gpu0_rect2)
    ax2.text(2.5, 0.7, 'GPU 0: [0,1,2,3,4]', ha='center', va='center', fontsize=10)
    
    # GPU 1 data  
    gpu1_rect2 = FancyBboxPatch((5, 0.6), 5, 0.2, boxstyle="round,pad=0.02",
                                facecolor=gpu1_color, edgecolor='black', linewidth=1)
    ax2.add_patch(gpu1_rect2)
    ax2.text(7.5, 0.7, 'GPU 1: [5,6,7,8,9]', ha='center', va='center', fontsize=10)
    
    # Communication arrows showing data requests
    # Index 8 from GPU 1
    ax2.annotate('', xy=(8, 0.6), xytext=(8, 0.4), 
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax2.text(8, 0.45, '8→GPU1', ha='center', va='center', fontsize=9, color='red', fontweight='bold')
    
    # Index 2 from GPU 0
    ax2.annotate('', xy=(2, 0.6), xytext=(2, 0.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue')) 
    ax2.text(2, 0.45, '2→GPU0', ha='center', va='center', fontsize=9, color='blue', fontweight='bold')
    
    # Index 6 from GPU 1
    ax2.annotate('', xy=(6, 0.6), xytext=(6, 0.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax2.text(6, 0.45, '6→GPU1', ha='center', va='center', fontsize=9, color='red', fontweight='bold')
    
    # Index 1 from GPU 0  
    ax2.annotate('', xy=(1, 0.6), xytext=(1, 0.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax2.text(1, 0.45, '1→GPU0', ha='center', va='center', fontsize=9, color='blue', fontweight='bold')
    
    # Communication coordination
    comm_rect = FancyBboxPatch((3, 0.1), 4, 0.2, boxstyle="round,pad=0.02",
                               facecolor=comm_color, edgecolor='black', linewidth=1)
    ax2.add_patch(comm_rect)
    ax2.text(5, 0.2, 'Legate Runtime Coordination\n(NCCL/MPI Communication)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlim(-0.5, 10.5)
    ax2.set_ylim(0, 1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    ############################################################################
    # Step 3: Result - Gathered Data
    ############################################################################
    ax3.set_title("Step 3: Result - Gathered Data on Each GPU", fontsize=14, fontweight='bold')
    
    # Result array [8,2,6,1] distributed across GPUs
    result_rect = FancyBboxPatch((2, 0.4), 6, 0.3, boxstyle="round,pad=0.05",
                                 facecolor=result_color, edgecolor='black', linewidth=2)
    ax3.add_patch(result_rect)
    
    # Show the gathered values
    gathered_values = [8, 2, 6, 1]
    for i, val in enumerate(gathered_values):
        ax3.text(2.75 + i*1.5, 0.55, str(val), ha='center', va='center', fontsize=14, fontweight='bold')
        if i < len(gathered_values)-1:
            ax3.axvline(x=2.75 + (i+0.5)*1.5, color='gray', linestyle='-', alpha=0.7)
    
    ax3.text(5, 0.55, 'Result: [8, 2, 6, 1]', ha='center', va='center', fontsize=12, fontweight='bold')
    ax3.text(5, 0.2, 'New distributed array with gathered elements\n(May be distributed differently)', 
             ha='center', va='center', fontsize=11, style='italic')
    
    ax3.set_xlim(-0.5, 10.5)
    ax3.set_ylim(0, 1)
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # Add legend
    gpu0_patch = mpatches.Patch(color=gpu0_color, label='GPU 0 Data')
    gpu1_patch = mpatches.Patch(color=gpu1_color, label='GPU 1 Data') 
    comm_patch = mpatches.Patch(color=comm_color, label='Communication Layer')
    result_patch = mpatches.Patch(color=result_color, label='Gathered Result')
    
    fig.legend(handles=[gpu0_patch, gpu1_patch, comm_patch, result_patch], 
               loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig('gather_coordination_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_gather_diagram()
    print("Diagram saved as 'gather_coordination_diagram.png'")
    print("\nKey Points:")
    print("1. Original array is distributed across GPUs")
    print("2. Advanced indexing A[indices] triggers gather operation")
    print("3. Legate runtime coordinates cross-GPU data access")
    print("4. Result is a new distributed array with gathered elements") 