# src/visualization.py
"""
Visualization functions
"""
import matplotlib
import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from src.utils import *
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Set Chinese font to Heiti
matplotlib.rcParams['axes.unicode_minus'] = False    # Correctly display negative signs

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")

def get_interval_array(melody):
    """Get the interval array of the melody (ignoring rests)"""
    intervals = []
    notes = melody.notes
    for i in range(len(notes) - 1):
        current_note = notes[i]
        next_note = notes[i + 1]
        # Skip rests
        if current_note > 0 and next_note > 0:
            interval = next_note - current_note
            intervals.append(interval)
    
    return intervals

def extract_all_melody_features(population):
    """Extract feature vectors of all melodies"""
    feature_list = []
    for i, individual in enumerate(population.individuals):
        melody = individual.melody
        melody_no_zero = [note for note in melody.notes if note > 0]# Remove rests
        average_pitch = np.mean(melody_no_zero)
        std_pitch = np.std(melody_no_zero, ddof=0)
        intervals = get_interval_array(melody)
        average_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        changes_count = 0
        if len(melody.notes) >= 3:  # At least three contour points for changes
            for j in range(2, len(melody.notes)):
                prev2, prev1, curr = melody.notes[j-2], melody.notes[j-1], melody.notes[j]
                dir1 = 1 if prev1 > prev2 else (-1 if prev1 < prev2 else 0)
                dir2 = 1 if curr > prev1 else (-1 if curr < prev1 else 0)
                change = 1 if dir1 != dir2 else 0
                changes_count += change
        features = [
            average_pitch/20.,
            std_pitch/8.0,
            average_interval/10.0,
            std_interval/6.0,
            changes_count/10
        ]# Normalization
        feature_list.append(features)
    return feature_list

def plot_fitness_progress(fitness_history):
    """Plot fitness progress graph"""
    plt.figure(figsize=(10, 6))
    
    generations = range(len(fitness_history))
    
    plt.plot(generations, fitness_history, 'b-', linewidth=2, label='Average Fitness')
    
    # Add trend line
    if len(fitness_history) > 1:
        z = np.polyfit(generations, fitness_history, 3)
        p = np.poly1d(z)
        plt.plot(generations, p(generations), "r--", alpha=0.5, label='Trend Line')
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution Process')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save image
    plt.savefig(f'data/outputs/{current_time}/fitness_progress.png', dpi=150)
    plt.show()

def plot_population_diversity(population_history):
    """Plot population diversity graph (simplified version)"""
    # Here we can calculate the differences between melodies in each generation's population
    # Extract feature vectors of all melody encodings [average pitch, pitch standard deviation, average interval, interval standard deviation, contour change rate], calculate distance from mean
    avg_distances = []
    for gen in range(0, 30):
        population = population_history[gen]
        feature_list = extract_all_melody_features(population)
        X = np.array(feature_list)
        n = X.shape[0]
        differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        squared_diffs = differences ** 2
        distances = np.sqrt(np.sum(squared_diffs, axis=2))
    
        # Take upper triangle average (excluding diagonal)
        avg_distance = np.mean(distances[np.triu_indices(n, k=1)])
        avg_distances.append(avg_distance)

    plt.figure(figsize=(10, 6))
    generations = range(0,10*len(avg_distances),10)
    # Plot main curve
    plt.plot(generations, avg_distances, 'b-', linewidth=2, label='Population Diversity')
    
    # Add trend line
    if len(avg_distances) > 1:
        z = np.polyfit(generations, avg_distances, 3)
        p = np.poly1d(z)
        plt.plot(generations, p(generations), "r--", alpha=0.5, label='Trend Line')
    
    plt.xlabel('Generation')
    plt.ylabel('Average Euclidean Distance of Feature Vectors')
    plt.title('Population Diversity Evolution Process')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_melody(melody, title=None):
    """Visualize melody (piano roll)"""
    
    from src.config import NOTES
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), 
                           gridspec_kw={'height_ratios': [3, 2, 2]})
    notes = melody.notes
    pitches = np.array(notes)
    time_points = np.arange(len(pitches))
    
    # 1. Main plot: Piano roll 
    ax1 = axes[0]
    # Analyze note sequence
    notes = []  # (start position, pitch, duration, contains sustain)
    i = 0
    note_counter = 1  # Note counter for labeling
    
    while i < len(pitches):
        if pitches[i] == 0:  # Rest
            i += 1
        elif pitches[i] == 28:  # Sustain
            # Find previous normal note
            j = i - 1
            while j >= 0 and pitches[j] in [0, 28]:
                j -= 1
            
            if j >= 0:
                # Update duration of previous note
                for idx, (start, pitch, dur, has_sustain) in enumerate(notes):
                    if start == j:
                        notes[idx] = (start, pitch, dur + 1, True)
                        break
            i += 1
        else:  # Normal note
            start = i
            pitch_val = pitches[i]
            duration = 1
            has_sustain = False
            i += 1
            
            # Check for immediate sustain
            while i < len(pitches) and pitches[i] == 28:
                duration += 1
                has_sustain = True
                i += 1
                
            notes.append((start, pitch_val, duration, has_sustain))
    
    # Plot piano roll
    for note_idx, (start, pitch, duration, has_sustain) in enumerate(notes):
        # Note body - draw a continuous rectangle without gaps
        rect_width = duration  # Width equals duration
        rect_height = 0.8  # Note height
        
        # Draw entire note rectangle (from start to end position)
        rect = plt.Rectangle((start, pitch - rect_height/2), 
                            rect_width, rect_height,
                            facecolor='skyblue', edgecolor='navy',
                            linewidth=2, alpha=1.0, zorder=3)
        ax1.add_patch(rect)
        
        # If note contains sustain, add special marker in sustain part
        if has_sustain and duration > 1:
            # Add diagonal pattern on entire rectangle
            for d in range(1, duration):
                # Add slash marker in sustain part
                sustain_rect = plt.Rectangle((start + d - 0.1, pitch - rect_height/2), 
                                           0.2, rect_height,
                                           facecolor='#FF9999', edgecolor='red',
                                           linewidth=1, alpha=0.7, zorder=4)
                ax1.add_patch(sustain_rect)
        
        # Mark note start position (on first beat)
        start_marker = plt.Rectangle((start, pitch - rect_height/2), 
                                   0.1, rect_height,
                                   facecolor='darkblue', edgecolor='darkblue',
                                   linewidth=1, alpha=1.0, zorder=5)
        ax1.add_patch(start_marker)
        
        # Label note number
        ax1.text(start + 0.5, pitch + 1.5, f'N{note_counter}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        ax1.text(start + duration/2, pitch - 1.5, f'{NOTES[pitch - 1]}', 
                    ha='center', va='top', fontsize=8, fontstyle='italic')
        
        # Label duration
        ax1.text(start + duration/2, pitch - 3, f'{duration} beats',
                ha='center', va='top', fontsize=8, fontstyle='italic')
        
        note_counter += 1
    
    # Plot rests
    rest_positions = []
    for i in range(len(pitches)):
        if pitches[i] == 0:
            # Draw rest rectangle (gray semi-transparent)
            rest_rect = plt.Rectangle((i, -2 - 0.4), 1, 0.8,
                                     facecolor='gray', edgecolor='darkgray',
                                     linewidth=1, alpha=0.3, zorder=2)
            ax1.add_patch(rest_rect)
            
            # Rest marker
            ax1.scatter(i + 0.5, -2, color='darkgray', marker='$\u266B$',
                       s=150, linewidths=2, zorder=3)
            rest_positions.append(i)
    
    # Set axis range
    all_pitches = [p for p in pitches if p not in [0, 28]]
    if all_pitches:
        min_pitch = min(all_pitches) - 8
        max_pitch = max(all_pitches) + 8
        ax1.set_ylim(min_pitch, max_pitch)
    else:
        ax1.set_ylim(-5, 35)
    
    ax1.set_xlim(-0.5, len(pitches) - 0.5)
    ax1.set_xlabel('Time Position (Beats)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pitch Value', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title} - Piano Roll', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add time grid lines
    for x in range(0, len(pitches) + 1):
        ax1.axvline(x=x, color='gray', linestyle=':', alpha=0.2, linewidth=0.5)
    
    # 2. Melody contour plot (unchanged)
    ax2 = axes[1]
    
    # Create smooth contour line (considering sustains)
    contour_pitches = np.full(len(pitches), np.nan)
    current_pitch = np.nan
    
    for i in range(len(pitches)):
        if pitches[i] == 0:
            contour_pitches[i] = np.nan
            current_pitch = np.nan
        elif pitches[i] == 28:
            if not np.isnan(current_pitch):
                contour_pitches[i] = current_pitch
        else:
            contour_pitches[i] = pitches[i]
            current_pitch = pitches[i]
    
    # Plot step plot for better duration display
    ax2.step(time_points, contour_pitches, where='post', 
            color='blue', linewidth=3, alpha=0.8, label='Melody Contour')
    
    # Fill area
    valid_indices = ~np.isnan(contour_pitches)
    if np.any(valid_indices):
        ax2.fill_between(time_points, 0, contour_pitches, 
                        where=valid_indices, alpha=0.2, color='blue')
    
    # Mark note starts
    for i in range(len(pitches)):
        if pitches[i] > 0 and pitches[i] != 28:
            ax2.scatter(i, pitches[i], color='darkblue', s=100, 
                       zorder=5, edgecolors='white', linewidths=2)
    
    # Mark rest positions
    for rest_pos in rest_positions:
        ax2.axvspan(rest_pos, rest_pos + 1, 
                   alpha=0.2, color='gray', zorder=1)
    
    ax2.set_xlabel('Time Position (Beats)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Pitch Value', fontsize=12, fontweight='bold')
    ax2.set_title('Melody Contour (Step Plot)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(-0.5, len(pitches) - 0.5)
    ax2.legend(loc='upper right')
    
    # 3. Note duration distribution plot (unchanged)
    ax3 = axes[2]
    
    if notes:
        durations = [note[2] for note in notes]
        note_pitches = [note[1] for note in notes]
        sustain_flags = [note[3] for note in notes]
        
        # Create color array (notes with sustain use different color)
        colors = ['#FF9999' if sustain else 'skyblue' for sustain in sustain_flags]
        
        # Plot duration bar chart
        x_positions = np.arange(len(notes))
        bars = ax3.bar(x_positions, durations, 
                      color=colors, edgecolor='black', alpha=0.8)
        
        # Add pitch labels
        for i, (bar, pitch) in enumerate(zip(bars, note_pitches)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                    f'{pitch}', ha='center', va='bottom', fontsize=9)
            
            
            ax3.text(bar.get_x() + bar.get_width()/2, -0.5,
                        f'{NOTES[pitch - 1]}', ha='center', va='top', fontsize=8,
                        rotation=45 if len(notes) > 8 else 0)
        
        # Add sustain markers
        for i, sustain in enumerate(sustain_flags):
            if sustain:
                ax3.text(i, durations[i] + 0.2, 'Sustain', 
                        ha='center', va='bottom', fontsize=8, color='red')
        
        ax3.set_xlabel('Note Number', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Duration (Beats)', fontsize=12, fontweight='bold')
        ax3.set_title('Note Duration Distribution', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_positions)
        ax3.set_xticklabels([f'#{i+1}' for i in range(len(notes))])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add duration statistics
        if durations:
            avg_duration = np.mean(durations)
            
            ax3.axhline(y=avg_duration, color='green', linestyle='--', 
                       alpha=0.5, label=f'Avg Duration: {avg_duration:.1f} beats')
            ax3.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def create_evolution_report(ga_instance, output_dir):
    """Generate evolution process report"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create report file
    report_file = os.path.join(output_dir, 'evolution_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("Genetic Algorithm Music Composition Evolution Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Generations: {ga_instance.generation}\n")
        f.write(f"Final Average Fitness: {ga_instance.population.get_average_fitness():.3f}\n")
        f.write(f"Final Best Fitness: {ga_instance.population.get_fittest().fitness:.3f}\n\n")
        
        f.write("Fitness History:\n")
        for gen, fitness in enumerate(ga_instance.fitness_history):
            f.write(f"  Generation {gen}: {fitness:.3f}\n")
        
        f.write("\nBest Melody Information:\n")
        best_melody = ga_instance.population.get_fittest().melody
        f.write(best_melody.to_string())
    
    print(f"Report saved to: {report_file}")

def compare_melodies(melody_1, melody_2, name1="Melody 1", name2="Melody 2", 
                                           title="Two Melodies Comparison - Piano Roll", save_path=None):
    """
    Side-by-side visualization of two melody arrays with pitch name labels and extended x-axis
    
    Parameters:
        melody1: First melody array, length 32, 0 for rest, 28 for sustain
        melody2: Second melody array, length 32, 0 for rest, 28 for sustain
        name1: Name of the first melody
        name2: Name of the second melody
        title: Chart title
        save_path: Image save path
    """
    melody1 = melody_1.notes
    melody2 = melody_2.notes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Set unified colormap for both subplots
    cmap = plt.cm.get_cmap('viridis')
    
    # Process first melody
    pitches1 = np.array(melody1)
    notes1 = []
    max_end1 = 0  # Record maximum end position of first melody
    
    i = 0
    while i < len(pitches1):
        if pitches1[i] == 0:
            i += 1
        elif pitches1[i] == 28:
            j = i - 1
            while j >= 0 and pitches1[j] in [0, 28]:
                j -= 1
            if j >= 0 and notes1:
                for note_idx in range(len(notes1)-1, -1, -1):
                    if notes1[note_idx][0] <= j <= notes1[note_idx][0] + notes1[note_idx][2] - 1:
                        notes1[note_idx] = (notes1[note_idx][0], notes1[note_idx][1], notes1[note_idx][2] + 1)
                        max_end1 = max(max_end1, notes1[note_idx][0] + notes1[note_idx][2])
                        break
            i += 1
        else:
            start = i
            pitch_val = pitches1[i]
            duration = 1
            i += 1
            while i < len(pitches1) and pitches1[i] == 28:
                duration += 1
                i += 1
            notes1.append((start, pitch_val, duration))
            max_end1 = max(max_end1, start + duration)
    
    # Process second melody
    pitches2 = np.array(melody2)
    notes2 = []
    max_end2 = 0  # Record maximum end position of second melody
    
    i = 0
    while i < len(pitches2):
        if pitches2[i] == 0:
            i += 1
        elif pitches2[i] == 28:
            j = i - 1
            while j >= 0 and pitches2[j] in [0, 28]:
                j -= 1
            if j >= 0 and notes2:
                for note_idx in range(len(notes2)-1, -1, -1):
                    if notes2[note_idx][0] <= j <= notes2[note_idx][0] + notes2[note_idx][2] - 1:
                        notes2[note_idx] = (notes2[note_idx][0], notes2[note_idx][1], notes2[note_idx][2] + 1)
                        max_end2 = max(max_end2, notes2[note_idx][0] + notes2[note_idx][2])
                        break
            i += 1
        else:
            start = i
            pitch_val = pitches2[i]
            duration = 1
            i += 1
            while i < len(pitches2) and pitches2[i] == 28:
                duration += 1
                i += 1
            notes2.append((start, pitch_val, duration))
            max_end2 = max(max_end2, start + duration)
    
    # Calculate maximum x-axis range to ensure all notes are fully displayed
    x_max1 = max(31.5, max_end1 + 1.0)  # At least 31.5, extend if notes are longer
    x_max2 = max(31.5, max_end2 + 1.0)
    
    # Plot first melody
    for start, pitch, duration in notes1:
        color = cmap((pitch - 1) / 26)
        rect = Rectangle((start, pitch - 0.4), duration, 0.8,
                        facecolor=color, edgecolor='black',
                        linewidth=1.5, alpha=0.8)
        ax1.add_patch(rect)
        
        # Label pitch name on note
        ax1.text(start + duration/2, pitch, NOTES[pitch - 1],
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white' if pitch < 14 else 'black')
    
    # Mark rests
    for i in range(len(pitches1)):
        if pitches1[i] == 0:
            ax1.scatter(i + 0.5, 14, color='gray', marker='x', 
                       s=80, linewidths=2.5, zorder=3, alpha=0.8)
    
    # Plot second melody
    for start, pitch, duration in notes2:
        color = cmap((pitch - 1) / 26)
        rect = Rectangle((start, pitch - 0.4), duration, 0.8,
                        facecolor=color, edgecolor='black',
                        linewidth=1.5, alpha=0.8)
        ax2.add_patch(rect)
        
        # Label pitch name on note
        ax2.text(start + duration/2, pitch, NOTES[pitch - 1],
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white' if pitch < 14 else 'black')
    
    # Set first subplot properties
    ax1.set_xlabel('Time Position', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pitch Value', fontsize=12, fontweight='bold')
    ax1.set_title(f'{name1}', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlim(-1.5, x_max1)  # Extend x-axis range
    ax1.set_ylim(0.5, 27.5)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Set second subplot properties
    ax2.set_xlabel('Time Position', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Pitch Value', fontsize=12, fontweight='bold')
    ax2.set_title(f'{name2}', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlim(-1.5, x_max2)  # Extend x-axis range
    ax2.set_ylim(0.5, 27.5)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add vertical reference lines (measure lines) - extend to maximum range
    max_x_ticks = max(x_max1, x_max2)
    for x in range(0, int(max_x_ticks) + 4, 4):
        ax1.axvline(x=x-0.5, color='blue', alpha=0.2, linestyle='--', linewidth=1)
        ax2.axvline(x=x-0.5, color='blue', alpha=0.2, linestyle='--', linewidth=1)
    
    # Add all pitch level labels on y-axis (1-27)
    y_ticks = list(range(1, 28))
    y_tick_labels = [str(i) for i in range(1, 28)]
    
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_tick_labels, fontsize=10)
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_tick_labels, fontsize=10)
    
    # Add x-axis ticks (mark every 2 positions)
    ax1.set_xticks(range(0, int(x_max1) + 1, 2))
    ax2.set_xticks(range(0, int(x_max2) + 1, 2))
    
    # Add statistics text
    stats1 = f"Notes: {len(notes1)}  Rests: {np.sum(pitches1 == 0)}  Sustains: {np.sum(pitches1 == 28)}"
    stats2 = f"Notes: {len(notes2)}  Rests: {np.sum(pitches2 == 0)}  Sustains: {np.sum(pitches2 == 28)}"
    
    ax1.text(0.02, 0.97, stats1, transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax2.text(0.02, 0.97, stats2, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='skyblue', edgecolor='black', alpha=0.8, label='Note'),
        plt.Line2D([0], [0], marker='x', color='gray', markersize=10, 
                  label='Rest', linewidth=0)
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Set main title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Add color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=27))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label('Pitch Value', fontsize=11)
    cbar.set_ticks(range(1, 28, 2))
    cbar.set_ticklabels([str(i) for i in range(1, 28, 2)])
    
    # Save image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()