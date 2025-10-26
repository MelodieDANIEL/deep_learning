#!/usr/bin/env python3
"""
Script pour générer une visualisation du concept de convolution avec/sans padding.
Illustre pourquoi on perd des pixels sans padding.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_convolution_visualization():
    """Crée une visualisation complète de la convolution avec et sans padding."""
    
    # Créer une figure avec 2 sous-graphiques
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ========== GAUCHE : SANS PADDING ==========
    ax1 = axes[0]
    ax1.set_xlim(-0.5, 5.5)
    ax1.set_ylim(-0.5, 7.5)  # Même hauteur totale que la figure de droite
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    ax1.set_title('Convolution SANS padding (padding=0)\nFiltre 3×3 sur image 5×5', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.axis('off')
    
    # Dessiner la grille 5×5
    for i in range(5):
        for j in range(5):
            rect = patches.Rectangle((j, i), 1, 1, linewidth=2, 
                                     edgecolor='black', facecolor='lightblue', alpha=0.3)
            ax1.add_patch(rect)
            ax1.text(j + 0.5, i + 0.5, f'{i*5+j+1}', 
                    ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Marquer les positions impossibles (bords)
    impossible_positions = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),  # haut
        (0, 4), (1, 4), (2, 4), (3, 4), (4, 4),  # bas
        (0, 1), (0, 2), (0, 3),  # gauche
        (4, 1), (4, 2), (4, 3),  # droite
    ]
    
    for (x, y) in impossible_positions:
        rect = patches.Rectangle((x, y), 1, 1, linewidth=3, 
                                edgecolor='red', facecolor='red', alpha=0.2)
        ax1.add_patch(rect)
        # Ajouter une croix rouge
        ax1.plot([x + 0.2, x + 0.8], [y + 0.2, y + 0.8], 'r-', linewidth=3)
        ax1.plot([x + 0.2, x + 0.8], [y + 0.8, y + 0.2], 'r-', linewidth=3)
    
    # Marquer les positions valides (centre)
    valid_positions = [
        (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3), (3, 3),
    ]
    
    for (x, y) in valid_positions:
        circle = patches.Circle((x + 0.5, y + 0.5), 0.15, 
                               color='green', alpha=0.7, zorder=10)
        ax1.add_patch(circle)
    
    # Dessiner un exemple de filtre 3×3 centré sur le pixel central (2,2) - pixel numéro 13
    # Le filtre couvre de (1,1) à (3,3) pour être centré sur (2,2)
    filter_rect = patches.Rectangle((1, 1), 3, 3,
                                    linewidth=4, edgecolor='blue', 
                                    facecolor='none', linestyle='--', zorder=5)
    ax1.add_patch(filter_rect)
    ax1.text(2.5, -0.3, 'Filtre 3×3 centré sur 13', ha='center', fontsize=11, 
            color='blue', fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7, edgecolor='blue'))
    
    # Ajouter la légende EN DESSOUS
    ax1.text(2.5, 5.5, '✗ Rouge : positions impossibles (filtre déborde)', 
            ha='center', fontsize=11, color='darkred', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7, edgecolor='red'))
    ax1.text(2.5, 6.2, '✓ Vert : positions valides (9 positions)', 
            ha='center', fontsize=11, color='darkgreen', fontweight='bold')
    ax1.text(2.5, 6.8, 'Résultat : 5×5 → 3×3 (perte de 2x2 pixels)', 
            ha='center', fontsize=12, fontweight='bold')
    
    # ========== DROITE : AVEC PADDING ==========
    ax2 = axes[1]
    ax2.set_xlim(-1.5, 6.5)
    ax2.set_ylim(-1.5, 7.5)  # Même hauteur totale que la figure de gauche
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    ax2.set_title('Convolution AVEC padding (padding=1)\nFiltre 3×3 sur image 5×5 + padding', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.axis('off')
    
    # Dessiner le padding (zeros)
    for i in range(-1, 6):
        for j in range(-1, 6):
            if i == -1 or i == 5 or j == -1 or j == 5:
                rect = patches.Rectangle((j, i), 1, 1, linewidth=2,
                                        edgecolor='gray', facecolor='lightyellow', alpha=0.3)
                ax2.add_patch(rect)
                ax2.text(j + 0.5, i + 0.5, '0', 
                        ha='center', va='center', fontsize=9, color='gray', style='italic')
    
    # Dessiner l'image originale 5×5
    for i in range(5):
        for j in range(5):
            rect = patches.Rectangle((j, i), 1, 1, linewidth=2,
                                    edgecolor='black', facecolor='lightblue', alpha=0.3)
            ax2.add_patch(rect)
            ax2.text(j + 0.5, i + 0.5, f'{i*5+j+1}', 
                    ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Toutes les positions sont maintenant valides !
    for i in range(5):
        for j in range(5):
            circle = patches.Circle((j + 0.5, i + 0.5), 0.15, 
                                   color='green', alpha=0.7, zorder=10)
            ax2.add_patch(circle)
    
    # Dessiner un exemple de filtre 3×3 centré sur le pixel (0,0) - coin supérieur gauche
    # Le filtre couvre de (-1,-1) à (1,1) pour être centré sur (0,0)
    filter_rect = patches.Rectangle((-1, -1), 3, 3,
                                    linewidth=4, edgecolor='blue', 
                                    facecolor='none', linestyle='--', zorder=5)
    ax2.add_patch(filter_rect)
    ax2.text(2.5, -1.3, 'Filtre 3×3 centré sur 1 ', ha='center', fontsize=11, 
            color='blue', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7, edgecolor='blue'))
    
    # Ajouter la légende EN DESSOUS de l'image
    ax2.text(2.5, 5.5, '□ Jaune : padding (zéros ajoutés)', 
            ha='center', fontsize=11, color='darkorange', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7, edgecolor='orange'))
    ax2.text(2.5, 6.2, '✓ Vert : toutes les positions sont valides (25 positions)', 
            ha='center', fontsize=11, color='darkgreen', fontweight='bold')
    ax2.text(2.5, 6.8, 'Résultat : 5×5 → 5×5 (taille préservée !)', 
            ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_formula_visualization():
    """Crée une visualisation de la formule de calcul de taille."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Titre
    ax.text(5, 9, 'Formule de calcul de la taille de sortie', 
           ha='center', fontsize=18, fontweight='bold')
    
    # Formule principale
    formula_text = r'$H_{out} = \left\lfloor \frac{H_{in} + 2 \times padding - kernel\_size}{stride} \right\rfloor + 1$'
    ax.text(5, 7.5, formula_text, ha='center', fontsize=20, 
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Exemples
    examples = [
        {
            'title': 'Sans padding (padding=0)',
            'params': 'H_in=224, kernel=3, padding=0, stride=1',
            'calc': r'$\left\lfloor \frac{224 + 0 - 3}{1} \right\rfloor + 1 = 222$',
            'result': '224×224 → 222×222 (perte de 2 pixels)',
            'color': 'lightcoral',
            'y': 5.5
        },
        {
            'title': 'Avec padding (padding=1)',
            'params': 'H_in=224, kernel=3, padding=1, stride=1',
            'calc': r'$\left\lfloor \frac{224 + 2 - 3}{1} \right\rfloor + 1 = 224$',
            'result': '224×224 → 224×224 (taille préservée !)',
            'color': 'lightgreen',
            'y': 3.5
        },
        {
            'title': 'Avec stride=2',
            'params': 'H_in=224, kernel=3, padding=1, stride=2',
            'calc': r'$\left\lfloor \frac{224 + 2 - 3}{2} \right\rfloor + 1 = 112$',
            'result': '224×224 → 112×112 (division par 2)',
            'color': 'lightsteelblue',
            'y': 1.5
        }
    ]
    
    for example in examples:
        # Cadre
        rect = patches.FancyBboxPatch((0.5, example['y'] - 0.4), 9, 1.5,
                                      boxstyle="round,pad=0.1",
                                      facecolor=example['color'], 
                                      edgecolor='black', linewidth=2, alpha=0.3)
        ax.add_patch(rect)
        
        # Texte
        ax.text(1, example['y'] + 0.7, example['title'], 
               fontsize=13, fontweight='bold')
        ax.text(1, example['y'] + 0.3, example['params'], 
               fontsize=10, style='italic')
        ax.text(5, example['y'], example['calc'], 
               ha='center', fontsize=12)
        ax.text(5, example['y'] - 0.3, example['result'], 
               ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    # Générer la visualisation principale
    print("Génération de la visualisation de convolution...")
    fig1 = create_convolution_visualization()
    output_path1 = 'convolution_padding_explanation.png'
    fig1.savefig(output_path1, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Image sauvegardée : {output_path1}")
    
    # Générer la visualisation de la formule
    print("Génération de la visualisation de la formule...")
    fig2 = create_formula_visualization()
    output_path2 = 'convolution_formula_examples.png'
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Image sauvegardée : {output_path2}")
    
    print("\n🎉 Images générées avec succès !")
    print("Vous pouvez maintenant les inclure dans votre document RST avec :")
    print(f"   .. image:: images/{output_path1}")
    print(f"      :width: 100%")
    print(f"      :align: center")
