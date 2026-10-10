#!/usr/bin/env python3
"""
Génère les figures du chapitre 4 (Manipulation d'images) à partir de ballon.jpg,
ainsi que la figure des filtres de convolution du chapitre 5 (images/chap5/).

Usage (depuis n'importe quel dossier) :
    python generate_figures_chap4.py

Dépendances : numpy, matplotlib, pillow, torch, torchvision.
Pour changer d'image d'exemple, remplacer ballon.jpg puis adapter BALL_BOX (zone du ballon).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(HERE, "ballon.jpg")

# Zone du ballon dans ballon.jpg : (y_min, y_max, x_min, x_max)
BALL_BOX = (95, 205, 405, 520)
BALL_CENTER = (461, 149)  # (x, y)
BALL_RADIUS = 52

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})


def save(fig, name, dossier=HERE):
    os.makedirs(dossier, exist_ok=True)
    path = os.path.join(dossier, name)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("->", name)


def load():
    img = np.array(Image.open(IMG_PATH).convert("RGB"))  # (H, W, 3) uint8
    gray = np.array(Image.open(IMG_PATH).convert("L"))   # (H, W) uint8
    return img, gray


def fig_pixels(img, gray):
    """Une image = un tableau de nombres : zoom sur quelques pixels."""
    y0, x0, n = 162, 486, 8  # bord entre un panneau noir et le blanc du ballon
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].add_patch(patches.Rectangle((x0 - 0.5, y0 - 0.5), n, n, fill=False, ec="red", lw=2.5))
    axes[0].annotate("zone zoomée", xy=(x0, y0 + n), xytext=(x0 - 230, y0 + 120), color="red",
                     fontsize=12, fontweight="bold", arrowprops=dict(arrowstyle="->", color="red", lw=2))
    axes[0].set_title("Image en niveaux de gris (482 × 644 pixels)")
    axes[0].axis("off")

    zone = gray[y0:y0 + n, x0:x0 + n]
    axes[1].imshow(zone, cmap="gray", vmin=0, vmax=255)
    for i in range(n):
        for j in range(n):
            v = zone[i, j]
            axes[1].text(j, i, str(v), ha="center", va="center", fontsize=10,
                         color="black" if v > 128 else "white")
    axes[1].set_xticks(range(n), [str(x0 + j) for j in range(n)], fontsize=9)
    axes[1].set_yticks(range(n), [str(y0 + i) for i in range(n)], fontsize=9)
    axes[1].set_xlabel("colonne (x)")
    axes[1].set_ylabel("ligne (y)")
    axes[1].set_title(f"Zoom sur {n} × {n} pixels : leurs valeurs")
    save(fig, "chap4_pixels_zoom.png")


def image_couleurs():
    """Image synthétique : 8 bandes de couleurs pures (même code que dans le cours)."""
    img = np.zeros((100, 400, 3), dtype=np.uint8)
    couleurs = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                [255, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0]]
    for i, c in enumerate(couleurs):
        img[:, i * 50:(i + 1) * 50] = c
    return img


def fig_canaux():
    img = image_couleurs()
    noms = ["rouge", "vert", "bleu", "jaune", "magenta", "cyan", "blanc", "noir"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 8.6))
    axes[0].imshow(img)
    axes[0].set_title("Image RGB : img.shape = (100, 400, 3)", fontsize=12)
    for i, n in enumerate(noms):
        axes[0].text(i * 50 + 25, 50, n, ha="center", va="center", fontsize=10,
                     color="white" if n in ["bleu", "noir"] else "black")
    for k, nom in enumerate(["R (rouge)", "G (vert)", "B (bleu)"]):
        canal = img[:, :, k]
        axes[k + 1].imshow(canal, cmap="gray", vmin=0, vmax=255)
        axes[k + 1].set_title(f"Canal {nom} : img[:, :, {k}] affiché en niveaux de gris", fontsize=11)
        for i in range(8):
            v = canal[50, i * 50]
            axes[k + 1].text(i * 50 + 25, 50, str(v), ha="center", va="center", fontsize=12,
                             fontweight="bold", color="black" if v > 128 else "white")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():   # cadre gris pour voir les bandes blanches
            spine.set_edgecolor("gray")
    plt.tight_layout()
    save(fig, "chap4_canaux_rgb.png")


def fig_slicing(img):
    # Reproduit exactement le code de la section 3.3 du chapitre 4
    y1, y2, x1, x2 = BALL_BOX
    ballon = img[y1:y2, x1:x2]
    rouge = img[:, :, 0]
    petite = img[::8, ::8]
    miroir_h = img[:, ::-1]
    miroir_v = img[::-1, :]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("img (originale)")
    axes[0, 1].imshow(ballon)
    axes[0, 1].set_title("ballon (recadrage)")
    axes[0, 2].imshow(rouge, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("rouge (canal R)")
    axes[1, 0].imshow(petite)
    axes[1, 0].set_title("petite (1 pixel sur 8)")
    axes[1, 1].imshow(miroir_h)
    axes[1, 1].set_title("miroir_h")
    axes[1, 2].imshow(miroir_v)
    axes[1, 2].set_title("miroir_v")
    save(fig, "chap4_slicing.png")


def fig_vue_copie(img):
    # Reproduit exactement le code de l'avertissement vue / copie (section 3.3 du chapitre 4)
    y1, y2, x1, x2 = BALL_BOX
    img1 = img.copy()
    ballon = img1[y1:y2, x1:x2]
    ballon[:] = 0
    img2 = img.copy()
    ballon2 = img2[y1:y2, x1:x2].copy()
    ballon2[:] = 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(img1)
    axes[0].set_title("img après ballon[:] = 0 (vue)")
    axes[1].imshow(img2)
    axes[1].set_title("img2 après ballon2[:] = 0 (copie)")
    save(fig, "chap4_vue_copie.png")


def decoupe_patchs(img, taille=160):
    # Même code que dans la section 3.5 du chapitre 4
    H, W, C = img.shape
    patchs, coins = [], []
    for y in range(0, H - taille + 1, taille):
        for x in range(0, W - taille + 1, taille):
            patchs.append(img[y:y+taille, x:x+taille])
            coins.append((y, x))
    return patchs, coins


def fig_patchs_schema(img, taille=160):
    # Schéma : ordre de parcours des deux boucles et coin en haut à gauche de chaque patch
    patchs, coins = decoupe_patchs(img, taille)
    H, W, _ = img.shape
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(img)
    for k, (y, x) in enumerate(coins):
        ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), taille, taille, fill=False, ec="yellow", lw=2))
        ax.plot(x, y, "o", color="red", ms=7)
        ax.text(x + taille / 2, y + taille / 2, f"patchs[{k}]\ny = {y}, x = {x}", ha="center", va="center",
                fontsize=11, bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.set_xticks(sorted({x for _, x in coins} | {W}))
    ax.set_yticks(sorted({y for y, _ in coins} | {H}))
    ax.set_xlabel("x (colonnes) : boucle intérieure")
    ax.set_ylabel("y (lignes) : boucle extérieure")
    ax.set_title("Point rouge = coin en haut à gauche img[y, x] de chaque patch")
    save(fig, "chap4_patchs_schema.png")


def fig_patchs(img, taille=160):
    # Reproduit exactement le code d'affichage de la section 3.5 du chapitre 4
    patchs, _ = decoupe_patchs(img, taille)
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for k in range(len(patchs)):
        axes[k // 4, k % 4].imshow(patchs[k])
        axes[k // 4, k % 4].set_title(f"patchs[{k}]")
        axes[k // 4, k % 4].axis("off")
    save(fig, "chap4_patchs.png")


def fig_operations(img, gray):
    f = img.astype(np.float32)
    lum = np.clip(f + 60, 0, 255).astype(np.uint8)
    contraste = np.clip((f - 128) * 1.5 + 128, 0, 255).astype(np.uint8)
    negatif = 255 - img
    gris = (0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]).astype(np.uint8)
    masque = (img[:, :, 0] > 150) & (img[:, :, 1] > 150) & (img[:, :, 2] > 150)
    recolore = img.copy()
    recolore[masque] = [255, 0, 0]

    vues = [
        (img, "Originale", None),
        (lum, "Luminosité : img + 60", None),
        (contraste, "Contraste : (img − 128) × 1.5 + 128", None),
        (negatif, "Négatif : 255 − img", None),
        (gris, "Niveaux de gris\n0.299 R + 0.587 G + 0.114 B", "gray"),
        (masque, "Masque : R, G et B > 150\n(True = blanc)", "gray"),
        (recolore, "img[masque] = [255, 0, 0]", None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.2))
    for ax, (v, titre, cmap) in zip(axes.ravel(), vues):
        ax.imshow(v, cmap=cmap)
        ax.set_title(titre, fontsize=11)
        ax.axis("off")
    axes.ravel()[-1].axis("off")
    save(fig, "chap4_operations_pixels.png")


def fig_histogramme(img, gray):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for k, (nom, c) in enumerate([("R", "red"), ("G", "green"), ("B", "blue")]):
        axes[0].hist(img[:, :, k].ravel(), bins=256, range=(0, 256), color=c, alpha=0.5, label=f"canal {nom}")
    axes[0].set_title("Histogramme par canal")
    axes[0].set_xlabel("valeur du pixel")
    axes[0].set_ylabel("nombre de pixels")
    axes[0].legend()
    axes[1].hist(gray.ravel(), bins=256, range=(0, 256), color="gray")
    axes[1].set_title("Histogramme en niveaux de gris")
    axes[1].set_xlabel("valeur du pixel (0 = noir, 255 = blanc)")
    axes[1].annotate("herbe sombre", xy=(75, 8000), xytext=(115, 8000), arrowprops=dict(arrowstyle="->"))
    axes[1].annotate("lignes blanches\net ballon", xy=(200, 700), xytext=(170, 5000), arrowprops=dict(arrowstyle="->"))
    for ax in axes:
        ax.set_xlim(0, 255)
    save(fig, "chap4_histogramme.png")


def rotation_visible(pil):
    torch.manual_seed(3)  # avec cette graine, l'angle tiré vaut environ -30°
    return transforms.RandomRotation(30)(pil)


def fig_resize():
    # Reproduit exactement le code de la section 5.1 du chapitre 4
    img_pil = Image.open(IMG_PATH)
    petite = transforms.Resize((64, 64))(img_pil)
    petite2 = img_pil.resize((128, 64))
    proportionnelle = transforms.Resize(240)(img_pil)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].imshow(img_pil)
    axes[0].set_title("img_pil (644 × 482)")
    axes[1].imshow(petite)
    axes[1].set_title("petite (64 × 64)")
    axes[2].imshow(petite2)
    axes[2].set_title("petite2 (128 × 64)")
    axes[3].imshow(proportionnelle)
    axes[3].set_title("proportionnelle (320 × 240)")
    save(fig, "chap4_resize.png")


def fig_normalize():
    # Reproduit exactement le code de la section 5.3 du chapitre 4
    img_t = transforms.ToTensor()(Image.open(IMG_PATH))
    img_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img_t)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].imshow(img_t.permute(1, 2, 0))
    axes[0].set_title("img_t : valeurs dans [0, 1]")
    axes[1].imshow(img_norm.permute(1, 2, 0).clamp(0, 1))   # même rendu que l'affichage direct, sans l'avertissement
    axes[1].set_title("img_norm affichée directement")
    axes[2].hist(img_t.flatten(), bins=100, alpha=0.5, label="img_t")
    axes[2].hist(img_norm.flatten(), bins=100, alpha=0.5, label="img_norm")
    axes[2].set_title("Valeurs des pixels (3 canaux)")
    axes[2].legend()
    save(fig, "chap4_normalize.png")


def fig_batch():
    # Reproduit exactement le code de la section 6.1 du chapitre 4
    pretraitement = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    images = [pretraitement(Image.open(IMG_PATH).convert("RGB")) for _ in range(4)]
    batch = torch.stack(images)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        axes[i].imshow(batch[i].permute(1, 2, 0))
        axes[i].set_title(f"batch[{i}]")
    save(fig, "chap4_batch.png")


def fig_batch_operations():
    # Reproduit exactement le code de la section 6.2 du chapitre 4
    pretraitement = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    batch = torch.stack([pretraitement(Image.open(IMG_PATH).convert("RGB")) for _ in range(4)])
    miroirs = batch.flip(dims=[3])
    gris = (0.299 * batch[:, 0] + 0.587 * batch[:, 1] + 0.114 * batch[:, 2]).unsqueeze(1)
    petits = F.interpolate(batch, size=(64, 64), mode="bilinear")

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(batch[0].permute(1, 2, 0))
    axes[0].set_title("batch[0]")
    axes[1].imshow(batch[0, 0], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("batch[0, 0] (canal rouge)")
    axes[2].imshow(miroirs[0].permute(1, 2, 0))
    axes[2].set_title("miroirs[0]")
    axes[3].imshow(gris[0, 0], cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("gris[0, 0]")
    axes[4].imshow(petits[0].permute(1, 2, 0))
    axes[4].set_title("petits[0] (64 × 64)")
    save(fig, "chap4_batch_operations.png")


def fig_interpolation():
    # Les tableaux de l'exemple d'interpolation de la section 6.2 du chapitre 4
    t = torch.tensor([[[[0., 100.], [200., 40.]]]])
    vues = [
        (t[0, 0], "Image d'origine (2 × 2)"),
        (F.interpolate(t, size=(4, 4), mode="nearest")[0, 0], "mode='nearest' (4 × 4)"),
        (F.interpolate(t, size=(4, 4), mode="bilinear")[0, 0], "mode='bilinear' (4 × 4)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (v, titre) in zip(axes, vues):
        ax.imshow(v, cmap="gray", vmin=0, vmax=200)
        for (i, j), val in np.ndenumerate(v.numpy()):
            ax.text(j, i, f"{val:g}", ha="center", va="center", fontsize=11,
                    color="black" if val > 100 else "white")
        ax.set_title(titre)
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, "chap4_interpolation.png")


def fig_transforms():
    # Même code que dans la section 5.2 du chapitre 4, avec des graines fixées pour des tirages lisibles
    pil = Image.open(IMG_PATH).convert("RGB")
    # Les tirages aléatoires sont faits dans cet ordre pour garder des résultats lisibles,
    # puis les images sont affichées dans l'ordre de la liste de la section 5.2
    torch.manual_seed(0)
    crop = transforms.CenterCrop(300)(pil)
    rotation = rotation_visible(pil)
    flip = transforms.RandomHorizontalFlip(p=1.0)(pil)
    jitter = transforms.ColorJitter(brightness=0.8, contrast=0.5, saturation=0.8)(pil)
    gris = transforms.Grayscale()(pil)
    resized_crop = transforms.RandomResizedCrop(224, scale=(0.2, 0.5))(pil)
    flou = transforms.GaussianBlur(kernel_size=15, sigma=5)(pil)
    torch.manual_seed(5)  # avec cette graine, le carré tiré contient le ballon
    random_crop = transforms.RandomCrop(224)(pil)
    vues = [
        (pil, "Originale (644 × 482)"),
        (crop, "CenterCrop(300)"),
        (random_crop, "RandomCrop(224)"),
        (resized_crop, "RandomResizedCrop(224)"),
        (flip, "RandomHorizontalFlip(p=1.0)"),
        (rotation, "RandomRotation(30)"),
        (jitter, "ColorJitter(...)"),
        (gris, "Grayscale()"),
        (flou, "GaussianBlur(15, sigma=5)"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, (v, titre) in zip(axes.ravel(), vues):
        ax.imshow(v, cmap="gray" if v.mode == "L" else None)
        ax.set_title(titre, fontsize=12)
        ax.axis("off")
    save(fig, "chap4_transforms.png")

    # Même code que dans la section 5.4 du chapitre 4 (augmentation, puis Normalize annulée pour l'affichage)
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    torch.manual_seed(1)
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.3))
    for i in range(6):
        x = augmentation(pil)
        x_affichable = x * std + mean
        axes[i].imshow(x_affichable.permute(1, 2, 0))
        axes[i].set_title(f"tirage n°{i + 1}")
        axes[i].axis("off")
    save(fig, "chap4_augmentations.png")


def fig_filtres(gray):
    zone = gray[40:320, 260:600]  # ballon + croisement des lignes
    x = torch.tensor(zone, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0  # (1, 1, H, W)

    g1d = torch.exp(-torch.arange(-3, 4, dtype=torch.float32) ** 2 / (2 * 1.5 ** 2))
    gauss = torch.outer(g1d, g1d)
    gauss = gauss / gauss.sum()
    noyaux = [
        ("Identité (3×3)", torch.tensor([[0., 0, 0], [0, 1, 0], [0, 0, 0]])),
        ("Flou moyen (7×7)", torch.ones(7, 7) / 49),
        ("Flou gaussien (7×7, σ = 1.5)", gauss),
        ("Accentuation (3×3)", torch.tensor([[0., -1, 0], [-1, 5, -1], [0, -1, 0]])),
        ("Sobel x (3×3) → contours verticaux\n(rouge > 0, blanc ≈ 0, bleu < 0)", torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])),
        ("Sobel y (3×3) → contours horizontaux\n(rouge > 0, blanc ≈ 0, bleu < 0)", torch.tensor([[-1., -2, -1], [0, 0, 0], [1, 2, 1]])),
    ]
    sorties = []
    for titre, k in noyaux:
        k = k.unsqueeze(0).unsqueeze(0)  # (1, 1, kH, kW)
        y = F.conv2d(x, k, padding=k.shape[-1] // 2)[0, 0]
        sorties.append((titre, y))
    gx, gy = sorties[4][1], sorties[5][1]
    sorties.append(("Norme du gradient\n√(Sobel x² + Sobel y²)", torch.sqrt(gx ** 2 + gy ** 2)))

    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6))
    for ax, (titre, y) in zip(axes.ravel(), sorties):
        y = y[4:-4, 4:-4]  # on n'affiche pas les bords (effet du padding par des zéros)
        if "Sobel" in titre and "Norme" not in titre:
            lim = y.abs().max().item()
            ax.imshow(y, cmap="seismic", vmin=-lim, vmax=lim)
        elif "Norme" in titre:
            ax.imshow(y, cmap="gray")
        else:
            ax.imshow(y.clamp(0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(titre, fontsize=11)
        ax.axis("off")
    axes.ravel()[-1].axis("off")
    save(fig, "chap5_filtres_convolution.png", dossier=os.path.join(HERE, "..", "chap5"))


def fig_taches(img):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (xx - BALL_CENTER[0]) ** 2 + (yy - BALL_CENTER[1]) ** 2
    ballon = d2 <= BALL_RADIUS ** 2

    def lisser(a, k):  # moyenne glissante k×k (pour des masques moins bruités)
        t = torch.tensor(a, dtype=torch.float32)[None, None]
        return F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="replicate"), k, stride=1)[0, 0].numpy()

    f = img.astype(np.float32)
    lignes = (lisser(f.min(axis=2), 9) > 95) & (d2 > (BALL_RADIUS + 8) ** 2)
    verdure = lisser(f[:, :, 1] - f[:, :, 0], 5)  # G − R : l'herbe est plus verte que le robot
    robot = (verdure < 7) & ~lignes & ~ballon & (xx < 380) & (yy > 180)

    seg = np.zeros((h, w, 3), dtype=np.uint8)
    seg[:] = [60, 140, 60]        # terrain
    seg[lignes] = [235, 235, 235]  # lignes
    seg[robot] = [120, 60, 160]    # robot
    seg[ballon] = [255, 150, 0]    # ballon

    y1, y2, x1, x2 = BALL_BOX
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    axes[0].imshow(img)
    axes[0].set_title("Classification\nétiquette de l'image : « ballon »")
    axes[1].imshow(img)
    axes[1].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=3))
    axes[1].text(x1, y1 - 8, "ballon", color="red", fontsize=13, fontweight="bold")
    axes[1].set_title("Détection\nune boîte englobante par objet")
    axes[2].imshow(seg)
    axes[2].set_title("Segmentation\nune classe par pixel")
    legend = [patches.Patch(color=np.array(c) / 255, label=n) for n, c in
              [("terrain", [60, 140, 60]), ("lignes", [235, 235, 235]), ("robot", [120, 60, 160]), ("ballon", [255, 150, 0])]]
    axes[2].legend(handles=legend, loc="lower right", fontsize=10, framealpha=0.9)
    for ax in axes:
        ax.axis("off")
    save(fig, "chap4_taches_vision.png")


if __name__ == "__main__":
    img, gray = load()
    fig_pixels(img, gray)
    fig_canaux()
    fig_slicing(img)
    fig_vue_copie(img)
    fig_patchs_schema(img)
    fig_patchs(img)
    fig_operations(img, gray)
    fig_histogramme(img, gray)
    fig_resize()
    fig_transforms()
    fig_normalize()
    fig_batch()
    fig_batch_operations()
    fig_interpolation()
    fig_filtres(gray)
    fig_taches(img)
