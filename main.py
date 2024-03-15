import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def difference(ng1, ng2):
    Gaus = cv.GaussianBlur(ng1, (5, 5), 0) 
    Seuil1, bin1 = cv.threshold(Gaus, 0, 255, cv.THRESH_OTSU) 
    Gaus = cv.GaussianBlur(ng2, (5, 5), 0)
    Seuil2, bin2 = cv.threshold(Gaus, 0, 255, cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    imFermeture1 = cv.morphologyEx(bin1, cv.MORPH_CLOSE, kernel)
    imFermeture2 = cv.morphologyEx(bin2, cv.MORPH_CLOSE, kernel)
    Diff = np.bitwise_xor(imFermeture1, imFermeture2)
    return Diff

def color_diff(img, diff_img):
    img_rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    for i in range(diff_img.shape[0]):
        for j in range(diff_img.shape[1]):
            if diff_img[i, j] != 0:
                img_rgb[i, j] = [255, 0, 0]  # Set pixel to red
    return img_rgb

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv.imread(img_path, 0)  # Read image as grayscale
            if img is not None:
                images.append(img)
    return images

def compute_deforestation(folder):
    images = load_images_from_folder(folder)
    diffs = []
    years = []
    pourcentages = []

    for i in range(len(images) - 1):
        diffs.append(difference(images[i], images[i+1]))
        years.append(f"{i+2000}")

    for diff in diffs:
        total_couverture_diff = np.sum(diff) / 255
        pourcentages.append(float('{0:.2f}'.format(total_couverture_diff / diff.size * 100)))

    return years, pourcentages, images, diffs

def plot_deforestation(years, pourcentages, diffs, images):
    plt.figure(figsize=(12, 6))
    for i in range(len(years)):
        plt.subplot(3, 5, i+1)
        img = color_diff(images[i], diffs[i])
        plt.imshow(img)
        plt.title(f"{years[i]}-{int(years[i])+1}")
        plt.axis('off')
    plt.savefig(os.path.join("IDHAR SAVE KARNE KA PATH", 'deforestation1.png'))

    
    
    x = [f"{years[i]}-{int(years[i])+1}" for i in range(len(years))]
    y = pourcentages
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o')
    plt.xlabel('Years')
    plt.ylabel('Percentages')
    plt.title('Temporal Study of Deforestation')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join("IDHAR SAVE KARNE KA PATH", 'deforestation2.png'))


def main():
    folder_path = ""
    years, pourcentages, images, diffs = compute_deforestation(folder_path)
    plot_deforestation(years, pourcentages, diffs, images)

if __name__ == "__main__":
    main()
