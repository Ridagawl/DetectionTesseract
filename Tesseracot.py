import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import re
import numpy as np
import requests

video_path = '/Users/fouad/Desktop/Tests-Mercredi/Test.mp4'
cap = cv2.VideoCapture(video_path)

# Vérification de l'ouverture de la vidéo
if not cap.isOpened():
    print("Erreur lors de l'ouverture du fichier vidéo.")
    exit()

# Dimensions de la vidéo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#x, y, w, h = 400, 580, 480, 124  # Region of interest

x, y, w, h = 400, 642, 480, 62  # Region of interest

# Taux de rafraîchissement de la vidéo
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

subtitles = []  # Liste pour stocker les sous-titres
frame_number = 0
last_text = ""
start_time = 0
subtitle_index = 1  # Indice pour le numéro de sous-titre
k = 0

c,d=640, 632

e,f=640, 624

special_caracters = ["/", "\\", "|", ">>", ">", "<<", "<", "=", "+", ")", "()", ")", "#", "©", "*", "٠","!","٠","”",".","-","«","-",":","0","1","2","3","4","5","6","7","8","9","١","_","[","]","»",","]


def nettoyer_texte(texte, caracteres_speciaux):
    regex_pattern = '[' + re.escape(''.join(caracteres_speciaux)) + ']'
    return re.sub(regex_pattern, '', texte)

def garder_deux_lignes_max(texte):
    lignes = texte.split('\n')
    lignes = [ligne for ligne in lignes if ligne.strip() and len(ligne.strip()) > 2]
    if len(lignes) > 2:
        lignes = sorted(lignes, key=len, reverse=True)[:2]
    return '\n'.join(lignes)

# Définir la fonction de comparaison de ressemblance

def correct_text_with_textgears(text, api_key):
    url = "https://api.textgears.com/spelling"
    params = {
        'text': text,
        'language': 'ar',
        'key': api_key
    }
    response = requests.get(url, params=params)
    corrections = response.json().get('response', {}).get('errors', [])
    
    # Applique les corrections au texte
    corrected_text = text
    for error in corrections:
        bad = error['bad']
        better = error['better'][0] if error['better'] else bad
        corrected_text = corrected_text.replace(bad, better)
    
    return corrected_text


threshold1=100
#a Distance de Levenshtein
l=0
def comparer_ressemblance(str1, str2, seuil=0.2):
    len_str1 = len(str1)
    len_str2 = len(str2)
    matrix = np.zeros((len_str1+1, len_str2+1))
    for x in range(len_str1+1):
        matrix[x, 0] = x
    for y in range(len_str2+1):
        matrix[0, y] = y
    for x in range(1, len_str1+1):
        for y in range(1, len_str2+1):
            if str1[x-1] == str2[y-1]:
                cost = 0
            else:
                cost = 1
            matrix[x, y] = min(matrix[x-1, y] + 1, matrix[x, y-1] + 1, matrix[x-1, y-1] + cost)
    distance = matrix[len_str1, len_str2]
    longest_length = max(len_str1, len_str2)
    similarity = 1 - (distance / longest_length)
    return similarity >= seuil

api_key="PqTG9jmSchQwRJmP"

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % 15 == 0:
            x, y, w, h = 400, 633, 480, 58


            b, g, r = frame[f, e]
            ba, ga, ra = frame[d, c]

            e1,e2,e3=abs(ba-b),abs(ga-g),abs(ra-r)
            if e1 < threshold1 and e2 < threshold1 and e3 < threshold1:
                 y,h = 575,116  # Region of interest
                            
            roi = frame[y:y+h, x:x+w]
            img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))  # Convertir en format PIL

            # Augmentation du contraste
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)

            # Conversion en niveaux de gris
            img = img.convert("L")

            # Augmentation du contraste et seuillage
            img = ImageOps.autocontrast(img, cutoff=2)
            threshold = 235
            img = img.point(lambda p: 255 if p > threshold else 0)
            img = ImageOps.invert(img)
            img = img.filter(ImageFilter.GaussianBlur(radius=1))

            """
            img.save("image"+str(l)+".png")
            l+=1

            """


            # Extraction du texte avec pytesseract
            extracted_text = pytesseract.image_to_string(img, lang='ara')

            extracted_text = nettoyer_texte(extracted_text, special_caracters)

            extracted_text = garder_deux_lignes_max(extracted_text)

            if len(extracted_text)<=2:
                extracted_text=""

            if extracted_text.strip():
                corrected_text = correct_text_with_textgears(extracted_text, api_key)


                if not last_text or not comparer_ressemblance(extracted_text, last_text):
                    if last_text:
                        end_time = frame_number / fps
                        subtitles.append(f"{subtitle_index}\n")
                        subtitles.append(f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},000 --> {int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},000\n")
                        subtitles.append(f"{last_text}\n\n")
                        subtitle_index += 1
                    start_time = frame_number / fps
                    last_text = extracted_text
            else:
                if last_text:  # Si du texte était enregistré précédemment, mais plus maintenant
                    end_time = frame_number / fps
                    subtitles.append(f"{subtitle_index}\n")
                    subtitles.append(f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},000 --> {int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},000\n")
                    subtitles.append(f"{last_text}\n\n")
                    subtitle_index += 1
                    last_text = ""  # Réinitialiser last_text car il n'y a plus de texte à afficher
        frame_number += 1

    if last_text:
        end_time = frame_number / fps
        subtitles.append(f"{subtitle_index}\n")
        subtitles.append(f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},000 --> {int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},000\n")
        subtitles.append(f"{last_text}\n\n")

finally:
    cap.release()

    # Écriture des sous-titres dans un fichier
    with open('Test13.srt', 'w') as f:
        f.writelines(subtitles)

print("Extraction des sous-titres terminée.")
