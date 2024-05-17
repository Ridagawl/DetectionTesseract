import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import re
import numpy as np
import requests


video_path = '/Users/fouad/Desktop/Tests-Mercredi/Lucas.mkv'
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
out = cv2.VideoWriter('sortie_sous_titre.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (output_width, output_height))


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

special_caracters = ["/", "\\", "|", ">>", ">", "<<", "<", "=", "+", ")", "()", ")", "#", "©", "*", "٠","!","٠","”",".","-","«","-",":","0","1","2","3","4","5","6","7","8","9","١","_","[","]","»",",",'"']


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

def correct_text_with_textgears(text):
    url = "https://api.textgears.com/spelling"
    params = {
        'text': text,
        'language': 'ar-AR',
        'ai': "true",
        'key': "RVib8LT5QgvBkwvx"
    }
    response = requests.get(url, params=params)
    corrections = response.json().get('response', {}).get('errors', [])
    
    # Applique les corrections au texte
    corrected_text = text
    for error in corrections:
        vide=""
        bad = error['bad']
        better = error['better'][0] if error['better'] else vide
        corrected_text = corrected_text.replace(bad, better)
    
    return corrected_text


threshold1=250
#a Distance de Levenshtein
l=0
def comparer_ressemblance(str1, str2, seuil=0.4):
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


def correct_text_with_languagetool(text):
    url = "https://api.languagetool.org/v2/check"
    data = {
        'text': text,
        'language': 'ar'  # Spécifiez 'ar' pour l'arabe
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(url, data=data, headers=headers)
    result = response.json()
    
    # Traitement des suggestions de corrections
    corrected_text = text
    for match in result.get('matches', []):
        start = match['offset']
        end = start + match['length']
        suggestions = match.get('replacements', [])
        if suggestions:
            corrected_text = corrected_text[:start] + suggestions[0]['value'] + corrected_text[end:]
    
    return corrected_text
k=0
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
            threshold = 249
            img = img.point(lambda p: 255 if p > threshold else 0)
            img = ImageOps.invert(img)
            img = img.filter(ImageFilter.GaussianBlur(radius=1))

            

            img.save("image"+str(l)+".png")
            l+=1

            


            # Extraction du texte avec pytesseract
            extracted_text = pytesseract.image_to_string(img, lang='ara')

            extracted_text = nettoyer_texte(extracted_text, special_caracters)

            extracted_text = garder_deux_lignes_max(extracted_text)

            if len(extracted_text)<=2:
                extracted_text=""

            if extracted_text.strip():

                bande_height = 60  # Hauteur de la bande noire
                frame_with_bande = cv2.copyMakeBorder(frame, 0, bande_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            

                frame[y:y+h, x:x+w] = 0

                cv2.putText(frame_with_bande, extracted_text.strip(), (50, output_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
                out.write(frame_with_bande)


                corrected_text=extracted_text


                if not last_text or not comparer_ressemblance(corrected_text, last_text, seuil=0.4):
                    k+=1
                    if last_text:
                        end_time = frame_number / fps
                        subtitles.append(f"{subtitle_index}\n")
                        subtitles.append(f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},000 --> {int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},000\n")
                        subtitles.append(f"{last_text}\n\n")
                        subtitle_index += 1
                    start_time = frame_number / fps
                    last_text = corrected_text
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
    out.release()

    # Écriture des sous-titres dans un fichier
    with open('Lucas.srt', 'w') as f:
        f.writelines(subtitles)

print("Extraction des sous-titres terminée.")

print(k)


#### Optimisation du process de la correction. 

def correct_text_with_textgears(text):
    url = "https://api.textgears.com/spelling"
    params = {
        'text': text,
        'language': 'ar-AR',
        'ai': "true",
        'key': "RVib8LT5QgvBkwvx"
    }
    response = requests.get(url, params=params)
    corrections = response.json().get('response', {}).get('errors', [])
    
    # Applique les corrections au texte
    corrected_text = text
    for error in corrections:
        vide=""
        bad = error['bad']
        better = error['better'][0] if error['better'] else vide
        corrected_text = corrected_text.replace(bad, better)
    
    return corrected_text
#### Optimisation du process de la correction. 
def lire_srt(fichier_srt):
    with open(fichier_srt, 'r', encoding='utf-8') as file:
        contenu = file.read()
    sous_titres = re.findall(r'\d+\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n(.*?)\n', contenu, re.DOTALL)
    return sous_titres

def concatener_sous_titres(sous_titres):
    return '. '.join(texte.replace('\n', ' ') for _, texte in sous_titres) + '.'


def corriger_texte(texte):
    z=0

    E = []  # Liste pour stocker les segments de texte
    n = len(texte)

    print("la valeur de n", n)
    i = 0  # Index pour suivre notre position dans le texte
    limit=3995
    S=""
    while i < n:  # Tant que nous n'avons pas atteint la fin du texte
        # Si ce qui reste du texte est plus court que la limite, on prend tout ce qui reste
        if i + limit > n:
            corrected_morceau=correct_text_with_textgears(texte[i:n])

            print(n-i)
            z+=1
            corrected_morceau=corrected_morceau
            E.append(corrected_morceau)
            break
        # Sinon, on se place à la limite
        fin = i + limit
        # On cherche le prochain espace pour éviter de couper un mot
        while fin < n and texte[fin] != ' ':
            fin += 1
        # On ajoute le segment de texte jusqu'à cet espace
        corrected_morceau=correct_text_with_textgears(texte[i:fin])

        print(fin-i)
        z+=1

        corrected_morceau=corrected_morceau

        E.append(corrected_morceau)
        # On met à jour l'index 'i' pour commencer après l'espace trouvé
        i = fin + 1

    print("le valeur de z ",z)
    k=len(E)
    for i in range (0,k):
        S+=E[i]
    return S


## Traduire en utilisant l'API de Deepl. 

import requests

def translate_with_deepl(text, target_lang="FR"):
    api_key = 'your_deepl_api_key_here'
    url = "https://api.deepl.com/v2/translate"

    # Paramètres pour la requête POST
    data = {
        'auth_key': api_key,
        'text': text,
        'target_lang': "Fr"
    }

    response = requests.post(url, data=data)

    if response.status_code == 200:
        
        json_response = response.json()
        translations = json_response.get('translations', [])
        if translations:
            return translations[0]['text']
        else:
            return "Aucune traduction disponible."
    else:
        return f"Erreur lors de la traduction: {response.text}"

translated_text = translate_with_deepl("Hello, how are you?", "FR")
print(translated_text)


def mettre_a_jour_srt(fichier_srt, sous_titres, nouveaux_textes):
    nouveaux_textes = iter(nouveaux_textes)
    contenu_mis_a_jour = []
    with open(fichier_srt, 'r', encoding='utf-8') as file:
        contenu = file.read()
    for temps, texte in sous_titres:
        texte_mis_a_jour = next(nouveaux_textes)
        contenu = re.sub(re.escape(texte), texte_mis_a_jour, contenu, count=1)
    with open(fichier_srt, 'w', encoding='utf-8') as file:
        file.write(contenu)

fichier_srt = 'Lucas.srt'

# Exécution des fonctions
sous_titres = lire_srt(fichier_srt)
texte_concatene = concatener_sous_titres(sous_titres)
texte_corrige = corriger_texte(texte_concatene)
nouveaux_textes = texte_corrige.split('.')

mettre_a_jour_srt(fichier_srt, sous_titres, nouveaux_textes)

print(len(nouveaux_textes))
print(nouveaux_textes[185])

print("hihi")


