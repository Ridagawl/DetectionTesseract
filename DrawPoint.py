import cv2

def draw_point_and_get_rgb(video_path, x, y,z,c,d, threshold=30):
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)

    # Vérifier si la vidéo a été ouverte correctement
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break



        # Obtenir la valeur RGB au point spécifié
        # Note: OpenCV stocke les couleurs en BGR, pas en RGB
        """
        for i in range (0,640,20):
            for j in range ()
        #b, g, r = frame[y, x]
        #print(f"Valeur RGB au point ({x}, {y}): ({r}, {g}, {b})")
        """
        
        # Dessiner un point sur la frame
        #cv2.circle(frame, (x, y), radius=1, color=(0,0,255), thickness=-1)  # Point rouge
        #cv2.circle(frame, (c, d), radius=1, color=(0,255,0), thickness=-1)  # Point rouge

        b, g, r = frame[y, x]
        ba, ga, ra = frame[d, c]
        print(f"Valeur RGB au point ({c}, {d}): ({ra}, {ga}, {ba})")

        """
        e1,e2,e3=abs(ba-b),abs(ga-g),abs(ra-r)
        if e1 < threshold and e2 < threshold and e3 < threshold:
            cv2.circle(frame, (x, y), radius=10, color=(0,0,255), thickness=-1)  # Point rouge
        """
        ba, ga, ra = frame[d, c]

        # Afficher la frame avec le point
        cv2.imshow('Video with Point', frame)

        # Quitter avec la touche 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()

# Exemple d'utilisation
video_path = 'Test.mp4'
x, y = 640, 624  # Coordonnées du point à tracer
c,d= 640, 632
z=(0,0,255)
a=(0,255,0)
draw_point_and_get_rgb(video_path, x, y, z,c,d)
