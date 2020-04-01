# Traitement de la base de donnees
# Inspire de : https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly

import numpy as np
import cv2

class Treatment():

    def __init__(self):
        # Constructeur de la classe
        pass
    
    def get_scalers(self, img_size, x_max, y_min):
        # Table de conversion
        h, w = img_size
        w = w * (w / (w + 1))
        h = h * (h / (h + 1))

        return w / x_max, h / y_min
    
    def get_mask_polygons(self, img_size, list_polygons):
        # Recuperer les polygones definis vis a vis de la classe choisie
        img_mask = np.zeros(img_size, np.uint8)
        exteriors = []
        interiors = []
        
        if list_polygons:
            int_coords = lambda x: np.array(x).round().astype(np.int32)
            
            for polygon in list_polygons:
                exteriors.append(int_coords(polygon.exterior.coords))
            
                for pi in polygon.interiors: 
                    interiors.append(int_coords(pi.coords))
            
            cv2.fillPoly(img_mask, exteriors, 1)
            cv2.fillPoly(img_mask, interiors, 0)
        
        return img_mask

    def get_img_rgb(self, matrix):
        # Traitement de l'image pour un affichage RGB
        w, h, d = matrix.shape

        matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)

        mins = np.percentile(matrix, 1, axis=0)
        maxs = np.percentile(matrix, 99, axis=0) - mins

        matrix = (matrix - mins[None, :]) / maxs[None, :]
        matrix = np.reshape(matrix, [w, h, d])
        matrix = matrix.clip(0, 1)

        return 255 * matrix
    
    def get_img_mask(self, matrix):
        # Traitment de l'image pour un affichage du masque
        return 255 * np.stack([matrix, matrix, matrix])
