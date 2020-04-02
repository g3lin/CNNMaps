# Ouverture des fichiers de la base de données
# Inspire de : https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly

import os
import csv
import numpy as np

import shapely.wkt
import tifffile as tiff

class DataOpening():

    def __init__(self):
        # Constructeur de la classe 
        self.path = "data"
        self.size = "grid_sizes.csv"
        self.polygons = "train_wkt_v4.csv"
        self.tiff_three_band = "three_band/{}.tif"
        self.tiff_sixteen_band = "sixteen_band/{}_M.tif"

    def get_size(self, IMG_ID):
        # Recuperer la taille de l'image
        x_max = None
        y_min = None

        for data_img_id, data_x, data_y in csv.reader(open(os.path.join(self.path, self.size))):
            if data_img_id == IMG_ID:
                x_max, y_min = float(data_x), float(data_y)
                
                break

        return x_max, y_min

    def get_polygons(self, IMG_ID, POLY_TYPE):
        # Recuperer les annotations de l'image en fonction de la classe choisie
        train_polygons = None

        for data_img_id, data_poly_type, data_poly in csv.reader(open(os.path.join(self.path, self.polygons))):
            if data_img_id == IMG_ID and data_poly_type == POLY_TYPE:
                train_polygons = shapely.wkt.loads(data_poly)
                
                break
        
        return train_polygons

    def get_tiff(self, IMG_ID):
        # Recuperer l'image au format .tif
        img_rgb = tiff.imread(os.path.join(self.path, self.tiff_three_band).format(IMG_ID)).transpose([1, 2, 0])
        img_rgb = img_rgb[:3345,:3338]
        
        img_size = img_rgb.shape[:2]

        return img_rgb, img_size