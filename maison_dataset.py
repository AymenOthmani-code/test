from torch.utils.data import Dataset
import cv2
import os
import pandas as pd
import torch

class MaisonDataSet(Dataset):
    """
    Dataset personnalisé hérité de la classe torch standard pour charger nos images
    Chaque maison a 4 images (bathroom, bedroom, frontal, kitchen)
    + nombre de chambres et de salles de bain, et un label (3 classes possibles)
    """
    
    def __init__(self, root_dir, csv_file):

        # root_dir: chemin du dossier contenant les images
        # csv_file: chemin du fichier csv "groundtruth" avec les classes
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)

        self.image_types = ["bathroom", "bedroom", "frontal", "kitchen"]
        self.transform = None

        # OPTIONNEL: calculer les class weights qu'on pourra donner à notre loss pour gérer le déséquilibre de classes
        weights = self.data['prix_class'].value_counts(normalize=True).sort_index().values
        self.class_weights = torch.tensor(1.0 / weights, dtype=torch.float32)
        
        # OPTIONNEL: si on veut utiliser les Area et Zipcode en + des images.
        # Vu qu'il n'y a aucun lien logique entre les valeurs de ces deux variables (pas de relation d'ordre),
        # il faut les considérer comme des variables catégorielles.
        # ce qui est différent des nombres de chambres/salles de bain par example.
        # on convertit donc chaque Area/Zipcode en un index unique que l'on va ensuite "one-hot encoder"
        self.area_to_idx = {area: idx for idx, area in enumerate(self.data['Area'].unique())}
        self.zipcode_to_idx = {zipcode: idx for idx, zipcode in enumerate(self.data['Zipcode'].unique())}
        self.n_areas = len(self.area_to_idx)
        self.n_zipcodes = len(self.zipcode_to_idx)
        
    def __len__(self):
        return len(self.data)
    
    def set_transform(self, transform):
        self.transform = transform
    
    def __getitem__(self, idx):
        """
        L"output est un dictionnaire avec:
            "images": tensor (4, channels, height, width)
            "n_chambres": tensor float
            "n_salle_bain": tensor float
            "prix_class": tensor int = NOTRE VERITE TERRAIN
            "area_onehot": tensor de size n_areas (one-hot encoding)
            "zipcode_onehot": tensor de size n_zipcodes (one-hot encoding)
            "id": je le mets car c'est utile si jamais on veut retrouver à quelle image chaque prediction correspond
        """

        img_id = self.data.iloc[idx]["id"]
        n_chambres = self.data.iloc[idx]["n_chambres"]
        n_salle_bain = self.data.iloc[idx]["n_salle_bain"]
        prix_class = self.data.iloc[idx]["prix_class"]
        area = self.data.iloc[idx]["Area"]
        zipcode = self.data.iloc[idx]["Zipcode"]
        
        images = []
        for img_type in self.image_types:
            img_path = os.path.join(self.root_dir, f"{int(img_id)}_{img_type}.jpg")
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                augmented_image = self.transform(image=image)["image"]
            images.append(augmented_image)
        
        # stack la liste vers shape (4, C, H, W) où 4 est le nombre d'images par maison
        images_tensor = torch.stack(images)

        ## on peut visualizer les 4 images avec le code suivant
        ## (penser à mettre num_workers=0 dans votre code pour éviter le parallélisme 
        ## et que plein de fenêtres s'ouvrent en même temps)
        
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        # for i in range(4):
        #     img = images_tensor[i].permute(1, 2, 0).numpy()
        #     axs[i].imshow(img)
        # plt.show()
        # stop

        # génération des one-hot encodings pour Area et Zipcode
        # il s'agit simplement de tensors remplis de 0, mais avec un 1 à l'index correspondant 
        # à la valeur de Area/Zipcode
        area_idx = self.area_to_idx[area]
        zipcode_idx = self.zipcode_to_idx[zipcode]
        area_onehot = torch.zeros(self.n_areas, dtype=torch.float32)
        area_onehot[area_idx] = 1.0
        zipcode_onehot = torch.zeros(self.n_zipcodes, dtype=torch.float32)
        zipcode_onehot[zipcode_idx] = 1.0
        
        out = {
            "images": images_tensor,
            "n_chambres": torch.tensor(n_chambres, dtype=torch.float32),
            "n_salle_bain": torch.tensor(n_salle_bain, dtype=torch.float32),
            "prix_class": torch.tensor(prix_class, dtype=torch.int64),
            "area_onehot": area_onehot,
            "zipcode_onehot": zipcode_onehot,
            "id": img_id
        }

        return out