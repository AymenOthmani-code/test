import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

column_names = ["n_chambres", "n_salle_bain", "Area", "Zipcode", "prix"]

# Resolve CSV path relative to this script so running from other CWDs still works.
script_dir = Path(__file__).parent
csv_path = script_dir / "projet_miage" / "Houses-dataset" / "HousesInfo.csv"
if not csv_path.exists():
	available = sorted([p.name for p in script_dir.iterdir()])
	raise FileNotFoundError(
		f"Houses dataset not found at: {csv_path}\n"
		f"Please place 'HousesInfo.csv' in that folder or update the path.\n"
		f"Current script dir: {script_dir}\n"
		f"Files in script dir: {available}"
	)

data = pd.read_csv(csv_path, sep=" ", names=column_names)
print(data.head())

# conversion n_chambres et n_salle_bain en int (car pas de sens pourquoi des floats ici ??)
data["n_chambres"] = data["n_chambres"].astype(int)
data["n_salle_bain"] = data["n_salle_bain"].astype(int)

# on affiche distribution prix
sns.histplot(data['prix'] / 1e6, bins=100, kde=True) # (/1e6 pour avoir prix en millions et bins=100 c'est arbitraire)
plt.title('Distribution du prix')
plt.xlabel('prix (en millions)')
plt.ylabel('fréquence')
plt.grid(True)
plt.show()

# on affiche les variables n_chambres et n_salle_bain par rapport au prix, 
# pour voir si intéressant de les inclure dans notre modèle
plt.subplot(1, 2, 1)
sns.scatterplot(x='n_chambres', y=data['prix'] / 1e6, data=data)
plt.title('Prix vs n_chambres')
plt.xlabel('Nombre de chambres')
plt.ylabel('prix (en millions)')
plt.grid(True)
plt.subplot(1, 2, 2)
sns.scatterplot(x='n_salle_bain', y=data['prix'] / 1e6, data=data)
plt.title('Prix vs n_salle_bain')
plt.xlabel('Nombre de salles de bain')
plt.ylabel('prix (en millions)')
plt.grid(True)
plt.tight_layout()
plt.show()


########## Discrétisation du prix en classes ##########
# Plutôt que de faire une tâche de regression, on vous propose une tâche de classification,
# donc on discrétise le prix en 3 classes: bas, moyen, élevé

from sklearn.preprocessing import KBinsDiscretizer

# on utilise la stratégie quantile, ce qui nous génère des classes qui ont un nombre d'éléments le plus proche possible
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
data['prix_class'] = est.fit_transform(data[['prix']])

# on peut afficher le même graphique que précédemment, mais avec les classes de prix
sns.histplot(data=data, x='prix', bins=100, hue='prix_class', palette='viridis', multiple='stack')
plt.title('Distribution du prix')
plt.xlabel('prix (en millions)')
plt.ylabel('fréquence')
plt.grid(True)
plt.show()

# on s'assure que chaque ligne correspond bien aux ids de 1 à N qui sont dans le dataset
data.reset_index(inplace=True)
data.rename(columns={'index': 'id'}, inplace=True)
data['id'] = data['id'] + 1

# on normalise n_chambres et n_salle_bain entre 0 et 1 pour faciliter l'apprentissage
data["n_chambres"] = data["n_chambres"] / data["n_chambres"].max()
data["n_salle_bain"] = data["n_salle_bain"] / data["n_salle_bain"].max()

# on sauvegarde notre nouveau fichier .csv modifié qui sera notre groundtruth
data.to_csv("groundtruth.csv", index=False)

