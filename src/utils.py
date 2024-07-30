"""
Methoden für Datenanalyse / Datenvariabilität
- Winkel: H - CENTRAL_ATOM - REST_ATOME  (siehe all_angles_H_central_neighbor)
- Bindungslängen
- Dihedralwinkel

Methoden für Regressionsgüte (Regression von Hydrogen position H', ground truth H)
- absoluter Abstand H zu H' (keine Implementierung notwendig, einfach mean_squared_error von sklearn verwenden)
- Winkel H zu H' (siehe metric_cosine_similarity_and_angle)
- Bindungslängendifferenz H zu H' (siehe metric_bindungslänge_differenz)
- Abstand zwischen zwei Verteilungen durch Wasserstein Distanz, z.b. Winkelverteilung in Daten vs. predictete Winkelverteilung (keine Implementierung notwendig, einfach wasserstein_distance von scipy verwenden)

--> Auch plotten von Verteilungen möglich (siehe plot_fit)

Siehe Beispiele durch Ausführung dieses Skripts (main-methode ganz unten)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import itertools
from tqdm import tqdm

# Wasserstein distance
from calc_wasserstein import *


# -------------------------------------------------
# Methoden: Datenanalyse / Datenvariabilität
# -------------------------------------------------

# Gibt alle Winkel zurück, nicht den Durchschnitt aller Winkel! Dies muss im Nachgang separat berechnet werden
def all_angles_H_central_neighbor(X_relative_coordinates, y):
    assert len(X_relative_coordinates) == len(y), "Length of X and y is not the same"

    all_angles = []
    for current_X, current_y in zip(X_relative_coordinates, y):
        for coords in current_X:
            _, angle = cosine_similarity_and_angle(coords, current_y)
            angle = abs(angle)  # for the angles H (or  H') - central -  neighbor, we just need the absolute angle
            all_angles.append(angle)
    return np.array(all_angles)


def compute_dihedral(p0, p1, p2, p3):
    """Calculate the dihedral angle between four points. p1 and p2 are the fixed points of the two planes"""
    # Cast inputs to numpy arrays
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1 for the calculation
    b1_norm = b1 / np.linalg.norm(b1)

    # Compute normal vectors to the planes
    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b1, b2)

    # Compute the dot and cross prducts needed for the angle
    y = np.dot(np.cross(b0xb1, b1xb2), b1_norm)
    x = np.dot(b0xb1, b1xb2)

    return np.degrees(np.arctan2(y, x))


def all_dihedral_angles(coordinates):
    """Compute all possible dihedral angles for a list of coordinates."""
    num_atoms = len(coordinates)
    if num_atoms < 4:
        raise Exception("Cannot compute dihedral angles for less than 4 coordinates")
    angles = []
    for fixed_pair in itertools.combinations(range(num_atoms), 2):
        remaining_indices = set(range(num_atoms)) - set(fixed_pair)
        for p2, p3 in itertools.permutations(remaining_indices, 2):
            p0, p1 = fixed_pair
            angle = compute_dihedral(coordinates[p0], coordinates[p1], coordinates[p2], coordinates[p3])
            angles.append(((p0, p1, p2, p3), angle))
    return angles


def fit_gaussian(data, n_components=1):
    data = data.reshape(-1, 1)  # Reshape for sklearn
    if n_components == 1:
        # Fit a single Gaussian
        mu, std = np.mean(data), np.std(data)
        return mu, std
    else:
        # Fit a Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_components, random_state=0).fit(data)
        # Extract means and variances
        means = gmm.means_.flatten()
        variances = gmm.covariances_.flatten()
        return gmm, means, variances


def plot_fit(data, n_components=1):
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000).reshape(-1, 1)
    
    if n_components == 1:
        mu, std = fit_gaussian(data, n_components)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        plt.title(f'Single Gaussian Fit: mu = {mu:.2f}, std = {std:.2f}')
    else:
        gmm, means, variances = fit_gaussian(data, n_components)
        logprob = gmm.score_samples(x)
        pdf = np.exp(logprob)
        plt.plot(x, pdf, 'k', linewidth=2)
        plt.title(f'Gaussian Mixture Model with {n_components} components')
        for mean, var in zip(means, variances):
            std = np.sqrt(var)
            p = norm.pdf(x, mean, std)
            plt.plot(x, p, '--', linewidth=2)
            plt.text(mean, norm.pdf(mean, mean, std), f'mu={mean:.2f}, std={std:.2f}', 
                     horizontalalignment='center')
    plt.show()


# -------------------------------------------------
# Methoden: Regressionsgüte
# -------------------------------------------------

def cosine_similarity_and_angle(vector1, vector2):
    # Convert inputs to numpy arrays if they are not already
    pred = np.array(vector1)
    truth = np.array(vector2)
    
    # Compute the dot product
    dot_product = np.dot(pred, truth)
    
    # Compute the norms (magnitudes) of the vectors
    norm_pred = np.linalg.norm(pred)
    norm_truth = np.linalg.norm(truth)
    
    # Compute the cosine similarity
    cosine_sim = dot_product / (norm_pred * norm_truth)
    
    # Clip the cosine similarity to the range [-1, 1] to avoid numerical issues
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    
    # Compute the angle in radians
    angle_radians = np.arccos(cosine_sim)
    
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)

    # Round the values to three decimal places
    cosine_sim = round(cosine_sim, 3)
    #angle_radians = round(angle_radians, 3)
    angle_degrees = round(angle_degrees, 3)
    
    return cosine_sim, angle_degrees


def metric_cosine_similarity_and_angle(y_test, y_pred):
    assert len(y_test) == len(y_pred), "Length of y_test and y_pred is not the same"
    
    cosine_similarities = []
    angles = []
    for t, p in zip(y_test, y_pred):
        cosine_sim, ang = cosine_similarity_and_angle(t, p)
        ang = abs(ang)  # for the angle between H and H' we just need the absolute angle
        cosine_similarities.append(cosine_sim)
        angles.append(ang)
    avg_cosine_similarity = np.average(cosine_similarities)
    avg_angle = np.average(angles)
    return avg_cosine_similarity, avg_angle


def bindungslängen(data):
    return np.linalg.norm(data, axis=1)


# Bindungslängendifferenz zwischen nur zwei Vektoren (Praktisch: |Länge Vektor 1 - Länge Vektor 2|)
def bindungslänge_differenz(vector1, vector2):
    # truth and pred should have same shape
    np_vector1 = np.array(vector1)
    np_vector2 = np.array(vector2)
    
    return abs(np.linalg.norm(np_vector1) - np.linalg.norm(np_vector2))


def metric_bindungslänge_differenz(y_test, y_pred):
    assert len(y_test) == len(y_pred), "Length of y_test and y_pred is not the same"
        
    return abs(np.average(np.linalg.norm(y_test, axis=1)) - np.average(np.linalg.norm(y_pred, axis=1)))


# combining all regression metrics in one method
def eval_regression(y_test, y_pred, X_test_coords=None):
    print("Alle folgenden Metriken sind H zu H'")
    # absoluter Abstand H zu H' (MSE Score)
    mse = mean_squared_error(y_test, y_pred)
    print("Mittlerer Abstand: ", mse)

    # R2 Bestimmtheitskoeffizient
    r2 = r2_score(y_test, y_pred)
    print("R2 Bestimmtheitsmaß: ", r2)

    # Durchschnittliche Cosine Similarity (kann ignoriert werden) und durchschnittliche Winkel
    avg_cosine_similarity, avg_angle = metric_cosine_similarity_and_angle(y_test, y_pred)
    print("Average Cosine Similarity: ", avg_cosine_similarity)
    print("Average Angle (degrees): ", avg_angle)

    # Duchschnittliche Bindungslängendifferenz
    avg_bindugslänge_diff = metric_bindungslänge_differenz(y_test, y_pred)
    print("Durschnittliche Bindungslänge Differenz: ", avg_bindugslänge_diff)

    if X_test_coords is not None and X_test_coords.size > 0:
        # Wasserstein Distanz zwischen zwei Winkelverteilungen
        all_angles_test = all_angles_H_central_neighbor(X_test_coords, y_test)  # alle Winkel im Test set
        all_angles_pred = all_angles_H_central_neighbor(X_test_coords, y_pred)  # alle Winkel durch Prediction
        wasserstein = calculate_wasserstein_distance(all_angles_test, all_angles_pred, bins=180, range=(0, 180))
        print("Die Wasserstein Distanz zwischen den beiden Winkelverteilungen beträgt: ", wasserstein)
        return mse, r2, avg_angle, avg_bindugslänge_diff, wasserstein
    
    return mse, r2, avg_angle, avg_bindugslänge_diff

# -------------------------------------------------
# Methoden: Zusätzliches
# -------------------------------------------------

# Gegeben feature Vektoren X, es entinimmt nur die relativen Koordinaten in den Feature Vektoren (One hot encodings kommen weg)
def extract_relative_coordinates(X):
    X_rel = []
    for sample_nr, sample in enumerate(X):
        X_rel_current = []
        i = 0
        while i < len(sample):
            x, y, z = sample[i+8], sample[i+9], sample[i+10]
            if x == 0 and y == 0 and z == 0:
                break # there should be no real value after zero padding happended once
            else:
                X_rel_current.append([x, y, z])
            i += 11
        if not X_rel_current:
            print("Broken sample: ", sample, sample_nr)
            raise Exception("Error occurred extracting relative coordinates")
        X_rel.append(X_rel_current)
    return X_rel

# Nützliche Methoden für die Dihedralwinkelberechnung
def build_graph(tuples_list):
    graph = {}
    for start, end in tuples_list:
        if start not in graph:
            graph[start] = []
        graph[start].append(end)
    return graph


def dfs_with_depth(graph, start_node, depth):
    stack = [(start_node, 0)]
    neighbors_at_depth = []
    while stack:
        current_node, current_depth = stack.pop()
        if current_depth < depth:
            if current_node in graph:
                for neighbor in graph[current_node]:
                    stack.append((neighbor, current_depth + 1))
                    if current_depth + 1 == depth:
                        neighbors_at_depth.append((current_node, neighbor))
    return neighbors_at_depth


def find_neighbors(tuples_list, start_atom, depth):
    graph = build_graph(tuples_list)
    neighbors_dict = {}
    for start, end in tuples_list:
        if start == start_atom:
            neighbors = dfs_with_depth(graph, end, depth - 1)
            neighbors_dict[(start, end)] = neighbors
    return neighbors_dict


if __name__ == "__main__":
    np.random.seed(42)
    mean = 0
    std_dev = 1

    # -------------------------------------------------
    # Beispiel: Datenanalyse / Datenvariabilität
    # -------------------------------------------------

    print("Beispiel für Methoden für für Datenanalyse / Datenvariabilität:")

    # Beispieldaten mit Shape (10, 3, 3) und (10, 3) im Wertebereich [0, 1]
    X_relative_coordinates = np.random.normal(loc=mean, scale=std_dev, size= (10, 5, 3))  # z.b. 10 Samples mit je 5 Nachbaratome mit x, y, z Werten
    y = np.random.normal(loc=mean, scale=std_dev, size=(10, 3))  # z.b. ground truth 10 Hydrogen positions mit x, y, z Werten

    # alle Winkel von: H - CENTRAL_ATOM - REST_ATOME
    all_angles = all_angles_H_central_neighbor(X_relative_coordinates, y)
    print("Durschnittswinkel H - CENTRAL_ATOM - REST_ATOM: ", np.average(all_angles))
    plt.hist(all_angles, bins=1000, density=True, alpha=0.6, color='g'); plt.show()
    plot_fit(all_angles, n_components=1)

    # alle Bindungslängen
    alle_bindungslängen = bindungslängen(y)
    print(alle_bindungslängen[:5])
    print("Durschnitts-Bindungslänge: ", np.average(alle_bindungslängen))
    plt.hist(alle_bindungslängen, bins=1000, density=True, alpha=0.6, color='g'); plt.show()
    plot_fit(alle_bindungslängen, n_components=1)

    # alle Dihedralwinkel nur für EINEN sample
    dihedral_angles = all_dihedral_angles(X_relative_coordinates[0])
    for angle_info in dihedral_angles:
        indices, angle = angle_info
        print(f"Dihedral Angle between points {indices}: {angle:.2f} degrees")

    # alle Dihedralwinkel
    alle_dihedral_winkel = [all_dihedral_angles(coords) for coords in tqdm(X_relative_coordinates)]# there are many dihedral angles per sample!
    alle_dihedral_winkel_filtered = []
    for dw in alle_dihedral_winkel:
        alle_dihedral_winkel_filtered.extend([dw_inner[-1] for dw_inner in dw])
    print(alle_dihedral_winkel_filtered)
    plt.hist(alle_dihedral_winkel_filtered, bins=1000, density=True, alpha=0.6, color='g'); plt.show()

    # -------------------------------------------------
    # Beispiel: Regressionsgüte
    # -------------------------------------------------

    print("Beispiel für Methoden für Regressionsgüte:")

    # Beispieldaten mit Shape (10, 3) im Wertebereich [0, 1]
    y_test = np.random.normal(loc=mean, scale=std_dev, size=(10, 3))  # z.b. ground truth 10 Hydrogen positions mit x, y, z Werten
    y_pred = np.random.normal(loc=mean, scale=std_dev, size=(10, 3))  # z.b. predicted 10 Hydrogen positions mit x, y, z Werten

    # absoluter Abstand H zu H' (MSE Score)
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE score: {mse}')

    # Durchschnittliche Cosine Similarity (kann ignoriert werden) und durchschnittliche Winkel
    avg_cosine_similarity, avg_angle = metric_cosine_similarity_and_angle(y_test, y_pred)
    print("Average Cosine Similarity:", avg_cosine_similarity)
    print("Average Angle (degrees):", avg_angle)

    # Duchschnittliche Bindungslängendifferenz
    avg_bindugslänge_diff = metric_bindungslänge_differenz(y_test, y_pred)
    print("Durschnittliche Bindungslänge Differenz:", avg_bindugslänge_diff)

    # Wasserstein Distanz zwischen zwei Winkelverteilungen
    all_angles_test = all_angles_H_central_neighbor(X_relative_coordinates, y_test)  # alle Winkel im Test set
    all_angles_pred = all_angles_H_central_neighbor(X_relative_coordinates, y_pred)  # alle Winkel durch Prediction
    wasserstein = calculate_wasserstein_distance(all_angles_test, all_angles_pred, bins=180, range=(0, 180))
    print("Die Wasserstein Distanz zwischen den beiden Winkelverteilungen beträgt: ", wasserstein)