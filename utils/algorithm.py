import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances




def inter_class_distance(np_class1: np.ndarray, np_class2: np.ndarray, method:str = "euclidean"):
    """
    Calculate the inter-class distance between two classes.

    Args:
        class1 (numpy.ndarray): Class 1.
        class2 (numpy.ndarray): Class 2.
        method (str, optional): Distance metric. Defaults to "euclidean".
    Returns:
        float: Inter-class distance.
    """
    
    if np_class1.shape[1] != np_class2.shape[1]:
        raise ValueError("Class dimensions must be equal")
    
    
    distance = 0
    
    if method == "euclidean":
        for i in range(np_class1.shape[0]):
            for j in range(np_class2.shape[0]):
                distance += np.linalg.norm(np_class1[i] - np_class2[j])
    elif method == "cosine":
        for i in range(np_class1.shape[0]):
            for j in range(np_class2.shape[0]):
                distance += 1 - np.dot(np_class1[i], np_class2[j]) / (np.linalg.norm(np_class1[i]) * np.linalg.norm(np_class2[j]))    
    else:
        raise ValueError("Invalid method")
    
    distance = distance / (np_class1.shape[0] * np_class2.shape[0])
    return distance

def intra_class_distance(np_class:np.ndarray, method:str = "euclidean"):
    """
    Calculate the intra-class distance of a class.

    Args:
        class (numpy.ndarray): Class.
        method (str, optional): Distance metric. Defaults to "euclidean". options: "euclidean", "cosine"
    Returns:
        float: Intra-class distance.
    """
    distance = 0
    
    if method == "euclidean":
        for i in range(np_class.shape[0]):
            for j in range(np_class.shape[0]):
                if i == j:
                    continue
                distance += np.linalg.norm(np_class[i] - np_class[j])
    elif method == "cosine":
        for i in range(np_class.shape[0]):
            for j in range(np_class.shape[0]):
                if i == j:
                    continue
                distance += 1 - np.dot(np_class[i], np_class[j]) / (np.linalg.norm(np_class[i]) * np.linalg.norm(np_class[j]))
    else:
        raise ValueError("Invalid method")
    distance = distance / (np_class.shape[0] * (np_class.shape[0] - 1))
    return distance

def cohens_d(np_class1: np.ndarray, np_class2: np.ndarray):
    """
    Calculate the Cohen's d between two classes.

    Args:
        class1 (numpy.ndarray): Class 1.
        class2 (numpy.ndarray): Class 2.

    Returns:
        float: Cohen's d.
    """
    if np_class1.shape[0] != np_class2.shape[0]:
        raise ValueError("Class dimensions must be equal")
    
    class1_mean = np.mean(np_class1, axis=0)
    class2_mean = np.mean(np_class2, axis=0)
    print("class1_mean", class1_mean)
    print("class2_mean", class2_mean)
    
    class1_std = np.std(np_class1, axis=0)
    class2_std = np.std(np_class2, axis=0)
    print("class1_std", class1_std)
    print("class2_std", class2_std)
    
    spool = np.sqrt(((np_class1.shape[0] - 1) *np.square(class1_std) + (np_class2.shape[0] -1 ) * np.square(class2_std)) / (np_class1.shape[0] + np_class2.shape[0] - 2))
    
    d = np.abs(class1_mean - class2_mean) / spool    
    return d

def visualize_inter_intra_class_distances(inter_class_distances:list, intra_class_distances:list, keywords : list = None):
    """
    Visualize inter-class and intra-class distances.

    Args:
        inter_class_list (list): List of inter-class distances for a classes.
        intra_class_list (list): List of intra-class distances for a classes.
        keyword (str): Keyword.
    Returns:
        None
    """
    cluster_indices = range(1, len(inter_class_distances) + 1)

    plt.figure(figsize=(10, 6))
    
    num_clusters = len(inter_class_distances)  # Adjust the number of clusters as needed
    colors = np.array(plt.cm.tab20.colors)  # Convert to NumPy array

    # Create a custom color map
    if num_clusters > len(colors):
        colors = np.concatenate([colors, plt.cm.tab20b.colors])
    cmap = ListedColormap(colors[:num_clusters])

    handles = []  # Collect handles for legend

    # Plot inter-class variance for each cluster
    if (type(inter_class_distances[0]) == list):
        for i, inter_class_var in enumerate(inter_class_distances):
            #handle = []
            for j, var in enumerate(inter_class_var):
                color = cmap(j)
                handles.append(plt.scatter(i + 1, var, color=color, marker='x',s=15))
    else:
        for i, icd in enumerate(inter_class_distances):
            color = cmap(i)
            handle = plt.scatter(i + 1, icd, color=color, marker='x',s=25)
            handles.append(handle)
        
    # Plot intra-class variance
    for i, icd in enumerate(intra_class_distances):
        color = cmap(i)
        handle = plt.scatter(i + 1, icd, color=color, marker='^',s=25)
        handles.append(handle)

    # Mark centroids
    for i in range(num_clusters):
        color = cmap(i)
        handle = plt.scatter(i + 1, 0, marker='o', color=color, s=20)
        handles.append(handle)

    # Create a legend with handles
    legend_elements = [
        Line2D([0], [0], marker='o', color='b', markerfacecolor='black', markersize=10, label='Centroid'),
        Line2D([0], [0], marker='x', color='b', markerfacecolor='black', markersize=10, label='Inter-Class Distance'),
        Line2D([0], [0], marker='^', color='b', markerfacecolor='black', markersize=10, label='Intra-Class Distance')
    ]

    # Add legend with handles
    plt.legend(handles=handles + legend_elements, loc='upper right')

    plt.xlabel('Classes')
    plt.ylabel('Distance')
    
    str_keywords = ""
    if keywords is not None:
        str_keywords = " ".join(keywords)
    plt.title(f'Inter-Class and Intra-Class Distance for {str_keywords}')
    plt.grid(True)
    plt.tight_layout()  # Improve layout

    return plt


