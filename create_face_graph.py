import cv2 as cv
import numpy as np
import os
import hdbscan
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data_path = '/home/alexxx/Downloads/crop_part1/'
path_db = 'kumskov/res_dbscan/'
path_fp = 'kumskov/res_extract_facepoints/'
path_init = 'kumskov/init/'
path_graph = 'kumskov/res_graph/'


def find_filepaths(directory: str) -> list:
    res_data = []
    for filename in os.listdir(data_path):
        if filename.endswith("jpg"):
            filepath = os.path.join(data_path, filename)
            res_data.append(filepath)
    return res_data


def fit_hdbscan(points_: list):
    points_ = StandardScaler().fit_transform(points_)
    hdb.fit(points_)
    hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                    gen_min_span_tree=True, leaf_size=40, metric='euclidean',
                    min_cluster_size=15, min_samples=None, p=None)


if __name__ == '__main__':

    data = find_filepaths(data_path)

    for i in range(0, 1000):
        img = cv.imread(data[i])
        path = path_init + str(i) + ".jpg"
        print(path)
        cv.imwrite(path, img)

        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_img = np.float32(gray_img)
        res_img = cv.cornerHarris(gray_img, 2, 3, None)
        res_img = cv.dilate(res_img, None)
        img[res_img > 0.005 * res_img.max()] = [0, 0, 255]

        path = path_fp + str(i) + ".jpg"
        cv.imwrite(path, img)

        points = []
        points_graph = []
        for x in range(0, img.shape[0]):
            for y in range(0, img.shape[1]):
                if img[y, x, 0] == 0 and img[y, x, 1] == 0 and img[y, x, 2] == 255:
                    points_graph.append((x, -y))
                    points.append((y, x))

        img_db = np.array(points)

        hdb = hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=True)

        fit_hdbscan(points)
        labels = hdb.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        gray_3ch = cv.merge((gray_img, gray_img, gray_img))
        unique_labels = set(labels)
        colors = [plt.cm.inferno_r(i / float(len(unique_labels))) for i in range(len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            class_member_mask = labels == k

            xy = img_db[class_member_mask]
            if k != -1:
                for j in range(0, len(xy)):
                    gray_3ch[xy[j, 0], xy[j, 1], :] = [int(255 * col[0]) + 20, int(255 * col[1]) + 30, int(255 * col[2]) + 50]
            else:
                for j in range(0, len(xy)):
                    gray_3ch[xy[j, 0], xy[j, 1], :] =  [0, 0, 255]  # red used for noise

        path = path_db + str(i) + "_" + str(n_clusters) + "cl" + ".jpg"
        cv.imwrite(path, gray_3ch)

        fit_hdbscan(points_graph)
        hdb.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=80,
                                        edge_linewidth=2, colorbar=False)
        path = path_graph + str(i) + ".jpg"
        plt.savefig(path)
        plt.close()

    print("Done extracting face points. Go to /fp folder")
    print("Done dbscan classification. Go to /db folder")
    print("Done building a graph. Go to /graphs folder")
