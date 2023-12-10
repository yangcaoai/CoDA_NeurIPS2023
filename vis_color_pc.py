##tsne
import os.path

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from open3d import geometry
def tsne_viw(X_, Y_):
    """

    Args:
        X_: List[N, f], features
        Y_: List[N,], labels
        '''
        for i, data in enumerate(dataloader):
            pc, _, targets, texts = map(lambda x: x.cuda(), data[:4])
            pc_features = model.encode_pc(pc)
            pc_features = model.loss.process_pc_features(pc_features)

            X_.append(pc_features.cpu().numpy())
            Y_.append(targets.cpu().numpy())
        '''
    Returns:

    """
    tsne = TSNE(n_components=2, random_state=0, early_exaggeration=50)
    X_ = np.concatenate(X_, axis=0)
    Y_ = np.concatenate(Y_, axis=0)
    # Y_ = [i for i in range(10)]
    X_tsne = tsne.fit_transform(X_)

    # Another algorithm as pca
    # pca1 = PCA(n_components=2)
    # pca1.fit(X_)
    # X_tsne = pca1.transform(X_)

    # colors = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])
    colors = np.array([0, 5, 10, 15, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap='Spectral')
    plt.show()

def color_point(points, point_color=None, mode='xyzrbg', basename=None, out_path=None):
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    pcd = geometry.PointCloud()
    if mode == 'xyz':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = np.tile(np.array(point_color), (points.shape[0], 1))
    elif mode == 'xyzrgb':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = points[:, 3:6]
        # normalize to [0, 1] for open3d drawing
        if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
            points_colors /= 255.0
    else:
        raise NotImplementedError
    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    o3d.io.write_point_cloud(out_path+ '/%s_color_pc.ply' % basename, pcd)
    # vis.add_geometry(pcd)
    # vis.run()
    # vis.destroy_window()

def read_pc_data(path, out_path=None):
    pc_path = path + '_pc.npz'
    basename = os.path.basename(pc_path)[:-7]
    # box_path = path + '_bbox.npy'
    pc_data = np.load(pc_path)["pc"]
    # box_data = np.load(box_path)
    color_point(pc_data, mode='xyzrgb', basename=basename, out_path=out_path)




# read_pc_data('Data/3d_indoor/sunrgb_d/sunrgbd_v1_0415/sunrgbd_pc_bbox_votes_50k_v1_all_classes_0415_val/000080')