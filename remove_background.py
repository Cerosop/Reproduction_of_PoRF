import open3d as o3d
import numpy as np

def remove_bkgd(pts, pcd):
    # 載入兩個 PLY 檔案
    pcd1 = o3d.io.read_point_cloud("C:/Users/hsu/Desktop/project/exp_dtu/scan37_1/dtu_sift_porf/meshes/00050000.ply")
    print(len(pcd1.points))
    pcd2 = o3d.io.read_point_cloud("C:/Users/hsu/Desktop/project/porf_data/dtu/scan37_1/sparse_points_interest.ply")


    # 使用下採樣降低點的密度（選擇性）
    pcd1_down = pcd1.voxel_down_sample(voxel_size=0.01)
    pcd2_down = pcd2.voxel_down_sample(voxel_size=0.01)

    # 計算法向量
    pcd1_down.estimate_normals()
    pcd2_down.estimate_normals()

    # 點雲對齊 (使用 ICP 演算法)
    threshold = 0.02  # 設定配準閾值
    trans_init = np.eye(4)  # 初始化轉換矩陣
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd2_down, pcd1_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # 應用對齊後的變換矩陣到第二個點雲
    pcd2.transform(reg_p2p.transformation)

    # 比較兩個點雲的距離，篩選出物體點雲
    distance = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    # 設定一個距離閾值，篩選出目標物體的點
    mask = distance < 0.25
    filtered_pcd = pcd1.select_by_index(np.where(mask)[0])
    print(len(filtered_pcd.points))
    # 儲存提取出的物體點雲
    o3d.io.write_point_cloud("filtered_object.ply", filtered_pcd)

    # 可視化結果
    o3d.visualization.draw_geometries([filtered_pcd], window_name="Extracted Object")
