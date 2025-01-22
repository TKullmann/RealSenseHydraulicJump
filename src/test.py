import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
o3d.visualization.draw_geometries([pcd])
