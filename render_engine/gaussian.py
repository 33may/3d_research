import numpy as np
import open3d as o3d

# 1. Параметры «шара»
center = np.array([0.0, 0.0, 0.0])   # центр гауссиана
sigma  = 0.05                         # дисперсия (чем меньше, тем плотнее облако)
num_pts = 50_000                      # сколько точек хотим «набрызгать»

# 2. Генерируем координаты X,Y,Z ~ N(center, sigma^2)
xyz = np.random.normal(loc=center, scale=sigma, size=(num_pts, 3))

# 3. Цвета (можно задать радиус-зависимый grad или что угодно)
colors = np.repeat([[1.0, 0.4, 0.0]], repeats=num_pts, axis=0)  # оранжевый

# 4. Собираем point-cloud
pcd = o3d.geometry.PointCloud()
pcd.points  = o3d.utility.Vector3dVector(xyz)
pcd.colors  = o3d.utility.Vector3dVector(colors)

# 5. Визуализация
o3d.visualization.draw_geometries([pcd], window_name='Single Gaussian')
