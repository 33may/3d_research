#!/usr/bin/env python3
"""
create_and_show_splats_filament.py

Генерация N искусственных 3D Gaussian Splats
и их немедленная визуализация с Filament-шейдером "3dgs".
"""

import numpy as np
import open3d as o3d
from open3d.visualization.rendering import (
    OffscreenRenderer,
    TGaussianSplatBuffersBuilder,
    MaterialRecord
)

def create_splats(N=50):
    tpc = o3d.t.geometry.PointCloud()
    # 1) Позиции (N×3)
    tpc.point["positions"] = o3d.core.Tensor(
        np.random.uniform(-1, 1, (N, 3)).astype(np.float32))
    # 2) Прозрачность (N×1)
    tpc.point["opacity"] = o3d.core.Tensor(
        np.linspace(0.2,1.0,N,dtype=np.float32).reshape(N,1))
    # 3) Поворот (N×4)
    tpc.point["rot"]   = o3d.core.Tensor(
        np.tile([1,0,0,0], (N,1)).astype(np.float32))
    # 4) Масштаб (N×3)
    tpc.point["scale"] = o3d.core.Tensor(
        np.full((N,3), 0.1, dtype=np.float32))
    # 5) Цвета DC (N×3)
    tpc.point["f_dc"]  = o3d.core.Tensor(
        np.vstack([np.linspace(0,1,N),
                   np.zeros(N),
                   np.linspace(1,0,N)]).T.astype(np.float32))
    # 6) SH-коэффициенты (N×3×3)
    tpc.point["f_rest"] = o3d.core.Tensor(
        np.zeros((N,3,3), dtype=np.float32))
    return tpc

def main():
    tpc = create_splats(200)
    print(f"Создано сплэтов: {tpc.point['positions'].shape[0]}")

    # 1) Инициализируем рендерер
    renderer = OffscreenRenderer(800,600)
    scene    = renderer.scene

    # 2) Строим GPU-буферы сплэтов
    builder = TGaussianSplatBuffersBuilder(tpc)  # :contentReference[oaicite:2]{index=2}
    buffers = builder.ConstructBuffers()

    # 3) Задаём материал "3dgs"
    mat = MaterialRecord()
    mat.shader = "3dgs"                          # :contentReference[oaicite:3]{index=3}

    # 4) Добавляем и рендерим
    scene.add_geometry("gaussian_splats", buffers, mat)
    img = renderer.render_to_image()
    o3d.io.write_image("out.png", img)
    print("Сохранено изображение out.png")

if __name__ == "__main__":
    main()
