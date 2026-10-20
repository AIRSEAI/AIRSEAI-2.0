# 最小 YAML / Isaac 原生 LUT 集成说明

版本：3，2026-10-20。基线为用户原始 `AIRSEAI-2.0-main.zip`，SHA256：`149d7a2d562dd17e7c54b8c0f96f00a063a68596d11255ca19c240fa9df2a5a8`。

## 1. 结构与修改范围

原项目已有 Harness。本版本从原 ZIP 重新建立，只新增独立文件，原项目全部 187 个已有文件逐字节保持一致。它替代此前覆盖训练、评估、采集、录像和 manager 的广泛集成；那些外围改动没有进入此包。

```text
config/environment.example.yaml
       ↓ 加载、校验、生成 LUT（初始化一次）
src/sim/yaml_scene.py
       ↓ 应用 USD 安装外参、四指参考帧和 RTX 镜头属性
benchmark/run/yaml_scene.py
       ↓ 原 Harness + 指定尺寸的 TiledCamera
Isaac RTX 原生 RGB → 原 cam_rgb()/image_batch() 或直接读取传感器
```

运行实现只有两个新增 Python 文件：加载与应用模块、启动入口。另附 YAML、额外依赖清单、离线测试、本文及验证/几何来源记录。没有修改原 Harness、manager、录像、数据 shape、其他 CLI、Gym 或 `pyproject.toml`；也没有全局环境变量自动启用行为。

本版本不存在 `Calibration.rgb()`、`cv2.remap()`、源针孔图二次采样、录像专用补丁或每帧 CPU 畸变处理。新入口创建原 Harness 时用已有的 `cameras=False`，随后把配置好的相机加入其公开的 `cams` 字典。原有读取方法直接获得 RTX 最终图像。

范围是三相机成像/安装和四指接触参考帧。机器人 USD、关节初态、世界放置、场景物体、光照、动力学仍由原项目提供；一份相机 YAML 不等于完整任务场景重建。四指参考帧不修改胶面、碰撞体、TacSL ROI 或 IK TCP。

## 2. 安装与使用

使用已配置的 Ubuntu / Isaac Sim 5.1.0 / Isaac Lab v2.3.2 / Python 3.11 环境，并准备原项目官方 `openarm_tactile.usd` 及其外部引用。原 ZIP 不包含这些 USD 依赖，本次代码包也不复制它们。

```bash
python -m pip install -e .
python -m pip install -r requirements-environment.txt
python benchmark/run/yaml_scene.py \
  --environment-yaml config/environment.example.yaml \
  --robot_usd /absolute/path/openarm_tactile.usd \
  --out /tmp/airseai-native-lut \
  --steps 90 --headless --enable_cameras
```

新增运行依赖为 PyYAML 和 OpenEXR 3.3.5，NumPy 使用原 Isaac 环境。OpenEXR 只在初始化时写入 float32 LUT 纹理；运行不需要 RealSense SDK，也不连接设备。服务器验证采用隔离依赖目录，没有覆盖已有环境包。

输出包含 `base.png`、`left_wrist.png`、`right_wrist.png`、`environment-receipt.json` 和 `lut/*.exr`。图片为传感器直接输出，已有畸变；不要再执行旧版 RGB 后处理。EXR 是自动推导结果，不需要手填或单独标定。receipt 保存 YAML 哈希、输入参数、实际安装 prim、两张 LUT 路径/哈希及所用后端。

其他原项目入口保持原行为，不会自动读取这份 YAML。若以后需要接入另一个入口，应在创建相机/physics reset 前显式调用 `apply_environment(stage, robot_root, spec, lut_directory)`，并按 YAML 尺寸创建相机；无需给每个图像消费者再写畸变代码。

## 3. YAML 参数填写规范

[完整示例](../config/environment.example.yaml) 只有 `cameras` 和 `fingertips`。未知字段、重复键、非法矩阵或不支持的镜头模型会报错。

| 字段 | 单位与标准 |
| --- | --- |
| `cameras.base/left_wrist/right_wrist` | 主、左腕、右腕 RGB，相应 prim 绑定在加载模块的 `CAMERA_PATHS` 中 |
| `intrinsics.width_px/height_px` | 目标图像宽高，正整数像素；创建传感器前确定 |
| `fx_px/fy_px/cx_px/cy_px` | 同一流、同一分辨率的焦距/主点，单位 px；左上像素中心 `(0,0)`，u 右、v 下 |
| `distortion_model` | 精简版支持 `realsense_inverse_brown_conrady` 或 `none` |
| `distortion_coeffs` | inverse Brown 为 `[k1,k2,p1,p2,k3]`，无量纲；none 为 `[]` 或五个零 |
| `image_domain` | `raw` 默认：保留声明畸变；`ideal`：生成针孔射线，真实图须另行正确校正 |
| 相机 `extrinsics.parent_frame` | 安装父 link：base 使用显式别名 `openarm_body_link0` → `openarm_body_link`；腕相机为各自 hand |
| 相机 `T_parent_from_child` | 4×4 刚体矩阵；平移 m，child 为光学系 X 右/Y 下/Z 前；最后一列为光心 |
| 指尖 `parent_frame` | 必须是随开合运动的对应 finger link |
| 指尖 `T_parent_from_child` | 接触中心参考系到 finger link，平移 m，旋转右手正交，末行 `[0,0,0,1]` |

列向量约定：`p_parent = T_parent_from_child @ p_child`。旋转矩阵三列为 child XYZ 在 parent 中的方向；`R.T @ R = I`、`det(R)=+1`。不填写欧拉角，不对平移单独取负来代替矩阵求逆。

相机输入没有预乘 optical→USD 换基；加载时只乘一次 `diag(1,-1,-1,1)`。适配器移动相机 mount，保留外壳到镜头的局部偏移。腕相机必须声明其实际安装父 link。stage 单位必须为米，安装变换不能含缩放。

修改分辨率或裁剪后必须同步更新 K，不能只改宽高。K 的主点已经烘入 LUT；渲染器中心设为图像正中，避免重复应用偏移。Isaac Lab 的 `camera.data.intrinsic_matrices` 是常规针孔元数据，不足以表达原生 LUT；投影/反投影应依据 YAML 和对应 SDK 模型，不能用这个缓存矩阵替代完整镜头模型。

四指名为 `right_hand_right_finger`、`right_hand_left_finger`、`left_hand_right_finger`、`left_hand_left_finger`，分别属于两手各自的右/左指，不依画面左右命名。官方 GelSight R1.5 胶面中心按 visual mesh 的 XZ 包围框中心线与前表面最小 Y 交点定义；接触 X 沿宽度、Y 沿指长、Z 朝夹持间隙。局部平移约 X=±0.0005 mm、Y=±8.058 mm、Z=78.000 mm，详见完整矩阵及 [源资产推导记录](official-contact-frames.json)。动态 base/world 变换由当前父指节 FK 组合，不作为另一项固定安装参数写入 YAML。

相机 K/畸变沿用历史 RealSense 彩色流记录，外参仍为 2026-10-15 候选；四指值是资产几何推导，未经实体标定。不要将这组示例视为最新实物安装的验收值。

## 4. 原生 LUT 如何工作

初始化时按 librealsense 2.56.5 的十轮 inverse Brown 反投影计算每个像素的射线，生成两张 float32 EXR：

1. 方向表：分辨率为输出的 2 倍，像素 → USD 相机射线 `(x,-y,-1)`，归一化后交给 RTX。
2. 位置表：2048×2048，八面体编码的射线方向 → 图像归一化坐标。Isaac 5.1 generalizedProjection 直接编码 USD 相机的 -Z 方向，不采用新版 LutAPI 文档的 Z 翻转；该约定已用运动向量验证。

像素中心坐标与纹理 texel 中心显式转换，主点只计入一次。LUT 有离散插值误差，不声称是无限精度解析镜头。当前是前向 RealSense 镜头范围；过宽视场或不收敛参数直接拒绝。没有使用普通 Brown 系数替换，也没有重新拟合成另一组近似参数。

**Isaac Sim 5.1 实测兼容细节：** 该安装的 `OmniLensDistortionLutAPI` 可创建属性，但实际图像仍按默认针孔投影；原生 OpenCV 对照则生效。最终使用 RTX 已实现的 `cameraProjectionType="generalizedProjection"`，通过 `generalizedProjectionDirectionTexturePath` 和 `generalizedProjectionNDCTexturePath` 绑定同两张 LUT，并设置 `fthetaWidth/Height/Cx/Cy/MaxFov`。这是 RTX 原生镜头采样，不是渲染后 CPU 处理。属性赋值集中在 `apply_lut()`；升级 Isaac 时应重新验证此绑定，不假定新旧 schema 等效。

已有源资产若启用了其他镜头畸变则报错，防止叠加。运行不保存/覆盖源 USD，只对当前 stage 写入覆盖。支持 RGB 的验证范围，不将普通针孔深度反投影、分割或六维力标定混称为已验收。

官方参考：[Replicator generalizedProjection 接口](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.12.27/source/extensions/omni.replicator.core/docs/API.html)、[RTX LUT 纹理约定](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html)。实现兼容性以本次服务器验证为准。

## 5. 验证与复跑

4090 上的独立 RealSense SDK 标靶验证：三路每路 63 点、共 189 点，覆盖画面中心与边缘。实际 `TiledCamera` 原生 LUT 最大像素误差为 base 0.2820、左腕 0.3147、右腕 0.3051；RMS 均小于 0.20 px。测试阈值预设为 max <0.65 px、RMS <0.35 px。小球中心检测包含渲染采样误差，不等于实物标定精度。

相同目标关闭畸变后的两腕最大误差为 5.6012/6.2729 px，验证了 LUT 不是被忽略；base 原设备系数全零，对照自然接近。另完成整机两姿态、四个官方接触帧和三路 native RGB 检阅，原 Harness 的 `image_batch()` 与 `cam_rgb()` 输出相同。全部验收摘要见 [validation.json](validation.json)。

另验证反向 LUT：目标连续移动 0.005 m/帧，三相机原生运动向量相对 SDK 理论像素位移的最大误差均小于 0.01 px/帧。在目标周围 9×9 区域筛选非零运动表面样本，取中位数；这不等于全图稠密光流验收。CPU 反向表插值在内部小于 0.01 px，八面体边界夹取小于 0.35 px。

离线 9 项测试使用真实 USD Core、RealSense SDK、OpenEXR，验证双向 LUT 插值、无效视场拒绝、文件通道精度、YAML 校验和四指父系随动。运行方式：

```bash
python -m pip install pytest usd-core==25.11 pyrealsense2==2.56.5.9235
python -m pytest tests/test_yaml_scene.py -q
```

上述额外包仅用于离线测试，建议独立环境安装；不要把另一个 USD Core 安装进 Isaac 的 Python。缺少对照依赖时相关测试会跳过，不能把跳过算作通过。RTX 标靶及双姿态脚本和原图随独立验证资料包保存；正常使用只需第 2 节的单一入口。

未执行真机操作。没有证明候选相机外参已与实物对齐，也没有复现桌面/孔板等完整任务场景。多环境、训练和录像的自动 YAML 集成已移除出本版本范围。

## 变更记录

| 日期 | 版本 | 内容 |
| --- | --- | --- |
| 2026-10-20 | 3 | 从原 ZIP 重建最小独立入口；采用已验证的 RTX 原生 LUT，移除上一集成的图像后处理和外围接入；保留官方四指、数值和实际渲染证据 |
