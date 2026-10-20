# YAML 环境参数集成与修改说明书

版本：2（2026-10-20，官方四指）；基线：用户提供的 `AIRSEAI-2.0-main.zip`。
适用运行环境沿用项目 README：Ubuntu 22.04、Isaac Sim 5.1、Isaac Lab 2.3.2、Python 3.11。
原压缩包 SHA256 和逐文件差异见交付目录的 `changes.json`、`environment-integration.patch`。

## 1. 实现范围

通过一个 YAML 设置三路 RGB 相机的分辨率、内参、畸变与安装外参，并生成双手四个官方 GelSight R1.5 手指的接触参考帧。
支持单环境 `Harness`、多环境 `make_env` / `SlotHarness`；录像与策略观测使用同一份处理后的图像。
未指定配置时保持原有相机和图像行为。

本功能复现的是 YAML 中定义的相机及接触坐标系。机器人 USD、Home、场景物体、
动力学、控制参数仍由原项目配置决定；这份 YAML 没有这些字段，不能单独复现全部物理环境。
指尖参考帧不改变胶面网格、碰撞体、TacSL taxel/ROI 或 IK TCP。

入口配置：[config/environment.example.yaml](../config/environment.example.yaml)。
它复制自用户指定的 SimHarness 示例，相机数值保持不变；指尖已按用户要求改为官方四指资产胶面中心，替换原 DaiMon/v11 两指候选。
也可以直接传入原 SimHarness YAML；省略 `image_domain` 时，此适配器默认 `raw`。

## 2. 安装与启动

先按原 README 准备 Isaac、机器人 USD 和触觉资产，然后在同一个仿真 Python 环境安装额外依赖：

```bash
python -m pip install -e ".[environment]"
```

新增依赖只有 PyYAML 和 OpenCV headless。运行不需要 RealSense SDK，也不访问真实相机或机器人。
`pyrealsense2` 仅作为离线测试的数值对照。

先做不启动 Isaac 的配置/射线覆盖检查：

```bash
python -m sim.environment config/environment.example.yaml --out /tmp/environment-offline.json
```

该 receipt 的源 K 是计划值，标记为 `offline_only`；不是渲染器读回结果。

在 Isaac 环境中检查裸机器人，并导出三个相机的针孔源图、最终图像和运行 receipt：

```bash
python benchmark/run/launch_env.py \
  --env minimal --environment-yaml config/environment.example.yaml \
  --robot_usd /absolute/path/openarm_tactile.usd \
  --steps 90 --camera-snapshots /tmp/airseai-camera-check
```

多环境示例：

```bash
python benchmark/run/launch_env.py \
  --env blind_peg_insertion --num_envs 2 --view_env 1 \
  --environment-yaml config/environment.example.yaml \
  --steps 90 --camera-snapshots /tmp/airseai-slot1 --report /tmp/airseai-preview.json
```

`--camera-snapshots` 导出选定 slot；`--view_env -1` 时导出 slot 0。
原有 `--snapshot`、`--video` 仍输出第三人称 review 相机；它不是 YAML 中的 base 相机。
GUI 的原生相机视口显示针孔源图，畸变后的图像在程序输出和 `*-output.png` 中查看。

以下入口均支持 `--environment-yaml`：

- `benchmark/run/launch_env.py`
- `benchmark/tasks/run_expert.py`
- `benchmark/eval/evaluate.py`
- `benchmark/teleop/run_teleop.py`（仿真遥操作入口）
- `benchmark/run/env_server.py`

原 ZIP 本来就包含 `src/sim/harness.py` 中的 `Harness`，不是此次新引入的框架。此次为它增加可选 YAML 配置。其他调用公共 Harness / manager 工厂的脚本，以及 Gym 启动的子进程，可使用环境变量：

```bash
export AIRSEAI_ENVIRONMENT_YAML=/absolute/path/environment.yaml
```

CLI 显式值优先于环境变量；传 `--environment-yaml ""` 可关闭继承的配置。
在 Linux 仿真机使用该机器上的路径，不能直接使用 Windows 的 `C:/...`。
Python 调用也可显式选择配置：

```python
h = Harness(scene_cfg, device="cuda:0", environment_yaml="/path/environment.yaml")
# task.make_scene_cfg 等其他参数按原项目提供
h.start()
h.calibration.write_receipt("/tmp/environment-receipt.json")

env = make_env("blind_peg_insertion", num_envs=2, environment_yaml="/path/environment.yaml")
```

## 3. 字段填写规范

只接受顶层 `cameras`、`fingertips`；至少定义一项。未指定的相机保留原设置。
未知字段、重复 YAML key、未知相机名、缺失字段、非法矩阵和畸变模型都会报错。

| 字段 | 单位 / 定义 | 填写方法 |
| --- | --- | --- |
| `cameras.base/left_wrist/right_wrist` | 数据流名称 | 分别绑定原项目的头部、左腕、右腕相机 |
| `intrinsics.width_px/height_px` | 整数像素 | 实际目标图像宽高；在传感器创建前应用，数据集 shape 同步更新 |
| `fx_px/fy_px` | px | 对应同一个分辨率、流类型和图像处理阶段的焦距 |
| `cx_px/cy_px` | px | 主点；左上像素中心为 (0,0)，u 向右、v 向下 |
| `distortion_model` | 模型枚举 | `none`、`opencv_brown_conrady`、`opencv_fisheye`、`realsense_inverse_brown_conrady` |
| `distortion_coeffs` | 无量纲数组 | Brown/inverse Brown 为 `[k1,k2,p1,p2,k3]`；fisheye 为 `[k1,k2,k3,k4]`；none 为 `[]` 或五个零 |
| `image_domain` | `raw` / `ideal` | raw 匹配真实原图；ideal 输出无畸变针孔图，真实输入须另行正确校正 |
| `extrinsics.parent_frame` | URDF/USD link 名 | 当前资产 base 使用 `openarm_body_link0` 别名，腕相机分别使用对应 hand link |
| `extrinsics.T_parent_from_child` | 4×4，平移 m | 父 link 到相机光学系的变换关系，见下文 |
| `fingertips.<name>.parent_frame` | finger link 名 | 必须使用随夹爪开合运动的父 link |
| `fingertips.<name>.T_parent_from_child` | 4×4，平移 m | 父指节到未压缩接触面中心参考系的安装变换 |

缩放/裁剪真实图像后要同步换算 K，不能只改宽高。输出分辨率还应符合所用策略、编码器的输入要求。
目标 K 和源渲染 K 有意不同：源相机需要稍宽视场，重映射后才得到 YAML 所定义的 K 和图像域。

列向量约定：

```text
p_parent = T_parent_from_child @ p_child
T_world_from_camera = T_world_from_parent @ T_parent_from_camera
```

相机 child 是光学系：X 右、Y 下、Z 前，最后一列是光心，不是相机外壳中心。
写 USD 时只转换一次 `T_parent_from_usd_camera = T_parent_from_optical @ diag(1,-1,-1,1)`。
实现移动整套相机 mount，保留相机相对外壳的局部偏移。相机不能声明与实际安装 link 不同的父系，
否则会在运动中失去外参意义，因此适配器拒绝这种配置。

`openarm_body_link0 -> openarm_body_link` 是针对 AIRSEAI 当前资产布局的显式别名；
它不是名称相似即可推断原点重合的通用规则。更换机器人资产时需重新核对该注册关系。
stage 必须使用米。相机及安装节点必须可编辑、无缩放，保留 canonical translate/orient/scale 操作顺序。

## 4. 畸变与图像出口

```text
YAML 目标 K、模型与系数
    → 计算每个目标像素对应的光线
    → 配置覆盖这些光线的针孔源相机
    → 读取每个实例的实际源 K
    → 建立一次逆向采样映射
    → cam_rgb() → 录像、image_batch()、build_obs()、Gym
```

RealSense inverse Brown 使用 librealsense 2.56.5 的十次迭代反投影公式。
不能把同一数组直接当作 OpenCV Brown 系数，也不能通过把系数取负实现逆变换。
映射计算使用 CPU NumPy；RGB 双线性采样使用 CPU OpenCV。多环境按相机、slot 分别保存映射。
策略输入需要传回 GPU，这会增加 CPU/GPU 传输；本次以清晰实现优先，未进行吞吐优化。

原生 USD 镜头畸变必须关闭，避免叠加。模型不支持、射线无效或源视场覆盖不足直接报错。
该映射仅用于 RGB/RGBA，不适用于深度、分割 ID、法向或触觉。
直接读 `camera.data.output["rgb"]`、RenderProduct、原生视口仍是源针孔图；自定义消费者应调用
`h.cam_rgb(name)` / `h.image_batch(name)`，或 `env.calibration.rgb(name, image, slot)`。

## 5. 官方四指接触帧

原 ZIP 自带 `src/sim/harness.py`，以及为左右手各安装两个官方 GelSight R1.5 模组的 `src/sim/assets/build_robot_asset.py`。本配置对应这四个官方模组，不对应自建 DaiMon/W2M 手指。

接触原点按官方 gel visual mesh 的 XZ 包围框中心取一条沿 Y 的直线，与网格求交并选最外侧（最小 Y）交点。X 沿宽度、Y 沿指长、Z 朝夹持间隙。四个局部平移约为 X=±0.0005 mm、Y=±8.058 mm、Z=78.000 mm；以 YAML 中完整矩阵为准。

每项配置在父指节下创建 `calibrated_contact_<name>` Xform，随父 link 运动。
它的 Z 轴指向夹持间隙，X 沿接触面内宽度，Y=Z×X，具体遵照填写的矩阵。
查询当前相对基座的变换：

```python
T_base_from_contact = h.contact_transform("right_hand_right_finger")
# 多环境使用 SlotHarness；输出相对指定 link，环境平移会自然抵消。
```

查询使用实时 articulation body pose 组合局部外参，避免 Fabric 没有同步 USD 时读取到旧位姿。启用 YAML 的相机同时设置 `update_latest_camera_pose=True`，确保运动后读出的 optical pose 不停留在初始化缓存。
不把示例注释里的 Home 位姿下 base 矩阵作为固定安装参数。
接触参考帧和实际胶面不一致时，需要另行标定并修改真实传感器/资产，不会自动“校正”物理接触。

## 6. 修改位置

| 文件 | 改动 |
| --- | --- |
| `src/sim/environment.py` | YAML 校验、局部 USD 安装、指尖帧、实际 K 读回、receipt、离线检查入口 |
| `src/sim/camera_model.py` | 独立于 Isaac 的相机射线和 RGB 重映射数学 |
| `src/sim/harness.py` | 初始化消费配置；统一录像和策略图像；暴露动态接触矩阵 |
| `src/sim/manager_based/common/env_cfg.py` | 传感器分辨率和 prestartup 外参事件 |
| `src/sim/manager_based/__init__.py` | 初始化所有实例的图像映射 |
| `src/sim/manager_based/common/slot.py` | slot 共享配置并选择对应映射 |
| `src/sim/recording.py` | 根据实际相机尺寸声明数据集；保存配置 receipt |
| `src/sim/collection.py` | 运行输出目录保存 receipt |
| 上述五个 CLI | 增加一个可选 YAML 参数；preview 增加相机对照图导出 |
| `pyproject.toml` | `environment` 可选依赖，不增加 Isaac 或硬件依赖 |
| `benchmark/run/verify_environment.py` | 双姿态 RTX 检阅、四指轴标记、实际相机外参与 FK 核对、录像/策略图像一致性 |
| `tests/test_environment.py` | CPU 数值、USD 层级、错误拒绝、录像/策略一致性测试 |

数据集 `meta/environment-receipt.json` 和 vector collection 输出目录会保存 receipt。
preview 对照图目录包含独立 receipt，manager 的 `--report` 也包含它。
自定义脚本/单环境评估可调用 `h.calibration.write_receipt(...)` 显式保存。
记录包括原 YAML SHA256、完整配置、实际 prim、每 slot 源/目标 K、畸变实现、映射哈希和覆盖率。
源 USD 只在当前 stage 上覆写，不调用 Save/Export 修改原资产。

## 7. 参数来源、验证与限制

- 三路彩色 K 和畸变来自历史 RealSense 设备内参记录，适用原始 640×480 彩色流；
  base 零系数是设备对该流的报告，不代表物理镜头完美。
- 示例相机外参为 2026-10-15 历史候选；四个指尖来自官方 GelSight R1.5 USD 胶面网格中心交点，非实体标定值。推导数值和资产 SHA 见 [official-contact-frames.json](official-contact-frames.json)。
- 代码按 Isaac Lab 2.3.2 的相机 API 和 prestartup 生命周期接入。官方参考：
  [Camera 源码](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/sensors/camera/camera.py)、
  [ManagerBasedEnv](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/envs/manager_based_env.py)、
  [RealSense 反投影实现](https://github.com/realsenseai/librealsense/blob/v2.56.5/src/rs.cpp)。
- 本次离线环境为 Windows / Python 3.12，测试使用真实 USD Core 和 RealSense SDK，
  Isaac 张量接口使用测试替身。测试结果见 `environment-validation.json`。
- Windows 仍只执行离线测试；另在 4090 server 的独立目录执行实际 Isaac Kit / RTX 单环境双姿态验证。Home 和双臂 joint1 增加 0.12 rad 的姿态均通过：三相机实际 optical pose 与父 link 实时 FK 组合矩阵最大元素差小于 5e-7；六幅策略图像与录像出口逐像素相同。多环境 RTX 尚未执行，不将 CPU 多实例测试当作物理验收。
- 原 ZIP 不含外部机器人/触觉资产。本次使用服务器原生 `openarm_tactile.usd` 的四个官方 GelSight R1.5 模组，未载入 DaiMon/W2M 替换体；资产几何和真实标定仍需区分。

离线验证共 26 项通过，Ruff 检查和 Python 3.11 语法检查通过；包含三路 SDK 对照、
两个 USD 实例的父子变换与随动、不同分辨率、四种畸变模型以及录像/策略出口一致性。
复跑测试可安装 `.[environment,dev]`，另装 `usd-core==25.11` 和
`pyrealsense2==2.56.5.9235`，然后执行 `python -m pytest tests/test_environment.py -q`。
不安装两个对照依赖时，对应测试会跳过，不能将跳过计作通过。

运行验收应核对三路 `*-output.png`、receipt 的 K/覆盖率、机器人运动后的相机随动，以及指尖 FK。
还应在真实标定观测上检查重投影误差；数值公式吻合 SDK 不等于实体环境对齐通过。
不修改历史外参来掩盖投影模型差异，也未连接任何真实硬件。

## 8. 4090 可视化复跑

使用官方 `openarm_tactile.usd`，先执行全景和相机核验，再根据第一遍的实际手指位置创建固定近景相机：

```bash
python benchmark/run/verify_environment.py --headless --enable_cameras \
  --robot_usd /absolute/path/openarm_tactile.usd \
  --environment-yaml config/environment.example.yaml --out /tmp/official-pass1
python benchmark/run/verify_environment.py --headless --enable_cameras \
  --robot_usd /absolute/path/openarm_tactile.usd \
  --environment-yaml config/environment.example.yaml \
  --reference-frames /tmp/official-pass1/render-validation.json --out /tmp/official-pass2
```

两遍必须使用相同资产、配置和初态。近景相机在 physics 启动前固定创建；它们不属于三路标定相机。诊断图中黄色球为原点，红/绿/蓝分别为 X/Y/Z 正轴；导出三路策略图像时隐藏这些标记。测试为便于观察将两手各指初始模型位移设为 0.018 m，运动后为 0.025 m，保存在运行关节记录中；不是实机 Home 标定。

每个输出目录保存全景、可选近景、其实际 K/optical pose 和四指 FK、三路源图与畸变输出、`render-validation.json`。运行结果另附 [official-render-validation.json](official-render-validation.json)。独立检查场景使用最小占位物体及官方默认光照，不代表复原桌面/孔板等任务场景。

## 变更记录

| 日期 | 版本 | 内容 |
| --- | --- | --- |
| 2026-10-20 | 1 | 基于用户 ZIP，加入可选 YAML 相机/接触参考帧适配、共享 RGB 出口、离线验证与修改说明 |

| 2026-10-20 | 2 | 替换为官方四指胶面参考帧；4090 单环境双姿态 RTX 检阅；更新运动相机位姿缓存及可复跑检查入口 |
