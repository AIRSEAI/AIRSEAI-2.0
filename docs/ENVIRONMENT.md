# YAML 公共初始化集成说明

版本：4，2026-10-21。基线为用户原始 AIRSEAI-2.0-main.zip；SHA256 `149d7a2d562dd17e7c54b8c0f96f00a063a68596d11255ca19c240fa9df2a5a8`。本版取代 v3 独立预览入口的接入方式；原生 LUT 数学模型不变。

## 1. 默认启动行为与修改范围

**原项目启动现在默认强制加载项目内 `config/environment.example.yaml`。** 路径以源码所在项目定位，不依赖工作目录。也可用 `AIRSEAI_ENVIRONMENT_YAML` 指定另一份文件；空路径、文件缺失、未知字段、三相机/四指不完整、矩阵非法、USD prim 不匹配或 LUT 写入失败都会抛错，停止当前初始化，不回退到原资产参数。

只修改五个原文件，其余原文件保留：

| 原文件 | 必要修改 |
| --- | --- |
| `src/sim/harness.py` | 构造时加载 YAML，场景创建后应用，首次 reset 前完成；三相机尺寸来自 YAML；默认触觉类复用原项目已有的 Isaac5.1 dtype 适配 |
| `src/sim/manager_based/common/env_cfg.py` | 公共配置加载一次 YAML；用 prestartup 事件在所有环境克隆完成后、首次 reset 前应用；三路 TiledCamera 按 YAML 配置 |
| `benchmark/run/launch_env.py` | 旧 minimal 预览绕过以上两个公共构造器，因此补一个相同加载/应用调用 |
| `benchmark/run/env_server.py` | 启动后先初始化触觉基准，保证 Gym 首次 meta 请求可读取触觉观测；原始 ZIP 也能复现缺失初始化的错误 |
| `src/sim/recording.py` | 录像元数据读取相机配置尺寸，避免 YAML 改分辨率后仍声明640×480；不改图像 |

新增运行实现仍集中于 `src/sim/yaml_scene.py`，包含加载校验、安装矩阵、原生 LUT 和公共启动适配。新增 `benchmark/run/yaml_scene.py` 只保留便捷的单环境图片导出，已改为复用 Harness 的公共接入，避免重复畸变。manager 工厂、Gym客户端、训练、评估、采集、动作、碰撞、控制和图像读取逻辑没有各自再加畸变补丁。

```text
原 Harness / manager 公共配置 / 旧 minimal 预览
  → 加载同一 YAML 快照
  → 场景与克隆建立 → 所有实例应用相机外参、四指帧、原生 LUT
  → 写 receipt → 首次 reset / 相机初始化 → 原控制、采集、录像、策略流程
```

默认启用意味着相机图像会按 YAML 改变。保留的是原有接口、动作/状态、任务和数据流程，不承诺既有策略在新图像域下的成功率。源 USD 不保存，四指仅创建参考帧，不修改胶面、碰撞、TacSL ROI 或 IK TCP。

## 2. 安装、原入口使用与核实加载

使用已有 Ubuntu / Isaac Sim 5.1.0 / Isaac Lab v2.3.2 / Python3.11 环境，按原项目准备官方 openarm_tactile.usd 及外部引用。代码包不复制这些资产。以可编辑方式安装此交付项目，避免仍导入另一份旧 checkout：

```bash
python -m pip install -e .
python -m pip install -r requirements-environment.txt
# 不需要新增参数：默认加载本项目 config/environment.example.yaml
python benchmark/run/launch_env.py --env minimal \
  --robot_usd /absolute/path/openarm_tactile.usd --snapshot /tmp/minimal.png
python benchmark/run/launch_env.py --env stack_cube --num_envs 2 \
  --robot_usd /absolute/path/openarm_tactile.usd --snapshot /tmp/two-envs.png
```

为当前命令指定校准文件和输出目录，原命令其他参数保持使用方式：

```bash
AIRSEAI_ENVIRONMENT_YAML=/absolute/path/environment.yaml \
AIRSEAI_ENVIRONMENT_OUTPUT=/tmp/calibration-receipts \
python benchmark/run/launch_env.py --env stack_cube --num_envs 2 \
  --robot_usd /absolute/path/openarm_tactile.usd --snapshot /tmp/two-envs.png
```

这两个变量由原 Gym 子进程正常继承。无需改 Gym 协议或给每个 CLI 增加选项。直接使用 Harness 时还可传 `environment_yaml=...`，优先于环境变量；其他公共入口采用环境变量或默认文件。文件选择优先级：显式 Harness 参数 > 环境变量 > 项目内默认文件。解析/应用错误始终抛出，不自动回退。依赖未安装也会明确失败。

成功时标准输出包含 `[environment] APPLIED ... sha256=... num_envs=... receipt=...`。默认结果目录为启动工作目录的 `outputs/environment/startup-*`；每次构造使用独立目录，保存 `environment-receipt.json` 和 `lut/*.exr`。receipt 记录读取文件的绝对路径/哈希、解析参数、全部实例 prim、LUT 哈希和后端。`applied_before_first_reset` 只表示参数应用完成，不能替代后续仿真/渲染成功检查。失败的初始化可能留下未完成纹理目录，但不会写成功 receipt。

每个环境实例分别设置相同父 link 局部外参，world 放置由原场景与 `env_origins` 决定，不假定各实例 world 原点相同。相同内参的不同实例共享本次生成的纹理。原多环境共享一个 USD stage，因此相机远处可能看见邻近实例；保留原项目的物理隔离机制，本版没有新增逐相机的渲染可见性隔离。reset 不重新应用或累积外参、不重复生成 LUT；腕相机和四指参考帧随父链接运动。

`cameras=False` 仍不创建 RGB 图像传感器，但仍校验并配置 USD 中相机与四指，便于保持同一安装定义。旧 minimal 预览仍显示原第三人称相机，三路本体相机 prim 已应用 YAML；它不会额外导出三路 RGB。原生 LUT 没有 CPU 图像后处理。直接在 Isaac GUI 打开原 USD、或另写不经过以上初始化路径的脚本，不会自动读取 YAML。

运行新增依赖仅 PyYAML 与 OpenEXR3.3.5；EXR 在初始化时生成，运行不需要 RealSense SDK。请保持每路 K、畸变、尺寸属于同一图像流。若更改尺寸，已有数据集/模型的预处理仍需匹配；新录像会按实际相机尺寸写入元数据，既有数据集不会被改写。

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

## 5. 验证记录

本轮验收通过，摘要见 [validation.json](validation.json)。12项YAML/LUT/USD离线测试、原项目11项Gym客户端测试均通过；Python3.11语法与Ruff检查通过。

| 4090检查 | 结果 |
| --- | --- |
| 原 `launch_env.py --env minimal` | 默认YAML成功应用，原第三人称截图输出，退出0 |
| 原 `launch_env.py --env stack_cube --num_envs 2` | 两实例默认YAML、原动作/触觉/渲染/15帧视频正常，退出0 |
| manager双环境专项 | 三相机/实例、四官方参考帧/实例；单实例运动与局部reset通过，另一实例状态不被reset改动 |
| YAML分辨率与录像 | 512×384、320×240、640×480三路，各3帧；实际视频解码尺寸/帧数与元数据相符 |
| 原Gym→原env_server→Harness | tacff变体的meta/reset/step/render/close通过，RGB三路、触觉四路、状态/动作16维 |
| 指定不存在的YAML、错误USD父链接 | 均退出1，包含具体错误，不生成成功receipt |

双环境world原点为(+1.5,0,0)/(-1.5,0,0)m；三相机姿态矩阵与所属父链接FK组合的最大元素差约2.05e-7。相同内参在两实例共享6张EXR，局部reset不重建纹理。该检查不等同于相机逐环境可见性隔离。

Gym对照发现原始ZIP同样在首次meta请求报“Nominal tactile is not set”。本版在env_server启动后初始化触觉基准，并让Harness默认触觉类复用原项目manager已有的CompatibleTactileSensor，处理当前Isaac5.1深度/RGB缓冲dtype；不重写触觉模型，不覆盖用户自定义传感器类。修复后原Gym完整链路通过。

静态 LUT 模型沿用 v3 已验证结果：三相机189点最大0.3147 px，原生运动向量误差小于0.01 px/帧。不能把这些数学/渲染误差当作实物标定精度。

离线测试：

```bash
# 以下仅用于独立离线 Python，不要把 usd-core 安装进 Isaac 环境
python -m pip install -r requirements-environment.txt
python -m pip install pytest usd-core==25.11 pyrealsense2==2.56.5.9235
python -m pytest tests/test_yaml_scene.py -q
```

未执行硬件操作。相机外参仍为历史候选、官方四指值为资产几何推导。没有验证每个任务/策略的完整成功率，也不声称任意环境数量都可在24GB显存上运行。多环境数量与传感器尺寸受实际资源约束。

## 变更记录

| 日期 | 版本 | 内容 |
| --- | --- | --- |
| 2026-10-21 | 4 | 接入原公共初始化，默认加载/失败报错，多实例应用共享LUT，录像尺寸元数据跟随相机；修复原Gym触觉首次观测初始化并复用已有dtype适配 |
| 2026-10-20 | 3 | 独立入口的原生 LUT；原始187文件未改；历史包保留 |
