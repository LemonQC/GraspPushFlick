# 资产来源、尺度和物理参数

统一注册表为 `assets/manifests/object_registry.json`。当前 30 种 YCB 均从[官方逐物体下载接口](https://ycb-benchmarks.s3.amazonaws.com/index.html)取得真实网格和纹理，优先 Google 16k 扫描，无该扫描时选官方 Berkeley processed mesh。许可为官方声明的 CC BY 4.0，需保留 YCB 项目及原作者署名。下载收据记录 URL、字节数、SHA256；官方未提供独立 SHA256 时明确使用本地哈希、HTTP 长度、gzip/tar 完整性校验，不把本地哈希冒充官方哈希。

`download_ycb.sh` 默认准备 command.md 中的 26 种及 strawberry/apple/lemon/peach 共 30 种；有限重试，支持 `.part` 断点续传，校验过的文件跳过。解压前拒绝路径穿越、软链接和特殊设备文件。原始网格、纹理与下载包均留在项目内部。

最初还合法复用本机用户已有 YCB.zip 的 16 种原网格，并单独标记其 repack 来源尚待确认。最终注册表采用已完成的 30 种官方直接下载，不将 repack 计作额外独立物体。原始压缩包未改动或删除。

另复制本机 Isaac 4.5 官方资产包中的 Franka Panda 子树和 21 种 YCB USD 子树。Franka 为七轴 Panda + 原装双指夹爪，所有必要 USD 层和几何位于项目内；材质中的 `OmniPBR.mdl` 来自项目内 Kit 自带 MDL 库。`check_asset_integrity.py` 检查 USD 递归引用、纹理及外部路径，报告中明确区分内置 MDL 搜索模块与 USD 文件依赖。静态检查不等于断网渲染测试。

转换器保留扫描真实米制尺度、三角形和纹理 UV，不用方块替代缺失模型。将网格包围范围中心移到刚体原点，并在注册表记录 mesh-to-body 平移。碰撞采用 `convexDecomposition`，不是统一包围盒。杯、碗、剪刀、扳手和夹具的接触形状仍需实际检查凸分解是否保留凹腔和薄壁；当前没有这方面的通过结论。

所有新生成物体的质量、质心、惯量、摩擦及恢复系数均明确标记 **estimated**。质量暂按包围体积 ×0.25 填充率 ×500 kg/m³ 估计并限幅 0.02–2 kg；惯量使用该质量下均匀包围长方体近似，质心为范围中心，静/动摩擦 0.6/0.5、恢复系数 0.02。这些是可追溯的 M0 初值，不是实测参数，不能据此作翻转性能论证。惯量近似为长方体不表示碰撞形状采用长方体。

扩展数据接口核查见 `extension_asset_audit.json`：GraspNet 官方 models.zip 为 4.3 GB，本机空间允许，但尝试的 Jbox 端点超时，浏览接口未返回可直接获取的 Google Drive 模型包；没有下载训练场景和标签。官方说明 GraspNet 0–32、71 来自 YCB，45 为其 pitcher cap，已建立保守重叠排除规则；精确逐扫描映射待模型可访问后完成。

DexGraspNet 2.0 官方 Hugging Face 列表提供约 4.62 GB 的 `meshdata.tar.gz`，没有核实可按 3–5 个物体下载的接口，故不下载完整大数据集，也不将灵巧手标签当成平行夹爪标签。GraspFactory 官方仓库提供五个 Panda 示例网格，可选择性获取；下载状态、固定 source tree revision 和 Git blob 哈希在 `graspfactory_samples.json`，不下载标签。上述扩展不阻塞 YCB 首批验证，但其动态验证同样尚未运行。
