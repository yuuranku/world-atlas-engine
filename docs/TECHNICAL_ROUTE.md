# 技术路线与实现边界

本文描述 1.2.1 实际代码，不是未来功能清单。计算包不调用 AI；skill 调用计算包并承担需求解释、参考选择、结果判断和人工验收。

## 1. 输入与随机性

`skills/generate-world-atlas/references/questionnaire.md` 定义 17 题；`prepare_brief.py` 保存答案、来源和待确认状态。问卷不是万能参数转换器；每个选项的支持状态须核对，再形成有效配方和 `worldgen.json`。

`api.load_recipe` 严格检查字段、有限数值和 unsigned 32-bit 种子。地形种子决定构造/形态，人文种子决定下游身份与命名；冻结配置与输入 SHA-256。`api._fresh_output` 禁止生成时覆盖旧目录。

## 2. 世界尺度的构造与地壳

主入口 `core/procedural_planet.py`，依赖 `tectonic_foundation.py`、`physical/tectonics.py`、`physical/tectonic_relief.py` 等。先构造板块归属、运动和边界，再建立地壳结构、抬升/裂解与大陆潜势。球面坐标模块承担距离和面积解释；展示采用经纬度展开图。

同一种子对应连续参数化形态，不预置“固定五种大陆图片”。生成过程有启发式约束，并非运行数亿年的完整岩石圈动力学。上游构造参考场为 720×360，v38 最终物理字段为 2176×1088。普通大陆的展示接缝放在海洋。选中的极地大陆覆盖极点，在经纬展开投影中必然横贯所有经度；这一极地条带是左右海洋约束的明确例外。

南北极分别用配方的 `north_polar_continent`、`south_polar_continent` 控制。选择先进入大陆地壳潜势，再进行共同海平面切分和地形计算；不是在成图后贴一块白色冰盖。海冰与陆地是不同字段。四种组合均保持独立选择和目标海陆面积。

## 3. 连续地表、海岸与海底

大陆潜势、地壳、构造抬升和多尺度地貌构成连续表面；海陆来自共同海平面切分。平原、盆地、高原、山麓和山系由地貌规则、构造背景和侵蚀共同产生，而非各画一张互不相关的图。

v38 的 `core/coastal_margins.py` 对连续地表做沿岸分段坐标位移，**先变换地表，再切海平面**。使用板块运动信息和有界位移预算，在不等长岸段形成内凹、外凸和方向转折；横岸影响有紧支撑，邻近地势随岸移动。后续重算水陆、侵蚀和海底，不靠显示层随机锯齿假装复杂。

v38 样本有 450 个修改岸段，实际位移约 16.58–102.44 km；这是单一种子的统计，不是所有海岸的硬性尺寸。稳定沉积岸仍可平缓。绝对板块速度不能完整解释被动大陆边缘的历史，因此这里只是受地质概念约束的程序化近似。

海底结合陆架、洋壳/构造背景、深海形态和沿岸衔接生成。颜色是相对高程/深度表达，不能将每条等值线都当成有测量精度的真实米制等高线。

参考锚点：[NOAA ETOPO](https://www.ncei.noaa.gov/products/etopo-global-relief-model) 提供连续陆海地形的观察基准；[NPS 岩岸地貌](https://www.nps.gov/articles/rocky-coast-landforms.htm) 用于比较基岩、构造与侵蚀相关形态。参考是检验细节的依据，不复制地球大陆布局。

## 4. 气候与水文

`physical/climate.py` 使用行星、纬度和地形参数近似气候；`physical/hydrology.py` 处理地表洼地、汇流、湖泊、流向与河网。河流路径依赖地表和汇水图，而非单独装饰性 SVG。D8 有向汇流仍是离散计算骨架；展示平滑不能替代水流连通与地势检查。

思路对照：[terrain-erosion-3-ways](https://github.com/dandrino/terrain-erosion-3-ways) 展示侵蚀与排水网络的不同路线；[Red Blob Games](https://www.redblobgames.com/maps/terrain-from-noise/) 帮助区分噪声地形手段与结构性地貌需求。它们是研究参考，不是本项目完整实现的替代说明。

## 5. 城市—交通—文化—政治链路

`core/society/pipeline.py` 的实际次序：

1. 从物理条件和宜居性导出人口与聚落位置。
2. 计算初始交通，供文化/语言扩散与接触使用，再给聚落命名。
3. 生成宗教和圣城；圣城改变路线需求，重新计算交通。
4. 根据地形、河流、通达性、文化和制度轮廓生成政权；将交通与政权关系对齐。
5. 最多两轮补行政中心，重算交通、制度轮廓和国家。这是有界耦合修正，不是假装各层毫无反馈。
6. 提升边境聚落的军事角色、加入战略据点，再生成省份和地理名称。
7. 程序化身份和排除词表处理名称；机器检查实体引用、覆盖和旧名残留。

交通见 `transport.py`，政权扩张见 `politics.py`、`territorial_simulation.py`，省份见 `provinces.py`。山河是阻隔，桥梁是具体跨河通路，已有道路降低部分通行成本；不将整条河的流域直接视为一个国家。海运与陆路有不同的可通行域，并在验收时检查跨陆/跨水错误。

这是规则驱动的历史地理模拟，不代表对真实文明高低或历史必然性的科学判定。若调整部落空间、时代、制度或名称习惯，须核对对应规则是否已经参数化，不能只改变题目文字。

## 6. 表达层与交互

`core/render.py` 及 review 模块生成 PNG 底图、SVG 等值线/河流/共享边界和本地 HTML。采用 [Mapshaper](https://github.com/mbloch/mapshaper) 处理共享拓扑几何；同一边界供相邻填色和边线使用，降低白缝/错位。Mapshaper 是实际 npm 依赖，不是可省略的概念性参考。

国界长虚线、省界细短虚线；标签按视图与缩放分层，城市形态/描边/文字位置分别验收。拖动流畅度属于展示责任层，不能靠偷偷降低真实计算分辨率解决。

## 7. 输出、验证与复现

地形输出：`source/physical-fields.npz`、源图和 provenance、`worldgen.json`、`terrain-run.json`、`review/index.html`/PNG/SVG。完整世界额外包含物理网格、人文实体/字段、命名和发布检查。

四类检查不能混同：程序正确性、同环境复现一致性、地理合理性、美术与交互可用性。`checks.py`、`acceptance.py` 做机器检查；`tools/verify_accepted_terrain.py` 对认可基准；浏览器工具读取实际页面并截图。路径等环境字段不作为地理语义一致性的证据。

同版本同环境 v38 的全尺寸字段、PNG 和 SVG 已独立复算一致。跨 NumPy/Python/平台不承诺位级一致；相同种子也不自动意味着任意新世界好看。

## 8. 下载和依赖边界

轻量 skill → `assets/release.json` 固定版本/大小/SHA-256 → HTTPS GitHub Release → 已校验缓存 → 本地 Python venv + npm renderer → `doctor` → CLI。

下载使用 Python 标准库，不需要先安装下载框架。HTTPS 重定向限定 GitHub 资源域；下载临时文件经长度/哈希核对后才原子加入缓存。发布包哈希属于完整性检查，不是独立签名或第三方安全审计；使用者仍需信任获取的 skill 和发布者。

Python 依赖：NumPy 2.3.5、SciPy 1.17.1、ContourPy 1.3.3、Pillow 12.3.0、Shapely 2.1.2。完整地图用 Node + 锁定 Mapshaper 0.7.56；截图才需要 Playwright/浏览器。使用者须自行提供 Python/Node 系统程序。npm 锁文件保留传递依赖版本，部分传递包仍有上游弃用提示，本次没有宣称安全审计完成。

## 9. 生成计算加速

保持生产网格、政治迭代次数和所有通行约束不变。用 [SciPy ndimage](https://docs.scipy.org/doc/scipy/reference/ndimage.html) 的连通标记与棋盘距离变换替换逐格 Python 洪泛和重复整图扩散，显式恢复经度环绕；聚落间距使用局部空间桶，但保留原来的候选顺序、平局规则与精确距离判断。

交通缓存只在同一次世界生成中生效。键包含通行域、代价场和精细修正场的完整 SHA-256，以及有方向的起终点；新增城市不重算未改变的路线，地理代价改变则重新寻路。

政治边界调整先筛选实际邻国与有收益的交换，再检查移走区域后的完整连通性；维护随领土转移同步更新的成员集合，避免每次扫描全世界。没有将“两个邻居”误当断点，也没有省去飞地检查。图连通性的概念参考 [NetworkX 割点说明](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.articulation_points.html)，实际这里保留精确遍历，不引入额外图框架。

`tools/profile_generation.py` 定位真实世界热点；`benchmark_spatial.py`、`benchmark_state_boundaries.py` 与发布的 1.1.0 在同输入上比较速度和数组一致性。完整生成把物理、人文、渲染耗时写入 `timing.json`，续跑累计计算时间，不把人工等待算入耗时。
