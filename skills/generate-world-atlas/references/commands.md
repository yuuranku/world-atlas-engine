# 引擎命令和资源

## 安装

本 skill 是轻量下载入口。`assets/release.json` 固定公开仓库、版本、文件长度和 SHA-256；`assets` 保留本地资产清单、当前配方/世界设置样例、历史 v38 配方/指纹与 renderer 的 npm 锁文件。

用 Python 3.14 执行 `scripts/install_engine.py --target <全新独立运行目录>`。脚本核对本地资产，再自动从固定 GitHub Release 下载 wheel，检查长度/哈希后创建局部虚拟环境，安装 wheel 与锁定的 Mapshaper，报告 `runtime.json` 和执行路径。不修改全局 Python/npm，不启动可见后台窗口。

默认缓存位于运行目录同级的 `.world-atlas-downloads/<版本>`；也可显式 `--cache <缓存目录>`。命中仍校验；损坏缓存保留并报错，让用户选新缓存目录，不强行覆盖。下载中断的临时文件不会变成可安装文件。缓存应放在支持硬链接的本地文件系统（已验证 NTFS）。不用 latest，不绕过 HTTPS/哈希，不把访问令牌写入 skill。

`--check-only` 只检查本地清单，不联网也不创建运行目录。仅下载计算包：`python scripts/download_engine.py --cache <缓存目录>`。历史 v5 输入与其匹配引擎留在 1.1.0 Release，不混入新版安装流程。

安装版本由资产清单 `engineVersion` 决定。配方必须明确两个极地选择；完整世界必须提供严格 v3 世界设置，包含 `technologyEra` 和 `travelCapabilities`。普通世界的额外交通能力填 `[]`；只有玩家世界观支持时才声明飞行能力。复现历史世界使用其原版引擎，新版不自动迁移或猜测缺失参数。

首次安装需要已有 Python 3.14、Node.js/npm 和网络；运行阶段离线，不需 API key。Mapshaper 0.7.56 是完整地图共享边界的实际依赖，不能漏列。

分享 [完整轻量 skill ZIP](https://github.com/yuuranku/world-atlas-engine/releases/download/v1.4.0.dev17/generate-world-atlas-skill.zip)，不能只发 SKILL.md。这是开发预发布。安装到宿主 skills 目录后重新加载技能；首次新世界仍先问 17 题，确认有效参数后才自动下载。源代码和单独 wheel 见 [版本下载页](https://github.com/yuuranku/world-atlas-engine/releases/tag/v1.4.0.dev17)。不要使用原作者的盘符路径。`--target` 必须是全新目录，安装后使用返回的解释器/Mapshaper 路径。仅安装计算 wheel 时，地形模式只需 Python 及 wheel 声明的库；完整人文地图另需锁定的 Node/Mapshaper。

## 入口

把以下 `python` 换成报告的虚拟环境解释器，`<mapshaper>` 换成报告的 JS 入口绝对路径。

```text
python -m world_atlas doctor --mapshaper <mapshaper>
python -m world_atlas seeds --root-seed 2116268501 --count 6
python -m world_atlas terrain --recipe <确认后的配方.json> --output <新地形目录>
python -m world_atlas world --terrain <认可地形目录> --settings <确认后的世界设置.json> --output <新世界目录> --exclude <旧名清单.json> --mapshaper <mapshaper>
python -m world_atlas reproduce --inputs <同版完整世界输入目录> --output <新复现目录> --mapshaper <mapshaper>
python -m world_atlas verify <完整世界目录>
```

`--exclude` 可重复，接受旧 `society.json` 或含 `forbidden` 数组的 JSON。排除已用专名，不全禁“河”“王国”等通称。

`terrain` 产出 source NPZ/PNG、`worldgen.json` 及 `review/index.html`。地形认可后，把已确认的行星、人文、地形/人文/命名种子写进严格的世界设置 JSON，再运行 `world`；不改已发布输入或 fieldBundle 哈希。设置会复制进成品并参与 SHA-256 校验。

`world` 重算物理网格和人文，保存 source、grid、society、review、命名及语义检查。普通生成复用这些内建记录，不额外重复 `verify` 或 `reproduce`。`verify` 用于交付文件有疑点或明确要求专项检查时，写检查报告，不修改计算数组。

`reproduce` 固定地形，重建后续链路并比较四项语义指纹：物理数组、人文数组、人文实体及导航网络；这不代表“所有新种子均已验证”。

新地形配方参考 `assets/terrain.json`，人文设置参考 `assets/world-settings.json`，两个极地布尔字段必填。历史 v38 用固定 1.1.0 引擎与 `terrain-v38.json` 复现，并对照原 `terrain-v38.acceptance.json`，不能用新版改写旧指纹。

Python API 与 CLI 共用实现：`from world_atlas import generate_terrain, generate_world, reproduce_world, verify_world`。

## 约束

- 输出必须不存在；不删除用户目录绕过保护，创建新版本。
- seed 为 unsigned 32-bit；配方严格匹配 `PlanetRecipe`，拒绝未知字段/NaN。
- 图片/字段各核对 SHA-256，不改哈希迁就未知输入。
- 复现要保留代码、依赖、配置、源及种子，不只是随机数。
- Python 3.14 为验证环境，其他版本不能承诺位级相同。
- 上游参考场 720×360，生产地形 2176×1088；低尺寸候选不代表最终细节。
- 部分人文和形态策略仍是代码规则，不存在万能 JSON；需要开发时明说。
- 长计算记录日志，定期简报，不重复启动同一任务。

## 续跑

中断后先查 `regeneration.json` 和 checkpoint。仅在 `status=building`、源/配置/引擎版本不变时，使用 `world_atlas.rebuild.publish_accepted_world(output)` 从物理 checkpoint 继续，或 `finish_accepted_world(output)` 从验证过的 society checkpoint 继续。不是忽略错误的备用算法。

不确定时保留 checkpoint，先定位失效的输入或阶段；不得默认从头重新生成整个世界。版本和输入绑定相同才续跑，绑定不符时说明需要重算的具体阶段。续跑复用已有有效检查，只补受影响或缺失部分；玩家明确要求复现时再与原预期语义指纹比较。
