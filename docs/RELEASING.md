# 发布流程

源码在 `src/world_atlas`，skill 在 `skills/generate-world-atlas`。不要编辑个人安装副本后遗漏仓库同步。计算依赖固定在 pyproject 和 npm lockfile；发布附件不提交 Git，放在 [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)。

1. 修改版本并补回归测试。修改算法时用同种子对照，更新认可指纹必须有真实结果依据。保持已发布版本只读。
2. 复用版本相符的 Python 和 renderer 环境，执行受影响模块的回归测试及必要的代表性基准。只有依赖或跨模块行为改变、已有失败或明确疑点时才扩大范围；不因为打包重新生成完整世界。skill 测试默认读取仓库内版本；`WORLD_ATLAS_SKILL` 仅用于验证部署副本。
3. `python -m pip wheel . --no-deps --wheel-dir dist`，再执行 `python tools/package_release.py`。生成 wheel、轻量 skill、源码 ZIP 与 release-manifest。历史输入只在对应旧版本发布。
4. 复查 `assets/release.json`、所有哈希、README 下载链接版本和 source ZIP 内容，确认当前配方与世界设置样例随 skill 分发。
5. 提交源码和更新的 skill 清单，创建匹配版本的新 tag，再用 `gh release create <tag> <全部附件> --verify-tag --notes-file <发布说明>` 发布。不要上传 `.venv`、node_modules、原用户工程、缓存、密钥、完整旧地图或验证临时目录。
6. 从实际 release 下载并校验发布附件，在新目录安装已封装的计算包。依赖版本未变时复用已记录的第三方环境；确认导入版本、wheel 与 manifest，沿用本次已有测试和计时结果，不再生成地形样本。`--check-only` 不等于网络下载已通过。

失败时保留可诊断错误，不转向旧项目路径，不禁用 TLS，不改期望哈希。中断安装留下的运行目录由用户保留检查，下次选择新目录，不递归删除用户目录自动重试。

当前工作区计算版本为 1.4.0.dev17 开发预发布；当前配方和严格 v3 世界设置在 `examples`，历史 v38 参数需搭配 1.1.0。发布维护不覆盖玩家原地图。普通世界交付复用引擎内建机器检查，打开一次网页后交付；独立复算与专项审计按任务需要显式执行。长绘图任务从同输入、同引擎的有效 checkpoint 续跑。
