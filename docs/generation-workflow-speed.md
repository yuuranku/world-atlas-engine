# 单次生成、续跑和阶段耗时

普通交付只计算一个已确认候选，依次运行地形和完整世界。`world` 本身已连续执行气候水文、人文、渲染和最终语义记录；不需要代理逐阶段重新启动，不在成功后另跑 `verify` 或 `reproduce`。

现有 CLI 分为 `terrain` 和 `world`，没有第三个完整世界命令。需要一次启动时，直接组合现有 Python API；不复制生成算法。下例保存为 `generate.py`，替换已确认的输入及 Mapshaper 路径，然后只执行一次 `python generate.py`。Windows 的进程生成必须保留 `__main__` 保护。

```python
from pathlib import Path

from world_atlas import generate_terrain, generate_world
from world_atlas.timing import measure_stage


def main():
    task = Path("new-world-task").resolve()
    task.mkdir(exist_ok=False)
    with measure_stage(task, "terrain"):
        generate_terrain("confirmed-recipe.json", task / "terrain")
    with measure_stage(task, "world"):
        generate_world(
            task / "terrain",
            "confirmed-world-settings.json",
            task / "world",
            mapshaper="installed-renderer/node_modules/mapshaper/bin/mapshaper",
        )


if __name__ == "__main__":
    main()
```

## 渲染调度

先构造共同海岸与河道约束，再在三个进程中重叠计算主要等高线、交通几何、生态数值主题和人口图层；主进程同时处理分类覆盖及地图组装。只在消费结果时等待，结果按固定源顺序写出。港口接线后的城市锚点与路线一同返回，保留原来的空间关系。几何进程退出后才启动最多四个切片进程，避免叠加进程池。

## 已有地形和中断续跑

已有认可地形直接调用 `world`，不再执行 `terrain`。`prepare_regeneration` 在不改变任何行星物理设置时寻找地形/世界根目录或物理源目录中的 `physical-contours`。只有当前代码、运行库、源海拔、诊断身份、文件哈希和图结构均符合时才复制原文件；随后提取器仍校验完整连续地形绑定。它不修改已有 manifest，不把旧算法的结果绑定到新算法。

只改人文或命名种子时，当前版本内可复用物理网格与已完成高度图。物理参数、求解代码或库版本改变则重算受影响部分。`regeneration.json` 的 `physicalContoursCheckpoint` 记录此项命中与实际来源，`physicalCheckpoint` 记录物理网格命中情况。

本次提速也改变制图等高线的提取约定：物理模拟、原生地形分辨率和连续地表计算保持不变，制图的连通关系先在原生格四倍的网格中发现（间距为原生格的四分之一），顶点随后仍回到原连续地表上求根。它不再为每一条分支逐一做形式化区间认证，因此非常窄、隐藏于采样格内部的闭环不再保证全部发现。新制图约定使用新的 checkpoint schema 和代码绑定，旧认证图不自动迁移。

中断后先查看原输出的 `regeneration.json` 和实际 checkpoint，保持输出不动。原引擎版本和输入绑定不变、且状态为 `building` 时：

- 只有完整物理网格时，调用 `world_atlas.rebuild.publish_accepted_world(output)`，继续人文和渲染。
- 已有完整、绑定有效的 `society` 时，调用 `world_atlas.rebuild.finish_accepted_world(output)`，只继续渲染；完整高度图继续复用，失败任务未完成的图重新提取。
- 渲染完成后写一次 `world_atlas.checks.semantic_checks(output)` 的结果到 `review/release-checks.json`，与普通 `generate_world` 交付相同。

续跑使用原有运行库和 `WORLD_ATLAS_MAPSHAPER` 的安装路径。指纹不符时定位具体改变的输入或代码，创建明确的新版本；不重写指纹来通过校验，不因为一个局部失败重复生成地形。

## 耗时的含义

顶层 `timing.json` 记录气候水文、人文和渲染。`review/timing.json` 记录渲染内部各阶段；上面的单次启动脚本另外在任务根目录记录地形和完整世界。每项保留实际起止、完成/失败状态和耗时，失败的计算时间也保留。

总 `elapsedSeconds` 是已完成或失败阶段的单调时钟区间并集，不是各项简单相加。父阶段已包含子阶段，并发阶段也只能按实际等待一次计时；根表的 `render` 与 `review` 的分项不能重复加总。续跑间的空闲等待不计入计算耗时。记录可在同一台机器、同一次系统启动内续用；旧计时 schema 必须由原引擎读取，不能自动迁移。

计时更新使用线程锁、跨进程文件锁和原子替换，嵌套或并发写入不会覆盖另一个阶段。只报告实际测得的阶段耗时；局部基准不能当作完整世界已经达到十几分钟的证据。
