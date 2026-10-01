"""Three independent fantasy languages, adapted to each generated sound family."""

from .model import NamingProfile
from .realistic import PROFILES as REALISTIC


_THEMES = (
    ("astral", "星月幻想", "星 月 霜 云 天 苍 银 灵 玉 紫 玄 瑶 霞 雪 辰 璇 曜 青",
     "澜 宸 墟 岚 阙 霁 川 陵 泽 渊 原 庭",
     "艾露 瑟莱 伊莲 星诺 萨维 艾瑟 洛瑞 维兰 诺伊 阿芙 米瑞 塔伊 露恩 希瓦 赛琳 奥瑞 瑟温 维露",
     "星落 望舒 霜华 云阙 天璇 银烬 月隐 玉衡 苍穹 灵渊 瑶光 流霞"),
    ("elder", "古国幻想", "玄 苍 烬 黑 赤 岩 铁 龙 古 石 朔 暮 霜 白 崇 远 寒 幽",
     "墟 岭 阙 原 岑 城 垣 渊 陵 川 泽",
     "阿兹 卡恩 巴洛 杜尔 格拉 麦格 索伦 瓦尔 奥格 塔尔 布兰 科尔 德拉 赫尔 萨尔 洛克 诺恩 瑟格",
     "龙眠 古烬 黑岩 铁脊 霜垣 苍骨 赤炉 朔风 石冠 暮谷 寒渊 幽岭"),
    ("sylvan", "森海幻想", "青 碧 苍 翠 白 松 榆 桐 柳 竹 风 云 雾 灵 花 鹿 羽 月",
     "岚 溪 川 汀 泽 原 岑 岸 野 林 洲 谷",
     "艾兰 蕾恩 希露 维娜 莉安 瑟拉 奥伦 梅瑞 法伦 伊芙 诺兰 洛恩 瑞亚 萨林 艾芙 塔琳 露维 米娅",
     "鹿隐 羽栖 风眠 碧落 苍松 雾林 云萝 灵溪 花镜 月榕 青羽 白鹿"),
)

PROFILES = tuple(
    NamingProfile(f"fantasy-{theme}-{style:02d}", label, style,
        tuple((east_stems if style == 0 else west_stems).split()),
        tuple(east_endings.split()) if style == 0 else REALISTIC[style].endings,
        landmarks=tuple(landmarks.split()) if style == 0 else ())
    for theme, label, east_stems, east_endings, west_stems, landmarks in _THEMES
    for style in range(12)
)
