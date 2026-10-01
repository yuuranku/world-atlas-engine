"""Historical naming layers: dynastic identities and older regional formations.

The Chinese polity inventory uses the requested Han/Tang/Qin/Jin/Chu pattern;
it does not assign those names any real-world geography or history.
"""

from .model import NamingProfile
from .realistic import PROFILES as REALISTIC


_OLD_STEMS = (
    "洛 汾 淮 泗 沅 湘 汉 衡 岷 丹 石 云 青 白 龙 武 永 广 安 临 会 鄢 梁 兰 陇 渭 泾 洮 岐 鄱 邺 汝",
    "埃德 阿尔 布伦 赫尔 林德 诺德 哈尔 沃尔 古德 莱恩 奥斯特 伊尔",
    "瓦伦 奥雷 卡斯特 卢卡 蒙特 弗洛 塞维 塔伦 阿尔 拉文 科尔 贝内",
    "贝洛 诺沃 罗斯 维索 切尔 米尔 卢博 斯塔罗 布雷 卡梅 多布 科罗",
    "乌鲁 阿克 科克 巴彦 铁穆 卡拉 阿尔 萨勒 科尔 阿斯 库图 巴特",
    "米赫 罗珊 巴赫 阿扎 法尔 沙赫 纳林 阿夫 扎尔 古尔 索赫 西尔",
    "努尔 哈迪 拉希 哈萨 巴希 纳西 塔里 扎伊 法里 萨希 卡西 赛义",
    "罗摩 因陀 梵达 那罗 索摩 毗罗 阿摩 摩诃 俱摩 苏里 莲达 迦兰",
    "苏拉 丹加 马鲁 兰加 巴拉 马纳 加拉 纳加 塔拉 拉玛 巴厘 乌纳",
    "格温 阿文 布林 杜恩 格伦 凯尔 达文 莫恩 内文 弗恩 埃文 布瑞",
    "伊尔 凯米 塔维 瓦拉 科伊 萨洛 艾诺 佩拉 拉赫 维萨 哈拉 伊瓦",
    "姆巴 恩加 基里 卢巴 恩多 桑加 卡桑 基桑 图马 马拉 卢马 纳巴",
)

PROFILES = tuple(
    NamingProfile(f"historical-{style:02d}", "历史风格", style,
        tuple(dict.fromkeys((*stems.split(), *REALISTIC[style].stems))),
        tuple("阳 阴 陵 安 平 宁 川 原 城 津".split()) if style == 0 else REALISTIC[style].endings,
        polities=tuple("汉 唐 秦 晋 楚 燕 赵 齐 魏 吴 蜀 梁 陈 隋 宋 周 郑 卫 鲁 越 夏 商 许 蔡 曹 邓 滕 纪 薛 莒".split()) if style == 0 else (),
        landmarks=tuple("苍梧 云梦 丹霞 鹿鸣 白石 青冥 雁回 石梁 长风 龙首 北邙 南屏 玉衡 清源 松陵 梁溪".split()) if style == 0 else ())
    for style, stems in enumerate(_OLD_STEMS)
)

