"""Minimize recognizable personal information at cross-group boundaries."""

import re
import unicodedata
from typing import Any

_MARKER = "[涉及个人隐私，已隐藏]"
_IDENTIFIERS = re.compile(
    r"(?<!\d)(?:\+?86[ -]?)?1[3-9]\d(?:[ -]?\d){8}(?!\d)"
    r"|(?<!\d)\d{17}[\dXx](?!\d)"
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"|(?:银行卡|银行账号|身份证号|微信(?:号|号码)?|QQ(?:号|号码)?|验证码|密码|密钥)\s*(?:是|为|[:：=])\s*[^\s，。；;]{4,}",
    re.IGNORECASE,
)
_PERSONAL = re.compile(
    r"(?:家庭住址|详细住址|家庭地址|现住址)\s*(?:是|为|[:：=])\s*[^\n，。；;]{3,}"
    r"|(?:我|他|她|家人|父母|妈妈|爸爸|孩子|朋友|同事|本人)[^\n。！？!?]{0,20}"
    r"(?:住院|确诊|患有|病历|诊断|做手术|手术治疗|抑郁症|生病|感冒|发烧|怀孕|离婚|家暴|工资|薪水|收入|欠债|失业|被辞退|性取向)"
    r"|[\u4e00-\u9fff]{2,4}(?:确诊了?|住院了?|患有|患了|抑郁症|怀孕了?|离婚了?|失业了?|被辞退|做了手术)"
    r"|(?:我|他|她|本人)(?:家)?(?:住在|住址在)[^\n，。；;]{4,}",
    re.IGNORECASE,
)
_ASK_PRIVATE = r"手机号|手机号码|电话号码|联系方式|微信号|QQ号|家庭住址|详细住址|身份证号|银行卡号|病历|诊断结果|工资|薪水|性取向|住院|确诊"
_PRIVATE_QUESTION = re.compile(
    rf"(?:他|她|本人|对方|别人|群友|成员|朋友|家人|[\w\u4e00-\u9fff]{{1,8}}的)[^\n。！？!?]{{0,20}}(?:{_ASK_PRIVATE})"
    rf"|(?:{_ASK_PRIVATE})[^\n。！？!?]{{0,8}}(?:是多少|是什么|多少|告诉我|给我|发给我|了吗)",
    re.IGNORECASE,
)
_GROUP_RECORDS = re.compile(r"聊天记录|群聊记录|发言记录|聊天原话|历史消息|成员名单|消息列表|最近(?:都|在)?聊(?:了?什么|啥|些啥|些什么)|(?:刚才|昨天)[^\n]{0,3}说(?:了?什么|啥)")
_SEND_INTENT = re.compile(r"发(?:给|到|去)|发送|转发|转告|转达|传(?:给|到|话)|告诉|通知|分享")
_SELF_DETAIL = re.compile(
    r"我(?:本人的|自己的|的)(?:手机号码|手机号|电话号码|手机|电话|联系方式|微信号|QQ号|邮箱|家庭住址|详细住址|身份证号|银行卡号|工资|薪水|性取向)"
    r"|(?:我|本人)(?:已经|最近|正在|今天|昨天|刚刚|要|得了|患有)?(?:住院|确诊|手术|抑郁症)"
    r"|(?:我本人|我自己)(?:明确|自愿)?(?:同意|授权)",
    re.IGNORECASE,
)
_OTHER_PERSON_DETAIL = re.compile(
    r"(?:他|她|别人|朋友|家人|父母|妈妈|爸爸|孩子|同事|[\u4e00-\u9fff]{2,4}的)"
    r"[^\n。！？!?]{0,15}(?:手机|电话|号码|联系方式|微信|QQ|邮箱|住址|地址|身份证|银行卡|住院|确诊|抑郁症|工资|薪水|性取向|1[3-9]\d(?:[ -]?\d){8})"
    r"|[\u4e00-\u9fff]{2,4}(?:也|已经|刚刚|最近)?(?:住院|确诊|患病|抑郁症|怀孕|离婚)",
    re.IGNORECASE,
)
INTERNAL_TOPICS = {
    "游戏": ("游戏", "osu", "开黑", "打图", "排位", "手游"),
    "比赛": ("比赛", "赛事", "赛程", "锦标", "联赛"),
    "动漫": ("动漫", "番剧", "漫画", "二次元"),
    "音乐": ("音乐", "歌曲", "唱歌", "演奏"),
    "影视": ("电影", "电视剧", "影视", "观影"),
    "美食": ("美食", "吃饭", "午饭", "晚饭", "做饭"),
    "旅行": ("旅行", "旅游", "郊游", "出游"),
    "技术": ("技术", "代码", "电脑", "软件", "网络", "编程"),
    "学习": ("学习", "考试", "数学", "作业", "上课"),
    "运动": ("运动", "跑步", "篮球", "足球", "健身"),
    "生活": ("生活", "天气", "睡觉", "日常"),
    "活动": ("活动", "聚会", "聚餐", "约玩"),
}


def hide_other_group_ids(text: str, current_scene: str, spaces: dict[str, list[str]]) -> str:
    """Hide configured group IDs belonging to other groups in user-facing text."""
    ids = {str(group) for groups in spaces.values() for group in groups
           if str(group) != str(current_scene) and str(group).isdigit() and len(str(group)) >= 5}
    for group_id in sorted(ids, key=len, reverse=True):
        text = re.sub(rf"(?<!\d){re.escape(group_id)}(?!\d)", "[群号已隐藏]", text)
    return text


def contains_personal_data(text: str) -> bool:
    text = unicodedata.normalize("NFKC", text or "")
    return bool(_IDENTIFIERS.search(text) or _PERSONAL.search(text))


def asks_for_personal_data(text: str) -> bool:
    return bool(_PRIVATE_QUESTION.search(unicodedata.normalize("NFKC", text or "")))


def asks_for_group_records(text: str) -> bool:
    return bool(_GROUP_RECORDS.search(unicodedata.normalize("NFKC", text or "")))


def explicitly_shares_own_data(request: str, content: str) -> bool:
    """Accept only a current, explicit request containing the same personal data."""
    request = unicodedata.normalize("NFKC", request)
    content = unicodedata.normalize("NFKC", content)
    outside_quotes = re.sub(r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"]*"|\'[^\']*\'', "", request)
    if not (_SEND_INTENT.search(request) and _SELF_DETAIL.search(outside_quotes)):
        return False
    if _OTHER_PERSON_DETAIL.search(_SELF_DETAIL.sub("", outside_quotes)):
        return False
    identifiers = [match.group() for match in _IDENTIFIERS.finditer(content)]
    if identifiers:
        if len(identifiers) != 1:
            return False
        compact_request = re.sub(r"[\s-]+", "", request).casefold()
        return all(re.sub(r"[\s-]+", "", value).casefold() in compact_request for value in identifiers)
    return bool(content.strip() and content.strip() in request)


def public_context_text(text: str) -> str:
    return _MARKER if contains_personal_data(text) else text


def is_general_topic(text: str) -> bool:
    """Only allow fixed broad topics, never arbitrary chat details."""
    return text in INTERNAL_TOPICS


def topic_matches(topic: str, text: str) -> bool:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return any(term.casefold() in text for term in INTERNAL_TOPICS.get(topic, ()))


def copies_source_text(text: str, sources: list[str], *, min_chars: int = 8) -> bool:
    """Catch substantial verbatim copies; paraphrases still need model judgment."""
    if min_chars < 1:
        raise ValueError("min_chars must be positive")
    normalized = re.sub(r"[^\w\u4e00-\u9fff]", "", text).casefold()
    for source in sources:
        original = re.sub(r"[^\w\u4e00-\u9fff]", "", source).casefold()
        if any(original[index:index + min_chars] in normalized for index in range(max(0, len(original) - min_chars + 1))):
            return True
    return False


def sanitize_linked_group_data(data: dict[str, Any]) -> dict[str, Any]:
    """Hide sensitive evidence while keeping cursors and ordinary topic data."""
    filtered = False
    text_keys = {"content", "text", "context", "fact", "description", "user_name", "nickname", "card"}

    def visit(value: Any, key: str = "") -> Any:
        nonlocal filtered
        if isinstance(value, str) and key in text_keys and contains_personal_data(value):
            filtered = True
            return _MARKER
        if isinstance(value, list):
            return [visit(item, key) for item in value]
        if isinstance(value, dict):
            result = {name: visit(item, name) for name, item in value.items()}
            if any(result.get(name) == _MARKER for name in ("content", "text")):
                if "user_name" in result:
                    result["user_name"] = "[已隐藏]"
                if "user_id" in result:
                    result["user_id"] = ""
            first = result.get("first_message")
            if isinstance(first, dict) and first.get("content") == _MARKER and "participants" in result:
                result["participants"] = []
            return result
        return value

    cleaned = visit(data)
    if filtered:
        cleaned["privacy_filtered"] = True
    return cleaned
