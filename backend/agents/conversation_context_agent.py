from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.config import get_settings
from models.schemas import AgentResult, ChatHistoryMessage
from utils.json_utils import parse_json_object

from .base_agent import BaseAgent


SYSTEM_PROMPT = """你是电商导购对话上下文理解Agent，擅长把自然口语里的购物意图整理成后续推荐系统容易使用的结构化信息。

你会结合当前消息、最近对话和图片摘要，写出一个独立可理解的购物需求；如果用户是在追问上一轮商品、预算、适用场景或对比建议，可以自然延续仍然相关的上下文。当前消息里出现的新预算、品牌、用途、尺寸、颜色、禁忌和偏好，通常更能代表这一轮的真实需求。

当前项目的商品类目包括：数码家电、美妆个护、家居用品、食品生鲜、运动户外、图书文娱、医药健康、汽车用品。类目理解可以参考这些语义范围：
- 数码家电：耳机、手机、电脑、平板、智能设备、小家电、影音和充电配件。
- 美妆个护：护肤、彩妆、洗护、香氛、剃须、口腔护理和身体护理。
- 家居用品：收纳、清洁、厨具、床品、家具、照明、母婴居家和日用百货。
- 食品生鲜：零食、饮料、水果、肉蛋奶、方便食品、茶咖酒水、营养补给和适合携带的吃喝。
- 运动户外：运动装备、户外装备、露营、徒步、骑行、健身、运动服饰和防护用品。
- 图书文娱：图书、文具、玩具、乐器、游戏、模型、影音娱乐和文化礼品。
- 医药健康：药品、医疗器械、保健护理、血糖血压监测、康复和健康管理用品。
- 汽车用品：车载电器、清洁养护、内饰、行车安全、维修工具和汽车周边。

用户表达里可能同时包含多个购物目标，比如主商品和补充商品、组合购买、场景搭配，或由“还有、顺便、另外、同时、搭配、以及”引出的另一类商品。理解这类句子时，可以先在心里区分“用户想买几类东西”，再把每一类东西分别放进 shopping_goals。categories 更像是这些目标涉及类目的汇总，而不是只选最显眼的那个类目。

例如“请推荐一些户外用品相关的商品，还有户外运动时候补充能量的吃的”，可以理解为两个并行目标：一类是户外运动装备，接近运动户外；另一类是可携带、补充能量的食物，接近食品生鲜。像“吃的”“零食”“水果”“饮料”“补给”“能量棒”“运动补能”“带点能量的东西”这类表达，在购物场景里通常更接近食品生鲜；“户外用品、运动装备、露营、徒步、越野、登山、防晒防雨装备”通常更接近运动户外。

再比如“买通勤耳机和桌面收纳”，可以整理成数码家电和家居用品两个目标；“给老人买血压计，再配点低糖零食”，可以整理成医药健康和食品生鲜两个目标。这样后续推荐系统能为每个目标分别召回商品，而不是让一个宽泛场景覆盖掉另一个真实需求。

如果用户只是泛泛地说“推荐点好用的东西”，而上下文也没有清晰方向，categories 和 shopping_goals 可以保持为空，让推荐链路用更通用的候选召回。长期画像可以作为背景线索，但当前消息和最近对话更适合作为本轮意图依据。

输出保持为一个JSON对象，字段如下:
{
  "standalone_query": "完整购物需求",
  "is_follow_up": true,
  "categories": ["类目"],
  "shopping_goals": [
    {
      "goal": "购物目标描述",
      "category": "类目",
      "keywords": ["关键词"],
      "constraints": ["约束"]
    }
  ],
  "budget": 3000,
  "constraints": ["约束1", "约束2"],
  "avoid_categories": ["不应混入的类目"],
  "reason": "简短原因"
}
"""

GOAL_REVIEW_PROMPT = """你是电商购物目标审查Agent，负责温和地复核上一轮意图解析是否覆盖了用户真实想买的东西。

请从 current_message、recent_history、image_summary、available_categories 和 initial_result 中判断购物目标是否完整。重点留意并列需求、组合购买、场景搭配、主商品加补充商品等表达。如果用户一句话里有“户外用品”和“补充能量的吃的”这类两个购买对象，而 initial_result 只留下了运动户外，就可以补上食品生鲜目标；如果 initial_result 已经清楚覆盖用户意图，可以返回等价结构。

当前项目类目语义可参考：数码家电、美妆个护、家居用品、食品生鲜、运动户外、图书文娱、医药健康、汽车用品。像“吃的、零食、水果、饮料、补给”这类购物表达通常可归入食品生鲜；户外、运动、露营、徒步相关用品通常可归入运动户外。

输出保持为与上一轮相同的JSON对象:
{
  "standalone_query": "完整购物需求",
  "is_follow_up": true,
  "categories": ["类目"],
  "shopping_goals": [
    {
      "goal": "购物目标描述",
      "category": "类目",
      "keywords": ["关键词"],
      "constraints": ["约束"]
    }
  ],
  "budget": 3000,
  "constraints": ["约束1", "约束2"],
  "avoid_categories": ["不应混入的类目"],
  "reason": "简短原因"
}
"""


class ConversationContextAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__(name="conversation_context", timeout=6.0)
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.0,
            max_tokens=700,
        )

    async def _execute(self, **kwargs: Any) -> AgentResult:
        message: str = kwargs.get("message", "")
        history: list[ChatHistoryMessage] = kwargs.get("history", [])
        available_categories: list[str] = kwargs.get("available_categories", [])
        image_summary: str = kwargs.get("image_summary", "")

        payload = {
            "current_message": message,
            "recent_history": [
                {"role": item.role, "content": item.content}
                for item in history[-8:]
                if item.content
            ],
            "available_categories": available_categories,
            "image_summary": image_summary,
        }
        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ]
            )
            data = parse_json_object(response.content)
            data = await self._review_goals(payload, data)
        except Exception as exc:
            return self._fallback_context(message, str(exc))

        standalone_query = str(data.get("standalone_query") or message).strip()
        shopping_goals = self._valid_shopping_goals(
            data.get("shopping_goals"),
            available_categories,
        )
        shopping_goals = self._augment_shopping_goals(
            message=message,
            standalone_query=standalone_query,
            shopping_goals=shopping_goals,
            available_categories=available_categories,
        )
        categories = self._merge_categories(
            self._valid_categories(data.get("categories"), available_categories),
            [goal["category"] for goal in shopping_goals if goal.get("category")],
        )
        budget = self._budget(data.get("budget"))
        constraints = data.get("constraints") if isinstance(data.get("constraints"), list) else []
        avoid_categories = self._valid_categories(
            data.get("avoid_categories"),
            available_categories,
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "original_query": message,
                "standalone_query": standalone_query or message,
                "is_follow_up": bool(data.get("is_follow_up")),
                "categories": categories,
                "shopping_goals": shopping_goals,
                "budget": budget,
                "constraints": [str(item) for item in constraints if item],
                "avoid_categories": avoid_categories,
                "reason": str(data.get("reason") or ""),
            },
            confidence=0.85,
        )

    async def _review_goals(
        self,
        payload: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        review_payload = {
            **payload,
            "initial_result": data,
        }
        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=GOAL_REVIEW_PROMPT),
                    HumanMessage(content=json.dumps(review_payload, ensure_ascii=False)),
                ]
            )
            reviewed = parse_json_object(response.content)
            return reviewed or data
        except Exception:
            return data

    def _fallback_context(self, message: str, error: str) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            success=False,
            error=error,
            data={
                "original_query": message,
                "standalone_query": message,
                "is_follow_up": False,
                "categories": [],
                "shopping_goals": [],
                "budget": None,
                "constraints": [],
                "avoid_categories": [],
                "reason": "fallback_to_current_message",
            },
            confidence=0.0,
        )

    def _valid_categories(self, raw: Any, available_categories: list[str]) -> list[str]:
        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, list):
            values = [str(item) for item in raw]
        else:
            values = []
        available = set(available_categories)
        return [value for value in values if value in available]

    def _valid_shopping_goals(
        self,
        raw: Any,
        available_categories: list[str],
    ) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        available = set(available_categories)
        goals: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category") or "")
            if category and category not in available:
                continue
            keywords = item.get("keywords") if isinstance(item.get("keywords"), list) else []
            constraints = (
                item.get("constraints") if isinstance(item.get("constraints"), list) else []
            )
            goals.append(
                {
                    "goal": str(item.get("goal") or "").strip(),
                    "category": category,
                    "keywords": [str(keyword) for keyword in keywords if keyword],
                    "constraints": [str(value) for value in constraints if value],
                }
            )
        return goals

    def _merge_categories(self, *category_groups: list[str]) -> list[str]:
        merged = []
        for categories in category_groups:
            for category in categories:
                if category and category not in merged:
                    merged.append(category)
        return merged

    def _augment_shopping_goals(
        self,
        message: str,
        standalone_query: str,
        shopping_goals: list[dict[str, Any]],
        available_categories: list[str],
    ) -> list[dict[str, Any]]:
        inferred = self._infer_goals_from_text(
            " ".join([message or "", standalone_query or ""]),
            available_categories,
        )
        if len(inferred) < 2:
            return shopping_goals

        merged = list(shopping_goals)
        existing_categories = {goal.get("category") for goal in merged}
        for goal in inferred:
            category = goal.get("category")
            if category in existing_categories:
                continue
            merged.append(goal)
            existing_categories.add(category)
        return merged

    def _infer_goals_from_text(
        self,
        text: str,
        available_categories: list[str],
    ) -> list[dict[str, Any]]:
        available = set(available_categories)
        normalized = str(text or "")
        inferred: list[dict[str, Any]] = []

        outdoor_terms = [
            "户外",
            "运动",
            "徒步",
            "登山",
            "露营",
            "越野",
            "骑行",
            "跑步",
        ]
        food_terms = [
            "吃的",
            "食品",
            "零食",
            "水果",
            "饮料",
            "补充能量",
            "补能",
            "能量",
            "能量棒",
            "便携食",
        ]

        outdoor_keywords = [
            term for term in outdoor_terms if term and term in normalized
        ]
        if "运动户外" in available and outdoor_keywords:
            inferred.append(
                {
                    "goal": "户外运动用品",
                    "category": "运动户外",
                    "keywords": outdoor_keywords,
                    "constraints": [],
                }
            )

        food_keywords = [term for term in food_terms if term and term in normalized]
        if "食品生鲜" in available and food_keywords:
            inferred.append(
                {
                    "goal": "便携补能食品",
                    "category": "食品生鲜",
                    "keywords": food_keywords,
                    "constraints": ["适合户外携带"] if "户外" in normalized else [],
                }
            )

        return inferred

    def _budget(self, raw: Any) -> float | None:
        if raw in ("", None):
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None
