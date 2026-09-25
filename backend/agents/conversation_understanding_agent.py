from __future__ import annotations

import json
from typing import Any

from core.agent_config import get_agent_system_config
from core.model_clients import create_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from models.schemas import AgentResult, ChatHistoryMessage
from utils.json_utils import parse_json_object

from .base_agent import BaseAgent

DATASET_CATEGORY_GUIDANCE: dict[str, str] = {
    "数码家电": "电子产品、手机及配件、电脑、软件、家用电器、影音设备和充电配件",
    "美妆个护": "护肤、彩妆、洗护、香氛、剃须、口腔护理和身体护理",
    "家居用品": "收纳、清洁、厨具、床品、家具、照明和日用家居",
    "食品饮料": "零食、饮料、方便食品、茶、咖啡、酒水、营养补给和便携吃喝",
    "运动户外": "运动装备、露营、徒步、骑行、健身、运动服饰和防护用品",
    "图书音像": "纸质书、电子书、杂志、电影电视、唱片和数字音乐",
    "医药健康": "药品、医疗器械、保健护理、健康监测、康复和家庭健康用品",
    "汽车用品": "车载电器、清洁养护、内饰、行车安全、维修工具和汽车周边",
    "服饰鞋包": "服装、鞋靴、箱包、珠宝首饰、手表和穿戴配饰",
    "母婴用品": "婴幼儿喂养、护理、出行、寝具、安全防护和孕产用品",
    "办公用品": "纸张、本册、书写工具、办公耗材、桌面用品和办公设备配件",
    "家装工具": "装修建材、五金工具、电工照明、庭院、园艺和家居维修用品",
    "宠物用品": "宠物食品、清洁护理、玩具、训练、出行和居住用品",
    "玩具游戏": "儿童玩具、桌游、拼图、模型、电子游戏和游戏设备配件",
    "艺术手工": "绘画、手工制作、缝纫、编织、雕刻材料和手工成品",
    "乐器": "乐器、演奏配件、录音设备和音乐制作用品",
    "工业科研": "工业工具、实验室设备、测量仪器、安全防护和科研耗材",
}

SYSTEM_PROMPT = """先识别对话意图，再按需提取购物需求；一次输出完整JSON。

请判断当前用户消息包含哪些意图。注意：同一句话可能同时需要商品推荐和长期记忆更新。

路由类型:
- recommendation: 用户明确想买、想推荐、想找商品、换一批商品、提出新的购买需求。
- product_follow_up: 用户追问上一轮推荐商品的原因、区别、价格、适用场景、对比、某个序号。
- preference_update: 用户表达长期偏好、排斥项、预算习惯、风格变化、要求记住或不要记住某偏好。
- recommendation_with_preference_update: 用户同时表达长期偏好变化并要求推荐商品。
- smalltalk: 问候、感谢、确认、结束语、闲聊。
- unsupported: 与购物导购无关且不适合回答的任务。

规则:
1. needs_recommendation 表示是否要进入商品召回和推荐链路。
2. needs_memory_update 表示是否要尝试抽取并更新长期记忆。
3. 两个字段彼此独立，可以同时为 true。
4. “这次/今天/送人/临时”的一次性需求通常不需要更新长期记忆，但仍可能需要推荐。
5. “以后/一直/通常/我现在更喜欢/不要再按这个推荐/别记这个”等长期偏好或记忆管理表达，通常需要更新长期记忆。
6. 如果用户同时说“以后耳机预算2000左右，帮我推荐一款”，needs_memory_update 和 needs_recommendation 都应为 true。


不需要推荐时，购物目标、类目和约束留空；不要把闲聊或偏好更新变成购买需求。
payload.force_recommendation=true 表示图片推荐入口，必须继续提取购物需求。

你是电商导购需求理解Agent，擅长把自然口语里的购物意图整理成后续推荐系统容易使用的结构化信息。

你会结合当前消息、最近对话和图片摘要，写出一个独立可理解的购物需求；如果用户是在追问上一轮商品、预算、适用场景或对比建议，可以自然延续仍然相关的上下文。当前消息里出现的新预算、品牌、用途、尺寸、颜色、禁忌和偏好，通常更能代表这一轮的真实需求。

payload 中的 available_categories 是当前商品库真实可用的类目，category_catalog 是这些类目的语义说明。categories、shopping_goals.category 和 avoid_categories 只能使用 available_categories 中的原始类目名称，不要创造近义类目，也不要把多个独立类目合并成一个宽泛类目。

用户表达里可能同时包含多个购物目标，比如主商品和补充商品、组合购买、场景搭配，或由“还有、顺便、另外、同时、搭配、以及”引出的另一类商品。理解这类句子时，可以先在心里区分“用户想买几类东西”，再把每一类东西分别放进 shopping_goals。categories 更像是这些目标涉及类目的汇总，而不是只选最显眼的那个类目。

例如“请推荐一些户外用品相关的商品，还有户外运动时候补充能量的吃的”，可以理解为两个并行目标：一类是户外运动装备，接近运动户外；另一类是可携带、补充能量的食物，接近食品饮料。像“吃的”“零食”“饮料”“补给”“能量棒”“运动补能”“带点能量的东西”这类表达，在购物场景里通常更接近食品饮料；“户外用品、运动装备、露营、徒步、越野、登山、防晒防雨装备”通常更接近运动户外。

再比如“买通勤耳机和桌面收纳”，可以整理成数码家电和家居用品两个目标；“给老人买血压计，再配点低糖零食”，可以整理成医药健康和食品饮料两个目标；“给孩子买积木和画笔”可以整理成玩具游戏和艺术手工两个目标。这样后续推荐系统能为每个目标分别召回商品，而不是让一个宽泛场景覆盖掉另一个真实需求。

如果用户只是泛泛地说“推荐点好用的东西”，而上下文也没有清晰方向，categories 和 shopping_goals 可以保持为空，让推荐链路用更通用的候选召回。长期画像可以作为背景线索，但当前消息和最近对话更适合作为本轮意图依据。

输出保持为一个JSON对象，字段如下:
{
  "route": "recommendation|product_follow_up|preference_update|recommendation_with_preference_update|smalltalk|unsupported",
  "needs_recommendation": true,
  "needs_memory_update": false,
  "memory_update_reason": "是否包含长期偏好或记忆管理要求",
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

请从 current_message、recent_history、image_summary、available_categories、category_catalog 和 initial_result 中判断购物目标是否完整。available_categories 是当前商品库真实可用类目，所有类目字段都必须使用其中的原始名称。重点留意并列需求、组合购买、场景搭配、主商品加补充商品等表达。如果用户一句话里有“户外用品”和“补充能量的吃的”这类两个购买对象，而 initial_result 只留下了运动户外，就可以补上食品饮料目标；如果 initial_result 已经清楚覆盖用户意图，可以返回等价结构。

类目语义以 category_catalog 为准。不要用未出现在 available_categories 中的近义名称替代真实类目。

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


class ConversationUnderstandingAgent(BaseAgent):
    def __init__(self):
        agent_config = get_agent_system_config()
        definition = agent_config.agent("conversation-understanding")
        model_config = agent_config.resolved_model("conversation-understanding")
        super().__init__(
            name="conversation_understanding",
            timeout=definition.runtime.timeout_seconds,
            max_retries=definition.runtime.max_attempts,
        )
        self.llm = create_chat_model(
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            timeout=definition.runtime.timeout_seconds,
            max_retries=0,
        )

    async def _execute(self, **kwargs: Any) -> AgentResult:
        message: str = kwargs.get("message", "")
        history: list[ChatHistoryMessage] = kwargs.get("history", [])
        available_categories: list[str] = kwargs.get("available_categories", [])
        image_summary: str = kwargs.get("image_summary", "")
        force_recommendation = bool(kwargs.get("force_recommendation", False))

        if not message.strip() and not image_summary and not force_recommendation:
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={
                    "route": "smalltalk",
                    "needs_recommendation": False,
                    "needs_memory_update": False,
                    "reason": "空消息不需要处理",
                },
                confidence=1.0,
            )

        payload = {
            "current_message": message,
            "recent_history": [
                {"role": item.role, "content": item.content}
                for item in history[-8:]
                if item.content
            ],
            "available_categories": available_categories,
            "category_catalog": {
                category: DATASET_CATEGORY_GUIDANCE.get(category, "按类目原名理解")
                for category in available_categories
            },
            "image_summary": image_summary,
            "force_recommendation": force_recommendation,
        }
        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ]
            )
            data = parse_json_object(response.content)
            route_data = self._route_data(data, force_recommendation)
            if route_data["needs_recommendation"]:
                data = await self._review_goals(payload, data)
            else:
                data = {}
            data.update(route_data)
        except Exception as exc:
            return self._fallback_context(message, str(exc))

        standalone_query = str(data.get("standalone_query") or message).strip()
        shopping_goals = self._valid_shopping_goals(
            data.get("shopping_goals"),
            available_categories,
        )
        if data["needs_recommendation"]:
            shopping_goals = self._augment_shopping_goals(
                message=message,
                standalone_query=standalone_query,
                shopping_goals=shopping_goals,
                available_categories=available_categories,
            )
        else:
            shopping_goals = []
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
                **route_data,
                "original_query": message,
                "standalone_query": standalone_query or message,
                "is_follow_up": bool(data.get("is_follow_up")),
                "categories": categories,
                "shopping_goals": shopping_goals,
                "budget": budget,
                "constraints": [str(item) for item in constraints if item],
                "avoid_categories": avoid_categories,
                "reason": str(data.get("reason") or route_data["reason"]),
            },
            confidence=0.85,
        )

    @staticmethod
    def _route_data(data: dict[str, Any], force_recommendation: bool) -> dict[str, Any]:
        for field in ("needs_recommendation", "needs_memory_update"):
            if not isinstance(data.get(field), bool):
                raise ValueError(f"需求理解缺少有效布尔字段: {field}")
        needs_recommendation = force_recommendation or data["needs_recommendation"]
        needs_memory_update = data["needs_memory_update"]
        route = str(data.get("route") or "")
        if needs_recommendation:
            route = (
                "recommendation_with_preference_update"
                if needs_memory_update else "recommendation"
            )
        elif needs_memory_update:
            route = "preference_update"
        elif route not in {"product_follow_up", "smalltalk", "unsupported"}:
            route = "smalltalk"
        return {
            "route": route,
            "needs_recommendation": needs_recommendation,
            "needs_memory_update": needs_memory_update,
            "memory_update_reason": str(data.get("memory_update_reason") or ""),
            "reason": str(data.get("reason") or ""),
        }

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
                "route": "recommendation",
                "needs_recommendation": True,
                "needs_memory_update": False,
                "memory_update_reason": "理解失败时不更新长期记忆",
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
        if "食品饮料" in available and food_keywords:
            inferred.append(
                {
                    "goal": "便携补能食品",
                    "category": "食品饮料",
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

    def plan_customer_service_tasks(self, user_text: str):
        """Customer-service routing is a mode of the shared understanding role."""
        from customer_service.task_schemas import ServiceSubTask, ServiceTaskPlan

        prompt = (
            "将客服请求拆成1到4个顺序任务。route可选general（咨询）、faq_specialist（FAQ）、"
            "sales_script_specialist（话术）、quality_reviewer（审核或改写回复）。"
            "仅多个明确目标才拆分；instruction须保留商品、渠道、语气、政策等约束。\n"
            + user_text
        )
        try:
            result = self.llm.with_structured_output(ServiceTaskPlan).invoke(prompt)
            return ServiceTaskPlan.model_validate(result).tasks
        except Exception:
            return [ServiceSubTask(route="general", instruction=user_text.strip() or "请帮助用户明确客服需求")]
