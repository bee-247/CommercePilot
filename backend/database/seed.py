"""Deterministic demo catalog used for a fresh local environment."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from sqlalchemy import func, select

from .models import (
    InventoryRecord,
    ProductRecord,
    UserBehaviorRecord,
    UserRecord,
)
from .session import ProductSessionLocal, UserSessionLocal


DEMO_PRODUCTS = [
    ("demo-headphone-1", "静旅 Pro 降噪耳机", "耳机", 699, "适合地铁通勤的头戴式主动降噪耳机，支持多设备连接。", "Acme", "audio-a", ["降噪", "通勤", "蓝牙"], 98),
    ("demo-headphone-2", "轻风开放式运动耳机", "耳机", 399, "轻量开放式设计，适合跑步和户外运动。", "Acme", "audio-b", ["运动", "防水", "轻便"], 80),
    ("demo-headphone-3", "会议通话真无线耳机", "耳机", 299, "双麦克风通话降噪，适合办公会议。", "North", "audio-c", ["无线", "办公", "通话"], 72),
    ("demo-phone-1", "远航 5G 手机", "手机", 2499, "大容量电池与高亮屏幕，满足日常拍摄和通勤使用。", "North", "mobile-a", ["续航", "快充", "5G"], 90),
    ("demo-phone-2", "映像轻旗舰手机", "手机", 3299, "轻薄机身与人像拍摄能力，支持快速充电。", "Vista", "mobile-b", ["轻便", "摄影", "快充"], 84),
    ("demo-power-1", "行野户外电源 600", "户外电源", 1599, "适合露营照明和小型设备供电，提供多种输出接口。", "Trail", "outdoor-a", ["露营", "大容量", "便携"], 76),
    ("demo-keyboard-1", "静音办公机械键盘", "电脑", 459, "紧凑布局与静音轴体，适合共享办公环境。", "North", "computer-a", ["办公", "静音", "无线"], 79),
    ("demo-monitor-1", "护眼 27 英寸显示器", "电脑", 1299, "低蓝光模式和可升降支架，适合长期办公学习。", "Vista", "computer-b", ["护眼", "办公", "学习"], 88),
    ("demo-tablet-1", "学伴 11 英寸平板", "平板", 1899, "支持手写笔与分屏，适合课程笔记和文档阅读。", "Vista", "tablet-a", ["学习", "护眼", "轻便"], 83),
    ("demo-watch-1", "跃动 GPS 运动手表", "户外", 899, "支持跑步轨迹、心率记录和日常防水。", "Trail", "outdoor-b", ["运动", "防水", "续航"], 74),
    ("demo-charger-1", "三口氮化镓快充", "配件", 229, "紧凑三口设计，可为手机、平板和轻薄电脑充电。", "Acme", "accessory-a", ["快充", "便携", "多设备"], 86),
    ("demo-shaver-1", "便携往复式剃须刀", "个护", 269, "可水洗刀头与旅行锁设计，适合出差携带。", "North", "care-a", ["便携", "防水", "续航"], 67),
]


def seed_demo_catalog() -> int:
    """Seed only an empty catalog; repeated startups do not alter user data."""
    with ProductSessionLocal() as session:
        count = session.scalar(select(func.count()).select_from(ProductRecord)) or 0
        if count:
            return 0
        for index, product in enumerate(DEMO_PRODUCTS):
            (
                product_id,
                name,
                category,
                price,
                description,
                brand,
                seller_id,
                tags,
                hot_score,
            ) = product
            session.add(
                ProductRecord(
                    product_id=product_id,
                    name=name,
                    category=category,
                    price=price,
                    description=description,
                    brand=brand,
                    seller_id=seller_id,
                    tags_json=json.dumps(tags, ensure_ascii=False),
                    status="active",
                    hot_score=hot_score,
                )
            )
            session.add(
                InventoryRecord(
                    product_id=product_id,
                    stock=15 + index * 3,
                    reserved_stock=0,
                )
            )
        session.commit()

    product_ids = [item[0] for item in DEMO_PRODUCTS]
    with UserSessionLocal() as session:
        for user_id, preferences in [
            ("web_user", ["耳机", "电脑"]),
            ("demo_commuter", ["耳机", "手机"]),
            ("demo_student", ["平板", "电脑"]),
        ]:
            session.merge(
                UserRecord(
                    user_id=user_id,
                    city="上海",
                    segments_json='["active"]',
                    preferred_categories_json=json.dumps(
                        preferences,
                        ensure_ascii=False,
                    ),
                )
            )
        now = datetime.utcnow()
        for index, product_id in enumerate(product_ids):
            session.add(
                UserBehaviorRecord(
                    user_id="demo_commuter" if index % 2 == 0 else "demo_student",
                    product_id=product_id,
                    behavior_type="view",
                    scene="demo_seed",
                    weight=1.0 + (index % 3) * 0.5,
                    created_at=now - timedelta(minutes=index),
                )
            )
        session.commit()
    return len(DEMO_PRODUCTS)
