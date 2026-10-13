# =====================================================
# CatBoost 모델 설정
# =====================================================

CATBOOST_CONFIG = {
    1: {
        "filename": "작업부하_catboost.cbm",
        "features": [
            ("근무시간", ["F.1.S.1"]),
            ("수면시간", ["F.1.S.3"]),
            ("작업시간", ["F.1.S.2"]),
            ("근무시간대", ["F.1.S.5"]),
            ("작업 허용 시간", ["F.1.S.4"]),
        ],
    },

    2: {
        "filename": "장비_및_기기_catboost.cbm",
        "features": [
            (
                "Equipment_Accessibility",
                ["F.2.S.1"],
            ),
            (
                "Equipment_Quality",
                ["F.2.S.2", "F.2.S.4"],
            ),
            (
                "System_Support",
                ["F.2.S.3", "F.2.S.6"],
            ),
            (
                "Documentation_Quality",
                ["F.2.S.5"],
            ),
            (
                "Maintenance_Support",
                ["F.2.S.7"],
            ),
        ],
    },

    3: {
        "filename": "절차_및_지침_catboost.cbm",
        "features": [
            (
                "(직무)절차의 일관성",
                ["F.3.S.5"],
            ),
            (
                "(직무)절차의 추적성",
                ["F.3.S.6"],
            ),
            (
                "(직무)절차의 유무(가용성)",
                ["F.3.S.1"],
            ),
            (
                "(직무)절차의 명료성(구체성)",
                ["F.3.S.3"],
            ),
            (
                "(직무)절차의 적절성(완전성 포함)",
                ["F.3.S.4"],
            ),
            (
                "(직무)절차 기술(description)의 복잡성",
                ["F.3.S.2"],
            ),
        ],
    },

    4: {
        "filename": "교육_및_훈련_catboost.cbm",
        "features": [
            (
                "Training_Program",
                ["F.4.S.1"],
            ),
            (
                "Training_Frequency",
                ["F.4.S.2", "F.4.S.4"],
            ),
            (
                "Experience_Level",
                ["F.4.S.3"],
            ),
            (
                "Technical_Knowledge",
                ["F.4.S.5", "F.4.S.7"],
            ),
            (
                "Procedural_Knowledge",
                ["F.4.S.6"],
            ),
        ],
    },

    5: {
        "filename": "의사소통_catboost.cbm",
        "features": [
            (
                "의사소통 메시지의 길이",
                ["F.5.S.1"],
            ),
            (
                "팀/조직 의사소통 자율성",
                ["F.5.S.6"],
            ),
            (
                "의사소통 프로토콜의 구체성",
                ["F.5.S.4"],
            ),
            (
                "의사소통자 간 상황인식 공유 정도",
                ["F.5.S.5"],
            ),
            (
                "의사소통을 위한 장비/기기의 적절성",
                ["F.5.S.2"],
            ),
            (
                "의사소통을 위한 전문용어 숙지 필요성",
                ["F.5.S.3"],
            ),
        ],
    },

    6: {
        "filename": "환경_catboost.cbm",
        "features": [
            (
                "Thermal_Comfort",
                [
                    "F.6.S.10",
                    "F.6.S.11",
                    "F.6.S.12",
                ],
            ),
            (
                "Visual_Environment",
                [
                    "F.6.S.14",
                    "F.6.S.6",
                    "F.6.S.7",
                ],
            ),
            (
                "Surface_Safety",
                [
                    "F.6.S.9",
                    "F.6.S.2",
                ],
            ),
            (
                "Workspace",
                ["F.6.S.8"],
            ),
            (
                "Acoustic_Environment",
                ["F.6.S.13"],
            ),
            (
                "Outdoor_Climate",
                [
                    "F.6.S.4",
                    "F.6.S.3",
                    "F.6.S.1",
                    "F.6.S.5",
                    "F.6.S.2",
                ],
            ),
        ],
    },

    7: {
        "filename": "작업절차_catboost.cbm",
        "features": [
            (
                "역할 분담 명료화",
                ["F.7.S.4"],
            ),
            (
                "작업 계획 적절성",
                ["F.7.S.2"],
            ),
            (
                "작업절차 및 규정",
                ["F.7.S.3"],
            ),
            (
                "작업자간 협업 정도",
                ["F.7.S.5"],
            ),
            (
                "작업 관리 및 감독 적절성",
                ["F.7.S.1"],
            ),
        ],
    },

    8: {
        "filename": "복잡성_catboost.cbm",
        "features": [
            (
                "Goal_Clarity",
                ["F.8.S.1"],
            ),
            (
                "Decision_Complexity",
                [
                    "F.8.S.2",
                    "F.8.S.3",
                ],
            ),
            (
                "Concurrent_Decision_Load",
                ["F.8.S.4"],
            ),
            (
                "Information_Complexity",
                [
                    "F.8.S.5",
                    "F.8.S.6",
                ],
            ),
            (
                "Information_Uncertainty",
                ["F.8.S.7"],
            ),
        ],
    },
}