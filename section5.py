import os
from datetime import datetime
from typing import List
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start, router, or_
from crewai.agent import LiteAgentOutput
from crewai import Agent, Task, Crew, LLM
from crewai.project import CrewBase, agent, task, crew
from env import OPENAI_API_KEY
from tools import web_search_tool

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class Post(BaseModel):
    title: str
    content: str
    hashtag: List[str]


class ScoreManager(BaseModel):
    score: int = 0
    reason: str = ""


class BlogContentMakerState(BaseModel):

    topic: str = ""
    max_length: int = 1000
    research_data: LiteAgentOutput | None = None
    score_manager: ScoreManager | None = None
    post: Post | None = None


@CrewBase
class SEOManagerCrew:

    @agent
    def seo_agent(self):
        return Agent(
            role="SEO 전문가",
            goal="블로그 게시물의 SEO 효율성을 엄격하고 정확하게 평가하여 검색 엔진 최적화 품질을 측정합니다. 각 평가 요소에 대해 구체적이고 실용적인 피드백을 제공하며, 객관적인 기준에 따라 정확한 점수를 산출합니다.",
            backstory="""
            당신은 10년 이상의 경력을 가진 SEO 전문 컨설턴트로, 구글 알고리즘 변화와 검색 트렌드에 정통합니다.
            키워드 밀도, 제목 최적화, 콘텐츠 구조화, 사용자 의도 분석, 가독성 평가 등 모든 SEO 요소를 체계적으로 분석합니다.
            데이터 기반의 정확한 평가를 통해 콘텐츠가 검색 결과에서 상위 랭킹을 달성할 수 있도록 구체적이고 실행 가능한 개선안을 제시합니다.
            """,
            llm="openai/o4-mini",
            verbose=True,
        )

    @task
    def check_seo_task(self):
        return Task(
            description="""
            주어진 블로그 게시물을 다음 SEO 기준으로 종합 분석하여 정확한 점수와 개선 방안을 제시하세요:

            ## 평가 기준 (각 항목별 세부 분석 필수):
            1. **키워드 최적화 (25점)**
               - 타겟 키워드의 자연스러운 배치와 밀도
               - 제목, 소제목, 본문 내 키워드 활용도
               - 관련 키워드 및 동의어 사용

            2. **제목 및 구조 최적화 (25점)**
               - 제목의 검색 친화성과 클릭 유도성
               - 헤딩 태그(H1, H2, H3) 구조화
               - 논리적 콘텐츠 흐름

            3. **콘텐츠 품질 및 길이 (25점)**
               - 정보의 정확성과 유용성
               - 적절한 콘텐츠 길이와 깊이
               - 독창성과 가치 제공

            4. **사용자 경험 및 가독성 (25점)**
               - 문장 길이와 가독성
               - 단락 구성과 시각적 구조
               - 검색 의도와의 일치도

            ## 출력 요구사항:
            - **총점**: 0-100점 (각 항목별 점수 합산)
            - **상세 분석**: 각 평가 기준별 현재 상태와 구체적 개선점
            - **우선순위**: 가장 중요한 개선 영역 3가지
            - **실행 가능한 개선안**: 구체적이고 즉시 적용 가능한 방법

            분석 대상 게시물: {post}
            타겟 주제: {topic}
            """,
            expected_output="""
            다음을 포함하는 Score 객체:
            - score: SEO 품질을 평가하는 0-100 사이의 정수
            - reason: 점수에 영향을 미치는 주요 요인을 설명하는 문자열
            """,
            agent=self.seo_agent(),
            output_pydantic=ScoreManager,
        )

    @crew
    def crew(self):
        return Crew(
            agents=[self.seo_agent()], tasks=[self.check_seo_task()], verbose=True
        )


class BlogContentMakerFlow(Flow[BlogContentMakerState]):

    @start()
    def init_make_blog_content(self):

        if self.state.topic == "":
            raise ValueError("주제는 비워둘 수 없습니다")

    @listen(init_make_blog_content)
    def research_by_topic(self):
        researcher = Agent(
            role="수석 연구원",
            backstory="당신은 다양한 분야의 전문 데이터베이스와 최신 트렌드에 정통한 전문 리서처입니다. 과학적 연구 방법론을 기반으로 신뢰성 높은 정보를 수집하고, 복잡한 데이터를 읽기 쉽게 정리하여 핵심 인사이트를 추출하는 능력을 갖추고 있습니다.",
            goal=f"{self.state.topic}에 대한 최신 트렌드, 과학적 근거, 실용적 활용 방안을 종합적으로 조사하여 독자에게 가치 있는 인사이트를 제공하세요.",
            tools=[web_search_tool],
            llm="openai/o4-mini",
        )

        self.state.research_data = researcher.kickoff(
            f"""
            '{self.state.topic}' 주제에 대해 다음 요소들을 중심으로 종합적인 리서치를 수행하세요:

            1. **최신 동향 및 트렌드**: 최근 1년 내 주요 발전 사항
            2. **과학적/기술적 근거**: 신뢰성 있는 연구 데이터나 전문가 의견
            3. **실용적 적용 사례**: 실제 활용 방법이나 사례 연구
            4. **미래 전망**: 향후 발전 방향이나 예상 대안
            5. **일반인을 위한 설명**: 전문용어를 쉽게 설명할 수 있는 자료

            각 정보는 출처와 신뢰도를 포함하여 제공해 주세요.
            """
        )

    @listen(or_(research_by_topic, "remake"))
    def handle_make_blog(self):
        llm = LLM(model="openai/o4-mini")

        score_reason = (
            self.state.score_manager.reason if self.state.score_manager else ""
        )
        if self.state.post is None:
            result = llm.call(
                f"""
                다음 리서치 데이터(resarch data)를 기반으로 '{self.state.topic}' 주제에 대한 고품질 SEO 최적화 블로그 글을 작성해 주세요.

                ## 작성 가이드라인:
                ### 내용 구성
                - **도입부**: 주제의 중요성과 현재 상황을 간결하게 설명
                - **본문**: 3-4개 소주제로 나눠 체계적으로 설명
                - **결론**: 핵심 인사이트와 실용적 시사점 제시

                ### SEO 최적화 요소
                - **제목**: 60자 이내, 주요 키워드 포함, 호기심 유발
                - **키워드**: 주제 관련 키워드를 자연스럽게 배치
                - **구조**: 논리적 흐름과 명확한 구성
                - **길이**: {self.state.max_length}자 이내로 충분한 정보 제공

                ### 타깃 독자
                - 주제에 관심 있는 일반인부터 전문가까지
                - 기초 개념부터 시작하여 심화 내용까지 포괄

                반드시 다음 JSON 형식으로 응답하세요:
                {{
                    "title": "검색에 최적화된 매력적인 제목",
                    "content": "구조화되고 유용한 블로그 내용",
                    "hashtag": ["주요키워드1", "주요키워드2", "주요키워드3"]
                }}

                <research data>
                -----------------------------
                {self.state.research_data}
                -----------------------------
                </research data>
                """
            )
        else:
            # 블로그 remake
            result = llm.call(
                f"""
            SEO 전문가의 분석에 따르면 '{self.state.topic}' 주제의 블로그 게시물(post)이 다음과 같은 이유로 개선이 필요합니다:
            **SEO 분석 결과**: {score_reason}

            ## 개선 전략:
            ### 핵심 개선 영역
            - **콘텐츠 품질 제고**: 더 깊이 있고 전문적인 정보 제공
            - **SEO 최적화**: 키워드 배치, 제목 최적화, 구조 개선
            - **사용자 경험**: 가독성과 유용성 향상
            - **검색 의도**: 사용자가 찾는 정보와 매칭 개선

            ### 리라이팅 가이드라인
            1. **제목 개선**: 더 구체적이고 검색 친화적으로 수정
            2. **콘텐츠 재구성**: 논리적 흐름과 명확한 구조
            3. **가치 추가**: 실용적 정보와 인사이트 강화
            4. **키워드 최적화**: 주요 키워드의 자연스러운 배치

            반드시 다음 JSON 형식으로 응답하세요:
            {{
                "title": "SEO 최적화된 개선된 제목",
                "content": "고품질로 리라이팅된 블로그 콘텐츠",
                "hashtag": ["전략적키워드1", "전략적키워드2", "전략적키워드3"]
            }}

            <post>
            --------------------------------
            {self.state.post.model_dump_json()}
            --------------------------------
            </post>

            다음 연구를 사용하세요.

            <research data>
            -----------------------------
            {self.state.research_data}
            -----------------------------
            </research data>
            """
            )

        self.state.post = Post.model_validate_json(result)

    @listen(handle_make_blog)
    def manage_seo(self):

        if self.state.post is None:
            raise ValueError("post가 없습니다.")

        result = (
            SEOManagerCrew()
            .crew()
            .kickoff(
                inputs={
                    "topic": self.state.topic,
                    "post": self.state.post.model_dump_json(),
                }
            )
        )
        self.state.score_manager = result.pydantic  # type:ignore

    @router(manage_seo)
    def manage_score_router(self):

        if self.state.score_manager is None:
            raise ValueError("score_manager가 없습니다.")

        if self.state.score_manager.score >= 80:
            self._save_to_markdown()
            return None

        else:
            return "remake"

    def _save_to_markdown(self):
        # 블로그 게시물을 마크다운 파일로 저장
        if self.state.post is None or self.state.score_manager is None:
            return
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{self.state.topic}_{timestamp}.md"

        markdown_content = f"""
        # {self.state.post.title}
        **주제**: {self.state.topic}
        **작성일**: {datetime.now().strftime("%Y년 %m월 %d일 %H:%M")}
        **SEO 점수**: {self.state.score_manager.score}/100

        ## 내용
        {self.state.post.content}

        ## 해시태그
        {' '.join(f'#{tag}' for tag in self.state.post.hashtag)}

        ## SEO 분석
        **점수**: {self.state.score_manager.score}/100
        **분석**: {self.state.score_manager.reason}
        """

        with open(filename, "w", encoding="utf-8") as file:
            file.write(markdown_content)

        print("블로그 게시물이 저장되었습니다.")


flow = BlogContentMakerFlow()

flow.kickoff(inputs={"topic": "AI 로보틱스"})

# flow.plot()