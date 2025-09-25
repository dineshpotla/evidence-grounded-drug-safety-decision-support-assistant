from drug_safety_assistant.llm.claude import ClaudeAgent
from drug_safety_assistant.retrieval.agentic import AgenticRetriever
from drug_safety_assistant.retrieval.aggregator import MultiSourceRetriever
from drug_safety_assistant.types import Intent, RetrievedEvidence, SafetyRequest


class StubSourceRetriever(MultiSourceRetriever):
    def __init__(self) -> None:
        pass

    def retrieve(self, request: SafetyRequest, intent: Intent) -> list[RetrievedEvidence]:
        return [
            RetrievedEvidence(
                source="openfda",
                citation_id="OPENFDA:1",
                title="Label",
                snippet="Boxed warning contraindication severe toxicity",
                metadata={"label_severity": 3},
                strength_score=3,
            ),
            RetrievedEvidence(
                source="pubmed",
                citation_id="PMID:1",
                title="Trial",
                snippet="Randomized trial reports adverse outcomes with combination therapy",
                metadata={"publication_types": ["randomized controlled trial"]},
                strength_score=2,
            ),
        ]


def test_agentic_retriever_returns_ranked_evidence() -> None:
    retriever = AgenticRetriever(
        source_retriever=StubSourceRetriever(),
        claude_agent=ClaudeAgent(api_key=""),
    )
    request = SafetyRequest(question="Is this combination high risk?", drug="example")

    items = retriever.retrieve(request=request, intent=Intent.GENERAL)

    assert items
    assert all(item.citation_id for item in items)
