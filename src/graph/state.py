from typing import List

from typing_extensions import TypedDict

from src.index.indexer import retriever
from src.llms.answer_grader import answer_grader
from src.llms.generator import rag_chain
from src.llms.hallucination_grader import hallucination_grader
from src.llms.retrieval_grader import retrieval_grader
from src.llms.rewriter import question_rewriter


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The question being asked
        generation: The generated answer from the LLM
        feedback: Feedback from the previous generation
        documents: List of retrieved documents
    """

    question: str
    generation: str
    feedback: str
    documents: List[str]


def retrieve(state):
    """
    Retrieve documents based on the question in the state.

    Args:
        state (dict): The current graph state

    Returns:
        dict: Updated state with retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieve documents
    documents = retriever.invoke(question)
    print(documents)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate an answer based on the question and documents in the state.

    Args:
        state (dict): The current graph state

    Returns:
        dict: Updated state with the generated answer
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    feedback = state.get("feedback", "")

    # Generate answer using RAG chain
    generation = rag_chain.invoke(
        {
            "context": documents,
            "question": question,
            "feedback": (
                f"Feedback from last LLM run: \n {feedback}, \ncan you now improve based on this response"
                if feedback
                else ""
            ),
        }
    )
    print(generation)
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determine the relevance of the retrieved documents to the question.

    Args:
        state (dict): The current graph state

    Returns:
        dict: Updated state with only relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Filter relevant documents
    filtered_docs = []
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        print(score)
        if score["score"] == "yes" or score["score"][0] == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        dict: Updated state with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    print(better_question)

    return {"documents": documents, "question": better_question}


### Edges ###


def decide_to_generate(state):
    """
    Determine whether to generate an answer or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for the next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered out
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # Relevant documents are available, generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determine whether the generation is grounded in the documents and answers the question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for the next node to call
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    print(score)
    state["feedback"] = score["explanation"]

    if score["score"] == "yes" or score["score"][0] == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        print(score)
        if (
            score["score"] == "yes"
            or score["score"][0] == "yes"
            and not score["weakness"]
        ):
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            feedback = state["feedback"]
            if score["strengths"]:
                feedback += "\n STRENGTH:\n" + "\n-".join(score["strengths"])
            if score["weaknesses"]:
                feedback += "\n WEAKNESS:\n" + "\n-".join(score["weaknesses"])
            if score["suggestion"]:
                feedback += "\n SUGGESTION:\n" + score["suggestion"]
            state["feedback"] = feedback
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
