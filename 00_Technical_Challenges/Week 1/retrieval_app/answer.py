"""Stage 5: turn retrieved chunks into an answer.

YOUR CODE GOES HERE. The prompt is yours to write — it is the cheapest thing
in the pipeline to change and often the most effective.
"""

from .index import Hit, client

CHAT_MODEL = "gpt-4.1-mini"

# def build_messages(question: str, hits: list[Hit]) -> list[dict]:
#     """Assemble the messages list sent to the chat model.

#     Args:
#         question: the user's question.
#         hits: what search() returned. May be empty.

#     Returns:
#         A list of {"role": ..., "content": ...} dicts — a system message
#         setting the rules, and a user message carrying the context and the
#         question.

#     Three things your prompt has to decide, and they are real decisions:
#       1. How do you label each chunk in the context so the model can refer to
#          it? (hit.chunk_id exists for this.)
#       2. What should it do when the context does not contain the answer? Say
#          so explicitly, or it will answer from the nearest chunk regardless.
#       3. What happens when `hits` is empty? Write that branch deliberately.

#     Hint: Session 2's RAG_SYSTEM_TEMPLATE is a good starting point, and a
#     better one once you have read Step 3's failures and edited it.
#     """   
#     if not hits:
#         return "No matching hits input for query context."
    
#     messages = []
#     # Add system prompt.
#     system_message = {}
#     system_message["role"] = "system"
#     system_message["content"] = RAG_SYSTEM_TEMPLATE.format(response_style="concise", response_length="brief")
#     messages.append(system_message)

#     user_message = {}
#     user_message["role"] = "user"
#     user_message["content"] = RAG_USER_TEMPLATE.format(
#         context=build_context(hits),
#          user_query=question, 
#          context_count=len(hits),
#          similarity_scores=[hit.score for hit in hits]
#     )
#     messages.append(user_message)
#     return messages

# def answer(question: str, hits: list[Hit]) -> str:
#     """Call the chat model and return its reply as a string.

#     Hint: `client().chat.completions.create(model=CHAT_MODEL, messages=...)`,
#     then dig out `.choices[0].message.content`. Look up what `temperature`
#     does to your results when you run the same question twice.
#     """
#     if not hits:
#         return "The query did not match any documents, so not making the LLM call." 
#     response = client().chat.completions.create(
#         model=CHAT_MODEL, 
#         messages = build_messages(question, hits)
#     )
#     return response.choices[0].message.content

RAG_SYSTEM_TEMPLATE_INTAKE_AGENT = """You are a pediatric neuropsychological intake assistant. Your only task is to map a parent or caregiver's description of a child's concerns to concern domains, using only the context provided. You do not diagnose, name conditions, or give advice.

Allowed concern domains (use these ids exactly; never output any other name):
    attention_executive
    learning_academic
    speech_language
    social_communication
    repetitive_restricted
    sensory_processing
    motor_coordination
    adaptive_daily_living
    memory_processing
    emotional_mood
    behavioral_regulation
    sleep
    skill_regression
    global_delay
    visuospatial_nonverbal

Input: the user message contains the child's age and the caregiver's statement. The context contains excerpts from a reference document. An excerpt may begin with header lines such as "Category: <id> | Age band: <band>" or "Category: <id> | Section: overview | Age band: all ages". A header may be missing.

Rules:
- A statement can map to more than one domain, or to none.
- Map a domain only if BOTH are true: (a) the caregiver actually describes the concern in their statement, and (b) the context describes that domain's coverage or signs. Do not use outside knowledge.
- A domain name or keyword appearing in the context is not enough. In particular, lines that begin "Useful to find out" are questions to ask, not descriptions of a domain; never use them as evidence.
- Use the child's age. Prefer context for the child's age band or "all ages". If the only supporting context is for a different age band, still map the domain, set "age_match" to "no", and say so in "note". If the child is 18 or older, rely on "all ages" context only and say so in "note".
- For every domain, give:
    "caregiver_phrase": a short phrase copied exactly from the caregiver's statement;
    "evidence": ONE contiguous quote copied exactly from the context (under 25 words, no ellipsis, not from the caregiver's statement);
    "source": the header line of the excerpt the quote came from, or "" if that excerpt has no header;
    "age_match": "yes" if that excerpt's age band matches the child's age, "all ages" if its header says all ages, "no" if it is for a different band, "unknown" if it has no header.
- List only supported domains. If none are supported, return an empty list and use this exact note: "I am not sure which concern domain this maps to."

Output only JSON, with no other text, in this format:
{{"domains": [{{"domain": "<id>", "caregiver_phrase": "<exact phrase>", "evidence": "<exact quote from context>", "source": "<header line or empty>", "age_match": "<yes|no|all ages|unknown>"}}], "note": "<empty string or a short note>"}}"""


RAG_USER_TEMPLATE_INTAKE_AGENT = """Context Information:
{context}

Number of relevant sources found: {context_count}
{similarity_scores}

Intake Input: {caregiver_input} 
Child's age: {child_age}
Please provide your answer based solely on the context above."""

def build_context(hits: list[Hit]) -> str:
    context_strs = [f"[Chunk {hit.chunk_id}] : {hit.text}" for hit in hits]
    return "\n".join(context_strs)

def build_messages_intake_agent(question: str, child_age:str, hits: list[Hit]) -> list[dict]:
    """Assemble the messages list sent to the chat model.

    Args:
        question: the user's question.
        hits: what search() returned. May be empty.

    Returns:
        A list of {"role": ..., "content": ...} dicts — a system message
        setting the rules, and a user message carrying the context and the
        question.

    Three things your prompt has to decide, and they are real decisions:
      1. How do you label each chunk in the context so the model can refer to
         it? (hit.chunk_id exists for this.)
      2. What should it do when the context does not contain the answer? Say
         so explicitly, or it will answer from the nearest chunk regardless.
      3. What happens when `hits` is empty? Write that branch deliberately.

    Hint: Session 2's RAG_SYSTEM_TEMPLATE is a good starting point, and a
    better one once you have read Step 3's failures and edited it.
    """   
    if not hits:
        return "No matching hits input for query context."
    
    messages = []
    # Add system prompt.
    system_message = {}
    system_message["role"] = "system"
    system_message["content"] = RAG_SYSTEM_TEMPLATE_INTAKE_AGENT
    messages.append(system_message)

    user_message = {}
    user_message["role"] = "user"
    user_message["content"] = RAG_USER_TEMPLATE_INTAKE_AGENT.format(
        context=build_context(hits),
         caregiver_input=question, 
         child_age=child_age,
         context_count=len(hits),
         similarity_scores=[hit.score for hit in hits]
    )
    messages.append(user_message)
    return messages

def answer_intake_agent(question: str, child_age: str, hits: list[Hit]) -> str:
    """Call the chat model and return its reply as a string.

    Hint: `client().chat.completions.create(model=CHAT_MODEL, messages=...)`,
    then dig out `.choices[0].message.content`. Look up what `temperature`
    does to your results when you run the same question twice.
    """
    if not hits:
        return "The query did not match any documents, so not making the LLM call." 
    response = client().chat.completions.create(
        model=CHAT_MODEL, 
        messages = build_messages_intake_agent(question, child_age, hits)
    )
    return response.choices[0].message.content