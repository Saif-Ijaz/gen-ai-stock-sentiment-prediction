from rag.prompts import prediction_prompt

def rag_explanation_chain(llm, ticker, prediction, prob, docs):
    context = "\n".join([d["text"] for d in docs[:3]])
    prompt = prediction_prompt(ticker, prediction, prob, context)

    return llm.invoke(prompt)
