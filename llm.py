from langchain import PromptTemplate
template = """Sentence: {sentence}
Translation in {language}:"""
prompt = PromptTemplate(template=template, input_variables=["sentence", "language"])

from langchain import OpenAI, LLMChain
llm = OpenAI(temperature=0)
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.predict(sentence="aashvi and vanya are quite annoyed today due to studies", language="hindi"))
