from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.llms import OpenAI
memory = ConversationSummaryMemory(llm=OpenAI(temperature=0))
memory.save_context({"input": "hi, I'm looking for some ideas to write an essay in AI"}, {"output": "hello, what about writing on LLMs?"})
memory.load_memory_variables({})
print(memory)
