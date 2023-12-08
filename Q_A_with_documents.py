from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
import sys

loader=DirectoryLoader(r'Run_llama2_local_cpu_upload\data',
                       glob="*.txt",
                       )

documents=loader.load()


text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


text_chunks=text_splitter.split_documents(documents)


embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})


#** Armazenar em vectores***
vector_store=FAISS.from_documents(text_chunks, embeddings)

query=""
vector_store.similarity_search(query)

#print(docs)
llm=CTransformers(model=r"MODEL__LLAMA2\llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':128,
                          'temperature':0.01})


template="""O contexto a seguir é uma coletanea de feedbacks do serviço de telefonia

Feedbacks:{context}
Pergunta:{question}

use esses feedbacks para responder a pergunta.
"""

qa_prompt=PromptTemplate(template=template, input_variables=['context', 'question'])

#start=timeit.default_timer()

chain = RetrievalQA.from_chain_type(llm=llm,
                                   chain_type='stuff',
                                   retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                   return_source_documents=True,
                                   chain_type_kwargs={'prompt': qa_prompt})

while True:
    user_input=input(f"prompt:")
    if query=='exit':
        print('Exiting')
        sys.exit()
    if query=='':
        continue
    result=chain({'query':user_input})
    print(f"Answer:{result['result']}")
