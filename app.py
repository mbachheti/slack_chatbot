import os
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# load environment variables from .env file
load_dotenv('.env')

# load documents from ./docs folder
documents = []

for file in os.listdir('docs'):
	path = './docs/' + file
	if file.endswith('.pdf'):
		loader = PyPDFLoader(path)
	elif file.endswith('.txt'):
		loader = TextLoader(path)
	elif file.endswith('.docx') or file.endswith('.doc'):
		loader = Docx2txtLoader(doc_path)
	else:
		print(f'Error: {file} does not does not end in .pdf, .txt, .doc, or .docx')
	documents.extend(loader.load())

# split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size = 500, chunk_overlap = 10)
documents = text_splitter.split_documents(documents)

vector_db = Chroma.from_documents(documents, embedding = OpenAIEmbeddings(), persist_directory = './data')

llm = ChatOpenAI(temperature = 0.7, model_name = 'gpt-4-turbo')

chain = ConversationalRetrievalChain.from_llm(
	llm = llm,
	retriever = vector_db.as_retriever(search_kwargs = {'k': 4}),
	return_source_documents = True
)

# initialize the app with your slack bot token
app = App(token = os.environ.get('SLACK_BOT_TOKEN'))

# store chat history
chat_history = []

# message handler for Slack
@app.message('.*')
def message_handler(message, say, logger):
	user = message['user']
	query = message['text']
	print(f'{user}: {query}')

	output = chain.invoke({'chat_history': chat_history, 'question': query})['answer']
	
	chat_history.append((query, output))
	print(f'Output: {output}')
	
	say(output)

# start the app
if __name__ == '__main__':
	SocketModeHandler(app, os.environ['SLACK_APP_TOKEN']).start()
