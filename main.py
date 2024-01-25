from PIL import Image
import random
import os
import openai
import streamlit as st 
from streamlit_chat import message

from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter

from serpapi import GoogleSearch


st.set_page_config(page_title='GPTをアップグレードしよう',layout='centered')
st.title('GPTをアップグレードしよう')
tab_titles = ['検索','PDF読み込み(工事中)','Youtube読み込み(工事中)']
tab1, tab2, tab3 = st.tabs(tab_titles)

with tab1:
    st.header('ウェブ検索')
    st.write('<font size="4">Google検索により、ChatGPTでは回答できない2021年9月以降のデータについても取り込んで回答します。</font>', unsafe_allow_html=True)
    st.write('  \n')

    image0 = Image.open('./pics/pic0.PNG')
    image1 = Image.open('./pics/ramen0.PNG')
    image2 = Image.open('./pics/run0.PNG')
    image3 = Image.open('./pics/pic3.PNG')
    image4 = Image.open('./pics/sakee0.PNG')
    image5 = Image.open('./pics/sea0.PNG')
    image6 = Image.open('./pics/teacher2.PNG')
    image7 = Image.open('./pics/pic7.PNG')

    image = [image0,image1,image2,image3,image4,image5,image6,image7]

    st.image(image[random.randint(0,7)])
    st.write('  \n')
    st.write('  \n') 

    user_api_key = st.sidebar.text_input(
    label="Your OpenAI API key",
    placeholder="あなたのOpenAI API keyをここにペーストして下さい。",
    type="password")
    openai.api_key = user_api_key
    os.environ['SERPAPI_API_KEY'] = "74fea45ccb646481e0c9ac1fd733fdce926cdb7694eb61e6e900adbec7cf08b1"

    llm = OpenAI(openai_api_key=user_api_key)
    tools = load_tools(["serpapi"],llm=llm)
    agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

    ques = st.text_input(
    label="ご質問は何でしょうか？",
    placeholder="ここに入力して下さい",
    type="default")

    st.write(agent.run(ques))

    st.sidebar.write('改善点やご要望等のフィードバックを頂けますと大変励みになります。ご連絡/お問合せは以下メールまでお願い致します。')
    st.sidebar.write('<a href="mailto:tadahisa.terao777@gmail.com">tadahisa.terao777@gmail.com</a>', unsafe_allow_html=True)
    profileImg = Image.open('./pics/profile.JPG')
    st.sidebar.image(profileImg)
    st.sidebar.write('(開発者略歴)現在関西を拠点に大手外資コンサルティング会社にてライフサイエンス/メドテック業界のデジタル化ご支援コンサルティングに従事。IT/インターネット/医療業界にて15年以上の新規事業開発経験、8年の海外駐在経験を有する。INSEAD MBA/東京大学大学院工学系研究科卒。趣味はマラソン(サブ3.5)とアプリ開発(Python/Swift)、2児(長男/次男)の父。')
    
    

with tab2:
    st.header('AI先生に聞いてみよう')
    st.write('ChatGPTのAPIを活用したチャットボットです。PDFをアップロードしたらその中身について教えてあげましょう。ただしOpenAIのAPIKeyが必要です。')
    st.image(image[random.randint(0,7)])

    uploaded_file = st.file_uploader("調べたいPDFをアップロードして下さい。", type="pdf")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap  = 100,
        length_function = len,
    
    )

    if uploaded_file :
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(file_path=tmp_file_path)  
        data = loader.load_and_split(text_splitter)

        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_documents(data, embeddings)

        chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo-16k'),
                                                                      retriever=vectors.as_retriever())
        # This function takes a query as input and returns a response from the ChatOpenAI model.
        def conversational_chat(query):

            # The ChatOpenAI model is a language model that can be used to generate text, translate languages, write different kinds of creative content, and answer your questions in an informative way.
            result = chain({"question": query, "chat_history": st.session_state['history']})
            # The chat history is a list of tuples, where each tuple contains a query and the response that was generated from that query.
            st.session_state['history'].append((query, result["answer"]))
            
            # The user's input is a string that the user enters into the chat interface.
            return result["answer"]
        
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["こんにちは! この資料について気軽に何でも聞いて下さいね♪" + uploaded_file.name]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["こんにちは!"]
            
        # This container will be used to display the chat history.
        response_container = st.container()
        # This container will be used to display the user's input and the response from the ChatOpenAI model.
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                
                user_input = st.text_input("質問を入力して下さい:", placeholder="Please enter your message regarding the PDF data.", key='input')
                submit_button = st.form_submit_button(label='Send')
                
            if submit_button and user_input:
                output = conversational_chat(user_input)
                
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="big-smile")
   
with tab３:
    st.write('Coming Soon !!!')