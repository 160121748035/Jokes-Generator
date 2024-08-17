import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

st.title("Jokes Generator")

type = st.sidebar.selectbox("Pick type of joke",("dark", "racist", "dad", "puns", "one-liner", "political", "clean", "offensive"))
value = st.sidebar.text_input('Enter number of jokes')
if st.sidebar.button("Submit"):
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
    template1 = PromptTemplate(
        input_variables = ['type','value'],
        template = " generate {value} {type} jokes "
    )
    chain = LLMChain(llm = llm , prompt = template1,output_key = 'jokes')
    seqchain = SequentialChain(
      chains = [chain],
      input_variables = ['type','value'],
      output_variables = ['jokes']
    )
    st.write(seqchain({'type':'sexist','value':value})['jokes'])


