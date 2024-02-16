import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

## Function To get response from LLAma 2 model

def getLLamaresponse(input_text,no_words,blog_style,saving_amount,medical_expense,age_box,destination_go):

    ### LLama2 model
    llm=CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    
    ## Prompt Template

    template="""
        Calculate a pension optimization plan for a person having pension of {input_text},
        with savings {saving_amount},living in a {blog_style},age of person{age_box},
        having medical condition{medical_expense},
        having travel plans{destination_go} in about {no_words} no of words, 
        allocate specific funds to differerent fields.
            """
    
    prompt=PromptTemplate(input_variables=["blog_style","input_text",'no_words','saving_amount','medical_expense','age_box','destination_go'],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words,saving_amount=saving_amount,medical_expense=medical_expense,age_box=age_box,destination_go=destination_go ))
    print(response)
    return response






st.set_page_config(page_title="Pension optimization Plan",
                    page_icon='🧓👵🏥🏦🩺',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Pension optimization Plan🧓👵")

input_text=st.number_input("Enter pension amount")

## creating to more columns for additonal 2 fields

col1,col2=st.columns([5,5])
col3,col4=st.columns([5,5])
col5,col6=st.columns([5,5])
with col1:
    no_words=st.text_input('No of Words')
with col3:
    medical_expense=st.text_input('enter your medical condition')
with col4:
    saving_amount=st.number_input('enter the saving amount')
with col2:
    blog_style=st.selectbox('Which type of family you live in ',
                            ('Joint Family','Nuclear Family'),index=0)
with col5:
    age_box=st.number_input('Enter your age ')
with col6:
    destination_go=st.text_input('Enter the place where you want to travel')
    
submit=st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text,no_words,blog_style,saving_amount,medical_expense,age_box,destination_go))