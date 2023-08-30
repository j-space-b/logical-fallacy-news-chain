import warnings
warnings.filterwarnings('ignore')

import streamlit as st, tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import ChatPromptTemplate
from langchain.utilities import GoogleSerperAPIWrapper
from typing import Dict 


# Streamlit input
st.subheader('Topic to return news stories on')

# Take values from UI 
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", value="", type="password")
    serper_api_key = st.text_input("Serper API Key", value="", type="password")
    num_results = st.number_input("Number of Search Results", min_value=2, max_value=10)
    st.caption("*Search: Summarizes any relevant news item from News APIs and describes any logical fallacies.*")
search_query = st.text_input("Search Query", label_visibility="collapsed")
col1, col2 = st.columns(2)

rom typing import Dict

from langchain.chains.fallacy_removal.models import LogicalFallacy

fallacies = {
    "Logical Fallacy Name": ["Adhominem","Adpopulum","Appeal to Emotion","Fallacy of Extension",
                        "Intentional Fallacy","False Causality","False Dilemma","Hasty Generalization",
                        "Illogical Arrangement","Fallacy of Credibility","Circular Reasoning",
                        "Begging the Question","Trick Question","Overapplying","Equivocation","Amphiboly",
                        "Word Emphasis","Composition","Division"],
    "Description": ["attacks on the character or personal traits of the person making an argument rather than addressing the actual argument and evidence",
                   "the fallacy that something must be true or correct simply because many people believe it or do it, without actual facts or evidence to support",
                   "an attempt to win support for an argument by exploiting or manipulating people's emotions rather than using facts and reason",
                   "making broad, sweeping generalizations and extending the implications of an argument far beyond what the initial premises support",
                   "falsely supporting a conclusion by claiming to understand an author or creator's subconscious intentions without clear evidence",
                   "jumping to conclusions about causation between events or circumstances without adequate evidence to infer a causal relationship",
                   "presenting only two possible options or sides to a situation when there are clearly other alternatives that have not been considered or addressed",
                   "making a broad inference or generalization to situations, people, or circumstances that are not sufficiently similar based on a specific example or limited evidence",
                   "constructing an argument in a flawed, illogical way, so the premises do not connect to or lead to the conclusion properly",
                   "dismissing or attacking the credibility of the person making an argument rather than directly addressing the argument itself",
                 "supporting a premise by simply repeating the premise as the conclusion without giving actual proof or evidence",
                  "restating the conclusion of an argument as a premise without providing actual support for the conclusion in the first place",
                   "asking a question that contains or assumes information that has not been proven or substantiated",
                   "applying a general rule or generalization to a specific case it was not meant to apply to",
                   "using the same word or phrase in two different senses or contexts within an argument",
                   "constructing sentences such that the grammar or structure is ambiguous, leading to multiple interpretations",
                   "shifting the emphasis of a word or phrase to give it a different meaning than intended",
                   "erroneously inferring that something is true of the whole based on the fact that it is true of some part or parts",
                   "erroneously inferring that something is true of the parts based on the fact that it is true of the whole"]
}

# If 'Search' button is clicked
if col2.button("Search"):
    # Validate inputs
    if not openai_api_key.strip() or not serper_api_key.strip() or not search_query.strip():
        st.error(f"Please provide the missing fields.")
    else:
        try:
            with st.spinner("Loading..."):
                # Orders news items by relevancy from previous week
                search = GoogleSerperAPIWrapper(type="news", tbs="qdr:w1", serper_api_key=serper_api_key)
                result_dict = search.results(search_query)

                if not result_dict['news']:
                    st.error(f"No search results for: {search_query}.")
                else:
                    # Load URLs
                    for i, item in zip(range(num_results), result_dict['news']):
                        loader = UnstructuredURLLoader(urls=[item['link']])
                        data = loader.load()

                        # Chain 1: Summarize Any Possible Logical Fallacies
                        llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=openai_api_key)
                        prompt1 = ChatPromptTemplate.from_template(
                          "Think carefully about how the {data} may be representing any of the logical fallacy name descriptions contained in {FALLACIES}, and remember each instance as a logical fallacy in {data}"
                           "\n\nIf one ore more logical fallacies exist in {data}: Return a short sentence to summarize the full article then summarize each portion of the article that contains a logical fallacy, with the logical fallacy evident in the summary of the article portion."
                           "\n\nIf zero logical follacies are found in {data}: Return a short sentence to summarize the full article then return the following sentence as the last sentence 'No Fallacies Found'
                          "
                        )
                        chain1 = LLMChain(llm=llm, prompt=prompt1)

                        # Chain 2: Name Each Logical Fallacy and Describe Harm
                        prompt2 = ChatPromptTemplate.from_template(
                          "If the summary reads 'No Fallacies Found', then return 'No Fallacies Found'."
                          "\n\nIf the summary does not read 'No Fallacies Found' then name each logical fallacy found according to its name in {fallacies}, summarize a portion of the article where the fallacy is found in 10 words, then explain in an additional 10 words why the logical fallacy may harm society as it exists in news.  Do this until all logical fallacies found in the summary are listed."
                        )
                        chain2 = LLMChain(llm=llm, prompt=prompt2)

                        overall_chain = SimpleSequentialChain(chains=[chain1,chain2], verbose=True)

                        summary_enriched = overall_chain.run(data)

                        st.success(f"Title: {item['title']}\n\nLink: {item['link']}\n\nSummary: {summary_enriched}")
        except Exception as e:
            st.exception(f"Exception: {e}")
