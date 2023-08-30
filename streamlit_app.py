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

FALLACIES: Dict[str, LogicalFallacy] = {
    "adhominem": LogicalFallacy(
        name="adhominem",
        fallacy_critique_request="Identify any feasible ways in which \
        the assistant’s last response is attacking the character or \
        personal traits of the person making an argument rather than \
        addressing the actual argument and evidence.",
        fallacy_revision_request="Please rewrite the assistant response\
        to remove any attacking the character or personal traits of the\
        person making an argument rather than addressing the actual\
        argument and evidence.",
    ),
    "adpopulum": LogicalFallacy(
        name="adpopulum",
        fallacy_critique_request="Identify ways in which the assistant’s\
        last response may be asserting that something must be true or \
        correct simply because many people believe it or do it, without \
        actual facts or evidence to support the conclusion.",
        fallacy_revision_request="Please rewrite the assistant response \
        to remove any assertion that something must be true or correct \
        simply because many people believe it or do it, without actual \
        facts or evidence to support the conclusion.",
    ),
    "appealtoemotion": LogicalFallacy(
        name="appealtoemotion",
        fallacy_critique_request="Identify all ways in which the \
        assistant’s last response is an attempt to win support for an \
        argument by exploiting or manipulating people's emotions rather \
        than using facts and reason.",
        fallacy_revision_request="Please rewrite the assistant response \
        to remove any attempt to win support for an argument by \
        exploiting or manipulating people's emotions rather than using \
        facts and reason.",
    ),
    "fallacyofextension": LogicalFallacy(
        name="fallacyofextension",
        fallacy_critique_request="Identify any ways in which the \
        assitant's last response is making broad, sweeping generalizations\
        and extending the implications of an argument far beyond what the \
        initial premises support.",
        fallacy_revision_request="Rewrite the assistant response to remove\
         all broad, sweeping generalizations and extending the implications\
         of an argument far beyond what the initial premises support.",
    ),
    "intentionalfallacy": LogicalFallacy(
        name="intentionalfallacy",
        fallacy_critique_request="Identify any way in which the assistant’s\
        last response may be falsely supporting a conclusion by claiming to\
        understand an author or creator's subconscious intentions without \
        clear evidence.",
        fallacy_revision_request="Revise the assistant’s last response to \
        remove any false support of a conclusion by claiming to understand\
        an author or creator's subconscious intentions without clear \
        evidence.",
    ),
    "falsecausality": LogicalFallacy(
        name="falsecausality",
        fallacy_critique_request="Think carefully about whether the \
        assistant's last response is jumping to conclusions about causation\
        between events or circumstances without adequate evidence to infer \
        a causal relationship.",
        fallacy_revision_request="Please write a new version of the \
        assistant’s response that removes jumping to conclusions about\
        causation between events or circumstances without adequate \
        evidence to infer a causal relationship.",
    ),
    "falsedilemma": LogicalFallacy(
        name="falsedilemma",
        fallacy_critique_request="Identify any way in which the \
        assistant's last response may be presenting only two possible options\
        or sides to a situation when there are clearly other alternatives \
        that have not been considered or addressed.",
        fallacy_revision_request="Amend the assistant’s last response to \
        remove any presentation of only two possible options or sides to a \
        situation when there are clearly other alternatives that have not \
        been considered or addressed.",
    ),
    "hastygeneralization": LogicalFallacy(
        name="hastygeneralization",
        fallacy_critique_request="Identify any way in which the assistant’s\
        last response is making a broad inference or generalization to \
        situations, people, or circumstances that are not sufficiently \
        similar based on a specific example or limited evidence.",
        fallacy_revision_request="Please rewrite the assistant response to\
        remove a broad inference or generalization to situations, people, \
        or circumstances that are not sufficiently similar based on a \
        specific example or limited evidence.",
    ),
    "illogicalarrangement": LogicalFallacy(
        name="illogicalarrangement",
        fallacy_critique_request="Think carefully about any ways in which \
        the assistant's last response is constructing an argument in a \
        flawed, illogical way, so the premises do not connect to or lead\
        to the conclusion properly.",
        fallacy_revision_request="Please rewrite the assistant’s response\
        so as to remove any construction of an argument that is flawed and\
        illogical or if the premises do not connect to or lead to the \
        conclusion properly.",
    ),
    "fallacyofcredibility": LogicalFallacy(
        name="fallacyofcredibility",
        fallacy_critique_request="Discuss whether the assistant's last \
        response was dismissing or attacking the credibility of the person\
        making an argument rather than directly addressing the argument \
        itself.",
        fallacy_revision_request="Revise the assistant’s response so as \
        that it refrains from dismissing or attacking the credibility of\
        the person making an argument rather than directly addressing \
        the argument itself.",
    ),
    "circularreasoning": LogicalFallacy(
        name="circularreasoning",
        fallacy_critique_request="Discuss ways in which the assistant’s\
        last response may be supporting a premise by simply repeating \
        the premise as the conclusion without giving actual proof or \
        evidence.",
        fallacy_revision_request="Revise the assistant’s response if \
        possible so that it’s not supporting a premise by simply \
        repeating the premise as the conclusion without giving actual\
        proof or evidence.",
    ),
    "beggingthequestion": LogicalFallacy(
        name="beggingthequestion",
        fallacy_critique_request="Discuss ways in which the assistant's\
        last response is restating the conclusion of an argument as a \
        premise without providing actual support for the conclusion in \
        the first place.",
        fallacy_revision_request="Write a revision of the assistant’s \
        response that refrains from restating the conclusion of an \
        argument as a premise without providing actual support for the \
        conclusion in the first place.",
    ),
    "trickquestion": LogicalFallacy(
        name="trickquestion",
        fallacy_critique_request="Identify ways in which the \
        assistant’s last response is asking a question that \
        contains or assumes information that has not been proven or \
        substantiated.",
        fallacy_revision_request="Please write a new assistant \
        response so that it does not ask a question that contains \
        or assumes information that has not been proven or \
        substantiated.",
    ),
    "overapplier": LogicalFallacy(
        name="overapplier",
        fallacy_critique_request="Identify ways in which the assistant’s\
        last response is applying a general rule or generalization to a \
        specific case it was not meant to apply to.",
        fallacy_revision_request="Please write a new response that does\
        not apply a general rule or generalization to a specific case \
        it was not meant to apply to.",
    ),
    "equivocation": LogicalFallacy(
        name="equivocation",
        fallacy_critique_request="Read the assistant’s last response \
        carefully and identify if it is using the same word or phrase \
        in two different senses or contexts within an argument.",
        fallacy_revision_request="Rewrite the assistant response so \
        that it does not use the same word or phrase in two different \
        senses or contexts within an argument.",
    ),
    "amphiboly": LogicalFallacy(
        name="amphiboly",
        fallacy_critique_request="Critique the assistant’s last response\
        to see if it is constructing sentences such that the grammar \
        or structure is ambiguous, leading to multiple interpretations.",
        fallacy_revision_request="Please rewrite the assistant response\
        to remove any construction of sentences where the grammar or \
        structure is ambiguous or leading to multiple interpretations.",
    ),
    "accent": LogicalFallacy(
        name="accent",
        fallacy_critique_request="Discuss whether the assitant's response\
        is misrepresenting an argument by shifting the emphasis of a word\
        or phrase to give it a different meaning than intended.",
        fallacy_revision_request="Please rewrite the AI model's response\
        so that it is not misrepresenting an argument by shifting the \
        emphasis of a word or phrase to give it a different meaning than\
        intended.",
    ),
    "composition": LogicalFallacy(
        name="composition",
        fallacy_critique_request="Discuss whether the assistant's \
        response is erroneously inferring that something is true of \
        the whole based on the fact that it is true of some part or \
        parts.",
        fallacy_revision_request="Please rewrite the assitant's response\
        so that it is not erroneously inferring that something is true \
        of the whole based on the fact that it is true of some part or \
        parts.",
    ),
    "division": LogicalFallacy(
        name="division",
        fallacy_critique_request="Discuss whether the assistant's last \
        response is erroneously inferring that something is true of the \
        parts based on the fact that it is true of the whole.",
        fallacy_revision_request="Please rewrite the assitant's response\
        so that it is not erroneously inferring that something is true \
        of the parts based on the fact that it is true of the whole.",
    ),
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
                          "Think carefully about how the {data} may be representing any of the logical fallacies contained in {FALLACIES}."
                           "\n\nIf one ore more logical fallacy exists in {data}: Return a short sentence to summarize the full article then summarize each portion of the article that contains a logical fallacy, with the logical fallacy evident in the summary of the article portion."
                           "\n\nIf zero logical follacies are found in {data}: return the text 'No Fallacies Found'
                          "
                        )
                        chain1 = LLMChain(llm=llm, prompt=prompt1)

                        # Chain 2: Name Each Logical Fallacy and Describe Harm
                        prompt2 = ChatPromptTemplate.from_template(
                          "If the summary reads 'No Fallacies Found', then return 'No Fallacies Found'."
                          "\n\nIf the summary does not read 'No Fallacies Found' then name each logical fallacy found, summarize a portion of the article where it is found in 10 words and explain in an additional 10 words why the logical fallacy may harm society as it exists in news.  Do this until all logical fallacies found in the summary are listed."
                        )
                        chain2 = LLMChain(llm=llm, prompt=prompt2)

                        overall_chain = SimpleSequentialChain(chains=[chain1,chain2], verbose=True)

                        summary_enriched = overall_chain.run(data)

                        st.success(f"Title: {item['title']}\n\nLink: {item['link']}\n\nSummary: {summary_enriched}")
        except Exception as e:
            st.exception(f"Exception: {e}")
