import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from gemini_key import api_key
import json
import os

os.environ["GOOGLE_API_KEY"] = api_key

gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_output_tokens=300)

system_prompt = """
You are a helpful AI Assistant which helps humans to do ABSA (Aspect Based Sentiment Analysis) on a user review.

Your task is to analyze the user review expressed towards specific aspects in the following sentence: {user_review}.
Also generate overall review (as positive, mild positive, neutral, mild negative, negative) based on the aspects and their sentiment respectively.

Identify the aspects and their corresponding sentiments.

For example, if the sentence is 'The mobile was good, but the delivery was late and there are some heating issues.', you would identify 'mobile' as an aspect with positive sentiment and 'delivery', 'heating' as two different aspects with negative sentiments.
O/P:
Aspect: mobile, Sentiment: positive
Aspect: delivery, Sentiment: negative
Aspect: heating, Sentiment: negative

Only return the output in a dictionary i.e. a json (all aspects in a list and their respective sentiments in another list) and don't include any formatting (give it as raw string).
"""

# Creating the prompt template
prompt_template = ChatPromptTemplate.from_template(system_prompt)

chain = prompt_template | gemini_llm

st.title("Aspect-Based Sentiment Analysis")

# Predefined reviews
predefined_reviews = {
    "Review 1": "The camera quality is great, but the battery life is poor.",
    "Review 2": "The gaming console is expensive, and the online features could be improved.",
    "Review 3": "The laptop is fast and has a great display, but it overheats quickly.",
    "Review 4": "The smartwatch has a small screen that's difficult to read in direct sunlight, and the software could be more intuitive.",
    "Review 5": "The headphones have great sound quality and are comfortable to wear.",
    "Review 6": "The laptop has a cheap build quality, an unresponsive trackpad, and a low-resolution screen.",
}

custom_review = st.text_area("Enter your own review:", "")

if custom_review:
    review_to_analyze = custom_review
else:
    # Show predefined reviews dropdown
    selected_review = st.selectbox("Choose a sample review:", list(predefined_reviews.keys()))
    review_to_analyze = predefined_reviews[selected_review]

st.write(f"Review to analyze: {review_to_analyze}")

if st.button("Analyze"):
    # Invoking the chain
    ai_response = chain.invoke({'user_review': review_to_analyze}).content

    # Parsing the response
    try:
        # Cleaning and converting the response to dictionary
        start = ai_response.find('{')
        end = ai_response.rfind('}') + 1
        json_response = ai_response[start:end]
        result_dict = json.loads(json_response)
        
        final_response = dict(zip(result_dict['aspects'], result_dict['sentiments']))

        st.write("### Analysis Results:")
        for aspect, sentiment in final_response.items():
            st.write(f"{aspect}: {sentiment}")
    except Exception as e:
        st.error(f"Error parsing the response: {e}")