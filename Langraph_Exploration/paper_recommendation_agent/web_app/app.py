import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import PaperRecommendationAgent
import json

def main():
    st.set_page_config(
        page_title="Academic Paper Recommender",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 AI Paper Recommendation Agent")
    st.markdown("Find relevant academic papers based on your research interests")
    
    # Initialize agent
    @st.cache_resource
    def load_agent():
        return PaperRecommendationAgent()
    
    agent = load_agent()
    
    # Sidebar
    st.sidebar.header("Configuration")
    default_query = st.sidebar.text_area(
        "Default research interests:",
        "machine learning deep learning neural networks"
    )
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_query = st.text_area(
            "Describe your research interests:",
            value=default_query,
            height=100,
            help="Be specific about your research area, techniques, and topics of interest"
        )
    
    with col2:
        st.markdown("### Examples:")
        examples = [
            "few-shot learning for natural language processing",
            "computer vision with transformers",
            "reinforcement learning for robotics",
            "graph neural networks for social networks"
        ]
        
        for example in examples:
            if st.button(example, key=example):
                user_query = example
                st.rerun()
    
    if st.button("Find Relevant Papers", type="primary"):
        with st.spinner("Searching for relevant papers..."):
            result = agent.recommend(user_query, save_output=False)
        
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            # Display results
            st.success(f"Found {result['total_candidates']} candidate papers")
            
            # Show plan
            with st.expander("Search Plan", expanded=False):
                st.json(result['plan'])
            
            # Display recommendations
            st.header("Recommended Papers")
            
            for i, rec in enumerate(result['recommendations']):
                paper = rec['paper']
                
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.subheader(f"{i+1}. {paper['title']}")
                        st.markdown(f"**Relevance Score:** `{rec['relevance_score']:.3f}`")
                        st.markdown(f"**Categories:** {', '.join(paper['categories'])}")
                        st.markdown(f"**Published:** {paper['published']}")
                        
                        st.markdown(f"**Why relevant:** {rec['justification']}")
                        
                        with st.expander("Abstract"):
                            st.write(paper['abstract'])
                    
                    with col2:
                        if paper.get('pdf_url'):
                            st.markdown(f"[📄 PDF]({paper['pdf_url']})")
                    
                    st.divider()

if __name__ == "__main__":
    main()