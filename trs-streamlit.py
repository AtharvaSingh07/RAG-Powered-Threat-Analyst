import os
import streamlit as st
import streamlit_mermaid as stmd
import pandas as pd

from streamlit_extras.badges import badge
from datetime import datetime
from loguru import logger

from trs.main import TRS


# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title='TRS - LLaMA3',
    page_icon='🦊',
    layout='wide',
    initial_sidebar_state='expanded'
)


# ---------------------------------------------------------
# CLEAR CHAT HISTORY
# ---------------------------------------------------------
def clear_chat_history():
    st.session_state.messages = []


# ---------------------------------------------------------
# SAVE UPLOADED FILE
# ---------------------------------------------------------
def save_uploaded_file(uploaded_file) -> str:
    save_dir = "uploads"
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
def main():
    st.header('TRS - Threat Report Summarizer (LLaMA-3)')
    st.subheader('Local File Playground', divider='rainbow')

    with st.sidebar:
        st.header('🦊 TRS (Local LLaMA-3)', divider='rainbow')
        badge(type='github', name='deadbits/trs')
        st.divider()

    page = st.sidebar.radio(
        'Select a page:',
        ['Analyze', 'Chat', 'Database', 'History']
    )

    # ---------------------------------------------------------
    # ANALYZE PAGE
    # ---------------------------------------------------------
    if page == 'Analyze':
        if 'history' not in st.session_state:
            st.session_state.history = []

        uploaded_file = st.file_uploader(
            "Upload Threat Report (.txt or .pdf)",
            type=["txt", "pdf"]
        )

        # Load prompts
        prompt_dir = 'prompts/'
        prompt_list = [
            p.replace('.txt', '')
            for p in os.listdir(prompt_dir)
            if p.endswith('.txt') and p.replace('.txt', '') not in ['qna', 'mindmap']
        ]

        prompt_name = st.selectbox(
            'Select a prompt:',
            prompt_list
        )

        response = None
        iocs = None
        mindmap = None

        if st.button('Submit'):
            if not uploaded_file:
                st.error("Please upload a threat report file.")
                return

            file_path = save_uploaded_file(uploaded_file)

            with st.spinner('Processing locally with LLaMA-3...'):
                try:
                    if prompt_name == 'detect':
                        response = trs.detections(file=file_path)

                    elif prompt_name == 'summary':
                        response, mindmap, iocs = trs.summarize(file=file_path)

                    else:
                        response = trs.custom(
                            file=file_path,
                            prompt_name=prompt_name
                        )

                except Exception as e:
                    logger.exception(e)
                    st.error(f"Processing failed: {e}")
                    return

            if response:
                st.session_state.history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'file': uploaded_file.name,
                    'prompt': prompt_name,
                    'response': response
                })

                st.subheader("Analysis Result")
                st.write(response)

                if iocs:
                    st.subheader('Indicators of Compromise (IOCs)')
                    st.write(iocs)

                if mindmap:
                    st.subheader('Attack Flow Mindmap')
                    stmd.st_mermaid(mindmap)
            else:
                st.error("No response generated.")

        else:
            st.info('Upload a file and select a prompt to begin analysis.')

    # ---------------------------------------------------------
    # CHAT PAGE
    # ---------------------------------------------------------
    elif page == 'Chat':
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        st.sidebar.button(
            'Clear Chat History',
            on_click=clear_chat_history
        )

        chat_input = st.text_input('Ask a question:')

        if chat_input:
            st.session_state.messages.append(
                {'role': 'user', 'content': chat_input}
            )

            with st.chat_message('user'):
                st.markdown(chat_input)

            with st.spinner('Thinking with LLaMA-3...'):
                response = trs.qna(prompt=chat_input)

            st.session_state.messages.append(
                {'role': 'assistant', 'content': response}
            )

            with st.chat_message('assistant'):
                st.markdown(response)

    # ---------------------------------------------------------
    # HISTORY PAGE
    # ---------------------------------------------------------
    elif page == 'History':
        st.title('Analysis History')

        history = sorted(
            st.session_state.get('history', []),
            key=lambda x: x['timestamp'],
            reverse=True
        )

        for item in history:
            st.write('Timestamp:', item['timestamp'])
            st.write('File:', item['file'])
            st.write('Prompt:', item['prompt'])
            st.write('Response:', item['response'])
            st.divider()

    # ---------------------------------------------------------
    # DATABASE PAGE
    # ---------------------------------------------------------
    elif page == 'Database':
        st.title('Vector Database Viewer')
        st.markdown(f'**Total records:** {trs.vdb.count()}')
        st.caption('Limited to first 10,000 records')

        data = trs.vdb.get()
        df = pd.DataFrame.from_dict(data).head(10000)

        st.dataframe(df)


# ---------------------------------------------------------
# RUN APP
# ---------------------------------------------------------
if __name__ == '__main__':
    trs = TRS()  # Local LLaMA-3 only
    main()
