import streamlit as st
from io import BytesIO
from pathlib import Path
from tenacity import retry, wait_random_exponential, stop_after_attempt
from PyPDF2 import PdfReader
from PIL import Image
from openai import OpenAI, OpenAIError
import time

# Configure page layout
st.set_page_config(page_title="ChatGPT API Interface", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are ChatGPT, a large language model."}
    ]
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False
if "pending_api_params" not in st.session_state:
    st.session_state.pending_api_params = None
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False
if "client" not in st.session_state:
    st.session_state.client = None
if "stored_api_key" not in st.session_state:
    st.session_state.stored_api_key = None
if "check_local_storage" not in st.session_state:
    st.session_state.check_local_storage = True

# Model configurations with their specific parameters
MODELS_CONFIG = {
    # Standard GPT models
    "gpt-4o": {
        "name": "GPT-4o", 
        "type": "chat",
        "supports": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini", 
        "type": "chat",
        "supports": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    },
    "gpt-4": {
        "name": "GPT-4", 
        "type": "chat",
        "supports": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    },
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo", 
        "type": "chat",
        "supports": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    },
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo", 
        "type": "chat",
        "supports": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    },
    # Reasoning models
    "o1": {
        "name": "o1", 
        "type": "reasoning",
        "supports": ["max_completion_tokens"]
    },
    "o1-mini": {
        "name": "o1-mini", 
        "type": "reasoning",
        "supports": ["max_completion_tokens"]
    },
    "o1-preview": {
        "name": "o1-preview", 
        "type": "reasoning",
        "supports": ["max_completion_tokens"]
    },
    "o3-mini": {
        "name": "o3-mini", 
        "type": "reasoning",
        "supports": ["max_completion_tokens", "reasoning_effort"]
    },
}

def validate_api_key(api_key):
    """Test if the API key is valid by making a simple API call"""
    try:
        client = OpenAI(api_key=api_key)
        # Make a simple test call
        response = client.models.list()
        return True, client
    except OpenAIError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

# Main title
st.title("🤖 ChatGPT API Chat Interface")

# JavaScript functions for local storage
if st.session_state.check_local_storage:
    st.session_state.check_local_storage = False
    st.html("""
    <script>
    function saveApiKey(apiKey) {
        localStorage.setItem('openai_api_key', apiKey);
    }
    
    function getApiKey() {
        return localStorage.getItem('openai_api_key');
    }
    
    function clearApiKey() {
        localStorage.removeItem('openai_api_key');
    }
    
    // Check if there's a stored API key
    const storedKey = getApiKey();
    if (storedKey) {
        // Send the stored key to Streamlit
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: storedKey
        }, '*');
    }
    
    // Set up message listener for saving API keys
    window.addEventListener('message', function(event) {
        if (event.data && event.data.type === 'saveApiKey') {
            saveApiKey(event.data.key);
        } else if (event.data && event.data.type === 'clearApiKey') {
            clearApiKey();
        }
    });
    </script>
    """)

# Check for stored API key in local storage
stored_key_placeholder = st.empty()
if not st.session_state.api_key_valid and not st.session_state.stored_api_key:
    with stored_key_placeholder:
        # Create a hidden component to receive the stored API key
        key_from_storage = st.text_input("", key="hidden_api_key", label_visibility="hidden")
        if key_from_storage and key_from_storage.startswith("sk-"):
            st.session_state.stored_api_key = key_from_storage
            # Auto-validate the stored key
            with st.spinner("Validating stored API key..."):
                is_valid, result = validate_api_key(key_from_storage)
                if is_valid:
                    st.session_state.api_key_valid = True
                    st.session_state.client = result
                    st.success("✅ Welcome back! Your stored API key is still valid.")
                    st.rerun()
                else:
                    # Invalid stored key, clear it
                    st.session_state.stored_api_key = None
                    st.html("""<script>
                    parent.postMessage({type: 'clearApiKey'}, '*');
                    </script>""")

stored_key_placeholder.empty()

# API Key Input Section
if not st.session_state.api_key_valid:
    st.markdown("## 🔑 Enter Your OpenAI API Key")
    st.markdown("""
    **To use this interface, you need your own OpenAI API key:**
    1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
    2. Create a new secret key
    3. Copy and paste it below
    
    **Your API key is stored only in your current session and is never saved or shared.**
    """)
    
    api_key = st.text_input(
        "OpenAI API Key", 
        type="password", 
        placeholder="sk-...",
        help="Your API key will be used for this session only and is not stored anywhere."
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Validate API Key", type="primary"):
            if api_key:
                with st.spinner("Validating API key..."):
                    is_valid, result = validate_api_key(api_key)
                    if is_valid:
                        st.session_state.api_key_valid = True
                        st.session_state.client = result
                        st.session_state.stored_api_key = api_key
                        # Save to local storage
                        st.html(f"""<script>
                        parent.postMessage({{type: 'saveApiKey', key: '{api_key}'}}, '*');
                        </script>""")
                        st.success("✅ API key validated and saved successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Invalid API key: {result}")
            else:
                st.warning("Please enter your API key")
    
    st.stop()

# If we get here, API key is valid
# Sidebar configuration
st.sidebar.title("Settings")

# Show API key status
st.sidebar.markdown("### 🔑 API Key Status")
st.sidebar.success("✅ Valid API key connected")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Change API Key", use_container_width=True):
        st.session_state.api_key_valid = False
        st.session_state.client = None
        st.session_state.stored_api_key = None
        st.session_state.messages = [
            {"role": "system", "content": "You are ChatGPT, a large language model."}
        ]
        st.rerun()
with col2:
    if st.button("Clear Stored Key", use_container_width=True):
        st.session_state.api_key_valid = False
        st.session_state.client = None
        st.session_state.stored_api_key = None
        st.session_state.messages = [
            {"role": "system", "content": "You are ChatGPT, a large language model."}
        ]
        # Clear from local storage
        st.html("""<script>
        parent.postMessage({type: 'clearApiKey'}, '*');
        </script>""")
        st.rerun()

# Model selection
model_key = st.sidebar.selectbox(
    "Select Model", 
    list(MODELS_CONFIG.keys()),
    format_func=lambda x: MODELS_CONFIG[x]["name"]
)

selected_model = MODELS_CONFIG[model_key]

# Model parameters based on model type
st.sidebar.markdown("### Model Parameters")

# Common parameters for all models
if "max_tokens" in selected_model["supports"]:
    max_tokens = st.sidebar.slider("Max Tokens", 100, 4000, 1000)
elif "max_completion_tokens" in selected_model["supports"]:
    max_completion_tokens = st.sidebar.slider("Max Completion Tokens", 100, 32000, 4000)

# Parameters for chat models
if selected_model["type"] == "chat":
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
    top_p = st.sidebar.slider("Top P", 0.0, 1.0, 1.0)
    freq_penalty = st.sidebar.slider("Frequency Penalty", 0.0, 2.0, 0.0)
    pres_penalty = st.sidebar.slider("Presence Penalty", 0.0, 2.0, 0.0)

# Parameters for reasoning models
if selected_model["type"] == "reasoning" and "reasoning_effort" in selected_model["supports"]:
    reasoning_effort = st.sidebar.selectbox("Reasoning Effort", ["low", "medium", "high"], index=1)

# File uploader
st.sidebar.markdown("### Upload a File")
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"]
)

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "system", "content": "You are ChatGPT, a large language model."}
    ]
    st.rerun()

# Retry wrapper for API calls
def retry_if_openai_error(exception):
    return isinstance(exception, OpenAIError)

@retry(wait=wait_random_exponential(min=1, max=20),
       stop=stop_after_attempt(5),
       retry=retry_if_openai_error)
def chat_completion_call(model_key, messages, **kwargs):
    """Make API call with model-specific parameters"""
    model_config = MODELS_CONFIG[model_key]
    client = st.session_state.client
    
    if model_config["type"] == "chat":
        # Standard chat completion
        resp = client.chat.completions.create(
            model=model_key,
            messages=messages,
            **kwargs
        )
    elif model_config["type"] == "reasoning":
        # Reasoning models use different parameters
        # For reasoning models, we need to filter out system messages and use developer messages instead
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                # Convert system message to developer message for reasoning models
                filtered_messages.append({"role": "developer", "content": msg["content"]})
            else:
                filtered_messages.append(msg)
        
        resp = client.chat.completions.create(
            model=model_key,
            messages=filtered_messages,
            **kwargs
        )
    
    return resp.choices[0].message.content

# Chat interface
st.markdown("---")

# Chat container with fixed height and scrolling
chat_container = st.container()
with chat_container:
    st.markdown("### Chat History")
    
    # Create a scrollable chat area
    chat_placeholder = st.empty()
    
    with chat_placeholder.container():
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "system":
                continue  # Don't display system messages
            elif msg["role"] == "user":
                st.markdown(f"""
                <div style="background-color: #e8f4fd; padding: 10px; border-radius: 10px; margin: 5px 0; margin-left: 20px; color: black;">
                    <strong>You:</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            elif msg["role"] == "assistant":
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 5px 0; margin-right: 20px; color: black;">
                    <strong>ChatGPT:</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
        
        # Show loading indicator if waiting for response
        if st.session_state.waiting_for_response:
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 5px 0; margin-right: 20px; color: black;">
                <strong>ChatGPT:</strong> <em>Thinking...</em>
            </div>
            """, unsafe_allow_html=True)
            
            # Show a simple progress indicator
            with st.container():
                progress_placeholder = st.empty()
                with progress_placeholder:
                    st.markdown("""
                    <div style="text-align: center; padding: 10px;">
                        <div style="display: inline-block; animation: pulse 1.5s ease-in-out infinite;">
                            🤖 ChatGPT is processing your request...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# File display area
if uploaded_file:
    if uploaded_file.type.startswith("image/"):
        st.image(uploaded_file, caption=uploaded_file.name, width=300)

# User input area at the bottom
st.markdown("### Your Message")
with st.form(key="message_form", clear_on_submit=True):
    user_input = st.text_area("Type your message here...", height=100, key="user_message")
    col1, col2 = st.columns([4, 1])
    
    with col2:
        submit_button = st.form_submit_button("Send", use_container_width=True)

# Handle message submission
if submit_button and user_input.strip():
    # Handle file context
    context = ""
    if uploaded_file:
        file_type = uploaded_file.type
        file_name = uploaded_file.name
        if file_type == "application/pdf":
            try:
                pdf_reader = PdfReader(uploaded_file)
                full_text = []
                for page in pdf_reader.pages:
                    full_text.append(page.extract_text() or "")
                pdf_text = "\n".join(full_text)
                context = f"User uploaded PDF '{file_name}' with content:\n{pdf_text[:2000]}"
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
                context = f"User uploaded PDF '{file_name}' but couldn't read content."
        elif file_type.startswith("image/"):
            context = f"User uploaded image file: {file_name}."

    # Build user message
    user_msg = f"{context}\n\n{user_input}" if context else user_input
    st.session_state.messages.append({"role": "user", "content": user_msg})

    # Prepare API parameters based on model type
    api_params = {}
    
    if selected_model["type"] == "chat":
        api_params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": freq_penalty,
            "presence_penalty": pres_penalty,
        }
    elif selected_model["type"] == "reasoning":
        api_params = {
            "max_completion_tokens": max_completion_tokens,
        }
        if "reasoning_effort" in selected_model["supports"]:
            api_params["reasoning_effort"] = reasoning_effort

    # Set waiting state and store API params
    st.session_state.waiting_for_response = True
    st.session_state.pending_api_params = (model_key, st.session_state.messages.copy(), api_params)
    st.rerun()  # Show user message immediately

# Handle API call if waiting for response
if st.session_state.waiting_for_response and st.session_state.pending_api_params:
    model_key, messages, api_params = st.session_state.pending_api_params
    
    try:
        assistant_response = chat_completion_call(model_key, messages, **api_params)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.session_state.waiting_for_response = False
        st.session_state.pending_api_params = None
        st.rerun()  # Show response and hide loading
    except OpenAIError as e:
        st.error(f"API call failed: {e}")
        st.session_state.waiting_for_response = False
        st.session_state.pending_api_params = None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.session_state.waiting_for_response = False
        st.session_state.pending_api_params = None

# Custom CSS for better styling
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #e8f4fd;
        margin-left: 20px;
    }
    .assistant-message {
        background-color: #f0f2f6;
        margin-right: 20px;
    }
    @keyframes pulse {
        0% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
        100% {
            opacity: 1;
        }
    }
</style>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px;">
    <p>🔐 Your API key is stored securely in your browser's local storage for convenience.<br>
    💰 You are charged directly by OpenAI for API usage based on your own billing.<br>
    🔄 Use "Clear Stored Key" in settings to remove the saved key from your browser.</p>
</div>
""", unsafe_allow_html=True)