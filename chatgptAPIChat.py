import streamlit as st
import streamlit.components.v1 as components
from io import BytesIO
from pathlib import Path
from tenacity import retry, wait_random_exponential, stop_after_attempt
from PyPDF2 import PdfReader
from PIL import Image
from openai import OpenAI, OpenAIError
import time
import base64
import json

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

# Model configurations with their specific parameters
MODELS_CONFIG = {
    # Newest GPT-4.1 series models (April 2025)
    "gpt-4.1": {
        "name": "GPT-4.1", 
        "type": "chat",
        "description": "Latest model with major improvements in coding and instruction following. 1M token context.",
        "supports": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    },
    "gpt-4.1-mini": {
        "name": "GPT-4.1 Mini", 
        "type": "chat",
        "description": "Lightweight variant with strong reasoning at lower cost and latency.",
        "supports": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    },
    "gpt-4.1-nano": {
        "name": "GPT-4.1 Nano", 
        "type": "chat",
        "description": "Fastest and cheapest model ideal for classification and autocompletion.",
        "supports": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    },
    
    # GPT-4.5 (Research Preview - being deprecated July 2025)
    "gpt-4.5-preview": {
        "name": "GPT-4.5 Preview", 
        "type": "chat",
        "description": "Large experimental model (being deprecated July 2025). Very expensive.",
        "supports": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    },
    
    # GPT-4o series
    "gpt-4o": {
        "name": "GPT-4o", 
        "type": "chat",
        "description": "Flagship multimodal model with audio, vision, and text capabilities.",
        "supports": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini", 
        "type": "chat",
        "description": "Faster and more affordable version of GPT-4o.",
        "supports": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    },
    
    # GPT-4 series
    "gpt-4": {
        "name": "GPT-4", 
        "type": "chat",
        "description": "Original GPT-4 model for creativity and advanced reasoning.",
        "supports": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    },
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo", 
        "type": "chat",
        "description": "Enhanced GPT-4 with larger context window and improved capabilities.",
        "supports": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    },
    
    # GPT-3.5 series
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo", 
        "type": "chat",
        "description": "Fast and affordable model for simpler tasks.",
        "supports": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    },
    
    # Latest o-series reasoning models (April 2025)
    "o3": {
        "name": "o3", 
        "type": "reasoning",
        "description": "Most powerful reasoning model for coding, math, science, and vision. Tops SWE-Bench.",
        "supports": ["max_completion_tokens", "reasoning_effort"]
    },
    "o4-mini": {
        "name": "o4-mini", 
        "type": "reasoning",
        "description": "Fast, cost-efficient reasoning model. Best on AIME benchmarks.",
        "supports": ["max_completion_tokens", "reasoning_effort"]
    },
    "o3-mini": {
        "name": "o3-mini", 
        "type": "reasoning",
        "description": "Enhanced reasoning model with web search capabilities.",
        "supports": ["max_completion_tokens", "reasoning_effort"]
    },
    
    # Previous o-series models
    "o1": {
        "name": "o1", 
        "type": "reasoning",
        "description": "First reasoning model with enhanced problem-solving capabilities.",
        "supports": ["max_completion_tokens"]
    },
    "o1-mini": {
        "name": "o1-mini", 
        "type": "reasoning",
        "description": "Smaller reasoning model for cost-sensitive applications.",
        "supports": ["max_completion_tokens"]
    },
    "o1-preview": {
        "name": "o1-preview", 
        "type": "reasoning",
        "description": "Preview version of the original o1 reasoning model.",
        "supports": ["max_completion_tokens"]
    },
}

# Initialize session state for stored key validation
if "check_stored_key" not in st.session_state:
    st.session_state.check_stored_key = False
if "stored_key_to_validate" not in st.session_state:
    st.session_state.stored_key_to_validate = None

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

# API Key Input Section - WORKING VERSION
if not st.session_state.api_key_valid:
    st.markdown("#### 🔑 Enter Your OpenAI API Key")
    st.markdown("""
    **To use this interface, you need your own OpenAI API key:**
    1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
    2. Create a new secret key
    3. Copy and paste it below
    
    **Your API key can be stored locally in your browser for convenience.**
    """)
    
    # Check for returning users
    st.markdown("#### 🔍 Check for Stored API Key")
    st.markdown("If you've used this app before on this browser, you may have a stored API key.")
    
    # Check for stored key button
    if st.button("🔍 Check for Stored API Key", type="secondary"):
        st.session_state.check_stored_key = True
    
    # Display stored key checker when button is clicked
    if st.session_state.check_stored_key:
        check_script = """
        <div id="key-check-result"></div>
        <script>
        (function(){
            const resultDiv = document.getElementById('key-check-result');
            const storedKey = localStorage.getItem('chatgpt_api_key');
            if (!storedKey) {
                resultDiv.innerHTML = '<div style="color: #666; padding: 10px; background: #f5f5f5; border-radius: 5px;">No stored key found</div>';
                return;
            }
            try {
                const decodedKey = atob(storedKey);
                if (!decodedKey.startsWith('sk-')) {
                    resultDiv.innerHTML = '<div style="color: #ff6b35; padding: 10px; background: #ffeaa7; border-radius: 5px;">Invalid stored key format</div>';
                    return;
                }
                const maskedKey = decodedKey.substring(0,7) + '...' + decodedKey.substring(decodedKey.length -4);
                resultDiv.innerHTML = `
                    <div style="color: #2e7d32; background: #f0f8f0; padding: 15px; border-radius: 5px; border: 1px solid #4CAF50;">
                        <div style="margin-bottom: 10px;"><b>Found stored key:</b> ${maskedKey}</div>
                        <button id="useStoredKeyBtn" style="padding: 8px 16px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">✓ Use This Key</button>
                    </div>
                `;
                document.getElementById('useStoredKeyBtn').onclick = function(){
                    // Instead of postMessage, put key into an input field
                    const input = document.createElement('input');
                    input.type = 'text';
                    input.value = decodedKey;
                    input.id = 'storedKeyInput';
                    input.style.position = 'fixed';
                    input.style.top = '10px';
                    input.style.left = '10px';
                    input.style.zIndex = 9999;
                    document.body.appendChild(input);
                    input.select();
                    document.execCommand('copy');
                    alert('Stored API key copied to clipboard. Please paste it into the API key input box.');
                }
            } catch (err) {
                resultDiv.innerHTML = `<div style="color: #d32f2f; padding: 10px; background: #ffebee; border-radius: 5px;">Error reading key: ${err.message}</div>`;
            }
        })();
        </script>
        """
        components.html(check_script, height=150)

    
    # Handle stored key validation from query params
    if 'validate_stored_key' in st.query_params:
        try:
            encoded_key = st.query_params.get('validate_stored_key')
            if encoded_key:
                stored_key = base64.b64decode(encoded_key).decode('utf-8')
                
                with st.spinner("Validating stored API key..."):
                    is_valid, result = validate_api_key(stored_key)
                    if is_valid:
                        st.session_state.api_key_valid = True
                        st.session_state.client = result
                        # Clear the query param and reset state
                        st.query_params.clear()
                        st.session_state.check_stored_key = False
                        st.success("✅ Welcome back! Your stored API key is valid.")
                        st.rerun()
                    else:
                        st.error(f"❌ Stored API key is invalid: {result}")
                        st.query_params.clear()
                        st.session_state.check_stored_key = False
        except Exception as e:
            st.error(f"Error processing stored key: {e}")
            if 'validate_stored_key' in st.query_params:
                st.query_params.clear()
            st.session_state.check_stored_key = False
    
    st.markdown("---")
    st.markdown("#### ✍️ Enter New API Key")
    
    api_key = st.text_input(
        "OpenAI API Key", 
        type="password", 
        placeholder="sk-...",
        help="Your API key will be securely stored in your browser's local storage for future use."
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        save_key = st.checkbox("💾 Save key for future use", value=True)
    
    with col2:
        if st.button("Validate API Key", type="primary"):
            if api_key:
                with st.spinner("Validating API key..."):
                    is_valid, result = validate_api_key(api_key)
                    if is_valid:
                        st.session_state.api_key_valid = True
                        st.session_state.client = result
                        
                        if save_key:
                            save_script = f"""
                            <script>
                                try {{
                                    const apiKey = `{api_key}`;
                                    const encodedKey = btoa(apiKey);
                                    localStorage.setItem('chatgpt_api_key', encodedKey);
                                }} catch (error) {{
                                    console.error('Error saving API key:', error);
                                }}
                            </script>
                            """
                            components.html(save_script, height=0)
                            st.success("✅ API key validated and saved successfully!")
                        else:
                            st.success("✅ API key validated successfully!")
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
        st.session_state.messages = [
            {"role": "system", "content": "You are ChatGPT, a large language model."}
        ]
        st.rerun()
with col2:
    if st.button("Clear Stored Key", use_container_width=True):
        # Properly clear from browser storage with confirmation
        clear_script = """
        <script>
        try {
            localStorage.removeItem('chatgpt_api_key');
            // Show immediate feedback
            alert('✅ Stored API key cleared successfully!');
        } catch (error) {
            alert('❌ Error clearing stored key: ' + error.message);
        }
        </script>
        """
        components.html(clear_script, height=0)

# Model selection with categorization
st.sidebar.markdown("### Select Model")

# Get reasoning and chat models separately for better organization
reasoning_models = {k: v for k, v in MODELS_CONFIG.items() if v["type"] == "reasoning"}
chat_models = {k: v for k, v in MODELS_CONFIG.items() if v["type"] == "chat"}

# Custom format function that shows model type
def format_model_name(model_key):
    model = MODELS_CONFIG[model_key]
    if model["type"] == "reasoning":
        return f"🧠 {model['name']}"
    else:
        return f"💬 {model['name']}"

model_key = st.sidebar.selectbox(
    "Choose a model", 
    list(MODELS_CONFIG.keys()),
    format_func=format_model_name
)

selected_model = MODELS_CONFIG[model_key]

# Model description as hover tooltip with info icon
st.sidebar.markdown(f"""
<div style="margin-bottom: 15px;">
    <span style="font-weight: bold;">{selected_model['name']}</span> 
    <span style="color: #666;">- {selected_model['type'].title()}</span>
    <div class="tooltip" style="display: inline-block; margin-left: 5px;">
        <span style="font-size: 16px; color: #0066cc; cursor: help;">ℹ️</span>
        <span class="tooltiptext">{selected_model.get('description', 'No description available')}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Model parameters based on model type with hover descriptions
st.sidebar.markdown("### Model Parameters")

# Helper function to create parameter with tooltip
def create_parameter_with_tooltip(label, tooltip, widget_func, key, *args, **kwargs):
    st.sidebar.markdown(f"""
    <div style="margin-bottom: 5px;">
        <span style="font-weight: 500;">{label}</span>
        <div class="tooltip" style="display: inline-block; margin-left: 5px;">
            <span style="font-size: 14px; color: #0066cc; cursor: help;">ℹ️</span>
            <span class="tooltiptext">{tooltip}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    return widget_func(label, key=key, label_visibility="hidden", *args, **kwargs)

# Common parameters for all models
if "max_tokens" in selected_model["supports"]:
    max_tokens = create_parameter_with_tooltip(
        "Max Tokens",
        "Maximum number of tokens to generate. Higher values allow longer responses but cost more. Lower values create shorter, more concise responses.",
        st.sidebar.slider,
        "max_tokens_slider",
        100, 4000, 1000
    )
elif "max_completion_tokens" in selected_model["supports"]:
    max_completion_tokens = create_parameter_with_tooltip(
        "Max Completion Tokens",
        "Maximum tokens in the response for reasoning models. Higher values allow more detailed reasoning and longer explanations.",
        st.sidebar.slider,
        "max_completion_tokens_slider",
        100, 32000, 4000
    )

# Parameters for chat models
if selected_model["type"] == "chat":
    temperature = create_parameter_with_tooltip(
        "Temperature",
        "Controls randomness. Lower values (0.0) make responses more focused and deterministic. Higher values (1.0) make responses more creative and varied.",
        st.sidebar.slider,
        "temperature_slider",
        0.0, 1.0, 0.7
    )
    
    top_p = create_parameter_with_tooltip(
        "Top P",
        "Controls diversity via nucleus sampling. Lower values focus on most likely tokens. Higher values consider more possibilities for creative responses.",
        st.sidebar.slider,
        "top_p_slider",
        0.0, 1.0, 1.0
    )
    
    freq_penalty = create_parameter_with_tooltip(
        "Frequency Penalty",
        "Reduces repetition based on token frequency. Higher values discourage repeating phrases and words.",
        st.sidebar.slider,
        "freq_penalty_slider",
        0.0, 2.0, 0.0
    )
    
    pres_penalty = create_parameter_with_tooltip(
        "Presence Penalty",
        "Encourages discussing new topics. Higher values make the model talk about different subjects rather than staying on the same topic.",
        st.sidebar.slider,
        "pres_penalty_slider",
        0.0, 2.0, 0.0
    )

# Parameters for reasoning models
if selected_model["type"] == "reasoning" and "reasoning_effort" in selected_model["supports"]:
    reasoning_effort = create_parameter_with_tooltip(
        "Reasoning Effort",
        "Controls how much thinking time the model uses. Higher effort leads to better quality but slower and more expensive responses.",
        st.sidebar.selectbox,
        "reasoning_effort_selectbox",
        ["low", "medium", "high"], index=1
    )

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
    st.markdown("#### Chat History")
    
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
st.markdown("#### Your Message")
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

# Custom CSS for better styling and tooltips
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
    
    /* Tooltip styles */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: calc(100vw - 40px); /* Full viewport width minus padding */
        max-width: 300px; /* Maximum width to prevent it from being too wide on large screens */
        background-color: #333;
        color: white;
        text-align: left;
        border-radius: 8px;
        padding: 10px;
        position: absolute;
        z-index: 9999;
        top: 100%;
        left: 0;
        margin-top: 5px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 13px;
        line-height: 1.4;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        word-wrap: break-word;
        white-space: normal;
    }
    
    /* For sidebar tooltips specifically */
    .stSidebar .tooltip .tooltiptext {
        width: calc(100% + 100px); /* Sidebar width plus some extra space */
        max-width: 280px;
        left: 25px; /* Position to the right of the info icon */
        top: 50%;
        transform: translateY(-50%);
        margin-top: 0;
    }
    
    .tooltip .tooltiptext::before {
        content: "";
        position: absolute;
        top: 50%;
        left: -5px;
        margin-top: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: transparent #333 transparent transparent;
    }
    
    /* Adjust arrow for top-positioned tooltips */
    .tooltip .tooltiptext:not(.stSidebar .tooltip .tooltiptext)::before {
        top: -5px;
        left: 10px;
        margin-top: 0;
        border-color: #333 transparent transparent transparent;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
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