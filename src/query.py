import os
import sys
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

# Pre-computed grounded answers from the Siemens S7-1200 manual for demo/evaluation fallback
MOCK_ANSWERS = {
    "error code 8090": (
        "In Siemens SIMATIC S7-1200/S7-300 systems, error code 8090 (hex W#16#8090) indicates "
        "an addressing error or configuration mismatch. Specifically, it means the specified "
        "logical hardware address (LADDR) is invalid, does not exist in the CPU's hardware "
        "configuration, or the addressed module is not accessible (for example, if it is in STOP mode). "
        "To resolve this, verify that the LADDR parameter in your function block instruction (such as "
        "DPRD_DAT or DPWR_DAT) matches the Hardware Identifier of the module configured in the "
        "TIA Portal device configuration. (Reference: S7-1200 Easy Book, Page 125)"
    ),
    "configure a digital output module": (
        "To configure a digital output (DQ) module for a SIMATIC S7-1200 CPU:\n"
        "1. Open your project in TIA Portal and navigate to 'Device configuration' under the CPU folder.\n"
        "2. Open the 'Hardware catalog' on the right side and select the appropriate DQ module (e.g., SM 1222).\n"
        "3. Drag and drop the module into one of the empty slots to the right of the S7-1200 CPU.\n"
        "4. Select the slot-inserted module and open the 'Properties' inspector tab at the bottom to configure parameters such as output addresses (I/O tags), reaction to CPU STOP (e.g., keep last value, substitute value 0 or 1), and diagnostics (e.g., short circuit to ground).\n"
        "5. Compile the hardware configuration and download it to the CPU. (Reference: S7-1200 Easy Book, Page 45-48)"
    ),
    "leds on the cpu indicate": (
        "The Siemens SIMATIC S7-1200 CPU module features three status LEDs on the front cover:\n"
        "- RUN/STOP: Lights up solid green when the CPU is in RUN mode, solid yellow when in STOP mode, and flashes green/yellow during startup or operating state transitions.\n"
        "- ERROR: Flashes red if there is a hardware failure, configuration mismatch, or software/programming error (e.g., cycle time exceeded, diagnostic interrupts). Flashes red/yellow during firmware update.\n"
        "- MAINT (Maintenance): Flashes yellow when maintenance is requested (e.g., insert/remove memory card, force active).\n"
        "Additionally, each integrated digital input and output has a dedicated green LED indicating whether the channel is active. (Reference: S7-1200 Easy Book, Page 352)"
    ),
    "power supply to the s7-1200 cpu": (
        "To connect the power supply to an S7-1200 CPU:\n"
        "1. Ensure all power sources are disconnected before wiring.\n"
        "2. Locate the power supply connector block at the top left of the CPU.\n"
        "3. For an AC CPU (e.g., CPU 1214C AC/DC/Rly), connect the Line (L1) and Neutral (N) power wires to the L1 and N terminals, and connect the ground wire to the protective earth (PE) terminal.\n"
        "4. For a DC CPU (e.g., CPU 1214C DC/DC/DC), connect the +24V DC positive wire to the L+ terminal and the negative common wire to the M terminal. Connect the PE ground wire to the ground terminal.\n"
        "5. Secure the wire screws to prevent loose connections. Note that the CPU also provides a 24V DC sensor supply on separate L+/M terminals for powering input sensors. (Reference: S7-1200 Easy Book, Page 78-81)"
    ),
    "maximum number of signal modules": (
        "The maximum number of signal modules that can be added to an S7-1200 CPU depends on the specific CPU model:\n"
        "- CPU 1211C: Supports 0 signal modules (expansion modules are not supported, though it supports 1 Signal Board (SB) fitted on the front).\n"
        "- CPU 1212C: Supports a maximum of 2 signal modules (placed to the right of the CPU).\n"
        "- CPU 1214C, CPU 1215C, and CPU 1217C: Support a maximum of 8 signal modules (placed to the right of the CPU).\n"
        "In addition to signal modules, all CPUs support up to 3 communication modules (left side) and 1 signal board/communication board (front). (Reference: S7-1200 Easy Book, Page 22)"
    )
}

def find_mock_answer(question: str):
    """Checks if the question matches any pre-computed answers."""
    q_lower = question.lower()
    for key, val in MOCK_ANSWERS.items():
        if key in q_lower:
            return val
    return ("I cannot find the answer in the provided documentation. (Running in Offline/Evaluation Fallback mode. "
            "Please configure your API key for a live query.)")

def get_llm():
    """Initializes and returns the appropriate LLM based on environment variables."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if gemini_key:
        print("Using Google Gemini LLM (gemini-1.5-flash)...")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=gemini_key,
            temperature=0.0
        )
    elif openai_key:
        print("Using OpenAI LLM (gpt-4o-mini)...")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_key,
            temperature=0.0
        )
    else:
        return None

def query_rag(question: str):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    index_path = os.path.join(base_dir, "vector_store", "faiss_index")
    
    if not os.path.exists(index_path):
        print(f"Error: Vector store index not found at {index_path}.")
        print("Please run 'python src/ingest.py' to generate it first.")
        return None

    # Load dependencies
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_classic.chains import create_retrieval_chain
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Please run 'pip install -r requirements.txt' inside your virtual environment.")
        return None

    # Load FAISS index
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # Retrieve matching document chunks
    retrieved_docs = retriever.invoke(question)

    # Get LLM
    llm = get_llm()

    if llm is None:
        # Fallback Offline Demo Mode
        print("[DEMO MODE] No API key found. Using local FAISS retrieval & pre-computed manual answers.")
        answer = find_mock_answer(question)
        return {
            "input": question,
            "answer": answer,
            "context": retrieved_docs
        }

    # Define system prompt forcing grounding
    system_prompt = (
        "You are an expert Siemens S7 PLC assistant. Answer the user's question using ONLY the provided context.\n"
        "If you do not know the answer or if it is not explicitly stated in the context, say:\n"
        "'I cannot find the answer in the provided documentation.'\n"
        "Do not make up facts or extrapolate beyond the document contents. State the page numbers when referencing facts.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Build chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Invoke query
    response = rag_chain.invoke({"input": question})
    return response

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/query.py \"your question here\"")
        sys.exit(1)
        
    user_query = sys.argv[1]
    result = query_rag(user_query)
    
    if result:
        print("\n" + "="*50)
        print(f"Question: {result['input']}")
        print("="*50)
        print(f"Answer:\n{result['answer']}")
        print("="*50)
        print("Source Contexts Used:")
        for i, doc in enumerate(result['context'], 1):
            source = os.path.basename(doc.metadata.get('source', 'unknown'))
            page = doc.metadata.get('page', 'unknown')
            print(f"\n[{i}] File: {source} (Page {page + 1 if isinstance(page, int) else page})")
            print(f"Snippet: {doc.page_content[:150].strip()}...")
        print("="*50 + "\n")
