from typing import List
import os
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from .settings import get_settings
from dotenv import load_dotenv


#Load project configuration
S = get_settings()


load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
_initialized = False


def _init():
    """Initialize Vertex AI once per session."""
    global _initialized
    if not _initialized:
        try:
            vertexai.init(
                project=S.gcp_project_id,
                location=S.gcp_location,
                credentials=None,  
            )
            print(f" Vertex AI initialized for {S.gcp_project_id} in {S.gcp_location}")
            print(f" Credentials: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
            _initialized = True
        except Exception as e:
            print("Vertex AI initialization failed:", e)
            raise


def get_embedding(text: str) -> List[float]:
    """Generate a text embedding using Vertex AI Gecko model."""
    _init()
    try:
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        emb = model.get_embeddings([text])[0].values
        return emb
    except Exception as e:
        print(" Error generating embedding:", e)
        raise


def generate_text(prompt: str) -> str:
    """Generate a text response using Gemini."""
    _init()
    try:
        model = GenerativeModel("gemini-2.0-flash-001")
        response = model.generate_content(prompt)
        return response.text or ""
    except Exception as e:
        print(" Error generating text:", e)
        raise


def vertex_test():
    """Test Vertex AI embedding model directly."""
    print("\n Running Vertex AI connection test...\n")
    _init()
    try:
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        emb = model.get_embeddings(["MemoraAI test run"])[0].values
        print(" Vertex AI test successful. Sample embedding:", emb[:5])
    except Exception as e:
        print(" Vertex AI test failed:", e)
        raise


# This  runs test automatically when executed directly
if __name__ == "__main__":
    vertex_test()
